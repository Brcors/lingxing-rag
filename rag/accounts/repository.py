import json
import logging
import secrets
import threading
from contextlib import contextmanager

import pymysql

from rag.knowledge.mysql_repository import MySQLKnowledgeRepository

logger = logging.getLogger(__name__)


class AccountStoreError(RuntimeError):
    pass


class ConversationNotFound(LookupError):
    pass


class ConversationBusy(RuntimeError):
    pass


class DuplicateUsername(ValueError):
    pass


class AccountRepository:
    owner_column = "user_id"
    turn_table = "chat_turns"

    def __init__(self, settings):
        self.backend = MySQLKnowledgeRepository(settings)
        self._ready = False
        self._lock = threading.Lock()

    def initialize(self):
        if self._ready:
            return
        with self._lock:
            if self._ready:
                return
            try:
                self.backend.initialize()
                with self.backend._connect() as connection:
                    with connection.cursor() as cursor:
                        lock_name = "rag_schema_" + self.backend.settings.mysql_database[:48]
                        cursor.execute("SELECT GET_LOCK(%s,15) AS acquired", (lock_name,))
                        if not cursor.fetchone()["acquired"]:
                            raise RuntimeError("账户表升级正在进行，请稍后重试。")
                        try:
                            for sql in SCHEMA:
                                cursor.execute(sql)
                            self._migrate_guests(cursor)
                        finally:
                            cursor.execute("SELECT RELEASE_LOCK(%s)", (lock_name,))
                    connection.commit()
                self._ready = True
            except Exception as error:
                logger.exception("账户数据库初始化失败")
                raise AccountStoreError("无法连接账户数据库，请确认 MySQL 服务已启动。") from error

    @staticmethod
    def _migrate_guests(cursor):
        cursor.execute("SHOW COLUMNS FROM chat_conversations")
        columns = {column["Field"]: column for column in cursor.fetchall()}
        if columns["user_id"]["Null"] != "YES":
            cursor.execute("ALTER TABLE chat_conversations MODIFY user_id BIGINT UNSIGNED NULL")
        if "guest_session_id" not in columns:
            cursor.execute("ALTER TABLE chat_conversations ADD guest_session_id CHAR(64) NULL, ADD KEY idx_guest_session(guest_session_id)")
        if "expires_at" not in columns:
            cursor.execute("ALTER TABLE chat_conversations ADD expires_at DATETIME(6) NULL, ADD KEY idx_guest_expiry(expires_at)")

    @contextmanager
    def connection(self):
        self.initialize()
        connection = None
        try:
            connection = self.backend._connect()
            with connection.cursor() as cursor:
                cursor.execute("SET time_zone = '+00:00'")
            yield connection
            connection.commit()
        except (ConversationNotFound, ConversationBusy, DuplicateUsername):
            if connection:
                connection.rollback()
            raise
        except Exception as error:
            if connection:
                connection.rollback()
            logger.exception("账户或对话数据库操作失败")
            raise AccountStoreError("账户数据库暂不可用，请确认 MySQL 连接后重试。") from error
        finally:
            if connection:
                connection.close()

    def create_user(self, username, password_hash):
        with self.connection() as conn, conn.cursor() as cur:
            try:
                cur.execute("INSERT INTO auth_users(username,password_hash) VALUES(%s,%s)",
                            (username, password_hash))
            except pymysql.err.IntegrityError as error:
                if error.args[0] == 1062:
                    raise DuplicateUsername("该用户名已被注册。") from None
                raise
            return {"id": cur.lastrowid, "username": username}

    def get_user(self, user_id=None, username=None):
        with self.connection() as conn, conn.cursor() as cur:
            if user_id is not None:
                cur.execute("SELECT id,username,password_hash FROM auth_users WHERE id=%s", (user_id,))
            else:
                cur.execute("SELECT id,username,password_hash FROM auth_users WHERE username=%s", (username,))
            return cur.fetchone()

    def create_conversation(self, user_id):
        conversation_id = secrets.token_hex(16)
        with self.connection() as conn, conn.cursor() as cur:
            cur.execute("INSERT INTO chat_conversations(id,user_id,title) VALUES(%s,%s,%s)",
                        (conversation_id, user_id, "新对话"))
        return {"id": conversation_id, "title": "新对话"}

    def list_conversations(self, user_id):
        with self.connection() as conn, conn.cursor() as cur:
            cur.execute(f"""SELECT id,title,created_at,updated_at,memory_version FROM chat_conversations
                           WHERE {self.owner_column}=%s ORDER BY updated_at DESC LIMIT 200""", (user_id,))
            return cur.fetchall()

    def owned(self, cur, user_id, conversation_id, lock=False):
        cur.execute(f"SELECT * FROM chat_conversations WHERE id=%s AND {self.owner_column}=%s" +
                    (" FOR UPDATE" if lock else ""), (conversation_id, user_id))
        row = cur.fetchone()
        if row is None:
            raise ConversationNotFound("对话不存在或无权访问。")
        return row

    def read_conversation(self, user_id, conversation_id, before=None):
        with self.connection() as conn, conn.cursor() as cur:
            row = self.owned(cur, user_id, conversation_id)
            sql = f"SELECT id,question,answer,response_json,feedback_id,liked_at,created_at FROM {self.turn_table} WHERE conversation_id=%s"
            args = [conversation_id]
            if before:
                sql += " AND id<%s"
                args.append(before)
            cur.execute(sql + " ORDER BY id DESC LIMIT 51", args)
            turns = cur.fetchall()
            more = len(turns) > 50
            turns = turns[:50]
            for turn in turns:
                turn["response"] = json.loads(turn.pop("response_json"))
                turn["liked"] = bool(turn.pop("liked_at"))
            return {"id": row["id"], "title": row["title"], "memory_version": row["memory_version"],
                    "memory_error": row["memory_error"], "turns": list(reversed(turns)),
                    "has_more": more, "before": turns[-1]["id"] if turns else None}

    def acquire(self, user_id, conversation_id):
        token = secrets.token_hex(16)
        with self.connection() as conn, conn.cursor() as cur:
            self.owned(cur, user_id, conversation_id, lock=True)
            cur.execute("""UPDATE chat_conversations SET busy_token=%s,busy_until=DATE_ADD(UTC_TIMESTAMP(),INTERVAL 20 MINUTE)
                           WHERE id=%s AND (busy_until IS NULL OR busy_until<UTC_TIMESTAMP())""",
                        (token, conversation_id))
            if cur.rowcount != 1:
                raise ConversationBusy("该对话正在生成回答，请完成后再发送。")
        return token

    def release(self, user_id, conversation_id, token):
        with self.connection() as conn, conn.cursor() as cur:
            cur.execute(f"""UPDATE chat_conversations SET busy_token=NULL,busy_until=NULL
                           WHERE id=%s AND {self.owner_column}=%s AND busy_token=%s""", (conversation_id, user_id, token))

    def context(self, user_id, conversation_id):
        with self.connection() as conn, conn.cursor() as cur:
            row = self.owned(cur, user_id, conversation_id)
            cur.execute(f"""SELECT question,answer FROM {self.turn_table} WHERE conversation_id=%s AND id>%s
                           ORDER BY id DESC LIMIT 4""", (conversation_id, row["covered_turn_id"]))
            history = []
            for turn in reversed(cur.fetchall()):
                history.extend([{"role": "user", "content": turn["question"][:4000]},
                                {"role": "assistant", "content": turn["answer"][:4000]}])
            hint = ""
            if not history and row["covered_turn_id"]:
                cur.execute(f"SELECT question FROM {self.turn_table} WHERE id=%s AND conversation_id=%s",
                            (row["covered_turn_id"], conversation_id))
                latest = cur.fetchone()
                hint = latest["question"] if latest else ""
            return {"summary": row["summary"] or "", "history": history,
                    "retrieval_hint": hint, "memory_version": row["memory_version"], "memory_error": row["memory_error"]}

    def find_request(self, user_id, conversation_id, request_id):
        with self.connection() as conn, conn.cursor() as cur:
            self.owned(cur, user_id, conversation_id)
            cur.execute(f"SELECT response_json FROM {self.turn_table} WHERE conversation_id=%s AND request_id=%s",
                        (conversation_id, request_id))
            row = cur.fetchone()
            return json.loads(row["response_json"]) if row else None

    def save_turn(self, user_id, conversation_id, token, request_id, question, category, event):
        with self.connection() as conn, conn.cursor() as cur:
            row = self.owned(cur, user_id, conversation_id, lock=True)
            if row["busy_token"] != token:
                raise ConversationBusy("对话锁已过期，请重新发送。")
            cur.execute(f"""INSERT INTO {self.turn_table}(conversation_id,request_id,question,answer,category,response_json,feedback_id)
                           VALUES(%s,%s,%s,%s,%s,%s,%s)""", (
                               conversation_id, request_id, question, event["answer"], category,
                               json.dumps(event, ensure_ascii=False), event.get("feedback_id")))
            turn_id = cur.lastrowid
            cur.execute("""UPDATE chat_conversations SET title=IF(title='新对话',%s,title),updated_at=UTC_TIMESTAMP(6)
                           WHERE id=%s""", (question[:60], conversation_id))
            return turn_id

    def liked_record(self, user_id, feedback_id):
        with self.connection() as conn, conn.cursor() as cur:
            cur.execute("""SELECT t.*,c.user_id FROM chat_turns t JOIN chat_conversations c ON c.id=t.conversation_id
                           WHERE t.feedback_id=%s AND c.user_id=%s""", (feedback_id, user_id))
            row = cur.fetchone()
            if row is None:
                raise ConversationNotFound("该问答不存在或无权点赞。")
            event = json.loads(row["response_json"])
            return {"question": row["question"], "answer": row["answer"],
                    "category": row["category"], "sources": event.get("sources", [])}

    def mark_liked(self, user_id, feedback_id):
        with self.connection() as conn, conn.cursor() as cur:
            cur.execute("""UPDATE chat_turns t JOIN chat_conversations c ON c.id=t.conversation_id
                           SET t.liked_at=UTC_TIMESTAMP(6) WHERE t.feedback_id=%s AND c.user_id=%s""",
                        (feedback_id, user_id))

    def claim_memory(self, user_id, conversation_id):
        token = secrets.token_hex(16)
        with self.connection() as conn, conn.cursor() as cur:
            row = self.owned(cur, user_id, conversation_id, lock=True)
            cur.execute("SELECT summary_until>UTC_TIMESTAMP() AS active FROM chat_conversations WHERE id=%s",
                        (conversation_id,))
            if cur.fetchone()["active"]:
                return None
            threshold = 4 if row["memory_version"] else 5
            cur.execute(f"SELECT id,question,answer FROM {self.turn_table} WHERE conversation_id=%s AND id>%s ORDER BY id LIMIT %s",
                        (conversation_id, row["covered_turn_id"], threshold))
            turns = cur.fetchall()
            if len(turns) < threshold:
                return None
            cur.execute("""UPDATE chat_conversations SET summary_token=%s,summary_until=DATE_ADD(UTC_TIMESTAMP(),INTERVAL 10 MINUTE)
                           WHERE id=%s""", (token, conversation_id))
            return {"token": token, "version": row["memory_version"], "summary": row["summary"] or "", "turns": turns}

    def finish_memory(self, user_id, conversation_id, job, summary=None, error=None):
        with self.connection() as conn, conn.cursor() as cur:
            row = self.owned(cur, user_id, conversation_id, lock=True)
            if row["summary_token"] != job["token"] or row["memory_version"] != job["version"]:
                return False
            if error:
                cur.execute("UPDATE chat_conversations SET summary_token=NULL,summary_until=NULL,memory_error=%s WHERE id=%s",
                            (error[:250], conversation_id))
                return False
            through_id = job["turns"][-1]["id"]
            cur.execute("""INSERT INTO chat_memory_versions(conversation_id,version,summary,covered_turn_id)
                           VALUES(%s,%s,%s,%s)""", (conversation_id, job["version"] + 1, summary, through_id))
            cur.execute("""UPDATE chat_conversations SET summary=%s,covered_turn_id=%s,memory_version=memory_version+1,
                           summary_token=NULL,summary_until=NULL,memory_error=NULL WHERE id=%s""",
                        (summary, through_id, conversation_id))
            return True


SCHEMA = [
    """CREATE TABLE IF NOT EXISTS auth_users (
        id BIGINT UNSIGNED NOT NULL AUTO_INCREMENT PRIMARY KEY,
        username VARCHAR(64) CHARACTER SET utf8mb4 COLLATE utf8mb4_bin NOT NULL UNIQUE,
        password_hash VARCHAR(255) NOT NULL,
        created_at DATETIME(6) NOT NULL DEFAULT CURRENT_TIMESTAMP(6)
    ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4""",
    """CREATE TABLE IF NOT EXISTS chat_conversations (
        id CHAR(32) NOT NULL PRIMARY KEY,user_id BIGINT UNSIGNED NULL,title VARCHAR(100) NOT NULL,
        guest_session_id CHAR(64) NULL,expires_at DATETIME(6) NULL,
        summary MEDIUMTEXT NULL,covered_turn_id BIGINT UNSIGNED NOT NULL DEFAULT 0,
        memory_version INT UNSIGNED NOT NULL DEFAULT 0,memory_error VARCHAR(255) NULL,
        busy_token CHAR(32) NULL,busy_until DATETIME NULL,summary_token CHAR(32) NULL,summary_until DATETIME NULL,
        created_at DATETIME(6) NOT NULL DEFAULT CURRENT_TIMESTAMP(6),
        updated_at DATETIME(6) NOT NULL DEFAULT CURRENT_TIMESTAMP(6),
        KEY idx_user_updated(user_id,updated_at),KEY idx_guest_session(guest_session_id),KEY idx_guest_expiry(expires_at),
        FOREIGN KEY(user_id) REFERENCES auth_users(id) ON DELETE CASCADE
    ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4""",
    """CREATE TABLE IF NOT EXISTS chat_turns (
        id BIGINT UNSIGNED NOT NULL AUTO_INCREMENT PRIMARY KEY,conversation_id CHAR(32) NOT NULL,
        request_id CHAR(36) NOT NULL,question TEXT NOT NULL,answer MEDIUMTEXT NOT NULL,
        category VARCHAR(128) NULL,response_json JSON NOT NULL,feedback_id VARCHAR(64) NULL UNIQUE,
        liked_at DATETIME(6) NULL,created_at DATETIME(6) NOT NULL DEFAULT CURRENT_TIMESTAMP(6),
        UNIQUE KEY uq_conversation_request(conversation_id,request_id),KEY idx_conversation_turn(conversation_id,id),
        FOREIGN KEY(conversation_id) REFERENCES chat_conversations(id) ON DELETE CASCADE
    ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4""",
    """CREATE TABLE IF NOT EXISTS chat_memory_versions (
        id BIGINT UNSIGNED NOT NULL AUTO_INCREMENT PRIMARY KEY,conversation_id CHAR(32) NOT NULL,
        version INT UNSIGNED NOT NULL,summary MEDIUMTEXT NOT NULL,covered_turn_id BIGINT UNSIGNED NOT NULL,
        created_at DATETIME(6) NOT NULL DEFAULT CURRENT_TIMESTAMP(6),
        UNIQUE KEY uq_conversation_memory(conversation_id,version),
        FOREIGN KEY(conversation_id) REFERENCES chat_conversations(id) ON DELETE CASCADE
    ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4""",
]

# 游客问答独立存储；复用对话元数据和摘要表，退出时由外键级联清除。
SCHEMA.append(SCHEMA[2].replace("chat_turns", "guest_chat_turns"))
