"""游客临时会话：独立问答表，24 小时未活动清理，退出立即清理。"""
import logging
import secrets
import threading

from rag.accounts.repository import AccountRepository, ConversationNotFound

logger = logging.getLogger(__name__)


class GuestRepository(AccountRepository):
    owner_column = "guest_session_id"
    turn_table = "guest_chat_turns"

    def create_guest(self):
        guest_id = secrets.token_hex(32)
        conversation = self._create_conversation(guest_id, initial=True)
        return guest_id, conversation

    def create_conversation(self, guest_id):
        return self._create_conversation(guest_id)

    def _create_conversation(self, guest_id, initial=False):
        conversation_id = secrets.token_hex(16)
        with self.connection() as conn, conn.cursor() as cur:
            # 已存在的游客必须仍然有效，避免退出后的异步请求复活已清除的会话。
            cur.execute("SELECT id FROM chat_conversations WHERE guest_session_id=%s FOR UPDATE", (guest_id,))
            rows = cur.fetchall()
            if not initial and not rows:
                raise ConversationNotFound("游客会话已退出，请重新进入。")
            if rows:
                cur.execute("SELECT COUNT(*) AS active FROM chat_conversations WHERE guest_session_id=%s AND expires_at>UTC_TIMESTAMP()", (guest_id,))
                if not cur.fetchone()["active"]:
                    raise ConversationNotFound("游客会话已过期，请重新进入。")
            cur.execute("""INSERT INTO chat_conversations(id,user_id,guest_session_id,title,expires_at)
                           VALUES(%s,NULL,%s,'新对话',DATE_ADD(UTC_TIMESTAMP(6),INTERVAL 24 HOUR))""",
                        (conversation_id, guest_id))
        return {"id": conversation_id, "title": "新对话"}

    def owned(self, cur, guest_id, conversation_id, lock=False):
        row = super().owned(cur, guest_id, conversation_id, lock)
        cur.execute("SELECT expires_at>UTC_TIMESTAMP() AS active FROM chat_conversations WHERE id=%s", (conversation_id,))
        if not cur.fetchone()["active"]:
            raise ConversationNotFound("游客会话已过期，请重新进入。")
        return row

    def touch_guest(self, guest_id):
        with self.connection() as conn, conn.cursor() as cur:
            cur.execute("""SELECT id FROM chat_conversations WHERE guest_session_id=%s
                           AND expires_at>UTC_TIMESTAMP() FOR UPDATE""", (guest_id,))
            if not cur.fetchall():
                return False
            cur.execute("""UPDATE chat_conversations SET expires_at=DATE_ADD(UTC_TIMESTAMP(6),INTERVAL 24 HOUR)
                           WHERE guest_session_id=%s AND expires_at>UTC_TIMESTAMP()""", (guest_id,))
            return True

    def delete_guest(self, guest_id):
        with self.connection() as conn, conn.cursor() as cur:
            cur.execute("DELETE FROM chat_conversations WHERE guest_session_id=%s", (guest_id,))
            return cur.rowcount

    def delete_conversation(self, guest_id, conversation_id):
        with self.connection() as conn, conn.cursor() as cur:
            cur.execute("SELECT id FROM chat_conversations WHERE guest_session_id=%s ORDER BY id FOR UPDATE", (guest_id,))
            cur.fetchall()
            self._delete_owned(cur, guest_id, conversation_id)
            cur.execute("SELECT id FROM chat_conversations WHERE guest_session_id=%s LIMIT 1", (guest_id,))
            if cur.fetchone():
                return None
            # 留下一条空白对话作为游客身份锚点，避免删除最后一条后退出游客身份。
            replacement = {"id": secrets.token_hex(16), "title": "新对话"}
            cur.execute("""INSERT INTO chat_conversations(id,user_id,guest_session_id,title,expires_at)
                           VALUES(%s,NULL,%s,%s,DATE_ADD(UTC_TIMESTAMP(6),INTERVAL 24 HOUR))""",
                        (replacement["id"], guest_id, replacement["title"]))
            return replacement

    def cleanup_expired(self):
        with self.connection() as conn, conn.cursor() as cur:
            cur.execute("""DELETE FROM chat_conversations WHERE guest_session_id IS NOT NULL
                           AND expires_at<=UTC_TIMESTAMP()""")
            return cur.rowcount


class GuestCleanup:
    def __init__(self, repository, interval=60):
        self.repository, self.interval = repository, interval
        self._stop = threading.Event()
        self._thread = None

    def start(self):
        if self._thread is not None:
            return
        self._thread = threading.Thread(target=self._run, name="guest-cleanup", daemon=True)
        self._thread.start()

    def stop(self):
        self._stop.set()

    def _run(self):
        while not self._stop.is_set():
            try:
                self.repository.cleanup_expired()
            except Exception:
                logger.warning("游客过期记录暂未清理，将在下一轮重试。", exc_info=True)
            self._stop.wait(self.interval)
