"""MySQL 中仅存储用户明确点赞的问答及其本地向量。"""
import json
import logging
import threading

import pymysql

logger = logging.getLogger(__name__)


class KnowledgeStoreError(RuntimeError):
    pass


class MySQLKnowledgeRepository:
    def __init__(self, settings):
        self.settings = settings
        self._schema_lock = threading.Lock()
        self._schema_ready = False
        self._snapshot_lock = threading.Lock()
        self._snapshot_cache = None

    def _connect(self, database=True):
        settings = self.settings
        return pymysql.connect(
            host=settings.mysql_host, port=settings.mysql_port, user=settings.mysql_user,
            password=settings.mysql_password, database=settings.mysql_database if database else None,
            charset=settings.mysql_charset, connect_timeout=settings.mysql_connect_timeout,
            read_timeout=5, write_timeout=5, cursorclass=pymysql.cursors.DictCursor,
            autocommit=False,
        )

    def initialize(self):
        if self._schema_ready:
            return
        with self._schema_lock:
            if self._schema_ready:
                return
            database = self.settings.mysql_database
            try:
                connection = self._connect(database=False)
                try:
                    with connection.cursor() as cursor:
                        cursor.execute(
                            f"CREATE DATABASE IF NOT EXISTS `{database}` "
                            "CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci"
                        )
                        cursor.execute(f"USE `{database}`")
                        cursor.execute(
                            """CREATE TABLE IF NOT EXISTS liked_answers (
                                id BIGINT UNSIGNED NOT NULL AUTO_INCREMENT,
                                feedback_id VARCHAR(64) NOT NULL,
                                question TEXT NOT NULL,
                                answer MEDIUMTEXT NOT NULL,
                                category VARCHAR(128) NULL,
                                sources JSON NOT NULL,
                                embedding LONGBLOB NOT NULL,
                                embedding_signature VARCHAR(64) NOT NULL,
                                created_at DATETIME(6) NOT NULL DEFAULT CURRENT_TIMESTAMP(6),
                                PRIMARY KEY (id),
                                UNIQUE KEY uq_feedback_id (feedback_id),
                                KEY idx_category_id (category, id)
                            ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci"""
                        )
                    connection.commit()
                finally:
                    connection.close()
                self._schema_ready = True
            except Exception as error:
                logger.exception("MySQL 点赞问答库初始化失败")
                raise KnowledgeStoreError("无法连接 MySQL 或初始化 lingxing_rag 数据库。") from error

    def save(self, record, vector_bytes, embedding_signature):
        self.initialize()
        try:
            connection = self._connect()
            try:
                with connection.cursor() as cursor:
                    cursor.execute(
                        """INSERT INTO liked_answers
                           (feedback_id, question, answer, category, sources, embedding, embedding_signature)
                           VALUES (%s, %s, %s, %s, %s, %s, %s)
                           ON DUPLICATE KEY UPDATE id=LAST_INSERT_ID(id)""",
                        (record["feedback_id"], record["question"], record["answer"], record.get("category"),
                         json.dumps(record["sources"], ensure_ascii=False), vector_bytes, embedding_signature),
                    )
                    inserted = cursor.rowcount == 1
                connection.commit()
            finally:
                connection.close()
            with self._snapshot_lock:
                self._snapshot_cache = None
            return inserted
        except Exception as error:
            logger.exception("保存用户点赞问答至 MySQL 失败")
            raise KnowledgeStoreError("点赞保存失败，请检查本机 MySQL 连接。") from error

    def snapshot(self):
        self.initialize()
        try:
            connection = self._connect()
            try:
                with connection.cursor() as cursor:
                    cursor.execute("SELECT COUNT(*) AS total, COALESCE(MAX(id), 0) AS latest_id FROM liked_answers")
                    version_row = cursor.fetchone()
                    version = (version_row["total"], version_row["latest_id"])
                    with self._snapshot_lock:
                        cached = self._snapshot_cache
                        if cached and cached[0] == version:
                            return cached
                    cursor.execute(
                        """SELECT id, feedback_id, question, answer, category, sources, embedding, embedding_signature
                           FROM liked_answers ORDER BY id DESC LIMIT %s""",
                        (self.settings.mysql_kb_max_records,),
                    )
                    rows = cursor.fetchall()
            finally:
                connection.close()
            normalized = []
            for row in rows:
                sources = row["sources"]
                if isinstance(sources, str):
                    try:
                        sources = json.loads(sources)
                    except json.JSONDecodeError:
                        sources = []
                row["sources"] = sources if isinstance(sources, list) else []
                normalized.append(row)
            result = (version, normalized)
            with self._snapshot_lock:
                self._snapshot_cache = result
            return result
        except KnowledgeStoreError:
            raise
        except Exception as error:
            logger.exception("读取 MySQL 点赞问答知识库失败")
            raise KnowledgeStoreError("读取 MySQL 问答知识库失败，本次将仅使用接口文档。") from error

    def count(self):
        self.initialize()
        try:
            connection = self._connect()
            try:
                with connection.cursor() as cursor:
                    cursor.execute("SELECT COUNT(*) AS total FROM liked_answers")
                    return int(cursor.fetchone()["total"])
            finally:
                connection.close()
        except Exception as error:
            raise KnowledgeStoreError("无法读取 MySQL 问答知识库状态。") from error
