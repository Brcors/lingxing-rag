"""游客的真实 MySQL 存储、清理、权限与过期测试；不请求真实 LLM。"""
import secrets
from dataclasses import replace
from datetime import datetime, timezone
import threading

import pytest

from rag.accounts.memory import MemoryManager
from rag.accounts.guests import GuestCleanup
from rag.accounts.repository import AccountRepository, ConversationNotFound, SCHEMA
from rag.settings import Settings
from rag.service import RAGService
from rag.types import ContextBundle
from test_accounts_memory import setup, csrf, headers, register, new_chat, ask


def enter_guest(app):
    client = app.test_client()
    response = client.post("/api/auth/guest", headers=csrf(client))
    assert response.status_code == 201, response.json
    with client.session_transaction() as session:
        guest_id = session["guest_id"]
        assert "user_id" not in session
    return client, guest_id, client.get("/api/conversations").json["conversations"][0]["id"]


def count(repo, table, where="", args=()):
    assert table in {"auth_users", "chat_conversations", "guest_chat_turns", "chat_turns", "chat_memory_versions", "liked_answers"}
    with repo.connection() as conn, conn.cursor() as cur:
        cur.execute(f"SELECT COUNT(*) AS n FROM {table} " + where, args)
        return cur.fetchone()["n"]


def test_guest_entry_and_independent_conversations(setup):
    app, repo, service, _ = setup
    anonymous = app.test_client()
    assert "以游客身份进入" in anonymous.get("/login").get_data(as_text=True)
    assert anonymous.post("/api/auth/guest").status_code == 403
    a, ga, ca = enter_guest(app)
    b, gb, cb = enter_guest(app)
    user = register(app, "alice")
    registered_chat = new_chat(user)
    assert len(ga) == 64 and ga != gb
    assert count(repo, "auth_users") == 1
    result = ask(a, ca, "游客的问题")
    assert result.status_code == 200 and "feedback_id" not in result.json
    assert service.last_allow_feedback is False
    assert count(repo, "guest_chat_turns") == 1
    assert count(repo, "chat_turns") == 0
    assert b.get(f"/api/conversations/{ca}").status_code == 404
    assert ask(b, ca).status_code == 404
    assert user.get(f"/api/conversations/{ca}").status_code == 404
    assert a.get(f"/api/conversations/{registered_chat}").status_code == 404
    assert a.get("/").status_code == 200
    assert 'content="guest"' in a.get("/").get_data(as_text=True)
    second = new_chat(a)
    assert len(a.get("/api/conversations").json["conversations"]) == 2
    assert a.get(f"/api/conversations/{ca}").json["turns"][0]["question"] == "游客的问题"
    assert a.get(f"/api/conversations/{second}").json["turns"] == []
    assert app.extensions["guests"].touch_guest(ga)
    assert app.extensions["guests"].touch_guest(ga)  # 同一秒连续访问也有效。
    lifetime = a.get_cookie("lingxing_session").expires - datetime.now(timezone.utc)
    assert 23 * 3600 < lifetime.total_seconds() <= 24 * 3600
    user_lifetime = user.get_cookie("lingxing_session").expires - datetime.now(timezone.utc)
    assert 7 * 3600 < user_lifetime.total_seconds() <= 8 * 3600


def test_guest_cannot_write_shared_knowledge_or_index(setup):
    app, repo, service, _ = setup
    user = register(app, "alice")
    uc = new_chat(user)
    result = ask(user, uc, "共享知识")
    fid = result.json["feedback_id"]
    assert user.post("/api/feedback", json={"feedback_id": fid}, headers=headers(user)).status_code == 200
    visitor, gid, cid = enter_guest(app)
    assert ask(visitor, cid, "查看共享知识").status_code == 200
    assert visitor.post("/api/feedback", json={"feedback_id": fid}, headers=headers(visitor)).status_code == 403
    assert visitor.post("/api/index", json={}, headers=headers(visitor)).status_code == 403
    assert count(repo, "liked_answers") == 1
    visitor.post("/api/auth/logout", headers=headers(visitor))
    assert count(repo, "liked_answers") == 1  # 退出不删全局点赞资料。


def test_guest_memory_five_then_four_and_logout_cascades_only_owner(setup):
    app, repo, service, _ = setup
    visitor, gid, cid = enter_guest(app)
    other, other_gid, other_cid = enter_guest(app)
    user = register(app, "alice")
    uc = new_chat(user)
    ask(user, uc, "永久记录")
    ask(other, other_cid, "另一个游客")
    guests = app.extensions["guests"]
    memory = MemoryManager(guests, service.llm)
    for n in range(5):
        assert ask(visitor, cid, f"第一批{n}").status_code == 200
    memory.compact(gid, cid)
    assert guests.context(gid, cid)["memory_version"] == 1
    for n in range(4):
        ask(visitor, cid, f"第二批{n}")
    memory.compact(gid, cid)
    assert guests.context(gid, cid)["memory_version"] == 2
    assert count(repo, "guest_chat_turns", "WHERE conversation_id=%s", (cid,)) == 9
    second = new_chat(visitor)
    ask(visitor, second)
    old_cookie = visitor.get_cookie("lingxing_session").value
    assert visitor.post("/api/auth/logout", headers=headers(visitor)).status_code == 200
    assert count(repo, "chat_conversations", "WHERE guest_session_id=%s", (gid,)) == 0
    assert count(repo, "guest_chat_turns", "WHERE conversation_id IN (%s,%s)", (cid, second)) == 0
    assert count(repo, "chat_memory_versions", "WHERE conversation_id=%s", (cid,)) == 0
    assert count(repo, "chat_turns") == 1
    assert other.get(f"/api/conversations/{other_cid}").json["turns"]
    replay = app.test_client()
    replay.set_cookie("lingxing_session", old_cookie)
    assert replay.get("/api/conversations").status_code == 401
    with pytest.raises(ConversationNotFound):
        guests.create_conversation(gid)


def test_idle_expiry_touch_and_cleanup(setup):
    app, repo, service, _ = setup
    visitor, gid, cid = enter_guest(app)
    active, active_gid, active_cid = enter_guest(app)
    ask(visitor, cid)
    with repo.connection() as conn, conn.cursor() as cur:
        cur.execute("UPDATE chat_conversations SET expires_at=DATE_ADD(UTC_TIMESTAMP(),INTERVAL 1 HOUR) WHERE guest_session_id=%s", (active_gid,))
    assert active.get("/api/conversations").status_code == 200
    with repo.connection() as conn, conn.cursor() as cur:
        cur.execute("SELECT expires_at>DATE_ADD(UTC_TIMESTAMP(),INTERVAL 23 HOUR) AS extended FROM chat_conversations WHERE id=%s", (active_cid,))
        assert cur.fetchone()["extended"]
        cur.execute("UPDATE chat_conversations SET expires_at=DATE_SUB(UTC_TIMESTAMP(),INTERVAL 1 SECOND) WHERE guest_session_id=%s", (gid,))
    assert visitor.get("/api/conversations").status_code == 401
    assert app.extensions["guests"].cleanup_expired() == 1
    assert count(repo, "guest_chat_turns", "WHERE conversation_id=%s", (cid,)) == 0
    assert active.get("/api/conversations").status_code == 200


def test_failed_login_keeps_guest_successful_account_switch_clears_guest(setup):
    app, repo, service, _ = setup
    registered = register(app, "alice")
    visitor, gid, cid = enter_guest(app)
    ask(visitor, cid)
    assert visitor.get("/login").status_code == 200
    response = visitor.post("/api/auth/login", headers=headers(visitor), json={"username": "alice", "password": "wrong-password"})
    assert response.status_code == 401
    assert count(repo, "guest_chat_turns") == 1
    response = visitor.post("/api/auth/login", headers=headers(visitor), json={"username": "alice", "password": "test-password-123"})
    assert response.status_code == 200
    with visitor.session_transaction() as session:
        assert "user_id" in session and "guest_id" not in session
    assert count(repo, "guest_chat_turns") == 0
    assert count(repo, "chat_conversations", "WHERE guest_session_id=%s", (gid,)) == 0
    assert visitor.post("/api/auth/guest", headers=headers(visitor)).status_code == 400


def test_late_summary_cannot_recreate_logged_out_guest(setup):
    app, repo, service, _ = setup
    visitor, gid, cid = enter_guest(app)
    for n in range(5):
        ask(visitor, cid, f"问{n}")
    guests = app.extensions["guests"]
    job = guests.claim_memory(gid, cid)
    visitor.post("/api/auth/logout", headers=headers(visitor))
    with pytest.raises(ConversationNotFound):
        guests.finish_memory(gid, cid, job, summary="迟到的摘要")
    assert count(repo, "chat_memory_versions") == 0
    assert count(repo, "guest_chat_turns") == 0


def test_upgrade_existing_schema_preserves_registered_data(tmp_path):
    database = "rag_auth_test_" + secrets.token_hex(8)
    settings = replace(Settings.load(), mysql_database=database, data_dir=tmp_path, api_key="")
    repo = AccountRepository(settings)
    connection = repo.backend._connect(database=False)
    try:
        with connection.cursor() as cur:
            cur.execute(f"CREATE DATABASE `{database}` CHARACTER SET utf8mb4")
            cur.execute(f"USE `{database}`")
            cur.execute(SCHEMA[0])
            old = SCHEMA[1].replace("user_id BIGINT UNSIGNED NULL", "user_id BIGINT UNSIGNED NOT NULL")
            old = old.replace("guest_session_id CHAR(64) NULL,expires_at DATETIME(6) NULL,", "")
            old = old.replace("KEY idx_guest_session(guest_session_id),KEY idx_guest_expiry(expires_at),", "")
            cur.execute(old)
            cur.execute(SCHEMA[2])
            cur.execute(SCHEMA[3])
            cur.execute("INSERT INTO auth_users(username,password_hash) VALUES('existing','existing-hash')")
            cur.execute("INSERT INTO chat_conversations(id,user_id,title) VALUES(%s,1,'原有对话')", ("a" * 32,))
        connection.commit()
        repo.initialize()
        assert repo.list_conversations(1)[0]["title"] == "原有对话"
        with repo.connection() as conn, conn.cursor() as cur:
            cur.execute("SHOW COLUMNS FROM chat_conversations LIKE 'user_id'")
            assert cur.fetchone()["Null"] == "YES"
            cur.execute("SHOW TABLES LIKE 'guest_chat_turns'")
            assert cur.fetchone()
    finally:
        with connection.cursor() as cur:
            cur.execute(f"DROP DATABASE IF EXISTS `{database}`")
        connection.commit()
        connection.close()


def test_guest_generation_does_not_fill_feedback_buffer(setup, monkeypatch):
    _app, _repo, _service, settings = setup
    rag = RAGService(settings)
    monkeypatch.setattr(rag, "retrieve", lambda *a, **k: (ContextBundle("测试证据", [{"id": "S1"}], 10), {}))
    monkeypatch.setattr(rag.llm, "stream", lambda *a, **k: iter(["测试回答[S1]"]))
    monkeypatch.setattr(rag.feedback_kb, "register_answer", lambda *a, **k: pytest.fail("游客回答不应写入点赞暂存区"))
    done = list(rag.events("查询参数", allow_feedback=False))[-1]
    assert done["mode"] == "answer" and "feedback_id" not in done


def test_cleanup_worker_retries_after_database_failure():
    complete = threading.Event()

    class Repository:
        calls = 0

        def cleanup_expired(self):
            self.calls += 1
            if self.calls == 1:
                raise RuntimeError("temporary database failure")
            complete.set()

    repo = Repository()
    worker = GuestCleanup(repo, interval=0.01)
    worker.start()
    try:
        assert complete.wait(2)
    finally:
        worker.stop()
        worker._thread.join(timeout=1)
    assert repo.calls >= 2
