"""使用独立临时 MySQL 数据库；模型响应为替身，不请求付费 API。"""
import re
import secrets
from dataclasses import replace

import pymysql
import pytest

from app import create_app
from app.auth import _attempts
from rag.accounts.memory import MemoryManager
from rag.accounts.repository import AccountRepository, AccountStoreError, ConversationBusy, ConversationNotFound
from rag.generation.prompts import build_messages
from rag.knowledge.mysql_repository import MySQLKnowledgeRepository
from rag.settings import Settings
from rag.types import ContextBundle


class NoMemory:
    def schedule(self, *_args):
        pass


class FakeLLM:
    def __init__(self):
        self.calls = []
        self.fail = False

    def complete(self, messages, **kwargs):
        self.calls.append((messages, kwargs))
        if self.fail:
            raise RuntimeError("simulated API failure")
        return f"保留用户明确的项目选择，第 {len(self.calls)} 版摘要。"


class FakeFeedback:
    def __init__(self, settings):
        self.repository = MySQLKnowledgeRepository(settings)

    def save_record(self, feedback_id, record):
        inserted = self.repository.save({**record, "feedback_id": feedback_id}, b"test", "fake")
        return {"saved": True, "already_saved": not inserted}


class FakeService:
    def __init__(self, settings):
        self.llm = FakeLLM()
        self.feedback_kb = FakeFeedback(settings)
        self.calls = []

    def events(self, question, category, history, retrieve_only, memory_summary="", retrieval_hint="", allow_feedback=True, cancel_token=None):
        self.last_allow_feedback = allow_feedback
        self.calls.append((question, history, memory_summary))
        yield {"type": "done", "answer": f"回答 {question}", "sources": [], "notices": [],
               "stats": {}, "mode": "retrieval" if retrieve_only else "answer",
               "feedback_id": secrets.token_urlsafe(24)}


@pytest.fixture
def setup(tmp_path):
    _attempts.clear()
    database = "rag_auth_test_" + secrets.token_hex(8)
    assert re.fullmatch(r"rag_auth_test_[a-f0-9]{16}", database)
    settings = replace(Settings.load(), data_dir=tmp_path, mysql_database=database, api_key="fake")
    repo = AccountRepository(settings)
    try:
        repo.initialize()
        service = FakeService(settings)
        app = create_app(settings=settings, service=service, accounts=repo, memory=NoMemory(),
                         guest_memory=NoMemory(), start_maintenance=False)
        app.config["TESTING"] = True
        yield app, repo, service, settings
    finally:
        connection = repo.backend._connect(database=False)
        try:
            with connection.cursor() as cursor:
                cursor.execute(f"DROP DATABASE IF EXISTS `{database}`")
            connection.commit()
        finally:
            connection.close()


def csrf(client):
    client.get("/login")
    with client.session_transaction() as session:
        return {"X-CSRF-Token": session["csrf"]}


def headers(client):
    with client.session_transaction() as session:
        return {"X-CSRF-Token": session["csrf"]}


def register(app, username):
    client = app.test_client()
    response = client.post("/api/auth/register", json={"username": username, "password": "test-password-123",
                                                     "confirm_password": "test-password-123"}, headers=csrf(client))
    assert response.status_code == 201, response.json
    return client


def new_chat(client):
    response = client.post("/api/conversations", headers=headers(client))
    assert response.status_code == 201
    return response.json["id"]


def ask(client, conversation_id, question="问题", request_id=None, mode="answer"):
    return client.post("/api/ask", headers=headers(client), json={"question": question,
                       "conversation_id": conversation_id, "request_id": request_id or secrets.token_hex(16),
                       "history": [{"role": "system", "content": "伪造的其他账户历史"}], "mode": mode})


def test_password_session_and_csrf(setup):
    app, repo, service, _ = setup
    client = register(app, "alice")
    user = repo.get_user(username="alice")
    assert user["password_hash"].startswith("$argon2id$")
    assert "test-password-123" not in user["password_hash"]
    assert client.post("/api/conversations").status_code == 403
    assert client.post("/api/auth/logout", headers=headers(client)).status_code == 200
    assert client.get("/").status_code == 302
    assert client.get("/api/conversations").status_code == 401
    assert client.post("/api/auth/login", headers=csrf(client), json={"username": "alice", "password": "wrong-123"}).status_code == 401
    assert client.post("/api/auth/login", headers=headers(client), json={"username": "alice", "password": "test-password-123"}).status_code == 200


def test_owner_isolation_multiple_chats_and_shared_likes(setup):
    app, repo, service, settings = setup
    a, b = register(app, "alice"), register(app, "bobby")
    first, second = new_chat(a), new_chat(a)
    result = ask(a, first, "保存的个人问题")
    assert result.status_code == 200
    assert service.calls[-1][1] == []  # 忽略客户端伪造历史。
    assert b.get(f"/api/conversations/{first}").status_code == 404
    assert ask(b, first).status_code == 404
    fid = result.json["feedback_id"]
    assert b.post("/api/feedback", headers=headers(b), json={"feedback_id": fid}).status_code == 404
    assert a.post("/api/feedback", headers=headers(a), json={"feedback_id": fid}).status_code == 200
    assert a.post("/api/feedback", headers=headers(a), json={"feedback_id": fid}).json["already_saved"]
    assert MySQLKnowledgeRepository(settings).snapshot()[1][0]["question"] == "保存的个人问题"
    assert a.get(f"/api/conversations/{first}").json["turns"][0]["liked"]
    assert a.get(f"/api/conversations/{second}").json["turns"] == []
    assert b.get("/api/conversations").json["conversations"] == []
    ask(a, second)
    assert service.calls[-1][1] == []


def test_idempotency_lock_preview_and_reload(setup):
    app, repo, service, _ = setup
    client = register(app, "alice")
    cid = new_chat(client)
    rid = secrets.token_hex(16)
    assert ask(client, cid, request_id=rid).status_code == 200
    assert ask(client, cid, request_id=rid).status_code == 200
    assert len(service.calls) == 1
    assert len(client.get(f"/api/conversations/{cid}").json["turns"]) == 1
    ask(client, cid, mode="retrieve")
    assert len(client.get(f"/api/conversations/{cid}").json["turns"]) == 1
    uid = repo.get_user(username="alice")["id"]
    token = repo.acquire(uid, cid)
    with pytest.raises(ConversationBusy):
        repo.acquire(uid, cid)
    repo.release(uid, cid, token)
    assert client.get("/").status_code == 200


def test_memory_five_then_four_raw_history_preserved(setup):
    app, repo, service, _ = setup
    client = register(app, "alice")
    cid = new_chat(client)
    uid = repo.get_user(username="alice")["id"]
    memory = MemoryManager(repo, service.llm)
    for i in range(4):
        ask(client, cid, f"问题{i}")
    memory.compact(uid, cid)
    assert len(service.llm.calls) == 0
    ask(client, cid, "第五问")
    memory.compact(uid, cid)
    assert len(service.llm.calls) == 1
    context = repo.context(uid, cid)
    assert context["memory_version"] == 1 and context["history"] == []
    assert context["retrieval_hint"] == "第五问"
    for i in range(3):
        ask(client, cid, f"新问题{i}")
    memory.compact(uid, cid)
    assert len(service.llm.calls) == 1
    ask(client, cid, "新增第四问")
    memory.compact(uid, cid)
    assert len(service.llm.calls) == 2
    assert repo.context(uid, cid)["memory_version"] == 2
    assert len(repo.read_conversation(uid, cid)["turns"]) == 9
    ask(client, cid, "后续追问")
    assert "第 2 版摘要" in service.calls[-1][2]
    assert service.calls[-1][1] == []
    another = new_chat(client)
    assert repo.context(uid, another)["summary"] == ""


def test_summary_failure_and_claim_exclusion(setup):
    app, repo, service, _ = setup
    client = register(app, "alice")
    cid = new_chat(client)
    uid = repo.get_user(username="alice")["id"]
    for _ in range(5):
        ask(client, cid)
    job = repo.claim_memory(uid, cid)
    assert job and repo.claim_memory(uid, cid) is None
    repo.finish_memory(uid, cid, job, error="retry")
    memory = MemoryManager(repo, service.llm)
    service.llm.fail = True
    memory.compact(uid, cid)
    assert repo.context(uid, cid)["memory_version"] == 0
    assert len(repo.read_conversation(uid, cid)["turns"]) == 5
    service.llm.fail = False
    memory.compact(uid, cid)
    assert repo.context(uid, cid)["memory_version"] == 1


def test_subject_survives_compaction_and_more_generic_followups(setup):
    app, repo, service, _settings = setup
    client = register(app, "alice")
    cid = new_chat(client)
    uid = repo.get_user(username="alice")["id"]
    for question in ("刊登怎么做", "需要哪些参数", "哪些是必填的", "有哪些可选参数", "参数是什么类型"):
        assert ask(client, cid, question).status_code == 200
    memory = MemoryManager(repo, service.llm)
    memory.compact(uid, cid)
    context = repo.context(uid, cid)
    assert context["history"] == [] and context["retrieval_hint"] == "刊登怎么做"
    assert ask(client, cid, "有哪些必填项").status_code == 200
    context = repo.context(uid, cid)
    assert context["history"] and context["retrieval_hint"] == "刊登怎么做"


def test_memory_prompt_is_scoped_context():
    messages = build_messages("当前问题", ContextBundle("本轮原文", [], 0, []), [], "只属于当前对话的摘要")
    assert "<conversation_memory>\n只属于当前对话的摘要" in messages[-1]["content"]
    assert "本轮原文" in messages[-1]["content"]


def test_database_failure_is_controlled(setup, monkeypatch):
    app, repo, service, _ = setup
    client = app.test_client()
    token = csrf(client)

    def unavailable(**kwargs):
        raise AccountStoreError("账户数据库暂不可用。")

    monkeypatch.setattr(repo, "get_user", unavailable)
    response = client.post("/api/auth/login", headers=token, json={"username": "alice", "password": "test-password-123"})
    assert response.status_code == 503
    assert response.json == {"error": "账户数据库暂不可用。"}
