import json
import secrets

from test_accounts_memory import setup, register, headers, new_chat, ask
from test_guests import enter_guest, count


def test_search_rename_delete_are_owner_scoped(setup):
    app, repo, service, settings = setup
    alice, bob = register(app, "alice"), register(app, "bob")
    a, b = new_chat(alice), new_chat(bob)
    ask(alice, a, "只属于甲的独特字段百分号%_")
    ask(bob, b, "只属于乙的秘密接口")
    response = alice.patch(f"/api/conversations/{a}", headers=headers(alice), json={"title": "我的接口笔记"})
    assert response.status_code == 200
    assert alice.get("/api/conversations?q=我的接口").json["conversations"][0]["id"] == a
    assert alice.get("/api/conversations?q=%25_").json["conversations"][0]["id"] == a
    assert not alice.get("/api/conversations?q=只属于乙").json["conversations"]
    assert alice.patch(f"/api/conversations/{b}", headers=headers(alice), json={"title": "越权"}).status_code == 404
    assert alice.delete(f"/api/conversations/{b}", headers=headers(alice)).status_code == 404
    assert alice.get(f"/api/conversations/{b}/export").status_code == 404
    for title in (None, "", "x" * 101, "换\n行"):
        assert alice.patch(f"/api/conversations/{a}", headers=headers(alice), json={"title": title}).status_code == 400
    assert alice.delete(f"/api/conversations/{a}", headers=headers(alice)).status_code == 200
    assert alice.get(f"/api/conversations/{a}").status_code == 404
    assert bob.get(f"/api/conversations/{b}").status_code == 200
    assert count(repo, "chat_turns") == 1


def test_export_includes_all_pages_and_document_dates(setup):
    app, repo, service, settings = setup
    client = register(app, "alice")
    cid = new_chat(client)
    owner = repo.get_user(username="alice")["id"]
    lease = repo.acquire(owner, cid)
    for i in range(205):
        event = {"answer": f"导出回答{i}", "mode": "answer", "sources": [
            {"id": "S1", "title": "接口文档", "source_url": "https://example.com/api", "updated_at": "2026-11-01T00:00:00Z"}]}
        repo.save_turn(owner, cid, lease, f"{i:032x}", f"导出问题{i}", None, event)
    repo.release(owner, cid, lease)
    response = client.get(f"/api/conversations/{cid}/export")
    assert response.status_code == 200 and "attachment" in response.headers["Content-Disposition"]
    text = response.data.decode("utf-8")
    assert text.count("## 问题\n") == 205 and "导出问题204" in text
    assert "2026-11-01T00:00:00Z" in text and "https://example.com/api" in text


def test_deleting_last_guest_chat_preserves_guest_identity(setup):
    app, repo, service, settings = setup
    visitor, gid, cid = enter_guest(app)
    other, other_gid, other_cid = enter_guest(app)
    assert visitor.get("/api/documents/versions").status_code == 200
    assert visitor.post("/api/documents/sync", headers=headers(visitor), json={"check_only": True}).status_code == 403
    assert visitor.delete(f"/api/conversations/{other_cid}", headers=headers(visitor)).status_code == 404
    response = visitor.delete(f"/api/conversations/{cid}", headers=headers(visitor))
    assert response.status_code == 200 and response.json["replacement"]
    assert visitor.get("/api/conversations").json["conversations"][0]["id"] == response.json["replacement"]["id"]
    with visitor.session_transaction() as session:
        assert session["guest_id"] == gid
    assert other.get(f"/api/conversations/{other_cid}").status_code == 200


def test_cancel_stops_save_and_releases_conversation_lease(setup):
    app, repo, service, settings = setup
    client = register(app, "alice")
    bob = register(app, "bob")
    cid = new_chat(client)
    rid = secrets.token_hex(16)
    def events(*args, cancel_token=None, **kwargs):
        yield {"type": "status", "message": "生成中"}
        cancel_token.check()
        yield {"type": "done", "mode": "answer", "answer": "不应保存", "sources": [], "stats": {}, "notices": []}
    service.events = events
    response = client.post("/api/ask/stream", headers=headers(client), buffered=False,
        json={"question": "问题", "conversation_id": cid, "request_id": rid})
    assert response.status_code == 200
    assert bob.post(f"/api/conversations/{cid}/cancel", headers=headers(bob), json={"request_id": rid}).status_code == 404
    second = app.test_client()
    second.set_cookie("lingxing_session", client.get_cookie("lingxing_session").value)
    cancelled = second.post(f"/api/conversations/{cid}/cancel", headers=headers(second), json={"request_id": rid})
    assert cancelled.status_code == 200
    events = [json.loads(line) for line in response.response]
    response.close()
    assert any(event["type"] == "cancelled" for event in events)
    assert count(repo, "chat_turns") == 0
    owner = repo.get_user(username="alice")["id"]
    token = repo.acquire(owner, cid)
    repo.release(owner, cid, token)


def test_cancel_before_stream_and_delete_busy_chat(setup):
    app, repo, service, settings = setup
    client = register(app, "alice")
    cid = new_chat(client)
    rid = secrets.token_hex(16)
    assert client.post(f"/api/conversations/{cid}/cancel", headers=headers(client), json={"request_id": rid}).status_code == 200
    response = client.post("/api/ask/stream", headers=headers(client),
        json={"question": "问题", "conversation_id": cid, "request_id": rid})
    assert b'"cancelled"' in response.data and not service.calls
    owner = repo.get_user(username="alice")["id"]
    lease = repo.acquire(owner, cid)
    assert client.delete(f"/api/conversations/{cid}", headers=headers(client)).status_code == 409
    repo.release(owner, cid, lease)
