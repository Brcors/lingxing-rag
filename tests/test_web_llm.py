from dataclasses import replace
import json

import pytest

from app import create_app
from conftest import authenticated_client
from app.rendering import render_markdown
from rag.generation.client import DeepSeekClient, LLMError


class StubService:
    def events(self, *args, **kwargs):
        yield {"type": "status", "message": "检索中"}
        yield {"type": "token", "text": "测试回答"}
        yield {"type": "done", "answer": "**测试回答**", "sources": [], "stats": {}, "notices": [], "mode": "answer"}

    def ask(self, *args):
        return list(self.events(*args))[-1]


def test_page_api_validation_and_stream_contract(settings):
    client = authenticated_client(settings, StubService())
    assert client.get("/").status_code == 200
    assert client.get("/api/status").json["llm_configured"] is False
    assert client.post("/api/ask", json={"question": ""}).status_code == 400
    assert client.post("/api/ask", json={"question": "a" * 4001}).status_code == 400
    assert client.post("/api/ask", json={"question": "问题", "history": "bad"}).status_code == 200
    assert client.post("/api/ask", json={"question": "问题"}, headers={"Origin": "https://evil.example"}).status_code == 403
    result = client.post("/api/ask/stream", json={"question": "问题"})
    events = [json.loads(line) for line in result.data.decode().splitlines()]
    assert [e["type"] for e in events] == ["status", "token", "done"]
    assert "<strong>测试回答</strong>" in events[-1]["answer_html"]
    assert client.post("/api/ask", json={"question": "问题"}).json["answer"] == "**测试回答**"


def test_markdown_blocks_scripts_and_unsafe_urls():
    html = render_markdown("<script>alert(1)</script>\n\n[x](javascript:alert(1))\n\n[S1]\n\n```python\nprint(1)\n```")
    assert "<script>" not in html
    assert 'href="javascript:' not in html
    assert 'href="#source-S1"' in html
    assert 'class="language-python"' in html


class FakeResponse:
    def __init__(self, status=200, lines=None, data=None):
        self.status_code, self.ok = status, 200 <= status < 300
        self.lines, self.data, self.closed = lines or [], data, False

    def iter_lines(self, **kwargs):
        yield from self.lines

    def json(self):
        return self.data

    def close(self):
        self.closed = True

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()


def test_deepseek_stream_handles_keepalive_usage_and_finish(settings, monkeypatch):
    response = FakeResponse(lines=["", ": keepalive", 'data: {"choices":[]}',
                                  'data: {"choices":[{"delta":{"content":"你好"},"finish_reason":null}]}',
                                  'data: {"choices":[{"delta":{},"finish_reason":"stop"}],"usage":{}}',
                                  "data: [DONE]"])
    calls = []

    def post(url, **kwargs):
        calls.append((url, kwargs))
        return response

    monkeypatch.setattr("rag.generation.client.requests.post", post)
    llm = DeepSeekClient(replace(settings, api_key="test-key"))
    assert "".join(llm.stream([{"role": "user", "content": "问题"}])) == "你好"
    assert calls[0][0] == "https://api.deepseek.com/chat/completions"
    payload = calls[0][1]["json"]
    assert payload["model"] == "deepseek-flash"
    assert payload["thinking"] == {"type": "disabled"}
    assert response.closed


def test_deepseek_reports_missing_key_and_sanitized_auth_failure(settings, monkeypatch):
    with pytest.raises(LLMError, match="DEEPSEEK_API_KEY"):
        DeepSeekClient(settings).complete([])
    monkeypatch.setattr("rag.generation.client.requests.post", lambda *_args, **_kwargs: FakeResponse(status=401, data={"secret": "sensitive"}))
    with pytest.raises(LLMError, match="API key 无效") as error:
        DeepSeekClient(replace(settings, api_key="private-key")).complete([])
    assert "private-key" not in str(error.value)


def test_incomplete_stream_is_an_error(settings, monkeypatch):
    monkeypatch.setattr("rag.generation.client.requests.post", lambda *_args, **_kwargs: FakeResponse(lines=[
        'data: {"choices":[{"delta":{"content":"一部分"}}]}'
    ]))
    with pytest.raises(LLMError, match="连接提前结束"):
        list(DeepSeekClient(replace(settings, api_key="test-key")).stream([]))


def test_complete_rag_web_flow_uses_retrieved_evidence_and_filters_bad_citations(settings, embedding, monkeypatch):
    from conftest import write_api
    from rag.indexing.builder import IndexBuilder
    from rag.service import RAGService
    write_api(settings.docs_dir / "FBA" / "货件.md")
    settings = replace(settings, api_key="test-key")
    IndexBuilder(settings, embedding).build()
    sent_payload = []

    def post(_url, **kwargs):
        sent_payload.append(kwargs["json"])
        answer = "接口路径为 `/erp/test/shipments`。[S1] 另一个引用[S999]"
        return FakeResponse(lines=[
            "data: " + json.dumps({"choices": [{"delta": {"content": answer}, "finish_reason": None}]}),
            'data: {"choices":[{"delta":{},"finish_reason":"stop"}]}', "data: [DONE]"
        ])

    monkeypatch.setattr("rag.generation.client.requests.post", post)
    rag_service = RAGService(settings)
    rag_service.embedding = embedding
    client = authenticated_client(settings, rag_service)
    result = client.post("/api/ask", json={"question": "查询货件列表需要 shipment_id 吗？"})
    assert result.status_code == 200
    data = result.json
    assert data["sources"]
    assert "[S999]" not in data["answer"]
    assert any("无效来源" in notice for notice in data["notices"])
    assert "/erp/test/shipments" in sent_payload[0]["messages"][-1]["content"]
    assert "shipment_id" in sent_payload[0]["messages"][-1]["content"]
    assert "检索" not in sent_payload[0]["model"]
    source = data["sources"][0]
    assert client.get("/documents/" + source["document_id"]).status_code == 200


def test_missing_key_still_allows_real_retrieval_preview(settings, embedding, monkeypatch):
    from conftest import write_api
    from rag.indexing.builder import IndexBuilder
    from rag.service import RAGService
    write_api(settings.docs_dir / "FBA" / "货件.md")
    IndexBuilder(settings, embedding).build()
    monkeypatch.setattr("rag.generation.client.requests.post", lambda *_a, **_kw: pytest.fail("不应发起付费请求"))
    rag_service = RAGService(settings)
    rag_service.embedding = embedding
    result = authenticated_client(settings, rag_service).post("/api/ask", json={"question": "货件查询"})
    assert result.status_code == 200
    assert result.json["mode"] == "retrieval"
    assert result.json["sources"]
