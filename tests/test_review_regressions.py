from dataclasses import replace
from types import SimpleNamespace

import pytest
import requests

from app import create_app
from app.rendering import render_markdown
from conftest import write_api, authenticated_client
from rag.context import ContextBuilder
from rag.embedding import LocalEmbedding
from rag.generation.client import DeepSeekClient, LLMError
from rag.indexing.builder import IndexBuilder
from rag.ingestion.chunker import SemanticChunker
from rag.ingestion.parser import MarkdownLoader
from rag.types import SearchHit
from test_web_llm import FakeResponse, StubService


def test_parent_expansion_does_not_evict_another_retrieved_source(settings, embedding):
    a = MarkdownLoader.parse("# 接口A\n## 请求参数\n命中A\n" + "说明" * 220, "a.md", "测试")
    b = MarkdownLoader.parse("# 接口B\n## 请求参数\n命中B", "b.md", "测试")
    chunks = SemanticChunker(settings, embedding).build([a, b])
    selected = [next(c for c in chunks if "命中A" in c.text), next(c for c in chunks if "命中B" in c.text)]
    result = ContextBuilder(replace(settings, context_budget=700)).build(
        "对比两个接口参数", [SearchHit(c) for c in selected],
        {d.document_id: d for d in (a, b)}, {c.chunk_id: c for c in chunks})
    assert "命中A" in result.text and "命中B" in result.text
    assert result.estimated_tokens <= 700


def test_four_backtick_fence_keeps_nested_three_backticks_inside_code():
    source = "# 示例\n## 请求示例\n````markdown\n```python\n## 这不是章节\nprint(1)\n```\n````\n## 返回结果\n完成"
    document = MarkdownLoader.parse(source, "a.md", "测试")
    assert [s.heading for s in document.sections] == ["请求示例", "返回结果"]
    assert "## 这不是章节" in document.sections[0].text


def test_split_indented_code_preserves_valid_closing_fence(settings, embedding):
    source = "# 示例\n## 请求示例\n  ````python\n" + "print('hello')\n" * 80 + "  ````"
    document = MarkdownLoader.parse(source, "a.md", "测试")
    chunks = SemanticChunker(settings, embedding).build([document])
    assert len(chunks) > 1
    assert all(c.text.splitlines()[-1].strip() == "````" for c in chunks)


def test_unclosed_code_block_does_not_lose_last_line(settings, embedding):
    source = "# 示例\n## 请求示例\n```python\n" + "print(1)\n" * 80 + "SENTINEL_LAST_LINE"
    document = MarkdownLoader.parse(source, "a.md", "测试")
    chunks = SemanticChunker(settings, embedding).build([document])
    assert any("SENTINEL_LAST_LINE" in c.text for c in chunks)


def test_citation_linking_does_not_modify_code_or_existing_links():
    rendered = render_markdown('```python\nlabel = "[S1]"\n```\n\n`[S2]`\n\n[S3]\n\n[S4](https://example.com)')
    assert '#source-S1' not in rendered
    assert '#source-S2' not in rendered
    assert 'href="#source-S3"' in rendered
    assert 'href="https://example.com"' in rendered


def test_read_timeout_does_not_repeat_a_potentially_billed_request(settings, monkeypatch):
    calls = []

    def post(*args, **kwargs):
        calls.append(1)
        raise requests.ReadTimeout("read timed out")

    monkeypatch.setattr("rag.generation.client.requests.post", post)
    monkeypatch.setattr("rag.generation.client.time.sleep", lambda _seconds: None)
    with pytest.raises(LLMError):
        DeepSeekClient(replace(settings, api_key="test-key")).complete([])
    assert len(calls) == 1


def test_malformed_stream_schema_is_a_controlled_error(settings, monkeypatch):
    monkeypatch.setattr("rag.generation.client.requests.post", lambda *_a, **_kw: FakeResponse(lines=['data: []']))
    with pytest.raises(LLMError):
        list(DeepSeekClient(replace(settings, api_key="test-key")).stream([]))


def test_unknown_api_modes_are_rejected_before_model_call(settings):
    client = authenticated_client(settings, StubService())
    assert client.post("/api/ask", json={"question": "问题", "mode": "retrive"}).status_code == 400
    assert client.post("/api/ask/stream", json={"question": "问题", "mode": None}).status_code == 400


def test_different_embedding_model_does_not_reuse_existing_weights(settings, monkeypatch):
    root = settings.model_dir
    root.mkdir()
    for name in ("model.safetensors", "modules.json", "config.json", "tokenizer_config.json", "_revision.txt"):
        (root / name).write_text("model-A", encoding="utf-8")
    (root / "_model_id.txt").write_text("example/model-A", encoding="utf-8")
    calls = []
    monkeypatch.setattr("huggingface_hub.HfApi", lambda: SimpleNamespace(model_info=lambda _id: SimpleNamespace(sha="revision-B")))

    def download(**kwargs):
        from pathlib import Path
        target = Path(kwargs["local_dir"])
        target.mkdir(parents=True, exist_ok=True)
        for name in ("model.safetensors", "modules.json", "config.json", "tokenizer_config.json", "tokenizer.json", "1_Pooling/config.json"):
            (target / name).parent.mkdir(parents=True, exist_ok=True)
            (target / name).write_text("model-B", encoding="utf-8")
        calls.append(kwargs)

    monkeypatch.setattr("huggingface_hub.snapshot_download", download)
    target = LocalEmbedding(replace(settings, embedding_model="example/model-B")).download()
    assert calls and calls[0]["repo_id"] == "example/model-B"
    assert target != root
    assert (root / "model.safetensors").read_text(encoding="utf-8") == "model-A"


def test_broken_optional_vector_cache_is_recomputed(settings, embedding):
    path = settings.docs_dir / "a.md"
    write_api(path)
    first = IndexBuilder(settings, embedding).build()
    cache = settings.data_dir / "generations" / first["generation"] / "vectors.npy"
    cache.write_bytes(b"broken cache")
    write_api(path, extra="| new_field | 新增参数 | 否 |")
    second = IndexBuilder(settings, embedding).build()
    assert second["changed"] is True
    assert second["reused_vectors"] == 0


def test_citation_validation_preserves_literal_code(settings):
    from rag.service import RAGService
    from rag.types import ContextBundle
    answer = '```python\nlabel = "[S999]"\n```\n\n`[S998]`\n\n事实[S1] 错误引用[S997]'
    checked, notices = RAGService.check_citations(answer, ContextBundle("", [{"id": "S1"}], 0))
    assert 'label = "[S999]"' in checked
    assert '`[S998]`' in checked
    assert '[S997]' not in checked
    assert '[S1]' in checked
    assert notices


@pytest.mark.parametrize("stream", [False, True])
def test_abnormal_model_finish_is_not_reported_as_success(settings, monkeypatch, stream):
    choice = {"delta": {"content": "部分内容"}, "message": {"content": "部分内容"},
              "finish_reason": "insufficient_system_resource"}
    import json
    response = FakeResponse(data={"choices": [choice]},
                            lines=["data: " + json.dumps({"choices": [choice]}), "data: [DONE]"])
    monkeypatch.setattr("rag.generation.client.requests.post", lambda *_a, **_kw: response)
    client = DeepSeekClient(replace(settings, api_key="test-key"))
    with pytest.raises(LLMError, match="未正常完成"):
        list(client.stream([])) if stream else client.complete([])


def test_tokenizer_is_not_used_while_the_same_model_encodes(settings):
    from concurrent.futures import ThreadPoolExecutor
    from threading import Event
    import numpy as np
    encoding, release, tokenizer_called, count_started = Event(), Event(), Event(), Event()

    class Tokenizer:
        def encode(self, _text, **kwargs):
            tokenizer_called.set()
            if encoding.is_set():
                raise RuntimeError("concurrent tokenizer access")
            return [1, 2]

    class Model:
        tokenizer = Tokenizer()

        def encode(self, _texts, **kwargs):
            encoding.set()
            assert release.wait(2)
            encoding.clear()
            return np.array([[1.0, 0.0]])

    local = LocalEmbedding(settings)
    local._model = Model()

    def count():
        count_started.set()
        return local.count("test")

    with ThreadPoolExecutor(max_workers=2) as pool:
        encode_future = pool.submit(local.encode_documents, ["test"])
        try:
            assert encoding.wait(1)
            count_future = pool.submit(count)
            assert count_started.wait(1)
            assert not tokenizer_called.wait(0.1)
        finally:
            release.set()
        assert encode_future.result() == [[1.0, 0.0]]
        assert count_future.result() == 2


def test_empty_environment_key_does_not_hide_configured_file_key(monkeypatch):
    from rag.settings import Settings
    monkeypatch.setattr("rag.settings.config.DEEPSEEK_API_KEY", "file-key")
    monkeypatch.setenv("DEEPSEEK_API_KEY", "   ")
    assert Settings.load().api_key == "file-key"
    monkeypatch.setenv("DEEPSEEK_API_KEY", "env-key")
    assert Settings.load().api_key == "env-key"
