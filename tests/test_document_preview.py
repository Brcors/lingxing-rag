from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from pathlib import Path
import threading

import pytest

from conftest import authenticated_client
from rag.indexing.store import CorpusStore, atomic_json
from rag.ingestion.parser import MarkdownLoader
from test_web_llm import StubService


def publish(settings, generation, documents):
    # 预览不应依赖切片文件、向量库或模型。
    root = settings.data_dir / "generations" / generation
    atomic_json(root / "documents.json", [d.to_dict() for d in documents])
    atomic_json(settings.data_dir / "active.json", {"generation": generation})
    return root


def example(title="查询货件列表", path="FBA/查询.md"):
    return MarkdownLoader.parse(f"# {title}\n## 请求参数\nshipment_id 货件编号", path, "FBA")


def count_corpus_reads(monkeypatch):
    reads = []
    original = Path.read_text

    def read(path, *args, **kwargs):
        if path.name in ("documents.json", "chunks.json"):
            reads.append(path)
        return original(path, *args, **kwargs)

    monkeypatch.setattr(Path, "read_text", read)
    return reads


def test_preview_reuses_documents_across_requests_without_loading_chunks(settings, monkeypatch):
    document = example()
    root = publish(settings, "first", [document])
    reads = count_corpus_reads(monkeypatch)
    client = authenticated_client(settings, StubService())
    for _ in range(3):
        response = client.get("/documents/" + document.document_id)
        assert response.status_code == 200
        assert document.title in response.get_data(as_text=True)
        assert "shipment_id" in response.get_data(as_text=True)
    assert client.get("/documents/missing").status_code == 404
    assert reads == [root.resolve() / "documents.json"]


def test_new_generation_updates_documents_and_removes_old_entries(settings, monkeypatch):
    document = example()
    removed = example("删除的接口", "FBA/删除.md")
    publish(settings, "first", [document, removed])
    client = authenticated_client(settings, StubService())
    assert client.get("/documents/" + removed.document_id).status_code == 200
    assert client.get("/documents/" + document.document_id).status_code == 200

    updated = replace(document, title="新版货件接口", content="## 新版参数\nnew_field")
    publish(settings, "second", [updated])
    reads = count_corpus_reads(monkeypatch)
    response = client.get("/documents/" + document.document_id)
    assert response.status_code == 200
    assert "新版货件接口" in response.get_data(as_text=True)
    assert "new_field" in response.get_data(as_text=True)
    assert client.get("/documents/" + removed.document_id).status_code == 404
    assert len(reads) == 1


def test_same_generation_in_different_data_directories_is_isolated(settings):
    document = example()
    other = replace(settings, data_dir=settings.data_dir.parent / "other-data")
    publish(settings, "same", [document])
    publish(other, "same", [replace(document, title="另一个项目")])
    assert CorpusStore(settings).get_document(document.document_id).title == document.title
    assert CorpusStore(other).get_document(document.document_id).title == "另一个项目"
    assert CorpusStore(settings).get_document(document.document_id).title == document.title


def test_concurrent_first_previews_only_parse_documents_once(settings, monkeypatch):
    document = example()
    publish(settings, "concurrent", [document])
    reads = count_corpus_reads(monkeypatch)
    ready = threading.Barrier(8)

    def preview(_):
        ready.wait(timeout=5)
        return CorpusStore(settings).get_document(document.document_id)

    with ThreadPoolExecutor(max_workers=8) as pool:
        documents = list(pool.map(preview, range(8)))
    assert all(d is documents[0] for d in documents)
    assert len(reads) == 1


def test_failed_generation_load_does_not_publish_partial_cache(settings):
    document = example()
    publish(settings, "first", [document])
    store = CorpusStore(settings)
    assert store.get_document(document.document_id).title == document.title

    root = publish(settings, "broken", [document])
    atomic_json(root / "documents.json", [document.to_dict(), {"document_id": "invalid"}])
    for _ in range(2):
        with pytest.raises(KeyError):
            store.get_document(document.document_id)
    updated = replace(document, title="修复后的文档")
    atomic_json(root / "documents.json", [updated.to_dict()])
    assert store.get_document(document.document_id).title == updated.title


def test_missing_index_and_missing_generation_return_404_without_stale_content(settings):
    document = example()
    client = authenticated_client(settings, StubService())
    url = "/documents/" + document.document_id
    assert client.get(url).status_code == 404
    publish(settings, "first", [document])
    assert client.get(url).status_code == 200
    (settings.data_dir / "active.json").unlink()
    assert client.get(url).status_code == 404
    atomic_json(settings.data_dir / "active.json", {"generation": "missing"})
    assert client.get(url).status_code == 404
