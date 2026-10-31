from dataclasses import replace

from rag.context import ContextBuilder
from rag.indexing.builder import IndexBuilder
from rag.indexing.store import CorpusStore
from rag.ingestion.parser import MarkdownLoader
from rag.ingestion.chunker import SemanticChunker
from rag.retrieval.hybrid import reciprocal_rank_fusion, search_query
from rag.retrieval.lexical import LexicalIndex
from rag.types import SearchHit
from conftest import write_api


def test_lexical_keeps_identifiers_and_nested_field_paths(settings, embedding):
    write_api(settings.docs_dir / "FBA" / "查询.md")
    chunks = SemanticChunker(settings, embedding).build(MarkdownLoader(settings).load())
    index = LexicalIndex(chunks)
    tokens = index.tokenize("data>>list>>shipment_id /erp/test/shipments sellerId")
    assert "data>>list>>shipment_id" in tokens
    assert "shipment_id" in tokens
    assert "sellerid" in tokens
    assert index.search("shipment_id", 3)


def test_rrf_rewards_agreement_and_uses_rank_not_raw_score(settings, embedding):
    write_api(settings.docs_dir / "查询.md")
    chunks = SemanticChunker(settings, embedding).build(MarkdownLoader(settings).load())
    a, b, c = chunks[:3]
    result = reciprocal_rank_fusion({"bm25": [(a.chunk_id, 100), (b.chunk_id, 1)],
                                     "vector": [(c.chunk_id, 0.99), (b.chunk_id, 0.7)]},
                                    {x.chunk_id: x for x in chunks})
    assert result[0].chunk.chunk_id == b.chunk_id
    assert set(result[0].channels) == {"bm25", "vector"}


def test_context_restores_interface_and_budget(settings, embedding):
    write_api(settings.docs_dir / "FBA" / "查询.md")
    docs = MarkdownLoader(settings).load()
    chunks = SemanticChunker(settings, embedding).build(docs)
    chunk = next(c for c in chunks if c.kind == "parameters")
    bundle = ContextBuilder(replace(settings, context_budget=600)).build(
        "shipment_id 是什么", [SearchHit(chunk)], {d.document_id: d for d in docs}, {c.chunk_id: c for c in chunks})
    assert bundle.sources
    assert "/erp/test/shipments" in bundle.text
    assert "接口概览" in bundle.text
    assert bundle.estimated_tokens <= 600


def test_incremental_reuses_vectors_and_deleted_documents_disappear(settings, embedding):
    path_a = settings.docs_dir / "FBA" / "a.md"
    path_b = settings.docs_dir / "基础数据" / "b.md"
    write_api(path_a)
    write_api(path_b, title="查询店铺列表")
    builder = IndexBuilder(settings, embedding)
    first = builder.build()
    calls = embedding.computed
    assert builder.build()["changed"] is False
    assert embedding.computed == calls
    path_b.unlink()
    second = builder.build()
    assert second["document_count"] == 1
    assert second["reused_vectors"] == second["chunk_count"]
    assert embedding.computed == calls
    docs, chunks = CorpusStore(settings).load()
    assert [d.title for d in docs] == ["查询货件列表"]
    assert first["generation"] != second["generation"]


def test_failed_rebuild_keeps_previous_active_index(settings, embedding, monkeypatch):
    path = settings.docs_dir / "a.md"
    write_api(path)
    builder = IndexBuilder(settings, embedding)
    first = builder.build()
    write_api(path, title="更新后的标题")
    monkeypatch.setattr(embedding, "encode_documents", lambda _texts: (_ for _ in ()).throw(RuntimeError("simulated failure")))
    import pytest
    with pytest.raises(RuntimeError, match="simulated failure"):
        builder.build()
    assert CorpusStore(settings).active()["generation"] == first["generation"]


def test_followup_preserves_previous_user_intent():
    history = [{"role": "user", "content": "查询FBA货件列表"}, {"role": "assistant", "content": "回答"}]
    assert "查询FBA货件列表" in search_query("它的分页参数是什么", history)
    assert search_query("查询店铺列表", history) == "查询店铺列表"
