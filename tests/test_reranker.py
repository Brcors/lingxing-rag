from collections import Counter
from dataclasses import replace
from types import SimpleNamespace

import pytest

from conftest import write_api
from rag.indexing.builder import IndexBuilder
from rag.retrieval.hybrid import HybridRetriever
from rag.retrieval.reranker import OptionalReranker, RerankError
from rag.service import RAGService
from rag.types import SearchHit


class ScoreModel:
    def __init__(self, scores):
        self.scores = scores
        self.calls = []

    def predict(self, pairs, **kwargs):
        self.calls.append((pairs, kwargs))
        return self.scores


def test_disabled_reranker_never_loads_or_downloads(settings, monkeypatch):
    reranker = OptionalReranker(settings)
    monkeypatch.setattr(reranker, "_load", lambda: pytest.fail("关闭时不应加载模型"))
    hits = [SearchHit(SimpleNamespace(embedding_text="接口片段"))]
    assert reranker.apply("问题", hits) is hits
    reranker.warmup()


def test_reranking_preserves_fusion_scores_and_stable_ties(settings):
    reranker = OptionalReranker(replace(settings, rerank_enabled=True))
    reranker._model = ScoreModel([-2.0, 4.0, 4.0])
    hits = [SearchHit(SimpleNamespace(embedding_text=text), score=0.03-i/1000)
            for i, text in enumerate(["不相关", "接口路径", "请求参数"])]
    original_scores = [h.score for h in hits]
    ranked = reranker.apply("接口参数", hits)
    assert ranked == [hits[1], hits[2], hits[0]]
    assert [h.score for h in hits] == original_scores
    assert hits[1].rerank_score == 4.0
    assert reranker._model.calls[0][0][1] == ("接口参数", "接口路径")


@pytest.mark.parametrize("scores", [[1.0], [float("nan"), 2.0], [float("inf"), 2.0]])
def test_invalid_scores_do_not_partially_modify_hits(settings, scores):
    reranker = OptionalReranker(replace(settings, rerank_enabled=True))
    reranker._model = ScoreModel(scores)
    hits = [SearchHit(SimpleNamespace(embedding_text="片段")) for _ in range(2)]
    with pytest.raises(RerankError):
        reranker.apply("问题", hits)
    assert all(h.rerank_score is None for h in hits)


def test_complete_model_cache_is_offline_and_model_identity_isolated(settings, monkeypatch):
    settings = replace(settings, rerank_backend="torch")
    reranker = OptionalReranker(settings)
    path = reranker.model_dir
    path.mkdir(parents=True)
    for name in ("config.json", "model.safetensors", "tokenizer.json", "tokenizer_config.json"):
        (path / name).write_text("test", encoding="utf-8")
    (path / "_revision.txt").write_text("pinned-revision", encoding="utf-8")
    (path / "_model_id.txt").write_text(settings.rerank_model, encoding="utf-8")
    monkeypatch.setattr("huggingface_hub.HfApi", lambda: pytest.fail("完整缓存不应请求网络"))
    assert reranker.download() == path
    other = OptionalReranker(replace(settings, rerank_model="BAAI/bge-reranker-v2-m3"))
    assert other.model_dir != path and not other.downloaded()
    (path / "_model_id.txt").write_text("wrong-model", encoding="utf-8")
    assert not reranker.downloaded()


def test_onnx_cache_requires_the_quantized_weight_and_pinned_identity(settings, monkeypatch):
    reranker = OptionalReranker(replace(settings, rerank_backend="onnx-int8"))
    path = reranker.model_dir
    (path / "onnx").mkdir(parents=True)
    for name in ("config.json", "tokenizer.json", "tokenizer_config.json"):
        (path / name).write_text("test", encoding="utf-8")
    (path / "_revision.txt").write_text("pinned", encoding="utf-8")
    (path / "_model_id.txt").write_text(settings.rerank_model, encoding="utf-8")
    assert not reranker.downloaded()
    (path / "onnx" / "model_int8.onnx").write_text("test", encoding="utf-8")
    monkeypatch.setattr("huggingface_hub.HfApi", lambda: pytest.fail("完整 ONNX 缓存不应联网"))
    assert reranker.download() == path


def test_onnx_length_batching_restores_scores_and_only_passes_model_inputs():
    import numpy as np
    from rag.retrieval.onnx_reranker import ONNXCrossEncoder

    encoder = ONNXCrossEncoder.__new__(ONNXCrossEncoder)
    encoder.max_length = 512
    encoder.input_names = {"input_ids", "attention_mask"}

    def tokenize(queries, documents, **kwargs):
        values = np.array([[int(d)] for d in documents], dtype=np.int64)
        return {"input_ids": values, "attention_mask": np.ones_like(values), "token_type_ids": values}

    class Session:
        batches = []

        def run(self, outputs, inputs):
            assert set(inputs) == {"input_ids", "attention_mask"}
            self.batches.append(inputs["input_ids"].shape[0])
            return [inputs["input_ids"] * 10.0]

    encoder.tokenizer, encoder.session = tokenize, Session()
    result = encoder.predict([("q", "100"), ("q", "2"), ("q", "30")], batch_size=2)
    assert result.tolist() == [1000.0, 20.0, 300.0]
    assert encoder.session.batches == [2, 1]


def test_reranker_sees_twenty_candidates_before_final_six_and_document_caps(settings, embedding):
    for i in range(8):
        write_api(settings.docs_dir / f"货件{i}.md", title=f"查询货件{i}")
    IndexBuilder(settings, embedding).build()
    retriever = HybridRetriever(replace(settings, rerank_enabled=True), embedding)

    class PreferLast:
        pairs = []

        def predict(self, pairs, **kwargs):
            self.pairs = pairs
            return [-i for i in range(len(pairs)-1)] + [100.0]

    model = PreferLast()
    retriever.reranker._model = model
    stats = {}
    _, hits, _, _ = retriever.search("查询货件流程", diagnostics=stats)
    assert len(model.pairs) == 20 and stats["rerank_candidates"] == 20
    assert stats["rerank_applied"] is True
    assert len(hits) == 6
    assert hits[0].chunk.embedding_text == model.pairs[-1][1]
    assert max(Counter(h.chunk.document_id for h in hits).values()) <= 2


def test_rerank_failure_keeps_retrieval_available_and_reports_fallback(settings, embedding):
    write_api(settings.docs_dir / "货件.md")
    IndexBuilder(settings, embedding).build()
    service = RAGService(replace(settings, rerank_enabled=True))
    service.embedding = embedding

    class BrokenModel:
        def predict(self, *args, **kwargs):
            raise RuntimeError("simulated inference failure")

    service.retriever.reranker._model = BrokenModel()
    bundle, stats = service.retrieve("查询货件")
    assert bundle.sources and stats["rerank_applied"] is False
    assert any("使用混合检索排序" in notice for notice in bundle.notices)
    assert "rerank_notice" not in stats
