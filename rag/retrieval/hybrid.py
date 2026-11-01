from collections import Counter
import logging
import re
import threading
import time

from rag.indexing.store import CorpusStore, VectorStore
from rag.retrieval.lexical import LexicalIndex
from rag.retrieval.reranker import OptionalReranker, RerankError
from rag.types import SearchHit
from rag.retrieval.query import context_anchor, is_detail_followup, search_query

logger = logging.getLogger(__name__)


def reciprocal_rank_fusion(rankings, chunks, constant=60):
    merged = {}
    for channel, ranking in rankings.items():
        for rank, (chunk_id, raw_score) in enumerate(ranking, start=1):
            hit = merged.setdefault(chunk_id, SearchHit(chunks[chunk_id]))
            hit.score += 1.0 / (constant + rank)
            hit.channels.append(channel)
            if channel == "vector":
                hit.vector_similarity = raw_score
            else:
                hit.bm25_score = raw_score
    return sorted(merged.values(), key=lambda hit: hit.score, reverse=True)


class HybridRetriever:
    def __init__(self, settings, embedding):
        self.settings, self.embedding = settings, embedding
        self.store = CorpusStore(settings)
        self.vectors = VectorStore(settings)
        self.reranker = OptionalReranker(settings)
        self._generation = None
        self._lock = threading.RLock()
        self.documents = {}
        self.chunks = {}

    def reload(self):
        active = self.store.active()
        if not active:
            raise FileNotFoundError("索引尚未建立，请运行 python manage.py index。")
        if active.get("embedding_signature") != self.embedding.signature():
            raise ValueError("模型或切块配置已经变化，请重新建立索引后再检索。")
        if active["generation"] != self._generation:
            documents, chunks = self.store.load(active)
            self.documents = {d.document_id: d for d in documents}
            self.chunks = {c.chunk_id: c for c in chunks}
            self.lexical = LexicalIndex(chunks)
            self.collection = self.vectors.get(active["collection"])
            self._generation = active["generation"]
        return active

    def search(self, question, category=None, history=None, *, diagnostics=None,
               rerank=True, finalize=True, intent=None):
        query = search_query(question, history)
        with self._lock:
            active = self.reload()
            if category and category not in {c.split("/")[0] for c in active["categories"]}:
                raise ValueError("指定的业务分类不存在。")
            lexical = self.lexical.search(query, self.settings.bm25_top_k, category)
            # Chroma category 存储完整层级，通过 $in 精确匹配所有子分类。
            matching_categories = [c for c in active["categories"] if not category or c.split("/")[0] == category]
            where = {"category": {"$in": matching_categories}} if category else None
            kwargs = {"where": where} if where else {}
            result = self.collection.query(
                query_embeddings=[self.embedding.encode_query(query)],
                n_results=min(self.settings.vector_top_k, active["chunk_count"]),
                include=["distances"], **kwargs,
            )
            vector = [(cid, 1.0 - distance) for cid, distance in zip(result["ids"][0], result["distances"][0])]
            hits = reciprocal_rank_fusion({"bm25": lexical, "vector": vector}, self.chunks, self.settings.rrf_k)
            compact = re.sub(r"\s+", "", question).lower()
            for hit in hits:
                chunk = hit.chunk
                title = re.sub(r"\s+", "", chunk.title).lower()
                if title and title in compact:
                    hit.score += 0.012
                if chunk.api_path and chunk.api_path.lower() in question.lower():
                    hit.score += 0.025
                preferred = {"parameters": {"parameters"}, "response": {"response"}, "endpoint": {"overview"}}
                if chunk.kind in preferred.get(intent, set()):
                    hit.score += 0.008
                document = self.documents[chunk.document_id]
                if document.is_legacy and "旧" not in question:
                    hit.score *= 0.8
            hits.sort(key=lambda hit: hit.score, reverse=True)
            hits = hits[:self.settings.rerank_candidates]
            rerank_start = time.monotonic()
            applied, notice = False, None
            if rerank and self.settings.rerank_enabled and hits:
                try:
                    hits = self.reranker.apply(query, hits)
                    applied = True
                except RerankError as error:
                    logger.warning("本地重排序失败，回退 RRF 顺序", exc_info=True)
                    notice = str(error)
            if diagnostics is not None:
                diagnostics.update(
                    rerank_applied=applied,
                    rerank_model=self.settings.rerank_model if self.settings.rerank_enabled else None,
                    rerank_backend=self.settings.rerank_backend if self.settings.rerank_enabled else None,
                    rerank_candidates=len(hits) if applied else 0,
                    rerank_seconds=round(time.monotonic() - rerank_start, 3),
                )
                if notice:
                    diagnostics["rerank_notice"] = notice
            if finalize:
                hits = self.select_final_hits(question, hits, applied, diagnostics, intent=intent)
            else:
                hits = hits[:self.settings.rerank_candidates]
            return query, hits, dict(self.documents), dict(self.chunks)

    def select_final_hits(self, question, hits, rerank_applied, diagnostics=None, intent=None):
        """Abstain if even the strongest reranked candidate is below the relevance floor."""
        if rerank_applied and hits:
            best_score = max(hit.rerank_score for hit in hits if hit.rerank_score is not None)
            if diagnostics is not None:
                diagnostics["rerank_best_score"] = round(best_score, 4)
                diagnostics["rerank_min_score"] = self.settings.rerank_min_score
            if best_score < self.settings.rerank_min_score:
                if diagnostics is not None:
                    diagnostics["rerank_low_confidence"] = True
                    diagnostics["rerank_filtered_candidates"] = len(hits)
                return []

        preferred = {"parameters": "parameters", "response": "response", "endpoint": "overview"}.get(intent)
        if hits and preferred:
            best_document = hits[0].chunk.document_id
            relevant = [hit for hit in hits if hit.chunk.document_id == best_document and
                        hit.chunk.kind == preferred and (not rerank_applied or
                        (hit.rerank_score is not None and hit.rerank_score >= self.settings.rerank_min_score))]
            hits = relevant + [hit for hit in hits if hit not in relevant]
        chosen, counts = [], Counter()
        workflow = any(word in question for word in ("流程", "步骤", "依次", "哪些接口", "前置", "依赖"))
        cap = 2 if workflow else 3
        for hit in hits:
            if counts[hit.chunk.document_id] < cap:
                chosen.append(hit)
                counts[hit.chunk.document_id] += 1
            if len(chosen) >= self.settings.final_top_k:
                break
        return chosen
