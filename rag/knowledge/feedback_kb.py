"""以用户点赞问答构建本地 BM25 + 向量辅助知识库。"""
import logging
import threading

import numpy as np

from rag.retrieval.lexical import LexicalIndex
from rag.types import Chunk, SearchHit
from rag.knowledge.feedback_buffer import FeedbackBuffer
from rag.knowledge.mysql_repository import MySQLKnowledgeRepository

logger = logging.getLogger(__name__)


def compact_sources(sources):
    fields = ("document_id", "title", "category", "api_path", "method", "heading", "source_url", "relative_path")
    return [{key: source.get(key, "") for key in fields} for source in sources[:10]]


def embed_text(question, answer):
    return f"问题：{question}\n\n回答：{answer[:6000]}"


class FeedbackKnowledgeBase:
    def __init__(self, settings, embedding):
        self.settings = settings
        self.embedding = embedding
        self.repository = MySQLKnowledgeRepository(settings)
        self.pending = FeedbackBuffer()
        self._lock = threading.RLock()
        self._snapshot_version = None
        self._chunks = {}
        self._vectors = {}
        self._lexical = None
        self._records = {}

    def initialize(self):
        self.repository.initialize()

    def register_answer(self, question, answer, sources, category=None):
        return self.pending.put(question, answer, compact_sources(sources), category)

    def save_liked_answer(self, feedback_id):
        record = self.pending.get(feedback_id)
        if record is None:
            raise KeyError("回答已过期或不属于当前服务进程，请重新提问后点赞。")
        return self.save_record(feedback_id, record)

    def save_record(self, feedback_id, record):
        record = {**record, "sources": compact_sources(record["sources"])}
        vector = np.asarray(self.embedding.encode_documents([embed_text(record["question"], record["answer"])])[0],
                            dtype="<f4")
        signature = self.embedding.signature()
        inserted = self.repository.save({"feedback_id": feedback_id, **record}, vector.tobytes(), signature)
        return {"saved": True, "already_saved": not inserted}

    def _refresh(self):
        version, rows = self.repository.snapshot()
        with self._lock:
            if version == self._snapshot_version:
                return
            chunks, vectors, records = [], {}, {}
            signature = self.embedding.signature()
            for row in rows:
                if row["embedding_signature"] != signature:
                    continue
                vector = np.frombuffer(row["embedding"], dtype="<f4")
                if vector.ndim != 1 or not len(vector) or not np.isfinite(vector).all():
                    continue
                chunk_id = f"mysql-qa-{row['id']}"
                text = embed_text(row["question"], row["answer"])
                chunk = Chunk(
                    chunk_id=chunk_id, document_id=chunk_id, parent_id=chunk_id,
                    section_order=0, order=0, title=row["question"], heading="用户点赞问答",
                    category=row["category"] or "通用问答", api_path="", method="", kind="liked_qa",
                    text=text, embedding_text=text, token_count=len(text),
                )
                chunks.append(chunk)
                vectors[chunk_id] = vector
                records[chunk_id] = row
            self._chunks = {chunk.chunk_id: chunk for chunk in chunks}
            self._vectors = vectors
            self._records = records
            self._lexical = LexicalIndex(chunks) if chunks else None
            self._snapshot_version = version

    def search(self, question, category=None, reranker=None):
        self._refresh()
        if not self._chunks:
            return []
        with self._lock:
            chunks = self._chunks
            vectors = self._vectors
            records = self._records
            lexical_index = self._lexical
            if category:
                ids = {chunk_id for chunk_id, chunk in chunks.items() if chunk.category == category}
            else:
                ids = set(chunks)
            if not ids:
                return []
            query_vector = np.asarray(self.embedding.encode_query(question), dtype=np.float32)
            vector_scores = {chunk_id: float(np.dot(vectors[chunk_id], query_vector))
                             for chunk_id in ids if vectors[chunk_id].shape == query_vector.shape}
            vector_ranking = sorted(vector_scores.items(), key=lambda item: item[1], reverse=True)[:20]
            lexical_ranking = lexical_index.search(question, min(20, len(ids)), category if category else None)
            lexical_ranking = [(chunk_id, score) for chunk_id, score in lexical_ranking if chunk_id in ids]

            ranks = {}
            for rank, (chunk_id, _) in enumerate(vector_ranking, 1):
                ranks.setdefault(chunk_id, {})["vector_rank"] = rank
            for rank, (chunk_id, _) in enumerate(lexical_ranking, 1):
                ranks.setdefault(chunk_id, {})["bm25_rank"] = rank
            hits = []
            bm25_scores = dict(lexical_ranking)
            for chunk_id, rank_data in ranks.items():
                score = (1.0 / (60 + rank_data["vector_rank"]) if "vector_rank" in rank_data else 0.0)
                score += (1.0 / (60 + rank_data["bm25_rank"]) if "bm25_rank" in rank_data else 0.0)
                hits.append(SearchHit(
                    chunks[chunk_id], score=score, vector_similarity=vector_scores.get(chunk_id),
                    bm25_score=bm25_scores.get(chunk_id),
                    channels=[name for name in ("vector" if "vector_rank" in rank_data else None,
                                                "bm25" if "bm25_rank" in rank_data else None) if name],
                ))
            hits.sort(key=lambda hit: hit.score, reverse=True)
            candidates = hits[:min(8, len(hits))]
            if reranker is not None and self.settings.rerank_enabled and candidates:
                try:
                    candidates = reranker.apply(question, candidates)
                except Exception:
                    logger.exception("MySQL 问答知识库重排序失败，回退混合排序")

            results = []
            for hit in candidates:
                # 词面精确匹配允许识别规范别名；纯语义命中需达到保守阈值。
                if (hit.vector_similarity or 0.0) < self.settings.mysql_kb_min_similarity and hit.bm25_score is None:
                    continue
                row = records[hit.chunk.chunk_id]
                if category and row["category"] != category:
                    continue
                results.append({
                    "id": int(row["id"]), "question": row["question"], "answer": row["answer"],
                    "category": row["category"], "sources": row["sources"],
                    "similarity": hit.vector_similarity, "score": hit.rerank_score if hit.rerank_score is not None else hit.score,
                })
                if len(results) >= self.settings.mysql_kb_top_k:
                    break
            return results
