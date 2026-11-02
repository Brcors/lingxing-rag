import re
import logging
import threading
import time

from rag.context import ContextBuilder
from rag.citations import transform_prose_citations
from rag.embedding import LocalEmbedding
from rag.generation.client import DeepSeekClient
from rag.generation.decomposer import QuestionDecomposer, should_decompose
from rag.generation.prompts import prepare_prompt
from rag.generation.budget import message_tokens
from rag.generation.errors import ContextLengthError
from rag.indexing.store import CorpusStore
from rag.knowledge.feedback_kb import FeedbackKnowledgeBase
from rag.retrieval.hybrid import HybridRetriever
from rag.retrieval.reranker import RerankError
from rag.retrieval.flow import RetrievalFlow
from rag.retrieval.routing import classify, parameter_clarification
from rag.types import SearchHit

logger = logging.getLogger(__name__)


class RAGService:
    def __init__(self, settings):
        self.settings = settings
        self.embedding = LocalEmbedding(settings)
        self.context_builder = ContextBuilder(settings)
        self.feedback_kb = FeedbackKnowledgeBase(settings, self.embedding)
        self.llm = DeepSeekClient(settings)
        self.decomposer = QuestionDecomposer(settings, self.llm)
        self._retriever = None
        self._lock = threading.Lock()

    @property
    def retriever(self):
        if self._retriever is None:
            with self._lock:
                if self._retriever is None:
                    self._retriever = HybridRetriever(self.settings, self.embedding)
        return self._retriever

    def retrieve(self, question, category=None, history=None, extra_queries=None, intent=None):
        start = time.monotonic()
        diagnostics = {}
        queries = []
        seen = set()
        for candidate in [question, *(extra_queries or [])]:
            if not isinstance(candidate, str) or not candidate.strip():
                continue
            key = re.sub(r"\W+", "", candidate).casefold()
            if key and key not in seen:
                seen.add(key)
                queries.append(candidate.strip())

        if len(queries) <= 1:
            query, hits, documents, chunks = self.retriever.search(
                question, category, history, diagnostics=diagnostics, intent=intent,
            )
        else:
            query_results = []
            documents, chunks = {}, {}
            normalized_queries = []
            for index, candidate in enumerate(queries):
                query, found, found_documents, found_chunks = self.retriever.search(
                    candidate, category, history if index == 0 else None,
                    diagnostics={}, rerank=False, finalize=False, intent=intent,
                )
                normalized_queries.append(query)
                query_results.append((query, found))
                documents.update(found_documents)
                chunks.update(found_chunks)

            fused = {}
            best_query = {}
            best_support = {}
            constant = self.settings.rrf_k
            for query_index, (candidate_query, candidate_hits) in enumerate(query_results):
                weight = 1.2 if query_index == 0 else 1.0
                for rank, hit in enumerate(candidate_hits, 1):
                    chunk_id = hit.chunk.chunk_id
                    contribution = weight / (constant + rank)
                    target = fused.get(chunk_id)
                    if target is None:
                        target = SearchHit(hit.chunk)
                        fused[chunk_id] = target
                    target.score += contribution
                    if hit.vector_similarity is not None:
                        target.vector_similarity = (
                            hit.vector_similarity if target.vector_similarity is None else
                            max(target.vector_similarity, hit.vector_similarity)
                        )
                    if hit.bm25_score is not None:
                        target.bm25_score = max(target.bm25_score or 0.0, hit.bm25_score)
                    target.channels.extend(hit.channels)
                    target.channels.append(f"query_{query_index + 1}")
                    if contribution > best_support.get(chunk_id, -1.0):
                        best_support[chunk_id] = contribution
                        best_query[chunk_id] = candidate_query

            candidates = sorted(fused.values(), key=lambda hit: hit.score, reverse=True)
            candidates = candidates[:self.settings.rerank_candidates]
            rerank_start = time.monotonic()
            applied, notice = False, None
            if self.settings.rerank_enabled and candidates:
                try:
                    query_for_candidate = [best_query[hit.chunk.chunk_id] for hit in candidates]
                    candidates = self.retriever.reranker.apply_queries(query_for_candidate, candidates)
                    applied = True
                except RerankError:
                    logger.warning("分解检索的候选重排序失败，回退查询融合顺序。", exc_info=True)
                    notice = "本地重排序失败，本次已使用混合检索排序。请查看启动日志。"
            diagnostics.update(
                rerank_applied=applied,
                rerank_model=self.settings.rerank_model if self.settings.rerank_enabled else None,
                rerank_backend=self.settings.rerank_backend if self.settings.rerank_enabled else None,
                rerank_candidates=len(candidates) if applied else 0,
                rerank_seconds=round(time.monotonic() - rerank_start, 3),
                decomposed_queries=len(queries) - 1,
            )
            if notice:
                diagnostics["rerank_notice"] = notice
            hits = self.retriever.select_final_hits(question, candidates, applied, diagnostics, intent=intent)
            query = normalized_queries[0] if normalized_queries else question

        knowledge_hits, knowledge_notice = [], None
        knowledge_start = time.monotonic()
        if self.settings.mysql_enabled:
            try:
                knowledge_hits = self.feedback_kb.search(
                    query, category, reranker=self.retriever.reranker,
                )
            except Exception:
                knowledge_notice = "MySQL 点赞问答知识库暂不可用，本次继续依据接口文档回答。"
        reserved = self.settings.mysql_kb_context_tokens if knowledge_hits else 0
        bundle = self.context_builder.build(question, hits, documents, chunks, reserved_tokens=reserved)
        if knowledge_hits:
            self.context_builder.add_feedback_knowledge(bundle, knowledge_hits, reserved)
        if knowledge_notice:
            bundle.notices.append(knowledge_notice)
        if diagnostics.get("rerank_low_confidence"):
            bundle.notices.append("最相关候选也未达到重排序相关性门槛，已停止引用这些接口片段；请补充接口名、参数名或业务模块。")
        if notice := diagnostics.pop("rerank_notice", None):
            bundle.notices.append(notice)
        return bundle, {
            **diagnostics,
            "mysql_qa_hits": len(knowledge_hits),
            "mysql_qa_seconds": round(time.monotonic() - knowledge_start, 3),
            "retrieval_seconds": round(time.monotonic() - start, 2),
            "retrieved_chunks": len(hits), "source_count": len(bundle.sources),
            "context_tokens_estimated": bundle.estimated_tokens,
            "rerank_enabled": self.settings.rerank_enabled,
            "query_expanded": query != question.strip(),
            "query_decomposition_applied": len(queries) > 1,
            "query_decomposition_count": max(0, len(queries) - 1),
        }

    @staticmethod
    def check_citations(answer, bundle):
        valid = {source["id"] for source in bundle.sources}
        cited = set()

        def validate(match):
            cited.add(match.group(1))
            return match.group() if match.group(1) in valid else ""

        answer = transform_prose_citations(answer, validate)
        notices = []
        if cited - valid:
            notices.append("已移除模型生成的无效来源编号，请结合原文核对。")
        if valid and not cited.intersection(valid):
            notices.append("模型未逐条标注引用，请结合下方文档核对回答。")
        return answer, notices

    def events(self, question, category=None, history=None, retrieve_only=False, memory_summary="", retrieval_hint="", allow_feedback=True, cancel_token=None):
        start = time.monotonic()
        if cancel_token:
            cancel_token.check()
        # 摘要刚整理完成时，保留上一问题作为检索指代线索；不重复发送整组旧问答。
        retrieval_history = list(history or [])
        if retrieval_hint and not any(item.get("role") == "user" and item.get("content") == retrieval_hint
                                      for item in retrieval_history):
            retrieval_history.insert(0, {"role": "user", "content": retrieval_hint})
        extra_queries, decomposition_notice = [], None
        if (classify(question, retrieval_history) != "version" and not retrieve_only and self.settings.api_key and self.settings.query_decomposition_enabled
                and should_decompose(question)):
            yield {"type": "status", "message": "正在拆解多步骤问题…"}
            extra = {"cancel_token": cancel_token} if cancel_token else {}
            extra_queries, decomposition_notice = self.decomposer.decompose(question, retrieval_history, **extra)
        bundle, stats = yield from RetrievalFlow(self).run(
            question, category, retrieval_history, extra_queries=extra_queries, cancel_token=cancel_token)
        if cancel_token:
            cancel_token.check()
        if decomposition_notice:
            bundle.notices.append(decomposition_notice)
        clarification = parameter_clarification(question, retrieval_history, bundle) if stats.get("needs_clarification") else None
        original_bundle = bundle
        retrieval_status = ("当前检索仍未确认全部所需信息。明确说明缺失的依据，不得补造未命中的接口或字段。"
                            if stats.get("retrieval_evidence_incomplete") else "")
        plan = None
        if not clarification and bundle.sources and not retrieve_only and self.settings.api_key:
            plan = prepare_prompt(self.settings, question, bundle, history, memory_summary,
                                  retrieval_status, retrieval_history)
            bundle = plan.bundle
            stats.update(plan.stats, source_count=len(bundle.sources),
                         context_tokens_estimated=bundle.estimated_tokens, context_length_retry_applied=False)
        yield {"type": "sources", "sources": bundle.sources, "notices": bundle.notices, "stats": dict(stats)}
        if clarification:
            answer, generated = clarification, False
        elif not bundle.sources:
            answer = ("没有当前文档源的可用版本记录，请先在侧栏检查或同步文档。"
                      if stats.get("retrieval_intent") == "version" else
                      "没有找到可用的文档证据。请补充业务模块、接口名称、参数名或错误码后重试。")
            generated = False
        elif retrieve_only or not self.settings.api_key:
            answer = (f"已找到 **{len(bundle.sources)} 处文档依据**，可展开下方来源查看。\n\n"
                      + ("当前为检索预览模式。" if retrieve_only else
                         "填写 `config.py` 中的 `DEEPSEEK_API_KEY`，保存并重启，即可生成回答。"))
            generated = False
        else:
            yield {"type": "status", "message": "正在生成回答…"}
            parts = []
            extra = {"cancel_token": cancel_token} if cancel_token else {}
            for attempt in range(2):
                if cancel_token:
                    cancel_token.check()
                stream = self.llm.stream(plan.messages, **extra)
                try:
                    for token in stream:
                        if cancel_token:
                            cancel_token.check()
                        parts.append(token)
                        yield {"type": "token", "text": token}
                    break
                except ContextLengthError:
                    # No replay after a partial answer, and never retry unrelated
                    # auth/validation/network failures as context errors.
                    if attempt or parts:
                        raise
                finally:
                    if hasattr(stream, "close"):
                        stream.close()
                if cancel_token:
                    cancel_token.check()
                yield {"type": "status", "message": "上下文超限，正在精简后重试一次…"}
                plan = prepare_prompt(self.settings, question, original_bundle, history, memory_summary,
                                      retrieval_status, retrieval_history,
                                      limit=int(message_tokens(plan.messages) * 0.7))
                bundle = plan.bundle
                bundle.notices.append("模型曾返回上下文超限，本次已进一步精简并重试一次。")
                stats.update(plan.stats, source_count=len(bundle.sources),
                             context_tokens_estimated=bundle.estimated_tokens, context_length_retry_applied=True)
                yield {"type": "sources", "sources": bundle.sources, "notices": bundle.notices, "stats": dict(stats)}
            answer = "".join(parts)
            generated = True
        if cancel_token:
            cancel_token.check()
        answer, notices = self.check_citations(answer, bundle) if self.settings.api_key and not retrieve_only else (answer, [])
        feedback_id = (self.feedback_kb.register_answer(question, answer, bundle.sources, category)
                       if generated and self.settings.mysql_enabled and allow_feedback and stats.get("retrieval_intent") != "version" else None)
        stats["total_seconds"] = round(time.monotonic() - start, 2)
        done = {"type": "done", "answer": answer, "sources": bundle.sources,
               "notices": bundle.notices + notices, "stats": stats,
               "mode": "clarification" if clarification and not retrieve_only else "answer" if generated else "retrieval"}
        if feedback_id:
            done["feedback_id"] = feedback_id
        yield done

    def save_liked_answer(self, feedback_id):
        if not self.settings.mysql_enabled:
            raise RuntimeError("MySQL 点赞问答功能已关闭。")
        return self.feedback_kb.save_liked_answer(feedback_id)

    def ask(self, question, category=None, history=None, retrieve_only=False):
        final = None
        for event in self.events(question, category, history, retrieve_only):
            if event["type"] == "done":
                final = event
        return final
