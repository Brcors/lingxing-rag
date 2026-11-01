"""Reuse existing retrieval and add at most one rule-based follow-up search."""
import logging
import time

from rag.retrieval.query import search_query
from rag.retrieval.routing import LABELS, PATHS, classify, merge_bundles, observe, parameter_clarification, version_subject
from rag.retrieval.versions import version_evidence

logger = logging.getLogger(__name__)


def followup_query(question, history, intent):
    suffix = {"parameters": "请求参数 必填 参数类型", "response": "返回结果 响应字段",
              "endpoint": "接口信息 API Path 请求方式", "workflow": "业务流程 前置条件 相关接口",
              "general": "接口说明 使用限制"}[intent]
    return search_query(question, history)[:600] + "\n" + suffix


class RetrievalFlow:
    def __init__(self, service):
        self.service, self.settings = service, service.settings

    def run(self, question, category=None, history=None, extra_queries=None, cancel_token=None):
        start = time.monotonic()
        intent = classify(question, history)
        stats = {"retrieval_intent": intent, "retrieval_rounds": 1,
                 "retrieval_retry_applied": False, "retrieval_steps": []}
        if cancel_token:
            cancel_token.check()
        yield {"type": "status", "message": f"正在检索{LABELS[intent]}…"}
        if cancel_token:
            cancel_token.check()
        if intent == "version":
            subject = version_subject(question, history)
            paths = []
            if subject:
                located, _ = self.service.retrieve(question, category, history, intent="endpoint")
                expanded = search_query(question, history)
                explicit_paths = PATHS.findall(expanded)
                paths = [s["relative_path"] for s in located.sources if s.get("relative_path") and
                    (any(path.casefold() == s.get("api_path", "").casefold() for path in explicit_paths)
                     if explicit_paths else (subject.casefold() in
                     (s.get("title", "") + s.get("api_path", "") + s["relative_path"]).casefold() or
                     (len(s.get("title", "")) >= 4 and s["title"].casefold() in expanded.casefold())))]
                stats["retrieval_steps"].append({"action": "document_search", "source_count": len(paths)})
                if cancel_token:
                    cancel_token.check()
            bundle = version_evidence(self.settings, subject, paths)
            stats["retrieval_steps"].append({"action": "local_versions", "source_count": len(bundle.sources)})
        else:
            # Keep the original rewriting/decomposition and its query fusion in the first round.
            bundle, initial = self.service.retrieve(question, category, history,
                                                    extra_queries=extra_queries, intent=intent)
            stats.update(initial)
            if cancel_token:
                cancel_token.check()
            gap = observe(search_query(question, history), intent, bundle, initial)
            needs_clarification = bool(parameter_clarification(question, history, bundle))
            stats["needs_clarification"] = needs_clarification
            stats["retrieval_steps"].append({"action": "document_search", "query": question,
                "source_count": len(bundle.sources), "observation": gap or "已有匹配证据"})
            if gap and self.settings.retrieval_retry_enabled and not needs_clarification:
                yield {"type": "status", "message": f"证据不足，正在补查{LABELS[intent]}（最多一次）…"}
                if cancel_token:
                    cancel_token.check()
                query = followup_query(question, history, intent)
                stats.update(retrieval_rounds=2, retrieval_retry_applied=True)
                try:
                    second, second_stats = self.service.retrieve(query, category, None, intent=intent)
                    if cancel_token:
                        cancel_token.check()
                    bundle = merge_bundles(bundle, second, self.settings.context_budget)
                    gap = observe(search_query(question, history), intent, bundle,
                                  {"rerank_low_confidence": not bundle.sources})
                    for key in ("rerank_seconds", "mysql_qa_seconds", "retrieved_chunks"):
                        stats[key] = round(initial.get(key, 0) + second_stats.get(key, 0), 3)
                    stats["rerank_applied"] = initial.get("rerank_applied", False) or second_stats.get("rerank_applied", False)
                    stats["rerank_low_confidence"] = bool(not bundle.sources)
                    stats.pop("rerank_best_score", None)
                    stats["retrieval_steps"].append({"action": "document_search", "query": query,
                        "source_count": len(second.sources), "observation": gap or "已有匹配证据"})
                except Exception:
                    if cancel_token:
                        cancel_token.check()
                    logger.warning("补查失败，保留首次检索证据。", exc_info=True)
                    bundle.notices.append("补查失败，已保留首次检索结果。")
                    stats["retrieval_steps"].append({"action": "document_search", "query": query,
                                                    "source_count": 0, "observation": "补查失败"})
            stats["retrieval_evidence_incomplete"] = bool(gap)
            if gap:
                bundle.notices.append("当前证据仍可能不足：" + gap + "。已停止继续检索，请结合来源核对或补充问题。")
        stats.update(source_count=len(bundle.sources), context_tokens_estimated=bundle.estimated_tokens,
                     retrieval_seconds=round(time.monotonic() - start, 2))
        return bundle, stats
