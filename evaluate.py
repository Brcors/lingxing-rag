"""使用本地真实向量模型评估一组固定检索案例，不调用 DeepSeek。"""
import json
import os
from pathlib import Path

from rag.indexing.store import atomic_json
from rag.service import RAGService
from rag.settings import Settings


def main():
    settings = Settings.load()
    os.chdir(settings.project_dir)
    service = RAGService(settings)
    cases = json.loads((Path(__file__).parent / "tests" / "retrieval_cases.json").read_text(encoding="utf-8"))
    results = []
    for case in cases:
        bundle, stats = service.retrieve(case["question"], case.get("category"))
        titles = list(dict.fromkeys(source["title"] for source in bundle.sources))
        success = any(case["expected"] in title for title in titles)
        row = {**case, "matched": success, "source_titles": titles, "stats": stats}
        results.append(row)
        print(("PASS " if success else "MISS ") + case["question"] + " -> " + " / ".join(titles), flush=True)
    count = sum(row["matched"] for row in results)
    reranked = sum(row["stats"].get("rerank_applied", False) for row in results)
    report = {"matched": count, "total": len(results), "rerank_applied": reranked, "cases": results}
    atomic_json(settings.data_dir / "retrieval-evaluation.json", report)
    print(f"固定案例召回命中：{count}/{len(results)}。这不是通用准确率。", flush=True)
    if settings.rerank_enabled:
        print(f"实际完成本地重排序：{reranked}/{len(results)}。", flush=True)
    return 0 if count == len(results) and (not settings.rerank_enabled or reranked == len(results)) else 1


if __name__ == "__main__":
    raise SystemExit(main())
