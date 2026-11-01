"""命令行入口：下载模型、更新索引、检索预览。"""
import argparse
import json
import logging
import os

from rag.embedding import LocalEmbedding
from rag.indexing.builder import IndexBuilder
from rag.settings import Settings
from rag.retrieval.reranker import OptionalReranker


def main():
    os.chdir(Settings.load().project_dir)
    parser = argparse.ArgumentParser(description="领星接口文档混合检索管理工具")
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("download-model", help="下载本地 BGE-small 模型")
    commands.add_parser("download-reranker", help="下载配置的本地重排序模型")
    index = commands.add_parser("index", help="更新文档索引，复用未改变的向量")
    index.add_argument("--force", action="store_true", help="强制重算全部向量")
    search = commands.add_parser("search", help="检索预览，不调用 DeepSeek")
    search.add_argument("question")
    commands.add_parser("check", help="显示配置与索引状态，不显示 API key")
    commands.add_parser("check-docs", help="检查文档源增删改，不下载或发布文档")
    commands.add_parser("sync-docs", help="同步文档、记录版本并增量更新索引")
    args = parser.parse_args()
    settings = Settings.load()
    settings.validate()
    if args.command in ("check-docs", "sync-docs"):
        from rag.document_sync import DocumentSync
        manager = DocumentSync(settings)
        manager.run(check_only=args.command == "check-docs")
        print(json.dumps(manager.status(), ensure_ascii=False, indent=2))
    elif args.command == "download-model":
        print(LocalEmbedding(settings).download())
    elif args.command == "download-reranker":
        print(OptionalReranker(settings).download())
    elif args.command == "index":
        IndexBuilder(settings, LocalEmbedding(settings)).build(args.force, progress=lambda text: print(text, flush=True))
    elif args.command == "search":
        from rag.service import RAGService
        result = RAGService(settings).ask(args.question, retrieve_only=True)
        print(json.dumps({"sources": result["sources"], "stats": result["stats"]}, ensure_ascii=False, indent=2))
    else:
        from rag.indexing.store import CorpusStore
        active = CorpusStore(settings).active()
        print(json.dumps({"llm_model": settings.llm_model, "api_key_configured": bool(settings.api_key),
                          "documents_dir": str(settings.docs_dir), "documents_dir_exists": settings.docs_dir.exists(),
                          "model_downloaded": (settings.model_dir / "model.safetensors").exists(),
                          "rerank_enabled": settings.rerank_enabled, "rerank_model": settings.rerank_model,
                          "rerank_backend": settings.rerank_backend,
                          "retrieval_retry_enabled": settings.retrieval_retry_enabled,
                          "rerank_model_downloaded": OptionalReranker(settings).downloaded(),
                          "indexed_documents": active["document_count"] if active else 0,
                          "indexed_chunks": active["chunk_count"] if active else 0}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    main()
