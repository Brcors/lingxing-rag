from dataclasses import dataclass, field
from pathlib import Path
import os

import config


@dataclass(frozen=True)
class Settings:
    project_dir: Path = config.PROJECT_DIR
    docs_dir: Path = config.DOCS_DIR
    doc_sync_enabled: bool = config.DOC_SYNC_ENABLED
    doc_sync_repository: str = config.DOC_SYNC_REPOSITORY
    doc_sync_branch: str = config.DOC_SYNC_BRANCH
    doc_sync_subdirectory: str = config.DOC_SYNC_SUBDIRECTORY
    doc_sync_base_revision: str = config.DOC_SYNC_BASE_REVISION
    doc_sync_interval: int = config.DOC_SYNC_INTERVAL_SECONDS
    supplement_dir: Path = config.SUPPLEMENT_DIR
    data_dir: Path = config.DATA_DIR
    model_dir: Path = config.MODEL_DIR
    embedding_model: str = config.EMBEDDING_MODEL
    device: str = config.EMBEDDING_DEVICE
    batch_size: int = config.EMBEDDING_BATCH_SIZE
    cpu_threads: int = config.CPU_THREADS
    chunk_target: int = config.CHUNK_TARGET_TOKENS
    chunk_max: int = config.CHUNK_MAX_TOKENS
    overlap: int = config.CHUNK_OVERLAP_TOKENS
    vector_top_k: int = config.VECTOR_TOP_K
    bm25_top_k: int = config.BM25_TOP_K
    rrf_k: int = config.RRF_K
    rerank_candidates: int = config.RERANK_CANDIDATES
    final_top_k: int = config.FINAL_TOP_K
    context_budget: int = config.CONTEXT_TOKEN_BUDGET
    rerank_enabled: bool = config.RERANK_ENABLED
    rerank_model: str = config.RERANK_MODEL
    rerank_backend: str = config.RERANK_BACKEND
    rerank_batch_size: int = config.RERANK_BATCH_SIZE
    rerank_max_length: int = config.RERANK_MAX_LENGTH
    rerank_min_score: float = config.RERANK_MIN_SCORE
    query_decomposition_enabled: bool = config.QUERY_DECOMPOSITION_ENABLED
    query_decomposition_max_subquestions: int = config.QUERY_DECOMPOSITION_MAX_SUBQUESTIONS
    query_decomposition_max_tokens: int = config.QUERY_DECOMPOSITION_MAX_TOKENS
    retrieval_retry_enabled: bool = config.RETRIEVAL_RETRY_ENABLED
    mysql_enabled: bool = config.MYSQL_ENABLED
    mysql_host: str = config.MYSQL_HOST
    mysql_port: int = config.MYSQL_PORT
    mysql_user: str = config.MYSQL_USER
    mysql_password: str = field(default=config.MYSQL_PASSWORD, repr=False)
    mysql_database: str = config.MYSQL_DATABASE
    mysql_charset: str = config.MYSQL_CHARSET
    mysql_connect_timeout: int = config.MYSQL_CONNECT_TIMEOUT
    mysql_kb_max_records: int = config.MYSQL_KB_MAX_RECORDS
    mysql_kb_top_k: int = config.MYSQL_KB_TOP_K
    mysql_kb_min_similarity: float = config.MYSQL_KB_MIN_SIMILARITY
    mysql_kb_context_tokens: int = config.MYSQL_KB_CONTEXT_TOKENS
    api_key: str = field(default="", repr=False)
    base_url: str = config.DEEPSEEK_BASE_URL
    llm_model: str = config.DEEPSEEK_MODEL
    temperature: float = config.DEEPSEEK_TEMPERATURE
    max_output_tokens: int = config.DEEPSEEK_MAX_TOKENS
    timeout: int = config.DEEPSEEK_TIMEOUT

    @classmethod
    def load(cls):
        environment_key = os.getenv("DEEPSEEK_API_KEY", "").strip()
        return cls(api_key=environment_key or config.DEEPSEEK_API_KEY.strip())

    def validate(self):
        if self.doc_sync_enabled:
            import re
            from pathlib import PurePosixPath
            sub = PurePosixPath(self.doc_sync_subdirectory)
            if (not re.fullmatch(r"[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+", self.doc_sync_repository)
                    or not self.doc_sync_branch or not self.doc_sync_subdirectory
                    or sub.is_absolute() or ".." in sub.parts or "\\" in self.doc_sync_subdirectory
                    or self.doc_sync_interval < 60):
                raise ValueError("文档同步来源或检查间隔配置不合法（最小 60 秒）。")
            if self.doc_sync_base_revision and not re.fullmatch(r"[a-f0-9]{40}", self.doc_sync_base_revision):
                raise ValueError("文档同步初始版本必须为空或完整的 Git 提交 SHA。")
        if not (64 <= self.chunk_target <= self.chunk_max <= 480):
            raise ValueError("切块参数需要满足 64 ≤ target ≤ max ≤ 480。")
        if not 0 <= self.overlap < self.chunk_target // 2:
            raise ValueError("重叠长度应小于目标块长的一半。")
        if self.context_budget < 512 or self.final_top_k < 1:
            raise ValueError("上下文预算或召回数量配置不合法。")
        if self.rerank_candidates < self.final_top_k or self.rerank_batch_size < 1:
            raise ValueError("重排候选数量不能少于最终数量，批大小必须大于 0。")
        if not 64 <= self.rerank_max_length <= 512:
            raise ValueError("重排输入长度应在 64 到 512 token 之间。")
        if not -20.0 <= self.rerank_min_score <= 20.0:
            raise ValueError("重排最低分数配置应在 -20 到 20 之间。")
        if not 1 <= self.query_decomposition_max_subquestions <= 3:
            raise ValueError("问题拆解最多允许 1 到 3 个子问题。")
        if not 64 <= self.query_decomposition_max_tokens <= 1024:
            raise ValueError("问题拆解生成长度应在 64 到 1024 token 之间。")
        if not isinstance(self.retrieval_retry_enabled, bool):
            raise ValueError("RETRIEVAL_RETRY_ENABLED 必须为 True 或 False。")
        if self.rerank_backend not in ("onnx-int8", "torch"):
            raise ValueError("重排序后端应为 onnx-int8 或 torch。")
        if self.mysql_enabled:
            import re
            if (not re.fullmatch(r"[A-Za-z][A-Za-z0-9_]{0,63}", self.mysql_database) or
                    self.mysql_port not in range(1, 65536) or self.mysql_charset.lower() != "utf8mb4"):
                raise ValueError("MySQL 数据库名、端口或字符集配置不合法。")
            if self.mysql_kb_max_records < 1 or not 1 <= self.mysql_kb_top_k <= 5:
                raise ValueError("MySQL 知识库记录上限或返回数量不合法。")
