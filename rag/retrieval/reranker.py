"""本地 CrossEncoder：独立缓存、离线加载及串行推理。"""
import hashlib
import logging
import math
import re
import threading

from filelock import FileLock

logger = logging.getLogger(__name__)


class RerankError(RuntimeError):
    pass


class OptionalReranker:
    def __init__(self, settings):
        self.settings = settings
        self._model = None
        self._lock = threading.RLock()

    @property
    def model_dir(self):
        model_id = self.settings.rerank_model
        slug = re.sub(r"[^A-Za-z0-9_.-]+", "_", model_id)[:80]
        suffix = hashlib.sha256(model_id.encode()).hexdigest()[:10]
        return self.settings.project_dir / "models" / "reranker" / f"{slug}_{suffix}"

    def downloaded(self):
        required = ["config.json", "tokenizer.json", "tokenizer_config.json", "_revision.txt"]
        if self.settings.rerank_backend == "onnx-int8":
            required += ["onnx/model_int8.onnx"]
        else:
            required += ["model.safetensors"]
        try:
            return (all((self.model_dir / name).is_file() for name in required) and
                    bool((self.model_dir / "_revision.txt").read_text(encoding="utf-8").strip()) and
                    (self.model_dir / "_model_id.txt").read_text(encoding="utf-8").strip() == self.settings.rerank_model)
        except (OSError, UnicodeError):
            return False

    def download(self):
        """仅下载推理所需文件，固定 revision；完整缓存不再请求网络。"""
        if self.downloaded():
            return self.model_dir
        from huggingface_hub import HfApi, snapshot_download

        self.model_dir.parent.mkdir(parents=True, exist_ok=True)
        with FileLock(str(self.model_dir) + ".download.lock", timeout=300):
            if self.downloaded():
                return self.model_dir
            revision_file = self.model_dir / "_revision.txt"
            identity = self.model_dir / "_model_id.txt"
            revision = None
            if revision_file.exists() and identity.exists() and identity.read_text(encoding="utf-8").strip() == self.settings.rerank_model:
                revision = revision_file.read_text(encoding="utf-8").strip()
            if not revision:
                revision = HfApi().model_info(self.settings.rerank_model).sha
            logger.info("正在下载本地重排序模型：%s", self.settings.rerank_model)
            weight = "onnx/model_int8.onnx" if self.settings.rerank_backend == "onnx-int8" else "model.safetensors"
            snapshot_download(
                repo_id=self.settings.rerank_model, revision=revision, local_dir=str(self.model_dir),
                allow_patterns=["config.json", weight, "tokenizer.json", "tokenizer_config.json",
                                "special_tokens_map.json", "sentencepiece.bpe.model", "LICENSE*", "README.md"],
            )
            (self.model_dir / "_revision.txt").write_text(revision, encoding="utf-8")
            (self.model_dir / "_model_id.txt").write_text(self.settings.rerank_model, encoding="utf-8")
            if not self.downloaded():
                raise RerankError("重排序模型下载不完整，请运行 python manage.py download-reranker 重试。")
        return self.model_dir

    def _load(self):
        if self._model is None:
            if self.settings.rerank_backend == "onnx-int8":
                from rag.retrieval.onnx_reranker import ONNXCrossEncoder
                self._model = ONNXCrossEncoder(self.download(), self.settings.rerank_max_length,
                                                self.settings.cpu_threads)
                return self._model
            from sentence_transformers import CrossEncoder
            import torch

            torch.set_num_threads(self.settings.cpu_threads)
            self._model = CrossEncoder(
                str(self.download()), device=self.settings.device, local_files_only=True,
                trust_remote_code=False, max_length=self.settings.rerank_max_length,
                activation_fn=torch.nn.Identity(),
            )
        return self._model

    def warmup(self):
        if self.settings.rerank_enabled:
            with self._lock:
                self._load().predict([("领星接口", "查询接口文档")], batch_size=1, show_progress_bar=False)

    def apply(self, query, hits):
        return self.apply_queries([query] * len(hits), hits)

    def apply_queries(self, queries, hits):
        if not self.settings.rerank_enabled or not hits:
            return hits
        if len(queries) != len(hits):
            raise RerankError("重排序问题数量与候选片段数量不一致。")
        try:
            with self._lock:
                scores = self._load().predict(
                    [(query, hit.chunk.embedding_text) for query, hit in zip(queries, hits)],
                    batch_size=self.settings.rerank_batch_size, show_progress_bar=False,
                )
                # 原始相关性 logits 仅用于同一次查询内排序，不作为概率或与 RRF 分数相加。
                scores = [float(score) for score in scores]
                if len(scores) != len(hits) or not all(math.isfinite(score) for score in scores):
                    raise ValueError("invalid rerank scores")
                for hit, score in zip(hits, scores):
                    hit.rerank_score = score
        except Exception as error:
            raise RerankError("本地重排序失败，本次已使用混合检索排序。请查看启动日志。") from error
        return sorted(hits, key=lambda hit: hit.rerank_score, reverse=True)
