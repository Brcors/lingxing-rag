"""本地 BGE 模型：一次加载，查询与文档使用一致的归一化向量。"""
import hashlib
import json
import os
import threading
import re

from filelock import FileLock

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")


class LocalEmbedding:
    query_instruction = "为这个句子生成表示以用于检索相关文章："

    def __init__(self, settings):
        self.settings = settings
        self._model = None
        self._load_lock = threading.Lock()
        self._encode_lock = threading.RLock()

    def _cache_directory(self):
        root = self.settings.model_dir
        identity = root / "_model_id.txt"
        try:
            model_id = identity.read_text(encoding="utf-8").strip()
        except (OSError, UnicodeError):
            model_id = None
        if (root / "model.safetensors").exists() and model_id != self.settings.embedding_model:
            slug = re.sub(r"[^A-Za-z0-9_.-]+", "_", self.settings.embedding_model)[:80]
            suffix = hashlib.sha256(self.settings.embedding_model.encode()).hexdigest()[:10]
            return root.parent / f"embedding_{slug}_{suffix}"
        return root

    def download(self):
        from huggingface_hub import HfApi, snapshot_download
        target = self._cache_directory()
        required = ("model.safetensors", "modules.json", "config.json", "tokenizer_config.json",
                    "tokenizer.json", "1_Pooling/config.json", "_revision.txt")

        def ready():
            try:
                return (all((target / name).is_file() for name in required) and
                        (target / "_model_id.txt").read_text(encoding="utf-8").strip() == self.settings.embedding_model)
            except (OSError, UnicodeError):
                return False

        if ready():
            return target
        target.parent.mkdir(parents=True, exist_ok=True)
        with FileLock(str(target) + ".download.lock", timeout=300):
            if ready():
                return target
            target.mkdir(parents=True, exist_ok=True)
            info = HfApi().model_info(self.settings.embedding_model)
            snapshot_download(
                repo_id=self.settings.embedding_model, revision=info.sha, local_dir=str(target),
                allow_patterns=["*.json", "*.txt", "*.safetensors", "1_Pooling/*"],
            )
            (target / "_revision.txt").write_text(info.sha, encoding="utf-8")
            if not all((target / name).is_file() for name in required):
                raise RuntimeError("模型文件下载不完整，请重新下载模型。")
            (target / "_model_id.txt").write_text(self.settings.embedding_model, encoding="utf-8")
        return target

    @property
    def model(self):
        if self._model is None:
            with self._load_lock:
                if self._model is None:
                    from sentence_transformers import SentenceTransformer
                    import torch
                    torch.set_num_threads(self.settings.cpu_threads)
                    path = self.download()
                    self._model = SentenceTransformer(str(path), device=self.settings.device, local_files_only=True)
                    self._model.max_seq_length = 512
        return self._model

    @property
    def tokenizer(self):
        return self.model.tokenizer

    def count(self, text):
        with self._encode_lock:
            return len(self.tokenizer.encode(text, add_special_tokens=False, verbose=False))

    def token_offsets(self, text):
        with self._encode_lock:
            return self.tokenizer(text, add_special_tokens=False, return_offsets_mapping=True, verbose=False)

    def signature(self):
        target = self.download()
        revision = target / "_revision.txt"
        material = {
            "model": self.settings.embedding_model,
            "revision": revision.read_text(encoding="utf-8").strip() if revision.exists() else "local",
            "normalize": True, "instruction": self.query_instruction,
            "chunk_target": self.settings.chunk_target, "chunk_max": self.settings.chunk_max,
            "overlap": self.settings.overlap, "parser_version": 1,
        }
        return hashlib.sha256(json.dumps(material, sort_keys=True).encode()).hexdigest()[:16]

    def encode_documents(self, texts):
        with self._encode_lock:
            return self.model.encode(texts, batch_size=self.settings.batch_size,
                                     normalize_embeddings=True, show_progress_bar=False).tolist()

    def encode_query(self, query):
        # 为 query 指令预留位置；使用原始字符偏移截取，避免 decode 改变字段名。
        with self._encode_lock:
            tokens = self.token_offsets(query)
            if len(tokens["input_ids"]) > 450:
                query = query[:tokens["offset_mapping"][449][1]]
            return self.model.encode([self.query_instruction + query], normalize_embeddings=True,
                                     show_progress_bar=False).tolist()[0]
