import json
import os
from pathlib import Path
import threading

from rag.types import Chunk, Document


def atomic_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(temporary, path)


class CorpusStore:
    _document_lock = threading.Lock()
    _document_cache = (None, {})

    def __init__(self, settings):
        self.settings = settings

    @property
    def active_path(self):
        return self.settings.data_dir / "active.json"

    def active(self):
        if not self.active_path.exists():
            return None
        return json.loads(self.active_path.read_text(encoding="utf-8"))

    def load(self, active=None):
        active = active or self.active()
        if not active:
            raise FileNotFoundError("文档索引尚未建立，请先运行 python manage.py index。")
        root = self.settings.data_dir / "generations" / active["generation"]
        documents = [Document.from_dict(row) for row in json.loads((root / "documents.json").read_text(encoding="utf-8"))]
        chunks = [Chunk(**row) for row in json.loads((root / "chunks.json").read_text(encoding="utf-8"))]
        return documents, chunks

    def get_document(self, document_id):
        """预览仅缓存当前代文档；多个 CorpusStore 实例共享进程内缓存。"""
        with CorpusStore._document_lock:
            active = self.active()
            if not active:
                CorpusStore._document_cache = (None, {})
                raise FileNotFoundError("文档索引尚未建立，请先运行 python manage.py index。")
            key = (self.settings.data_dir.resolve(), active["generation"])
            cached_key, documents = CorpusStore._document_cache
            if key != cached_key:
                root = key[0] / "generations" / key[1]
                rows = json.loads((root / "documents.json").read_text(encoding="utf-8"))
                documents = {row["document_id"]: Document.from_dict(row) for row in rows}
                # 完整解析后再发布，失败时不会留下半份缓存。
                CorpusStore._document_cache = (key, documents)
            return documents.get(document_id)


class VectorStore:
    def __init__(self, settings):
        # 禁止 Chroma 默认遥测；向量由本项目显式传入，不下载默认 embedding。
        os.environ.setdefault("ANONYMIZED_TELEMETRY", "false")
        import chromadb
        from chromadb.config import Settings as ChromaSettings
        persist = (settings.data_dir / "chroma").resolve()
        persist.mkdir(parents=True, exist_ok=True)
        client_path = str(persist)
        if os.name == "nt" and not client_path.isascii():
            # Windows HNSW 的窄字符串文件接口不能可靠处理中文绝对路径。
            # 在项目目录启动时传入 ASCII 相对路径，文件仍保存在原项目内。
            try:
                relative = os.path.relpath(persist, Path.cwd())
            except ValueError:
                relative = ""
            if not relative or not relative.isascii():
                raise ValueError("Windows 中文项目目录请先切换到项目目录再运行，或使用 start.bat。")
            client_path = relative
        self.client = chromadb.PersistentClient(
            path=client_path,
            settings=ChromaSettings(anonymized_telemetry=False),
        )

    def create(self, name):
        return self.client.create_collection(name=name, embedding_function=None,
                                              configuration={"hnsw": {"space": "cosine"}})

    def get(self, name):
        return self.client.get_collection(name=name, embedding_function=None)
