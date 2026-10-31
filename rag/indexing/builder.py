from datetime import datetime, timezone
import hashlib
import json
import logging
import uuid

import numpy as np

from filelock import FileLock

from rag.indexing.store import CorpusStore, VectorStore, atomic_json
from rag.ingestion.chunker import SemanticChunker
from rag.ingestion.parser import MarkdownLoader

logger = logging.getLogger(__name__)


class IndexBuilder:
    def __init__(self, settings, embedding):
        self.settings = settings
        self.embedding = embedding
        self.store = CorpusStore(settings)

    def build(self, force=False, progress=None):
        progress = progress or (lambda message: logger.info(message))
        self.settings.validate()
        self.settings.data_dir.mkdir(parents=True, exist_ok=True)
        with FileLock(str(self.settings.data_dir / "index.lock"), timeout=1):
            progress("读取文档与本地模型……")
            documents = MarkdownLoader(self.settings).load()
            if not documents:
                raise ValueError("文档目录中没有可索引的 Markdown 内容。")
            signature = self.embedding.signature()
            hashes = {document.relative_path: document.content_hash for document in documents}
            sources = {d.document_id: d.source_url for d in documents}
            fingerprint = hashlib.sha256(json.dumps([signature, hashes, sources, "storage-v3-fences"], sort_keys=True).encode()).hexdigest()
            old = self.store.active()
            vectors = VectorStore(self.settings)
            if not force and old and old.get("fingerprint") == fingerprint:
                # 检查持久化集合确实存在，不能仅凭 manifest 宣称成功。
                try:
                    collection = vectors.get(old["collection"])
                    count_matches = collection.count() == old["chunk_count"]
                    if count_matches:
                        # 必须真正查询 HNSW；count 成功不代表跨进程索引能读取。
                        collection.query(query_embeddings=[self.embedding.encode_query("领星接口")],
                                         n_results=1, include=["distances"])
                        progress("文档和索引配置未变化，复用已有索引。")
                        return {**old, "changed": False}
                except Exception:
                    progress("活动向量集合无法查询，将使用独立向量缓存重建。")
            progress(f"解析 {len(documents)} 篇文档，按章节、段落和表格行切分……")
            chunks = SemanticChunker(self.settings, self.embedding).build(documents)
            if not chunks:
                raise ValueError("解析后没有可检索的片段。")
            generation = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S") + "_" + uuid.uuid4().hex[:8]
            collection_name = "lingxing_" + generation.lower()
            collection = vectors.create(collection_name)
            previous_vectors, previous_offsets = None, {}
            if not force and old and old.get("embedding_signature") == signature:
                previous_root = self.settings.data_dir / "generations" / old["generation"]
                cache_path = previous_root / "vectors.npy"
                if cache_path.exists():
                    try:
                        _, previous_chunks = self.store.load(old)
                        previous_vectors = np.load(cache_path, mmap_mode="r", allow_pickle=False)
                        valid_cache = (previous_vectors.ndim == 2 and
                                       previous_vectors.shape[0] == len(previous_chunks) and
                                       previous_vectors.shape[1] > 0 and
                                       previous_vectors.dtype.kind == "f" and
                                       np.isfinite(previous_vectors).all())
                        if not valid_cache:
                            raise ValueError("向量缓存的形状或数值不合法")
                        previous_offsets = {c.chunk_id: i for i, c in enumerate(previous_chunks)}
                    except (OSError, ValueError, TypeError, IndexError):
                        previous_vectors, previous_offsets = None, {}
                        progress("独立向量缓存损坏或不完整，本次重新计算向量。")
                else:
                    progress("旧索引没有独立向量缓存，本次重新计算，随后可安全复用。")
            reused = 0
            vector_batches = []
            for start in range(0, len(chunks), 96):
                batch = chunks[start:start + 96]
                existing = {}
                if previous_vectors is not None:
                    existing = {c.chunk_id: previous_vectors[previous_offsets[c.chunk_id]].tolist()
                                for c in batch if c.chunk_id in previous_offsets}
                missing = [c for c in batch if c.chunk_id not in existing]
                encoded = self.embedding.encode_documents([c.embedding_text for c in missing]) if missing else []
                for chunk, value in zip(missing, encoded):
                    existing[chunk.chunk_id] = value
                reused += len(batch) - len(missing)
                batch_vectors = [existing[c.chunk_id] for c in batch]
                vector_batches.append(np.asarray(batch_vectors, dtype=np.float32))
                collection.add(
                    ids=[c.chunk_id for c in batch], embeddings=batch_vectors,
                    documents=[c.embedding_text for c in batch],
                    metadatas=[{"document_id": c.document_id, "parent_id": c.parent_id,
                                "category": c.category, "api_path": c.api_path,
                                "kind": c.kind, "heading": c.heading} for c in batch],
                )
                progress(f"向量索引 {min(start + len(batch), len(chunks))}/{len(chunks)}；复用 {reused} 个向量")
            if collection.count() != len(chunks):
                raise RuntimeError("索引数量校验失败，继续保留上一代索引。")
            collection.query(query_embeddings=[vector_batches[0][0].tolist()], n_results=1, include=["distances"])
            root = self.settings.data_dir / "generations" / generation
            root.mkdir(parents=True, exist_ok=True)
            np.save(root / "vectors.npy", np.concatenate(vector_batches), allow_pickle=False)
            atomic_json(root / "documents.json", [d.to_dict() for d in documents])
            atomic_json(root / "chunks.json", [c.to_dict() for c in chunks])
            metadata = {
                "generation": generation, "collection": collection_name,
                "fingerprint": fingerprint, "embedding_signature": signature,
                "embedding_model": self.settings.embedding_model,
                "document_count": len(documents), "chunk_count": len(chunks),
                "reused_vectors": reused, "categories": sorted({d.category for d in documents}),
                "built_at": datetime.now(timezone.utc).isoformat(),
                "document_hashes": hashes,
            }
            atomic_json(root / "manifest.json", metadata)
            # 构建失败时旧指针保持有效；删除文档也通过新一代语料自然移除。
            atomic_json(self.store.active_path, metadata)
            progress(f"索引就绪：{len(documents)} 篇文档，{len(chunks)} 个片段。")
            return {**metadata, "changed": True}
