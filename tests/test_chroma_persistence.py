"""Windows 中文项目目录超过 HNSW 默认刷盘阈值后的跨进程查询回归。"""
import os
from pathlib import Path
import subprocess
import sys

import pytest


@pytest.mark.skipif(os.name != "nt", reason="Windows 中文路径回归")
def test_chroma_unicode_project_reopens_after_flush(tmp_path):
    project = tmp_path / "中文项目"
    project.mkdir()
    env = dict(os.environ)
    env["PYTHONPATH"] = str(Path(__file__).resolve().parents[1])
    common = """
from dataclasses import replace
from pathlib import Path
from rag.indexing.store import VectorStore
from rag.settings import Settings
settings = replace(Settings(), data_dir=Path.cwd() / 'data')
store = VectorStore(settings)
"""
    create = common + """
collection = store.create('persistence_regression')
collection.add(ids=[str(i) for i in range(1200)], embeddings=[[float(i % 7), 1.0, 0.5] for i in range(1200)])
assert collection.query(query_embeddings=[[1.0, 1.0, 0.5]], n_results=1)['ids'][0]
"""
    reopen = common + """
collection = store.get('persistence_regression')
assert collection.count() == 1200
assert collection.query(query_embeddings=[[1.0, 1.0, 0.5]], n_results=1)['ids'][0]
"""
    for script in (create, reopen):
        result = subprocess.run([sys.executable, "-X", "utf8", "-c", script], cwd=project,
                                env=env, capture_output=True, text=True, encoding="utf-8", timeout=60)
        assert result.returncode == 0, result.stderr
