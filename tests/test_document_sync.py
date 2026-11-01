from dataclasses import replace
import json
from pathlib import Path

import pytest

from conftest import write_api
from rag.document_sync import DocumentSync, DocumentSyncError, GitHubDocumentSource, blob_sha
from rag.ingestion.parser import MarkdownLoader
from rag.indexing.builder import IndexBuilder
from rag.indexing.store import CorpusStore, atomic_json


class Source:
    def __init__(self, files):
        self.files = files
        self.revision = "a" * 40
        self.date = "2026-11-01T01:00:00Z"

    def snapshot(self):
        return {"revision": self.revision, "source_time": self.date,
                "files": {path: blob_sha(content) for path, content in self.files.items()}}

    def download(self, snapshot, paths):
        return {path: self.files[path] for path in paths}


def manager(settings, files):
    settings = replace(settings, doc_sync_enabled=True, doc_sync_base_revision="")
    source = Source(files)
    calls = []
    sync = DocumentSync(settings, indexer=lambda progress: calls.append(True), source=source)
    return sync, source, calls


def test_sync_checks_versions_preserves_custom_docs_and_indexes(settings):
    (settings.docs_dir / "a.md").write_text("# 原文\n旧内容", encoding="utf-8")
    (settings.docs_dir / "notes.md").write_text("# 本地笔记", encoding="utf-8")
    sync, source, indexed = manager(settings, {"a.md": b"# Original\nold"})
    sync.run()
    assert len(indexed) == 1
    baseline = sync.history()[0]
    source.files = {"a.md": b"# Original\nnew", "FBA/b.md": b"# New\ncontent"}
    source.revision = "b" * 40
    source.date = "2026-11-02T01:00:00Z"
    checked = sync.run(check_only=True)
    assert checked["available_changes_counts"] == {"added": 1, "modified": 1, "deleted": 0}
    assert (settings.docs_dir / "a.md").read_bytes().endswith(b"old")
    assert len(indexed) == 1
    synced = sync.run()
    assert not synced["running"] and not synced["index_pending"]
    assert synced["available_changes_counts"] == {"added": 0, "modified": 0, "deleted": 0}
    assert len(sync.history()) == 2 and len(indexed) == 2
    assert (settings.docs_dir / "notes.md").read_text(encoding="utf-8") == "# 本地笔记"
    assert (sync.root / "versions" / synced_version(sync) / "before" / "a.md").read_bytes().endswith(b"old")
    doc = next(doc for doc in MarkdownLoader(sync.settings).load() if doc.relative_path == "a.md")
    assert doc.updated_at == source.date and doc.source_revision == source.revision
    assert doc.update_basis == "source_commit"
    sync.run()
    assert len(sync.history()) == 2 and len(indexed) == 2
    source.files.pop("FBA/b.md")
    source.revision = "c" * 40
    sync.run()
    assert not (settings.docs_dir / "FBA/b.md").exists()
    assert sync.history()[0]["deleted"] == ["FBA/b.md"]


def synced_version(sync):
    return sync.history()[0]["id"]


def test_local_edit_conflict_preserves_docs_and_last_index(settings):
    sync, source, indexed = manager(settings, {"a.md": b"# A\ninitial"})
    sync.run()
    (settings.docs_dir / "a.md").write_bytes(b"# A\nlocal edit")
    source.files["a.md"] = b"# A\nupstream change"
    with pytest.raises(DocumentSyncError, match="本地修改"):
        sync.run()
    assert (settings.docs_dir / "a.md").read_bytes().endswith(b"local edit")
    assert len(indexed) == 1 and len(sync.history()) == 1


def test_failed_index_retries_without_losing_previous_active(settings):
    sync, source, _ = manager(settings, {"a.md": b"# A\nfirst"})
    sync.run()
    atomic_json(settings.data_dir / "active.json", {"generation": "old-good-index"})
    source.files["a.md"] = b"# A\nchanged"
    source.revision = "b" * 40
    def fail(progress):
        raise RuntimeError("index failure")
    sync.indexer = fail
    result = sync.run()
    assert result["index_pending"]
    assert json.loads((settings.data_dir / "active.json").read_text())["generation"] == "old-good-index"
    attempts = []
    sync.indexer = lambda progress: attempts.append(True)
    result = sync.run()
    assert attempts == [True] and not result["index_pending"] and not result["last_error"]
    assert len(sync.history()) == 2


def test_invalid_archive_and_tree_paths_are_rejected(settings, monkeypatch):
    source = GitHubDocumentSource(settings)
    monkeypatch.setattr(source, "_json", lambda suffix: {"tree": [
        {"path": settings.doc_sync_subdirectory + "/../escape.md", "type": "blob", "mode": "100644", "sha": "a" * 40}]})
    with pytest.raises(DocumentSyncError, match="路径"):
        source.tree("a" * 40)
    import io
    import zipfile
    archive = io.BytesIO()
    with zipfile.ZipFile(archive, "w") as package:
        package.writestr("repo/" + settings.doc_sync_subdirectory + "/a.md", b"tampered")
    monkeypatch.setattr(source, "_read", lambda url, limit: archive.getvalue())
    with pytest.raises(DocumentSyncError, match="校验"):
        source.download({"revision": "a" * 40, "files": {"a.md": blob_sha(b"expected")}}, ["a.md"])


def test_metadata_only_index_update_reuses_vectors(settings, embedding):
    write_api(settings.docs_dir / "a.md")
    builder = IndexBuilder(settings, embedding)
    initial = builder.build()
    computed = embedding.computed
    atomic_json(settings.data_dir / "document-sync" / "state.json", {
        "source": {"docs_dir": str(settings.docs_dir.resolve())},
        "files": {"a.md": {"source_revision": "a" * 40, "updated_at": "2026-11-01T00:00:00Z",
                            "synced_at": "2026-11-01T01:00:00Z", "date_basis": "source_commit"}}})
    updated = builder.build()
    assert updated["generation"] != initial["generation"]
    assert embedding.computed == computed and updated["reused_vectors"] == updated["chunk_count"]
    documents, _ = CorpusStore(settings).load()
    assert documents[0].source_revision == "a" * 40


def test_windows_checkout_newlines_are_not_local_conflicts(settings):
    content = b"# A\ncontent\n"
    (settings.docs_dir / "a.md").write_bytes(content.replace(b"\n", b"\r\n"))
    sync, source, calls = manager(settings, {"a.md": content})
    sync.settings = replace(sync.settings, doc_sync_base_revision=source.revision)
    sync.run()
    assert calls == [True]
    document = MarkdownLoader(sync.settings).load()[0]
    assert document.source_revision == source.revision
    (settings.docs_dir / "a.md").write_bytes(b"# A\nmanual edit\n")
    assert MarkdownLoader(sync.settings).load()[0].source_revision == ""


def test_publish_failure_restores_documents_and_metadata(settings, monkeypatch):
    sync, source, calls = manager(settings, {"a.md": b"# A\nold"})
    sync.run()
    source.files["a.md"] = b"# A\nnew"
    source.revision = "b" * 40
    import rag.document_sync as module
    original = module.atomic_json
    def fail_new_state(path, data):
        if Path(path).name == "state.json" and data.get("revision") == "b" * 40:
            raise OSError("simulated write failure")
        original(path, data)
    monkeypatch.setattr(module, "atomic_json", fail_new_state)
    with pytest.raises(DocumentSyncError):
        sync.run()
    assert (settings.docs_dir / "a.md").read_bytes() == b"# A\nold"
    assert sync.status()["revision"] == "a" * 40
    assert len(sync.history()) == 1 and len(calls) == 1
