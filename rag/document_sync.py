"""固定 Git 提交下载、校验、版本记录与定时文档同步。"""
import hashlib
import io
import json
import logging
import os
from pathlib import Path, PurePosixPath
import re
import shutil
import threading
from datetime import datetime, timezone
from urllib.error import HTTPError, URLError
from urllib.parse import quote
from urllib.request import Request, urlopen
import zipfile

from filelock import FileLock, Timeout

from rag.indexing.store import atomic_json

logger = logging.getLogger(__name__)


class DocumentSyncError(RuntimeError):
    pass


class DocumentSyncBusy(DocumentSyncError):
    pass


def now():
    return datetime.now(timezone.utc).isoformat()


def blob_sha(content):
    return hashlib.sha1(b"blob " + str(len(content)).encode("ascii") + b"\0" + content).hexdigest()


class GitHubDocumentSource:
    def __init__(self, settings):
        self.settings = settings
        self.prefix = settings.doc_sync_subdirectory.strip("/") + "/"

    def _read(self, url, limit):
        request = Request(url, headers={"User-Agent": "Lingxing-RAG-document-sync", "Accept": "application/vnd.github+json"})
        try:
            with urlopen(request, timeout=25) as response:
                content = response.read(limit + 1)
        except HTTPError as error:
            raise DocumentSyncError(f"文档源请求失败（HTTP {error.code}），请检查网络、来源配置或 GitHub 请求限额。") from None
        except (URLError, TimeoutError, OSError):
            raise DocumentSyncError("文档源连接失败，请稍后重试。") from None
        if len(content) > limit:
            raise DocumentSyncError("文档源超过下载大小限制。")
        return content

    def _json(self, suffix):
        try:
            return json.loads(self._read("https://api.github.com/repos/" + self.settings.doc_sync_repository + suffix, 12 * 1024 * 1024))
        except (ValueError, UnicodeError):
            raise DocumentSyncError("文档源返回了无效的 JSON。") from None

    def tree(self, revision):
        data = self._json(f"/git/trees/{revision}?recursive=1")
        if data.get("truncated"):
            raise DocumentSyncError("GitHub 文档清单不完整，已停止同步。")
        files = {}
        folded = set()
        for item in data.get("tree", []):
            path = item["path"]
            if item.get("type") != "blob" or not path.startswith(self.prefix):
                continue
            relative = path[len(self.prefix):]
            parts = PurePosixPath(relative)
            if not relative.endswith(".md") and relative != "tasks.json":
                continue
            if (parts.is_absolute() or ".." in parts.parts or "\\" in relative or ":" in relative
                    or any(part.startswith(".") for part in parts.parts)
                    or item.get("mode") not in ("100644", "100755")):
                raise DocumentSyncError("文档源存在不支持的文件路径或类型。")
            if relative.casefold() in folded or item.get("size", 0) > 5 * 1024 * 1024:
                raise DocumentSyncError("文档源存在路径冲突或过大的文件。")
            folded.add(relative.casefold())
            files[relative] = item["sha"]
        if not files or len(files) > 5000:
            raise DocumentSyncError("文档源为空或文件数量超过限制。")
        return files

    def snapshot(self):
        commit = self._json("/commits/" + quote(self.settings.doc_sync_branch, safe=""))
        revision = commit["sha"]
        if not re.fullmatch(r"[a-f0-9]{40}", revision):
            raise DocumentSyncError("文档源版本格式错误。")
        timestamp = commit["commit"]["committer"]["date"]
        datetime.fromisoformat(timestamp.replace("Z", "+00:00"))
        return {"revision": revision, "source_time": timestamp, "files": self.tree(revision)}

    def download(self, snapshot, paths):
        archive = self._read("https://codeload.github.com/" + self.settings.doc_sync_repository + "/zip/" + snapshot["revision"], 32 * 1024 * 1024)
        result = {}
        try:
            with zipfile.ZipFile(io.BytesIO(archive)) as package:
                if sum(item.file_size for item in package.infolist()) > 200 * 1024 * 1024:
                    raise DocumentSyncError("解压后的文档源超过大小限制。")
                for item in package.infolist():
                    parts = PurePosixPath(item.filename).parts
                    if len(parts) < 2 or item.is_dir():
                        continue
                    repo_path = PurePosixPath(*parts[1:]).as_posix()
                    if not repo_path.startswith(self.prefix):
                        continue
                    relative = repo_path[len(self.prefix):]
                    if relative not in paths:
                        continue
                    if relative in result:
                        raise DocumentSyncError("压缩包内存在重复文档。")
                    content = package.read(item)
                    if blob_sha(content) != snapshot["files"][relative]:
                        raise DocumentSyncError("文档内容与来源版本校验不一致。")
                    text = content.decode("utf-8-sig")
                    if relative == "tasks.json" and not isinstance(json.loads(text), list):
                        raise DocumentSyncError("文档来源清单格式不正确。")
                    result[relative] = content
        except (zipfile.BadZipFile, UnicodeError, ValueError, KeyError):
            raise DocumentSyncError("文档压缩包或文件内容校验失败。") from None
        if set(result) != set(paths):
            raise DocumentSyncError("下载的文档不完整，已保留原文档。")
        return result


class DocumentSync:
    def __init__(self, settings, indexer=None, source=None):
        self.settings = settings
        self.root = settings.data_dir / "document-sync"
        self.source = source or GitHubDocumentSource(settings)
        self.indexer = indexer
        self._guard = threading.Lock()
        self._stop = threading.Event()
        self._thread = None
        self._job = {"running": False, "message": "", "error": None}

    def _load(self, name, default):
        try:
            return json.loads((self.root / name).read_text(encoding="utf-8"))
        except FileNotFoundError:
            return default

    def identity(self):
        return {"repository": self.settings.doc_sync_repository, "branch": self.settings.doc_sync_branch,
                "subdirectory": self.settings.doc_sync_subdirectory, "docs_dir": str(self.settings.docs_dir.resolve())}

    def status(self):
        state = self._load("state.json", {})
        result = {"enabled": self.settings.doc_sync_enabled, **self._job,
                "interval_seconds": self.settings.doc_sync_interval,
                "source_url": "https://github.com/" + self.settings.doc_sync_repository,
                **{key: state.get(key) for key in ("revision", "source_time", "checked_at", "synced_at",
                    "index_pending", "last_error")}}
        for key in ("last_changes", "available_changes"):
            changes = state.get(key, {})
            result[key] = {kind: paths[:10] for kind, paths in changes.items()}
            result[key + "_counts"] = {kind: len(paths) for kind, paths in changes.items()}
        return result

    def history(self):
        return self._load("history.json", [])[:30]

    def _path(self, root, relative):
        path = root.joinpath(*PurePosixPath(relative).parts)
        if not path.resolve().is_relative_to(root.resolve()):
            raise DocumentSyncError("文档路径超出了同步目录。")
        return path

    def _local_sha(self, path, known=()):
        if not path.is_file():
            return None
        content = path.read_bytes()
        digest = blob_sha(content)
        # Git for Windows 可能把签出的 LF 转为 CRLF，这不是用户修改。
        return digest if digest in known else blob_sha(content.replace(b"\r\n", b"\n"))

    def _move(self, source, destination):
        # 所有目录迁移都限于明确配置的文档根目录和本服务的备份目录。
        docs = self.settings.docs_dir.resolve()
        storage = self.root.resolve()
        for path in (source, destination):
            resolved = path.resolve()
            if resolved != docs and not resolved.is_relative_to(storage):
                raise DocumentSyncError("目录迁移目标超出了文档和备份范围。")
        os.replace(source, destination)

    def _index(self, state):
        if not state.get("index_pending"):
            return
        self._job["message"] = "文档已同步，正在增量更新索引…"
        try:
            if self.indexer:
                self.indexer(lambda message: self._job.update(message=message))
            else:
                from rag.embedding import LocalEmbedding
                from rag.indexing.builder import IndexBuilder
                IndexBuilder(self.settings, LocalEmbedding(self.settings)).build(progress=lambda message: self._job.update(message=message))
        except Exception:
            state["last_error"] = "文档已同步，但索引更新失败；继续使用上一代索引，下次同步会重试。"
            self._job["error"] = state["last_error"]
            logger.exception("文档同步后的索引更新失败")
        else:
            state["index_pending"] = False
            state["last_error"] = None
        atomic_json(self.root / "state.json", state)

    def _sync(self, check_only):
        state = self._load("state.json", {})
        if state.get("source") and state["source"] != self.identity():
            raise DocumentSyncError("同步来源或目录已改变，请为新来源使用独立的数据目录。")
        snapshot = self.source.snapshot()
        expected = snapshot["files"]
        old_files = state.get("files", {})
        if old_files:
            baseline = {path: item["sha"] for path, item in old_files.items()}
        elif self.settings.doc_sync_base_revision:
            baseline = (expected if self.settings.doc_sync_base_revision == snapshot["revision"]
                        else self.source.tree(self.settings.doc_sync_base_revision))
        else:
            baseline = {path: blob_sha(self._path(self.settings.docs_dir, path).read_bytes())
                        for path in expected if self._path(self.settings.docs_dir, path).is_file()}
        actual = {}
        for path in set(expected) | set(baseline):
            file = self._path(self.settings.docs_dir, path)
            actual[path] = self._local_sha(file, (expected.get(path), baseline.get(path)))
        conflicts = [path for path, digest in actual.items() if digest is not None
                     and digest != expected.get(path) and digest != baseline.get(path)]
        if conflicts:
            raise DocumentSyncError("检测到本地修改，已停止同步以保留修改：" + "、".join(sorted(conflicts)[:8]))
        changes = {"added": sorted(set(expected) - set(baseline)),
                   "modified": sorted(path for path in expected.keys() & baseline.keys() if expected[path] != baseline[path]),
                   "deleted": sorted(set(baseline) - set(expected))}
        previous_index_error = state.get("last_error") if state.get("index_pending") else None
        state.update(source=self.identity(), checked_at=now(), available_changes=changes, last_error=previous_index_error)
        if check_only:
            atomic_json(self.root / "state.json", state)
            self._job["message"] = "文档更新检查完成。"
            return self.status()
        writes = [path for path, sha in expected.items() if actual.get(path) != sha]
        deletes = [path for path in baseline.keys() - expected.keys() if actual.get(path) is not None]
        new_version = bool(writes or deletes or not old_files)
        if not new_version:
            state["revision"] = snapshot["revision"]
            state["source_time"] = snapshot["source_time"]
            atomic_json(self.root / "state.json", state)
            self._index(state)
            self._job["message"] = "文档没有变化。" if not state.get("index_pending") else "等待索引更新重试。"
            return self.status()
        payload = self.source.download(snapshot, writes) if writes else {}
        version_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%f") + "_" + snapshot["revision"][:10]
        version = self.root / "versions" / version_id
        version.mkdir(parents=True, exist_ok=False)
        stage = version / "staged"
        docs = self.settings.docs_dir.resolve()
        if docs == self.settings.project_dir.resolve() or self.root.resolve().is_relative_to(docs):
            raise DocumentSyncError("文档目录不能是项目根目录或包含同步数据目录。")
        if docs.exists():
            shutil.copytree(docs, stage, symlinks=True)
        else:
            stage.mkdir()
        for path in deletes:
            self._path(stage, path).unlink()
        for path, content in payload.items():
            file = self._path(stage, path)
            file.parent.mkdir(parents=True, exist_ok=True)
            file.write_bytes(content)
        files = {}
        for path, sha in expected.items():
            previous = old_files.get(path, {})
            files[path] = previous if previous.get("sha") == sha else {
                "sha": sha, "source_revision": snapshot["revision"], "updated_at": snapshot["source_time"],
                "synced_at": state["checked_at"], "date_basis": "source_commit"}
        state.update(revision=snapshot["revision"], source_time=snapshot["source_time"],
                     synced_at=state["checked_at"], files=files, last_changes=changes,
                     available_changes={key: [] for key in changes}, index_pending=True)
        old_state = (self.root / "state.json").read_bytes() if (self.root / "state.json").exists() else None
        backup = version / "before"
        moved = False
        # 与所有索引构建共用锁，索引永远不会读到半套文档。
        with FileLock(str(self.settings.data_dir / "index.lock"), timeout=1):
            for path, digest in actual.items():
                file = self._path(docs, path)
                current = self._local_sha(file, (expected.get(path), baseline.get(path)))
                if current != digest:
                    raise DocumentSyncError("同步期间本地文档发生变化，请重新检查。")
            try:
                if docs.exists():
                    self._move(docs, backup)
                    moved = True
                self._move(stage, docs)
                atomic_json(self.root / "state.json", state)
            except Exception:
                if moved:
                    if docs.exists():
                        self._move(docs, version / "failed")
                    self._move(backup, docs)
                if old_state is not None:
                    (self.root / "state.json").write_bytes(old_state)
                raise
        record = {"id": version_id, "revision": snapshot["revision"], "source_time": snapshot["source_time"],
                  "synced_at": state["synced_at"], **changes,
                  "compare_url": "https://github.com/" + self.settings.doc_sync_repository + "/commit/" + snapshot["revision"]}
        atomic_json(version / "manifest.json", record)
        atomic_json(self.root / "history.json", [record, *self.history()][:30])
        self._index(state)
        self._job["message"] = "文档同步和索引更新完成。" if not state.get("index_pending") else state["last_error"]
        return self.status()

    def _execute(self, check_only):
        self._job.update(running=True, message="正在检查文档来源…", error=None)
        try:
            self.root.mkdir(parents=True, exist_ok=True)
            with FileLock(str(self.root / "sync.lock"), timeout=0):
                return self._sync(check_only)
        except Timeout:
            self._job["error"] = "文档同步或索引任务正在进行，请稍后重试。"
            raise DocumentSyncBusy(self._job["error"]) from None
        except Exception as error:
            message = str(error) if isinstance(error, DocumentSyncError) else "文档同步失败，请查看服务日志。"
            self._job["error"] = message
            state = self._load("state.json", {})
            state.update(checked_at=now(), last_error=message)
            atomic_json(self.root / "state.json", state)
            logger.exception("文档同步失败")
            raise DocumentSyncError(message) from error
        finally:
            self._job["running"] = False
            self._guard.release()

    def run(self, check_only=False):
        if not self._guard.acquire(blocking=False):
            raise DocumentSyncBusy("文档同步正在进行。")
        self._execute(check_only)
        return self.status()

    def trigger(self, check_only=False):
        if not self._guard.acquire(blocking=False):
            return False
        self._job["running"] = True
        def work():
            try:
                self._execute(check_only)
            except DocumentSyncError:
                pass
        threading.Thread(target=work, name="document-sync", daemon=True).start()
        return True

    def start(self):
        if not self.settings.doc_sync_enabled or self._thread is not None:
            return
        self._thread = threading.Thread(target=self._loop, name="document-sync-scheduler", daemon=True)
        self._thread.start()

    def stop(self):
        self._stop.set()

    def _loop(self):
        while not self._stop.is_set():
            checked = self.status().get("checked_at")
            due = not checked or (datetime.now(timezone.utc) - datetime.fromisoformat(checked.replace("Z", "+00:00"))).total_seconds() >= self.settings.doc_sync_interval
            if due:
                try:
                    self.run()
                except DocumentSyncError:
                    pass
            self._stop.wait(min(60, self.settings.doc_sync_interval))
