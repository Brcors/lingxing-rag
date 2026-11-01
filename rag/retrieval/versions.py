"""Read local sync records as citable evidence; never contact or modify the upstream."""
import json

from rag.context import estimated_llm_tokens
from rag.types import ContextBundle


def version_evidence(settings, subject="", paths=()):
    root = settings.data_dir / "document-sync"
    try:
        state = json.loads((root / "state.json").read_text(encoding="utf-8"))
        history_path = root / "history.json"
        history = json.loads(history_path.read_text(encoding="utf-8")) if history_path.exists() else []
        if not isinstance(state, dict) or not isinstance(history, list):
            raise ValueError("invalid version records")
        identity = state.get("source", {})
        if (identity.get("docs_dir") != str(settings.docs_dir.resolve()) or
                identity.get("repository") != settings.doc_sync_repository or
                identity.get("branch") != settings.doc_sync_branch or
                identity.get("subdirectory") != settings.doc_sync_subdirectory):
            raise ValueError("unrelated document source")
    except (OSError, ValueError, TypeError, AttributeError):
        return ContextBundle("", [], 0, ["没有当前文档源的可用本机版本记录，请先在侧栏检查或同步文档。"])

    scoped = bool(subject)
    matched = set(paths)
    if scoped and not matched:
        matched = {path for path in state.get("files", {}) if subject.casefold() in path.casefold()}
    files = state.get("files", {})
    file_versions = [{"path": path, **{key: files[path].get(key) for key in
                     ("source_revision", "updated_at", "synced_at")}}
                     for path in sorted(matched)[:5] if path in files]
    records = []
    for record in history[:30]:
        changes = {kind: record.get(kind, []) for kind in ("added", "modified", "deleted")}
        if scoped:
            changes = {kind: [path for path in values if path in matched or
                       (not matched and subject.casefold() in path.casefold())]
                       for kind, values in changes.items()}
            if not any(changes.values()):
                continue
        records.append({"revision": record.get("revision"), "source_time": record.get("source_time"),
                        "synced_at": record.get("synced_at"),
                        "counts": {kind: len(values) for kind, values in changes.items()},
                        "file_preview": {kind: values[:8] for kind, values in changes.items()}})
        if len(records) >= 5:
            break
    data = {"scope": "指定文档" if scoped else "整个文档仓库",
            "repository": settings.doc_sync_repository,
            "repository_revision": state.get("revision"),
            "repository_commit_time": state.get("source_time"),
            "last_checked_at": state.get("checked_at"), "synced_at": state.get("synced_at"),
            "index_pending": bool(state.get("index_pending")), "last_error": state.get("last_error"),
            "file_versions": file_versions, "recent_records": records}
    description = ("以下是本机保存的 GitHub 来源版本记录；没有在本次回答中联网检查。"
                   "上游提交时间不是官方文档发布日期，也不能证明这是官方最新 API。"
                   "记录仅含文件增删改，不含字段级差异。预览未列出全部文件，历史最多保留30条。")
    if scoped:
        description += "仓库整体提交时间不能当作指定接口的更新时间；未命中文件记录不代表该接口从未修改。"
    if state.get("index_pending"):
        description += "同步文档尚未完成索引更新，接口回答可能仍使用上一代索引。"
    text = description + "\n" + json.dumps(data, ensure_ascii=False, indent=2)
    # Keep one atomic record within the same context budget used for document evidence.
    prefix = "[S1] 本机文档同步与版本记录\n"
    while estimated_llm_tokens(prefix + text) > settings.context_budget and records:
        records.pop()
        text = description + "\n" + json.dumps(data, ensure_ascii=False, indent=2)
    if estimated_llm_tokens(prefix + text) > settings.context_budget:
        return ContextBundle("", [], 0, ["版本记录超过当前上下文预算，请缩小到具体接口或文件。"])
    source = {"id": "S1", "title": "本机文档同步与版本记录", "category": "文档版本",
              "heading": "指定文档记录" if scoped else "仓库同步记录", "api_path": "", "method": "",
              "relative_path": "", "source_type": "document_version", "excerpt": text,
              "source_url": "https://github.com/" + settings.doc_sync_repository,
              "channels": ["local_versions"], "source_revision": state.get("revision") or "",
              "updated_at": "", "synced_at": state.get("synced_at") or "", "update_basis": "source_commit"}
    context = prefix + text
    return ContextBundle(context, [source], estimated_llm_tokens(context), [description])
