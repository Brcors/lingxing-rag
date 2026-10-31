import hashlib
import json
import re
from pathlib import Path
from urllib.parse import urlparse

from rag.types import Document, Section
from rag.ingestion.fences import closes_fence, opening_fence


def digest(value):
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def section_kind(heading):
    if any(k in heading for k in ("示例", "cURL", "curl")):
        return "example"
    if any(k in heading for k in ("返回", "响应")):
        return "response"
    if "参数" in heading:
        return "parameters"
    if any(k in heading for k in ("错误", "注意", "限制")):
        return "constraint"
    return "overview"


def safe_url(value):
    parsed = urlparse(value or "")
    return value if parsed.scheme in ("http", "https") and parsed.netloc else ""


def normalize_category(value):
    return re.sub(r"\s*/\s*", "/", value.strip())


def title_key(value):
    # 抓取器移除了 Windows 文件名不允许的字符；仅在候选来源唯一时回填。
    return re.sub(r'[\s<>:"/\\|?*、]', "", value).casefold()


class MarkdownLoader:
    def __init__(self, settings):
        self.settings = settings

    def load(self):
        root = Path(self.settings.docs_dir)
        if not root.is_dir():
            raise FileNotFoundError(f"找不到文档目录：{root}")
        lookup = {}
        normalized_lookup = {}
        tasks = root / "tasks.json"
        if tasks.exists():
            for task in json.loads(tasks.read_text(encoding="utf-8-sig")):
                category = normalize_category(task.get("category", ""))
                title = task.get("title", "")
                url = safe_url(task.get("url", ""))
                lookup[(category, title)] = url
                if url:
                    normalized_lookup.setdefault((category, title_key(title)), set()).add(url)
        documents = []
        for path in sorted(root.rglob("*.md")):
            if path.name.lower() == "index.md" or any(p.startswith(".") for p in path.relative_to(root).parts):
                continue
            relative = path.relative_to(root).as_posix()
            category = relative.rsplit("/", 1)[0] if "/" in relative else "General"
            content = path.read_text(encoding="utf-8-sig").strip()
            if not content:
                continue
            document = self.parse(content, relative, category, lookup.get((category, path.stem), ""))
            if not document.source_url:
                document.source_url = lookup.get((category, document.title), "")
            if not document.source_url:
                candidates = set()
                for title in (path.stem, document.title):
                    candidates.update(normalized_lookup.get((category, title_key(title)), set()))
                if len(candidates) == 1:
                    document.source_url = next(iter(candidates))
            documents.append(document)
        supplements = Path(self.settings.supplement_dir)
        if supplements.is_dir():
            for path in sorted(supplements.glob("*.md")):
                content = path.read_text(encoding="utf-8-sig").strip()
                source = re.search(r"^来源：\s*(https?://\S+)", content, flags=re.M)
                documents.append(self.parse(content, "补充说明/" + path.name, "接入指引",
                                            safe_url(source.group(1)) if source else "", verified=True))
        return documents

    @staticmethod
    def parse(content, relative, category, source_url="", verified=False):
        document_id = digest(relative)[:24]
        title_match = re.search(r"^#\s+(.+)", content, re.M)
        title = title_match.group(1).strip() if title_match else Path(relative).stem
        api_path, method = "", ""
        api_header = re.search(r"\|\s*API Path\s*\|[^\n]*\n[^\n]*\n([^\n]+)", content, re.I)
        if api_header:
            row = api_header.group(1)
            match = re.search(r"`?(/[^`|\s]+)`?", row)
            api_path = match.group(1) if match else ""
            method_match = re.search(r"\b(GET|POST|PUT|DELETE|PATCH)\b", row, re.I)
            method = method_match.group(1).upper() if method_match else ""
        sections, lines, stack, fence = [], [], [], None
        current_heading = "接口概览"

        def flush():
            text = "\n".join(lines).strip()
            if text:
                order = len(sections)
                sections.append(Section(f"{document_id}:{order}", current_heading, order,
                                        section_kind(current_heading), text))
            lines.clear()

        for line in content.splitlines():
            marker = opening_fence(line)
            if fence is not None:
                if closes_fence(line, fence):
                    fence = None
                lines.append(line)
                continue
            if marker is not None:
                fence = marker
                lines.append(line)
                continue
            heading = re.match(r"^(#{1,6})\s+(.+?)\s*#*\s*$", line)
            if heading:
                level, label = len(heading.group(1)), heading.group(2).replace("**", "")
                if level == 1:
                    continue
                flush()
                stack = [(depth, text) for depth, text in stack if depth < level]
                stack.append((level, label))
                current_heading = " / ".join(text for _, text in stack)
            else:
                lines.append(line)
        flush()
        return Document(document_id, relative, title, category, api_path, method, source_url,
                        digest(content), content, sections,
                        is_legacy=bool(re.search(r"（旧）|\(旧\)|已废弃|已弃用", title)),
                        verified_update=verified)
