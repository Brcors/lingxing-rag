"""优先章节、段落、完整表格行；仅超长单元格和代码行才按 token 偏移继续切分。"""
import re

from rag.ingestion.parser import digest
from rag.ingestion.fences import closes_fence, opening_fence
from rag.types import Chunk


def markdown_blocks(text):
    lines = text.splitlines()
    result, i = [], 0
    while i < len(lines):
        if not lines[i].strip():
            i += 1
            continue
        start = i
        marker = opening_fence(lines[i])
        if marker:
            i += 1
            while i < len(lines):
                if closes_fence(lines[i], marker):
                    i += 1
                    break
                i += 1
            result.append(("code", "\n".join(lines[start:i])))
        elif lines[i].lstrip().startswith("|"):
            while i < len(lines) and lines[i].lstrip().startswith("|"):
                i += 1
            result.append(("table", "\n".join(lines[start:i])))
        else:
            while i < len(lines) and lines[i].strip():
                if i > start and (lines[i].lstrip().startswith("|") or re.match(r"^\s*(`{3,}|~{3,})", lines[i])):
                    break
                i += 1
            result.append(("paragraph", "\n".join(lines[start:i])))
    return result


class SemanticChunker:
    def __init__(self, settings, embedding):
        self.settings = settings
        self.embedding = embedding

    def _split_text(self, text, limit, overlap=0):
        encoded = self.embedding.token_offsets(text)
        offsets = encoded["offset_mapping"]
        if not offsets:
            return [text] if text.strip() else []
        pieces, start = [], 0
        while start < len(offsets):
            end = min(start + limit, len(offsets))
            # 仅在块尾部寻找句子或换行边界，避免把短句切得过碎。
            if end < len(offsets):
                low = start + max(1, int((end - start) * 0.7))
                for point in range(end - 1, low, -1):
                    boundary = text[offsets[point][1] - 1:offsets[point][1]]
                    if boundary in "。；;\n":
                        end = point + 1
                        break
            left = 0 if start == 0 else offsets[start][0]
            right = len(text) if end == len(offsets) else offsets[end][0]
            piece = text[left:right].strip()
            if piece:
                pieces.append(piece)
            if end == len(offsets):
                break
            start = max(start + 1, end - overlap)
        return pieces

    def _table(self, text, limit):
        rows = text.splitlines()
        if len(rows) < 3:
            return self._split_text(text, limit)
        header = "\n".join(rows[:2])
        pieces, current = [], []
        for row in rows[2:]:
            candidate = header + "\n" + "\n".join(current + [row])
            if self.embedding.count(candidate) <= limit:
                current.append(row)
                continue
            if current:
                pieces.append(header + "\n" + "\n".join(current))
                # 仅在放得下时重叠上一条完整字段。
                previous = current[-1]
                current = [previous] if self.embedding.count(header + "\n" + previous + "\n" + row) <= limit else []
            if self.embedding.count(header + "\n" + row) <= limit:
                current.append(row)
            else:
                cells = [cell.strip() for cell in row.strip("|").split("|")]
                label = cells[0] if cells else "长字段"
                note = f"表格长字段 {label}（分段，完整内容见所属章节）：\n"
                budget = max(32, limit - self.embedding.count(note))
                pieces.extend(note + part for part in self._split_text(row, budget, min(20, budget // 5)))
                current = []
        if current:
            pieces.append(header + "\n" + "\n".join(current))
        return pieces

    def build(self, documents):
        chunks = []
        for document in documents:
            for section in document.sections:
                prefix = (f"文档：{document.title}\n分类：{document.category}\n"
                          f"接口：{document.method} {document.api_path}\n章节：{section.heading}\n\n")
                overhead = self.embedding.count(prefix)
                limit = max(64, min(self.settings.chunk_target, self.settings.chunk_max - overhead - 4))
                for kind, block in markdown_blocks(section.text):
                    if self.embedding.count(block) <= limit:
                        pieces = [block]
                    elif kind == "table":
                        pieces = self._table(block, limit)
                    elif kind == "code":
                        code_lines = block.splitlines()
                        opening = code_lines[0]
                        closing = opening_fence(opening)
                        has_closing = len(code_lines) > 1 and closes_fence(code_lines[-1], closing)
                        code = "\n".join(code_lines[1:-1] if has_closing else code_lines[1:])
                        note = "代码示例片段（完整示例见原文）：\n"
                        budget = max(32, limit - self.embedding.count(note + opening + closing) - 8)
                        pieces = [note + opening + "\n" + piece + "\n" + closing
                                  for piece in self._split_text(code, budget, 20)]
                    else:
                        pieces = self._split_text(block, limit, min(self.settings.overlap, limit // 4))
                    for piece in pieces:
                        embedding_text = prefix + piece
                        # 极端长标题也必须显式报错，不能交由模型静默截断。
                        tokens = self.embedding.count(embedding_text)
                        if tokens > self.settings.chunk_max:
                            raise ValueError(f"片段超过 token 上限：{document.relative_path} / {section.heading} ({tokens})")
                        chunk_id = digest(section.section_id + "\n" + embedding_text)[:32]
                        if chunks and chunks[-1].chunk_id == chunk_id:
                            continue
                        chunks.append(Chunk(chunk_id, document.document_id, section.section_id,
                                            section.order, len(chunks), document.title, section.heading,
                                            document.category, document.api_path, document.method,
                                            section.kind, piece, embedding_text, tokens))
        # 相同片段可能在一个章节内出现多次；Chroma ID 必须唯一。
        return list({chunk.chunk_id: chunk for chunk in chunks}.values())
