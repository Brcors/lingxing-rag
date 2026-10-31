"""只变换正文引用，保留 fenced / indented / inline code 的原始文本。"""
import re

from markdown_it import MarkdownIt

CITATION = re.compile(r"(?<!!)\[(S\d+)\](?!\()")


def _outside_inline_code(text, callback):
    result, cursor = [], 0
    runs = list(re.finditer(r"`+", text))
    i = 0
    while i < len(runs):
        opening = runs[i]
        escapes = len(text[:opening.start()]) - len(text[:opening.start()].rstrip("\\"))
        if escapes % 2:
            i += 1
            continue
        closing = next((j for j in range(i + 1, len(runs))
                        if len(runs[j].group()) == len(opening.group())), None)
        if closing is None:
            i += 1
            continue
        end = runs[closing].end()
        result.append(CITATION.sub(callback, text[cursor:opening.start()]))
        result.append(text[opening.start():end])
        cursor, i = end, closing + 1
    result.append(CITATION.sub(callback, text[cursor:]))
    return "".join(result)


def transform_prose_citations(text, callback):
    tokens = MarkdownIt("commonmark").parse(text)
    lines = text.splitlines(keepends=True)
    offsets = [0]
    for line in lines:
        offsets.append(offsets[-1] + len(line))
    result, cursor = [], 0
    for token in tokens:
        if token.type not in ("fence", "code_block") or not token.map:
            continue
        start, end = offsets[token.map[0]], offsets[token.map[1]]
        result.append(_outside_inline_code(text[cursor:start], callback))
        result.append(text[start:end])
        cursor = end
    result.append(_outside_inline_code(text[cursor:], callback))
    return "".join(result)
