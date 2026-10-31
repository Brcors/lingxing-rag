"""共享的 Markdown 代码围栏规则，防止嵌套短围栏被当作结束标记。"""
import re


def opening_fence(line):
    match = re.match(r"^\s*(`{3,}|~{3,})(.*)$", line)
    return match.group(1) if match else None


def closes_fence(line, marker):
    match = re.match(r"^\s*(`{3,}|~{3,})\s*$", line)
    return bool(match and match.group(1)[0] == marker[0] and len(match.group(1)) >= len(marker))
