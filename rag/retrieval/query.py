"""Local query rewriting shared by retrieval and conversation context."""
import re


def is_detail_followup(question):
    compact = re.sub(r"[\s，。？！、,?!：:]+", "", question or "")
    return bool(len(compact) < 80 and re.search(r"参数|字段|必填|分页|请求方式|返回结果", compact) and
                re.fullmatch(r"(?:那|那么|请问|请|还|需要|要|有|哪些|什么|具体|必填|可选|请求|返回|响应|参数|字段|"
                             r"类型|格式|分页|方式|结果|项|怎么|如何|传|填写|校验|吗|呢|的|是|这个|它|该接口|是否|和|及|多少)+", compact))


def context_anchor(history):
    questions = [m["content"] for m in (history or []) if m.get("role") == "user" and isinstance(m.get("content"), str)]
    return next((value for value in reversed(questions) if not is_detail_followup(value)), questions[-1] if questions else "")


def search_query(question, history=None):
    query = question.strip()
    if history and len(query) < 80 and (is_detail_followup(query) or
            any(word in query for word in ("这个", "该接口", "它的", "上面", "刚才", "那它"))):
        previous = context_anchor(history)
        if previous:
            query = previous[:160] + "\n追问：" + query
    aliases = {"FNSKU": "FNSKU 商品标签", "access token": "access_token", "STA": "STA 货件"}
    for word, expansion in aliases.items():
        if word.lower() in query.lower() and expansion.lower() not in query.lower():
            query += " " + expansion
    return query
