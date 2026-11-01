"""Local intent rules, evidence checks and context packing for document retrieval."""
import re
from pathlib import PurePosixPath

from rag.context import estimated_llm_tokens
from rag.generation.decomposer import should_decompose
from rag.retrieval.query import context_anchor, is_detail_followup, search_query
from rag.retrieval.versions import version_evidence
from rag.types import ContextBundle

LABELS = {"endpoint": "接口", "parameters": "请求参数", "response": "返回字段",
          "version": "版本记录", "workflow": "业务流程", "general": "文档"}
PATHS = re.compile(r"/[A-Za-z0-9_.{}-]+(?:/[A-Za-z0-9_.{}-]+)+")
VERSION = re.compile(r"版本记录|更新记录|变更记录|更新时间|何时更新|什么时候更新|最近.*(?:变更|更新|变化)|"
                     r"接口.{0,16}(?:有更新|更新过|有变化|有变更|更新日期)|"
                     r"(?:文档|资料|快照|知识库).{0,12}(?:版本|同步|更新|变更)|changelog", re.I)


def classify(question, history=None):
    query = search_query(question, history)
    time_field = re.search(r"更新时间.{0,3}(?:参数|字段)|(?:参数|字段).{0,12}更新时间", question)
    if VERSION.search(question) and not time_field:
        return "version"
    if re.search(r"返回|响应|\bresponse\b", question, re.I):
        return "response"
    if re.search(r"参数|字段|必填|必传|传参|必须|枚举|分页|类型|\bparam", question, re.I):
        return "parameters"
    if should_decompose(question) or re.search(r"区别|对比|比较", question):
        return "workflow"
    if re.search(r"接口|路径|请求方式|请求方法|endpoint|\bHTTP\b", query, re.I) or PATHS.search(query):
        return "endpoint"
    return "general"


def version_subject(question, history=None):
    query = search_query(question, history)
    paths = PATHS.findall(query)
    if paths:
        return paths[0]
    query = re.sub(r"文档|资料|知识库|快照|版本|记录|更新|变更|变化|同步|检查|接口|最后一次|最近一次|最近|最新|最后|何时|什么时候|"
                   r"时候|时间|日期|整个|所有|领星|上游|github|查看|一下|帮我|请问|请|怎么|哪些|什么|是否|有没有|"
                   r"这个|该|它的|它|的|是|了|吗|有|追问", "", query, flags=re.I)
    query = re.sub(r"[^\w\u3400-\u9fff]+", "", query)
    return query if len(query) > 1 else ""


def identifiers(question):
    question = PATHS.sub("", question)
    words = re.findall(r"[A-Za-z_][A-Za-z0-9_]*(?:>>[A-Za-z_][A-Za-z0-9_]*)*", question)
    ignored = {"api", "get", "post", "put", "patch", "delete", "http", "https", "fba", "sta", "sku", "json"}
    return list(dict.fromkeys(word for word in words if word.lower() not in ignored))[:8]


def observe(question, intent, bundle, stats):
    documents = [s for s in bundle.sources if s.get("source_type") != "liked_qa"]
    if not documents or stats.get("rerank_low_confidence"):
        return "没有达到相关性要求的文档证据"
    paths = PATHS.findall(question)
    if paths and not all(any(path.casefold() == s.get("api_path", "").casefold() for s in documents) for path in paths):
        return "尚未命中问题中指定的接口路径"
    if paths:
        documents = [source for source in documents if source.get("api_path", "").casefold()
                     in {path.casefold() for path in paths}]
    else:
        anchored = [source for source in documents if len(source.get("title", "")) >= 4 and
                    source["title"].casefold() in question.casefold()]
        if anchored:
            documents = anchored
    evidence = "\n".join(s.get("excerpt", "") for s in documents).casefold()
    if any(not re.search(r"(?<![a-z0-9_])" + re.escape(word.casefold()) + r"(?![a-z0-9_])", evidence)
           for word in identifiers(question)):
        return "指定字段或标识符尚未出现在证据中"
    if intent in ("parameters", "response"):
        cue = r"参数|必填|请求字段" if intent == "parameters" else r"返回|响应|response"
        if not any(re.search(cue, s.get("heading", ""), re.I) for s in documents):
            return "尚未命中所需章节"
    if intent == "workflow" and re.search(r"哪些接口|依赖|全流程|调用链|对比|区别|比较|先.+(?:再|然后)", question):
        if len({s.get("document_id") for s in documents}) < 2:
            return "跨接口问题目前只有一份文档依据"
    return ""


def parameter_clarification(question, history, bundle):
    """A bare parameter follow-up cannot silently choose a step in a multi-API workflow."""
    if not history or not is_detail_followup(question):
        return None
    expanded = search_query(question, history).casefold()
    explicit_paths = set(PATHS.findall(expanded))
    if len(explicit_paths) == 1:
        return None
    endpoints = {}
    anchored = set()
    for source in bundle.sources:
        path = source.get("api_path", "")
        if not path or source.get("source_type") == "liked_qa":
            continue
        endpoints.setdefault(path, source)
        names = (source.get("title", ""), PurePosixPath(source.get("relative_path", "")).stem)
        if any(len(name) >= 4 and name.casefold() in expanded for name in names):
            anchored.add(path)
    if len(anchored) == 1:
        return None
    broad_topic = re.search(r"怎么做|如何|流程|步骤|哪些接口|全流程", context_anchor(history))
    if len(endpoints) < 2 and not broad_topic:
        return None
    answer = "你是在问上面哪一步或哪个接口的参数？请先选定接口，我再列出对应的必填项、类型和示例。"
    if endpoints:
        answer += "\n\n本次检索到的候选接口：\n" + "\n".join(
            f"- {source.get('title', path)}：`{path}` [{source['id']}]"
            for path, source in list(endpoints.items())[:6])
    else:
        answer += "\n\n请补充具体步骤、接口名称或接口路径。"
    return answer


def merge_bundles(first, second, budget):
    # Interleave independent source units so either retrieval can contribute within the budget.
    candidates = []
    for i in range(max(len(first.sources), len(second.sources))):
        for sources in (second.sources, first.sources):
            if i < len(sources):
                candidates.append(sources[i])
    sources, parts, seen = [], [], set()
    for source in candidates:
        key = (source.get("document_id"), source.get("heading"), source.get("excerpt"))
        if key in seen:
            continue
        seen.add(key)
        sid = f"S{len(sources) + 1}"
        part = (f"[{sid}] {source.get('title', '')}\n章节：{source.get('heading', '')}\n"
                f"接口：{source.get('method', '')} {source.get('api_path', '')}\n"
                f"来源：{source.get('source_url') or source.get('relative_path', '')}\n"
                f"{source.get('excerpt', '')}")
        if source.get("verified_update"):
            part = part.replace("\n章节：", "\n来源属性：已核验的官方更新说明。\n章节：", 1)
        if estimated_llm_tokens("\n\n---\n\n".join(parts + [part])) <= budget:
            sources.append({**source, "id": sid})
            parts.append(part)
    text = "\n\n---\n\n".join(parts)
    first_notices = ["首次检索：" + note if "门槛" in note else note for note in first.notices]
    notices = list(dict.fromkeys(first_notices + second.notices))
    if len(sources) < len(seen):
        notices.append("两轮证据超过上下文预算，已保留完整片段；可缩小问题范围后继续查询。")
    return ContextBundle(text, sources, estimated_llm_tokens(text) if text else 0, notices)
