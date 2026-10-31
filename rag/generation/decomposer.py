"""仅对流程和前置条件类问题生成少量独立检索查询。"""
import json
import logging
import re

logger = logging.getLogger(__name__)

WORKFLOW_CUES = re.compile(
    r"流程|全流程|全过程|工作流|步骤|依次|调用顺序|调用链|前置条件|依赖关系|"
    r"跳过.{0,12}(?:步骤|接口|环节)|从.{1,40}(?:到|至)|"
    r"先.{1,24}(?:再|然后|之后)|端到端|完整实现|涉及.{0,16}接口|"
    r"(?:创建|生成|确认|获取|上传|绑定|查询|修改|更新|释放|发货|同步)"
    r".{0,24}(?:并且|同时|然后|接着|之后|再调用|以及).{0,24}"
    r"(?:创建|生成|确认|获取|上传|绑定|查询|修改|更新|释放|发货|同步)"
)


def should_decompose(question):
    """用便宜的本地规则筛选，普通的单接口/单字段查询不额外调用 LLM。"""
    return bool(WORKFLOW_CUES.search(question or ""))


class QuestionDecomposer:
    def __init__(self, settings, llm):
        self.settings = settings
        self.llm = llm

    def decompose(self, question, history=None):
        if (not self.settings.query_decomposition_enabled or not self.settings.api_key or
                not should_decompose(question)):
            return [], None

        recent_user_questions = [
            item["content"][:300]
            for item in (history or [])
            if item.get("role") == "user" and isinstance(item.get("content"), str)
        ][-2:]
        payload = json.dumps(
            {"question": question, "recent_user_questions": recent_user_questions},
            ensure_ascii=False,
        )
        messages = [
            {
                "role": "system",
                "content": (
                    f"你是检索查询拆分器，不回答用户问题。仅当问题确实包含多个业务阶段或前置依赖时，"
                    f"拆成 2 到 {self.settings.query_decomposition_max_subquestions} 条可独立检索的中文子问题；每条写清原问题中的业务对象。"
                    "只使用原问题与最近用户问题明确提供的对象和阶段，不要补造接口名、流程步骤或依赖。"
                    "简单问题不要拆。不要输出分析过程。严格输出 JSON："
                    '{"queries":["子问题1","子问题2"]}。'
                ),
            },
            {"role": "user", "content": payload},
        ]
        try:
            raw = self.llm.complete(
                messages,
                max_tokens=self.settings.query_decomposition_max_tokens,
                temperature=0.0,
            )
            return self._parse(raw, question), None
        except Exception:
            logger.warning("多步骤问题拆解失败，回退原问题检索。", exc_info=True)
            return [], "问题拆解暂不可用，本次将按原问题检索。"

    def _parse(self, raw, original_question):
        start = raw.find("{")
        if start < 0:
            return []
        try:
            data, _ = json.JSONDecoder().raw_decode(raw[start:])
        except json.JSONDecodeError:
            return []
        candidates = data.get("queries") if isinstance(data, dict) else None
        if not isinstance(candidates, list):
            return []

        original_key = re.sub(r"\W+", "", original_question).casefold()
        seen = {original_key}
        result = []
        for candidate in candidates:
            if not isinstance(candidate, str):
                continue
            query = re.sub(r"\s+", " ", candidate).strip()
            key = re.sub(r"\W+", "", query).casefold()
            if len(query) < 6 or len(query) > 240 or not key or key in seen:
                continue
            seen.add(key)
            result.append(query)
            if len(result) >= self.settings.query_decomposition_max_subquestions:
                break
        return result if len(result) >= 2 else []
