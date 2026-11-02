"""Conservative, offline accounting shared by every model request.

UTF-8 bytes deliberately overestimate text tokens. This is a local capacity
guard, not the provider's tokenizer or billing count. Chat framing and a
separate safety margin account for protocol overhead and estimation error.
"""
from rag.generation.errors import ContextBudgetError


def message_tokens(messages):
    return 32 + sum(16 + len(item["role"].encode("utf-8")) +
                    len(item["content"].encode("utf-8")) for item in messages)


def input_budget(settings, max_tokens=None):
    output = settings.max_output_tokens if max_tokens is None else max_tokens
    if not isinstance(output, int) or isinstance(output, bool) or output < 1:
        raise ContextBudgetError("回答长度配置无效，请检查 max_tokens。")
    available = settings.llm_context_window - output - settings.llm_context_safety_margin
    if available <= 0:
        raise ContextBudgetError("上下文预算不足以预留回答空间，请检查模型配置。")
    return available


def ensure_request_fits(settings, messages, max_tokens=None):
    if message_tokens(messages) > input_budget(settings, max_tokens):
        raise ContextBudgetError("问题或必要资料超过本次上下文预算，请缩小问题范围或分步提问。")
