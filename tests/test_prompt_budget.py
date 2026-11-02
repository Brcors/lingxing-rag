"""No paid API calls: exercise full payload limits and bounded recovery."""
from dataclasses import replace
import json

import pytest

from rag.generation.budget import ensure_request_fits, message_tokens
from rag.generation.cancellation import CancellationToken, GenerationCancelled
from rag.generation.client import DeepSeekClient, LLMError
from rag.generation.errors import ContextBudgetError, ContextLengthError
from rag.generation.prompts import build_messages, prepare_prompt
from rag.service import RAGService
from rag.types import ContextBundle


def evidence(*excerpts):
    sources = [{"id": f"S{i}", "title": f"接口资料{i}", "document_id": str(i),
                "api_path": "/listing/publish", "method": "POST", "heading": "请求参数",
                "relative_path": f"{i}.md", "excerpt": text}
               for i, text in enumerate(excerpts, 1)]
    return ContextBundle("\n\n---\n\n".join(f"[{s['id']}]\n{s['excerpt']}" for s in sources), sources, 0)


def with_input_budget(settings, size):
    return replace(settings, llm_context_window=size + settings.max_output_tokens + settings.llm_context_safety_margin)


def test_full_payload_accounting_includes_unicode_roles_output_and_margin(settings):
    messages = [{"role": "system", "content": "系统"},
                {"role": "user", "content": '参数😀 ```json\n{"sku":"A"}\n```'}]
    size = message_tokens(messages)
    assert size > sum(len(m["content"]) for m in messages)
    exact = with_input_budget(settings, size)
    ensure_request_fits(exact, messages)
    with pytest.raises(ContextBudgetError):
        ensure_request_fits(replace(exact, llm_context_window=exact.llm_context_window - 1), messages)
    # An explicit output override is reserved too (decomposition / memory calls).
    with pytest.raises(ContextBudgetError):
        ensure_request_fits(exact, messages, max_tokens=settings.max_output_tokens + 1)


@pytest.mark.parametrize("method", ["complete", "stream"])
def test_final_client_guard_sends_no_over_budget_request(settings, monkeypatch, method):
    monkeypatch.setattr("rag.generation.client.requests.post", lambda *a, **kw: pytest.fail("must not send"))
    llm = DeepSeekClient(replace(settings, api_key="test-key"))
    with pytest.raises(ContextBudgetError):
        result = getattr(llm, method)([{"role": "user", "content": "超长" * 20000}])
        if method == "stream":
            list(result)


def test_under_budget_retains_complete_history_summary_and_evidence(settings):
    bundle = evidence("| 字段 | 必填 |\n| --- | --- |\n| store_id | 是 |")
    history = [{"role": "user", "content": "问题"}, {"role": "assistant", "content": "a" * 4500}]
    summary = "x" * 6100
    plan = prepare_prompt(settings, "提交商品资料需要哪些参数", bundle, history, summary)
    assert plan.messages[2]["content"] == "a" * 4500
    assert summary in plan.messages[-1]["content"]
    assert bundle.text in plan.messages[-1]["content"]
    assert not plan.stats["prompt_budget_applied"]
    ensure_request_fits(settings, plan.messages)


def test_trim_oldest_whole_pairs_first_and_keep_followup_subject(settings):
    bundle = evidence("完整参数表")
    history = [{"role": "user", "content": "提交商品资料 /listing/publish 怎么调用"},
               {"role": "assistant", "content": "早期说明" * 1500},
               {"role": "user", "content": "需要哪些参数"}, {"role": "assistant", "content": "最近回答"}]
    base = prepare_prompt(settings, "哪些是必填的", bundle, history)
    budget = message_tokens(base.messages) - len(history[1]["content"].encode("utf-8")) + 100
    plan = prepare_prompt(with_input_budget(settings, budget), "哪些是必填的", bundle, history)
    assert plan.stats["prompt_history_turns_removed"] == 1
    assert [m["role"] for m in plan.messages] == ["system", "user", "assistant", "user"]
    assert plan.messages[2]["content"] == "最近回答"
    assert "提交商品资料 /listing/publish 怎么调用" in plan.messages[-1]["content"]
    assert history[1]["content"] == "早期说明" * 1500
    assert plan.bundle.sources == bundle.sources


def test_summary_extraction_preserves_whole_sentences_and_source_tables(settings):
    bundle = evidence("| store_id | number | 是 |\n| data | array | 是 |")
    summary = "已确认站点为美国，productType 为 BOOK。" + "早期无关背景。" * 100
    required = message_tokens(build_messages("需要哪些参数", bundle, memory_summary="已确认站点为美国，productType 为 BOOK。"))
    plan = prepare_prompt(with_input_budget(settings, required + 20), "需要哪些参数", bundle, memory_summary=summary)
    assert plan.stats["prompt_memory_reduced"]
    assert "已确认站点为美国，productType 为 BOOK。" in plan.messages[-1]["content"]
    assert bundle.text in plan.messages[-1]["content"]
    assert len(plan.bundle.sources) == 1


def test_drop_low_priority_whole_sources_and_align_citations(settings):
    table = "| 参数 | 类型 | 必填 |\n| --- | --- | --- |\n| store_id | number | 是 |"
    bundle = evidence("无关旧答案" * 800, table, "其他接口说明" * 600)
    bundle.sources[0]["source_type"] = "liked_qa"
    bundle.sources[1]["title"] = "提交商品资料"
    one = ContextBundle(table, [bundle.sources[1]], 0)
    needed = message_tokens(build_messages("提交商品资料需要哪些参数", one))
    plan = prepare_prompt(with_input_budget(settings, needed + 1200), "提交商品资料需要哪些参数", bundle)
    assert [s["id"] for s in plan.bundle.sources] == ["S2"]
    assert table in plan.bundle.text
    assert "[S1]" not in plan.bundle.text and "[S3]" not in plan.bundle.text
    answer, notices = RAGService.check_citations("必填 store_id[S2]，旧证据[S1]", plan.bundle)
    assert "[S2]" in answer and "[S1]" not in answer and notices
    assert [s["id"] for s in bundle.sources] == ["S1", "S2", "S3"]
    assert message_tokens(plan.messages) <= plan.stats["prompt_input_budget"]


def test_required_question_and_atomic_json_are_never_silently_cut(settings):
    schema = json.dumps({"properties": {"long": "属性" * 10000}}, ensure_ascii=False)
    with pytest.raises(ContextBudgetError, match="指定具体接口"):
        prepare_prompt(settings, "列出全部必填属性", evidence(schema))
    with pytest.raises(ContextBudgetError):
        prepare_prompt(settings, "当前问题" * 10000, evidence("固定证据"))


def test_structured_memory_is_removed_atomically_and_retrieval_hint_survives(settings):
    bundle = evidence("完整证据")
    memory = '```json\n{"productType":"BOOK","other":"' + "x" * 15000 + '"}\n```'
    hint = [{"role": "user", "content": "提交商品资料接口 /listing/publish"}]
    baseline = prepare_prompt(settings, "需要哪些参数", bundle, retrieval_history=hint)
    limited = with_input_budget(settings, message_tokens(baseline.messages) + 100)
    plan = prepare_prompt(limited, "需要哪些参数", bundle, memory_summary=memory, retrieval_history=hint)
    assert "提交商品资料接口 /listing/publish" in plan.messages[-1]["content"]
    assert "```json" not in plan.messages[-1]["content"]
    assert plan.stats["prompt_memory_reduced"]


class Response:
    def __init__(self, status=400, data=None, lines=()):
        self.status_code, self.ok = status, status == 200
        self.data, self.lines, self.closed = data, lines, False

    def json(self):
        return self.data

    def iter_lines(self, **kwargs):
        yield from self.lines

    def close(self):
        self.closed = True


@pytest.mark.parametrize("error", [
    {"code": "context_length_exceeded", "message": "private prompt"},
    {"message": "This model's maximum context length is 1000 tokens; request exceeds it. private prompt"},
    {"message": "上下文长度超过限制 private prompt"},
])
def test_http_context_errors_are_typed_closed_and_sanitized(settings, monkeypatch, error):
    response = Response(data={"error": error})
    calls = []
    monkeypatch.setattr("rag.generation.client.requests.post", lambda *a, **k: calls.append(k) or response)
    with pytest.raises(ContextLengthError) as raised:
        list(DeepSeekClient(replace(settings, api_key="test-key")).stream([]))
    assert "private prompt" not in str(raised.value)
    assert len(calls) == 1 and response.closed


def test_other_400_is_not_a_context_error(settings, monkeypatch):
    monkeypatch.setattr("rag.generation.client.requests.post", lambda *a, **k:
                        Response(data={"error": {"message": "Invalid response_format private detail"}}))
    with pytest.raises(LLMError) as raised:
        DeepSeekClient(replace(settings, api_key="test-key")).complete([])
    assert not isinstance(raised.value, ContextLengthError)
    assert "private detail" not in str(raised.value)


@pytest.mark.parametrize("partial", [False, True])
def test_stream_context_error_is_retryable_only_before_any_text(settings, monkeypatch, partial):
    error = 'data: {"error":{"code":"context_length_exceeded","message":"private detail"}}'
    lines = ['data: {"choices":[{"delta":{"content":"部分回答"}}]}', error] if partial else [error]
    response = Response(status=200, lines=lines)
    monkeypatch.setattr("rag.generation.client.requests.post", lambda *a, **k: response)
    with pytest.raises(LLMError) as raised:
        list(DeepSeekClient(replace(settings, api_key="test-key")).stream([]))
    assert isinstance(raised.value, ContextLengthError) is (not partial)
    assert response.closed and "private detail" not in str(raised.value)


def service_with_bundle(settings, monkeypatch, bundle):
    from rag.retrieval.flow import RetrievalFlow
    service = RAGService(replace(settings, api_key="test-key", query_decomposition_enabled=False))
    def run(*args, **kwargs):
        yield {"type": "status", "message": "检索中"}
        return bundle, {"retrieval_intent": "parameters"}
    monkeypatch.setattr(RetrievalFlow, "run", run)
    return service


@pytest.mark.parametrize("second_fails", [False, True])
def test_service_context_retry_is_once_smaller_and_does_not_retrieve_again(settings, monkeypatch, second_fails):
    bundle = evidence("store_id 必填", "a" * 6000, "b" * 6000)
    service = service_with_bundle(settings, monkeypatch, bundle)
    sent, closed = [], []
    def stream(messages, **kwargs):
        sent.append(messages)
        try:
            if len(sent) == 1 or second_fails:
                raise ContextLengthError("上下文超限")
            yield "store_id 必填[S1]"
        finally:
            closed.append(True)
    monkeypatch.setattr(service.llm, "stream", stream)
    if second_fails:
        with pytest.raises(ContextLengthError):
            list(service.events("需要哪些参数"))
    else:
        events = list(service.events("需要哪些参数"))
        done = events[-1]
        assert done["stats"]["context_length_retry_applied"]
        assert done["stats"]["source_count"] == len(done["sources"]) < 3
        assert [event for event in events if event["type"] == "sources"][-1]["sources"] == done["sources"]
        assert "store_id" in done["answer"]
    assert len(sent) == len(closed) == 2
    assert message_tokens(sent[1]) < message_tokens(sent[0])
    for messages in sent:
        ensure_request_fits(settings, messages)


@pytest.mark.parametrize("partial", [False, True])
def test_service_does_not_replay_other_errors_or_partial_answers(settings, monkeypatch, partial):
    service = service_with_bundle(settings, monkeypatch, evidence("完整参数表"))
    sent = []
    def stream(messages, **kwargs):
        sent.append(messages)
        if partial:
            yield "部分回答"
            raise ContextLengthError("超限")
        raise LLMError("网络失败")
    monkeypatch.setattr(service.llm, "stream", stream)
    with pytest.raises(LLMError):
        list(service.events("需要哪些参数"))
    assert len(sent) == 1


def test_cancel_before_retry_prevents_second_request(settings, monkeypatch):
    service = service_with_bundle(settings, monkeypatch, evidence("store_id", "x" * 15000))
    token, sent = CancellationToken(), []
    def stream(messages, **kwargs):
        sent.append(messages)
        token.cancel()
        raise ContextLengthError("超限")
        yield  # generator contract
    monkeypatch.setattr(service.llm, "stream", stream)
    with pytest.raises(GenerationCancelled):
        list(service.events("需要哪些参数", cancel_token=token))
    assert len(sent) == 1


def test_full_service_http_recovery_updates_final_sources_and_payload(settings, monkeypatch):
    service = service_with_bundle(settings, monkeypatch, evidence("store_id 是必填参数", "x" * 6000, "y" * 6000))
    responses = [Response(data={"error": {"code": "context_length_exceeded", "message": "private"}}),
                 Response(status=200, lines=[
                     'data: {"choices":[{"delta":{"content":"store_id 必填[S1]"},"finish_reason":null}]}',
                     'data: {"choices":[{"delta":{},"finish_reason":"stop"}]}', "data: [DONE]"])]
    sent = []
    def post(*args, **kwargs):
        sent.append(kwargs["json"])
        return responses[len(sent) - 1]
    monkeypatch.setattr("rag.generation.client.requests.post", post)
    done = list(service.events("需要哪些参数"))[-1]
    assert len(sent) == 2 and all(response.closed for response in responses)
    assert done["stats"]["context_length_retry_applied"]
    assert done["stats"]["prompt_tokens_estimated"] == message_tokens(sent[-1]["messages"])
    assert done["answer"] == "store_id 必填[S1]"
    for source in done["sources"]:
        assert f"[{source['id']}]" in sent[-1]["messages"][-1]["content"]
    assert done["stats"]["prompt_sources_removed"] > 0


def test_invalid_budget_configuration_is_rejected(settings):
    for changes in ({"llm_context_safety_margin": -1}, {"llm_context_window": 6000},
                    {"max_output_tokens": 0}, {"llm_context_window": True}):
        with pytest.raises(ValueError):
            replace(settings, **changes).validate()
