from dataclasses import replace
import json

import pytest

from conftest import write_api
from rag.generation.cancellation import CancellationToken, GenerationCancelled
from rag.indexing.builder import IndexBuilder
from rag.indexing.store import atomic_json
from rag.retrieval.react import BoundedRetriever, classify, merge_bundles, observe, version_subject
from rag.retrieval.versions import version_evidence
from rag.service import RAGService
from rag.types import ContextBundle


def evidence(text="shipment_id 货件编号 必填", heading="请求参数", path="/erp/test/shipments", doc="a"):
    source = {"id": "S1", "document_id": doc, "heading": heading, "api_path": path, "method": "POST",
              "title": "查询货件列表", "category": "FBA", "relative_path": "FBA/查询货件列表.md",
              "excerpt": text, "source_url": "https://example.com/docs"}
    return ContextBundle("[S1] " + text, [source], 50)


class FakeLLM:
    def __init__(self, result='{"action":"search","query":"shipment_id 请求参数 必填"}'):
        self.result, self.calls = result, []

    def complete(self, messages, **kwargs):
        self.calls.append((messages, kwargs))
        if isinstance(self.result, Exception):
            raise self.result
        return self.result


class FakeService:
    def __init__(self, settings, responses, llm=None):
        self.settings, self.responses, self.calls = settings, list(responses), []
        self.llm = llm or FakeLLM()

    def retrieve(self, question, category=None, history=None, **kwargs):
        self.calls.append((question, category, history, kwargs))
        value = self.responses.pop(0)
        if isinstance(value, Exception):
            raise value
        return value, {"rerank_applied": True, "retrieved_chunks": len(value.sources), "rerank_seconds": .1}


def drain(iterator, on_event=None):
    events = []
    while True:
        try:
            event = next(iterator)
            events.append(event)
            if on_event:
                on_event(event)
        except StopIteration as result:
            return result.value, events


@pytest.mark.parametrize("question,intent", [
    ("查询货件列表需要哪些参数", "parameters"),
    ("response_type 参数必填吗", "parameters"),
    ("这个接口返回哪些字段", "response"),
    ("/erp/test/shipments 使用什么请求方式", "endpoint"),
    ("先创建再确认的调用流程", "workflow"),
    ("文档最后一次更新是什么时候", "version"),
    ("这个接口有更新吗", "version"),
    ("更新库存接口需要什么参数", "parameters"),
    ("update_time 更新时间参数是什么类型", "parameters"),
])
def test_local_question_routing(question, intent):
    assert classify(question) == intent


def test_sufficient_simple_question_uses_one_tool_and_no_planner(settings):
    service = FakeService(replace(settings, api_key="fake"), [evidence()])
    (bundle, stats), _ = drain(BoundedRetriever(service).run("shipment_id 参数必填吗", "FBA"))
    assert bundle.sources and stats["react_tool_calls"] == 1
    assert service.calls[0][1] == "FBA" and service.calls[0][3]["intent"] == "parameters"
    assert not service.llm.calls and not stats["react_evidence_incomplete"]


def test_one_grounded_followup_and_hard_tool_limit(settings):
    service = FakeService(replace(settings, api_key="fake"), [evidence("概览", "接口信息"), evidence()])
    (bundle, stats), events = drain(BoundedRetriever(service).run("shipment_id 参数必填吗", "FBA"))
    assert len(service.calls) == 2 and stats["react_retried"] and stats["react_planner_calls"] == 1
    assert len(service.llm.calls) == 1 and service.llm.calls[0][1]["max_attempts"] == 1
    assert service.llm.calls[0][1]["timeout"] == 15
    assert all(call[1] == "FBA" for call in service.calls)
    assert not stats["react_evidence_incomplete"] and "shipment_id" in bundle.text
    assert len({s["id"] for s in bundle.sources}) == len(bundle.sources)
    assert any("最后一次" in e["message"] for e in events)


@pytest.mark.parametrize("result", ["invalid JSON", '{"action":"search","query":"/invented/api token_new"}', RuntimeError("offline")])
def test_invalid_or_failed_planner_uses_local_fallback_without_another_model_call(settings, result):
    service = FakeService(replace(settings, api_key="fake"), [evidence("概览", "接口信息"), evidence()], FakeLLM(result))
    (_, stats), _ = drain(BoundedRetriever(service).run("shipment_id 参数必填吗"))
    assert len(service.calls) == 2 and len(service.llm.calls) == 1
    assert "shipment_id" in service.calls[-1][0] and "/invented" not in service.calls[-1][0]
    assert stats["react_tool_calls"] == 2


def test_planner_can_stop_and_evidence_gap_is_reported(settings):
    service = FakeService(replace(settings, api_key="fake"), [evidence("概览", "接口信息")], FakeLLM('{"action":"finish"}'))
    (bundle, stats), _ = drain(BoundedRetriever(service).run("参数是什么"))
    assert len(service.calls) == 1 and stats["react_tool_calls"] == 1
    assert stats["react_evidence_incomplete"] and any("证据仍可能不足" in note for note in bundle.notices)


@pytest.mark.parametrize("retrieve_only,api_key", [(True, "fake"), (False, "")])
def test_preview_or_missing_key_never_uses_planner(settings, retrieve_only, api_key):
    service = FakeService(replace(settings, api_key=api_key), [evidence("概览", "接口信息"), evidence()])
    (_, stats), _ = drain(BoundedRetriever(service).run("shipment_id 参数必填吗", retrieve_only=retrieve_only))
    assert len(service.calls) == 2 and stats["react_planner_calls"] == 0
    assert not service.llm.calls


def test_single_call_configuration_stops_before_planner(settings):
    service = FakeService(replace(settings, api_key="fake", react_max_tool_calls=1), [ContextBundle("", [], 0)])
    (_, stats), _ = drain(BoundedRetriever(service).run("参数是什么"))
    assert stats["react_tool_calls"] == 1 and not service.llm.calls


def test_other_endpoint_parameter_cannot_fill_the_requested_endpoint_gap():
    bundle = evidence("接口概览", "接口信息")
    other = evidence("shipment_id 必填", path="/other/endpoint", doc="b")
    bundle.sources.extend(other.sources)
    assert observe("/erp/test/shipments 的 shipment_id 参数必填吗", "parameters", bundle, {})


def test_failed_second_retrieval_keeps_first_evidence(settings):
    service = FakeService(settings, [evidence("概览", "接口信息"), RuntimeError("index busy")])
    (bundle, stats), _ = drain(BoundedRetriever(service).run("参数是什么"))
    assert bundle.sources[0]["excerpt"] == "概览" and stats["react_tool_calls"] == 2
    assert "补查失败" in " ".join(bundle.notices)


def test_cancel_after_status_prevents_tool_and_planner_calls(settings):
    token = CancellationToken()
    service = FakeService(settings, [])
    with pytest.raises(GenerationCancelled):
        drain(BoundedRetriever(service).run("参数是什么", cancel_token=token), lambda e: token.cancel())
    assert not service.calls and not service.llm.calls


def test_cancel_in_planner_does_not_fall_back_to_another_search(settings):
    token = CancellationToken()
    service = FakeService(replace(settings, api_key="fake"), [evidence("概览", "接口信息")])
    def complete(*args, **kwargs):
        assert kwargs["cancel_token"] is token
        token.cancel()
        token.check()
    service.llm.complete = complete
    with pytest.raises(GenerationCancelled):
        drain(BoundedRetriever(service).run("参数是什么", cancel_token=token))
    assert len(service.calls) == 1


def seed_versions(settings):
    root = settings.data_dir / "document-sync"
    atomic_json(root / "state.json", {
        "source": {"repository": settings.doc_sync_repository, "branch": settings.doc_sync_branch,
                   "subdirectory": settings.doc_sync_subdirectory, "docs_dir": str(settings.docs_dir.resolve())},
        "revision": "a" * 40, "source_time": "2026-06-03T00:00:00Z", "checked_at": "2026-11-01T00:00:00Z",
        "index_pending": True, "files": {"FBA/查询货件列表.md": {"source_revision": "b" * 40,
            "updated_at": "2026-05-01T00:00:00Z"}}})
    atomic_json(root / "history.json", [{"revision": "a" * 40, "source_time": "2026-06-03T00:00:00Z",
        "added": [], "modified": ["其他/店铺.md"], "deleted": []}])


def test_global_versions_use_local_records_without_vector_or_llm(settings):
    seed_versions(settings)
    service = FakeService(settings, [])
    (bundle, stats), _ = drain(BoundedRetriever(service).run("文档最后一次更新是什么时候"))
    assert version_subject("文档最后一次更新是什么时候") == ""
    assert not service.calls and not service.llm.calls and stats["react_tool_calls"] == 1
    assert bundle.sources[0]["source_type"] == "document_version"
    assert "2026-06-03" in bundle.text and "尚未完成索引更新" in bundle.text


def test_scoped_versions_do_not_assign_repository_date_to_endpoint(settings):
    seed_versions(settings)
    service = FakeService(settings, [evidence()])
    (bundle, stats), _ = drain(BoundedRetriever(service).run("/erp/test/shipments 的版本记录"))
    assert len(service.calls) == 1 and stats["react_tool_calls"] == 2
    assert "2026-05-01" in bundle.text and "不能当作指定接口" in bundle.text
    assert "其他/店铺.md" not in bundle.text


def test_unrelated_document_cannot_supply_endpoint_version(settings):
    seed_versions(settings)
    service = FakeService(settings, [evidence(path="/unrelated/api")])
    (bundle, _), _ = drain(BoundedRetriever(service).run("/missing/api 的版本记录"))
    assert '"file_versions": []' in bundle.text
    assert "2026-05-01" not in bundle.text


def test_version_followup_uses_previous_user_question_to_locate_document(settings):
    seed_versions(settings)
    service = FakeService(settings, [evidence()])
    (bundle, _), _ = drain(BoundedRetriever(service).run("这个接口有更新吗",
        history=[{"role": "user", "content": "查询货件列表需要哪些参数"}]))
    assert "2026-05-01" in bundle.text


def test_missing_or_unrelated_version_state_is_not_evidence(settings):
    assert not version_evidence(settings).sources
    seed_versions(settings)
    other = replace(settings, doc_sync_branch="other")
    assert not version_evidence(other).sources


def test_merge_preserves_atomic_sources_and_remaps_ids_within_budget():
    first, second = evidence("甲" * 100), evidence("乙" * 100, doc="b")
    merged = merge_bundles(first, second, 520)
    assert merged.estimated_tokens <= 520
    assert "乙" * 100 in merged.text
    for source in merged.sources:
        assert f'[{source["id"]}]' in merged.text
        assert len(source["excerpt"]) == 100


def test_real_retrieval_routes_parameter_section_without_paid_planner(settings, embedding, monkeypatch):
    write_api(settings.docs_dir / "FBA/货件.md")
    IndexBuilder(settings, embedding).build()
    service = RAGService(settings)
    service.embedding = embedding
    monkeypatch.setattr(service.decomposer, "decompose", lambda *a, **k: pytest.fail("ReAct 不应叠加旧拆解"))
    result = service.ask("查询货件列表需要哪些参数", retrieve_only=True)
    assert result["stats"]["react_intent"] == "parameters"
    assert result["stats"]["react_tool_calls"] == 1
    assert any("请求参数" in source["heading"] for source in result["sources"])


def test_disabled_react_preserves_original_pipeline(settings, monkeypatch):
    service = RAGService(replace(settings, react_retrieval_enabled=False))
    monkeypatch.setattr(service, "retrieve", lambda *a, **k: (evidence(), {}))
    result = service.ask("参数是什么", retrieve_only=True)
    assert "react_enabled" not in result["stats"] and result["sources"]


def test_dynamic_versions_do_not_enter_shared_feedback_knowledge(settings, monkeypatch):
    seed_versions(settings)
    service = RAGService(replace(settings, api_key="fake", mysql_enabled=True))
    monkeypatch.setattr(service.llm, "stream", lambda *a, **k: iter(["本机记录的来源版本为六月。[S1]"]))
    monkeypatch.setattr(service.feedback_kb, "register_answer", lambda *a, **k: pytest.fail("版本记录不能进入共享点赞库"))
    result = service.ask("文档更新时间是什么")
    assert result["mode"] == "answer" and "feedback_id" not in result


def test_invalid_tool_budget_is_rejected(settings):
    with pytest.raises(ValueError, match="ReAct"):
        replace(settings, react_max_tool_calls=3).validate()
