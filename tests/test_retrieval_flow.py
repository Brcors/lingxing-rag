from dataclasses import replace

import pytest

from conftest import write_api
from rag.generation.cancellation import CancellationToken, GenerationCancelled
from rag.indexing.builder import IndexBuilder
from rag.indexing.store import atomic_json
from rag.retrieval.flow import RetrievalFlow
from rag.retrieval.hybrid import search_query
from rag.retrieval.routing import classify, merge_bundles, observe, version_subject
from rag.retrieval.versions import version_evidence
from rag.service import RAGService
from rag.types import ContextBundle


def evidence(text="shipment_id 货件编号 必填", heading="请求参数", path="/erp/test/shipments", doc="a", title="查询货件列表"):
    source = {"id": "S1", "document_id": doc, "heading": heading, "api_path": path, "method": "POST",
              "title": title, "category": "FBA", "relative_path": f"FBA/{title}.md",
              "excerpt": text, "source_url": "https://example.com/docs"}
    return ContextBundle("[S1] " + text, [source], 50)


class FakeService:
    def __init__(self, settings, responses):
        self.settings, self.responses, self.calls = settings, list(responses), []

    @property
    def llm(self):
        pytest.fail("检索分流与补查不应访问模型")

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
    ("查询货件列表需要哪些参数", "parameters"), ("response_type 参数必填吗", "parameters"),
    ("这个接口返回哪些字段", "response"), ("/erp/test/shipments 使用什么请求方式", "endpoint"),
    ("先创建再确认的调用流程", "workflow"), ("文档最后一次更新是什么时候", "version"),
    ("这个接口有更新吗", "version"), ("更新库存接口需要什么参数", "parameters"),
    ("update_time 更新时间参数是什么类型", "parameters"),
])
def test_local_question_routing(question, intent):
    assert classify(question) == intent


def test_sufficient_question_uses_one_round(settings):
    service = FakeService(replace(settings, api_key="fake"), [evidence()])
    (bundle, stats), _ = drain(RetrievalFlow(service).run("shipment_id 参数必填吗", "FBA"))
    assert bundle.sources and stats["retrieval_rounds"] == 1
    assert service.calls[0][1] == "FBA" and service.calls[0][3]["intent"] == "parameters"
    assert not stats["retrieval_evidence_incomplete"]


@pytest.mark.parametrize("key", ["", "fake"])
def test_retry_uses_local_terms_and_never_a_model(settings, key):
    service = FakeService(replace(settings, api_key=key), [evidence("概览", "接口信息"), evidence()])
    (bundle, stats), events = drain(RetrievalFlow(service).run("shipment_id 参数必填吗", "FBA"))
    assert len(service.calls) == 2 and stats["retrieval_retry_applied"] and stats["retrieval_rounds"] == 2
    assert all(call[1] == "FBA" for call in service.calls)
    assert "shipment_id" in service.calls[-1][0] and "shipment_id" in bundle.text
    assert not stats["retrieval_evidence_incomplete"]
    assert len({s["id"] for s in bundle.sources}) == len(bundle.sources)
    assert any("最多一次" in e["message"] for e in events)


def test_no_third_round_when_evidence_is_still_missing(settings):
    service = FakeService(settings, [ContextBundle("", [], 0), ContextBundle("", [], 0)])
    (bundle, stats), _ = drain(RetrievalFlow(service).run("参数是什么"))
    assert len(service.calls) == 2 and stats["retrieval_evidence_incomplete"]
    assert any("证据仍可能不足" in note for note in bundle.notices)


def test_disabled_retry_keeps_first_result(settings):
    service = FakeService(replace(settings, retrieval_retry_enabled=False), [ContextBundle("", [], 0)])
    (_, stats), _ = drain(RetrievalFlow(service).run("参数是什么"))
    assert len(service.calls) == 1 and stats["retrieval_rounds"] == 1


def test_failed_retry_keeps_first_evidence(settings):
    service = FakeService(settings, [evidence("概览", "接口信息"), RuntimeError("index busy")])
    (bundle, stats), _ = drain(RetrievalFlow(service).run("参数是什么"))
    assert bundle.sources[0]["excerpt"] == "概览" and stats["retrieval_rounds"] == 2
    assert "补查失败" in " ".join(bundle.notices)


def test_cancel_before_retry_prevents_second_search(settings):
    token = CancellationToken()
    service = FakeService(settings, [evidence("概览", "接口信息")])
    def stop_on_retry(event):
        if "补查" in event["message"]:
            token.cancel()
    with pytest.raises(GenerationCancelled):
        drain(RetrievalFlow(service).run("参数是什么", cancel_token=token), stop_on_retry)
    assert len(service.calls) == 1


def test_other_endpoint_cannot_fill_requested_parameter_gap():
    bundle = evidence("接口概览", "接口信息")
    bundle.sources.extend(evidence("shipment_id 必填", path="/other/endpoint", doc="b").sources)
    assert observe("/erp/test/shipments 的 shipment_id 参数必填吗", "parameters", bundle, {})


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


def test_global_versions_only_read_local_records(settings):
    seed_versions(settings)
    service = FakeService(settings, [])
    (bundle, stats), _ = drain(RetrievalFlow(service).run("文档最后一次更新是什么时候"))
    assert version_subject("文档最后一次更新是什么时候") == ""
    assert not service.calls and len(stats["retrieval_steps"]) == 1
    assert bundle.sources[0]["source_type"] == "document_version"
    assert "2026-06-03" in bundle.text and "尚未完成索引更新" in bundle.text


def test_scoped_versions_do_not_assign_repository_date_to_endpoint(settings):
    seed_versions(settings)
    service = FakeService(settings, [evidence()])
    (bundle, stats), _ = drain(RetrievalFlow(service).run("/erp/test/shipments 的版本记录"))
    assert len(service.calls) == 1 and len(stats["retrieval_steps"]) == 2
    assert "2026-05-01" in bundle.text and "不能当作指定接口" in bundle.text
    assert "其他/店铺.md" not in bundle.text


def test_unrelated_document_cannot_supply_endpoint_version(settings):
    seed_versions(settings)
    service = FakeService(settings, [evidence(path="/unrelated/api")])
    (bundle, _), _ = drain(RetrievalFlow(service).run("/missing/api 的版本记录"))
    assert '"file_versions": []' in bundle.text and "2026-05-01" not in bundle.text


def test_version_followup_uses_previous_question(settings):
    seed_versions(settings)
    service = FakeService(settings, [evidence()])
    (bundle, _), _ = drain(RetrievalFlow(service).run("这个接口有更新吗",
        history=[{"role": "user", "content": "查询货件列表需要哪些参数"}]))
    assert "2026-05-01" in bundle.text


def test_unrelated_version_state_is_not_evidence(settings):
    assert not version_evidence(settings).sources
    seed_versions(settings)
    assert not version_evidence(replace(settings, doc_sync_branch="other")).sources


def test_merge_preserves_atomic_evidence_within_budget():
    merged = merge_bundles(evidence("甲" * 100), evidence("乙" * 100, doc="b"), 520)
    assert merged.estimated_tokens <= 520 and "乙" * 100 in merged.text
    for source in merged.sources:
        assert f'[{source["id"]}]' in merged.text and len(source["excerpt"]) == 100


def test_existing_workflow_rewrite_runs_once_and_retry_does_not_repeat_it(settings, monkeypatch):
    service = RAGService(replace(settings, api_key="fake"))
    rewrites, searches = [], []
    def decompose(question, *args, **kwargs):
        rewrites.append(question)
        return ["创建草稿的接口", "发布商品的接口"], None
    def retrieve(*args, **kwargs):
        searches.append(kwargs)
        return (ContextBundle("", [], 0), {}) if len(searches) == 1 else (evidence(), {})
    monkeypatch.setattr(service.decomposer, "decompose", decompose)
    monkeypatch.setattr(service, "retrieve", retrieve)
    monkeypatch.setattr(service.llm, "complete", lambda *a, **k: pytest.fail("不得增加规划模型调用"))
    monkeypatch.setattr(service.llm, "stream", lambda *a, **k: iter(["文档回答[S1]"]))
    done = service.ask("创建草稿和发布商品的流程")
    assert len(rewrites) == 1 and len(searches) == 2
    assert len(searches[0]["extra_queries"]) == 2 and not searches[1].get("extra_queries")
    assert done["stats"]["retrieval_rounds"] == 2


def test_bare_followup_keeps_subject_across_repeated_short_questions():
    history = [{"role": "user", "content": "刊登怎么做"}, {"role": "user", "content": "需要哪些参数"}]
    assert "刊登怎么做" in search_query("哪些是必填的", history)
    assert "刊登怎么做" in search_query("需要哪些参数", history)
    assert "刊登" not in search_query("查询店铺列表需要哪些参数", history)


def test_listing_followup_clarifies_before_selecting_an_interface(settings, monkeypatch):
    service = RAGService(replace(settings, api_key="fake"))
    bundle = evidence("store_id 参数", path="/listing/product/publish", title="提交商品资料")
    other = evidence("productTypeOrigin 参数", path="/listing/product/schema", doc="b", title="获取商品类型JSON Schema")
    other.sources[0]["id"] = "S2"
    bundle.sources.extend(other.sources)
    searches = []
    def retrieve(*args, **kwargs):
        searches.append(args)
        return bundle, {}
    monkeypatch.setattr(service, "retrieve", retrieve)
    monkeypatch.setattr(service.llm, "complete", lambda *a, **k: pytest.fail("澄清不调用模型"))
    monkeypatch.setattr(service.llm, "stream", lambda *a, **k: pytest.fail("澄清不调用模型"))
    done = service.ask("需要哪些参数", history=[{"role": "user", "content": "刊登怎么做"}])
    assert done["mode"] == "clarification" and done["stats"]["needs_clarification"]
    assert "提交商品资料" in done["answer"] and "获取商品类型" in done["answer"]
    assert len(searches) == 1 and "feedback_id" not in done


def test_concrete_interface_followup_uses_current_parameter_evidence(settings, embedding):
    write_api(settings.docs_dir / "刊登/提交商品资料.md", title="提交商品资料")
    IndexBuilder(settings, embedding).build()
    service = RAGService(settings)
    service.embedding = embedding
    done = service.ask("需要哪些参数", history=[{"role": "user", "content": "提交商品资料怎么做"}], retrieve_only=True)
    assert done["stats"]["query_expanded"] and not done["stats"]["needs_clarification"]
    assert any("请求参数" in source["heading"] for source in done["sources"])


def test_dynamic_versions_are_not_shared_feedback(settings, monkeypatch):
    seed_versions(settings)
    service = RAGService(replace(settings, api_key="fake", mysql_enabled=True))
    monkeypatch.setattr(service.llm, "stream", lambda *a, **k: iter(["本机记录的版本为六月[S1]"]))
    monkeypatch.setattr(service.feedback_kb, "register_answer", lambda *a, **k: pytest.fail("版本记录不能共享"))
    assert "feedback_id" not in service.ask("文档更新时间是什么")
