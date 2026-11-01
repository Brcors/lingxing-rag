import json
import logging
import threading
import re
import secrets
from urllib.parse import urlparse

from flask import Blueprint, Response, abort, current_app, g, jsonify, render_template, request, stream_with_context
from filelock import Timeout

from app.rendering import render_markdown
from rag.generation.client import LLMError
from rag.generation.cancellation import GenerationCancelled
from rag.indexing.builder import IndexBuilder
from rag.indexing.store import CorpusStore
from rag.knowledge.mysql_repository import KnowledgeStoreError
from rag.runtime import SERVICE_NAME, project_identity
from rag.retrieval.reranker import OptionalReranker
from rag.accounts.repository import AccountStoreError, ConversationNotFound, ConversationBusy

pages = Blueprint("pages", __name__)
api = Blueprint("api", __name__)
logger = logging.getLogger(__name__)
_job_lock = threading.Lock()
_job = {"running": False, "message": "", "error": None}


def settings():
    return current_app.extensions["rag_settings"]


def service():
    return current_app.extensions["rag_service"]


def read_question():
    if request.headers.get("Origin"):
        origin = urlparse(request.headers["Origin"])
        if origin.netloc != request.host:
            raise ValueError("仅接受当前页面发出的请求。")
    data = request.get_json(silent=True)
    if not isinstance(data, dict):
        raise ValueError("请求体必须是 JSON 对象。")
    question = data.get("question")
    if not isinstance(question, str) or not question.strip():
        raise ValueError("请输入问题。")
    if len(question) > 4000:
        raise ValueError("单个问题请控制在 4000 字符以内。")
    category = data.get("category")
    if category is not None and (not isinstance(category, str) or len(category) > 100):
        raise ValueError("业务分类参数不合法。")
    category = category.strip() or None if isinstance(category, str) else None
    mode = data.get("mode", "answer")
    if mode not in ("answer", "retrieve"):
        raise ValueError("mode 只能是 answer 或 retrieve。")
    conversation_id = data.get("conversation_id", "")
    if not isinstance(conversation_id, str) or not re.fullmatch(r"[a-f0-9]{32}", conversation_id):
        raise ValueError("请先新建或选择一个对话。")
    request_id = data.get("request_id") or secrets.token_hex(16)
    if not isinstance(request_id, str) or not re.fullmatch(r"[a-zA-Z0-9-]{20,36}", request_id):
        raise ValueError("提问标识无效。")
    return question.strip(), category, mode == "retrieve", conversation_id, request_id


def accounts():
    return current_app.extensions["guests" if g.user.get("is_guest") else "accounts"]


def memory_manager():
    return current_app.extensions["guest_memory" if g.user.get("is_guest") else "memory"]


def generation_key(conversation_id, request_id):
    return ("guest" if g.user.get("is_guest") else "account", str(g.user["id"]), conversation_id, request_id)


def conversation_events(args, cancel_token=None):
    question, category, retrieve_only, conversation_id, request_id = args
    repo, rag_service = accounts(), service()
    memory = memory_manager()
    is_guest = g.user.get("is_guest", False)
    user_id = g.user["id"]
    if cancel_token:
        cancel_token.check()
    token = repo.acquire(user_id, conversation_id)
    events = None
    try:
        if cancel_token:
            cancel_token.bind_lease(lambda: repo.release(user_id, conversation_id, token))
            cancel_token.check()
        previous = repo.find_request(user_id, conversation_id, request_id)
        if previous:
            yield previous
            return
        context = repo.context(user_id, conversation_id)
        extra = {"cancel_token": cancel_token} if cancel_token else {}
        events = rag_service.events(question, category, context["history"], retrieve_only,
                                        memory_summary=context["summary"], retrieval_hint=context.get("retrieval_hint", ""),
                                        allow_feedback=not is_guest, **extra)
        for event in events:
            if cancel_token:
                cancel_token.check()
            if event["type"] == "done":
                event = {**event, "conversation_id": conversation_id,
                         "memory_version": context["memory_version"]}
                if event.get("mode") == "answer":
                    if is_guest:
                        event.pop("feedback_id", None)
                    else:
                        event["feedback_id"] = event.get("feedback_id") or secrets.token_urlsafe(24)
                    event["request_id"] = request_id
                    repo.save_turn(user_id, conversation_id, token, request_id, question, category, event)
                    memory.schedule(user_id, conversation_id)
                if context["memory_error"]:
                    event["notices"] = [*event.get("notices", []), context["memory_error"]]
            yield event
    finally:
        try:
            if cancel_token:
                cancel_token.cancel()
            else:
                repo.release(user_id, conversation_id, token)
        except AccountStoreError:
            logger.exception("释放对话锁失败，将在租约到期后自动恢复")
        finally:
            if events is not None and hasattr(events, "close"):
                events.close()


def format_event(event):
    if event["type"] == "done":
        event = {**event, "answer_html": render_markdown(event["answer"])}
    return json.dumps(event, ensure_ascii=False) + "\n"


@pages.get("/")
def index():
    return render_template("index.html", user=g.user)


@api.get("conversations")
def list_conversations():
    query = request.args.get("q", "").strip()
    if len(query) > 200:
        return jsonify(error="搜索关键词请控制在 200 字符以内。"), 400
    return jsonify(conversations=accounts().list_conversations(g.user["id"], query))


@api.post("conversations")
def create_conversation():
    return jsonify(accounts().create_conversation(g.user["id"])), 201


@api.patch("conversations/<conversation_id>")
def rename_conversation(conversation_id):
    data = request.get_json(silent=True)
    try:
        return jsonify(accounts().rename_conversation(g.user["id"], conversation_id,
                                                     data.get("title") if isinstance(data, dict) else None))
    except ValueError as error:
        return jsonify(error=str(error)), 400


@api.delete("conversations/<conversation_id>")
def delete_conversation(conversation_id):
    replacement = accounts().delete_conversation(g.user["id"], conversation_id)
    return jsonify(ok=True, replacement=replacement)


@api.get("conversations/<conversation_id>/export")
def export_conversation(conversation_id):
    content = accounts().export_markdown(g.user["id"], conversation_id)
    return Response(stream_with_context(content), content_type="text/markdown; charset=utf-8",
                    headers={"Content-Disposition": f'attachment; filename="conversation-{conversation_id}.md"'})


@api.post("conversations/<conversation_id>/cancel")
def cancel_generation(conversation_id):
    data = request.get_json(silent=True)
    request_id = data.get("request_id") if isinstance(data, dict) else None
    if not isinstance(request_id, str) or not re.fullmatch(r"[a-zA-Z0-9-]{20,36}", request_id):
        return jsonify(error="提问标识无效。"), 400
    accounts().assert_owned(g.user["id"], conversation_id)
    if accounts().find_request(g.user["id"], conversation_id, request_id):
        return jsonify(cancelled=False, already_completed=True)
    active = current_app.extensions["generations"].cancel(generation_key(conversation_id, request_id))
    return jsonify(cancelled=True, was_running=active)


@api.get("conversations/<conversation_id>")
def get_conversation(conversation_id):
    before = request.args.get("before", type=int)
    result = accounts().read_conversation(g.user["id"], conversation_id, before)
    for turn in result["turns"]:
        turn["response"]["answer_html"] = render_markdown(turn["answer"])
    memory_manager().schedule(g.user["id"], conversation_id)
    return jsonify(result)


@pages.get("/documents/<document_id>")
def document(document_id):
    try:
        document = CorpusStore(settings()).get_document(document_id)
    except FileNotFoundError:
        abort(404)
    if document is None:
        abort(404)
    return render_template("document.html", document=document, content_html=render_markdown(document.content))


@api.get("status")
def status():
    active = CorpusStore(settings()).active()
    return jsonify({
        "service": SERVICE_NAME, "project_id": project_identity(settings().project_dir),
        "index_ready": bool(active), "llm_configured": bool(settings().api_key),
        "model": settings().llm_model, "embedding_model": settings().embedding_model,
        "rerank_enabled": settings().rerank_enabled,
        "rerank_model": settings().rerank_model,
        "rerank_backend": settings().rerank_backend,
        "rerank_model_downloaded": OptionalReranker(settings()).downloaded(),
        "mysql_knowledge_enabled": settings().mysql_enabled,
        "document_count": active["document_count"] if active else 0,
        "chunk_count": active["chunk_count"] if active else 0,
        "categories": sorted({c.split("/")[0] for c in active["categories"]}) if active else [],
        "built_at": active.get("built_at") if active else None,
        "job": dict(_job),
        "document_sync": current_app.extensions["document_sync"].status(),
    })


@api.get("documents/versions")
def document_versions():
    manager = current_app.extensions["document_sync"]
    versions = []
    for item in manager.history():
        record = dict(item)
        record["counts"] = {key: len(item.get(key, [])) for key in ("added", "modified", "deleted")}
        for key in record["counts"]:
            record[key] = item.get(key, [])[:30]
        versions.append(record)
    return jsonify(status=manager.status(), versions=versions)


@api.post("documents/sync")
def sync_documents():
    if g.user.get("is_guest"):
        return jsonify(error="游客不能同步共享文档。"), 403
    manager = current_app.extensions["document_sync"]
    if not settings().doc_sync_enabled:
        return jsonify(error="自动文档同步已关闭，请在本机配置中启用。"), 503
    data = request.get_json(silent=True)
    if not isinstance(data, dict) or not isinstance(data.get("check_only", False), bool):
        return jsonify(error="同步请求格式不合法。"), 400
    if not manager.trigger(check_only=data.get("check_only", False)):
        return jsonify(error="文档同步正在进行。"), 409
    return jsonify(message="已开始检查文档。" if data.get("check_only") else "已开始同步文档并更新索引。"), 202


@api.post("ask")
def ask():
    try:
        result = None
        for event in conversation_events(read_question()):
            if event["type"] == "done":
                result = event
        return jsonify({**result, "answer_html": render_markdown(result["answer"])})
    except (AccountStoreError, ConversationNotFound, ConversationBusy):
        raise
    except (ValueError, FileNotFoundError) as error:
        return jsonify(error=str(error)), 400
    except LLMError as error:
        return jsonify(error=str(error)), 502
    except Exception:
        logger.exception("问答处理失败")
        return jsonify(error="问答处理失败，请查看启动窗口中的错误信息。"), 500


@api.post("ask/stream")
def ask_stream():
    try:
        args = read_question()
    except ValueError as error:
        return jsonify(error=str(error)), 400
    registry = current_app.extensions["generations"]
    key = generation_key(args[3], args[4])
    control = registry.start(key)

    @stream_with_context
    def generate():
        try:
            for event in conversation_events(args, control):
                yield format_event(event)
        except GenerationCancelled:
            yield format_event({"type": "cancelled", "message": "已停止生成，未完成的回答不会保存。"})
        except (AccountStoreError, ConversationNotFound, ConversationBusy) as error:
            yield format_event({"type": "error", "message": str(error)})
        except (ValueError, FileNotFoundError, LLMError) as error:
            yield format_event({"type": "error", "message": str(error)})
        except Exception:
            logger.exception("流式问答处理失败")
            yield format_event({"type": "error", "message": "处理失败，请查看启动窗口中的错误信息后重试。"})
        finally:
            registry.finish(key, control)

    response = Response(generate(), mimetype="application/x-ndjson", headers={"X-Accel-Buffering": "no"})
    response.call_on_close(lambda: registry.finish(key, control))
    return response


@api.post("feedback")
def save_feedback():
    if g.user.get("is_guest"):
        return jsonify(error="游客可以查看共享知识，但不能向共享知识库写入问答。请注册或登录后点赞。"), 403
    origin = request.headers.get("Origin")
    if origin and urlparse(origin).netloc != request.host:
        return jsonify(error="请求来源不合法。"), 403
    data = request.get_json(silent=True)
    feedback_id = data.get("feedback_id") if isinstance(data, dict) else None
    if not isinstance(feedback_id, str) or not 20 <= len(feedback_id) <= 64:
        return jsonify(error="反馈标识无效，请重新提问后再点赞。"), 400
    try:
        if not settings().mysql_enabled:
            return jsonify(error="共享点赞知识库已关闭。"), 503
        record = accounts().liked_record(g.user["id"], feedback_id)
        result = service().feedback_kb.save_record(feedback_id, record)
        accounts().mark_liked(g.user["id"], feedback_id)
        return jsonify(result)
    except (AccountStoreError, ConversationNotFound):
        raise
    except KeyError as error:
        return jsonify(error=str(error)), 410
    except (KnowledgeStoreError, RuntimeError) as error:
        return jsonify(error=str(error)), 503
    except Exception:
        logger.exception("保存点赞反馈失败")
        return jsonify(error="保存失败，请稍后重试。"), 500


@api.post("index")
def update_index():
    if g.user.get("is_guest"):
        return jsonify(error="游客不能更新共享文档索引。"), 403
    if not request.is_json:
        return jsonify(error="请通过页面提交更新请求。"), 400
    origin = request.headers.get("Origin")
    if origin and urlparse(origin).netloc != request.host:
        return jsonify(error="请求来源不合法。"), 403
    with _job_lock:
        if _job["running"]:
            return jsonify(error="文档更新正在进行。"), 409
        _job.update(running=True, message="开始更新文档索引…", error=None)
    app = current_app._get_current_object()
    rag_service = service()

    def work():
        try:
            builder = IndexBuilder(app.extensions["rag_settings"], rag_service.embedding)
            builder.build(progress=lambda text: _job.update(message=text))
            _job.update(message="文档索引已更新。")
        except Timeout:
            _job.update(error="另一个进程正在建立索引，请稍后重试。")
        except Exception:
            logger.exception("文档更新失败")
            _job.update(error="文档更新失败，请查看启动窗口中的错误信息。")
        finally:
            _job["running"] = False

    threading.Thread(target=work, name="index-builder", daemon=True).start()
    return jsonify(message="已开始更新文档索引。"), 202
