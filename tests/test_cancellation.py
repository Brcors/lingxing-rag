import threading

import pytest

from rag.generation.cancellation import CancellationToken, GenerationCancelled, GenerationRegistry
from rag.generation.client import DeepSeekClient
from dataclasses import replace


def test_cancel_before_headers_releases_request_thread(settings, monkeypatch):
    entered, unblock, finished, closed = (threading.Event() for _ in range(4))
    class Response:
        def close(self):
            closed.set()
    def post(*args, **kwargs):
        entered.set()
        unblock.wait(3)
        return Response()
    monkeypatch.setattr("rag.generation.client.requests.post", post)
    token = CancellationToken()
    errors = []
    def run():
        try:
            list(DeepSeekClient(replace(settings, api_key="fake")).stream([], cancel_token=token))
        except Exception as error:
            errors.append(error)
        finally:
            finished.set()
    worker = threading.Thread(target=run)
    worker.start()
    assert entered.wait(1)
    token.cancel()
    assert finished.wait(1)
    unblock.set()
    worker.join(1)
    assert closed.wait(1) and isinstance(errors[0], GenerationCancelled)


def test_cancel_interrupts_blocked_stream_read(settings, monkeypatch):
    entered, unblock, closed, finished = (threading.Event() for _ in range(4))
    class Response:
        status_code = 200
        ok = True
        def iter_lines(self, **kwargs):
            yield 'data: {"choices":[{"delta":{"content":"片段"}}]}'
            entered.set()
            unblock.wait(3)
        def close(self):
            closed.set()
            unblock.set()
    monkeypatch.setattr("rag.generation.client.requests.post", lambda *a, **k: Response())
    token = CancellationToken()
    errors = []
    def run():
        try:
            list(DeepSeekClient(replace(settings, api_key="fake")).stream([], cancel_token=token))
        except Exception as error:
            errors.append(error)
        finally:
            finished.set()
    worker = threading.Thread(target=run)
    worker.start()
    assert entered.wait(1)
    token.cancel()
    assert finished.wait(1) and closed.wait(1)
    worker.join(1)
    assert isinstance(errors[0], GenerationCancelled)


def test_complete_uses_json_protocol_even_when_transport_is_cancellable(settings, monkeypatch):
    observed = []
    class Response:
        status_code = 200
        ok = True
        def json(self):
            return {"choices": [{"message": {"content": "完成"}, "finish_reason": "stop"}]}
        def close(self):
            pass
    def post(*args, **kwargs):
        observed.append(kwargs)
        return Response()
    monkeypatch.setattr("rag.generation.client.requests.post", post)
    result = DeepSeekClient(replace(settings, api_key="fake")).complete([], cancel_token=CancellationToken())
    assert result == "完成" and observed[0]["json"]["stream"] is False
    assert observed[0]["stream"] is True


def test_early_cancel_and_owner_isolation():
    registry = GenerationRegistry()
    a = ("account", "1", "conversation", "request")
    b = ("guest", "1", "conversation", "request")
    registry.cancel(a)
    token = registry.start(a)
    with pytest.raises(GenerationCancelled):
        token.check()
    registry.start(b).check()
    registry.finish(a, token)
    registry.start(a).check()


def test_failed_lease_release_can_be_retried():
    token = CancellationToken()
    attempts = []
    def release():
        attempts.append(True)
        if len(attempts) == 1:
            raise RuntimeError("temporary database failure")
    token.bind_lease(release)
    with pytest.raises(RuntimeError):
        token.cancel()
    token.release_lease()
    assert len(attempts) == 2
