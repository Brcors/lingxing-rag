"""单进程请求取消；取消身份包含账户类型、所有者、对话和请求 ID。"""
import logging
import queue
import threading
import time

logger = logging.getLogger(__name__)


def call_cancellable(operation, token, response=False):
    """网络等待在守护线程中进行，让 WSGI 线程及时响应取消。"""
    if token is None:
        return operation()
    token.check()
    result = queue.Queue(maxsize=1)
    def work():
        try:
            value = operation()
            if response:
                token.add_response(value)
            result.put((True, value))
        except Exception as error:
            result.put((False, error))
    threading.Thread(target=work, name="cancellable-api", daemon=True).start()
    while True:
        token.check()
        try:
            success, value = result.get(timeout=0.1)
        except queue.Empty:
            continue
        token.check()
        if not success:
            raise value
        return value


def iter_cancellable(iterator, token):
    if token is None:
        yield from iterator
        return
    values = queue.Queue(maxsize=8)
    stopped = threading.Event()
    def put(kind, value=None):
        while not stopped.is_set() and not token.event.is_set():
            try:
                values.put((kind, value), timeout=0.1)
                return True
            except queue.Full:
                pass
        return False
    def read():
        try:
            for value in iterator:
                if not put("value", value):
                    return
            put("end")
        except Exception as error:
            put("error", error)
    threading.Thread(target=read, name="cancellable-stream", daemon=True).start()
    try:
        while True:
            token.check()
            try:
                kind, value = values.get(timeout=0.1)
            except queue.Empty:
                continue
            token.check()
            if kind == "end":
                return
            if kind == "error":
                raise value
            yield value
    finally:
        stopped.set()


class GenerationCancelled(Exception):
    pass


class CancellationToken:
    def __init__(self):
        self.event = threading.Event()
        self._lock = threading.Lock()
        self._closers = set()
        self._release = None

    def check(self):
        if self.event.is_set():
            raise GenerationCancelled("已停止生成。")

    def bind_lease(self, callback):
        with self._lock:
            self._release = callback
            cancelled = self.event.is_set()
        if cancelled:
            self.release_lease()
            self.check()

    def release_lease(self):
        with self._lock:
            callback, self._release = self._release, None
        if callback:
            try:
                callback()
            except Exception:
                with self._lock:
                    if self._release is None:
                        self._release = callback
                raise

    def add_response(self, response):
        with self._lock:
            self._closers.add(response)
            cancelled = self.event.is_set()
        if cancelled:
            response.close()
            self.check()

    def remove_response(self, response):
        with self._lock:
            self._closers.discard(response)

    def cancel(self):
        self.event.set()
        with self._lock:
            responses = list(self._closers)
        # 某些网络 read 正在持锁，关闭操作不能堵住取消接口。
        for response in responses:
            threading.Thread(target=self._close, args=(response,), daemon=True).start()
        self.release_lease()

    @staticmethod
    def _close(response):
        try:
            response.close()
        except Exception:
            logger.debug("关闭已取消的上游响应失败", exc_info=True)


class GenerationRegistry:
    def __init__(self):
        self._lock = threading.Lock()
        self._items = {}

    def start(self, key):
        with self._lock:
            self._prune()
            existing = self._items.get(key)
            if existing and existing[2]:
                from rag.accounts.repository import ConversationBusy
                raise ConversationBusy("该请求已经在生成。")
            token = existing[0] if existing else CancellationToken()
            self._items[key] = (token, time.monotonic(), True)
            return token

    def cancel(self, key):
        with self._lock:
            self._prune()
            existing = self._items.get(key)
            token = existing[0] if existing else CancellationToken()
            if not existing:
                # 接住取消请求比流式请求更早到达的情况。
                self._items[key] = (token, time.monotonic(), False)
        token.cancel()
        return bool(existing and existing[2])

    def finish(self, key, token):
        with self._lock:
            if self._items.get(key, (None,))[0] is token:
                self._items.pop(key, None)

    def _prune(self):
        now = time.monotonic()
        for key, (_, created, active) in list(self._items.items()):
            if not active and (now - created > 60 or len(self._items) > 512):
                self._items.pop(key, None)
