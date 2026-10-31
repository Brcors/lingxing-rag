"""只在当前进程中暂存刚生成的回答，等待用户明确点赞。"""
from collections import OrderedDict
import secrets
import threading
import time


class FeedbackBuffer:
    def __init__(self, max_items=1000, ttl_seconds=7200):
        self.max_items = max_items
        self.ttl_seconds = ttl_seconds
        self._items = OrderedDict()
        self._lock = threading.Lock()

    def put(self, question, answer, sources, category=None):
        feedback_id = secrets.token_urlsafe(24)
        value = {"question": question, "answer": answer, "sources": sources,
                 "category": category, "created_at": time.time()}
        with self._lock:
            self._expire(time.time())
            self._items[feedback_id] = value
            while len(self._items) > self.max_items:
                self._items.popitem(last=False)
        return feedback_id

    def get(self, feedback_id):
        with self._lock:
            self._expire(time.time())
            value = self._items.get(feedback_id)
            return dict(value) if value else None

    def _expire(self, now):
        expired = [key for key, value in self._items.items()
                   if now - value["created_at"] > self.ttl_seconds]
        for key in expired:
            self._items.pop(key, None)
