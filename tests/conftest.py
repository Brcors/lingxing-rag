from dataclasses import replace
import math

import pytest

from rag.settings import Settings


class CharacterTokenizer:
    def encode(self, text, add_special_tokens=False):
        return list(range(len(text)))

    def __call__(self, text, **kwargs):
        return {"input_ids": self.encode(text), "offset_mapping": [(i, i + 1) for i in range(len(text))]}


class FakeEmbedding:
    tokenizer = CharacterTokenizer()

    def __init__(self):
        self.computed = 0

    def signature(self):
        return "fake-v1"

    def count(self, text):
        return len(text)

    def token_offsets(self, text):
        return self.tokenizer(text, add_special_tokens=False, return_offsets_mapping=True)

    def encode_documents(self, texts):
        self.computed += len(texts)
        return [self._vector(text) for text in texts]

    @staticmethod
    def _vector(text):
        vector = [float(text.count("货件")), float(text.count("店铺")), 1.0]
        norm = math.sqrt(sum(x*x for x in vector))
        return [x / norm for x in vector]

    def encode_query(self, text):
        return self._vector(text)


@pytest.fixture
def settings(tmp_path):
    docs = tmp_path / "docs"
    docs.mkdir()
    return replace(Settings(), project_dir=tmp_path, docs_dir=docs, supplement_dir=tmp_path / "knowledge",
                   data_dir=tmp_path / "data", model_dir=tmp_path / "models", api_key="", rerank_enabled=False,
                   mysql_enabled=False)


@pytest.fixture
def embedding():
    return FakeEmbedding()


def authenticated_client(settings, service):
    """隔离既有 RAG 协议测试；真实认证和 MySQL 所有权由账户集成测试覆盖。"""
    from flask.testing import FlaskClient
    from app import create_app

    class Repository:
        def get_user(self, **kwargs):
            return {"id": 1, "username": "tester"}

        def acquire(self, *args):
            return "lease"

        def release(self, *args):
            pass

        def find_request(self, *args):
            return None

        def context(self, *args):
            return {"history": [], "summary": "", "memory_version": 0, "memory_error": None}

        def save_turn(self, *args):
            pass

    class Memory:
        def schedule(self, *args):
            pass

    class Client(FlaskClient):
        def open(self, *args, **kwargs):
            if kwargs.get("json") is not None:
                kwargs["json"] = {"conversation_id": "a" * 32, **kwargs["json"]}
                kwargs["headers"] = {"X-CSRF-Token": "test-csrf", **kwargs.get("headers", {})}
            return super().open(*args, **kwargs)

    app = create_app(settings, service, accounts=Repository(), memory=Memory(), start_maintenance=False)
    app.test_client_class = Client
    client = app.test_client()
    with client.session_transaction() as session:
        session["user_id"] = 1
        session["csrf"] = "test-csrf"
    return client


def write_api(path, title="查询货件列表", extra=""):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(f"""# {title}
支持查询FBA货件。

## 接口信息
| API Path | 请求协议 | 请求方式 |
| --- | --- | --- |
| `/erp/test/shipments` | HTTPS | POST |

## 请求参数
| 参数名 | 说明 | 必填 |
| --- | --- | --- |
| shipment_id | 货件编号 | 是 |
{extra}

## 返回结果
| 参数名 | 说明 |
| --- | --- |
| data>>list>>shipment_id | 货件编号 |
""", encoding="utf-8")

