"""按批次导出已保存的完整问答，不触发摘要或生成请求。"""
import json
from datetime import datetime, timezone
from urllib.parse import urlparse


def markdown_export(repository, user_id, conversation_id, title, last_id):
    yield f"# {title}\n\n导出时间：{datetime.now(timezone.utc).isoformat()}\n\n"
    after = 0
    while after < last_id:
        with repository.connection() as conn, conn.cursor() as cur:
            repository.owned(cur, user_id, conversation_id)
            cur.execute(f"""SELECT id,question,answer,response_json,created_at FROM {repository.turn_table}
                            WHERE conversation_id=%s AND id>%s AND id<=%s ORDER BY id LIMIT 200""",
                        (conversation_id, after, last_id))
            turns = cur.fetchall()
        if not turns:
            break
        for turn in turns:
            yield f"## 问题\n\n{turn['question']}\n\n### 回答\n\n{turn['answer']}\n\n"
            yield f"保存时间（UTC）：{turn['created_at'].isoformat()}\n\n"
            sources = json.loads(turn["response_json"]).get("sources", [])
            if sources:
                yield "### 参考资料\n\n"
                for source in sources:
                    label = f"[{source.get('id', '')}] {source.get('title', '文档')}"
                    url = source.get("source_url", "")
                    if urlparse(url).scheme in {"http", "https"}:
                        label += f" — {url}"
                    if source.get("api_path"):
                        label += f" · {source.get('method', '')} `{source['api_path']}`"
                    if source.get("updated_at"):
                        label += f" · 资料版本时间：{source['updated_at']}"
                    yield f"- {label}\n"
                yield "\n"
            yield "---\n\n"
        after = turns[-1]["id"]
