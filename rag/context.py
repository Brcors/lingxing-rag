"""在命中片段周围补充接口信息，预算不足时缩小扩展，绝不静默切断表格。"""
import re

from rag.types import ContextBundle
from rag.citations import transform_prose_citations


def estimated_llm_tokens(text):
    # DeepSeek 分词器与 BGE 不同；这里按中文 1.5、其他字符 / 3 保守估算。
    cjk = len(re.findall(r"[\u3400-\u9fff]", text))
    return int(cjk * 1.5 + (len(text) - cjk) / 3) + 1


class ContextBuilder:
    def __init__(self, settings):
        self.settings = settings

    def build(self, question, hits, documents, chunks, reserved_tokens=0):
        context_budget = max(512, self.settings.context_budget - max(0, reserved_tokens))
        sources = []
        notices = ["回答依据本地文档快照；引用页面可能存在后续更新。"]
        candidates = list(hits)
        # 调用代码/鉴权问题补充已核验的官方迁移说明，覆盖旧快照中的过渡期示例。
        verified = [d for d in documents.values() if d.verified_update]
        require_auth = any(word in question.lower() for word in ("代码", "调用", "curl", "签名", "token", "鉴权", "认证"))
        extras = []
        if require_auth:
            for document in verified[:1]:
                text = "\n\n".join(s.text for s in document.sections)
                extras.append((document, "官方接入更新", text, [], [text], ""))
        units = extras
        complete_documents = {item[0].document_id for item in extras}
        groups = {}
        for hit in candidates:
            groups.setdefault(hit.chunk.parent_id, []).append(hit)
        for group in groups.values():
            hit = group[0]
            if hit.chunk.document_id in complete_documents:
                continue
            document = documents[hit.chunk.document_id]
            section = next(s for s in document.sections if s.section_id == hit.chunk.parent_id)
            # 小章节整体补齐；过长章节使用命中块和同节邻块。
            if estimated_llm_tokens(section.text) <= 1100:
                evidence = section.text
            else:
                siblings = sorted([c for c in chunks.values() if c.parent_id == section.section_id], key=lambda c: c.order)
                # 同一个长章节中可能命中多个相距很远的字段，先保留全部命中，
                # 再补邻块，避免只扩展第一个命中而丢掉其余问题依据。
                selected = {h.chunk.chunk_id: h.chunk for h in group}
                for matched in group:
                    position = next(i for i, c in enumerate(siblings) if c.chunk_id == matched.chunk.chunk_id)
                    for neighbor in siblings[max(0, position - 1):position + 2]:
                        if neighbor.chunk_id in selected:
                            continue
                        trial = "\n\n".join(c.text for c in [*selected.values(), neighbor])
                        if estimated_llm_tokens(trial) <= 1400:
                            selected[neighbor.chunk_id] = neighbor
                evidence = "\n\n".join(c.text for c in sorted(selected.values(), key=lambda c: c.order))
            overview = next((s.text for s in document.sections if s.heading == "接口概览"), "")
            if overview and estimated_llm_tokens(overview) < 220 and section.heading != "接口概览":
                evidence = "接口概览：\n" + overview + "\n\n" + evidence
            core = list(dict.fromkeys(h.chunk.text for h in group))
            channels = list(dict.fromkeys(channel for h in group for channel in h.channels))
            units.append((document, section.heading, evidence, channels, core, overview))

        packed = []

        def payload(items):
            return "\n\n---\n\n".join(item["header"] + item["evidence"] for item in items)

        def fits(items):
            return estimated_llm_tokens(payload(items)) <= context_budget

        # 先给每个命中章节保留一块最小证据，避免先扩展的父章节挤掉后续命中。
        for document, heading, expanded, channels, core, overview in units:
            source_id = f"S{len(packed) + 1}"
            header = (f"[{source_id}] {document.title}\n分类：{document.category}\n"
                      f"接口：{document.method} {document.api_path}\n章节：{heading}\n"
                      f"来源：{document.source_url or document.relative_path}\n")
            if document.verified_update:
                header += "来源属性：已核验的官方更新说明；发生冲突时以此说明为准。\n"
            item = {"document": document, "heading": heading, "id": source_id,
                    "header": header, "evidence": core[0], "core": core,
                    "expanded": expanded, "overview": overview, "channels": channels}
            if fits([*packed, item]):
                packed.append(item)

        # 然后补同章的其他命中，最后才用剩余预算扩展父级或相邻片段。
        for item in packed:
            for piece in item["core"][1:]:
                previous = item["evidence"]
                item["evidence"] = previous + "\n\n" + piece
                if not fits(packed):
                    item["evidence"] = previous
        for item in packed:
            previous = item["evidence"]
            item["evidence"] = item["expanded"]
            if not fits(packed):
                item["evidence"] = previous
                overview = item["overview"]
                if overview and estimated_llm_tokens(overview) < 220 and item["heading"] != "接口概览":
                    item["evidence"] = "接口概览：\n" + overview + "\n\n" + previous
                    if not fits(packed):
                        item["evidence"] = previous

        for item in packed:
            document = item["document"]
            sources.append({
                "id": item["id"], "document_id": document.document_id, "title": document.title,
                "category": document.category, "api_path": document.api_path,
                "method": document.method, "heading": item["heading"], "source_url": document.source_url,
                "relative_path": document.relative_path, "excerpt": item["evidence"], "channels": item["channels"],
                "verified_update": document.verified_update,
                "source_revision": document.source_revision, "updated_at": document.updated_at,
                "synced_at": document.synced_at, "update_basis": document.update_basis,
            })
        if len(packed) < len(units):
            notices.append("部分候选证据未能放入本次上下文预算，可缩小问题范围后继续查询。")
        text = payload(packed)
        return ContextBundle(text, sources, estimated_llm_tokens(text) if text else 0, notices)

    def add_feedback_knowledge(self, bundle, records, token_budget):
        for record in records:
            source_id = f"S{len(bundle.sources) + 1}"
            answer = transform_prose_citations(record["answer"], lambda _match: "").strip()
            references = []
            for source in record.get("sources", [])[:5]:
                label = source.get("title") or source.get("relative_path") or "接口文档"
                endpoint = " ".join(x for x in (source.get("method"), source.get("api_path")) if x)
                url = source.get("source_url") or ""
                references.append("- " + " · ".join(x for x in (label, endpoint, url) if x))
            reference_text = "\n".join(references) if references else "原回答未保存可访问的来源链接。"
            header = (f"[{source_id}] 用户点赞问答（辅助参考，非官方接口文档）\n"
                      f"分类：{record.get('category') or '未分类'}\n"
                      f"章节：相似问题与已认可回答\n")
            prefix = (f"相似问题：{record['question']}\n\n"
                      f"已点赞回答（需以本轮官方接口资料核对）：\n")
            suffix = f"\n\n原回答引用的接口文档：\n{reference_text}"
            room = min(max(0, token_budget),
                       self.settings.context_budget - bundle.estimated_tokens)
            if room <= 0:
                break

            def candidate(answer_part, truncated=False):
                ending = "\n\n（已按上下文预算截取）" if truncated else ""
                addition = f"\n\n---\n\n{header}{prefix}{answer_part}{ending}{suffix}"
                return addition, estimated_llm_tokens(bundle.text + addition)

            _, empty_size = candidate("")
            if empty_size > self.settings.context_budget:
                continue
            full_addition, full_size = candidate(answer)
            if full_size <= self.settings.context_budget and full_size - bundle.estimated_tokens <= room:
                answer_part, addition = answer, full_addition
            else:
                low, high, best = 0, len(answer), ""
                while low <= high:
                    mid = (low + high) // 2
                    part = answer[:mid]
                    addition_try, size = candidate(part, truncated=mid < len(answer))
                    if size <= self.settings.context_budget and size - bundle.estimated_tokens <= room:
                        best = part
                        low = mid + 1
                    else:
                        high = mid - 1
                if len(best) < 120:
                    continue
                answer_part = best
                addition, _ = candidate(answer_part, truncated=len(best) < len(answer))

            bundle.text += addition
            bundle.estimated_tokens = estimated_llm_tokens(bundle.text)
            first_reference = next((source for source in record.get("sources", []) if source.get("source_url")), {})
            bundle.sources.append({
                "id": source_id, "document_id": "", "title": "用户点赞问答（辅助参考）",
                "category": record.get("category") or "历史问答", "api_path": "", "method": "",
                "heading": record["question"], "source_url": first_reference.get("source_url", ""),
                "relative_path": f"MySQL liked_answers/{record['id']}",
                "excerpt": prefix + answer_part + suffix, "channels": ["mysql_bm25", "mysql_vector"],
                "verified_update": False, "source_type": "liked_qa",
                "similarity": round(record.get("similarity") or 0.0, 3),
            })
        return bundle
