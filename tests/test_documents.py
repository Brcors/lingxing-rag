from rag.ingestion.parser import MarkdownLoader
from rag.ingestion.chunker import SemanticChunker
from conftest import write_api


def test_parser_keeps_heading_hierarchy_and_does_not_parse_code_heading():
    document = MarkdownLoader.parse("""# 示例接口
介绍。
## 请求
### 嵌套参数
````python
## 代码内的注释
print('hello')
````
## 返回
结果。
""", "销售/示例.md", "销售")
    assert [s.heading for s in document.sections] == ["接口概览", "请求 / 嵌套参数", "返回"]
    assert "## 代码内的注释" in document.sections[1].text


def test_api_metadata_and_table_chunks_cover_all_rows(settings, embedding):
    long_rows = "\n".join(f"| field_{i} | 第{i}个返回字段，包含完整说明 | 否 |" for i in range(80))
    path = settings.docs_dir / "FBA" / "查询货件列表.md"
    write_api(path, extra=long_rows)
    (settings.docs_dir / "index.md").write_text("目录", encoding="utf-8")
    documents = MarkdownLoader(settings).load()
    assert len(documents) == 1
    document = documents[0]
    assert document.api_path == "/erp/test/shipments"
    assert document.method == "POST"
    chunks = SemanticChunker(settings, embedding).build(documents)
    assert all(c.token_count <= settings.chunk_max for c in chunks)
    joined = "\n".join(c.text for c in chunks)
    for i in range(80):
        assert f"| field_{i} |" in joined
    parameter_chunks = [c for c in chunks if "field_" in c.text]
    assert all("| 参数名 |" in c.text for c in parameter_chunks)
    assert all(c.parent_id and c.document_id and c.api_path for c in chunks)


def test_oversized_cell_has_labeled_continuations_and_no_truncation(settings, embedding):
    content = "# 长字段\n## 返回结果\n| 字段 | 说明 |\n| --- | --- |\n| long_field | " + "很长的业务说明。" * 120 + "结束标记 |"
    document = MarkdownLoader.parse(content, "长字段.md", "测试")
    chunks = SemanticChunker(settings, embedding).build([document])
    assert all(c.token_count <= settings.chunk_max for c in chunks)
    assert any("结束标记" in c.text for c in chunks)
    assert any("long_field" in c.text and "长字段" in c.text for c in chunks)


def test_task_category_spaces_do_not_drop_original_source_url(settings):
    import json
    path = settings.docs_dir / "FBA" / "FBA货件（STA）" / "查询货件列表.md"
    write_api(path)
    (settings.docs_dir / "tasks.json").write_text(json.dumps([{
        "category": "FBA / FBA货件（STA）", "title": "查询货件列表",
        "url": "https://apidoc.lingxing.com/#/docs/FBA/FBAShipmentList"
    }]), encoding="utf-8")
    assert MarkdownLoader(settings).load()[0].source_url.endswith("FBAShipmentList")
