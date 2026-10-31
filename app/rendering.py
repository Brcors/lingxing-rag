import re
import bleach
from markdown_it import MarkdownIt
from markdown_it.token import Token

MARKDOWN = MarkdownIt("commonmark", {"html": False, "linkify": False}).enable("table")
TAGS = ["p", "br", "strong", "em", "s", "blockquote", "pre", "code", "ul", "ol", "li",
        "h1", "h2", "h3", "h4", "h5", "h6", "hr", "a", "table", "thead", "tbody", "tr", "th", "td"]


def render_markdown(text):
    tokens = MARKDOWN.parse(text)
    for block in tokens:
        if not block.children:
            continue
        children, link_depth = [], 0
        for token in block.children:
            if token.type == "link_open":
                link_depth += 1
            if token.type == "text" and not link_depth:
                position = 0
                for match in re.finditer(r"\[(S\d+)\]", token.content):
                    if match.start() > position:
                        children.append(Token("text", "", 0, content=token.content[position:match.start()]))
                    children.extend([
                        Token("link_open", "a", 1, attrs={"href": "#source-" + match.group(1)}),
                        Token("text", "", 0, content=match.group()),
                        Token("link_close", "a", -1),
                    ])
                    position = match.end()
                if position < len(token.content):
                    children.append(Token("text", "", 0, content=token.content[position:]))
            else:
                children.append(token)
            if token.type == "link_close":
                link_depth -= 1
        block.children = children
    rendered = MARKDOWN.renderer.render(tokens, MARKDOWN.options, {})
    # 本地 Markdown 中的旧站相对链接需还原到官方文档站点。
    rendered = re.sub(r'href="(/?docs/[^\"]+)"',
                      lambda match: 'href="https://apidoc.lingxing.com/#/' + match.group(1).lstrip('/') + '"',
                      rendered)
    return bleach.clean(rendered, tags=TAGS,
                        attributes={"a": ["href", "title"], "code": ["class"], "th": ["align"], "td": ["align"]},
                        protocols=["http", "https"], strip=True)
