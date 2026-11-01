import argparse
import logging
import os
import socket
import threading
import webbrowser

import config
from app import create_app
from rag.settings import Settings
from rag.runtime import existing_instance


def open_page(url):
    try:
        opened = webbrowser.open(url)
    except webbrowser.Error:
        opened = False
    if not opened:
        print(f"未能自动打开浏览器，请手动访问：{url}", flush=True)


def main():
    os.chdir(config.PROJECT_DIR)
    parser = argparse.ArgumentParser(description="启动领星接口文档问答")
    parser.add_argument("--no-browser", action="store_true")
    parser.add_argument("--port", type=int, default=config.PORT)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    settings = Settings.load()
    settings.validate()
    url_host = "127.0.0.1" if config.HOST in ("0.0.0.0", "") else config.HOST
    url = f"http://{url_host}:{args.port}"
    # 提前检查端口，避免建立索引后才发现服务无法启动。
    with socket.socket() as probe:
        try:
            probe.bind((config.HOST, args.port))
        except OSError:
            if existing_instance(config.HOST, args.port, settings.project_dir):
                print(f"领星文档问答已在运行：{url}", flush=True)
                if config.OPEN_BROWSER and not args.no_browser:
                    open_page(url)
                return
            raise SystemExit(f"端口 {args.port} 被其他程序占用，请使用 start.bat --port 5001 更换端口。") from None
    from local_mysql import ensure_started
    ensure_started(settings)
    app = create_app(settings)
    if config.AUTO_BUILD_INDEX:
        from rag.indexing.builder import IndexBuilder
        IndexBuilder(settings, app.extensions["rag_service"].embedding).build(progress=lambda text: print(text, flush=True))
    if (settings.data_dir / "active.json").exists():
        print("正在准备本地检索服务…", flush=True)
        app.extensions["rag_service"].embedding.encode_query("领星接口")
        app.extensions["rag_service"].retriever.reload()
        if settings.rerank_enabled:
            print(f"正在准备本地重排序模型：{settings.rerank_model}…", flush=True)
            try:
                app.extensions["rag_service"].retriever.reranker.warmup()
            except Exception:
                logging.exception("重排序模型预热失败；检索时会重试，失败则回退混合检索顺序。")
    print(f"\n领星文档问答已就绪：{url}", flush=True)
    app.extensions["document_sync"].start()
    if not settings.api_key:
        print("尚未填写 API key；可先在页面使用检索预览。填写 config.py 后重启即可生成回答。", flush=True)
    if config.OPEN_BROWSER and not args.no_browser:
        timer = threading.Timer(1.5, lambda: open_page(url))
        timer.daemon = True
        timer.start()
    from waitress import serve
    serve(app, host=config.HOST, port=args.port, threads=4, channel_timeout=300)


if __name__ == "__main__":
    main()
