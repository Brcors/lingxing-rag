from flask import Flask

from rag.service import RAGService
from rag.settings import Settings
from rag.accounts.repository import AccountRepository, AccountStoreError, ConversationNotFound, ConversationBusy
from rag.accounts.memory import MemoryManager
from rag.accounts.guests import GuestRepository, GuestCleanup
from rag.generation.client import DeepSeekClient


def create_app(settings=None, service=None, accounts=None, memory=None, guests=None, guest_memory=None, start_maintenance=True):
    app = Flask(__name__)
    app.config.update(MAX_CONTENT_LENGTH=128 * 1024, JSON_AS_ASCII=False)
    app.json.ensure_ascii = False
    settings = settings or Settings.load()
    settings.validate()
    app.extensions["rag_settings"] = settings
    app.extensions["rag_service"] = service or RAGService(settings)
    app.extensions["accounts"] = accounts or AccountRepository(settings)
    app.extensions["memory"] = memory or MemoryManager(
        app.extensions["accounts"], getattr(app.extensions["rag_service"], "llm", None) or DeepSeekClient(settings),
        enabled=bool(settings.api_key))
    app.extensions["guests"] = guests or GuestRepository(settings)
    app.extensions["guest_memory"] = guest_memory or MemoryManager(
        app.extensions["guests"], getattr(app.extensions["rag_service"], "llm", None) or DeepSeekClient(settings),
        enabled=bool(settings.api_key))
    app.extensions["guest_cleanup"] = GuestCleanup(app.extensions["guests"])
    from app.auth import configure_auth
    configure_auth(app, settings)
    from app.routes import pages, api
    app.register_blueprint(pages)
    app.register_blueprint(api, url_prefix="/api")
    if start_maintenance:
        app.extensions["guest_cleanup"].start()

    @app.errorhandler(AccountStoreError)
    def account_error(error):
        return {"error": str(error)}, 503

    @app.errorhandler(ConversationNotFound)
    def conversation_missing(error):
        return {"error": str(error)}, 404

    @app.errorhandler(ConversationBusy)
    def conversation_busy(error):
        return {"error": str(error)}, 409

    @app.after_request
    def security_headers(response):
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["Referrer-Policy"] = "same-origin"
        response.headers["Cache-Control"] = "no-store"
        response.headers["Content-Security-Policy"] = (
            "default-src 'self'; script-src 'self'; style-src 'self'; img-src 'self' data:; "
            "connect-src 'self'; base-uri 'self'; frame-ancestors 'none'"
        )
        return response

    @app.errorhandler(413)
    def too_large(_error):
        return {"error": "请求过大，请缩短问题或清空较长的会话。"}, 413

    return app
