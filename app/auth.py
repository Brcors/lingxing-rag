import os
import re
import secrets
import threading
import time
import unicodedata
from datetime import datetime, timedelta, timezone

from argon2 import PasswordHasher
from argon2.exceptions import InvalidHashError, VerificationError
from flask import Blueprint, current_app, g, jsonify, redirect, render_template, request, session
from flask.sessions import SecureCookieSessionInterface
from itsdangerous import BadSignature

from rag.accounts.repository import AccountStoreError, DuplicateUsername

auth = Blueprint("auth", __name__)
hasher = PasswordHasher(time_cost=2, memory_cost=19456, parallelism=1)
_dummy_hash = hasher.hash(secrets.token_urlsafe(32))
_attempts = {}
_attempt_lock = threading.Lock()


def csrf_token():
    if "csrf" not in session:
        session["csrf"] = secrets.token_urlsafe(32)
    return session["csrf"]


class LoginSessionInterface(SecureCookieSessionInterface):
    def open_session(self, app, request):
        result = super().open_session(app, request)
        if result and result.get("user_id"):
            try:
                self.get_signing_serializer(app).loads(request.cookies.get(self.get_cookie_name(app), ""), max_age=8 * 3600)
            except BadSignature:
                return self.session_class()
        return result

    def get_expiration_time(self, app, session):
        if session.permanent:
            hours = 24 if session.get("guest_id") else 8
            return datetime.now(timezone.utc) + timedelta(hours=hours)
        return None


def configure_auth(app, settings):
    # 随机密钥只生成一次；密码和问答均不进入 cookie。
    secret = os.getenv("FLASK_SECRET_KEY", "").strip()
    if not secret:
        path = settings.data_dir / "auth-secret.key"
        path.parent.mkdir(parents=True, exist_ok=True)
        try:
            descriptor = os.open(str(path), os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
        except FileExistsError:
            pass
        else:
            with os.fdopen(descriptor, "w", encoding="ascii") as file:
                file.write(secrets.token_hex(32))
        secret = path.read_text(encoding="ascii").strip()
        if len(secret) < 32:
            raise RuntimeError("登录会话密钥文件无效，请检查 data/auth-secret.key。")
    app.config.update(SECRET_KEY=secret, SESSION_COOKIE_HTTPONLY=True,
                      SESSION_COOKIE_SAMESITE="Lax", SESSION_COOKIE_NAME="lingxing_session",
                      SESSION_COOKIE_SECURE=os.getenv("SESSION_COOKIE_SECURE", "").lower() == "true",
                      PERMANENT_SESSION_LIFETIME=timedelta(hours=24))
    app.session_interface = LoginSessionInterface()
    app.jinja_env.globals["csrf_token"] = csrf_token
    app.register_blueprint(auth)

    @app.before_request
    def protect_routes():
        public = request.endpoint in {"auth.login_page", "auth.register_page", "auth.login", "auth.register", "auth.guest", "api.status", "static"}
        g.user = None
        if not public or request.endpoint in {"auth.login_page", "auth.register_page"}:
            user_id = session.get("user_id")
            if user_id:
                user = app.extensions["accounts"].get_user(user_id=user_id)
                if user:
                    g.user = {"id": user["id"], "username": user["username"], "is_guest": False}
                else:
                    session.clear()
            elif session.get("guest_id"):
                guest_id = session["guest_id"]
                if app.extensions["guests"].touch_guest(guest_id):
                    g.user = {"id": guest_id, "username": "游客", "is_guest": True}
                else:
                    session.clear()
            if not public and not g.user:
                if request.path.startswith("/api/"):
                    return jsonify(error="请先登录。"), 401
                return redirect("/login")
        if request.method in {"POST", "PUT", "PATCH", "DELETE"}:
            from urllib.parse import urlparse
            origin = request.headers.get("Origin")
            if origin and urlparse(origin).netloc != request.host:
                return jsonify(error="请求来源不合法。"), 403
            expected = session.get("csrf", "")
            supplied = request.headers.get("X-CSRF-Token", "")
            if not expected or not secrets.compare_digest(expected, supplied):
                return jsonify(error="页面凭证已过期，请刷新页面后重试。"), 403


def credentials():
    body = request.get_json(silent=True)
    if not isinstance(body, dict):
        raise ValueError("请填写用户名和密码。")
    username, password = body.get("username"), body.get("password")
    if not isinstance(username, str) or not isinstance(password, str):
        raise ValueError("请填写用户名和密码。")
    username = unicodedata.normalize("NFKC", username.strip()).casefold()
    if not re.fullmatch(r"[a-z0-9_\u4e00-\u9fff]{3,32}", username):
        raise ValueError("用户名需为 3–32 位汉字、字母、数字或下划线。")
    if not 8 <= len(password) <= 128:
        raise ValueError("密码需为 8–128 个字符。")
    return username, password


def establish_session(user):
    discard_guest()
    session.clear()
    session["user_id"] = user["id"]
    session.permanent = True
    csrf_token()


def discard_guest():
    if session.get("guest_id"):
        current_app.extensions["guests"].delete_guest(session["guest_id"])


def throttle():
    key = request.remote_addr or "local"
    now = time.monotonic()
    with _attempt_lock:
        for address in list(_attempts):
            _attempts[address] = [stamp for stamp in _attempts[address] if now - stamp < 60]
            if not _attempts[address]:
                del _attempts[address]
        attempts = _attempts.setdefault(key, [])
        if len(attempts) >= 10:
            return False
        attempts.append(now)
        return True


@auth.get("/login")
def login_page():
    return redirect("/") if g.user and not g.user.get("is_guest") else render_template("auth.html", register=False)


@auth.get("/register")
def register_page():
    return redirect("/") if g.user and not g.user.get("is_guest") else render_template("auth.html", register=True)


@auth.post("/api/auth/guest")
def guest():
    if session.get("user_id"):
        return jsonify(error="请先退出当前账户，再以游客进入。"), 400
    repo = current_app.extensions["guests"]
    if session.get("guest_id") and repo.touch_guest(session["guest_id"]):
        return jsonify(ok=True, redirect="/")
    if not throttle():
        return jsonify(error="尝试过于频繁，请一分钟后重试。"), 429
    repo.cleanup_expired()
    guest_id, _conversation = repo.create_guest()
    session.clear()
    session["guest_id"] = guest_id
    session.permanent = True
    csrf_token()
    return jsonify(ok=True, redirect="/"), 201


@auth.post("/api/auth/register")
def register():
    if not throttle():
        return jsonify(error="尝试过于频繁，请一分钟后重试。"), 429
    try:
        username, password = credentials()
        body = request.get_json()
        if body.get("confirm_password") != password:
            raise ValueError("两次输入的密码不一致。")
        user = current_app.extensions["accounts"].create_user(username, hasher.hash(password))
    except (ValueError, DuplicateUsername) as error:
        return jsonify(error=str(error)), 400
    establish_session(user)
    return jsonify(ok=True, redirect="/"), 201


@auth.post("/api/auth/login")
def login():
    if not throttle():
        return jsonify(error="尝试过于频繁，请一分钟后重试。"), 429
    try:
        username, password = credentials()
    except ValueError as error:
        return jsonify(error=str(error)), 400
    user = current_app.extensions["accounts"].get_user(username=username)
    try:
        hasher.verify(user["password_hash"] if user else _dummy_hash, password)
        valid = bool(user)
    except (VerificationError, InvalidHashError):
        valid = False
    if not valid:
        return jsonify(error="用户名或密码错误。"), 401
    establish_session(user)
    return jsonify(ok=True, redirect="/")


@auth.post("/api/auth/logout")
def logout():
    discard_guest()
    session.clear()
    return jsonify(ok=True, redirect="/login")
