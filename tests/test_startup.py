import json
import os
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
import subprocess
import threading

import pytest

from app import create_app
from rag.runtime import SERVICE_NAME, existing_instance, project_identity
from test_web_llm import StubService

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_batch_has_windows_line_endings_and_ascii_commands():
    raw = (PROJECT_ROOT / "start.bat").read_bytes()
    assert raw.isascii()
    assert b"\r\n" in raw
    assert b"\n" not in raw.replace(b"\r\n", b"")
    assert b"run.py %*" in raw


@pytest.mark.skipif(os.name != "nt", reason="Windows BAT 执行回归")
def test_actual_batch_executes_from_another_directory_and_forwards_flags(tmp_path):
    env = dict(os.environ, LINGXING_NO_PAUSE="1")
    result = subprocess.run(["cmd.exe", "/d", "/c", "call", str(PROJECT_ROOT / "start.bat"),
                             "--no-browser", "--help"],
                            cwd=tmp_path, env=env, input=b"\r\n", capture_output=True, timeout=30)
    output = (result.stdout + result.stderr).decode("utf-8", errors="replace")
    assert result.returncode == 0, output
    assert "--no-browser" in output and "--port" in output
    assert "not recognized" not in output


def test_health_check_identifies_only_this_project(settings):
    payload = {"service": SERVICE_NAME, "project_id": project_identity(settings.project_dir)}

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            body = json.dumps(payload).encode()
            self.send_response(200)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *_args):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    worker = threading.Thread(target=server.serve_forever, daemon=True)
    worker.start()
    try:
        assert existing_instance("127.0.0.1", server.server_port, settings.project_dir)
        assert not existing_instance("127.0.0.1", server.server_port, settings.project_dir / "other")
        payload["service"] = "another-app"
        assert not existing_instance("127.0.0.1", server.server_port, settings.project_dir)
    finally:
        server.shutdown()
        server.server_close()
        worker.join(timeout=2)


def _busy_port(monkeypatch, settings, is_ours):
    import run

    class Probe:
        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

        def bind(self, _address):
            raise OSError("port in use")

    monkeypatch.chdir(settings.project_dir)
    monkeypatch.setattr("run.config.PROJECT_DIR", settings.project_dir)
    monkeypatch.setattr("run.Settings.load", lambda: settings)
    monkeypatch.setattr("run.socket.socket", lambda: Probe())
    monkeypatch.setattr("run.existing_instance", lambda *_args: is_ours)
    import sys
    monkeypatch.setattr(sys, "argv", ["run.py", "--no-browser"])
    monkeypatch.setattr("run.create_app", lambda *_args: pytest.fail("重复启动不应重新构建应用或索引"))
    monkeypatch.setattr("run.webbrowser.open", lambda *_args: pytest.fail("--no-browser 不应打开浏览器"))
    return run


def test_repeated_launch_reuses_existing_service(settings, monkeypatch, capsys):
    run = _busy_port(monkeypatch, settings, True)
    assert run.main() is None
    assert "已在运行" in capsys.readouterr().out


def test_occupied_port_from_other_program_has_clear_error(settings, monkeypatch):
    run = _busy_port(monkeypatch, settings, False)
    with pytest.raises(SystemExit, match="其他程序占用"):
        run.main()


def test_missing_browser_handler_does_not_fail_startup(monkeypatch, capsys):
    import run
    monkeypatch.setattr("run.webbrowser.open", lambda *_args: False)
    run.open_page("http://127.0.0.1:5000")
    assert "手动访问" in capsys.readouterr().out


def test_status_exposes_project_identity_without_file_paths(settings):
    result = create_app(settings, StubService(), start_maintenance=False).test_client().get("/api/status").json
    assert result["service"] == SERVICE_NAME
    assert result["project_id"] == project_identity(settings.project_dir)
    assert str(settings.project_dir) not in json.dumps(result)
