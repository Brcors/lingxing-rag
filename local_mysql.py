"""启动或停止本项目的便携 MySQL，不需要安装 Windows 系统服务。"""
import argparse
import json
import os
from pathlib import Path
import socket
import subprocess
import time

ROOT = Path(__file__).resolve().parent
MANIFEST = ROOT / "data" / "local-mysql.json"


def metadata():
    return json.loads(MANIFEST.read_text(encoding="utf-8"))


def port_open(info):
    try:
        with socket.create_connection((info["host"], info["port"]), timeout=1):
            return True
    except OSError:
        return False


def connect(info, admin=False):
    import pymysql
    return pymysql.connect(
        host=info["host"], port=info["port"],
        user="root" if admin else info["user"],
        password=info["admin_password"] if admin else info["password"],
        charset="utf8mb4", connect_timeout=3, read_timeout=5, write_timeout=5,
    )


def verify(info, connection):
    with connection.cursor() as cursor:
        # MySQL 的路径变量可能保留 Windows 本地编码，先取原始字节。
        cursor.execute("SELECT HEX(@@datadir)")
        raw_path = bytes.fromhex(cursor.fetchone()[0])
    expected = (ROOT / info["data_dir"]).resolve()
    for encoding in ("utf-8", "mbcs") if os.name == "nt" else ("utf-8",):
        try:
            if Path(raw_path.decode(encoding)).resolve() == expected:
                return
        except UnicodeError:
            pass
    raise RuntimeError("此端口上的 MySQL 不是本项目的数据库。")


def ensure_started(settings=None):
    if not MANIFEST.exists():
        return
    info = metadata()
    if settings and (not settings.mysql_enabled or
                     (settings.mysql_host, settings.mysql_port, settings.mysql_user) !=
                     (info["host"], info["port"], info["user"])):
        return
    if port_open(info):
        with connect(info) as connection:
            verify(info, connection)
        return
    # MySQL 的 Windows 参数解析器在 argv[0] 包含中文时会丢失参数。
    # 传入相对的 ASCII 程序名，cwd 仍为真实项目目录。
    executable = Path(info["binary_dir"]) / "bin" / "mysqld.exe"
    log_path = ROOT / "data" / "mysql-process.log"
    flags = (subprocess.CREATE_NO_WINDOW | subprocess.CREATE_NEW_PROCESS_GROUP) if os.name == "nt" else 0
    with log_path.open("ab") as log:
        process = subprocess.Popen(
            [str(executable), "--defaults-file=data/mysql.ini", "--no-monitor", "--console"],
            cwd=ROOT, stdin=subprocess.DEVNULL, stdout=log, stderr=log,
            creationflags=flags,
        )
    (ROOT / "data" / "mysql-process.pid").write_text(str(process.pid), encoding="ascii")
    deadline = time.monotonic() + 45
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise RuntimeError(f"本地 MySQL 启动失败，请查看 {log_path}")
        if port_open(info):
            with connect(info) as connection:
                verify(info, connection)
            print(f"本地 MySQL 已就绪：{info['host']}:{info['port']}", flush=True)
            return
        time.sleep(0.5)
    raise RuntimeError(f"本地 MySQL 启动超时，请查看 {log_path}")


def stop():
    if not MANIFEST.exists():
        return
    info = metadata()
    if not port_open(info):
        print("本地 MySQL 已停止。")
        return
    with connect(info, admin=True) as connection:
        verify(info, connection)
        with connection.cursor() as cursor:
            cursor.execute("SHUTDOWN")
    deadline = time.monotonic() + 30
    while time.monotonic() < deadline:
        if not port_open(info):
            print("本地 MySQL 已停止。")
            return
        time.sleep(0.5)
    raise RuntimeError("等待 MySQL 停止超时。")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("start", "stop"), default="start", nargs="?")
    args = parser.parse_args()
    ensure_started() if args.command == "start" else stop()
