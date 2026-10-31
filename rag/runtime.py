"""启动入口与运行中服务之间的本机身份校验。"""
import hashlib
from http.client import HTTPConnection, HTTPException
import json
import os

SERVICE_NAME = "lingxing-rag"


def project_identity(project_dir):
    path = os.path.normcase(str(project_dir.resolve()))
    return hashlib.sha256(path.encode("utf-8")).hexdigest()[:20]


def existing_instance(host, port, project_dir):
    connect_host = "127.0.0.1" if host in ("0.0.0.0", "") else host
    connection = HTTPConnection(connect_host, port, timeout=2)
    try:
        connection.request("GET", "/api/status")
        response = connection.getresponse()
        if response.status != 200:
            return False
        payload = response.read(65537)
        if len(payload) > 65536:
            return False
        data = json.loads(payload)
        return (isinstance(data, dict) and data.get("service") == SERVICE_NAME and
                data.get("project_id") == project_identity(project_dir))
    except (OSError, HTTPException, ValueError, UnicodeError):
        return False
    finally:
        connection.close()
