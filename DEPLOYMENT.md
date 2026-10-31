# 本机部署

项目目录：`D:\cursor\yo_project\领星RAG`。访问地址：http://127.0.0.1:5000 。

双击 `start.bat` 启动。入口会检查依赖、启动本项目的 MySQL、检查文档索引，再启动 Waitress 服务。支持 `start.bat --no-browser`，或 `start.bat --port 5001` 更换 Web 端口。

双击 `stop.bat` 可停止 Web 服务并正常关闭数据库。若使用其他 Web 端口，例如 5001，执行 `stop.bat -Port 5001`。停止后再次双击 `start.bat` 即可重启。

## 已准备的组件

- Python 3.13 独立环境：`.venv/`，使用 CPU 版 PyTorch。
- 文档：`领星接口API文档/`，来源为 `lpj2026/one2` 的 GitHub 文档快照。
- 向量模型：`models/bge-small-zh-v1.5/`，`BAAI/bge-small-zh-v1.5`。
- 重排序模型：`models/reranker/Xenova_bge-reranker-base_060632277f/`，ONNX INT8。
- 向量数据库：ChromaDB，数据在 `data/chroma/`，无需单独安装服务。
- 账户和聊天数据库：MySQL Community Server 8.4.13，程序在 `runtime/mysql-8.4.13-winx64/`，数据在 `data/mysql-db/`，仅监听 `127.0.0.1:3306`。

模型权重已与 Hugging Face 固定 revision 的 SHA-256 对照校验。MySQL ZIP 包已与官方 MD5 对照校验。模型齐全后，本地向量与重排序均可离线运行。

数据库使用专用账户 `lingxing_rag`，仅有 `lingxing_rag` 数据库权限。应用密码和管理员密码随机生成，保存在已被 Git 忽略的 `data/local-mysql.json`；`config_local.py` 读取本机配置。正常使用无需输入数据库密码。

## 生成回答的配置

当前 key 保存在已被 Git 忽略的 `config_local.py` 的 `DEEPSEEK_API_KEY` 中。需要更换时编辑此文件，停止后重新启动 Web 服务。

`start.bat` 检测到本机文件已明确配置 key 时，会在本项目进程内忽略继承的 `DEEPSEEK_API_KEY` 环境变量，避免其他环境的旧 key 覆盖本机配置；系统环境变量本身不变。直接运行 `python run.py` 时仍按原规则优先使用非空环境变量。

未使用 `config_local.py` 时，可以在 `config.py` 填写 key。所有 key 留空时可以使用检索预览；页面也可直接选择“仅检索”。

## 管理命令

在项目目录执行：

```powershell
.\.venv\Scripts\python.exe -X utf8 manage.py check
.\.venv\Scripts\python.exe -X utf8 manage.py index
.\.venv\Scripts\python.exe -X utf8 manage.py search "如何查询 FBA 货件列表？"
.\.venv\Scripts\python.exe -X utf8 local_mysql.py start
.\.venv\Scripts\python.exe -X utf8 local_mysql.py stop
```

`local_mysql.py stop` 会正常关闭本项目数据库。停止数据库前先关闭 Web 服务。MySQL 无需系统管理员权限，也未注册为系统服务；电脑重启后再次运行 `start.bat` 即可。

MySQL 日志：`data/mysql-process.log`、`data/mysql-error.log`。后台 Web 服务的日志为 `data/server.stdout.log`、`data/server.stderr.log`。

备份时保存接口文档、`knowledge/`、`data/` 和 `config_local.py`。数据库目录需在 MySQL 正常停止后复制。`data/auth-secret.key` 用于会话签名，也应保留。原电脑的不可用 Python 环境保存在 `.venv.previous-machine-20261030/`。

MySQL 的 Windows 配置文件 `data/mysql.ini` 使用本机系统编码，以兼容中文路径。迁移目录或电脑后，需要更新其中的绝对路径；Python 模型与文档路径随项目目录变化。

官方下载来源：[MySQL Community Server](https://dev.mysql.com/downloads/mysql/8.4.html?os=3)、[BGE 向量模型](https://huggingface.co/BAAI/bge-small-zh-v1.5)、[BGE 重排序模型](https://huggingface.co/Xenova/bge-reranker-base)。
