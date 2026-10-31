# 部署说明

本项目默认在本机运行，监听 `127.0.0.1:5000`。首次安装请先按 [README](README.md#快速开始) 完成依赖、MySQL、本机配置和模型准备。

## 普通安装与已有便携部署

从 GitHub 克隆项目后，需要自行安装并启动 MySQL。仓库不包含 MySQL 程序、数据库文件或现成账户，`start.bat` 也不会为首次克隆自动安装 MySQL。

已有的 Windows 便携部署可以继续使用 `local_mysql.py`：当本机存在 `data/local-mysql.json`、`data/mysql.ini` 和对应 MySQL 程序时，`run.py` 会按配置启动本项目的数据库。没有这份本机清单时，应用使用 `config_local.py` 指定的外部 MySQL。

便携部署的路径及密码仅保存在本机，未包含在开源仓库中。安装 MySQL 可参考 [官方 Windows 安装说明](https://dev.mysql.com/doc/refman/8.4/en/windows-installation.html)。

## 本机配置

```powershell
Copy-Item config_local.example.py config_local.py
```

编辑 `config_local.py`，填写自己的 `DEEPSEEK_API_KEY` 和 `MYSQL_*` 参数。不要覆盖已经配置好的本机文件；更新代码时保留此文件即可。

配置加载规则：

1. `config.py` 提供通用默认值。
2. `config_local.py` 存在时覆盖同名参数。
3. 直接执行 Python 时，非空 `DEEPSEEK_API_KEY` 环境变量优先于文件配置。
4. 使用 `start.bat` 且 `config_local.py` 已明确配置非空 key 时，本项目子进程优先使用本机文件；系统环境变量本身不变。

所有 API key 留空时，Web 页面可使用检索预览；登录和游客模式仍需要 MySQL。数据库开关关闭时，账户与游客功能不可用。

## 模型与文档

```powershell
.\.venv\Scripts\python.exe -X utf8 manage.py download-model
.\.venv\Scripts\python.exe -X utf8 manage.py download-reranker
.\.venv\Scripts\python.exe -X utf8 manage.py index
.\.venv\Scripts\python.exe -X utf8 manage.py check
```

模型默认使用 CPU，缓存在 `models/`。修改 `EMBEDDING_MODEL` 后需要建立新索引；仅更换重排序模型不需要重新计算文档向量。

接口文档目录为项目内的 `领星接口API文档/`。程序忽略 `index.md`、隐藏目录和抓取任务 JSON；`tasks.json` 用于恢复来源 URL。额外的 Markdown 目录可通过 `DOCS_DIR` 配置。

启动及更新索引时会复用未改变的向量。新一代索引完成校验后才切换活动指针，构建失败保留上一代；旧代不会自动删除。Windows 中文绝对路径的 ChromaDB 索引使用英文相对路径访问，因此从其他程序导入核心模块时也请以项目目录作为工作目录。

## 启动和停止

```powershell
.\start.bat
.\start.bat --no-browser
.\start.bat --port 5001
.\stop.bat
.\stop.bat -Port 5001
```

`start.bat` 会创建缺失的虚拟环境、安装缺失依赖，启动服务前检查索引并预热模型。已有模型及索引会被复用，首次运行可能需要下载和建立索引。

也可以直接运行：

```powershell
.\.venv\Scripts\python.exe -X utf8 run.py --no-browser --port 5001
```

Web 服务由 Waitress 提供。`stop.bat` 会核对运行中服务的项目身份，再停止 Web 进程；存在本项目便携 MySQL 清单时，会继续正常关闭该数据库。外部安装的 MySQL 不受此脚本管理。

仅管理已有便携数据库：

```powershell
.\.venv\Scripts\python.exe -X utf8 local_mysql.py start
.\.venv\Scripts\python.exe -X utf8 local_mysql.py stop
```

便携部署无需注册 Windows 系统服务。`start.bat`、`stop.bat` 和便携 MySQL 管理脚本针对 Windows；Linux/macOS 可使用 `.venv/bin/python` 执行模型、索引与 Web 入口，并独立管理 MySQL。

## 数据与备份

- `data/chroma/`、`data/generations/`、`data/active.json`：向量索引、文档与切块缓存。
- MySQL：账户、独立对话、历史问答、摘要和点赞辅助知识。
- `data/auth-secret.key`：自动生成的会话签名密钥，可用 `FLASK_SECRET_KEY` 环境变量替代。
- `models/`：可重新下载的本地模型缓存。
- `config_local.py`：本机配置，不提交到 Git。

MySQL 备份可使用数据库自己的备份工具；若复制便携部署的数据目录，应先正常停止数据库。恢复原有会话时保留会话签名密钥。

便携 MySQL 日志为 `data/mysql-process.log`、`data/mysql-error.log`。直接运行 Web 服务时日志输出到终端；采用后台重定向启动时，以自行指定的日志文件为准。

## 常见问题

- **无法连接账户数据库**：检查 MySQL 是否运行、主机和端口是否正确、专用账户密码及数据库权限是否匹配。游客模式也需要数据库。
- **生成接口返回 401**：检查自己的 DeepSeek key；若直接执行 Python，还需检查同名环境变量是否覆盖文件配置。修改配置后重启 Web 服务。
- **只能检索，不能生成**：确认 key 已填写且账户可用。页面选择“仅检索”时不会调用生成 API。
- **模型下载失败**：检查网络后重新运行相应下载命令，程序会复用已有完整缓存。
- **端口占用**：使用 `start.bat --port 5001`；停止时带上对应端口。
- **完整测试无法建库**：为测试单独配置具备临时建库和删库权限的账户；日常应用账户无需全局数据库权限。

本机默认仅监听回环地址。需要通过 HTTPS 提供访问时，可使用反向代理，并设置 `SESSION_COOKIE_SECURE=true`；部署方需自行配置域名、TLS 和访问范围。
