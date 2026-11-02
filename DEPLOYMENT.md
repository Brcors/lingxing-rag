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

接口文档目录为项目内的 `领星接口API文档/`。知识目录及 `knowledge/` 补充目录支持 Markdown、TXT、PDF、DOCX、XLSX、XLSM、XLS、CSV 和 TSV；程序忽略 `index.md`、隐藏目录和抓取任务 JSON。`tasks.json` 仅用于恢复来源 URL，不会作为知识内容入库。可通过 `DOCS_DIR` 配置其他本机文档目录。

PDF 优先提取内嵌文本；扫描页和文字稀少页面使用 PaddleOCR。安装 PaddleOCR CPU 版及中文识别依赖：

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements-ocr.txt
```

PaddleOCR 的识别模型在首次处理扫描 PDF 时下载并缓存在本机。普通数字 PDF 不需要启动 OCR。

启动及更新索引时会复用未改变的向量。新一代索引完成校验后才切换活动指针，构建失败保留上一代；旧代不会自动删除。Windows 中文绝对路径的 ChromaDB 索引使用英文相对路径访问，因此从其他程序导入核心模块时也请以项目目录作为工作目录。

## 启动和停止

### 检索补查与追问

`RETRIEVAL_RETRY_ENABLED` 默认为 `True`。首轮证据不足时最多追加一次本地规则查询，设为 `False` 可关闭补查。原有指代补全、别名和可选流程问题拆解继续生效；分流及补查不会增加规划模型调用。流程问题的首轮可按 `QUERY_DECOMPOSITION_*` 融合多个子查询，补查只追加一条查询且不重复拆解。

修改配置后重启即可，无需重新计算文档向量。旧 `REACT_*` 参数已移除。省略主语的参数追问会补入本对话先前的业务对象；多接口流程未确定目标时先澄清，再回答具体参数。澄清记录与个人对话一起保存，不进入共享点赞库。版本查询读取本机保存的记录，停止生成仍可取消检索后的生成及已有的拆解请求。

### 模型请求总预算

在 `config_local.py` 中可设置 `LLM_CONTEXT_WINDOW=32768` 和 `LLM_CONTEXT_SAFETY_MARGIN=1024`。输入预算为总预算减去 `DEEPSEEK_MAX_TOKENS` 与安全余量；需至少留下 8192 的输入预算，安全余量至少 256。切换模型或 API 服务时，应确保总预算不超过其实际上下文窗口；默认值为本项目主动设置的保守上限。

预算统计以 UTF-8 字节加消息开销保守估算，并非提供商精确分词或计费量。回答按最早整轮问答、摘要完整句子、低优先级整份证据的顺序精简；不修改数据库原始记录。系统规则、当前问题及必要证据仍超限时不发送请求，提示缩小范围。已保留的证据不从表格或 JSON 中间截断。

所有生成、问题拆解和摘要请求发送前都有检查。服务端明确返回上下文超限、且尚未输出回答时，回答流程最多精简重试一次，不重复检索或拆解。摘要超限会保留原始问答及既有摘要，按原有失败机制等待重试。修改这两项配置后重启服务，无需重建索引。API 回答统计包含 `prompt_tokens_estimated`、`prompt_input_budget`、`prompt_output_reserved`、`prompt_safety_margin`、`prompt_context_window`、`prompt_history_turns_removed`、`prompt_memory_reduced`、`prompt_sources_removed` 及 `context_length_retry_applied`。

### 文档同步配置

`run.py` 会启动后台检查线程，默认每小时检查已配置的 GitHub 文档源。手动执行同步命令或点击页面按钮会立即检查。`create_app()` 作为自定义 WSGI 入口时，需要自行调用 `app.extensions["document_sync"].start()` 才会启用定时检查。

- `DOC_SYNC_ENABLED`：是否启用后台和页面同步入口，默认 `True`。
- `DOC_SYNC_REPOSITORY`、`DOC_SYNC_BRANCH`、`DOC_SYNC_SUBDIRECTORY`：文档仓库、分支和子目录。
- `DOC_SYNC_BASE_REVISION`：现有文档快照的完整初始提交，用于首次检查本地修改；接入自有来源且没有初始提交时可设为空。
- `DOC_SYNC_INTERVAL_SECONDS`：检查间隔，默认 3600，最小 60。

切换来源或文档目录时，请使用独立的 `DATA_DIR`，避免混用旧版本记录和索引。GitHub 同步只处理 Markdown 和根目录 `tasks.json`；本地额外的笔记及其他支持格式会保留并参与索引。每个版本的完整增删改清单与修改前备份保存在 `data/document-sync/versions/`。

后台同步依赖 GitHub 公共 API 的网络与限额，失败信息会显示在侧栏。现有索引继续提供检索。文档版本时间采用上游提交时间，不代表官方发布日期。

### Web 服务

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
