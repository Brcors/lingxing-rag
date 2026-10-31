# 领星接口文档问答

Flask 简洁问答页面 + BGE-small 本地向量 + ChromaDB + jieba BM25 + RRF + BGE 本地重排序。回答由 DeepSeek V4.1 Flash 生成，附来源文档和检索片段。账户登录后，完整生成的问答按用户和独立对话写入 MySQL，提供持久历史与滚动记忆。只有点击 👍 的问答才额外加入所有账户共享的辅助知识库。

## 开始使用

当前电脑已完成本地部署，直接运行 `start.bat`；便携 MySQL 会随项目自动启动。数据库位置、管理命令和本机部署说明见 [DEPLOYMENT.md](DEPLOYMENT.md)。

从 GitHub 首次下载时，将 `config_local.example.py` 复制为 `config_local.py` 并填写自己的 API key 和 MySQL 参数。仓库仅包含代码、文档和空白配置示例；本机凭证、账户与聊天记录、数据库文件、模型权重、虚拟环境和运行日志均不随仓库上传。模型可用下方管理命令下载，数据库需要在自己的电脑上安装和配置。

1. 本机已经配置好 key 和 MySQL；更换 key 请编辑 `config_local.py` 中的 `DEEPSEEK_API_KEY`。全新安装且没有本机配置文件时，在 `config.py` 填写 key 和数据库连接参数。
2. 双击 `start.bat`。启动时检查文档变动并复用已有模型和索引，浏览器会打开 `http://127.0.0.1:5000`。
3. 确认 MySQL 服务已启动，在页面注册账户并登录。可新建多个独立对话，也可从左侧继续已有对话。默认流式生成回答；“仅检索”不调用 DeepSeek，也不保存成个人问答。

重复双击 `start.bat` 会识别本项目已运行的服务并打开页面，不会重复启动或报端口占用。若 5000 被其他程序占用，可执行 `start.bat --port 5001`。脚本支持 `--no-browser`，缺少依赖时会自动安装。批处理使用 Windows CRLF 换行，脚本命令保持 ASCII，中文日志由 UTF-8 的 Python 进程输出。

填写 key 后需要重启服务。直接运行 Python 时，非空的 `DEEPSEEK_API_KEY` 环境变量优先于文件中的值；空环境变量不会覆盖文件配置。`start.bat` 在 `config_local.py` 已明确配置 key 时优先使用该本机配置，详见部署说明。配置文件包含明文 key 时请自行保管；项目不会向前端返回 key。

当前官方调用名称是 `deepseek-flash`，对应 V4.1 Flash；参考 https://api-docs.deepseek.com/api/create-chat-completion/ 和 https://www.deepseek.com/en/news/deepseek-v4-1-flash/ 。用户提供的 `LLM.py` 保持原文件，项目适配器位于 `rag/generation/client.py`。

## 环境与文档

Python 3.11 及以上，当前在 Windows / Python 3.13 上准备。依赖安装于项目 `.venv`；本机环境复用了系统已安装的部分库。迁移到其他机器时新建环境并执行：

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe run.py
```

默认文档目录：项目内的 `领星接口API文档/`。更改位置在 `config.py` 设置 `DOCS_DIR`。原始文档只读，排除索引目录页、隐藏目录和抓取任务 JSON。`tasks.json` 用来恢复原文 URL。

本地模型为 `BAAI/bge-small-zh-v1.5`，约 96 MB 权重、512 维，默认 CPU。首次下载需要联网；模型保存在 `models/`，完整下载后向量计算可离线运行。生成回答需要联网调用 DeepSeek。

模型缓存记录仓库 ID 与具体 revision。更换 `EMBEDDING_MODEL` 时使用单独的模型缓存，避免沿用旧权重；索引更新期间的分词计数与编码共享锁，防止并发访问同一 tokenizer。

`knowledge/` 保存注明来源与日期的官方接入更新摘要，用来处理旧快照的 token 迁移示例；其余接口资料仍是本地快照，不承诺自动同步官网。

## 模块

| 路径 | 职责 |
| --- | --- |
| `app/` | Flask 应用工厂、Blueprint、页面、静态资源 |
| `rag/ingestion/` | 文档解析，保留标题层级、表格行和代码块 |
| `rag/embedding.py` | 本地模型下载、token 计数、归一化向量 |
| `rag/indexing/` | 版本化 Chroma 集合、增量向量复用、原子切换 |
| `rag/retrieval/` | jieba + 标识符分词、BM25、向量、RRF、可选重排序 |
| `rag/context.py` | 补齐所属接口与章节、去重、上下文预算 |
| `rag/generation/` | 提示词、DeepSeek 请求、流式响应与错误处理 |
| `rag/knowledge/` | 点赞暂存、MySQL 持久化，以及历史点赞问答的混合检索 |
| `rag/service.py` | 统一检索、生成与点赞反馈编排 |

MySQL 使用 `lingxing_rag.liked_answers` 表（首次检索或点赞时自动创建）。该表只保存用户明确点击 👍 的共享问答、来源和本地向量。完整问答的个人记录另存在 chat_turns，不参与其他账户的个人上下文。后续检索使用 jieba BM25 与本地 BGE 向量进行融合，可选复用本地重排序模型；每次最多补充 1 条点赞问答，上下文最多占用 350 token。点赞问答会明确标成低优先级辅助参考，当前接口文档优先；原回答引用的来源信息会一并保留。修改 `config.py` 的 `MYSQL_*` 项可调整数据库连接和检索阈值，将 `MYSQL_ENABLED` 设为 `False` 可关闭此功能。依赖为 `PyMySQL`。

检索块按章节、段落和表格结构划分，目标约 400 个 BGE tokenizer token，包含元数据后不超过 480；长段落重叠约 50 token。超长表格单行或代码行无法整体放入模型时，生成带有字段/示例标签的续片，原文可完整查看。最终上下文的 4000 token 是针对生成模型的保守估算，与 BGE tokenizer 不同。

BM25 与向量各取 20 条，按 chunk 去重、RRF 融合及标题/路径规则评分后截取前 20 条；本地重排序模型重新判断问题与各候选的相关性，再限制同接口片段占比，选约 6 条并补齐父级信息。返回的证据数量可能因同章去重和长度预算少于 6。多步流程问题仍需文档有明确依赖；本工具不会自动发起领星业务 API 请求。

仅当问题含流程、步骤、前置条件、依赖或先后顺序等线索时，才额外调用一次 DeepSeek 将问题拆成最多 3 个检索子问题；普通问题和“仅检索”模式不触发拆解。拆解结果与原问题分别检索，合并去重后统一重排序；拆解失败时回退原问题。最终候选的最佳重排分数低于 `RERANK_MIN_SCORE`（默认 -1.0）时，系统不引用这些候选并提示补充问题信息。阈值是本地模型原始分数，不是概率，可在 `config.py` 调整；重排序不可用时仍按混合检索顺序回退。

上下文先为各命中章节保留最小证据，再补其他命中和父级内容，避免一个长章节挤掉其余来源。引用处理保留代码中的 `[S1]` 等字面文本。DeepSeek 读取超时和连接中断不自动重发已可能被处理的请求；只有连接建立超时及可重试 HTTP 状态会进行有限重试。

## 管理命令

```powershell
.\.venv\Scripts\python.exe manage.py check
.\.venv\Scripts\python.exe manage.py download-model
.\.venv\Scripts\python.exe manage.py download-reranker
.\.venv\Scripts\python.exe manage.py index
.\.venv\Scripts\python.exe manage.py index --force
.\.venv\Scripts\python.exe manage.py search "如何查询 FBA 货件列表？"
.\.venv\Scripts\python.exe run.py --no-browser --port 5001
```

页面“更新文档索引”也可增量更新。新增、变更和删除的文档都体现在新一代语料中；相同 chunk 的旧向量复用。模型、切块设置或解析器版本变化触发重新索引。构建完成后才切换活动指针，构建失败继续使用上一代索引。旧代集合留作恢复，不自动删除。

Windows 中文绝对路径可能导致 ChromaDB 的 HNSW 文件无法重启读取；本项目从项目目录向 Chroma 传递英文相对路径，数据仍保存在本项目 `data/chroma` 内。入口脚本自动切换到项目目录；从其他程序导入核心模块时也应从项目目录启动。索引保留独立的 `vectors.npy` 缓存，更新和修复可以复用向量。发布活动索引前会执行实际向量查询，不只检查记录数量。

重排序默认开启，使用支持中英文的 `BAAI/bge-reranker-base`（MIT 许可）的预量化 ONNX INT8 版本，来源为 `Xenova/bge-reranker-base`：https://huggingface.co/Xenova/bge-reranker-base 。模型权重约 279 MB，在本机 CPU 运行，不调用付费 API。直接下载预量化权重，避免本机转换时占用大量内存。完整缓存位于 `models/reranker/` 的独立模型目录，记录仓库 ID 与 revision，后续离线加载。启动时预热一次，同一进程复用模型并串行推理，避免并发争用 CPU。

在 `config.py` 可设置 `RERANK_ENABLED`、`RERANK_MODEL`、`RERANK_BACKEND`、`RERANK_CANDIDATES`（20）、`RERANK_BATCH_SIZE`（4）、`RERANK_MAX_LENGTH`（512，问题和片段合计的模型 token 上限）和最低分 `RERANK_MIN_SCORE`。`onnx-int8` 需要模型仓库提供 `onnx/model_int8.onnx`；如改用 `BAAI/bge-reranker-v2-m3`，需同时设置后端为 `torch`，按原精度加载。重排使用原始相关性分数，不与量纲不同的 RRF 分数直接相加。同分保持融合后的先后顺序。重排失败时本次回退 RRF 顺序，页面显示提示，详细原因写入启动日志。更换重排模型无需重算文档向量，修改配置后重启服务。

`manage.py check` 可检查重排权重是否下载；检索返回的统计包含 `rerank_applied`、`rerank_model`、`rerank_candidates` 和 `rerank_seconds`，页面显示实际重排耗时。关闭重排可将 `RERANK_ENABLED` 改为 `False`。

## API

- `GET /api/status`：索引、模型配置状态（不返回 key）。
- `POST /api/ask`：JSON 一次性返回答案、来源和耗时。
- `POST /api/ask/stream`：NDJSON，依次返回 `status`、`sources`、`token`、`done` 或 `error`。
- `POST /api/feedback`：提交流式 `done` 事件中的 `feedback_id`，仅用于明确点赞后的持久化；接口不接受浏览器提交的问答正文。
- `POST /api/index`：后台更新索引，状态通过 `/api/status` 查看。
- `GET /documents/<document_id>`：本地完整文档预览。

提问 JSON：

```json
{"question":"如何查询货件列表？","category":"FBA","mode":"answer","conversation_id":"对话ID","request_id":"每次提问的唯一UUID"}
```

`category` 可省略；`mode` 可设为 `retrieve`；历史和记忆从服务器按登录账户与 conversation_id 读取，忽略浏览器提交的 history。无 API key 时自动返回检索预览。页面 Markdown 经后端清理；不使用外部 CDN。

## 验证

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements-dev.txt
.\.venv\Scripts\python.exe -m pytest -q
.\.venv\Scripts\python.exe evaluate.py
```

单元与接口测试包含 Markdown 层级、表格切分、标识符分词、融合、上下文预算、向量复用/删除、流式协议、完整检索到生成编排和 API 错误处理。它们使用本地替身验证模型调用协议，不消耗 DeepSeek 余额；真实 API key 的授权及付费生成需填写 key 后验证。`evaluate.py` 使用真实本地向量模型检查 8 条固定案例，并将结果保存到 `data/retrieval-evaluation.json`；该小样本不代表总体准确率。

若 Windows 的公共临时目录存在权限冲突，可在索引已建立后使用 `python -m pytest -q --basetemp data/pytest-tmp`，该目录专用于可丢弃的测试数据。


## 账户、独立对话与个人记忆

MySQL 新增 auth_users（用户名与 Argon2id 密码哈希）、chat_conversations（用户所属对话及当前摘要）、chat_turns（完整问答、来源与点赞标识）、chat_memory_versions（可追溯的摘要版本）。首次访问账户功能时自动建表；既有 liked_answers 继续全局共享，未点赞的个人问答不进入共享检索。

每个独立对话首次积累 5 组完整问答后调用 DeepSeek 整理一条摘要；此后每新增 4 组，将旧摘要与这 4 组更新成新摘要。原始问答始终保留。摘要后台执行，模型失败保留待整理记录并显示提示，之后打开或继续该对话时重试。数据库租约避免多个进程同时整理相同批次；进程中断后过期租约会自动恢复。连续对话使用当前摘要和最多 4 组尚未整理的近期问答，摘要仅提供对话背景，接口事实仍以本轮检索资料为依据。整理完成后，上一个问题仍作为简短追问的检索指代线索。

个人页面与问答 API 需要登录，GET /api/status 保留为公开健康检查。会话 cookie 只记录登录标识及 CSRF 值，HttpOnly、SameSite=Lax；默认 8 小时过期。签名密钥自动生成在 data/auth-secret.key，可用 FLASK_SECRET_KEY 环境变量覆盖。所有修改请求需发送页面 meta csrf-token 的值到 X-CSRF-Token 请求头。通过 HTTPS 部署时设置 SESSION_COOKIE_SECURE=true；本机 HTTP 保持默认。

- GET /login、GET /register：登录注册页面。
- POST /api/auth/register、POST /api/auth/login、POST /api/auth/logout：注册、登录和退出。
- GET /api/conversations、POST /api/conversations：列出个人对话、新建对话。
- GET /api/conversations/<id>：读取当前用户拥有的对话，最近 50 组问答；用 ?before=<问答ID> 继续向前翻页。
- POST /api/ask、POST /api/ask/stream：携带 conversation_id 和唯一 request_id。完整回答保存后再发送 done；重复 request_id 返回已保存的回答，防止重复计入记忆。
- POST /api/feedback：只能点赞自己的已保存问答；刷新和服务重启后仍可点赞。点赞内容进入共享知识库。

账户功能需要 MySQL；数据库关闭时登录和持久化操作会返回明确错误。个人上下文按用户和对话双重校验，其他对话不会自动带入当前上下文。记忆整理额外消耗 DeepSeek token，首次第 5 组后一次，此后每新增 4 组一次。

新增的 test_accounts_memory.py 使用随机命名的独立 MySQL 测试库，测试结束删除；模型响应采用替身，不请求真实 DeepSeek。需本机 MySQL 可用且测试账户有建库权限。


## 游客模式

登录和注册页面都提供“以游客身份进入”。游客无需创建账户，可以新建多个临时独立对话，使用官方文档及全局点赞知识库检索。游客不能点赞写入共享知识库，也不能更新共享文档索引；这些限制由后端校验。

游客问答独立存入 guest_chat_turns 表，使用 guest_session_id 隔离不同游客。对话元数据与临时摘要复用 chat_conversations、chat_memory_versions；原有账户数据保持按 user_id 隔离。游客临时记忆采用同样的首次 5 组、之后每新增 4 组滚动摘要规则。

点击“退出游客”后，服务器删除该游客的全部临时对话，外键级联删除问答与摘要版本，其他游客和注册账户不受影响。旧游客 cookie 无法恢复已经清除的记录。游客成功注册或登录正式账户时也会清理该游客记录，不自动迁移到正式账户。

关闭页面、断网或关闭浏览器不会可靠通知服务器。游客每次活动会把有效期延长到 24 小时后；服务运行期间每分钟清理一次过期数据，重启时也会检查。服务或 MySQL 停止时无法执行删除，恢复后继续清理。正式账户登录保持 8 小时的滑动会话有效期，游客为 24 小时。

新增接口 POST /api/auth/guest；需和其他修改请求一样提供 X-CSRF-Token。账户表升级是幂等的，自动增加游客标识、过期时间和独立游客问答表，不删除注册账户历史。游客模式同样需要 MySQL 服务运行。
