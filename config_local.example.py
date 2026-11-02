"""复制为 config_local.py 后填写自己的配置；请勿提交真实凭证。"""

# 留空时仅检索；也可通过 DEEPSEEK_API_KEY 环境变量配置。
DEEPSEEK_API_KEY = ""

MYSQL_ENABLED = True
MYSQL_HOST = "127.0.0.1"
MYSQL_PORT = 3306
MYSQL_USER = "lingxing_rag"
MYSQL_PASSWORD = ""
MYSQL_DATABASE = "lingxing_rag"

# 启动后每小时检查现有 GitHub 文档源；改为 False 可关闭后台同步。
DOC_SYNC_ENABLED = True
DOC_SYNC_INTERVAL_SECONDS = 3600

# 默认本地判断问题类型，必要时补查一次。
RETRIEVAL_RETRY_ENABLED = True

# 整次模型请求的本地保守预算，包含输入、预留回答及余量。
# 切换模型时请确保不超过其实际上下文窗口；统计采用 UTF-8 字节保守估算。
LLM_CONTEXT_WINDOW = 32768
LLM_CONTEXT_SAFETY_MARGIN = 1024
