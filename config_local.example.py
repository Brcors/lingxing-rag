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
REACT_RETRIEVAL_ENABLED = True
REACT_MAX_TOOL_CALLS = 2
