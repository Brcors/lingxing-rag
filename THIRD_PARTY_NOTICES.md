# 第三方资料说明

## 许可范围

根目录的 MIT 许可证覆盖本项目原创程序代码及原创说明。第三方接口文档、模型、依赖和商标不包含在这项授权中，其原有权利及许可保持不变。

## 领星接口文档快照

- 本地目录：`领星接口API文档/`。
- 来源：[lpj2026/one2](https://github.com/lpj2026/one2)。
- 固定提交：`ae142f1b0b33a8ac834c1e7c27482ff9badbcd02`。
- 快照内容：542 份 Markdown 文档及抓取任务清单。
- 官方文档入口：[apidoc.lingxing.com](https://apidoc.lingxing.com/)。

该上游仓库在本次检查时未声明开源许可证。本项目保留其来源信息，不将第三方文档重新授权为 MIT，也不对其授予额外使用或再分发权利。文档及商标权利归相应权利人所有，使用时应遵循原始来源的条款。

快照可能滞后于实际 API。`knowledge/` 中的更新说明注明了参考来源，不能替代官方实时文档。

## 模型与依赖

模型权重不随本仓库发布，运行下载命令时从对应 Hugging Face 仓库获取：

- [BAAI/bge-small-zh-v1.5](https://huggingface.co/BAAI/bge-small-zh-v1.5)。
- [Xenova/bge-reranker-base](https://huggingface.co/Xenova/bge-reranker-base)，对应 [BAAI/bge-reranker-base](https://huggingface.co/BAAI/bge-reranker-base) 的 ONNX 版本。

模型的许可和使用条件以各自仓库的声明为准。Python 依赖由包管理器安装，各依赖保留自身许可证。DeepSeek 与领星服务的使用还应遵循各自服务条款。

本项目不是领星、DeepSeek 或上述模型发布方的官方产品。
