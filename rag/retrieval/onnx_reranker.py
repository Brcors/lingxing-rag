"""BGE CrossEncoder 的 ONNX INT8 CPU 推理。"""
import os
from pathlib import Path


def runtime_path(path):
    # Windows 下 ONNX 的 C++ 文件操作对中文绝对路径支持不完整。
    if os.name == "nt" and not str(path).isascii():
        path = Path(os.path.relpath(path))
        if not str(path).isascii():
            raise RuntimeError("请从项目目录启动，以便 ONNX 使用英文相对路径加载模型。")
    return path


class ONNXCrossEncoder:
    def __init__(self, model_dir, max_length, threads):
        import onnxruntime as ort
        from transformers import AutoTokenizer

        self.max_length = max_length
        self.tokenizer = AutoTokenizer.from_pretrained(str(model_dir), local_files_only=True,
                                                       trust_remote_code=False)
        options = ort.SessionOptions()
        options.intra_op_num_threads = threads
        options.inter_op_num_threads = 1
        self.session = ort.InferenceSession(str(runtime_path(model_dir / "onnx" / "model_int8.onnx")),
                                            sess_options=options, providers=["CPUExecutionProvider"])
        self.input_names = {item.name for item in self.session.get_inputs()}

    def predict(self, pairs, batch_size=4, show_progress_bar=False):
        import numpy as np

        # 相近长度放在同一批次，减少 padding；结果恢复到输入顺序。
        order = sorted(range(len(pairs)), key=lambda i: len(pairs[i][0]) + len(pairs[i][1]))
        scores = np.empty(len(pairs), dtype=np.float32)
        for start in range(0, len(order), batch_size):
            indices = order[start:start + batch_size]
            batch = [pairs[i] for i in indices]
            encoded = self.tokenizer([p[0] for p in batch], [p[1] for p in batch],
                                     padding=True, truncation=True, max_length=self.max_length,
                                     return_tensors="np")
            inputs = {name: encoded[name] for name in self.input_names}
            output = np.asarray(self.session.run(None, inputs)[0]).reshape(-1)
            if len(output) != len(indices):
                raise ValueError("invalid ONNX rerank output")
            scores[indices] = output
        return scores
