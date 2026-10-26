"""路线 B：cmj-chinese-aigc-text-detector（ONNX，去相关的第二意见）。

与 A 训练数据不同源、错误不完全相关；CPU 上 ONNX Runtime 是无 GPU
用户的主力（ALGORITHM §2.2）。模型仓库自带 onnx/ 子目录（含 fast
tokenizer），无需自行转换。
"""

from __future__ import annotations

import json

import numpy as np

from .base import Detector, DetectorResult, ai_label_index, chunk_long_text, softmax_ai_score


class CmjOnnxDetector(Detector):
    path_key = "b"
    model_id = "mokawa3018/cmj-chinese-aigc-text-detector"

    def __init__(self, device: str = "auto"):
        super().__init__(device)
        from huggingface_hub import hf_hub_download
        from onnxruntime import InferenceSession
        from transformers import AutoTokenizer

        # ONNX 路恒走 CPU EP（计划 §3：不跟随 --device）
        self.session = InferenceSession(
            hf_hub_download(self.model_id, "onnx/model.onnx"),
            providers=["CPUExecutionProvider"],
        )
        self.input_names = [item.name for item in self.session.get_inputs()]
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_id, subfolder="onnx", trust_remote_code=False
        )
        config_path = hf_hub_download(self.model_id, "config.json")
        with open(config_path, encoding="utf-8") as fh:
            config = json.load(fh)
        self.ai_index = ai_label_index(config.get("id2label"))

    def detect(self, text: str, context: str = "") -> DetectorResult:
        chunks = chunk_long_text(self.tokenizer, text)
        truncated = len(chunks) > 1
        scores = []
        for chunk in chunks:
            enc = self.tokenizer(
                chunk,
                truncation=True,
                max_length=512,
                return_tensors="np",
            )
            # 只喂 session 声明需要的输入（有的导出不含 token_type_ids）
            feed = {name: enc[name].astype(np.int64) for name in self.input_names if name in enc}
            logits = self.session.run(None, feed)[0][0]
            scores.append(softmax_ai_score(np.asarray(logits), self.ai_index))
        return DetectorResult(
            score=max(scores),
            meta={"model": self.model_id, "device": "cpu-onnx", "chunks": len(chunks), "truncated": truncated},
        )
