"""路线 A：AIGC_detector_zhv2 微调分类器（主力，transformers 本地推理）。

RoBERTa-wwm-ext 底座（ICLR'24 Spotlight，MPU 训练）。已知局限：
训练数据的 AI 文本以 2023-2024 模型为主，对新模型倾向低估——
融合时不能给独大权重（ALGORITHM §2.1）。
"""

from __future__ import annotations

from .base import (
    Detector,
    DetectorResult,
    ai_label_index,
    chunk_long_text,
    resolve_device,
    softmax,
)


class Zhv2Detector(Detector):
    path_key = "a"
    model_id = "yuchuantian/AIGC_detector_zhv2"

    def __init__(self, device: str = "auto", max_batch: int = 16):
        super().__init__(device)
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        self._torch = torch
        self.device = resolve_device(device)
        self.max_batch = max_batch
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_id, trust_remote_code=False
        )
        if self.tokenizer.pad_token is None:  # 批量推理需要 pad（RoBERTa-wwm 系通常自带）
            self.tokenizer.pad_token = self.tokenizer.unk_token or self.tokenizer.eos_token
        self.model = (
            AutoModelForSequenceClassification.from_pretrained(
                self.model_id, trust_remote_code=False
            )
            .to(self.device)
            .eval()
        )
        self.ai_index = ai_label_index(self.model.config.id2label)

    def _meta(self, chunks: int) -> dict:
        return {
            "model": self.model_id,
            "device": self.device,
            "ai_label": self.model.config.id2label[str(self.ai_index)]
            if str(self.ai_index) in (self.model.config.id2label or {})
            else self.ai_index,
            "chunks": chunks,
            "truncated": chunks > 1,
        }

    def detect_batch(self, texts: list[str], contexts: list[str] | None = None) -> list[DetectorResult]:
        torch = self._torch
        flat: list[str] = []
        spans: list[tuple[int, int]] = []
        for text in texts:
            chunks = chunk_long_text(self.tokenizer, text)
            spans.append((len(flat), len(flat) + len(chunks)))
            flat.extend(chunks)
        if not flat:
            return []
        scores: list[float] = []
        with torch.no_grad():
            for start in range(0, len(flat), self.max_batch):
                enc = self.tokenizer(
                    flat[start : start + self.max_batch],
                    truncation=True,
                    max_length=512,
                    padding=True,
                    return_tensors="pt",
                ).to(self.device)
                logits = self.model(**enc).logits.float().cpu().numpy()
                probs = softmax(logits)
                scores.extend(float(probs[row, self.ai_index]) for row in range(len(probs)))
        return [
            DetectorResult(score=max(scores[lo:hi]), meta=self._meta(hi - lo)) for lo, hi in spans
        ]

    def detect(self, text: str, context: str = "") -> DetectorResult:
        return self.detect_batch([text])[0]
