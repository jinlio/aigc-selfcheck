"""窗口分数缓存：sha256(路径键 + 模型 id + 窗口文本 + 可选上下文) → {score, meta}。

改一句话不用全文重跑。JSONL 追加存储，
put 先落内存，close 时统一刷盘并去重。

上下文只对使用条件打分的路（C 路）参与哈希；空上下文不写入摘要，
因此 a/b 路与各段首窗的旧缓存键保持完全不变。
"""

from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path

DEFAULT_CACHE_DIR = Path.home() / ".cache" / "aigc-selfcheck"


class ScoreCache:
    def __init__(self, path: str | Path | None = None, enabled: bool = True):
        self.enabled = enabled
        self.path = Path(path) if path else DEFAULT_CACHE_DIR / "scores.jsonl"
        self._data: dict[str, dict] = {}
        self._new: dict[str, dict] = {}
        if enabled and self.path.exists():
            with self.path.open(encoding="utf-8") as fh:
                for line in fh:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        item = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    self._data[item["k"]] = item

    @staticmethod
    def key(path_key: str, model_id: str, text: str, context: str = "") -> str:
        digest = hashlib.sha256()
        digest.update(path_key.encode())
        digest.update(b"\x00")
        digest.update(model_id.encode())
        digest.update(b"\x00")
        digest.update(text.encode("utf-8"))
        if context:
            digest.update(b"\x00")
            digest.update(context.encode("utf-8"))
        return digest.hexdigest()

    def get(
        self, path_key: str, model_id: str, text: str, *, context: str = ""
    ) -> dict | None:
        if not self.enabled:
            return None
        return self._data.get(self.key(path_key, model_id, text, context))

    def put(
        self,
        path_key: str,
        model_id: str,
        text: str,
        value: dict,
        *,
        context: str = "",
    ) -> None:
        if not self.enabled:
            return
        item = {"k": self.key(path_key, model_id, text, context), "ts": time.time(), **value}
        self._data[item["k"]] = item
        self._new[item["k"]] = item

    def flush(self) -> None:
        if not self.enabled or not self._new:
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as fh:
            for item in self._new.values():
                fh.write(json.dumps(item, ensure_ascii=False) + "\n")
        self._new.clear()
