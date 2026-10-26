"""API 批量生成 AI 语料样本（v4 扩容轮）：OpenAI 兼容聚合端点 × 5 条标准提示词。

用途：用聚合平台（如阿里云百炼 compatible-mode）低成本补充**不同模型家族**的
AI 段落，扩大 LOGO-生成器 CV 的生成器池。与网页版样本的区别（溯源注释如实记录）：
API 调用无联网搜索、无深度思考，参数为服务端默认。

凭据纪律：密钥只从环境变量读取（OPENAI_API_KEY / OPENAI_API_BASE_URL），
源码与仓库不落任何凭据。

网络安全（与 fetch_human_corpus.py 同款约束）：端点仅允许 https + 域名白名单
（阿里云百炼及其网关），DNS 解析后阻断私网/环回/链路本地地址，禁用重定向。

    # 列出端点可用模型
    OPENAI_API_KEY=... OPENAI_API_BASE_URL=... uv run python scripts/gen_ai_api.py --list-models

    # 用指定模型生成 5 段（序号从 06 起），存入 calibration/ai/
    OPENAI_API_KEY=... OPENAI_API_BASE_URL=... uv run python scripts/gen_ai_api.py \
        --model qwen-max --prefix qwapi --start-no 6
    # 只补特定提示词（如上次中断在 2/5，从第 3 条继续）
    OPENAI_API_KEY=... OPENAI_API_BASE_URL=... uv run python scripts/gen_ai_api.py \
        --model qwen3.8-flash --prefix v7q38f --start-no 3 --only 3,4,5
"""

from __future__ import annotations

import ipaddress
import json
import os
import re
import socket
import sys
import urllib.error
import urllib.request
from pathlib import Path
from urllib.parse import urlparse

ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "calibration" / "ai"

# 端点域名白名单：阿里云百炼官方端点及其聚合网关（ trusted 后缀匹配）
ALLOWED_HOST_SUFFIXES = ("dashscope.aliyuncs.com", ".qianwenaiapi.com")

# 与 calibration/AI_PROMPTS.md 保持一致的 5 条标准提示词（主题与人类语料刻意重叠）
PROMPTS: list[tuple[str, str]] = [
    ("技术说明", "请用一段话（150~250字）介绍什么是数据库索引，说明它为什么能加快查询速度。不要分点，不要标题，写成一整段。"),
    ("议论", "请用一段话（150~250字）论述「开源软件对技术发展的利大于弊」这一观点，给出你的理由。不要分点，写成一整段。"),
    ("学术摘要", "假设你刚完成一篇关于在线学习效果的实证研究论文，请写一段150~250字的中文摘要，包含研究目的、方法、结果与结论。不要分点，写成一整段。"),
    ("科普", "请用一段话（150~250字）向普通读者解释深度学习是怎么「认出」图片里的猫的。语言通俗，不要分点，写成一整段。"),
    ("方法建议", "请用一段话（150~250字）说明大学生如何制定有效的时间管理计划并坚持执行。不要分点，写成一整段。"),
]

TOPIC_SUFFIX = {
    "技术说明": "数据库索引",
    "议论": "开源利弊",
    "学术摘要": "学术摘要",
    "科普": "深度学习认猫",
    "方法建议": "时间管理",
}

MAX_TOKENS = 800  # 150~250 字中文远小于此；上限只为兜底控费
TIMEOUT_S = 120


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):  # noqa: D102
        raise urllib.error.URLError(f"重定向被拒绝：{newurl}")


_OPENER = urllib.request.build_opener(_NoRedirect)


def _env(name: str) -> str:
    value = os.environ.get(name, "").strip()
    if not value:
        raise SystemExit(f"缺少环境变量 {name}（密钥不入库，运行前从密钥存储/本地 .env 导出）")
    return value.rstrip("/")


def _validate_base(base: str) -> str:
    """SSRF 防护：https + 域名白名单 + 解析后阻断私网/环回/链路本地地址。"""
    parsed = urlparse(base)
    if parsed.scheme != "https":
        raise SystemExit("OPENAI_API_BASE_URL 仅允许 https 端点")
    host = (parsed.hostname or "").lower()
    if not host or not host.endswith(ALLOWED_HOST_SUFFIXES):
        raise SystemExit(f"端点主机不在白名单：{host}（白名单：{ALLOWED_HOST_SUFFIXES}）")
    port = parsed.port or 443
    for *_, sockaddr in socket.getaddrinfo(host, port, proto=socket.IPPROTO_TCP):
        ip = ipaddress.ip_address(sockaddr[0])
        if not ip.is_global:
            raise SystemExit(f"端点解析到非公网地址，拒绝连接：{host} -> {ip}")
    if parsed.path in ("", "/"):
        raise SystemExit("OPENAI_API_BASE_URL 需带 compatible-mode 路径（如 …/compatible-mode/v1）")
    return base


def _request(base: str, key: str, method: str, path: str, payload: dict | None = None) -> dict:
    req = urllib.request.Request(
        f"{base}{path}",
        data=json.dumps(payload).encode("utf-8") if payload else None,
        headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
        method=method,
    )
    with _OPENER.open(req, timeout=TIMEOUT_S) as resp:
        return json.loads(resp.read().decode("utf-8"))


def _cjk(text: str) -> int:
    return sum(1 for ch in text if "\u4e00" <= ch <= "\u9fff")


def list_models(base: str, key: str) -> None:
    data = _request(base, key, "GET", "/models")
    ids = sorted(item["id"] for item in data.get("data", []))
    print(f"{len(ids)} 个可用模型：")
    for mid in ids:
        print(f"  {mid}")


def generate(
    base: str,
    key: str,
    model: str,
    prefix: str,
    start_no: int,
    date: str,
    only: set[int] | None = None,
) -> None:
    created = []
    for idx, (style, prompt) in enumerate(PROMPTS, 1):
        if only and idx not in only:
            continue
        body = None
        usage: dict = {}
        for attempt in (1, 2):
            resp = _request(
                base,
                key,
                "POST",
                "/chat/completions",
                {
                    "model": model,
                    "messages": [{"role": "user", "content": prompt}],
                    "max_tokens": MAX_TOKENS,
                },
            )
            content = (resp.get("choices") or [{}])[0].get("message", {}).get("content", "")
            # 聚合端点偶尔包一层 markdown 代码块，剥掉
            content = re.sub(r"^```[a-z]*\n?|```$", "", content.strip(), flags=re.M).strip()
            usage = resp.get("usage", {})
            if 100 <= _cjk(content) <= 450:
                body = content
                break
            print(f"  [重试{attempt}] 提示词{idx} 汉字数 {_cjk(content)} 越界")
        if body is None:
            raise SystemExit(f"提示词 {idx} 两次生成都越界，中止（已生成 {len(created)} 段）")
        header = (
            f"<!-- 平台: 阿里云百炼(API) | 模型: {model} | 日期: {date} | 文体: {style} | "
            f"提示词: {idx} | 搜索: 关闭（API 无联网搜索） | 思考: 关闭 | 参数: 服务端默认 -->"
        )
        out = OUT_DIR / f"{prefix}{start_no + len(created):02d}-{TOPIC_SUFFIX[style]}.txt"
        if out.exists():
            raise SystemExit(f"已存在：{out}（换 --start-no 或 --prefix）")
        out.write_text(f"{header}\n\n{body}\n", encoding="utf-8")
        created.append((out.name, _cjk(body), usage.get("total_tokens")))
        print(f"{out.name:32s} {_cjk(body):4d} 汉字  tokens={usage.get('total_tokens')}")
    total = sum(t for _, _, t in created if t)
    print(f"共 {len(created)} 段，总 token 消耗 ≈ {total}")


def main() -> int:
    if "--list-models" in sys.argv:
        base = _validate_base(_env("OPENAI_API_BASE_URL"))
        list_models(base, _env("OPENAI_API_KEY"))
        return 0
    args = dict(zip(sys.argv[1::2], sys.argv[2::2], strict=False))
    for need in ("--model", "--prefix"):
        if need not in args:
            raise SystemExit(__doc__)
    base = _validate_base(_env("OPENAI_API_BASE_URL"))
    only_arg = args.get("--only", "")
    only = {int(part) for part in only_arg.split(",") if part.strip()} if only_arg else None
    generate(
        base,
        _env("OPENAI_API_KEY"),
        args["--model"],
        args["--prefix"],
        int(args.get("--start-no", "1")),
        args.get("--date", "2026-10-25"),
        only=only,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
