"""火山引擎方舟（豆包 Lite）批量生成 AI 语料。

固定提示词循环采样，默认通用包 5 题 × 100 轮 = 500 段；`--pack academic`
走论文章节包 7 题 × 20 轮 = 140 段。输出到独立暂存目录，不直接写进
`calibration/ai/`：同模型样本若直接进语料，会压垮按生成器分层的切分与
LOGO-生成器 CV，需先抽样/加权再并入下一版切分。

凭据只从环境变量 `ARK_API_KEY` 读取；源码、参数和输出都不落密钥。
网络只允许 https + 固定 host 白名单 + DNS 公网校验，禁用重定向与代理。

    ARK_API_KEY=... python scripts/gen_ai_ark_lite.py --loops 100 --workers 6
    ARK_API_KEY=... python scripts/gen_ai_ark_lite.py --pack academic --loops 20
    ARK_API_KEY=... python scripts/gen_ai_ark_lite.py --loops 1 --limit 5  # 冒烟
"""

from __future__ import annotations

import argparse
import concurrent.futures
import datetime as dt
import hashlib
import ipaddress
import json
import os
import re
import secrets
import socket
import sys
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path
from urllib.parse import urlparse

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from gen_ai_api import PROMPTS, TOPIC_SUFFIX  # noqa: E402

ROOT = SCRIPT_DIR.parent
DEFAULT_BASE_URL = "https://ark.cn-beijing.volces.com/api/coding/v3"
ALLOWED_HOST = "ark.cn-beijing.volces.com"
DEFAULT_MODEL = "doubao-seed-2-1-lite-260915"
MIN_CJK = 100
MAX_CJK = 450
MAX_RETRY_TOKENS = 2400

# 学术场景包：7 个论文常见章节 × 20 个跨学科主题。与通用包的固定 5 题
# 分开跑，避免学术样本被通用文体稀释；同一 loop 的 7 个章节共享主题，
# 便于后续按「同一篇论文」聚合分析。
ACADEMIC_TOPICS = [
    "深度学习在医学影像诊断中的应用",
    "在线学习效果的影响因素",
    "低碳农业发展的生态效应",
    "传统村落旅游的可持续发展",
    "大语言模型在文本分类中的表现",
    "城市交通拥堵的治理策略",
    "企业数字化转型的绩效影响",
    "青少年心理健康的影响因素",
    "新能源汽车产业的商业模式",
    "乡村振兴背景下的土地利用",
    "社交媒体中的信息传播机制",
    "人工智能生成内容的检测方法",
    "供应链金融的风险控制",
    "教师专业发展的支持体系",
    "气候变化的农业适应策略",
    "医疗资源配置的公平性",
    "文化遗产的数字化保护",
    "大学生就业能力的影响因素",
    "智能制造中的质量控制",
    "公共图书馆的阅读推广",
]

ACADEMIC_PROMPTS: list[tuple[str, str]] = [
    (
        "学术论文-摘要",
        "请写一段中文论文摘要（200~300字），主题为「{topic}」。"
        "包含研究目的、方法、主要结果和结论；不要分点，不要标题，"
        "不要列出参考文献，写成一整段。",
    ),
    (
        "学术论文-引言",
        "请写一段中文论文引言（200~300字），主题为「{topic}」。"
        "说明研究背景、现有研究的不足和本文的贡献；不要分点，不要标题，"
        "不要列出参考文献，写成一整段。",
    ),
    (
        "学术论文-相关工作",
        "请写一段中文论文的相关工作（200~300字），主题为「{topic}」。"
        "概括已有研究的主要思路并比较其优缺点；不要分点，不要标题，"
        "不要列出参考文献，写成一整段。",
    ),
    (
        "学术论文-方法",
        "请写一段中文论文的研究方法（200~300字），主题为「{topic}」。"
        "描述研究设计、数据来源、模型或算法和关键步骤；不要分点，不要标题，"
        "不要列出参考文献，写成一整段。",
    ),
    (
        "学术论文-结果",
        "请写一段中文论文的结果与分析（200~300字），主题为「{topic}」。"
        "描述主要发现、对比结果和可能的原因；不要分点，不要标题，"
        "不要列出参考文献，写成一整段。",
    ),
    (
        "学术论文-讨论",
        "请写一段中文论文的讨论（200~300字），主题为「{topic}」。"
        "解释结果的含义、研究局限和未来方向；不要分点，不要标题，"
        "不要列出参考文献，写成一整段。",
    ),
    (
        "学术论文-结论",
        "请写一段中文论文的结论（150~250字），主题为「{topic}」。"
        "总结主要工作、贡献和实践启示；不要分点，不要标题，"
        "不要列出参考文献，写成一整段。",
    ),
]

ACADEMIC_TOPIC_SUFFIX = {style: style.removeprefix("学术论文-") for style, _ in ACADEMIC_PROMPTS}
PROMPT_PACKS = {
    "general": (PROMPTS, TOPIC_SUFFIX),
    "academic": (ACADEMIC_PROMPTS, ACADEMIC_TOPIC_SUFFIX),
}
PACK_PREFIX = {"general": "arklite", "academic": "arkacad"}
PACK_OUT_DIR = {
    "general": "calibration/ai_ark_lite",
    "academic": "calibration/ai_academic_lite",
}


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):  # noqa: D102
        raise urllib.error.URLError(f"重定向被拒绝：{newurl}")


def _opener() -> urllib.request.OpenerDirector:
    # 显式禁用代理，避免请求被环境里的 HTTP(S)_PROXY 改道。
    return urllib.request.build_opener(urllib.request.ProxyHandler({}), _NoRedirect)


def validate_base_url(base: str) -> str:
    """SSRF 防护：https + 固定 host + 公网 DNS，拒绝私网/环回/保留地址。"""
    parsed = urlparse(base)
    if parsed.scheme != "https":
        raise ValueError("ARK_BASE_URL 仅允许 https")
    if parsed.hostname != ALLOWED_HOST:
        raise ValueError(f"ARK_BASE_URL host 不在白名单：{parsed.hostname}")
    if not (parsed.path or "").startswith("/api/coding/v3"):
        raise ValueError("ARK_BASE_URL 路径需以 /api/coding/v3 开头")
    try:
        port = parsed.port or 443
    except ValueError as exc:
        raise ValueError(f"ARK_BASE_URL 端口非法：{base}") from exc
    try:
        infos = socket.getaddrinfo(parsed.hostname, port, proto=socket.IPPROTO_TCP)
    except socket.gaierror as exc:
        raise ValueError(f"ARK_BASE_URL 解析失败：{parsed.hostname}") from exc
    for *_, sockaddr in infos:
        ip = ipaddress.ip_address(sockaddr[0])
        if (
            not ip.is_global
            or ip.is_private
            or ip.is_loopback
            or ip.is_link_local
            or ip.is_reserved
            or ip.is_multicast
            or ip.is_unspecified
        ):
            raise ValueError(f"ARK_BASE_URL 解析到非公网地址：{parsed.hostname} -> {ip}")
    return base.rstrip("/")


def count_cjk(text: str) -> int:
    return sum(1 for ch in text if "\u4e00" <= ch <= "\u9fff")


def count_sentences(text: str) -> int:
    return sum(text.count(mark) for mark in "。！？!?")


_FENCE_START = re.compile(r"^```[a-zA-Z0-9_-]*\s*")
_FENCE_END = re.compile(r"\s*```$")
_TRAILING_META = re.compile(r"[（(]\s*(?:全文|共|约)?\s*\d+\s*字\s*[）)]\s*$")


def clean_content(text: str) -> str:
    """剥掉代码围栏与豆包常见的段尾自报字数。"""
    text = _TRAILING_META.sub("", text.strip()).strip()
    text = _FENCE_START.sub("", text)
    text = _FENCE_END.sub("", text).strip()
    return text


def body_ok(text: str) -> bool:
    return MIN_CJK <= count_cjk(text) <= MAX_CJK and count_sentences(text) >= 1


def retry_delay(attempt: int, headers: object | None = None) -> float:
    if headers is not None:
        raw = getattr(headers, "get", lambda _name: None)("Retry-After")
        if raw:
            try:
                return min(60.0, max(0.0, float(raw)))
            except ValueError:
                pass
    return min(30.0, float(2**attempt) + secrets.randbelow(1000) / 1000)


def post_chat(base: str, key: str, payload: dict, timeout: float) -> dict:
    req = urllib.request.Request(
        f"{base}/chat/completions",
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {key}",
            "Content-Type": "application/json",
        },
        method="POST",
    )
    with _opener().open(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


def sample_path(loop: int, prompt_idx: int, topic: str, prefix: str = "arklite") -> str:
    return f"texts/{prefix}-{loop:03d}-{prompt_idx:02d}-{topic}.txt"


def provenance(
    model: str,
    date: str,
    style: str,
    prompt_idx: int,
    loop: int,
    temperature: float,
    top_p: float,
    max_tokens: int,
) -> str:
    return (
        f"<!-- 平台: 火山引擎方舟(API) | 模型: {model} | 日期: {date} | "
        f"文体: {style} | 提示词: {prompt_idx} | 轮次: {loop:03d} | "
        f"搜索: 关闭（API 无联网搜索） | 思考: 关闭 | "
        f"参数: temperature={temperature}, top_p={top_p}, max_tokens={max_tokens} -->"
    )


def generate_one(
    task: dict,
    *,
    base: str,
    key: str,
    model: str,
    temperature: float,
    top_p: float,
    max_tokens: int,
    timeout: float,
    attempts: int,
) -> dict:
    started = time.monotonic()
    last_error = ""
    token_budget = max_tokens
    last_usage: dict = {}
    for attempt in range(1, attempts + 1):
        payload = {
            "model": model,
            "messages": [{"role": "user", "content": task["prompt"]}],
            "max_tokens": token_budget,
            "temperature": temperature,
            "top_p": top_p,
            "thinking": {"type": "disabled"},
        }
        try:
            data = post_chat(base, key, payload, timeout)
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", "replace")
            last_error = f"HTTP {exc.code}: {detail[:300]}"
            if exc.code == 429 or 500 <= exc.code < 600:
                time.sleep(retry_delay(attempt, exc.headers))
                continue
            break
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
            last_error = f"{type(exc).__name__}: {exc}"
            time.sleep(retry_delay(attempt))
            continue

        choice = (data.get("choices") or [{}])[0]
        message = choice.get("message") or {}
        content = clean_content(message.get("content") or "")
        last_usage = data.get("usage") or {}
        if body_ok(content):
            return {
                "status": "ok",
                "loop": task["loop"],
                "prompt_idx": task["prompt_idx"],
                "style": task["style"],
                "topic": task["topic"],
                "model": model,
                "file": sample_path(
                    task["loop"],
                    task["prompt_idx"],
                    task["topic"],
                    prefix=task.get("prefix", "arklite"),
                ),
                "sha256": hashlib.sha256(content.encode("utf-8")).hexdigest(),
                "cjk": count_cjk(content),
                "sentences": count_sentences(content),
                "finish_reason": choice.get("finish_reason"),
                "usage": last_usage,
                "attempts": attempt,
                "elapsed_s": round(time.monotonic() - started, 2),
                "created_at": dt.datetime.now().isoformat(timespec="seconds"),
                "error": None,
                "_body": content,
            }
        last_error = (
            f"正文不合格：cjk={count_cjk(content)} sentences={count_sentences(content)} "
            f"finish={choice.get('finish_reason')}"
        )
        if choice.get("finish_reason") == "length":
            token_budget = min(MAX_RETRY_TOKENS, int(token_budget * 1.5))
        time.sleep(0.5)

    return {
        "status": "failed",
        "loop": task["loop"],
        "prompt_idx": task["prompt_idx"],
        "style": task["style"],
        "topic": task["topic"],
        "model": model,
        "file": None,
        "sha256": None,
        "cjk": None,
        "sentences": None,
        "finish_reason": None,
        "usage": last_usage,
        "attempts": attempts,
        "elapsed_s": round(time.monotonic() - started, 2),
        "created_at": dt.datetime.now().isoformat(timespec="seconds"),
        "error": last_error,
    }


def build_tasks(loops: int, pack: str = "general") -> list[dict]:
    prompts, suffixes = PROMPT_PACKS[pack]
    tasks = []
    for loop in range(1, loops + 1):
        prompt_topic = ACADEMIC_TOPICS[(loop - 1) % len(ACADEMIC_TOPICS)]
        for prompt_idx, (style, prompt) in enumerate(prompts, 1):
            text = prompt.format(topic=prompt_topic) if pack == "academic" else prompt
            tasks.append(
                {
                    "loop": loop,
                    "prompt_idx": prompt_idx,
                    "style": style,
                    "prompt": text,
                    "topic": suffixes[style],
                    "prefix": PACK_PREFIX[pack],
                    "pack": pack,
                }
            )
    return tasks


def load_done(manifest_path: Path, out_dir: Path) -> set[tuple[int, int]]:
    if not manifest_path.exists():
        return set()
    latest: dict[tuple[int, int], dict] = {}
    for line in manifest_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError:
            continue
        latest[(int(record.get("loop", 0)), int(record.get("prompt_idx", 0)))] = record
    return {
        key
        for key, record in latest.items()
        if record.get("status") == "ok"
        and record.get("file")
        and (out_dir / str(record["file"])).exists()
    }


def write_sample(
    out_dir: Path,
    record: dict,
    body: str,
    date: str,
    temperature: float,
    top_p: float,
    max_tokens: int,
) -> None:
    path = out_dir / str(record["file"])
    path.parent.mkdir(parents=True, exist_ok=True)
    header = provenance(
        str(record["model"]),
        date,
        str(record["style"]),
        int(record["prompt_idx"]),
        int(record["loop"]),
        temperature,
        top_p,
        max_tokens,
    )
    tmp = path.with_name(f"{path.name}.tmp-{os.getpid()}-{threading.get_ident()}")
    tmp.write_text(f"{header}\n\n{body}\n", encoding="utf-8")
    tmp.replace(path)


def append_manifest(path: Path, record: dict, lock: threading.Lock) -> None:
    line = json.dumps(record, ensure_ascii=False, sort_keys=True)
    with lock:
        with path.open("a", encoding="utf-8") as fh:
            fh.write(line + "\n")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--loops", type=int, default=100, help="每个提示词循环次数，默认 100")
    parser.add_argument("--workers", type=int, default=6, help="并发请求数，默认 6")
    parser.add_argument("--attempts", type=int, default=3, help="单条最大尝试次数，默认 3")
    parser.add_argument("--max-tokens", type=int, default=800, help="首次生成 max_tokens")
    parser.add_argument("--temperature", type=float, default=1.0, help="采样温度")
    parser.add_argument("--top-p", type=float, default=0.95, help="top_p")
    parser.add_argument("--timeout", type=float, default=120.0, help="单请求超时秒数")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="只允许豆包 Lite 模型")
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL, help="方舟 coding plan 端点")
    parser.add_argument(
        "--pack",
        choices=sorted(PROMPT_PACKS),
        default="general",
        help="提示词包：general（通用 5 题）或 academic（论文章节 7 题）",
    )
    parser.add_argument(
        "--out-dir",
        default="",
        help="输出目录（相对项目根目录）；留空按提示词包选默认目录",
    )
    parser.add_argument(
        "--prefix",
        default="",
        help="样本文件名前缀；留空按提示词包选 arklite/arkacad",
    )
    parser.add_argument("--date", default=dt.date.today().isoformat(), help="溯源头日期")
    parser.add_argument("--limit", type=int, default=0, help="只跑前 N 条，0 表示全部")
    parser.add_argument("--dry-run", action="store_true", help="只打印计划，不发请求")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if "lite" not in args.model.lower():
        raise SystemExit("本脚本只允许豆包 Lite 模型，避免误用高价模型")
    if args.loops < 1 or args.workers < 1 or args.attempts < 1:
        raise SystemExit("--loops/--workers/--attempts 必须 >= 1")
    base = validate_base_url(args.base_url)
    key = os.environ.get("ARK_API_KEY", "").strip()
    if not key:
        raise SystemExit("缺少环境变量 ARK_API_KEY（只从环境变量读取，不写进源码）")

    out_dir = Path(args.out_dir or PACK_OUT_DIR[args.pack])
    if not out_dir.is_absolute():
        out_dir = ROOT / out_dir
    manifest_path = out_dir / "manifest.jsonl"
    out_dir.mkdir(parents=True, exist_ok=True)
    prefix = args.prefix or PACK_PREFIX[args.pack]
    (out_dir / "run-config.json").write_text(
        json.dumps(
            {
                "model": args.model,
                "base_url": base,
                "pack": args.pack,
                "prefix": prefix,
                "loops": args.loops,
                "workers": args.workers,
                "attempts": args.attempts,
                "max_tokens": args.max_tokens,
                "temperature": args.temperature,
                "top_p": args.top_p,
                "date": args.date,
                "started_at": dt.datetime.now().isoformat(timespec="seconds"),
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    tasks = build_tasks(args.loops, args.pack)
    for task in tasks:
        task["prefix"] = prefix
    done = load_done(manifest_path, out_dir)
    pending = [task for task in tasks if (task["loop"], task["prompt_idx"]) not in done]
    if args.limit:
        pending = pending[: args.limit]
    print(
        f"计划 {len(tasks)} 条（{args.pack}），已完成 {len(done)} 条，"
        f"本次待跑 {len(pending)} 条；模型 {args.model}，并发 {args.workers}",
        flush=True,
    )
    if args.dry_run or not pending:
        return 0

    lock = threading.Lock()
    ok = failed = duplicates = 0
    total_tokens = 0
    seen_shas: dict[str, tuple[int, int]] = {}
    started = time.monotonic()
    completed = 0
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {
            pool.submit(
                generate_one,
                task,
                base=base,
                key=key,
                model=args.model,
                temperature=args.temperature,
                top_p=args.top_p,
                max_tokens=args.max_tokens,
                timeout=args.timeout,
                attempts=args.attempts,
            ): task
            for task in pending
        }
        for future in concurrent.futures.as_completed(futures):
            record = future.result()
            completed += 1
            total_tokens += int((record.get("usage") or {}).get("total_tokens") or 0)
            if record["status"] == "ok":
                body = record.pop("_body")
                sha = str(record["sha256"])
                if sha in seen_shas:
                    record["duplicate"] = True
                    record["duplicate_of"] = list(seen_shas[sha])
                    duplicates += 1
                else:
                    record["duplicate"] = False
                    seen_shas[sha] = (int(record["loop"]), int(record["prompt_idx"]))
                try:
                    write_sample(
                        out_dir,
                        record,
                        body,
                        args.date,
                        args.temperature,
                        args.top_p,
                        args.max_tokens,
                    )
                    ok += 1
                except OSError as exc:
                    record["status"] = "failed"
                    record["error"] = f"写文件失败：{exc}"
                    failed += 1
            else:
                failed += 1
                print(
                    f"  [失败] loop={record['loop']:03d} prompt={record['prompt_idx']} "
                    f"{record.get('error')}",
                    flush=True,
                )
            append_manifest(manifest_path, record, lock)
            if completed % 10 == 0 or completed == len(pending):
                elapsed = time.monotonic() - started
                rate = completed / elapsed if elapsed else 0
                print(
                    f"[{completed}/{len(pending)}] ok={ok} fail={failed} dup={duplicates} "
                    f"tokens={total_tokens} {rate:.2f} 条/秒",
                    flush=True,
                )

    elapsed = time.monotonic() - started
    print(
        f"完成：成功 {ok}，失败 {failed}，重复 {duplicates}，"
        f"tokens {total_tokens}，耗时 {elapsed:.1f}s",
        flush=True,
    )
    print(f"输出目录：{out_dir}", flush=True)
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
