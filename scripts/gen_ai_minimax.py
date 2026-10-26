"""MiniMax（M2.5，Anthropic 兼容端点）批量生成新语域 AI 语料。

v9 扩容轮：现有 19 个生成器集中在科技/学术语域，本脚本为四个新语域
（news 新闻报道 / wiki 百科说明 / gov 政策公文 / literary 文学散文）
各配一组提示词包，扩大跨语域的生成器覆盖。输出到暂存目录（默认
calibration/ai_register_mm/），不直接进 calibration/ai/：单一生成器样本
直接并主语料会压垮按生成器分层的切分与 LOGO-生成器 CV，需先抽样再并入。

凭据纪律：密钥只从环境变量 MINIMAX_API_KEY 读取；源码、参数和输出都不落密钥。
网络安全：https + 固定 host 白名单（api.minimaxi.com）+ DNS 公网校验 +
禁用重定向 + 显式禁用环境代理。

    MINIMAX_API_KEY=... uv run python scripts/gen_ai_minimax.py --loops 6
    MINIMAX_API_KEY=... uv run python scripts/gen_ai_minimax.py --packs news,wiki --loops 2 --limit 4  # 冒烟
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
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path
from urllib.parse import urlparse

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_BASE_URL = "https://api.minimaxi.com/anthropic"
ALLOWED_HOST = "api.minimaxi.com"
DEFAULT_MODEL = "MiniMax-M2.5"
MIN_CJK = 100
MAX_CJK = 450
MAX_RETRY_TOKENS = 4000

NEWS_TOPICS = [
    "全国碳排放权交易市场扩容后首批行业完成首月交易",
    "某中部省会城市开通第四条地铁线路",
    "国产大飞机新增两条高原航线",
    "多地下调首套房贷款利率",
    "一款国产抗癌药进入国家医保目录",
    "大型进出口商品交易会在广州开幕",
    "国内首个深水深层大气田建成投产",
    "三北地区启动新一轮荒漠化治理工程",
    "多所高校新增人工智能本科专业",
    "新能源汽车零售渗透率首次过半",
    "某省启动县域医共体全覆盖建设",
    "全国铁路实行新列车运行图",
]
NEWS_PROMPTS = [
    (
        "新闻-时政",
        "请以新闻媒体的报道口吻，写一段150~250字的中文报道，主题：{topic}。"
        "需包含时间、地点、事件经过与意义，语气客观，不要分点，不要标题，写成一整段。",
    ),
    (
        "新闻-经济",
        "请以财经媒体的报道口吻，写一段150~250字的中文报道，主题：{topic}。"
        "交代背景、数据或进展与影响，语气客观克制，不要分点，不要标题，写成一整段。",
    ),
]

WIKI_TOPICS = [
    "深海热液喷口生态系统",
    "mRNA疫苗的作用原理",
    "量子纠缠的实验验证",
    "IPv6 的地址结构",
    "青藏高原的隆升过程",
    "碳-14 测年法",
    "共振现象及其应用",
    "洋流对气候的影响",
    "超导体的迈斯纳效应",
    "昆虫复眼的结构",
    "土壤盐碱化的成因与治理",
    "冰川期形成的主要假说",
]
WIKI_PROMPTS = [
    (
        "百科-科技",
        "请以百科全书的词条口吻，写一段150~250字的中文介绍，词条：{topic}。"
        "客观说明其定义、原理或构成与主要应用，不要分点，不要标题，不要加参考文献，写成一整段。",
    ),
    (
        "百科-自然",
        "请以科普百科的词条口吻，写一段150~250字的中文介绍，词条：{topic}。"
        "客观说明其成因、机制与影响，语言平实准确，不要分点，不要标题，写成一整段。",
    ),
]

GOV_TOPICS = [
    "加强县城基层医疗卫生服务能力建设",
    "推进高标准农田建设的政策措施",
    "制造业数字化转型的支持政策",
    "城镇老旧小区改造的实施路径",
    "职业教育产教融合的推进措施",
    "生态保护补偿机制的完善方向",
    "公立医院高质量发展的重点任务",
    "县域商业体系建设的推进安排",
    "知识产权保护体系的建设重点",
    "养老服务体系建设的主要举措",
    "营商环境持续优化的重点任务",
    "粮食安全保障体系的完善措施",
]
GOV_PROMPTS = [
    (
        "公文-政策解读",
        "请以政策解读公文的口吻，写一段150~250字的中文说明，主题：{topic}。"
        "说明政策目标、主要措施与预期成效，用语规范稳妥，不要分点，不要标题，写成一整段。",
    ),
    (
        "公文-工作部署",
        "请以机关工作部署的口吻，写一段150~250字的中文说明，主题：{topic}。"
        "按照总体要求、重点任务、保障措施的顺序概述，用语规范，不要分点，不要标题，写成一整段。",
    ),
]

LIT_TOPICS = [
    "深秋清晨的火车站月台",
    "老城区雨后的青石板巷",
    "夏夜外婆家院子里的乘凉",
    "离开家乡那天的长途汽车",
    "冬夜书房里的台灯",
    "海边渔村清晨的鱼市",
    "山间古寺的黄昏",
    "旧书店里偶然翻到的一封旧信",
    "麦收季节的平原",
    "南方梅雨季的屋檐",
    "清晨五点的早点铺",
    "绿皮火车穿过隧道的那一刻",
]
LIT_PROMPTS = [
    (
        "散文-叙事",
        "请以文学散文的笔法，写一段150~250字的中文文字，主题：{topic}。"
        "注重细节、意象与情感的自然流露，避免概括性总结，不要分点，不要标题，写成一整段。",
    ),
    (
        "散文-写景",
        "请以抒情散文的笔法，写一段150~250字的中文文字，主题：{topic}。"
        "以具体物象与感官细节为主，克制抒情，不要分点，不要标题，写成一整段。",
    ),
]


def _pack_spec(pack: str) -> tuple[list[tuple[str, str]], list[str], str]:
    return {
        "news": (NEWS_PROMPTS, NEWS_TOPICS, "news"),
        "wiki": (WIKI_PROMPTS, WIKI_TOPICS, "wiki"),
        "gov": (GOV_PROMPTS, GOV_TOPICS, "gov"),
        "literary": (LIT_PROMPTS, LIT_TOPICS, "lit"),
    }[pack]


PROMPT_PACKS = {name: _pack_spec(name) for name in ("news", "wiki", "gov", "literary")}
PACK_OUT_DIR = "calibration/ai_register_mm"


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
        raise ValueError("BASE_URL 仅允许 https")
    if parsed.hostname != ALLOWED_HOST:
        raise ValueError(f"BASE_URL host 不在白名单：{parsed.hostname}")
    try:
        port = parsed.port or 443
    except ValueError as exc:
        raise ValueError(f"BASE_URL 端口非法：{base}") from exc
    try:
        infos = socket.getaddrinfo(parsed.hostname, port, proto=socket.IPPROTO_TCP)
    except socket.gaierror as exc:
        raise ValueError(f"BASE_URL 解析失败：{parsed.hostname}") from exc
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
            raise ValueError(f"BASE_URL 解析到非公网地址：{parsed.hostname} -> {ip}")
    return base.rstrip("/")


def count_cjk(text: str) -> int:
    return sum(1 for ch in text if "\u4e00" <= ch <= "\u9fff")


def count_sentences(text: str) -> int:
    return sum(text.count(mark) for mark in "。！？!?")


_FENCE_START = re.compile(r"^```[a-zA-Z0-9_-]*\s*")
_FENCE_END = re.compile(r"\s*```$")
_TRAILING_META = re.compile(r"[（(]\s*(?:全文|共|约)?\s*\d+\s*字\s*[）)]\s*$")


def clean_content(text: str) -> str:
    text = _TRAILING_META.sub("", text.strip()).strip()
    text = _FENCE_START.sub("", text)
    return _FENCE_END.sub("", text).strip()


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


def post_messages(base: str, key: str, payload: dict, timeout: float) -> dict:
    req = urllib.request.Request(
        f"{base}/v1/messages",
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {key}",
            "Content-Type": "application/json",
            "anthropic-version": "2023-06-01",
        },
        method="POST",
    )
    with _opener().open(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


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
            "max_tokens": token_budget,
            "temperature": temperature,
            "top_p": top_p,
            "messages": [{"role": "user", "content": task["prompt"]}],
        }
        try:
            data = post_messages(base, key, payload, timeout)
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

        # Anthropic 兼容响应：只取 text 块（M2.5 的 thinking 块不进语料）
        content = "".join(
            block.get("text", "") for block in data.get("content", []) if block.get("type") == "text"
        )
        content = clean_content(content)
        last_usage = data.get("usage") or {}
        if body_ok(content):
            return {
                "status": "ok",
                "loop": task["loop"],
                "prompt_idx": task["prompt_idx"],
                "style": task["style"],
                "pack": task["pack"],
                "model": model,
                "file": task["file"],
                "sha256": hashlib.sha256(content.encode("utf-8")).hexdigest(),
                "cjk": count_cjk(content),
                "sentences": count_sentences(content),
                "finish_reason": data.get("stop_reason"),
                "usage": last_usage,
                "attempts": attempt,
                "elapsed_s": round(time.monotonic() - started, 2),
                "created_at": dt.datetime.now().isoformat(timespec="seconds"),
                "error": None,
                "_body": content,
            }
        last_error = (
            f"正文不合格：cjk={count_cjk(content)} sentences={count_sentences(content)} "
            f"finish={data.get('stop_reason')}"
        )
        if data.get("stop_reason") == "max_tokens":
            token_budget = min(MAX_RETRY_TOKENS, int(token_budget * 1.5))
        time.sleep(0.5)

    return {
        "status": "failed",
        "loop": task["loop"],
        "prompt_idx": task["prompt_idx"],
        "style": task["style"],
        "pack": task["pack"],
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


def build_tasks(loops: int, packs: list[str], prefix: str) -> list[dict]:
    tasks = []
    for pack in packs:
        prompts, topics, slug = PROMPT_PACKS[pack]
        for topic_idx, topic in enumerate(topics, 1):
            for prompt_idx, (style, prompt) in enumerate(prompts, 1):
                for loop in range(1, loops + 1):
                    tasks.append(
                        {
                            "loop": loop,
                            "prompt_idx": prompt_idx,
                            "style": style,
                            "pack": pack,
                            "prompt": prompt.format(topic=topic),
                            "file": f"texts/{prefix}-{pack}{topic_idx:02d}-{prompt_idx:02d}-{loop:03d}-{slug}.txt",
                        }
                    )
    return tasks


def load_done(manifest_path: Path, out_dir: Path) -> set[str]:
    if not manifest_path.exists():
        return set()
    done = set()
    for line in manifest_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError:
            continue
        if (
            record.get("status") == "ok"
            and record.get("file")
            and (out_dir / str(record["file"])).exists()
        ):
            done.add(str(record["file"]))
    return done


def write_sample(out_dir: Path, record: dict, body: str, date: str) -> None:
    path = out_dir / str(record["file"])
    path.parent.mkdir(parents=True, exist_ok=True)
    usage = record.get("usage") or {}
    header = (
        f"<!-- 平台: MiniMax(API) | 模型: {record['model']} | 日期: {date} | "
        f"文体: {record['style']} | 语域: {record['pack']} | "
        f"提示词: {record['prompt_idx']} | 轮次: {record['loop']:03d} | "
        f"搜索: 关闭（API 无联网搜索） | "
        f"参数: temperature=1.0, top_p=0.95, max_tokens=3000, "
        f"usage={usage.get('input_tokens')}+{usage.get('output_tokens')}tok -->"
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
    parser.add_argument("--loops", type=int, default=6, help="每个（主题×提示词）的重复轮数，默认 6")
    parser.add_argument("--workers", type=int, default=5, help="并发请求数，默认 5")
    parser.add_argument("--attempts", type=int, default=3, help="单条最大尝试次数，默认 3")
    parser.add_argument("--max-tokens", type=int, default=3000, help="首次生成 max_tokens（含思考预算）")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL)
    parser.add_argument("--packs", default="news,wiki,gov,literary", help="逗号分隔的提示词包")
    parser.add_argument("--out-dir", default=PACK_OUT_DIR, help="输出目录（相对项目根或绝对路径）")
    parser.add_argument("--prefix", default="mm25")
    parser.add_argument("--date", default=dt.date.today().isoformat(), help="溯源头日期")
    parser.add_argument("--limit", type=int, default=0, help="只跑前 N 条，0 表示全部")
    parser.add_argument("--dry-run", action="store_true", help="只打印计划，不发请求")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    packs = [p.strip() for p in args.packs.split(",") if p.strip()]
    for pack in packs:
        if pack not in PROMPT_PACKS:
            raise SystemExit(f"未知提示词包：{pack}（可选：{', '.join(PROMPT_PACKS)}）")
    if args.loops < 1 or args.workers < 1 or args.attempts < 1:
        raise SystemExit("--loops/--workers/--attempts 必须 >= 1")
    base = validate_base_url(args.base_url)
    key = os.environ.get("MINIMAX_API_KEY", "").strip()
    if not key:
        raise SystemExit("缺少环境变量 MINIMAX_API_KEY（只从环境变量读取，不写进源码）")

    out_dir = Path(args.out_dir)
    if not out_dir.is_absolute():
        out_dir = ROOT / out_dir
    manifest_path = out_dir / "manifest.jsonl"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "run-config.json").write_text(
        json.dumps(
            {
                "model": args.model,
                "base_url": base,
                "packs": packs,
                "prefix": args.prefix,
                "loops": args.loops,
                "workers": args.workers,
                "attempts": args.attempts,
                "max_tokens": args.max_tokens,
                "date": args.date,
                "started_at": dt.datetime.now().isoformat(timespec="seconds"),
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    tasks = build_tasks(args.loops, packs, args.prefix)
    done = load_done(manifest_path, out_dir)
    pending = [task for task in tasks if task["file"] not in done]
    if args.limit:
        pending = pending[: args.limit]
    print(
        f"计划 {len(tasks)} 条（{','.join(packs)}），已完成 {len(done)} 条，"
        f"本次待跑 {len(pending)} 条；模型 {args.model}，并发 {args.workers}",
        flush=True,
    )
    if args.dry_run or not pending:
        return 0

    lock = threading.Lock()
    ok = failed = duplicates = 0
    total_tokens = 0
    seen_shas: set[str] = set()
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
                temperature=1.0,
                top_p=0.95,
                max_tokens=args.max_tokens,
                timeout=180.0,
                attempts=args.attempts,
            ): task
            for task in pending
        }
        for future in concurrent.futures.as_completed(futures):
            record = future.result()
            completed += 1
            usage = record.get("usage") or {}
            total_tokens += int(usage.get("input_tokens") or 0) + int(usage.get("output_tokens") or 0)
            if record["status"] == "ok":
                body = record.pop("_body")
                if record["sha256"] in seen_shas:
                    record["duplicate"] = True
                    duplicates += 1
                else:
                    record["duplicate"] = False
                    seen_shas.add(record["sha256"])
                    try:
                        write_sample(out_dir, record, body, args.date)
                        ok += 1
                    except OSError as exc:
                        record["status"] = "failed"
                        record["error"] = f"写文件失败：{exc}"
                        failed += 1
            else:
                failed += 1
                print(
                    f"  [失败] {record.get('pack')} loop={record['loop']:03d} "
                    f"prompt={record['prompt_idx']} {record.get('error')}",
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
        f"完成：成功 {ok}，失败 {failed}，重复 {duplicates}，tokens {total_tokens}，"
        f"耗时 {elapsed:.1f}s",
        flush=True,
    )
    print(f"输出目录：{out_dir}", flush=True)
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
