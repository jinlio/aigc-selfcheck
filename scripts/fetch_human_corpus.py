"""人写语料抓取：四道保真关，确保「人写」成立。

第一道 时间截点：只收发布日期 ≤ 2021-12-31 的文本；无可靠日期的来源直接弃用
        （公版书补充层除外——年代本身即证明，单独标记 layer=classic 不与主语料混算）。
第二道 作者手写史：种子由使用者自备，须为有多年手写记录的作者/站点，每作者
        段落上限 5 段。公开版不内置任何站点专属的发现逻辑。
第三道 provenance：每段记录 URL、发布日期、抓取时间、sha256 → manifest.jsonl（入库可审计）。
        manifest 与正文都含第三方来源信息，保存在私有训练归档，不入公开仓库。
第四道 双重抽检：统计预检剔除 AI 典型分布段落（仅作剔除依据，不作准入依据），
        并生成 REVIEW.md 供人工逐段审读勾选。

网络安全：safe_fetch 强制 https + 主机白名单 + 解析 IP 私网/环回校验 + 手动逐跳重定向。

用法：
    uv run python scripts/fetch_human_corpus.py --only-seeds --seeds-file 种子.json
种子文件为具体文章 URL 清单（必须自带可靠发布日期）：
    [{"url": "...", "date": "2020-05-01", "author": "某作者", "source": "example.com"}]
默认白名单只放行 Project Gutenberg 与 Wayback Machine；使用其他种子站点时，
用 --allow-host 显式放行（可重复）：
    uv run python scripts/fetch_human_corpus.py --only-seeds \
        --allow-host www.example.com --seeds-file 种子.json
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import http.client
import ipaddress
import json
import re
import socket
import ssl
import sys
import urllib.error
import urllib.request
from datetime import UTC, datetime
from html.parser import HTMLParser
from pathlib import Path
from urllib.parse import urlparse

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from aigc_selfcheck.features import (  # noqa: E402
    cjk_count,
    connective_density,
    sentence_length_variance,
)
from aigc_selfcheck.preprocess import split_sentences  # noqa: E402

RAW_DIR = Path("calibration/raw/human")
MANIFEST = Path("calibration/manifest.jsonl")
REVIEW = Path("calibration/raw/REVIEW.md")
SEEDS_FILE = Path("calibration/seeds.json")

DATE_CUTOFF = "2021-12-31"  # 第一道保真关
PER_AUTHOR_CAP = 5
PER_POST_CAP = 2  # 单篇文章取段上限：避免一篇文章主导该作者的风格样本
MIN_CJK, MAX_CJK = 100, 400
MIN_SENTS = 3
MAX_REDIRECTS = 5
UA = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/126.0 Safari/537.36"
)

# 第二道保真关：来源准入白名单。公开版默认只放行公版书与 Wayback 存档
# （Wayback 快照时间戳即第三方「早于 LLM 时代」的存在证明）；其他种子站点
# 由使用者用 --allow-host 显式放行，并自行核验其手写史与内容授权。
ALLOWED_HOSTS = {
    "gutenberg.org",
    "www.gutenberg.org",
    "web.archive.org",
}
# 运行时经 --allow-host 追加；safe_fetch 的生效白名单 = 内置 ∪ 追加。
EXTRA_ALLOWED_HOSTS: set[str] = set()


class FetchError(Exception):
    pass


def _validate_url(url: str, allowed_hosts: set[str] | None = None) -> None:
    parsed = urlparse(url)
    if parsed.scheme != "https":
        raise FetchError(f"拒绝非 https 地址：{url}")
    allowed = (ALLOWED_HOSTS | EXTRA_ALLOWED_HOSTS) if allowed_hosts is None else allowed_hosts
    if parsed.hostname not in allowed:
        raise FetchError(f"主机不在白名单：{parsed.hostname}")
    if not _egress_addresses(parsed.hostname):
        raise FetchError(f"无可用出口地址：{parsed.hostname}")


def _egress_addresses(host: str) -> list[str]:
    """按优先级返回可建连的 (family, ip)：公网 IPv4 → 公网 IPv6 → 沙箱出口段。

    SSRF 高危段（环回 / 链路本地 / RFC1918 私网 / 保留 / 组播）在任何层级都被拒绝。
    第三层是本开发环境特有的出口：部分外网目标（如 web.archive.org）的真实 IP
    直连不可达，只能经沙箱代理的 2001::/32（Teredo 段）合成地址出网；该段
    无法寻址本机/内网服务，且主机白名单仍然生效，故作为最后回退是安全的。
    """
    try:
        infos = socket.getaddrinfo(host, 443, proto=socket.IPPROTO_TCP)
    except socket.gaierror as exc:
        raise FetchError(f"DNS 解析失败：{host}") from exc
    v4: list[str] = []
    v6: list[str] = []
    synth: list[str] = []
    for info in infos:
        ip = ipaddress.ip_address(info[4][0])
        if (
            ip.is_loopback
            or ip.is_link_local
            or ip.is_reserved
            or ip.is_multicast
            or ip.is_unspecified
            or (ip.version == 4 and ip.is_private)
        ):
            raise FetchError(f"目标解析到内网/保留地址，拒绝：{ip}")
        if ip.version == 4:
            v4.append(str(ip))
        elif ip.is_global:
            v6.append(str(ip))
        elif ip in ipaddress.ip_network("2001::/32"):
            synth.append(str(ip))
    return v4 + v6 + synth


class _PinnedHTTPSConnection(http.client.HTTPSConnection):
    """建连时重新解析并按 _egress_addresses 顺序尝试（防 rebinding + 环境出口回退）。"""

    def connect(self) -> None:
        last: OSError | None = None
        for addr in _egress_addresses(self.host):
            try:
                self.sock = self._create_connection(
                    (addr, self.port or 443), self.timeout, source_address=self.source_address
                )
                break
            except OSError as exc:
                last = exc
        else:
            raise last if last else OSError("无可用地址")
        try:
            self.sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        except OSError:  # noqa: PERF203 —— 与标准库 connect 同款容错
            pass
        if self._tunnel_host:
            self._tunnel()
        self.sock = self._context.wrap_socket(self.sock, server_hostname=self._tunnel_host or self.host)


class _PinnedHTTPSHandler(urllib.request.HTTPSHandler):
    """建连只经 _PinnedHTTPSConnection（IPv4 钉扎）；证书校验沿用 context 默认。"""

    def https_open(self, req):  # noqa: D102
        kwargs = {"context": self._context}
        if (chk := getattr(self, "_check_hostname", None)) is not None:
            kwargs["check_hostname"] = chk
        return self.do_open(_PinnedHTTPSConnection, req, **kwargs)


def _ssl_context() -> ssl.SSLContext:
    """优先用系统信任库（macOS /etc/ssl/cert.pem）——部分网络环境有 MITM 中间证书。"""
    for candidate in ("/etc/ssl/cert.pem", "/etc/ssl/certs/ca-certificates.crt"):
        if Path(candidate).exists():
            return ssl.create_default_context(cafile=candidate)
    return ssl.create_default_context()


def _decompress(raw: bytes, content_encoding: str) -> bytes:
    """按 Content-Encoding 解压；部分 CDN 即使没收到 Accept-Encoding 也回 gzip。"""
    encoding = (content_encoding or "").lower()
    if "gzip" in encoding:
        try:
            return gzip.decompress(raw)
        except OSError as exc:
            raise FetchError("响应声明 gzip 但解压失败") from exc
    if "deflate" in encoding:
        import zlib

        try:
            return zlib.decompress(raw)
        except zlib.error:
            try:
                return zlib.decompress(raw, -zlib.MAX_WBITS)
            except zlib.error as exc:
                raise FetchError("响应声明 deflate 但解压失败") from exc
    return raw


def _fetch_bytes(
    url: str, timeout: int = 20, allowed_hosts: set[str] | None = None
) -> tuple[bytes, str]:
    """逐跳校验的受限抓取：https + 白名单 + 私网 IP 拦截 + 手动重定向。"""

    class _NoRedirect(urllib.request.HTTPRedirectHandler):
        def redirect_request(self, req, fp, code, msg, headers, newurl):  # noqa: ARG002
            return None  # 手动处理重定向

    opener = urllib.request.build_opener(
        _PinnedHTTPSHandler(context=_ssl_context()), _NoRedirect
    )
    current = url
    for _ in range(MAX_REDIRECTS + 1):
        _validate_url(current, allowed_hosts)
        request = urllib.request.Request(current, headers={"User-Agent": UA})
        try:
            with opener.open(request, timeout=timeout) as resp:
                raw = _decompress(
                    resp.read(), resp.headers.get("Content-Encoding", "")
                )
                return raw, resp.headers.get("Content-Type", "")
        except urllib.error.HTTPError as exc:
            location = exc.headers.get("Location")
            if exc.code in (301, 302, 303, 307, 308) and location:
                current = urllib.request.urljoin(current, location)
                continue
            raise FetchError(f"HTTP {exc.code}: {current}") from exc
    raise FetchError(f"重定向次数超限：{url}")


def safe_fetch(
    url: str, timeout: int = 20, allowed_hosts: set[str] | None = None
) -> str:
    """抓取并解码文本；字符集按响应头 / 前 1KB meta 探测。"""
    raw, content_type = _fetch_bytes(url, timeout, allowed_hosts)
    return _decode_body(raw, content_type)


def safe_fetch_bytes(
    url: str, timeout: int = 30, allowed_hosts: set[str] | None = None
) -> bytes:
    """抓取二进制（PDF）；仍走同一套白名单 / 私网拦截 / 逐跳重定向校验。"""
    raw, _ = _fetch_bytes(url, timeout, allowed_hosts)
    return raw


def _decode_body(raw: bytes, content_type: str) -> str:
    """按 HTTP 头 / HTML meta 探测字符集；gb2312 一律升级 gb18030（超集）。"""
    encoding = "utf-8"
    if match := re.search(r"charset=([\w-]+)", content_type, re.I):
        encoding = match.group(1)
    else:
        if meta := re.search(br"""charset=["']?([\w-]+)""", raw[:1024]):
            encoding = meta.group(1).decode("ascii", "ignore")
    encoding = encoding.lower()
    if encoding in ("gb2312", "gbk"):
        encoding = "gb18030"
    return raw.decode(encoding, errors="replace")


def sha256_of(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


class _ParaExtractor(HTMLParser):
    """极简 <p> 文本提取器（不引入 bs4 依赖）。"""

    def __init__(self) -> None:
        super().__init__()
        self.paragraphs: list[str] = []
        self._buf: list[str] = []
        self._depth = 0
        self._skip = 0

    def handle_starttag(self, tag, attrs):
        if tag in ("script", "style", "pre", "code"):
            self._skip += 1
            return
        if tag == "p":
            self._depth += 1
            self._buf = []

    def handle_endtag(self, tag):
        if tag in ("script", "style", "pre", "code") and self._skip:
            self._skip -= 1
            return
        if tag == "p" and self._depth:
            self._depth -= 1
            text = "".join(self._buf).strip()
            if text:
                self.paragraphs.append(text)
            self._buf = []

    def handle_data(self, data):
        if not self._skip and self._depth:
            self._buf.append(data)


def extract_paragraphs(html: str) -> list[str]:
    parser = _ParaExtractor()
    try:
        parser.feed(html)
    except Exception:  # noqa: BLE001 —— 页面畸形时返回已提取部分
        pass
    return parser.paragraphs


# ── 统计预检（第四道关的一半；只用于剔除，不用于准入）───────────────────
def stats_precheck(text: str) -> str | None:
    """返回剔除原因；None 表示通过预检（仍需人工审读）。"""
    sentences = split_sentences(text)
    lengths = [len(s.text) for s in sentences]
    conn = connective_density(text)
    if conn > 12:  # 骨架词密度异常高：AI 典型分布
        return f"connective_density={conn:.1f} 异常"
    if len(lengths) >= 4 and sentence_length_variance(lengths) < 20:
        return f"句长方差过低 var={sentence_length_variance(lengths):.0f}"
    if re.search(r"作为一(个|名)[^，。]{2,12}[，,]", text) and "值得注意的是" in text:
        return "AI 模板句式"
    return None


# ── 种子源发现 ────────────────────────────────────────────────────────
def discover_gutenberg(max_books: int = 2) -> list[dict]:
    """公版书补充层：年代即证明，layer=classic（不与主语料混算）。"""
    seeds: list[dict] = []
    try:
        html = safe_fetch("https://www.gutenberg.org/browse/languages/zh")
    except Exception as exc:  # noqa: BLE001
        print(f"  [skip] gutenberg 不可达：{exc}")
        return seeds
    links = re.findall(r'href="(/ebooks/(\d+))"', html)
    seen: set[str] = set()
    for _path, book_id in links:
        if book_id in seen:
            continue
        seen.add(book_id)
        seeds.append(
            {
                # ebooks/*.txt.utf-8 会 302 到 http，被安全策略拒绝；直接取 https cache 文本。
                "url": f"https://www.gutenberg.org/cache/epub/{book_id}/pg{book_id}.txt",
                "date": None,
                "author": f"gutenberg-{book_id}",
                "source": "gutenberg.org",
                "layer": "classic",
            }
        )
        if len(seen) >= max_books:
            break
    return seeds


def load_seed_file(path: Path = SEEDS_FILE) -> list[dict]:
    """使用者自备的具体文章 URL 种子，必须自带可靠发布日期。"""
    if not path.exists():
        return []
    items = json.loads(path.read_text(encoding="utf-8"))
    kept = []
    for item in items:
        if not item.get("date"):
            print(f"  [gate1] 丢弃无日期种子：{item.get('url')}")
            continue
        if urlparse(item["url"]).hostname not in ALLOWED_HOSTS | EXTRA_ALLOWED_HOSTS:
            print(f"  [gate2] 丢弃白名单外种子：{item.get('url')}")
            continue
        kept.append(item)
    return kept


def gutenberg_paragraphs(text: str) -> list[str]:
    """纯文本公版书 → 段落（去掉 Gutenberg 头尾说明）。"""
    body = re.split(r"\*\*\* START OF .*?\*\*\*", text, flags=re.S)
    body = body[1] if len(body) > 1 else text
    body = re.split(r"\*\*\* END OF", body)[0]
    return [p.strip().replace("\n", "") for p in re.split(r"\n\s*\n", body) if p.strip()]


# ── 主流程 ────────────────────────────────────────────────────────────
def collect(
    existing_counts: dict[str, int],
    known_shas: set[str],
    existing_post_counts: dict[tuple[str, str], int] | None = None,
    *,
    only_seeds: bool = False,
    seeds_file: Path = SEEDS_FILE,
) -> list[dict]:
    if only_seeds:
        seeds = load_seed_file(seeds_file)
    else:
        seeds = load_seed_file(seeds_file) + discover_gutenberg()
    print(f"种子共 {len(seeds)} 条")
    records: list[dict] = []
    # 配额以 manifest 已有计数为基线：新增段与已入库段合计不超每作者/每篇上限
    per_author: dict[str, int] = dict(existing_counts)
    per_post: dict[tuple[str, str], int] = dict(existing_post_counts or {})
    for seed in seeds:
        if seed.get("layer") != "classic" and seed["date"] > DATE_CUTOFF:
            continue
        if per_post.get((seed["author"], seed["url"]), 0) >= PER_POST_CAP:
            continue  # 该篇跨次运行已取满，不再碰（否则每次运行都会从同一篇续取）
        paragraphs = seed.get("paragraphs")  # 种子可自带全文段落（如 RSS 全文），有则不再抓文章页
        if paragraphs is None:
            try:
                raw = safe_fetch(seed["url"])
            except Exception as exc:  # noqa: BLE001
                print(f"  [skip] {seed['url']}: {exc}")
                continue
            if seed["source"] == "gutenberg.org":
                paragraphs = gutenberg_paragraphs(raw)
            else:
                paragraphs = extract_paragraphs(raw)
        for para in paragraphs:
            cjk = cjk_count(para)
            if not (MIN_CJK <= cjk <= MAX_CJK) or len(split_sentences(para)) < MIN_SENTS:
                continue
            digest = sha256_of(para)
            if digest in known_shas:
                continue  # 已入库或已被审读否决：跳过（删除 manifest 行不会让段落复活）
            reason = stats_precheck(para)
            record = {
                "sha256": digest,
                "text": para,
                "url": seed["url"],
                "author": seed["author"],
                "source": seed["source"],
                "publish_date": seed["date"],
                "publish_date_precision": seed.get("publish_date_precision", "month"),
                "layer": seed.get("layer", "modern"),
                "cjk": cjk,
                "fetched_at": datetime.now(UTC).isoformat(timespec="seconds"),
            }
            if seed.get("register"):
                record["register"] = seed["register"]  # 周刊=资讯评述，与博客散文区分
            if reason:
                record["precheck_rejected"] = reason  # 仅留档不占配额（第四道关）
                records.append(record)
                known_shas.add(digest)  # 同段经别的 URL 再出现也不重复留档
                continue
            if per_author.get(seed["author"], 0) >= PER_AUTHOR_CAP:
                break
            records.append(record)
            per_author[seed["author"]] = per_author.get(seed["author"], 0) + 1
            key = (seed["author"], seed["url"])
            per_post[key] = per_post.get(key, 0) + 1
            if per_post[key] >= PER_POST_CAP:
                break  # 单篇取满即止（含此前运行已入库的段），避免一篇文章主导风格样本
    return records


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--only-seeds",
        action="store_true",
        help="只处理种子文件，跳过公版书自动发现（用于受控的定向扩容）",
    )
    parser.add_argument(
        "--seeds-file",
        type=Path,
        default=SEEDS_FILE,
        help="种子文件路径（默认 calibration/seeds.json）",
    )
    parser.add_argument(
        "--allow-host",
        action="append",
        default=[],
        metavar="HOST",
        help="额外放行的种子站点主机名（可重复）；内置白名单仅 gutenberg.org 与 web.archive.org",
    )
    args = parser.parse_args()  # 每作者上限使用常量 PER_AUTHOR_CAP
    EXTRA_ALLOWED_HOSTS.update(args.allow_host)

    RAW_DIR.mkdir(parents=True, exist_ok=True)
    MANIFEST.parent.mkdir(parents=True, exist_ok=True)
    # manifest 全量读取：known_shas 含已否决段（防复活），author_total/post_total 只计有效段。
    # 剔除协议：审读否决改打 review_rejected 标记、不删除行——删行会让发现逻辑重新抓回。
    known_shas: set[str] = set()
    author_total: dict[str, int] = {}
    post_total: dict[tuple[str, str], int] = {}  # (作者, URL) → 已取段数，跨次运行的单篇上限
    if MANIFEST.exists():
        for line in MANIFEST.read_text(encoding="utf-8").splitlines():
            if line.strip():
                try:
                    record = json.loads(line)
                except (json.JSONDecodeError, KeyError):
                    continue
                known_shas.add(record["sha256"])
                if not record.get("precheck_rejected") and not record.get("review_rejected"):
                    author_total[record["author"]] = author_total.get(record["author"], 0) + 1
                    key = (record["author"], record["url"])
                    post_total[key] = post_total.get(key, 0) + 1
    # 标记协议启用前删除的否决段：sha 只存在于 rejected/ 文件里，不入黑名单会复活
    rejected_dir = RAW_DIR.parent / "rejected"
    if rejected_dir.exists():
        for stale in rejected_dir.glob("*.txt"):
            known_shas.add(sha256_of(stale.read_text(encoding="utf-8")))
    records = collect(
        author_total,
        known_shas,
        post_total,
        only_seeds=args.only_seeds,
        seeds_file=args.seeds_file,
    )
    print(f"新增 {len(records)} 段")

    accepted = 0
    with MANIFEST.open("a", encoding="utf-8") as manifest:
        for record in records:
            (RAW_DIR / f"{record['sha256'][:16]}.txt").write_text(record["text"], encoding="utf-8")
            # manifest 入库不含正文（版权约束）：sha256 对应 raw/ 下的本地文件，审计链完整
            provenance = {k: v for k, v in record.items() if k != "text"}
            manifest.write(json.dumps(provenance, ensure_ascii=False) + "\n")
            if not record.get("precheck_rejected"):
                accepted += 1
    print(f"共落盘 {len(records)} 段（预检通过 {accepted} 段，其余带剔除原因留档）")

    # 第四道关的另一半：人工审读清单（追加，不覆盖历史记录）
    lines = [f"\n## 待审读（{datetime.now(UTC).date()} 运行新增）\n"]
    for record in records:
        if record.get("precheck_rejected"):
            continue
        preview = record["text"][:80].replace("\n", " ")
        lines.append(f"- [ ] `{record['sha256'][:16]}` {record['source']} {record['publish_date']} | {preview}…")
    REVIEW.parent.mkdir(parents=True, exist_ok=True)
    with REVIEW.open("a", encoding="utf-8") as review:
        review.write("\n".join(lines) + "\n")
    print(f"人工审读清单：{REVIEW}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
