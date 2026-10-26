"""可选的大模型具体建议层。

用已经加载的 C 路基座模型（Qwen）为高风险窗口生成一条贴合原文的改写示例；
规则引擎仍然是默认和兜底。约束写进 prompt：引用原文、不改结构、不编造数据。

注意：这里生成的是「改写示例」，不是自动替换。报告会把它标成实验性内容，
要求用户人工确认；任何数字/百分比只要原文没有，整条建议就会被丢弃。
"""

from __future__ import annotations

import re

from .fuse import FusedWindow

PROMPT_TEMPLATE = """你是一名中文学术写作编辑。下面这段文字被本地检测器标为“像 AI 写的”。
请把它改写得更像真人写作，但必须遵守：
1. 只改表达，不改变原意、段落结构、论证顺序、事实、数据、引用；
2. 不增加原文没有的数据、文献、案例或结论；
3. 保留专业术语和关键名词；
4. 只输出改写后的文字，不要解释，不要加引号。

原文：
{text}

检测器提示：
{reasons}

改写："""

_CJK = re.compile(r"[\u4e00-\u9fff]")
_NUMBER = re.compile(r"\d+(?:\.\d+)?%?")
_PREFIX = re.compile(r"^(?:改写|修改|建议|可以改为|可改为|改为)\s*[：:]\s*")
_EXPLANATION = re.compile(
    r"\n\s*(?:检测器提示|改写说明|修改说明|风格提示|说明|解释|注)\s*[：:]?"
)
_META_WORDS = ("这段文字", "作为AI", "作为 AI")
# 模型常把「表达改写」升级成「效果承诺」；原文没有这些说法时整条丢弃
_NEW_CLAIM_TERMS = (
    "鲁棒性",
    "准确性",
    "有效性",
    "优越性",
    "可靠性",
    "泛化能力",
    "显著提升",
    "大幅提升",
    "充分证明",
    "毫无疑问",
)


def _truncate(text: str, limit: int) -> str:
    text = (text or "").strip()
    return text if len(text) <= limit else text[:limit] + "…"


def build_prompt(window: FusedWindow) -> str:
    reasons = "；".join(
        f"{item.get('label', '风格特征')}（{item.get('evidence') or item.get('why', '')}）"
        for item in (window.reason_items or [])[:3]
    )
    return PROMPT_TEMPLATE.format(text=_truncate(window.text, 700), reasons=reasons or "未给出")


def _shares_cjk_span(output: str, source: str) -> bool:
    chars = [ch for ch in source if _CJK.match(ch)]
    shared = 0
    for i in range(len(chars) - 1):
        if "".join(chars[i : i + 2]) in output:
            shared += 1
            if shared >= 2:
                return True
    return False


def _numbers(text: str) -> set[str]:
    return set(_NUMBER.findall(text or ""))


def clean_advice(output: str, source: str) -> str | None:
    """清洗生成结果；不贴合原文或不安全的结果返回 None，由规则建议兜底。"""
    text = (output or "").strip()
    if "改写：" in text:
        text = text.rsplit("改写：", 1)[-1].strip()
    text = _PREFIX.sub("", text)
    # 基座模型常会附上「改写说明」；只保留前面的改写正文
    marker = _EXPLANATION.search(text)
    if marker:
        text = text[: marker.start()].strip()
    text = text.strip(" \n\t\"'“”‘’：:")
    if not text or len(text) < 8 or len(text) > 240:
        return None
    if any(word in text for word in ("抱歉", "无法", "不能提供", "作为AI", "作为 AI")):
        return None
    if any(word in text for word in _META_WORDS):
        return None
    if not _shares_cjk_span(text, source):
        return None
    # 只改写半句/一句就停下的结果信息量不足
    source_cjk = sum(1 for ch in source if _CJK.match(ch))
    output_cjk = sum(1 for ch in text if _CJK.match(ch))
    if source_cjk >= 30 and output_cjk < source_cjk * 0.35:
        return None
    # 改写示例不能凭空出现原文没有的数字/百分比
    if not _numbers(text) <= _numbers(source):
        return None
    # 不能把改写升级成原文没有的效果承诺
    if any(term in text and term not in source for term in _NEW_CLAIM_TERMS):
        return None
    # 几乎原文照抄的“改写”没有信息量
    if text.strip("。！？；，、 ") == source.strip("。！？；，、 "):
        return None
    return text[:160]


def generate_for_windows(
    fused: list[FusedWindow],
    detectors: list,
    max_windows: int = 3,
    max_new_tokens: int = 96,
) -> tuple[dict[str, dict], dict]:
    """为风险最高的少量窗口生成改写示例；返回 (window_key → advice, info)。"""
    generator = next(
        (detector for detector in detectors if hasattr(detector, "generate_batch")),
        None,
    )
    if generator is None:
        return {}, {
            "enabled": False,
            "reason": "C 路检测器不支持文本生成；已使用规则建议。",
            "n_generated": 0,
        }

    candidates = sorted(
        (window for window in fused if window.p_fuse >= 0.55 and window.text.strip()),
        key=lambda window: window.p_fuse,
        reverse=True,
    )[:max_windows]
    if not candidates:
        return {}, {
            "enabled": True,
            "model": getattr(generator, "model_id", ""),
            "n_generated": 0,
            "reason": "没有达到 0.55 的高风险窗口。",
        }

    prompts = [build_prompt(window) for window in candidates]
    try:
        outputs = generator.generate_batch(prompts, max_new_tokens=max_new_tokens)
    except Exception as exc:  # noqa: BLE001 —— 生成失败不能阻断主报告
        return {}, {
            "enabled": False,
            "model": getattr(generator, "model_id", ""),
            "n_generated": 0,
            "reason": f"生成失败：{type(exc).__name__}: {exc}",
        }
    if len(outputs) != len(candidates):
        return {}, {
            "enabled": False,
            "model": getattr(generator, "model_id", ""),
            "n_generated": 0,
            "reason": f"生成结果数量不匹配（{len(outputs)}/{len(candidates)}）；已使用规则建议。",
        }

    advice: dict[str, dict] = {}
    for window, output in zip(candidates, outputs, strict=True):
        cleaned = clean_advice(output, window.text)
        if cleaned:
            advice[window.window_key] = {
                "text": cleaned,
                "kind": "rewrite",
                "model": getattr(generator, "model_id", ""),
                "experimental": True,
            }
    reason = "已约束保留原意/结构/数据，并过滤原文没有的数字；请人工确认后再使用。"
    if not advice:
        reason = "生成内容未通过安全校验（可能偏离原文或含新数字）；已使用规则建议。"
    return advice, {
        "enabled": True,
        "model": getattr(generator, "model_id", ""),
        "n_generated": len(advice),
        "n_candidates": len(candidates),
        "reason": reason,
    }
