"""可解释理由生成（ALGORITHM §3.5 规则引擎）。

面向普通用户（大学生）的解释分三层：
- label：一句话说清是什么问题；
- why：白话解释，不出现 PPL/σ 等术语；
- suggestion：可以直接照做的修改建议；
- evidence：从原文里摘出的具体词句，让建议贴合这一段的实际写法；
- detail：原始技术指标，只在报告里折叠展示，供想深入看的同学参考。

纪律不变：
- 只列 z ≥ +EXPLAIN_Z_THRESHOLD 的前 EXPLAIN_TOP_K 个特征（z 已按
  「越大越像 AI」定向，负 z 是「为什么像人写」，不进理由）；
- 没有触发特征的红色窗口输出空列表，报告标注「模型直觉，无显著特征」
  ——不编造解释；
- 文本证据只引用原文，不生成原文没有的数据、文献或事实。
"""

from __future__ import annotations

import math
import re

from .constants import EXPLAIN_TOP_K, EXPLAIN_Z_THRESHOLD
from .features import AI_CONNECTIVES, COLLOQUIAL_MARKERS
from .preprocess import split_sentences

# 每个特征对应一段白话解释 + 一条可执行的修改建议。
# z 已按「越大越像 AI」定向（direction 并入 constants.STYLE_STATS），
# 因此 why / suggestion 描述的都是 z 偏高的一侧；不要在这里写反方向。
REASON_LIBRARY: dict[str, dict[str, str]] = {
    "punct_density": {
        "label": "句子太长、停顿少",
        "why": "这段的标点偏少，长句一个接一个，读起来像一口气说完。",
        "suggestion": "在语义转折或列举结束后断句，把长句拆成 2–3 个短句，让停顿更自然。",
        "detail": "标点密度 {z:+.1f}σ（低于人写基准）",
    },
    "char_mtld": {
        "label": "用词变化偏大",
        "why": "这段的用词变化比常见人写文本更大，可能有较多同义替换或术语堆叠。",
        "suggestion": "检查是否有为了换词而换词、同义替换或术语堆叠的地方；保留最准确的说法，必要时重复关键词比生硬替换更清楚。",
        "detail": "词汇多样性 {z:+.1f}σ（高于人写基准）",
    },
    "period_comma_ratio": {
        "label": "长句串联、句号少",
        "why": "句号和逗号的比例偏低，多个分句用逗号串在一起，读起来节奏比较单一。",
        "suggestion": "把长从句拆开，适当用句号收束；列举项多时改成短句或分条。",
        "detail": "句读节奏 {z:+.1f}σ（句号/逗号比偏低）",
    },
    "de_density": {
        "label": "句子过于凝练",
        "why": "「的」字比常见人写文本少，句子显得很紧，解释和修饰不足。",
        "suggestion": "在关键概念后补充一句解释或限定条件，适当增加“的”字结构，让表达更自然、更好懂。",
        "detail": "「的」字密度 {z:+.1f}σ（低于人写基准）",
    },
    "punct_diversity": {
        "label": "标点种类偏多",
        "why": "这段用了较多不同种类的标点（括号、分号、破折号、编号等），看起来像排版稿。",
        "suggestion": "统一标点用法；能用一个逗号或句号说清的地方，不要叠加括号、分号和破折号。",
        "detail": "标点多样性 {z:+.1f}σ（高于人写基准）",
    },
    "colloquial_density": {
        "label": "口语化偏多",
        "why": "这段的口语词（比如“其实、挺、吧”）比较多，读起来像讲解或聊天。",
        "suggestion": "如果是论文，改成书面表达；如果是科普，可以保留，但注意和全文语气一致。",
        "detail": "口语词密度 {z:+.1f}σ（高于人写基准）",
    },
    "connective_density": {
        "label": "连接词偏多",
        "why": "“首先、其次、因此、此外、同时、综上所述”这类连接词出现得比较密，读起来像模板。",
        "suggestion": "删掉一部分连接词，用具体内容、数据或短句自然过渡；一段里保留 1–2 个关键连接词就够了。",
        "detail": "连接词密度 {z:+.1f}σ（高于人写基准）",
    },
    "sent_initial_rate": {
        "label": "句首连接词偏多",
        "why": "好几句话都用“因此、此外、同时、然而”开头，句式比较整齐。",
        "suggestion": "换几种开头方式：用主语开头、用具体事实开头，或者把两句合并；不要每句都用连接词起头。",
        "detail": "句首连接词比例 {z:+.1f}σ（高于人写基准）",
    },
}

PPL_REASON = {
    "label": "句子太顺、太可预测",
    "why": "这段读起来非常顺，语言模型很容易预测下一个词；检测器会把这种“过于流畅”当成 AI 特征。",
    "suggestion": "补充具体数据、实验过程、失败案例或限制条件，让内容有更多“只有作者才知道”的信息；不要故意写病句。",
}

# 兼容旧调用：需要原始技术模板的地方仍可用。
FEATURE_LABELS = {name: item["detail"] for name, item in REASON_LIBRARY.items()}

_EXTRA_CONNECTIVES = ["因此", "此外", "同时", "然而", "总之", "最后", "另外"]
_TEMPLATE_OPENINGS = ["随着", "近年来", "在当今", "众所周知", "不可否认", "在……背景下", "在……时代"]
_EMPTY_SUMMARIES = ["综上所述", "总而言之", "总的来说", "由此可见", "不难看出"]
_VAGUE_CLAIMS = [
    "显著",
    "明显",
    "大幅",
    "有效",
    "进一步",
    "不断",
    "越来越",
    "一定程度上",
    "某种程度上",
    "具有重要",
    "起到关键",
]
_PARALLEL_PATTERNS = [
    re.compile(r"一方面[^。！？；]{0,40}另一方面"),
    re.compile(r"不仅[^。！？；]{0,40}而且"),
    re.compile(r"既[^。！？；]{0,30}又"),
]
_PUNCT_MARKS = ["；", "、", "（", "）", "——", "：", "“", "”", "《", "》"]


def _is_number(value) -> bool:
    return isinstance(value, (int, float)) and math.isfinite(value)


def _matched_terms(text: str, terms: list[str]) -> list[str]:
    hits = [(text.find(term), term) for term in terms if term in text]
    # 同一位置优先长词，避免「与此同时」又额外命中一次「同时」
    hits.sort(key=lambda item: (item[0], -len(item[1])))
    selected: list[str] = []
    occupied: list[tuple[int, int]] = []
    for position, term in hits:
        end = position + len(term)
        if any(position >= start and end <= stop for start, stop in occupied):
            continue
        occupied.append((position, end))
        selected.append(term)
    return selected


def _quote(text: str, limit: int = 30) -> str:
    text = (text or "").strip()
    return text if len(text) <= limit else text[:limit] + "…"


def _break_hint(sentence: str) -> str:
    for marker in ("与此同时", "因此", "但是", "然而", "其中", "；"):
        index = sentence.find(marker, 10)
        if index > 0:
            return marker
    return "第二个逗号处"


def _repeated_openings(sentences: list[str]) -> list[tuple[str, int]]:
    counts: dict[str, int] = {}
    for sentence in sentences:
        prefix = sentence[:2]
        if len(prefix) < 2 or all(ch in "，。！？；、 " for ch in prefix):
            continue
        counts[prefix] = counts.get(prefix, 0) + 1
    repeated = [(prefix, count) for prefix, count in counts.items() if count >= 2]
    repeated.sort(key=lambda item: (-item[1], item[0]))
    return repeated


def _leading_connective(sentence: str) -> str:
    for term in AI_CONNECTIVES + _EXTRA_CONNECTIVES:
        if sentence.startswith(term):
            return term
    return ""


def _opening(sentence: str) -> str:
    return _leading_connective(sentence) or sentence[:4]


def _text_context(text: str) -> dict:
    sentences = [s.text.strip() for s in split_sentences(text) if s.text.strip()]
    longest = max(sentences, key=len) if sentences else ""
    parallel = []
    for pattern in _PARALLEL_PATTERNS:
        match = pattern.search(text)
        if match:
            parallel.append(match.group(0))
    return {
        "text": text,
        "sentences": sentences,
        "longest": longest,
        "connectives": _matched_terms(text, AI_CONNECTIVES + _EXTRA_CONNECTIVES),
        "colloquial": _matched_terms(text, COLLOQUIAL_MARKERS),
        "vague": _matched_terms(text, _VAGUE_CLAIMS),
        "template": _matched_terms(text, _TEMPLATE_OPENINGS),
        "summary": _matched_terms(text, _EMPTY_SUMMARIES),
        "parallel": parallel,
        "repeated": _repeated_openings(sentences),
        "openings": [_opening(sentence) for sentence in sentences],
        "connective_openings": [
            term for sentence in sentences if (term := _leading_connective(sentence))
        ],
        "marks": [mark for mark in _PUNCT_MARKS if mark in text],
        "break_hint": _break_hint(longest),
    }


def _enrich(item: dict[str, str], ctx: dict) -> dict[str, str]:
    """把通用建议替换成引用本段原文的具体建议；没有证据时保持原样。"""
    out = dict(item)
    feature = item["feature"]
    evidence = ""

    if feature == "ppl":
        if ctx["template"]:
            span = ctx["template"][0]
            out["why"] = f"这段用「{span}」开头，句子又很顺，语言模型很容易预测下一个词。"
            out["suggestion"] = f"把「{span}」换成你研究里的具体时间、对象或数据；不要只做同义替换。"
            evidence = span
        elif ctx["vague"]:
            span = ctx["vague"][0]
            out["why"] = f"这段用了「{span}」这类评价词，但没有给出具体幅度。"
            out["suggestion"] = f"把「{span}」换成具体指标和数值；没有数据就删掉形容词，直接陈述结果。"
            evidence = span
        elif ctx["longest"] and len(ctx["longest"]) > 50:
            span = _quote(ctx["longest"])
            out["why"] = f"最长的一句有 {len(ctx['longest'])} 字，读起来很顺但信息密度低。"
            out["suggestion"] = f"把「{span}」拆成 2 句，并在后面补一个具体数据、过程或限制条件。"
            evidence = span
        else:
            out["suggestion"] = (
                "这段没有明显的模板词，但句子太顺；挑一句最像结论的句子，"
                "补上具体数据、过程或限制条件。"
            )
    elif feature == "connective_density" and ctx["connectives"]:
        spans = "、".join(ctx["connectives"][:5])
        out["why"] = f"本段出现了「{spans}」等连接词，密度偏高，读起来像模板。"
        out["suggestion"] = (
            f"本段出现了「{spans}」。删掉其中 2–3 个，直接从动作或对象写起；"
            "保留 1 个表转折或收束即可。"
        )
        evidence = spans
    elif feature == "sent_initial_rate":
        openings = ctx["connective_openings"]
        if len(openings) >= 2:
            spans = "、".join(openings[:4])
            out["why"] = f"多句用「{spans}」开头，句式比较整齐。"
            out["suggestion"] = "换几种开头方式：用具体对象、方法名或结果开头；不要每句都用连接词起头。"
            evidence = spans
        elif openings:
            span = openings[0]
            out["label"] = "句首用连接词"
            out["why"] = f"这句用「{span}」开头，句式比较整齐。"
            out["suggestion"] = "换一种开头方式：用具体对象、方法名或结果开头，不要用连接词起句。"
            evidence = span
        elif ctx["repeated"]:
            prefix, count = ctx["repeated"][0]
            out["label"] = "句首重复"
            out["why"] = f"有 {count} 句以「{prefix}」开头，句式比较整齐。"
            out["suggestion"] = (
                f"有 {count} 句以「{prefix}」开头；把其中 {max(1, count - 1)} 句"
                "改成具体对象、方法或结果开头。"
            )
            evidence = f"{prefix}（{count} 句）"
    elif feature in ("punct_density", "period_comma_ratio") and ctx["longest"]:
        sentence = ctx["longest"]
        span = _quote(sentence, 28)
        commas = sentence.count("，")
        out["why"] = f"最长的一句有 {len(sentence)} 字、{commas} 个逗号，读起来像一口气说完。"
        out["suggestion"] = (
            f"最长的一句是「{span}」（{len(sentence)} 字，{commas} 个逗号）；"
            f"在「{ctx['break_hint']}」前断成两句。"
        )
        evidence = f"「{span}」（{len(sentence)} 字，{commas} 个逗号）"
    elif feature == "de_density" and ctx["sentences"]:
        n_de = ctx["text"].count("的")
        out["why"] = f"本段 {len(ctx['sentences'])} 句里只有 {n_de} 个「的」，句子显得很紧。"
        out["suggestion"] = (
            f"在「{_quote(ctx['longest'], 24)}」后补一句限定条件或解释（适用数据/场景），"
            "不要只加「的」字。"
        )
        evidence = f"{len(ctx['sentences'])} 句 / {n_de} 个「的」"
    elif feature == "punct_diversity" and ctx["marks"]:
        spans = "、".join(ctx["marks"])
        out["why"] = f"本段用了「{spans}」等多种标点，看起来像排版稿。"
        out["suggestion"] = (
            f"本段用了「{spans}」；统一成一种分隔方式，能用一个逗号或句号说清的地方不要叠加。"
        )
        evidence = spans
    elif feature == "colloquial_density" and ctx["colloquial"]:
        spans = "、".join(ctx["colloquial"][:5])
        out["why"] = f"本段有「{spans}」等口语词，读起来像讲解或聊天。"
        out["suggestion"] = f"本段有「{spans}」；论文里改成书面表达，或确认全文语气一致。"
        evidence = spans
    elif feature == "char_mtld" and ctx["longest"]:
        span = _quote(ctx["longest"], 24)
        out["why"] = f"「{span}」这类句子用词变化偏大，可能有同义替换或术语堆叠。"
        out["suggestion"] = (
            f"检查「{span}」里是否有为了换词而换词；保留最准确的说法，"
            "必要时重复关键词比生硬替换更清楚。"
        )
        evidence = span

    if evidence:
        out["evidence"] = evidence
    return out


def _text_items(ctx: dict) -> list[dict[str, str]]:
    """从原文直接观察到的模板/空泛表达；不是模型特征，单独标注为文本证据。"""
    items: list[dict[str, str]] = []
    if ctx["template"]:
        span = ctx["template"][0]
        items.append(
            {
                "feature": "text:template_opening",
                "label": "模板化开头",
                "why": f"这段用「{span}」这类常见开头，读起来像综述模板。",
                "suggestion": (
                    f"把「{span}」换成你研究里的具体时间、对象或问题；"
                    "不要保留空泛的时代背景。"
                ),
                "detail": "文本证据：模板化开头",
                "evidence": span,
            }
        )
    if ctx["summary"]:
        span = ctx["summary"][0]
        items.append(
            {
                "feature": "text:empty_summary",
                "label": "总结句空泛",
                "why": f"这段用「{span}」收束，但前文没有对应的具体结论。",
                "suggestion": (
                    f"删掉「{span}」，直接写出你的具体结论：哪个指标、什么变化、适用条件。"
                ),
                "detail": "文本证据：空泛总结",
                "evidence": span,
            }
        )
    if ctx["vague"]:
        span = ctx["vague"][0]
        items.append(
            {
                "feature": "text:vague_claim",
                "label": "结论空泛、缺具体信息",
                "why": f"这段用了「{span}」这类评价词，但没有给出幅度或指标。",
                "suggestion": (
                    f"把「{span}」换成具体指标和数值；没有数据就删掉形容词，直接陈述结果。"
                ),
                "detail": "文本证据：空泛评价词",
                "evidence": span,
            }
        )
    if ctx["parallel"]:
        span = ctx["parallel"][0]
        items.append(
            {
                "feature": "text:parallel_template",
                "label": "排比/对仗句式",
                "why": f"这段用了「{_quote(span, 40)}」的整齐结构。",
                "suggestion": "保留一层对比即可；把第二层改成具体事实或拆成两句，避免为了对仗而重复。",
                "detail": "文本证据：排比/对仗",
                "evidence": _quote(span, 40),
            }
        )
    if ctx["repeated"]:
        prefix, count = ctx["repeated"][0]
        items.append(
            {
                "feature": "text:repeated_opening",
                "label": "句首重复",
                "why": f"有 {count} 句以「{prefix}」开头。",
                "suggestion": (
                    f"把其中 {max(1, count - 1)} 句改成具体对象、方法或结果开头，"
                    "不要连续用同一个主语起句。"
                ),
                "detail": "文本证据：句首重复",
                "evidence": f"{prefix}（{count} 句）",
            }
        )
    if ctx["longest"] and len(ctx["longest"]) >= 60:
        span = _quote(ctx["longest"], 28)
        items.append(
            {
                "feature": "text:long_sentence",
                "label": "单句过长",
                "why": f"最长的一句有 {len(ctx['longest'])} 字、{ctx['longest'].count('，')} 个逗号。",
                "suggestion": (
                    f"在「{ctx['break_hint']}」前断句，把「{span}」拆成两句；"
                    "拆完再补一个具体数据或限制条件。"
                ),
                "detail": "文本证据：单句过长",
                "evidence": f"「{span}」（{len(ctx['longest'])} 字）",
            }
        )
    return items


def _evidence_terms(evidence: str) -> list[str]:
    return [part.strip() for part in re.split(r"、", evidence or "") if part.strip()]


def _covered_by(evidence: str, covered: set[str]) -> bool:
    terms = _evidence_terms(evidence)
    if not terms:
        return False
    return all(any(term in span or span in term for span in covered) for term in terms)


def dedupe_reason_items(items: list[dict[str, str]]) -> list[dict[str, str]]:
    """同一处原文证据只保留第一条理由，避免「总结空泛 + 连接词偏多」刷屏。"""
    kept: list[dict[str, str]] = []
    covered: set[str] = set()
    for item in items:
        evidence = item.get("evidence")
        if evidence and _covered_by(evidence, covered):
            continue
        kept.append(item)
        if evidence:
            covered.add(evidence)
    return kept


def sentence_reason_items(
    meta: dict,
    text: str,
    risk: float | None,
    top_k: int = EXPLAIN_TOP_K,
) -> list[dict[str, str]]:
    """句级理由：只引用这一句里真实出现的证据。

    窗口级理由会把整段（多句）的模板词、连接词算进来；直接挂到句子上会出现
    “引用的原文不在这句里”。这里用句子自身的文本重新生成理由，并丢掉
    没有句子内证据的统计特征；一句里实在没有可引用的写法时，明确说明
    风险来自所在段落，而不是编造原因。
    """
    items = reason_items_for(meta, top_k=top_k + 4, text=text, risk=risk)
    text_items = [item for item in items if item.get("feature", "").startswith("text:")]
    ppl_items = [item for item in items if item.get("feature") == "ppl"]
    feature_items = [
        item
        for item in items
        if not item.get("feature", "").startswith("text:")
        and item.get("feature") not in ("ppl", "sent_initial_rate")
    ]
    if risk is not None and risk < 0.5:
        # 低风险句子不挂窗口统计特征，避免把整段特征说成这一句的问题
        feature_items = []

    covered: set[str] = {item["evidence"] for item in text_items if item.get("evidence")}
    grounded: list[dict[str, str]] = list(text_items)

    for item in ppl_items:
        evidence = item.get("evidence")
        if not evidence or _covered_by(evidence, covered):
            continue
        grounded.append(item)
        covered.add(evidence)

    for item in feature_items:
        evidence = item.get("evidence")
        if not evidence or _covered_by(evidence, covered):
            continue
        item = dict(item)
        item["detail"] = f"{item.get('detail', '')}；窗口统计"
        grounded.append(item)
        covered.add(evidence)

    if not grounded and (risk is None or risk >= 0.5):
        grounded.append(
            {
                "feature": "window_context",
                "label": "风险来自所在段落",
                "why": "这句本身没有明显的模板词或空泛表达，风险主要来自所在段落的整体统计特征。",
                "suggestion": (
                    "点开「窗口明细」看整段上下文；如果要改，优先补充具体数据、"
                    "过程或限制条件，不要只换同义词。"
                ),
                "detail": "句级风险继承自窗口融合分",
            }
        )
    return grounded[:top_k]


def reason_items_for(
    meta: dict,
    top_k: int = EXPLAIN_TOP_K,
    z_threshold: float = EXPLAIN_Z_THRESHOLD,
    text: str = "",
    risk: float | None = None,
) -> list[dict[str, str]]:
    """返回结构化理由：feature / label / why / suggestion / detail / evidence。

    顺序：PPL 理由 → 文本证据（最多 2 条）→ 风格特征（按 z 从大到小）。
    `text` 为空时退化为旧版纯特征解释；`risk < 0.5` 时不追加文本证据，
    避免低风险段落被模板词提示淹没。
    """
    z_values = meta.get("z_values") or {}
    candidates = [
        (name, z)
        for name, z in z_values.items()
        if _is_number(z) and z >= z_threshold and name in REASON_LIBRARY
    ]
    candidates.sort(key=lambda item: item[1], reverse=True)

    ctx = _text_context(text) if text.strip() else None
    items: list[dict[str, str]] = []

    p_ppl = meta.get("p_ppl")
    if _is_number(p_ppl) and p_ppl >= 1.0 / (1.0 + math.exp(-z_threshold)):
        ppl = meta.get("ppl")
        log_ppl = meta.get("log_ppl")
        if _is_number(ppl) and ppl > 0:
            detail = f"困惑度偏低（PPL {ppl:.0f}）"
            if _is_number(log_ppl):
                detail = f"困惑度偏低（log PPL {log_ppl:.2f}，PPL {ppl:.0f}）"
        else:
            detail = "困惑度偏低（语言模型很容易预测）"
        item = {
            "feature": "ppl",
            "label": PPL_REASON["label"],
            "why": PPL_REASON["why"],
            "suggestion": PPL_REASON["suggestion"],
            "detail": detail,
        }
        items.append(_enrich(item, ctx) if ctx else item)

    if ctx and (risk is None or risk >= 0.5):
        items.extend(_text_items(ctx)[:2])

    for name, z in candidates:
        library = REASON_LIBRARY[name]
        item = {
            "feature": name,
            "label": library["label"],
            "why": library["why"],
            "suggestion": library["suggestion"],
            "detail": library["detail"].format(z=z),
        }
        items.append(_enrich(item, ctx) if ctx else item)
    return items[:top_k]


def reasons_for(
    meta: dict,
    top_k: int = EXPLAIN_TOP_K,
    z_threshold: float = EXPLAIN_Z_THRESHOLD,
) -> list[str]:
    """兼容旧接口：只返回白话原因，不含技术细节。"""
    return [item["why"] for item in reason_items_for(meta, top_k, z_threshold)]


def suggestions_for(items: list[dict[str, str]]) -> list[str]:
    """从理由列表提取去重后的修改建议。"""
    seen: set[str] = set()
    suggestions: list[str] = []
    for item in items:
        suggestion = item.get("suggestion")
        if suggestion and suggestion not in seen:
            seen.add(suggestion)
            suggestions.append(suggestion)
    return suggestions
