/* aigc-selfcheck 报告交互（vanilla JS，无框架依赖，数据来自内联 JSON）。 */
(function () {
  "use strict";
  var DATA = JSON.parse(document.getElementById("selfcheck-data").textContent);
  var PATH_LABELS = { a: "A · zhv2 分类器", b: "B · cmj ONNX", c: "C · 统计特征" };

  function el(tag, attrs) {
    var node = document.createElement(tag);
    attrs = attrs || {};
    Object.keys(attrs).forEach(function (k) {
      if (k === "text") node.textContent = attrs[k];
      else if (k === "html") node.innerHTML = attrs[k];
      else if (k.indexOf("on") === 0) node.addEventListener(k.slice(2), attrs[k]);
      else node.setAttribute(k, attrs[k]);
    });
    for (var i = 2; i < arguments.length; i++) {
      var child = arguments[i];
      if (child == null) continue;
      node.appendChild(typeof child === "string" ? document.createTextNode(child) : child);
    }
    return node;
  }

  function fmt(x, digits) {
    if (x === null || x === undefined || x !== x) return "—";
    return Number(x).toFixed(digits === undefined ? 2 : digits);
  }

  function heatClass(risk) {
    if (risk < 0.35) return "r-low";
    if (risk > 0.65) return "r-high";
    return "r-mid";
  }

  function riskBadge(risk) {
    return el("span", { class: "risk-badge " + heatClass(risk), text: fmt(risk) });
  }

  function bandClass(band) {
    return band === "低" ? "r-low" : band === "高" ? "r-high" : "r-mid";
  }

  function bandBadge(band) {
    return el("span", { class: "risk-badge " + bandClass(band), text: band || "—" });
  }

  function renderReviewNote(note) {
    if (!note || !note.why) return null;
    return el("div", { class: "review-note review-" + (note.level || "review") },
      el("b", { text: note.label || "先人工确认" }),
      el("span", { text: "：" + note.why })
    );
  }

  function renderLlmAdvice(advice) {
    if (!advice || !advice.text) return null;
    return el("div", { class: "llm-advice" },
      el("b", { text: "所在段落改写示例（实验性）" }),
      el("div", { class: "llm-text", text: advice.text }),
      el("div", { class: "llm-caveat", text: "只改表达，不改变结构、事实和数据；请人工确认后再使用。" })
    );
  }

  function bar(score) {
    var fill = el("div", { class: "fill" });
    fill.style.width = Math.max(0, Math.min(100, score * 100)) + "%";
    var color = score > 0.65 ? "var(--red)" : score < 0.35 ? "var(--green)" : "var(--yellow)";
    fill.style.background = color;
    return el("span", { class: "bar" }, fill);
  }

  /* ── 白话理由与修改建议 ─────────────────────────── */
  function normalizeReasonItems(obj) {
    var items = (obj && obj.reason_items) || [];
    if (!items.length && obj && (obj.reasons || []).length) {
      items = obj.reasons.map(function (text) {
        return { label: "风格特征", why: text };
      });
    }
    return items;
  }

  function uniqueSuggestions(items) {
    var out = [];
    (items || []).forEach(function (item) {
      if (item && item.suggestion && out.indexOf(item.suggestion) < 0) out.push(item.suggestion);
    });
    return out;
  }

  function renderReasonWhy(items, risk, limit, note, advice) {
    var wrap = el("div", { class: "reasons" });
    var flag = renderReviewNote(note);
    if (flag) wrap.appendChild(flag);
    if (!items || !items.length) {
      wrap.textContent = risk !== null && risk !== undefined && risk > 0.65
        ? "模型觉得这段有风险，但没有找到特别明显的风格原因。"
        : "风险不高，未发现明显的模板词或空泛表达。";
      var emptyAdvice = renderLlmAdvice(advice);
      if (emptyAdvice) wrap.appendChild(emptyAdvice);
      return wrap;
    }
    var n = limit || 2;
    items.slice(0, n).forEach(function (item) {
      var row = el("div", { class: "reason-item" },
        el("b", { text: item.label || "风格特征" }),
        el("span", { class: "why", text: "：" + (item.why || "") })
      );
      if (item.evidence) row.appendChild(el("span", { class: "evidence", text: "（原文：" + item.evidence + "）" }));
      wrap.appendChild(row);
    });
    if (items.length > n) {
      wrap.appendChild(el("div", { class: "reason-more", text: "还有 " + (items.length - n) + " 条，点句子查看" }));
    }
    var adv = renderLlmAdvice(advice);
    if (adv) wrap.appendChild(adv);
    return wrap;
  }

  function renderSuggestionCell(items, risk, limit) {
    var wrap = el("div", { class: "reasons fix" });
    var fixes = uniqueSuggestions(items);
    if (!fixes.length) {
      if (risk !== null && risk !== undefined && risk > 0.65) {
        wrap.textContent = "通读这一段，检查是否有空泛总结或直接粘贴的 AI 内容。";
      }
      return wrap;
    }
    var n = limit || 2;
    fixes.slice(0, n).forEach(function (text) {
      wrap.appendChild(el("div", { class: "fix-item", text: "建议：" + text }));
    });
    if (fixes.length > n) {
      wrap.appendChild(el("div", { class: "reason-more", text: "还有 " + (fixes.length - n) + " 条建议" }));
    }
    return wrap;
  }

  function renderReasonAndFix(items, risk, limit, note, advice) {
    var wrap = el("div", { class: "reasons" });
    var flag = renderReviewNote(note);
    if (flag) wrap.appendChild(flag);
    if (!items || !items.length) {
      wrap.textContent = risk > 0.65
        ? "模型觉得有风险，但没有找到特别明显的风格原因。"
        : "风险不高，未发现明显的模板词或空泛表达。";
      var emptyAdvice = renderLlmAdvice(advice);
      if (emptyAdvice) wrap.appendChild(emptyAdvice);
      return wrap;
    }
    var n = limit || 2;
    items.slice(0, n).forEach(function (item) {
      var block = el("div", { class: "reason-item" });
      block.appendChild(el("b", { text: item.label || "风格特征" }));
      block.appendChild(el("div", { class: "why", text: item.why || "" }));
      if (item.evidence) block.appendChild(el("div", { class: "evidence", text: "原文证据：" + item.evidence }));
      if (item.suggestion) block.appendChild(el("div", { class: "fix-item", text: "建议：" + item.suggestion }));
      wrap.appendChild(block);
    });
    if (items.length > n) {
      wrap.appendChild(el("div", { class: "reason-more", text: "还有 " + (items.length - n) + " 条" }));
    }
    var adv = renderLlmAdvice(advice);
    if (adv) wrap.appendChild(adv);
    return wrap;
  }

  function reasonHeading(risk) {
    if (risk !== null && risk !== undefined && risk < 0.35) {
      return {
        why: "模型注意到的风格特征（这段风险不高，仅供参考）：",
        fix: "可以怎么调整（可选）：",
      };
    }
    if (risk !== null && risk !== undefined && risk <= 0.65) {
      return { why: "可能被注意到的原因：", fix: "可以怎么调整：" };
    }
    return { why: "为什么会被标：", fix: "可以怎么改：" };
  }

  function renderReasonDetails(items, risk, advice) {
    var wrap = el("div", {});
    var heads = reasonHeading(risk);
    if (!items || !items.length) {
      if (risk !== null && risk !== undefined && risk > 0.65) {
        wrap.appendChild(el("div", { class: "k", text: heads.why }));
        wrap.appendChild(el("p", { class: "reasons", text: "模型觉得这段有风险，但没有找到特别明显的风格原因。" }));
        wrap.appendChild(el("div", { class: "k", text: heads.fix }));
        wrap.appendChild(el("p", { class: "reasons", text: "通读这一段，检查是否有大段直接粘贴的 AI 内容、空泛的总结句，或缺少自己的分析。" }));
      }
      var emptyAdvice = renderLlmAdvice(advice);
      if (emptyAdvice) wrap.appendChild(emptyAdvice);
      return wrap;
    }
    wrap.appendChild(el("div", { class: "k", text: heads.why }));
    var whyList = el("ul", { class: "reason-list" });
    items.forEach(function (item) {
      var li = el("li", {},
        el("b", { text: item.label || "风格特征" }),
        el("span", { text: "：" + (item.why || "") })
      );
      if (item.evidence) li.appendChild(el("div", { class: "evidence", text: "原文证据：" + item.evidence }));
      whyList.appendChild(li);
    });
    wrap.appendChild(whyList);
    var fixes = uniqueSuggestions(items);
    if (fixes.length) {
      wrap.appendChild(el("div", { class: "k", text: heads.fix }));
      var fixList = el("ul", { class: "reason-list fix-list" });
      fixes.forEach(function (text) { fixList.appendChild(el("li", { text: text })); });
      wrap.appendChild(fixList);
    }
    var details = items.filter(function (item) { return item.detail; });
    if (details.length) {
      var tech = el("details", { class: "tech-details" });
      tech.appendChild(el("summary", { text: "技术细节（给想深入看的同学）" }));
      var techList = el("ul", { class: "reason-list" });
      details.forEach(function (item) { techList.appendChild(el("li", { text: item.detail })); });
      tech.appendChild(techList);
      wrap.appendChild(tech);
    }
    var adv = renderLlmAdvice(advice);
    if (adv) wrap.appendChild(adv);
    return wrap;
  }

  /* ── 句子 → 窗口映射 ─────────────────────────────── */
  var sentenceWindows = {};  // gid -> [window...]
  (DATA.windows || []).forEach(function (w) {
    (w.sentences || []).forEach(function (gid) {
      (sentenceWindows[gid] = sentenceWindows[gid] || []).push(w);
    });
  });
  Object.keys(sentenceWindows).forEach(function (gid) {
    sentenceWindows[gid].sort(function (a, b) { return b.risk - a.risk; });
  });
  var lastSentenceTrigger = null;

  /* ── 头部 chips ─────────────────────────────────── */
  function renderHeader() {
    document.getElementById("disclaimer-text").textContent = DATA.disclaimer || "";
    document.getElementById("provisional-note").textContent = DATA.provisional || "";
    var doc = DATA.doc || {};
    var bandClass = doc.band === "低" ? "band-low" : doc.band === "高" ? "band-high" : "band-mid";
    var scoreCard = document.getElementById("score-card");
    scoreCard.className = "score-card " + bandClass;
    scoreCard.textContent = "";
    scoreCard.appendChild(el("div", {
      class: "score-value",
      text: doc.score === null || doc.score === undefined ? "—" : String(doc.score),
    }));
    var scoreMeta = el("div", { class: "score-meta" });
    scoreMeta.appendChild(el("div", { class: "score-label", text: "综合风险分 / 100" }));
    scoreMeta.appendChild(el("div", { class: "score-band", text: "区间带：" + (doc.band || "—") }));
    scoreMeta.appendChild(el("div", {
      class: "score-note",
      text: "0–34 低 · 35–64 中 · 65–100 高；综合分用于快速比较，不改变区间带判定。",
    }));
    scoreCard.appendChild(scoreMeta);
    var chips = el("div", { class: "chips" });
    chips.appendChild(el("span", { class: "chip " + bandClass, html: "风险区间带 <b>" + (doc.band || "—") + "</b>" }));
    chips.appendChild(el("span", { class: "chip", html: "基调 median <b>" + fmt(doc.median) + "</b>" }));
    chips.appendChild(el("span", { class: "chip", html: "P90 <b>" + fmt(doc.p90) + "</b>" }));
    chips.appendChild(el("span", { class: "chip", html: "窗口 <b>" + (doc.n_windows || 0) + "</b>" }));
    chips.appendChild(el("span", { class: "chip", html: "句子 <b>" + (doc.n_sentences || 0) + "</b>" }));
    if (doc.burstiness !== null && doc.burstiness !== undefined) {
      chips.appendChild(el("span", { class: "chip", html: "突发性 <b>" + fmt(doc.burstiness) + "</b>" }));
    }
    chips.appendChild(el("span", { class: "chip", html: "用时 <b>" + fmt(DATA.duration_s, 1) + "s</b>" }));
    var profile = DATA.profile || {};
    if (profile.name && profile.name !== "custom") {
      chips.appendChild(el("span", {
        class: "chip",
        html: "运行档位 <b>" + (profile.label || profile.name) + "</b>",
        title: profile.description || "",
      }));
    }
    Object.keys(DATA.detectors || {}).forEach(function (key) {
      chips.appendChild(el("span", { class: "chip", text: PATH_LABELS[key] || key }));
    });
    if (DATA.enhanced && DATA.enhanced.enabled) {
      chips.appendChild(el("span", { class: "chip band-mid", html: "增强召回 <b>实验</b> · 额外 " + (DATA.enhanced.n_extra || 0) + " 窗" }));
    }
    var llm = DATA.llm_advice || {};
    if (llm.enabled) {
      chips.appendChild(el("span", {
        class: "chip band-mid",
        html: "具体建议 <b>实验</b> · " + (llm.n_generated || 0) + " 条",
        title: llm.reason || "",
      }));
    }
    if (DATA.guidance && DATA.guidance.evidence && DATA.guidance.evidence.sufficient === false) {
      chips.appendChild(el("span", { class: "chip band-mid", text: "证据不足 · 短文本仅供参考" }));
    }
    document.getElementById("chips").replaceWith(chips);
  }

  /* ── 热力图 ─────────────────────────────────────── */
  function renderHotCards() {
    var wrap = document.getElementById("hot-cards");
    wrap.textContent = "";
    (DATA.hot_intervals || []).forEach(function (item) {
      wrap.appendChild(
        el("div", { class: "hot-card" },
          el("b", { text: "重点复核区间 " }),
          el("span", { text: "块 " + item.block + " · 句 " + item.sentences.join("-") + " · 连续 " + item.n_windows + " 窗 · 均分 " + fmt(item.mean) }))
      );
    });
    (DATA.warnings || []).forEach(function (warning) {
      wrap.appendChild(el("div", { class: "hot-card", text: "⚠ " + warning }));
    });
    var guidance = DATA.guidance || {};
    if (guidance.headline) {
      wrap.appendChild(
        el("div", { class: "hot-card" },
          el("b", { text: guidance.headline }),
          el("span", { text: " " + ((guidance.evidence || {}).note || "") }))
      );
    }
    var enh = DATA.enhanced || {};
    if (enh.enabled) {
      wrap.appendChild(
        el("div", { class: "hot-card" },
          el("b", { text: "实验性增强召回" }),
          el("span", { text: " 额外标记 " + (enh.n_extra || 0) + " 个灰度窗口（增强命中共 " + (enh.n_flagged || 0) + " 个）。" + (enh.note || "") }))
      );
      (enh.windows || []).slice(0, 8).forEach(function (item) {
        wrap.appendChild(
          el("div", { class: "hot-card" },
            el("b", { text: "增强线索 · 窗口 " + item.key + " · 融合 " + fmt(item.risk) + " · meta " + fmt(item.meta_prob) }),
            el("div", { class: "quote-box", text: item.text.length > 160 ? item.text.slice(0, 160) + "…" : item.text }))
        );
      });
    }
  }

  function sentenceClick(gid, trigger) {
    lastSentenceTrigger = trigger || null;
    var windows = sentenceWindows[gid] || [];
    var best = windows[0];
    var text = "";
    var risk = null, disputed = false, reasonItems = [], section = "", reviewNote = null, llmAdvice = null;
    (DATA.blocks || []).some(function (block) {
      return (block.sentences || []).some(function (s) {
        if (s.gid === gid) {
          text = s.text; risk = s.risk; disputed = s.disputed; reasonItems = normalizeReasonItems(s);
          section = s.section || ""; reviewNote = s.review_note || null; llmAdvice = s.llm_advice || null;
          return true;
        }
        return false;
      });
    });
    var body = document.getElementById("drawer-body");
    body.textContent = "";
    document.getElementById("drawer-title").textContent = "句子 g" + gid + " · 风险 " + fmt(risk);
    body.appendChild(el("div", { class: "quote-box", text: text }));
    body.appendChild(el("div", { class: "kv" },
      el("span", { class: "k", text: "句级风险" }), el("span", { class: "v" }, riskBadge(risk)),
      el("span", { class: "k", text: "三路分歧" }), el("span", { class: "v", text: disputed ? "⚠ 是（取 δ 最高的窗口展示）" : "否" }),
      el("span", { class: "k", text: "所属章节" }), el("span", { class: "v", text: section || "未识别" })
    ));
    var flag = renderReviewNote(reviewNote);
    if (flag) body.appendChild(flag);
    body.appendChild(renderReasonDetails(reasonItems, risk, llmAdvice));
    if (best) {
      body.appendChild(el("div", { class: "k", text: "覆盖窗口 " + best.key + " 三路分数：" }));
      var kv = el("div", { class: "kv" });
      Object.keys(best.scores || {}).forEach(function (pathKey) {
        kv.appendChild(el("span", { class: "k", text: PATH_LABELS[pathKey] || pathKey }));
        var v = el("span", { class: "v" }, bar(best.scores[pathKey]), el("span", { text: " " + fmt(best.scores[pathKey]) }));
        kv.appendChild(v);
      });
      body.appendChild(kv);
      var cMeta = (best.metas || {}).c || {};
      if (cMeta.ppl !== undefined && cMeta.ppl !== null) {
        body.appendChild(el("div", { class: "kv" },
          el("span", { class: "k", text: "窗口 PPL" }), el("span", { class: "v", text: fmt(cMeta.ppl, 1) }),
          el("span", { class: "k", text: "p_ppl / p_style" }), el("span", { class: "v", text: fmt(cMeta.p_ppl) + " / " + fmt(cMeta.p_style) })
        ));
      }
      if (DATA.enhanced && DATA.enhanced.enabled && best.meta_prob !== undefined && best.meta_prob !== null) {
        body.appendChild(el("div", { class: "kv" },
          el("span", { class: "k", text: "增强召回" }),
          el("span", { class: "v", text: (best.enhanced ? "已标记" : "未标记") + " · meta " + fmt(best.meta_prob) })
        ));
      }
      body.appendChild(el("details", {}, el("summary", { text: "窗口原文" }), el("div", { class: "quote-box", text: best.text })));
    }
    openDrawer();
  }

  function renderBlocks() {
    var wrap = document.getElementById("blocks");
    wrap.textContent = "";
    (DATA.blocks || []).forEach(function (block) {
      if (!block.detectable) {
        wrap.appendChild(
          el("p", { class: "block-skipped" },
            el("span", { class: "skip-note", text: "〔" + (block.note || "已跳过") + "〕" }),
            el("span", { text: block.text.length > 120 ? block.text.slice(0, 120) + "…" : block.text }))
        );
        return;
      }
      var p = el("p", { class: "block-para" });
      (block.sentences || []).forEach(function (s) {
        p.appendChild(
          el("span", {
            class: "sent " + heatClass(s.risk) + (s.disputed ? " disputed" : "") + (s.enhanced ? " enhanced" : ""),
            text: s.text,
            role: "button",
            tabindex: "0",
            "aria-haspopup": "dialog",
            onclick: function (event) { sentenceClick(s.gid, event.currentTarget); },
            onkeydown: function (event) {
              if (event.key === "Enter" || event.key === " ") {
                event.preventDefault();
                sentenceClick(s.gid, event.currentTarget);
              }
            },
          })
        );
      });
      wrap.appendChild(p);
    });
  }

  /* ── 特征面板 ───────────────────────────────────── */
  function renderFeatures() {
    var wrap = document.getElementById("feature-chips");
    wrap.textContent = "";
    var doc = DATA.doc || {};
    wrap.appendChild(el("span", { class: "chip", html: "突发性 B = std/mean <b>" + fmt(doc.burstiness) + "</b>（人写偏高，AI 偏低）" }));
    wrap.appendChild(el("span", { class: "chip", html: "PPL 曲线窗口数 <b>" + ((DATA.ppl_curve || []).length) + "</b>" }));
    var chartDom = document.getElementById("ppl-chart");
    if (!window.echarts) {
      chartDom.textContent = "图表库未能加载，PPL 数据见窗口明细表。";
      return;
    }
    var chart = window.echarts.getInstanceByDom(chartDom);
    if (!chart) chart = window.echarts.init(chartDom, null, { backgroundColor: "transparent" });
    var points = DATA.ppl_curve || [];
    chart.setOption({
      tooltip: {
        trigger: "axis",
        backgroundColor: "rgba(255,255,255,0.97)",
        borderColor: "#e4ebf7",
        borderWidth: 1,
        padding: [8, 11],
        textStyle: { color: "#14224a", fontSize: 12 },
        extraCssText: "box-shadow:0 8px 24px rgba(35,72,140,.12);border-radius:10px;",
        axisPointer: { type: "line", lineStyle: { color: "#c9d8f0" } },
      },
      grid: { left: 56, right: 24, top: 34, bottom: 40 },
      xAxis: {
        type: "category",
        data: points.map(function (p) { return p.key; }),
        axisLine: { lineStyle: { color: "#d6e2f4" } },
        axisTick: { show: false },
        axisLabel: { color: "#7c8ba8", fontSize: 9, interval: "auto", hideOverlap: true },
      },
      yAxis: {
        type: "value",
        name: "PPL",
        scale: true,
        nameTextStyle: { color: "#7c8ba8", fontSize: 10 },
        axisLine: { show: false },
        axisTick: { show: false },
        axisLabel: { color: "#7c8ba8", fontSize: 10 },
        splitLine: { lineStyle: { color: "#edf2fa" } },
      },
      series: [{
        name: "窗口 PPL", type: "line", data: points.map(function (p) { return p.ppl; }),
        symbol: "circle", symbolSize: 5,
        lineStyle: { width: 2, color: "#598bfd" },
        itemStyle: { color: "#598bfd", borderColor: "#fff", borderWidth: 1.5 },
        areaStyle: { color: "rgba(89,139,253,0.10)" },
        markLine: {
          silent: true,
          symbol: "none",
          lineStyle: { color: "#d99022", type: "dashed", width: 1 },
          label: { color: "#a96a0b", fontSize: 10 },
          data: [{ yAxis: 0, name: "" }],
        },
      }],
    });
    chart.resize();
    if (!chartDom.__selfcheckResize) {
      chartDom.__selfcheckResize = true;
      window.addEventListener("resize", function () { chart.resize(); });
    }
  }

  /* ── 高风险清单 / 窗口明细 ──────────────────────── */
  function renderGuidance() {
    var wrap = document.getElementById("guidance");
    if (!wrap) return;
    var guidance = DATA.guidance || {};
    var evidence = guidance.evidence || {};
    wrap.textContent = "";
    wrap.appendChild(el("h3", { text: guidance.headline || "自查建议" }));
    if (guidance.intro) {
      wrap.appendChild(el("p", { class: "hint", text: guidance.intro }));
    }
    if (evidence.note) {
      wrap.appendChild(el("p", { class: evidence.sufficient === false ? "warn-flag" : "hint", text: evidence.note }));
    }
    if ((guidance.steps || []).length) {
      wrap.appendChild(el("div", { class: "k", text: "使用步骤" }));
      var steps = el("ol", { class: "reason-list" });
      guidance.steps.forEach(function (step) { steps.appendChild(el("li", { text: step })); });
      wrap.appendChild(steps);
    }
    if ((guidance.suggestions || []).length) {
      wrap.appendChild(el("div", { class: "k", text: "修改建议" }));
      var suggestions = el("ul", { class: "reason-list" });
      guidance.suggestions.forEach(function (item) { suggestions.appendChild(el("li", { text: item })); });
      wrap.appendChild(suggestions);
    }
  }

  function copyButton(text) {
    return el("button", {
      class: "copy",
      text: "复制",
      onclick: function () {
        var self = this;
        navigator.clipboard.writeText(text).then(function () {
          self.textContent = "已复制";
          setTimeout(function () { self.textContent = "复制"; }, 1200);
        });
      },
    });
  }

  function renderTop() {
    var tbody = document.querySelector("#top-table tbody");
    tbody.textContent = "";
    var rows = (DATA.top_sentences || []).filter(function (item) { return item.risk >= 0.5; });
    if (!rows.length) {
      tbody.appendChild(el("tr", {},
        el("td", { colspan: 6, class: "hint", text: "没有达到关注阈值的句子；请看热力图或窗口明细。" })
      ));
      return;
    }
    rows.forEach(function (item, i) {
      var items = normalizeReasonItems(item);
      var whyCell = renderReasonWhy(items, item.risk, 2, item.review_note, item.llm_advice);
      if (item.enhanced) {
        whyCell.insertBefore(el("div", { class: "reason-more", text: "【增强召回标记】" }), whyCell.firstChild);
      }
      tbody.appendChild(
        el("tr", {},
          el("td", { class: "num", text: String(i + 1) }),
          el("td", {}, riskBadge(item.risk)),
          el("td", {}, el("div", { class: "quote-box", text: item.text })),
          el("td", {}, whyCell),
          el("td", {}, renderSuggestionCell(items, item.risk)),
          el("td", {}, copyButton(item.text)))
      );
    });
  }

  function renderWindows() {
    var tbody = document.querySelector("#window-table tbody");
    tbody.textContent = "";
    (DATA.windows || []).forEach(function (w) {
      var scores = w.scores || {};
      var cells = [
        el("td", { class: "num", text: w.key }),
        el("td", { class: "num", text: String(w.block) }),
        el("td", {}, riskBadge(w.risk)),
        el("td", { class: "num" }, el("span", { text: fmt(w.delta) + (w.disputed ? " ⚠" : ""), class: w.disputed ? "warn-flag" : "" })),
        el("td", { class: "num", text: fmt(scores.a) }),
        el("td", { class: "num", text: fmt(scores.b) }),
        el("td", { class: "num", text: fmt(scores.c) }),
        el("td", { class: "num" + (w.enhanced ? " warn-flag" : ""), text: w.meta_prob === null || w.meta_prob === undefined ? "—" : fmt(w.meta_prob) + (w.enhanced ? " ⚠" : "") }),
        el("td", {}, renderReasonAndFix(normalizeReasonItems(w), w.risk, 2, w.review_note, w.llm_advice)),
        el("td", {}, copyButton(w.text)),
      ];
      var tr = el("tr", {});
      cells.forEach(function (cell) { tr.appendChild(cell); });
      tbody.appendChild(tr);
    });
  }

  /* ── 章节地图 / 修改计划 / 校准卡 ─────────────────── */
  function renderSections() {
    var tbody = document.querySelector("#section-table tbody");
    if (!tbody) return;
    tbody.textContent = "";
    var sections = DATA.sections || [];
    if (!sections.length) {
      tbody.appendChild(el("tr", {},
        el("td", { colspan: "8", class: "hint", text: "没有可聚合的章节：文档可能没有标题结构，或内容都被结构过滤跳过。" })));
      return;
    }
    sections.forEach(function (s) {
      var title = el("div", { class: "section-title", text: s.title || "未命名章节" });
      if (s.method_like) title.appendChild(el("span", { class: "tag", text: "结构敏感段" }));
      tbody.appendChild(el("tr", {},
        el("td", {}, title),
        el("td", { class: "num", text: String(s.n_windows || 0) }),
        el("td", { class: "num", text: String(s.n_cjk || 0) }),
        el("td", { class: "num", text: fmt(s.median) }),
        el("td", { class: "num", text: fmt(s.p90) }),
        el("td", { class: "num" }, el("b", {
          class: "score-mini " + bandClass(s.band),
          text: s.score === null || s.score === undefined ? "—" : String(s.score),
        })),
        el("td", {}, bandBadge(s.band)),
        el("td", { class: "hint", text: s.hint || "" })
      ));
    });
  }

  function renderPlan() {
    var wrap = document.getElementById("plan-cards");
    var summary = document.getElementById("plan-summary");
    if (!wrap || !summary) return;
    wrap.textContent = "";
    summary.textContent = "";
    summary.appendChild(el("h3", { text: "优先级说明" }));
    summary.appendChild(el("p", {
      class: "hint",
      text: "按「单独把这一段的风险归零后，全文综合分会下降多少」排序。它是敏感性估计，不是分数承诺；改完请重新检测。",
    }));
    var plan = DATA.plan || [];
    if (!plan.length) {
      summary.appendChild(el("p", {
        class: "hint",
        text: "没有单段主导风险。建议按「高风险清单」逐句复核，不要为了降分大改全文。",
      }));
      return;
    }
    plan.forEach(function (item, i) {
      var card = el("div", { class: "plan-card" });
      var impactText = item.impact > 0
        ? "预计影响 -" + item.impact + " 分"
        : "单窗风险高，分数影响小";
      card.appendChild(el("div", { class: "plan-head" },
        el("span", { class: "plan-rank", text: "#" + (i + 1) }),
        el("span", { class: "plan-impact", text: impactText }),
        el("span", {
          class: "hint",
          text: "若单独处理这一段：综合分 " + (item.score_without === undefined ? "—" : item.score_without)
            + " · 区间带 " + (item.band_without || "—"),
        })
      ));
      card.appendChild(el("div", { class: "plan-meta" },
        el("span", { text: "块 " + item.block }),
        el("span", { text: "章节：" + (item.section || "未识别") }),
        el("span", { text: "窗口 " + item.n_windows }),
        el("span", { text: "最高风险 " + fmt(item.max_risk) })
      ));
      var note = renderReviewNote(item.review_note);
      if (note) card.appendChild(note);
      var text = item.text || "";
      card.appendChild(el("div", {
        class: "quote-box",
        text: text.length > 220 ? text.slice(0, 220) + "…" : text,
      }));
      var items = normalizeReasonItems(item);
      if (items.length) card.appendChild(renderReasonAndFix(items, item.max_risk, 2, null, item.llm_advice));
      if (text) card.appendChild(copyButton(text));
      wrap.appendChild(card);
    });
  }

  function renderCalibration() {
    var wrap = document.getElementById("calibration-card");
    if (!wrap) return;
    wrap.textContent = "";
    var cal = DATA.calibration || {};
    wrap.appendChild(el("h3", { text: "校准卡 · " + (cal.version || "—") }));
    var table = el("table", { class: "grid" });
    table.appendChild(el("thead", {}, el("tr", {},
      el("th", { text: "数据集" }),
      el("th", { text: "指标" }),
      el("th", { text: "结果" })
    )));
    var tbody = el("tbody");
    function addRow(label, metric, value) {
      if (value === undefined || value === null || value === "") return;
      tbody.appendChild(el("tr", {},
        el("td", { text: label }),
        el("td", { text: metric }),
        el("td", { class: "num", text: String(value) })
      ));
    }
    var g = cal.general || {};
    if (g.label) {
      addRow(g.label, "融合 AUC", g.auc);
      addRow("", "人写误伤", g.human_fp);
      addRow("", "召回", g.recall);
    }
    var ad = cal.academic_default || {};
    if (ad.label) {
      addRow(ad.label, "AI 窗口召回", ad.ai_window_recall);
      addRow("", "人写窗口误伤", ad.human_window_fp);
      addRow("", "人写文档「高」带", ad.human_doc_high);
    }
    var ae = cal.academic_enhanced || {};
    if (ae.label) {
      addRow(ae.label, "AI 窗口召回", ae.ai_window_recall);
      addRow("", "人写窗口误伤", ae.human_window_fp);
    }
    var c = cal.combined || {};
    if (c.label) {
      addRow(c.label, "默认 P / R / F1", c.default_prf);
      addRow("", "增强 P / R / F1", c.enhanced_prf);
    }
    table.appendChild(tbody);
    wrap.appendChild(table);
    var notes = el("ul", { class: "reason-list" });
    (cal.notes || []).forEach(function (note) { notes.appendChild(el("li", { text: note })); });
    wrap.appendChild(notes);
  }

  /* ── 抽屉与标签页 ───────────────────────────────── */
  function openDrawer() {
    var drawer = document.getElementById("drawer");
    drawer.classList.add("open");
    drawer.setAttribute("aria-hidden", "false");
    document.getElementById("drawer-mask").classList.add("open");
    document.body.classList.add("drawer-open");
    document.getElementById("drawer-close").focus();
  }
  function closeDrawer() {
    var drawer = document.getElementById("drawer");
    var wasOpen = drawer.classList.contains("open");
    drawer.classList.remove("open");
    drawer.setAttribute("aria-hidden", "true");
    document.getElementById("drawer-mask").classList.remove("open");
    document.body.classList.remove("drawer-open");
    if (wasOpen && lastSentenceTrigger && lastSentenceTrigger.isConnected) {
      lastSentenceTrigger.focus();
    }
  }
  document.getElementById("drawer-close").addEventListener("click", closeDrawer);
  document.getElementById("drawer-mask").addEventListener("click", closeDrawer);
  document.addEventListener("keydown", function (e) { if (e.key === "Escape") closeDrawer(); });

  function switchTab(name) {
    ["overview", "sections", "plan", "top", "windows", "features"].forEach(function (tab) {
      document.getElementById("view-" + tab).hidden = tab !== name;
    });
    document.querySelectorAll("#tabs button").forEach(function (btn) {
      btn.classList.toggle("active", btn.getAttribute("data-tab") === name);
    });
    if (name === "sections") renderSections();
    else if (name === "plan") renderPlan();
    else if (name === "features") renderFeatures();
  }
  document.querySelectorAll("#tabs button").forEach(function (btn) {
    btn.addEventListener("click", function () { switchTab(btn.getAttribute("data-tab")); });
  });

  /* ── 启动 ───────────────────────────────────────── */
  renderHeader();
  renderHotCards();
  renderBlocks();
  renderGuidance();
  renderSections();
  renderPlan();
  renderTop();
  renderWindows();
  renderCalibration();
  document.getElementById("footer-meta").textContent =
    "aigc-selfcheck v" + DATA.tool_version + " · 生成于 " + DATA.generated_at + " · 本地分析，文本未离开本机";
})();
