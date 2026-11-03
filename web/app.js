/* Static replay + results viewer. No build step, no network except local JSON/video. */
(function () {
  "use strict";

  const ARMS = {
    no_verifier: { label: "No verifier", note: "One attempt; assumes it worked." },
    verifier_blind: { label: "Verifier + blind retry", note: "Checks from pixels; retries the same way." },
    verifier_corrected: { label: "Verifier + corrected retry", note: "Checks from pixels; shifts the release by the measured error." },
    verifier_damped: { label: "Verifier + damped correction", note: "Feeds back only 40% of the measured error." },
    oracle_verifier_corrected: { label: "Oracle verifier + corrected retry", note: "Upper bound: judges from simulator truth." },
  };
  const PROFILES = {
    nominal: "No injected faults.",
    noisy: "Random grasp and release error (sigma 2.0 / 2.5 cm).",
    biased: "Constant 5.4 cm release calibration error.",
    slippery: "Half of the grasps drop the block mid-carry.",
  };
  const TASKS = { pick_place: "Pick & place", stack: "Stack", two_step: "Two-step" };
  const STORIES = {
    "biased-fix": ["Silent failure, then fixed", "A systematic 5.4 cm release error. Compare what each arm does about it."],
    "slip-recover": ["Dropped mid-carry", "The block slips out of the gripper. A verifier notices; a retry re-grasps where it fell."],
    "nominal-twostep": ["Two steps, no faults", "Put one block in the bin, then stack another. Perception drives both."],
    "budget-exhausted": ["When it still fails", "Noisy execution and the attempt budget runs out. Failures are part of the result."],
  };
  const LIMITS = [
    "Simulation only: a small 6-DOF arm in MuJoCo with four coloured cubes and a bin.",
    "Skills are scripted waypoints, not a learned policy. The carry uses a kinematic grasp assist; arm motion, collisions and the final drop are physics.",
    "Faults are injected on purpose and their sizes are my choice, so the size of any gain depends on them. The profiles are listed in the table above.",
    "The camera is a fixed overhead RGB-D sensor with Gaussian noise. Perception is colour segmentation plus depth. Between looks the arm is moved out of view instantly.",
    "The verifier and the ground-truth scorer share one geometric definition of success, so their agreement measures perception, not task definition.",
    "The ManiSkill3 results (see the repository) are a separate, smaller study and are not mixed into these tables.",
  ];

  const $ = (sel, root = document) => root.querySelector(sel);
  const pct = (x) => (100 * x).toFixed(0) + "%";
  const pct1 = (x) => (100 * x).toFixed(x > 0.98 && x < 1 ? 1 : 0) + "%";
  function h(tag, attrs, ...kids) {
    const el = document.createElement(tag);
    for (const [k, v] of Object.entries(attrs || {})) {
      if (v === false || v == null) continue;
      if (k === "class") el.className = v;
      else if (k === "text") el.textContent = v;
      else if (k.startsWith("on")) el.addEventListener(k.slice(2), v);
      else el.setAttribute(k, v === true ? "" : v);
    }
    for (const kid of kids.flat()) if (kid != null) el.append(kid.nodeType ? kid : document.createTextNode(kid));
    return el;
  }
  async function getJSON(path) {
    try { const r = await fetch(path); return r.ok ? await r.json() : null; } catch { return null; }
  }

  const state = { results: null, index: null, planners: null, mani: null, episodes: {}, profile: "slippery",
                  story: null, arm: null, sub: 0, attempt: 0, media: "rollout" };

  /* Router ----------------------------------------------------------- */
  const VIEWS = ["overview", "replays", "results", "method"];
  function show(view) {
    if (!VIEWS.includes(view)) view = "overview";
    for (const v of VIEWS) {
      $("#view-" + v).hidden = v !== view;
      $("#tab-" + v).setAttribute("aria-selected", v === view ? "true" : "false");
      $("#tab-" + v).tabIndex = v === view ? 0 : -1;
    }
    if (location.hash.slice(1) !== view) history.replaceState(null, "", "#" + view);
    window.scrollTo({ top: 0 });
  }
  document.querySelectorAll("[data-view]").forEach((b) => b.addEventListener("click", () => show(b.dataset.view)));
  document.querySelectorAll("[data-go]").forEach((b) => b.addEventListener("click", () => show(b.dataset.go)));
  window.addEventListener("hashchange", () => show(location.hash.slice(1)));
  $(".tabs").addEventListener("keydown", (e) => {
    if (e.key !== "ArrowRight" && e.key !== "ArrowLeft") return;
    const i = VIEWS.indexOf(location.hash.slice(1) || "overview");
    const n = (i + (e.key === "ArrowRight" ? 1 : VIEWS.length - 1)) % VIEWS.length;
    show(VIEWS[n]); $("#tab-" + VIEWS[n]).focus();
  });

  $("#theme").addEventListener("click", () => {
    const cur = document.documentElement.dataset.theme ||
      (matchMedia("(prefers-color-scheme: light)").matches ? "light" : "dark");
    const next = cur === "light" ? "dark" : "light";
    document.documentElement.dataset.theme = next;
    try { localStorage.setItem("theme", next); } catch { /* private mode */ }
  });

  /* Overview stats ---------------------------------------------------- */
  function renderStats() {
    const box = $("#stats"); box.replaceChildren();
    const R = state.results;
    if (!R) { box.append(h("div", { class: "stat" }, h("div", { class: "k", text: "Results not generated yet" }),
      h("div", { class: "d", text: "Run python -m bench.run, then python -m bench.report." }))); return; }
    const profiles = Object.keys(R.pooled);
    let best = null, bestC = null;
    for (const p of profiles) {
      const a = R.pooled[p].no_verifier, b = R.pooled[p].verifier_blind, c = R.pooled[p].verifier_corrected;
      if (a && b && (!best || b.rate - a.rate > best.gain)) best = { p, a, b, gain: b.rate - a.rate, pair: R.paired[p].verifier_blind_vs_none };
      if (b && c && (!bestC || c.rate - b.rate > bestC.gain)) bestC = { p, b, c, gain: c.rate - b.rate, pair: R.paired[p].corrected_vs_blind };
    }
    const sig = (pr) => (pr.p < 0.001 ? "p < 0.001" : "p = " + pr.p.toFixed(3));
    if (best) box.append(stat("Verifier, vs. no verifier", [pct(best.a.rate), pct(best.b.rate)],
      `Under “${best.p}” faults, ${best.a.n} paired episodes, ${sig(best.pair)} (exact McNemar).`));
    if (bestC) box.append(stat("Corrected retry, vs. blind retry", [pct(bestC.b.rate), pct(bestC.c.rate)],
      `Under “${bestC.p}” faults, ${bestC.b.n} paired episodes, ${sig(bestC.pair)}.`));
    const c = R.verifier_confusion, tot = c.accept_correct + c.false_accept + c.reject_correct + c.false_reject;
    box.append(stat("Camera verifier agrees with ground truth", [pct1((c.accept_correct + c.reject_correct) / (tot || 1))],
      `${tot} attempt-level verdicts: ${c.false_accept} false accepts, ${c.false_reject} false rejects.`));
  }
  function stat(k, vals, d) {
    const v = h("div", { class: "v" });
    vals.forEach((x, i) => { if (i) v.append(h("span", { class: "arrow", text: "→" })); v.append(x); });
    return h("div", { class: "stat" }, h("div", { class: "k", text: k }), v, h("div", { class: "d", text: d }));
  }

  /* Replays ------------------------------------------------------------ */
  function groups() {
    const g = {};
    for (const e of state.index.episodes) {
      const label = e.id.slice(0, e.id.length - e.arm.length - 1);
      (g[label] = g[label] || []).push(e);
    }
    const order = Object.keys(STORIES);
    return Object.entries(g).sort((a, b) => order.indexOf(a[0]) - order.indexOf(b[0]));
  }
  function renderStories() {
    const list = $("#stories"); list.replaceChildren();
    for (const [label, eps] of groups()) {
      const [title, sub] = STORIES[label] || [label, ""];
      const btn = h("button", { class: "story", type: "button", "aria-current": state.story === label ? "true" : "false",
        onclick: () => { state.story = label; state.arm = eps[eps.length - 1].arm; state.sub = 0; state.attempt = 0; state.media = "rollout"; renderStories(); renderPlayer(); } },
        h("span", { class: "t", text: title }), h("span", { class: "s", text: sub }),
        h("span", { class: "chips" },
          eps.map((e) => h("span", { class: "chip " + (e.oracle_success ? "ok" : "bad"), text: ARMS[e.arm].label.replace(" + ", " + ") + ": " + (e.oracle_success ? "solved" : "failed") }))));
      list.append(btn);
    }
  }
  async function loadEpisode(id) {
    if (!state.episodes[id]) state.episodes[id] = await getJSON(`data/episodes/${id}.json`);
    return state.episodes[id];
  }
  async function renderPlayer() {
    const box = $("#player");
    if (!state.index || !state.index.episodes.length) {
      box.replaceChildren(h("p", { class: "empty", text: "No recorded rollouts yet. Run python -m bench.record." })); return;
    }
    if (!state.story) { state.story = groups()[0][0]; state.arm = groups()[0][1].slice(-1)[0].arm; }
    const eps = Object.fromEntries(groups())[state.story];
    const meta = eps.find((e) => e.arm === state.arm) || eps[0];
    state.arm = meta.arm;
    const ep = await loadEpisode(meta.id);
    if (!ep) { box.replaceChildren(h("p", { class: "empty", text: "Episode data missing." })); return; }
    state.sub = Math.min(state.sub, ep.subtasks.length - 1);
    const st = ep.subtasks[state.sub];
    state.attempt = Math.min(state.attempt, st.attempts.length - 1);
    const at = st.attempts[state.attempt];
    const [title] = STORIES[state.story] || [state.story];

    const armRow = eps.length > 1 ? h("div", { class: "row", role: "group", "aria-label": "Arm" }, h("span", { class: "label", text: "Arm" }),
      eps.map((e) => h("button", { class: "pill", type: "button", "aria-pressed": String(e.arm === state.arm),
        onclick: () => { state.arm = e.arm; state.sub = 0; state.attempt = 0; renderPlayer(); }, text: ARMS[e.arm].label }))) : null;
    const subRow = ep.subtasks.length > 1 ? h("div", { class: "row", role: "group", "aria-label": "Subtask" }, h("span", { class: "label", text: "Subtask" }),
      ep.subtasks.map((s, i) => h("button", { class: "pill", type: "button", "aria-pressed": String(i === state.sub),
        onclick: () => { state.sub = i; state.attempt = 0; renderPlayer(); }, text: `${i + 1}. ${s.action.replace("_", " ")} ${s.object.split("_")[0]}${s.target === "bin" ? " → bin" : " on " + s.target.split("_")[0]}` }))) : null;
    const attRow = h("div", { class: "row", role: "group", "aria-label": "Attempt" }, h("span", { class: "label", text: "Attempt" }),
      st.attempts.map((a, i) => h("button", { class: "pill", type: "button", "aria-pressed": String(i === state.attempt),
        onclick: () => { state.attempt = i; renderPlayer(); } },
        h("span", { class: "g " + (a.observed_by === "none" ? "" : a.verdict ? "ok" : "bad"), text: a.observed_by === "none" ? "·" : a.verdict ? "✓" : "✗" }), "Attempt " + a.attempt)));

    const showImg = state.media === "verify" && at.verify_image;
    const media = h("div", { class: "media" },
      showImg ? h("img", { src: "data/" + at.verify_image, alt: "Overhead camera view with detected blocks and the target marked" })
              : at.video ? h("video", { src: "data/" + at.video, controls: true, muted: true, autoplay: true, loop: true, playsinline: true, preload: "metadata", "aria-label": "Rollout video" })
                         : h("p", { class: "empty", text: "No clip for this attempt (no motion was executed)." }),
      h("span", { class: "cap", text: showImg ? "What the verifier saw" : "Rollout" }));
    const mediaRow = at.verify_image ? h("div", { class: "row", role: "group", "aria-label": "View" },
      h("button", { class: "pill", type: "button", "aria-pressed": String(!showImg), onclick: () => { state.media = "rollout"; renderPlayer(); }, text: "Rollout" }),
      h("button", { class: "pill", type: "button", "aria-pressed": String(showImg), onclick: () => { state.media = "verify"; renderPlayer(); }, text: "Verifier's view" })) : null;

    const fmt = (v) => (v == null ? "n/a" : `(${v[0] >= 0 ? "+" : ""}${v[0].toFixed(1)}, ${v[1] >= 0 ? "+" : ""}${v[1].toFixed(1)}) cm`);
    const unverified = at.observed_by === "none";
    const verdict = h("div", { class: "verdict " + (unverified ? "na" : at.verdict ? "ok" : "bad") },
      h("div", { class: "h", text: unverified ? "Robot's check" : `Robot's check (${at.observed_by})` }),
      h("div", { class: "b", text: unverified ? "Not checked" : at.verdict ? "Looks done" : "Not done" }),
      h("div", { class: "n", text: unverified ? "No verifier: it assumes the attempt worked." : at.reason }));
    const truth = h("div", { class: "verdict " + (at.oracle_success ? "ok" : "bad") },
      h("div", { class: "h", text: "Ground truth (scoring only)" }),
      h("div", { class: "b", text: at.oracle_success ? "Succeeded" : "Failed" }),
      h("div", { class: "n", text: !unverified && at.verdict !== at.oracle_success ? "The camera verifier was wrong here." : "The robot never sees this." }));
    const kv = h("dl", { class: "kv" },
      h("dt", { text: "Measured error" }), h("dd", { text: fmt(at.error_cm) }),
      h("dt", { text: "Correction applied" }), h("dd", { text: at.correction_applied_cm ? fmt(at.correction_applied_cm) : "none" }),
      h("dt", { text: "Next correction" }), h("dd", { text: at.correction_next_cm ? fmt(at.correction_next_cm) : "none" }),
      h("dt", { text: "Gripper held block" }), h("dd", { text: at.grasp_detected ? "yes" : "no" }));

    box.replaceChildren(...[
      h("h3", { text: title }),
      h("p", { class: "instr", text: `“${ep.instruction}” · ${ep.profile} faults · seed ${ep.seed}` }),
      h("div", { class: "row" },
        h("span", { class: "chip " + (ep.believed_success ? "ok" : "bad"), text: "Robot reports: " + (ep.believed_success ? "success" : "failure") }),
        h("span", { class: "chip " + (ep.oracle_success ? "ok" : "bad"), text: "Ground truth: " + (ep.oracle_success ? "success" : "failure") }),
        h("span", { class: "chip", text: ep.attempts + " attempt" + (ep.attempts === 1 ? "" : "s") })),
      armRow, subRow, attRow, mediaRow,
      h("div", { class: "stage" }, media, h("div", { class: "facts" }, verdict, truth, kv)),
      h("p", { class: "hint" }, "Keys: ", h("kbd", { text: "←" }), " ", h("kbd", { text: "→" }), " attempts · ", h("kbd", { text: "↑" }), " ", h("kbd", { text: "↓" }), " arms")].filter(Boolean));
  }
  document.addEventListener("keydown", (e) => {
    if (location.hash !== "#replays" || /input|textarea|select/i.test(e.target.tagName) || !state.index) return;
    const eps = state.story && Object.fromEntries(groups())[state.story]; if (!eps) return;
    const ep = state.episodes[(eps.find((x) => x.arm === state.arm) || eps[0]).id]; if (!ep) return;
    const n = ep.subtasks[state.sub].attempts.length;
    if (e.key === "ArrowRight") state.attempt = Math.min(n - 1, state.attempt + 1);
    else if (e.key === "ArrowLeft") state.attempt = Math.max(0, state.attempt - 1);
    else if (e.key === "ArrowDown" || e.key === "ArrowUp") {
      const i = eps.findIndex((x) => x.arm === state.arm);
      state.arm = eps[(i + (e.key === "ArrowDown" ? 1 : eps.length - 1)) % eps.length].arm; state.attempt = 0; state.sub = 0;
    } else return;
    e.preventDefault(); renderPlayer();
  });

  /* Results ------------------------------------------------------------ */
  function cell(c) {
    if (!c) return h("td", { text: "–" });
    return h("td", {}, h("div", { class: "cellwrap" },
      h("span", { class: "num", text: pct(c.rate) }), h("span", { class: "ci", text: `${pct(c.lo)}–${pct(c.hi)} · n=${c.n}` }),
      h("div", { class: "bar", role: "img", "aria-label": `${pct(c.rate)} success, 95% interval ${pct(c.lo)} to ${pct(c.hi)}` },
        h("span", { style: `width:${100 * c.rate}%` }), h("i", { style: `left:${100 * c.lo}%;width:${Math.max(0.6, 100 * (c.hi - c.lo))}%` }))));
  }
  function renderResults() {
    const R = state.results;
    const seg = $("#profile-seg"); seg.replaceChildren();
    if (!R) { $("#results-table").replaceChildren(h("p", { class: "empty", text: "Results not generated yet." })); return; }
    const m = R.meta;
    $("#results-sub").textContent = `${m.episodes} seeds per task × fault profile × arm, attempt budget ${m.attempt_budget}. Success is judged by simulator ground truth at the end. Bars show 95% Wilson intervals.`;
    for (const p of Object.keys(PROFILES)) if (R.pooled[p]) seg.append(h("button", { type: "button", role: "radio", "aria-checked": String(p === state.profile),
      text: p, onclick: () => { state.profile = p; renderResults(); } }));
    $("#profile-desc").textContent = PROFILES[state.profile] || "";
    const pooled = R.pooled[state.profile] || {};
    const best = Object.entries(pooled).filter(([a]) => a !== "oracle_verifier_corrected").sort((a, b) => b[1].rate - a[1].rate)[0];
    const rows = Object.keys(ARMS).filter((a) => pooled[a]).map((a) =>
      h("tr", { class: best && best[0] === a ? "hl" : "" },
        h("td", { class: "arm" }, ARMS[a].label, h("small", { text: ARMS[a].note })),
        cell(pooled[a]),
        Object.keys(TASKS).map((t) => cell(((R.per_task[t] || {})[state.profile] || {})[a]))));
    $("#results-table").replaceChildren(h("div", { class: "tbl-scroll" }, h("table", {},
      h("thead", {}, h("tr", {}, h("th", { text: "Arm" }), h("th", { text: "All tasks" }), Object.values(TASKS).map((t) => h("th", { text: t })))),
      h("tbody", {}, rows))));

    const P = R.paired[state.profile];
    const line = (q, a, pr) => h("div", { class: "pair" }, h("div", {}, h("div", { class: "q", text: q }),
      h("div", { class: "a", text: `${pr.only_a} episodes only the first solved, ${pr.only_b} only the second, of ${pr.n_pairs} paired.` })),
      h("div", { class: "sig " + (pr.p < 0.05 ? "yes" : "no"), text: (pr.p < 0.001 ? "p < 0.001" : "p = " + pr.p.toFixed(3)) + (pr.p < 0.05 ? " ✓" : " (n.s.)") }));
    $("#paired").replaceChildren(h("h3", { text: "Paired comparisons on identical seeds" }),
      h("p", { class: "muted small", text: "Exact McNemar test on episodes where exactly one of the two arms succeeded." }),
      h("div", { class: "pairs" },
        line("Verifier (blind retry) vs. no verifier", 1, P.verifier_blind_vs_none),
        line("Corrected retry vs. blind retry", 1, P.corrected_vs_blind),
        line("Camera verifier vs. oracle verifier (both corrected)", 1, P.perception_vs_oracle_verifier)));

    const c = R.verifier_confusion;
    $("#confusion").replaceChildren(h("h3", { text: "Is the camera verifier right?" }),
      h("p", { class: "muted small", text: "Every attempt’s verdict against ground truth, across all runs with the camera verifier." }),
      h("div", { class: "conf" },
        h("div", {}, h("b", { text: c.accept_correct }), h("span", { text: "accepted, and it was done" })),
        h("div", {}, h("b", { text: c.false_accept }), h("span", { text: "false accepts (said done, wasn’t)" })),
        h("div", {}, h("b", { text: c.reject_correct }), h("span", { text: "rejected, and it wasn’t done" })),
        h("div", {}, h("b", { text: c.false_reject }), h("span", { text: "false rejects (said not done, was)" }))));
    renderPlanners();
    renderMani();
  }
  function renderMani() {
    const box = $("#maniskill"), M = state.mani;
    box.hidden = !M;
    if (!M) return;
    const rows = Object.entries(M.summary).map(([prof, s]) => h("tr", {}, h("td", { class: "arm", text: prof }),
      ["no_verifier", "verifier_blind", "verifier_corrected"].map((a) => s.arms[a] ? cell(s.arms[a]) : h("td", { text: "–" }))));
    box.replaceChildren(h("h3", { text: `Standard benchmark: ManiSkill3 ${M.env}` }),
      h("p", { class: "muted small", text: `Official environment and success metric, ${M.episodes} seeds per cell, attempt budget ${M.attempt_budget}. Cube localised from the RGB-D camera; ManiSkill's own motion planner executes; the same fault profiles are injected. Runs in Docker (CPU simulation, software rendering).` }),
      h("div", { class: "tbl-scroll" }, h("table", {}, h("thead", {}, h("tr", {}, ["Fault profile", "No verifier", "Verifier + blind retry", "Verifier + corrected retry"].map((t) => h("th", { text: t })))), h("tbody", {}, rows))));
  }
  function renderPlanners() {
    const box = $("#planners"); const P = state.planners;
    if (!P) { box.replaceChildren(h("h3", { text: "Local planner models" }), h("p", { class: "empty", text: "Planner comparison not generated yet." })); return; }
    const rows = (P.planner_eval || []).map((r) => h("tr", {}, h("td", { class: "arm", text: r.model }), h("td", { text: pct(r.valid_json) }),
      h("td", { class: "num", text: pct(r.plan_accuracy) }), h("td", { text: r.median_latency_s.toFixed(2) + " s" })));
    const e2e = (P.end_to_end || []).map((r) => h("tr", {}, h("td", { class: "arm", text: r.model }),
      h("td", { text: r.arms.no_verifier ? pct(r.arms.no_verifier.rate) : "–" }),
      h("td", { class: "num", text: r.arms.verifier_corrected ? pct(r.arms.verifier_corrected.rate) : "–" })));
    box.replaceChildren(h("h3", { text: "Local planner models" }),
      h("p", { class: "muted small", text: "Quantised models served by Ollama on a 6 GB laptop GPU. “heuristic” is the regex fallback." }),
      h("div", { class: "tbl-scroll" }, h("table", {}, h("thead", {}, h("tr", {}, ["Model", "Valid JSON", "Plan accuracy", "Median latency"].map((t) => h("th", { text: t })))), h("tbody", {}, rows))),
      e2e.length ? h("div", {}, h("h3", { text: "End to end", style: "margin-top:16px" }),
        h("p", { class: "muted small", text: `Instruction → plan → closed loop, ${P.end_to_end[0].profile} faults. Task success, no verifier vs. corrected verifier.` }),
        h("div", { class: "tbl-scroll" }, h("table", {}, h("thead", {}, h("tr", {}, ["Model", "No verifier", "Verifier + corrected"].map((t) => h("th", { text: t })))), h("tbody", {}, e2e)))) : null);
  }

  /* Method ------------------------------------------------------------- */
  function renderMethod() {
    const box = (x, y, w, t, s, cls = "") => `<rect class="node ${cls}" x="${x}" y="${y}" width="${w}" height="64" rx="10"/><text x="${x + w / 2}" y="${y + 28}" text-anchor="middle">${t}</text><text class="sub" x="${x + w / 2}" y="${y + 47}" text-anchor="middle">${s}</text>`;
    $("#diagram").innerHTML = `<svg viewBox="0 0 900 270" role="img" aria-label="Pipeline: instruction to planner to skill to camera verifier, looping back on failure; ground truth sits outside the loop and only scores.">
      ${box(10, 40, 150, "Instruction", "natural language")}
      ${box(205, 40, 170, "Planner", "local LLM (Ollama)", "accent")}
      ${box(420, 40, 190, "Skill", "scripted waypoints + IK", "accent")}
      ${box(655, 40, 230, "Verifier", "overhead RGB-D, no ground truth", "accent")}
      <path d="M160 72H205"/><polygon points="205,72 197,68 197,76"/>
      <path d="M375 72H420"/><polygon points="420,72 412,68 412,76"/>
      <path d="M610 72H655"/><polygon points="655,72 647,68 647,76"/>
      <path class="loop" d="M770 104V156H515V112"/><polygon style="fill:var(--accent)" points="515,106 511,114 519,114"/>
      <text x="642" y="148" text-anchor="middle" style="fill:var(--accent)">not done: retry with the measured error</text>
      <text x="770" y="200" text-anchor="middle" style="fill:var(--ok)">done: next subtask</text>
      <rect class="node oracle" x="205" y="196" width="405" height="58" rx="10"/>
      <text x="407" y="221" text-anchor="middle">Ground-truth oracle</text>
      <text class="sub" x="407" y="240" text-anchor="middle">reads simulator state after the episode, only to score it</text></svg>`;
    const ul = $("#limits"); ul.replaceChildren(...LIMITS.map((t) => h("li", { text: t })));
  }

  /* Boot --------------------------------------------------------------- */
  async function boot() {
    renderMethod();
    [state.results, state.index, state.planners, state.mani] = await Promise.all([
      getJSON("data/results.json"), getJSON("data/index.json"), getJSON("data/planners.json"), getJSON("data/maniskill.json")]);
    renderStats(); renderResults();
    if (state.index) { renderStories(); await renderPlayer(); }
    else { $("#stories").replaceChildren(); $("#player").replaceChildren(h("p", { class: "empty", text: "No recorded rollouts yet." })); }
    show(location.hash.slice(1) || "overview");
  }
  boot();
})();
