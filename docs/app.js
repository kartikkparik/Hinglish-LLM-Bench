/* Hinglish-LLM-Bench dashboard.
 * Zero-backend: loads results/eval_results.json + summary_metrics.csv
 * relative to this file. Configure RESULTS_BASE if the dashboard is hosted
 * somewhere the results/ folder is not a sibling of docs/. */

/* Paths are relative to docs/ because GitHub Pages serves docs/ as the
 * site root. Mirrored copies of results/, assets/, and data/ live inside
 * docs/ — regenerate them with scripts/sync_docs.sh after a new GPU run. */
const RESULTS_BASE = "./results/";
const DATASET_URL = "./data/hinglish_bench_150.jsonl";

const state = {
  summary: [],          // rows from summary_metrics.csv
  perItem: [],          // flattened per-item results across models
  dataset: [],          // fallback prompt list when results are absent
  hasResults: false,
};

const $ = (s) => document.querySelector(s);
const esc = (s) => String(s ?? "").replace(/[&<>"']/g,
  c => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));

/* ---------------- data loading ---------------- */

async function fetchJSON(url) {
  const r = await fetch(url);
  if (!r.ok) throw new Error(`${r.status}`);
  return r.json();
}

async function fetchText(url) {
  const r = await fetch(url);
  if (!r.ok) throw new Error(`${r.status}`);
  return r.text();
}

function parseCSV(text) {
  const lines = text.trim().split(/\r?\n/);
  const head = lines[0].split(",");
  return lines.slice(1).map(l => {
    // simple CSV (no quoted commas in our generated file)
    const cells = l.split(",");
    return Object.fromEntries(head.map((h, i) => [h, cells[i]]));
  });
}

async function loadResults() {
  try {
    const [json, csvText] = await Promise.all([
      fetchJSON(RESULTS_BASE + "eval_results.json"),
      fetchText(RESULTS_BASE + "summary_metrics.csv").catch(() => ""),
    ]);
    state.summary = csvText ? parseCSV(csvText) : (json.summary || []);
    state.perItem = [];
    for (const [model, items] of Object.entries(json.models || {})) {
      for (const it of items) state.perItem.push({ ...it, model });
    }
    state.hasResults = state.perItem.length > 0;
    return true;
  } catch (e) {
    return false;
  }
}

async function loadDatasetFallback() {
  try {
    const text = await fetchText(DATASET_URL);
    state.dataset = text.trim().split(/\r?\n/).map(l => JSON.parse(l));
  } catch (e) {
    state.dataset = [];
  }
}

/* ---------------- rendering ---------------- */

function renderLeaderboard() {
  const el = $("#leaderboard-table");
  if (!state.summary.length) {
    el.innerHTML = `<p class="muted">No summary metrics yet — pending GPU run.</p>`;
    return;
  }
  const cols = ["model", "n_items", "exact_match_acc", "normalized_match_acc",
                "mean_cosine_sim", "ms_per_token", "cmi_robustness_drop"];
  const labels = ["Model", "N", "EM Acc", "NMatch Acc", "Cosine Sim",
                  "ms/token", "CMI Drop"];
  let html = "<table><thead><tr>" + labels.map(l => `<th>${l}</th>`).join("") +
             "</tr></thead><tbody>";
  for (const row of state.summary) {
    html += "<tr>" + cols.map(c => `<td>${esc(row[c] ?? "—")}</td>`).join("") + "</tr>";
  }
  el.innerHTML = html + "</tbody></table>";
}

function renderCharts() {
  if (!state.hasResults || typeof Chart === "undefined") return;
  const models = [...new Set(state.perItem.map(r => r.model))];
  const short = models.map(m => m.split("/").pop());
  const tiers = ["low", "medium", "high"];
  const tierLabels = ["Low CMI", "Medium CMI", "High CMI"];
  const palette = ["#2563EB", "#D97706", "#DC2626", "#059669"];

  const byModelTier = {};
  for (const r of state.perItem) {
    const ok = r.nmatch ?? r.em;
    (byModelTier[r.model] ||= { low: [], medium: [], high: [] })[r.cmi_tier]?.push(ok);
  }
  const acc = (arr) => arr.length ? arr.filter(Boolean).length / arr.length : 0;

  Chart.defaults.font.family = "ui-sans-serif, system-ui, sans-serif";
  Chart.defaults.color = "#374151";

  // CMI degradation (grouped bar)
  new Chart($("#chart-cmi"), {
    type: "bar",
    data: {
      labels: tierLabels,
      datasets: models.map((m, i) => ({
        label: short[i],
        data: tiers.map(t => +(acc(byModelTier[m]?.[t] || [])).toFixed(3)),
        backgroundColor: palette[i % palette.length],
        borderRadius: 3,
      })),
    },
    options: { responsive: true, scales: { y: { min: 0, max: 1 } },
               plugins: { legend: { position: "bottom" } } },
  });

  // Cosine similarity
  const cosData = models.map(m => {
    const vals = state.perItem.filter(r => r.model === m)
      .map(r => r.cosine_sim).filter(v => v != null);
    return vals.length ? +(vals.reduce((a, b) => a + b, 0) / vals.length).toFixed(3) : 0;
  });
  new Chart($("#chart-cosine"), {
    type: "bar",
    data: { labels: short,
      datasets: [{ label: "Mean cosine sim", data: cosData,
        backgroundColor: palette.slice(0, models.length), borderRadius: 3 }] },
    options: { indexAxis: "y", scales: { x: { min: 0, max: 1 } },
               plugins: { legend: { display: false } } },
  });

  // Latency ms/token from summary
  const lat = state.summary.map(r => +r.ms_per_token || 0);
  new Chart($("#chart-latency"), {
    type: "bar",
    data: { labels: state.summary.map(r => r.model.split("/").pop()),
      datasets: [{ label: "ms/token", data: lat,
        backgroundColor: "#7C3AED", borderRadius: 3 }] },
    options: { indexAxis: "y", plugins: { legend: { display: false } } },
  });
}

/* ---------------- explorer ---------------- */

function explorerRows() {
  if (state.hasResults) return state.perItem;
  // pending state: show dataset prompts without predictions
  return state.dataset.map(d => ({
    id: d.id, category: d.category, cmi_tier: d.cmi_tier, prompt: d.prompt,
    reference_answer: d.reference_answer, model: "—", predicted_answer: "—",
    nmatch: null,
  }));
}

function renderExplorer() {
  const q = $("#f-search").value.toLowerCase();
  const model = $("#f-model").value;
  const cat = $("#f-category").value;
  const tier = $("#f-tier").value;
  const corr = $("#f-correct").value;

  const rows = explorerRows().filter(r => {
    if (q && !r.prompt.toLowerCase().includes(q) &&
        !String(r.reference_answer).toLowerCase().includes(q)) return false;
    if (model && r.model !== model) return false;
    if (cat && r.category !== cat) return false;
    if (tier && r.cmi_tier !== tier) return false;
    if (corr === "correct" && !(r.nmatch ?? r.em)) return false;
    if (corr === "incorrect" && (r.nmatch ?? r.em)) return false;
    return true;
  });

  $("#explorer-count").textContent = `(${rows.length} shown)`;
  const tbody = $("#explorer-table tbody");
  tbody.innerHTML = rows.slice(0, 300).map(r => `
    <tr>
      <td>${esc(r.id)}</td>
      <td>${esc(r.category)}</td>
      <td><span class="tier tier-${esc(r.cmi_tier)}">${esc(r.cmi_tier)}</span></td>
      <td class="prompt-cell" title="${esc(r.prompt)}">${esc(r.prompt.slice(0, 140))}${r.prompt.length > 140 ? "…" : ""}</td>
      <td>${esc(r.reference_answer)}</td>
      <td>${esc((r.model || "").split("/").pop())}</td>
      <td title="${esc(r.raw_completion || "")}">${esc((r.predicted_answer || "").slice(0, 60))}</td>
      <td>${r.nmatch == null ? "—" : (r.nmatch ? "✓" : "✗")}</td>
    </tr>`).join("");
}

function populateModelFilter() {
  const sel = $("#f-model");
  const models = [...new Set(state.perItem.map(r => r.model))];
  for (const m of models) {
    const o = document.createElement("option");
    o.value = m; o.textContent = m.split("/").pop();
    sel.appendChild(o);
  }
}

/* ---------------- init ---------------- */

(async function init() {
  const ok = await loadResults();
  $("#pending-banner").classList.toggle("hidden", ok);
  if (!ok) await loadDatasetFallback();
  populateModelFilter();
  renderLeaderboard();
  renderCharts();
  renderExplorer();

  for (const id of ["#f-search", "#f-model", "#f-category", "#f-tier", "#f-correct"]) {
    $(id).addEventListener("input", renderExplorer);
    $(id).addEventListener("change", renderExplorer);
  }
})();
