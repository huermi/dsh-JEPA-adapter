# dsh-JEPA-adapter

> An OpenAI-compatible adapter for a **local JEPA cognitive agent** (retrieval-based implementation of DCA-4.0) that **runs on a consumer CPU**.

This project provides a local model service (model ID `jepa-1`) compatible with any harness that supports OpenAI function-calling (DeepSeek / Claude Code / custom clients): pick `jepa-1` in the model list and start chatting or calling tools. The kernel is **not** LLM generation — it is a **"situation → memory retrieval → decision"** hybrid architecture (vectors + symbolic tools). It learns continuously (blocked-triggered learning, pass-outcome consolidation, selective forgetting, contradiction adjudication) and monitors its own cognitive health (regeneration metrics + pathology flags).

**中文 README: [README.md](README.md)**

---

## Feature Overview

| Capability | Description |
|---|---|
| 🧠 Retrieval-based cognition | Answers are retrieved from memory (not generated) — traceable and falsifiable |
| 📚 Lifelong learning | Blocked (retrieval miss / low confidence) triggers learning; judgment feedback drives consolidation |
| 🧬 Layered memory | fast (working) → core (consolidated) → archive (recoverable forgetting) |
| ⚖️ Contradiction handling | Similar situations with different answers → contradiction pairs → practice-based adjudication (Wilson CI) |
| 🗑 Selective forgetting | Quality×recency weighted eviction, consolidation-score demotion, weight decay + verification-strength protection |
| ⚡ Weight memory (LAM) | Linear associative memory: O(1) prediction, 96 KB, generalizes relational knowledge |
| 🔧 Tool calling | Call-pattern retrieval (tool + args), multi-step tasks driven by situation |
| 🩺 Health monitoring | Flux / consolidation rate / margin / calibration shape / tool entropy + 4 pathology flags |
| 🌐 OpenAI compatible | Any harness can select `jepa-1`; browser settings UI at `/ui` |

---

## Core Mechanisms (Overview)

| Mechanism | Summary |
|---|---|
| **Retrieval-based responding & layered memory** | Answers are retrieved from memory (not generated) — traceable and falsifiable. Three layers: fast / core / archive; pass-score threshold promotes to core, score drops below threshold demotes to archive (recoverable) |
| **Pass-outcome consolidation** | Correct +1 / wrong −2 (with frequency weight); only experience that passes repeated practical tests is consolidated — wrong entries never consolidate and are evicted first |
| **Blocked-triggered learning** | Retrieval miss / low confidence / contradiction = blocked → explore and verify. Trigger = expectation mismatch, not curiosity (curiosity misses familiar-but-important knowledge) |
| **Contradiction adjudication** | Similar situations with different answers → contradiction pairs → practice-based adjudication (Wilson CI): exclusive → decide a winner (loser abstains); complementary → coexist; insufficient evidence → pending (tolerant waiting) |
| **Falsifiable paradigm** | Each tool call = (situation, action pattern, outcome status); falsified calls are never reused (counterexample-driven accumulation); task→tool preference is learned statistically from verified calls (replacing the hand-crafted keyword list) |
| **Weight memory (LAM)** | Linear associative memory W: situation→answer vector, online delta-rule updates (O(1) prediction, 96 KB) — generalizes relational knowledge (factual knowledge relies on external retrieval) |
| **Soft alignment (AdaJEPA)** | On correct judgment, move the hit entry's representation toward the query (small step α=0.1) — similar inputs benefit automatically; wrong judgments never calibrate |
| **Layered forgetting** | Forgetting priority ∝ discriminative value × verification history: fast → quality×recency weighted eviction; core → demote to archive on score drop; W → slow decay + verification-strength protection |
| **Regeneration metrics** | `v1/status` reports flux (paradigm update rate), consolidation rate, retrieval margin, calibration shape, tool entropy + 4 pathology flags (retrieval degradation / representation distortion / cognitive rigidity / value atrophy) |
| **LeCun-style gradual replacement** | Hand-crafted thresholds progressively replaced by data-driven decision surfaces: 2D calibration table (decision boundary) / statistical prior (keyword list) / Wilson CIs (adjudication) |

## Architecture

```
body/
  kernel.py           core: decision chain (call memory → retrieval → explore → done)
  respond_learner.py  retrieval responder: layered memory + pass-outcome + contradictions
                      + LAM + soft alignment + forgetting
  call_memory.py      call-pattern memory: (situation→tool+args+result) + falsifiability gate
                      + task→tool statistical prior
  mini_encoder.py     MiniLM encoder (lazy load, LRU cache, offline by default,
                      hash-bag fallback on failure)
  plugin_config.py    PluginConfig: 22 configurable fields (dataclass, JEPA_* env overrides)
  plugin_server.py    OpenAI-compatible HTTP server + settings UI (/ui)
components/           DCA components (InfoDrives / Configurator / energy, etc.)
```

### Decision Chain (`chat_completion`)

```
full message history → situation z_ctx
  1. call_mem.select_k top-K → world-model prediction verification (_select_planned)
     → hit → tool_calls (via=call_memory+planned)
  2. responder.respond (retrieval answering)
     → hit → content (via=retrieval)
     → miss → record blocked type (retrieval_miss / low_confidence / contradiction)
  3. _select_explore (after blocked, within budget, temperature decay)
     → tool_calls (via=explore) → tool_result_step learns → next retrieval hits
  4. default finish (via=default)
```

### Three Memory Layers × Three Forgetting Modes

```
        entry (quality×frequency)        exit (evidence synthesis)
fast ───────────────────→ core ──→ archive (recoverable)
  │                         ↑
  └── W (decay + G protection) ─┘
```

---

## API Reference

Run: `python body/plugin_server.py` (port 8045, override with `JEPA_PORT`). OpenAI compatible.

| Endpoint | Method | Description |
|---|---|---|
| `/v1/chat/completions` | POST | Chat + tool calls (OpenAI function-calling, supports `stream`) |
| `/v1/models` | GET | Model list (`jepa-1` / JEPA DCA-4.0) — for harness model selection |
| `/v1/status` | GET | Model status + switch matrix + responder/call_mem stats + **cognition health block** |
| `/v1/config` | GET | Current config + schema (settings-UI data source) |
| `/v1/config` | POST | Hot-update config (type-checked; responder params take effect immediately) |
| `/v1/sleep` | POST | Trigger sleep consolidation (memory replay) |
| `/v1/learn_response` | POST | Explicitly teach a response (situation→text) |
| `/v1/archive` | POST | Snapshot (save/load) |
| `/ui` | GET | Browser settings UI (dark theme, all config items visualized) |

### Chat Example

```bash
curl http://127.0.0.1:8045/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "jepa-1",
    "messages": [{"role": "user", "content": "list python files in the project"}],
    "tools": [{
      "type": "function",
      "function": {
        "name": "glob",
        "description": "list files matching a glob pattern",
        "parameters": {"type": "object", "properties": {"pattern": {"type": "string"}}}
      }
    }]
  }'
```

After receiving `tool_calls`, the harness executes and returns results as `role: tool` messages — JEPA learns the full call pattern via `tool_result_step`; each step of a multi-step task is situation-driven (list → read → done).

### Configuration (22 fields; `/ui` or `/v1/config`)

| Field | Default | Description |
|---|---|---|
| `learning` | true | online weight gradients |
| `memory` | true | memory writes |
| `sleep` / `sleep_epochs` / `sleep_lr_scale` / `sleep_prio_mix` | false / 3 / 0.3 / 0.5 | sleep consolidation (replay) |
| `tools` | true | tool calling |
| `respond_mode` | retrieval | retrieval = learn to speak via retrieval / llm = bridge external LLM |
| `archive` | false | snapshots (save/load) |
| `surprise_thresh` | 0.3 | surprise gate (E1 tool-call trigger) |
| `max_memory` | 200 | memory cap |
| `explore` / `explore_budget` / `explore_decay` | true / 3 / 0.9 | free exploration (budget + temperature decay) |
| `benchmark_mode` | false | benchmark mode (no explore/network; internalized knowledge only) |
| `respond_cap` | 300 | responder experience cap |
| `respond_min_sim` | 0.45 | responder retrieval threshold |
| `soft_align` / `soft_align_alpha` | true / 0.1 | AdaJEPA soft alignment (step α) |

Env vars: `JEPA_EXPLORE` / `JEPA_SOFT_ALIGN` / `JEPA_BENCHMARK` / `JEPA_RESPOND_CAP` / `JEPA_PORT` etc. (`PluginConfig.from_env()`).

---

## Quick Start

```bash
# 1. Dependencies (Python ≥3.10)
pip install -r requirements.txt
# or install separately (smaller CPU-only torch):
#   pip install numpy pyarrow torch --index-url https://download.pytorch.org/whl/cpu
#   pip install transformers sentence-transformers

# 2. Model: auto-downloaded on first run — sentence-transformers/all-MiniLM-L6-v2 (~90MB)
#    Default mirror: hf-mirror.com; cache in models/ (override with JEPA_MODEL_DIR)

# 3. Start server
python body/plugin_server.py

# 4. Chat / tool calls (see example above); settings UI: http://127.0.0.1:8045/ui
```

> Offline load is the default (`HF_HUB_OFFLINE`): when the model is cached locally, no network check is performed, avoiding the transformers import hang on unreachable networks.

## Learning & Training

```bash
# Autonomous verification learning (MMLU 8 subjects; knowledge base in benchmark/library/)
python learner_loop.py --per 12
# → post-learning accuracy 27.1% (global_facts 83%; 26/96 learned from scratch)

# Automated training (material ingestion → evaluation; snapshots in benchmark/snapshots/)
python auto_train.py --rounds 3
# → material ingestion: training items (question|correct choice) ingested directly,
#   correctness guaranteed by dataset labels (decoupled from model performance)
# → evaluation: fresh body per question, tests whether material generalizes to unseen
#   questions (global_facts 83.3%)

# Web retrieval learning (fetch→internalize→reuse→confusion-correction→generalize)
python train_web_learning.py
```

**Known benchmark results** (honest statement, 2026-08-25):

| Scenario | Result | Note |
|---|---|---|
| Autonomous learning (learner_loop, 96 questions) | 27.1% | limited by "isomorphic confusion + knowledge-base coverage" |
| Material generalization (auto_train, global_facts) | 83.3% | material supports unseen questions (generalization beats memorization) |
| Internalization eval (material learned into the model) | 1.0% | inherent boundary of retrieval architecture on factual knowledge |
| econometrics | 0% | zero knowledge-base coverage (data boundary, not mechanism) |

## Verification Scripts (mechanism-level regression)

```bash
python memory_layers_check.py         # layered memory: learn new without destroying old
python falsification_check.py         # falsifiability: falsified calls are not reused
python soft_alignment_check.py        # AdaJEPA soft alignment (0.890→0.927)
python selective_forgetting_check.py  # five layered-forgetting mechanisms
python contradiction_check.py         # contradiction protocol (four scenarios)
python status_metrics_check.py        # regeneration metrics + pathology flags (four scenarios)
python multi_step_check.py            # continuous multi-step tool calling
python dsh_harness_check.py           # end-to-end integration (OpenAI compatible)
python study_trigger_experiment.py    # trigger-source / consolidation-criterion experiments
```

---

## Third-Party Models & Training Attribution

| Component | Reference | License |
|---|---|---|
| Semantic encoder | [sentence-transformers/all-MiniLM-L6-v2](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2) (HuggingFace, downloaded via hf-mirror.com by default) | Apache-2.0 |
| Runtime | torch (CPU) / transformers / sentence-transformers / numpy / pyarrow | their respective licenses |

**This project does not train or fine-tune any third-party weights** — MiniLM is used strictly as a **frozen text encoder** (situation vectorization, 384d). All in-model learning (responder memory, LAM weights, contradiction adjudication, soft alignment) happens online at inference time and produces no derivatives of third-party weights. If you benchmark with the MMLU dataset, follow its original license (**the repo contains no MMLU data files**).

---

## Known Boundaries (honest statement)

1. **Inherent limit of retrieval architecture**: retrieval discernibility on isomorphic statistical questions (similar question forms, different answers) — cosine + lexical gate + LAM cannot fully separate them; exact memory or external verification is required
2. **Factual vs relational knowledge**: weight memory (LAM) works only for relational knowledge; factual knowledge (MMLU-like) relies on external retrieval / knowledge base
3. **LLM positioning**: in this architecture the LLM is a "symbol translator" (language↔intent↔symbols) and does not hold factual memory — facts must come from the cognition layer (traceable, hallucination-resistant)
4. **Internalization eval at 1%**: material learned directly into a retrieval model generalizes ≈0 — "training a database ≠ training a model"; improving in-model capacity needs generative/parametric routes (beyond this repo)

## Open-Source Notes

- **Internal research documents** (`benchmark/*.md`: theory development, training plans, LeCun comparison) and **diagnostic scripts** (`*_check.py`, `jpi*_*.py`, etc.) are public for researchers to trace the full design lineage and experimental data
- **Project memory logs** are published as redacted copies (`memory_public/`): personal paths, usernames, and workspace IDs are anonymized (`<repo-root>` / `~/` / `<project-id>`); raw logs in `.workbuddy/` are not published
- All hard-coded local absolute paths have been relativized (`REPO_ROOT` auto-detection) — clone and run directly

## License

[MIT](LICENSE) © 2026 huermi
