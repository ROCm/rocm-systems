# Spec-Driven Development Proposal for rocprofiler-compute

**Branch:** `users/feizheng10/spec-driven-dev`  
**Status:** Design proposal (not yet adopted as team policy)  
**Audience:** rocprofiler-compute maintainers and AI-assisted contributors

---

## Executive Summary

**ROCm Compute Profiler** (`rocprofiler-compute`) is a **GPU system performance profiling
tool** for ML/HPC workloads on AMD Instinct GPUs ([`README.md`](../../README.md)).
Day-to-day development spans metrics, SOC configuration, profiling replay, and PC
sampling, while also maintaining support across multiple OSes and GPU
architectures.

As AI-assisted development becomes normal, this proposal combines **OpenSpec**,
**Superpowers**, and **graphify** to improve end-to-end efficiency and
collaboration: keep requirements and design decisions durable, make execution
steps explicit and test-driven, reduce re-discovery work for both humans and AI
assistants, and lower token cost through scoped context.

Roles at a glance:

| Layer | Tool | Responsibility |
|-------|------|----------------|
| **Design & persistent contract** | [OpenSpec](https://github.com/Fission-AI/OpenSpec) | Early requirement fit; specs/design/tasks/archive as durable memory |
| **Engineering discipline** | [Superpowers](https://github.com/obra/superpowers) | Clear SDD/TDD execution: brainstorm, plan, test-first, debug, review |
| **Codebase structure** | graphify (`graphify-out/`) | Scoped navigation: how modules connect, where to edit, impact paths |

**OpenSpec** is used to: (1) **left-shift design**—pressure-test how the solution
fits requirements while change is still cheap, so early development aligns with
intent before large code lands; and (2) **record persistent design**—specs and
design artifacts that remain the contract for later AI-assisted work, so
follow-on sessions do not unknowingly break agreed behavior.

**Superpowers** is used to **enforce sound software engineering**—clear,
repeatable execution steps (brainstorming, planning, TDD, debugging, review) so
implementation follows SDD/TDD discipline instead of skipping tests, structure,
or verification.

**graphify** is used to **ground design and implementation in the real
codebase**—before and during changes, agents query `graphify-out/graph.json`
(via `graphify query`, `graphify path`, `graphify explain`) instead of broad
grep or loading entire trees. OpenSpec records *what* must stay true; graphify
shows *where* the code lives and how pieces connect. After code edits, run
`graphify update .` so the graph stays current (AST-only, no API cost).

The repo already has `openspec/config.yaml`, `graphify-out/`, OpenSpec skills
under `.cursor/skills/`, `.claude/skills/`, and `.github/skills/`, plus Cursor
rules for graphify-first navigation. Superpowers is available via the Cursor
Superpowers plugin. This document defines how to **combine** them deliberately
for rocprofiler-compute.

---

## 1. Problem Statement

### 1.1 Typical AI-assisted failure modes

- **Spec drift:** Long chat sessions lose the original requirement; the agent “fixes” symptoms unrelated to the ticket.
- **Skipped validation:** Code merges without the right test tier (unit vs Docker/ctest vs GPU replay).
- **Domain blindness:** PMC/metric naming, `gfx*` SOC tables, and schema compatibility are treated like generic CRUD.
- **Monolithic dumps:** One-shot large diffs across profile + analyze + docs without reviewable steps.
- **Blind edits:** Changing analyze/CLI/SOC paths without understanding cross-module dependencies (profile ↔ analyze ↔ `rocprof_compute_soc`).

### 1.2 What rocprofiler-compute needs

- Traceable **change records** (proposal → spec scenarios → design → tasks → verify → archive).
- **Enforced process** (design approval before code, red-green tests where feasible, root-cause debugging).
- **Context hygiene:** After spec is written, new sessions load files—not 50k tokens of brainstorming.
- Alignment with existing **`.ai/rules/`**, **Ruff**, **pre-commit**, **graphify**, and **rocm-systems** contribution norms.

---

## 2. Framework Roles

### 2.1 OpenSpec — specification & change tracking

OpenSpec provides **Spec-Driven Development (SDD)**: structured artifacts per change under `openspec/changes/<change-name>/`.

Typical artifact flow:

```text
proposal.md   →  specs/ (requirements & scenarios)
                design.md
                tasks.md
                → implement → verify → archive
```

**Commands / skills (project-local):**

- `/opsx:new <name>` or `openspec new change "<name>"` — scaffold a change
- `/opsx:ff` — fast-forward generate artifacts from conversation
- `/opsx:apply` — implementation guided by `tasks.md`
- `/opsx:archive` — move change to archive; update global spec state
- Skills: `openspec-propose`, `openspec-apply-change`, `openspec-archive-change`, `openspec-explore`

**Single source of truth:** `openspec/changes/<name>/` is the contract for reviewers and agents. User stories should be **testable scenarios** (Given/When/Then), not vague bullet lists.

### 2.2 Superpowers — engineering discipline

Superpowers is a **composable skills library** (Markdown `SKILL.md` files with hard gates). Skills trigger on task type—brainstorming before creative work, TDD before production code, systematic debugging before fixes.

Core skills for this workflow:

| Skill | When |
|-------|------|
| `brainstorming` | Before any feature/fix design; no code until design approved |
| `writing-plans` | After OpenSpec `tasks.md`; micro-tasks (2–5 min) with file paths & verification |
| `using-git-worktrees` | Isolate feature work from `develop` |
| `test-driven-development` | Implementation loops (red → green → refactor) |
| `systematic-debugging` | Failures in pytest/ctest/GPU runs |
| `subagent-driven-development` / `executing-plans` | Execute micro-tasks with clean context |
| `requesting-code-review` | Compare implementation to OpenSpec `specs/` |
| `verification-before-completion` | Evidence before “done” claims |
| `finishing-a-development-branch` | Merge/PR/worktree cleanup |

Installation (Cursor): Superpowers plugin (`/add-plugin superpowers` or marketplace). OpenSpec: `openspec init` (already present in this repo).

### 2.3 graphify — codebase navigation (rocprofiler-compute–local)

graphify is **not** a third process framework. It is a **knowledge graph** of this repository (`graphify-out/graph.json`, optional `graphify-out/wiki/`, `GRAPH_REPORT.md`) used to shrink context and reduce wrong-file edits.

**When to use (project rules in `.cursor/rules/graphify.mdc`):**

| Command | Use |
|---------|-----|
| `graphify query "<question>"` | Architecture or behavior questions before design |
| `graphify path "<A>" "<B>"` | Call/data flow between symbols or modules |
| `graphify explain "<concept>"` | Focused subgraph for a domain term (e.g. metric load path) |
| `graphify update .` | After modifying code in a session (AST refresh, no API cost) |

**Division of labor:**

```text
OpenSpec     →  WHAT must be true (requirements, scenarios, design decisions)
Superpowers  →  HOW to build it with discipline (TDD, plans, review)
graphify     →  WHERE in the repo (files, dependencies, touch surfaces)
```

Record graphify findings in OpenSpec `design.md` (e.g. “touch `analysis_cli.py` → `rocprof_compute_soc`”) so later sessions do not re-discover structure from scratch. Prefer `graphify query` over raw grep or loading `GRAPH_REPORT.md` unless doing a broad architecture review.

### 2.4 Why combine OpenSpec + Superpowers + graphify

| Approach | Gap |
|----------|-----|
| **OpenSpec only** | Strong documents, weak enforcement of TDD, worktrees, and debug discipline |
| **Superpowers only** | Strong process, weak long-lived **project spec** and cross-change governance |
| **graphify only** | Knows structure, not requirements or engineering gates |
| **Ad-hoc prompts** | Suggestions ignored under pressure; no archived change history |
| **OpenSpec + Superpowers + graphify** | Intent + discipline + accurate code map |

This aligns with the **combined model** described in community write-ups—OpenSpec for **governance**, Superpowers for **execution**—extended with graphify for **navigation** in a large, multi-module profiler codebase ([Josh, hands-on workflow](https://vocus.cc/article/699e6bf2fd8978000156935e)).

---

## 3. Combined Workflow (Five Phases)

Adapted from [Josh’s OpenSpec + Superpowers guide](https://vocus.cc/article/699e6bf2fd8978000156935e) and [Termdock’s Superpowers overview](https://www.termdock.com/en/blog/superpowers-framework-agent-skills).

```mermaid
flowchart LR
  subgraph phase1 [Phase 1: Clarify]
    B[Superpowers: brainstorming]
    O1[OpenSpec: /opsx:new + /opsx:ff]
    B --> O1
  end
  subgraph phase2 [Phase 2: Isolate]
    W[Superpowers: git worktree]
  end
  subgraph phase3 [Phase 3: Plan]
    T[OpenSpec: tasks.md]
    P[Superpowers: writing-plans]
    T --> P
  end
  subgraph phase4 [Phase 4: Implement]
    C[Context reset]
    A[Superpowers: TDD + subagents]
    C --> A
  end
  subgraph phase5 [Phase 5: Finish]
    R[code review + verify]
    F[finishing branch]
    AR[/opsx:archive]
    R --> F --> AR
  end
  phase1 --> phase2 --> phase3 --> phase4 --> phase5
```

### Phase 1 — Clarify & define (Superpowers + OpenSpec + graphify)

1. Natural-language intent → **`brainstorming`** (questions, trade-offs, SOC/metric impact).
2. For unfamiliar areas → **`graphify query`** / **`graphify explain`** (e.g. “how are gfx metrics loaded into analyze?”).
3. After design approval → **`/opsx:new <kebab-name>`** (e.g. `add-rdna35-metric-filter`).
4. **`/opsx:ff`** → `proposal.md`, `specs/`, `design.md`, `tasks.md`; capture **modules and paths** from graphify in `design.md`.
5. Enrich **`openspec/config.yaml`** `context:` with rocprof-compute stack (see §6.1).

### Phase 2 — Isolate environment (Superpowers)

- **`using-git-worktrees`**: branch `users/<user>/<change>` off latest `develop`.
- Baseline: `pytest` (fast subset) + document whether full **`ctest`** in Docker is required for this change.

### Phase 3 — Decompose tasks (OpenSpec + Superpowers + graphify)

- OpenSpec `tasks.md` = **milestones** (e.g. “Add metric to gfx1151 table + docs + tests”).
- **`graphify path`** between entry points (e.g. CLI flag → parser → SOC YAML) to list real touch surfaces.
- **`writing-plans`** = **micro-tasks** with exact paths (`src/...`, `docs/data/metrics/...`, `tests/...`).

### Phase 4 — Disciplined implementation (Superpowers, spec as law, graphify for discovery)

**Context hygiene:** New chat/session; read only `openspec/changes/<name>/` + micro-plan.

1. Before editing unfamiliar code → **`rocprof-compute-graphify-nav`** or **`graphify query`** (see §6.5).
2. **`test-driven-development`** where tests exist or can be added without GPU.
3. For GPU-only behavior: spec must name **which workload** under `tests/workloads/` validates the change.
4. **`subagent-driven-development`** for independent micro-tasks (e.g. docs vs Python parser).
5. Never contradict **`specs/`** without updating the spec first.
6. After code changes → **`graphify update .`** in the project root.

### Phase 5 — Review & finalize

1. **`requesting-code-review`** against OpenSpec scenarios.
2. **`verification-before-completion`**: paste command output (Ruff, pytest, ctest scope).
3. **`finishing-a-development-branch`**: PR per `.ai/rules/pr-workflow.md`, `gh` CLI.
4. **`/opsx:archive`** after merge.

---

## 4. Rationale — Why This Solution

1. **Fits existing repo investment** — OpenSpec already initialized; skills duplicated for Cursor/Claude/GitHub Actions agents.
2. **Matches super-repo reality** — rocm-systems sparse checkouts need **written** design for cross-project impact.
3. **Profiling domain needs scenarios** — Metric and SOC changes deserve Given/When/Then acceptance tests in `specs/`, not only code review intuition.
4. **Hardware test pyramid** — Superpowers TDD for pure Python; OpenSpec documents which tier needs Docker/GPU.
5. **Context limits** — Archiving specs + clearing chat reduces hallucination ([context hygiene](https://vocus.cc/article/699e6bf2fd8978000156935e)).
6. **Contributor onboarding** — New developers (human or AI) read `openspec/changes/archive/` for prior decisions.
7. **Industry momentum** — Both tools are widely adopted; skills are portable across Claude Code, Cursor, Codex CLI ([Termdock](https://www.termdock.com/en/blog/superpowers-framework-agent-skills)).

---

## 5. Pros and Cons

### 5.1 Pros

| Benefit | Detail |
|---------|--------|
| **Traceability** | Every feature has proposal, design, and archived history |
| **Reviewability** | PRs link to `openspec/changes/<name>/`; reviewers judge against scenarios |
| **Process gates** | Brainstorming + TDD + debug skills reduce “vibe-only” patches |
| **Parallel work** | Subagents + worktrees suit large changes (new SOC, CLI flags) |
| **AI/tool agnostic** | Markdown specs + skills work across agents |
| **Large codebase** | graphify scopes context; OpenSpec + Superpowers avoid wrong intent and sloppy execution |

### 5.2 Cons

| Drawback | Mitigation |
|----------|------------|
| **Overhead for tiny fixes** | Use “no OpenSpec” path for typos/docs-only (team norm); optional `openspec-explore` for spikes |
| **Double setup** | Document one-time install in CONTRIBUTING |
| **TDD friction on GPU tests** | Spec declares unit vs integration vs manual GPU validation |
| **Skill conflict** | Prefer explicit “use OpenSpec change X” in prompt; list skill order in AGENTS.md |
| **Archive drift** | Require `/opsx:archive` in PR checklist before merge |
| **Learning curve** | This doc + one pilot change (pilot checklist in §6) |

---

## 6. rocprofiler-compute–Specific Guidance

### 6.1 Recommended `openspec/config.yaml` context block

Add project context so `/opsx:ff` artifacts are accurate:

```yaml
context: |
  Project: ROCm Compute Profiler (rocprofiler-compute) in rocm-systems super-repo.
  Stack: Python 3.9+ (src/), CMake/C++ native tooling, YAML metric definitions.
  GPUs: MI100/MI200/MI300/MI350 (CDNA), RDNA35_HALO, multiple gfx* SOCs.
  Layout: src/rocprof_compute_*, src/rocprof_compute_soc, tests/workloads/<case>/<SOC>/.
  Quality: Ruff on src/, pytest (pyproject.toml), ctest in Docker (README Testing).
  Docs: Sphinx under docs/; metrics in docs/data/metrics/gfx*_metrics.yaml.
  Agent rules: AGENTS.md, .ai/rules/python-style.md, ruff-tooling.md, pr-workflow.md.
  Code navigation: graphify query (graphify-out/) after code changes.
```

### 6.2 Change taxonomy (when to open an OpenSpec change)

| Change type | OpenSpec? | Minimum verification |
|-------------|-----------|----------------------|
| Metric add/rename | **Yes** | Scenario + YAML + pytest; note gfx ID |
| New SOC support | **Yes** | design.md: perfmon YAML, workloads, docs |
| CLI / analyze behavior | **Yes** | pytest markers; sample CSV fixtures |
| Profiler replay / rocprof SDK | **Yes** | ctest + named workload |
| Docs-only | Optional | Sphinx build if structural |
| Ruff/formatting | No | CI only |

### 6.3 Test strategy mapping (Superpowers TDD vs GPU reality)

```text
                    ┌─────────────────────────────────────┐
                    │  OpenSpec specs/: acceptance cases   │
                    └─────────────────┬───────────────────┘
                                      │
        ┌─────────────────────────────┼─────────────────────────────┐
        ▼                             ▼                             ▼
  Tier A: pytest                  Tier B: ctest                 Tier C: manual
  (parsers, analyze logic,        (Docker README workflow)      (new hardware,
   mocked CSV, unit markers)       install + workload replay)     lab-only SOCs)
        │                             │                             │
        └──────── Superpowers TDD ────┴── systematic-debugging ─────┘
```

**Rules:**

- Prefer **Tier A** for TDD loops; check in minimal golden CSV under `tests/` when adding analyze features.
- **Tier B** must be listed in `design.md` with Docker image tag and `ctest -R` filter if full suite is too heavy.
- **Tier C** requires explicit “manual validation” section in `specs/` — not a substitute for Tier A/B when CI coverage exists.

### 6.4 Profiling-project patterns to encode in specs

Borrow from observability / performance-tool practice (skills or OpenSpec `rules:`):

| Pattern | Application in rocprofiler-compute |
|---------|----------------------------------|
| **Golden-file testing** | Frozen `tests/workloads/.../*.csv` + `profiling_config.yaml` |
| **Schema versioning** | Document breaking changes to analysis dump (`docs/data/analyze/`) |
| **SOC capability matrix** | Table in `design.md`: gfx → features (roofline, PC sampling, etc.) |
| **Metric catalog discipline** | Single source: `docs/data/metrics/gfx*_metrics.yaml` + generated tables |
| **Replay determinism** | Scenarios for multiplexing, iteration counts, attach/detach |
| **Fail-closed CLI** | Invalid SOC/path exits non-zero; spec negative cases |
| **Performance regression budget** | Optional: benchmark task in `tasks.md` for hot paths |

### 6.5 Suggested custom skills (future “skills market”)

Package as repo skills under `.cursor/skills/` or `.claude/skills/` when needed:

| Skill name | Purpose |
|------------|---------|
| `rocprof-compute-metric-change` | Checklist: YAML + Sphinx + pytest + compatible-accelerators.rst |
| `rocprof-compute-soc-onboarding` | SOC fork checklist (perfmon, workloads, conceptual docs) |
| `rocprof-compute-docker-verify` | Wrap README Docker/ctest steps with expected filters |
| `rocprof-compute-graphify-nav` | Query graphify before editing unknown modules |

These **compose** with Superpowers (e.g. `rocprof-compute-metric-change` runs after `brainstorming`, before `writing-plans`).

### 6.6 Branch & PR conventions (rocm-systems)

- Branch: `users/<ldap>/<short-topic>` (this proposal branch is an example).
- Base: `develop`; sparse-checkout path `projects/rocprofiler-compute`.
- PR body: link OpenSpec change dir; paste pytest/ctest evidence.
- JIRA: per `.ai/rules/pr-workflow.md`.

### 6.7 Pilot adoption checklist

1. [ ] Team review of this proposal on branch `users/feizheng10/spec-driven-dev`.
2. [ ] Fill `openspec/config.yaml` `context` (§6.1).
3. [ ] Run one pilot change end-to-end (e.g. small metric doc fix → full archive).
4. [ ] Add CONTRIBUTING section pointing to `docs/design/spec-driven-development-proposal.md`.
5. [ ] PR template checkbox: “OpenSpec change archived or N/A”.
6. [ ] Confirm agents run `graphify update .` after code edits (or CI/doc note if graphify not on all machines).

---

## 7. Example Session Scripts (Cursor / Claude Code)

### 7.1 Start a feature

```text
I want to add <feature> for <SOC>. Use superpowers brainstorming first.
Use graphify query to map existing code paths before we lock design.
After I approve the design, run openspec new change "<kebab-name>" and /opsx:ff.
Put graphify touch surfaces into design.md.
```

### 7.2 Start implementation (new session)

```text
Read openspec/changes/<kebab-name>/design.md and specs/.
Use graphify path/query for any module not already listed in design.md.
Use writing-plans to break tasks.md into micro-tasks.
Implement with strict TDD; follow AGENTS.md and .ai/rules/.
For GPU validation use tests/workloads/<...>.
Run graphify update . when code changes are done.
```

### 7.3 Finish

```text
requesting-code-review against openspec/changes/<kebab-name>/specs/
Run ruff + pytest; document ctest scope.
finishing-a-development-branch per .ai/rules/pr-workflow.md
Then /opsx:archive
```

---

## 8. Alternatives Considered

| Alternative | Why not primary |
|-------------|-----------------|
| **Superpowers only** | No durable change specs in-repo |
| **OpenSpec only** | Weak enforcement of test/debug/worktree discipline |
| **Plain AGENTS.md / Cursor rules** | Suggestions, not gated workflows |
| **Spec-kit / other SDD tools** | OpenSpec already integrated in repo |
| **Full custom skill suite** | Higher maintenance; start with upstream skills + §6.5 |

---

## 9. References

1. Josh — *實戰教學：在 AI 終端機完美結合 OpenSpec 與 Superpowers 開發工作流*
   (five-phase CLI walkthrough; combined model, context hygiene).  
   https://vocus.cc/article/699e6bf2fd8978000156935e

2. Danny Huang, Termdock — *Superpowers: Skills Framework Reshaping AI Dev*
   (skills, TDD iron law, subagents).  
   https://www.termdock.com/en/blog/superpowers-framework-agent-skills

3. BSWEN — *OpenSpec vs Superpowers: Which SDD Framework Should You Choose?*  
   https://docs.bswen.com/blog/2026-03-27-openspec-vs-superpowers/

4. OpenSpec repository.  
   https://github.com/Fission-AI/OpenSpec

5. Superpowers repository (obra/superpowers).  
   https://github.com/obra/superpowers

6. rocprofiler-compute — `README.md` (testing, Docker, ctest).  
7. rocprofiler-compute — `AGENTS.md`, `.ai/rules/*`.  
8. rocprofiler-compute — `openspec/config.yaml` (existing init).  
9. rocprofiler-compute — `graphify-out/`, `.cursor/rules/graphify.mdc` (code navigation).

---

*Document generated on branch `users/feizheng10/spec-driven-dev` after syncing `develop`.*
