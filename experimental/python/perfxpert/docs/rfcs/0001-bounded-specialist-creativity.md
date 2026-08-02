---
status: draft
author: ephoukong
start_date: 2026-08-01
pr: TBD
implements: []
supersedes: []
---

# RFC-0001: Bounded creative freedom for Layer-2 specialists

## Summary

Add an opt-in, open-set **exploratory proposal channel** to the Compute,
Memory, Latency, and Trace-Diff specialists. The existing `techniques` list,
diff arithmetic, routing, and gate inputs remain a deterministic, vetted
control plane. Creative proposals are returned in a separate, strictly typed
field; are stamped `exploratory`; carry evidence, provenance, risk, and a
falsifiable measurement plan; and are never automatically ranked with,
promoted into, or executed as vetted recommendations.

The shipped policy has two tiers: `strict` (default, deterministic
catalog-only baseline) and `exploratory` (one bounded LLM augmentation pass). Air-gap mode
always has an effective tier of `strict`, returning an empty exploratory
channel. This makes nondeterminism additive: it may change advisory prose and
exploratory hypotheses, but it cannot change a handoff, a vetted technique, a
diff verdict, a correctness-gate input, or a correctness-gate decision.

This RFC also makes several currently documented-but-not-enforced guardrails
real before the exploratory tier can be enabled: live fence composition,
model-output validation, token/turn budgets, full Layer-2 coverage in the
guardrail tests, and a checked-in PerfXpert CI lane.

## Implementation status

Two prerequisites have already landed ahead of this RFC, because both are
correctness fixes rather than design changes:

1. **Trace-Diff determinism (finding 13).** The Trace-Diff specialist no
   longer lets a model replace `verdict`, `kernel_deltas`, or
   `wall_delta_pct`. Its fence already required that behaviour, so the code
   was contradicting its own documented contract. See
   `agents/diff_specialist.py` and `tests/test_agents/test_diff_specialist.py`.
2. **Canonical agent inventory (finding 16, partial).** `perfxpert.agents`
   claimed eight agents in its docstring but exported seven — Diff was absent
   from the package surface, which is why every guardrail test that
   enumerated agents inherited the same blind spot. The package now exposes
   `AGENT_BUILDERS`, and the schema-field-cap, fence-size, and tool-allowlist
   tests derive from it instead of keeping private lists.
   `tests/test_agents/test_agent_inventory.py` discovers builders from the
   package and fails if any is unregistered, so this cannot silently recur.

3. **Phase 11A guardrail enforcement (findings 3, 7, 8, 9).** The framework
   now validates model output against the declared schema and discards a
   response carrying unknown or mistyped keys; `token_budget` reaches the SDK
   as `ModelSettings(max_tokens=...)`; a session-scoped `RunPolicy` applies
   `MAX_SESSION_LLM_TURNS`; SDK tool wrappers re-check the allowlist; and the
   SDK-import guard recognises the canonical `agents` module it had been
   missing. `FenceBuilder` now reads the canonical `agents/fence` files,
   knows the Diff role, and is the single composition path — so `always.md`
   reaches every live prompt, verified by test. The duplicate
   `perfxpert/fence/slices/` copies are removed; they were never packaged in
   the wheel, so the builder would have raised `FileNotFoundError` in an
   installed environment.
4. **Analysis determinism (finding 11, related).** `run_analysis` no longer
   lets a live model replace `primary_bottleneck`, `confidence`,
   `time_breakdown`, `hot_kernels`, or `counter_data_available`. This closes a
   safety hole an existing test had encoded: with `counter_data_available`
   false the rule returns `data_insufficient`, but a model claiming a real
   bottleneck was accepted, producing exactly the recommendations the
   data-insufficient warning promises never to emit.

Still outstanding from finding 16: the fence/tool alignment parser in
`test_root.py`. Unresolved question 7 is answered by item 3 — the duplicate
slices are deleted while the `FenceBuilder` API is retained.

Everything else in this document is still proposed and unimplemented.

## Motivation

The Layer-2 specialists are prompted as closed-set catalog selectors. This is
safe and predictable, but it prevents a specialist from expressing a
workload-specific hypothesis that is not already represented by a named YAML
entry. Adding more YAML entries improves coverage but does not provide the
requested creative reasoning loop or a path for newly discovered ideas to
become reviewed knowledge.

The right boundary is not “deterministic agent” versus “creative agent.” It is
**deterministic control plane** versus **nondeterministic advisory plane**.
PerfXpert already has natural control-plane boundaries: rule-based routing,
knowledge catalogs, immutable Pydantic handoffs, read-only specialist tools,
and the correctness-gate cascade. The feature should place creativity beside
those boundaries, not inside them.

### First-hand current-state findings

The implementation differs materially from several comments and architecture
documents. These findings are prerequisites for the design:

1. `perfxpert/agents/fence/*.md` is the live fence system. Each agent builder
   passes its role file directly to `Agent.fence_path`, and
   `framework.Agent.__post_init__` reads that one file verbatim.
2. `perfxpert/fence/` is a second, divergent system. Its `FenceBuilder`
   composes `fence/slices/always.md`, a role slice, and YAML excerpts, but it
   is used only by fence/integration tests in the tracked tree. No production
   agent calls it. It does not include the Diff specialist and its slices lag
   the live copies.
3. Consequently, `agents/fence/always.md` is **not** prepended to live agent
   prompts, despite the architecture documentation saying it is loaded into
   every agent.
4. The specialist fence “enum” is prompt prose, not a schema enum.
   `ComputeSpecialistOutput`, `MemorySpecialistOutput`, and
   `LatencySpecialistOutput` each accept `List[Dict[str, Any]]`. Existing tests
   successfully inject names outside the fence list. A live model can already
   emit an arbitrary technique dictionary.
5. The deterministic catalog ranking does not override a live model result.
   It is used in air-gap mode and when the `techniques` key is absent. An
   explicitly empty live list remains empty. A novel live item is accepted,
   and `_predict_attach.py` leaves it untouched when it has no catalog match.
6. The role fences describe a singular `technique` object, while the actual
   model contract expects `techniques`, `confidence`, and `citations`. Tool
   lists in the fences also omit the catalog binding that code counts as a
   tool. This drift likely increases fallback frequency and makes the
   specialist appear more constrained than its Python contract is.
7. `Agent.output_schema` and `Agent.input_schema` are metadata to
   `run_agent()`. The framework does not validate the raw model response
   against `output_schema`, and the SDK agent is not constructed with an
   output type. Final Pydantic constructors validate selected top-level
   values, but technique dictionaries remain untyped and untrusted fields are
   manually ignored.
8. `Agent.token_budget` is not passed to the OpenAI Agents SDK or the opencode
   provider. `PERFXPERT_AGENTS_MAX_TURNS` limits one SDK run (default 10), but
   `MAX_SESSION_LLM_TURNS=100` is not connected to the session runtime.
9. `dispatch_tool()` and `dispatch_handoff()` enforce their policies when
   called directly, but neither is on the live SDK dispatch path. The SDK is
   given only declared tools, which provides an effective exposure allowlist;
   no SDK handoffs are configured at all. `Handoff` construction rules are
   tested but no production agent constructs a `Handoff`.
10. `allowed_handoffs=[]` is therefore not what limits specialist creativity.
    Specialists are ordinary leaf function calls that return to
    Recommendation. Relaxing Layer-2 handoffs would enlarge the control
    surface without improving open-set reasoning.
11. Root computes a deterministic route label but does not execute a Layer-1
    handoff. Analysis has no specialist handoffs. Recommendation directly
    calls Compute, Memory, or Latency as Python functions and does not call its
    own LLM. Diff is directly callable but is not routed by Recommendation.
    The three-layer hierarchy is an API/ownership model, not an SDK handoff
    graph in the current implementation.
12. Recommendation does not thread enough evidence into specialists:
    Compute receives empty `counter_data`, Memory receives neither counter nor
    memcpy data, Latency omits average duration, and the three inputs have no
    database path even though key bound tools require one. This is a separate
    barrier to grounded creative reasoning.
13. Trace-Diff calculated a deterministic verdict first but then permitted a
    live model to replace `verdict`, regressions, and improvements. That
    contradicted its fence and the stated parity invariant. **Fixed ahead of
    this RFC** — see Implementation status.
14. The five-gate cascade is deterministic and well tested when explicitly
    invoked, but there is no production call site for
    `runtime.gate_cascade.evaluate()` or `run_gate_cascade()` in the tracked
    package. Direct agent MCP tools are read-only; backend TUIs can execute
    native tools after a separate intent-first gate. The architecture claim
    that every proposed edit is automatically interposed by the five-gate
    cascade is not true on this branch.
15. The air-gap tests cover Root route equality, one Analysis classification
    comparison, and gate status/failing-gate equality. They do not compare
    specialist vetted outputs live versus air-gapped. The `tests/test_parity`
    suite runs the agentic path in air-gap mode against fixture expectations;
    it is not a live-versus-air-gap specialist comparison.
16. Several “CI guard” lists omit Diff, including schema field caps, the
    explicit fence-size list, execution-tool checks, fence/tool alignment, and
    `FenceBuilder` roles. Construction still applies the tool and role-fence
    caps when Diff is built, but the claimed full-tree CI inventory is
    incomplete. The root cause is that `perfxpert/agents/__init__.py` itself
    omitted Diff. **Partly fixed ahead of this RFC** — see Implementation
    status; fence/tool alignment and `FenceBuilder` roles remain.
17. No checked-in workflow under the repository's top-level
    `.github/workflows/` references PerfXpert. The tests are suitable CI
    checks, but their execution currently depends on an external job not
    represented in this tree.
18. `scripts/exit_dashboard.py` invokes a nonexistent
    `tests/test_agents/test_narrow_scope.py`, treats pending metrics as
    non-blocking, and expects air-gap snapshots that the cited tests do not
    write. It must not be treated as a reliable merge gate until repaired.
19. The proven-optimization documentation says fixture cases pass all five
    gates, while `ProvenOptimizationRunner` explicitly skips compile,
    bitwise, and anchors. This matters for promotion of a creative proposal.

These gaps do not justify weakening any intended guardrail. They justify
closing the gaps before adding a new nondeterministic surface.

### Guardrail inventory

“Enforced” below means the current production path structurally applies the
rule. “Partial” means a helper or test exists but the live path does not fully
use it. “Test/document only” means the claim is not mechanically present in
the tracked production path.

| Guardrail | Enforcement point | Current status and relevant tests |
|---|---|---|
| Agent layer is 0/1/2 | `Agent.__post_init__` | Enforced; `test_agents/test_framework.py` |
| Agent metadata and tool/handoff sequences are immutable | Frozen `Agent`; lists normalized to tuples | Enforced for the container; callable behavior is still external |
| Maximum five tools per agent | `Agent.__post_init__` | Enforced at construction; repeated in `test_tool_allowlist_guardrail.py` |
| Maximum 400 role-fence lines | `Agent.__post_init__` on the one live role file | Enforced for a built agent; direct test list omits Diff; composed shared fence is not live |
| Handoffs move down exactly one layer; no Layer-2 lateral handoff | `Handoff.__post_init__` | Partial: constructor is tested, but no production handoff graph uses it |
| Per-agent handoff allowlist | `dispatch_handoff()` | Partial: tested directly, not connected to SDK; specialists expose no SDK handoffs |
| Per-agent tool allowlist | Only declared tools are given to SDK; `dispatch_tool()` checks explicit calls | Effective exposure allowlist; direct dispatcher exception is not the SDK path |
| No execution-class gate tools on agents | Builder-enumeration tests | Partial: known forbidden names are checked; Diff is omitted; Root/Correctness task tools intentionally mutate local task state |
| OpenAI Agents SDK import isolation | Convention plus AST test | Implementation currently imports the SDK only in `framework.py`; test misses canonical `from agents ...` imports |
| Immutable, extra-forbid agent handoffs | `_FrozenModel` in `agents/schemas.py` | Enforced when Pydantic models are constructed |
| Input ≤10 / output ≤5 top-level fields | `test_schema_field_caps.py` | Test-enforced for seven agents; Diff is omitted |
| Literal/range/cross-field schema coherence | Pydantic fields and validators | Enforced for typed fields, including confidence and verdict/action coherence |
| Raw model output matches declared schema | Intended by `Agent.output_schema` | Not enforced by `run_agent()` or SDK configuration |
| Per-agent token budget | `Agent.token_budget` declaration | Not enforced |
| Per-run turn limit | `PERFXPERT_AGENTS_MAX_TURNS`, default 10 | Enforced on the SDK runner only; opencode path differs |
| Per-session 100-turn and optimization/failure caps | Constants in `gate_cascade.py` | Partial/test-only: 100-turn cap is unused; synthetic helper checks `loop_counter >= 5`, not the declared three-consecutive-failure constant |
| Provider allowlist | `build_session()` registry validation | Enforced |
| Retry only retryable provider failures | `AnalysisSession._run_live()` | Enforced for rate-limit/transient errors; auth/fatal errors fail closed |
| Scoped explicit API key | process lock plus temporary environment override | Enforced by session runtime tests |
| Opencode recursion prevention | `PERFXPERT_IN_OPENCODE_SESSION` provider guard; separate `PERFXPERT_IN_AGENT_SESSION` launcher guard | Enforced on their respective paths; launcher has an explicit `--force` override |
| Air-gap skips all model calls | `run_agent()` and `build_session()` | Enforced |
| Deterministic intent before Root | `runtime.intent_classifier` | Enforced; Root route metadata cannot be changed by the model |
| Deterministic Analysis baseline | deterministic metric collection and classifier | Partial: live model may replace the final classification and facts |
| Deterministic specialist fallback | catalog ranking in specialist runners | Enforced in air-gap/missing-key paths, not as the live vetted baseline |
| Knowledge schema validation | YAML plus JSON schemas and `tests/test_knowledge` | Test-enforced; specialist catalogs actually derive from `proven_optimizations.yaml` via `_technique_catalog.py` |
| Architecture applicability | `catalog_for()` filters `applies_to_gfx` | Enforced for catalog entries |
| Conservative impact prediction | unknown-technique, Amdahl, counter-data, and 0.85 high-bound rules in `predict_impact.py` | Enforced for recognized catalog IDs; unknown live techniques simply receive no prediction |
| Gate ownership outside Correctness | Correctness has no gate tools and consumes a frozen verdict | Enforced inside Correctness; automatic middleware interposition is absent |
| Five gates, strict order, first-failure short circuit | `gate_cascade.evaluate()` | Enforced when explicitly invoked: compile, SOL, bitwise/numeric, regression, anchors |
| SOL hard cap and hardware peak | 50x hard cap plus optional absolute-FLOPS check | Enforced when gate 2 runs; unit and red-team coverage |
| Regression thresholds | total >3%, weighted tail >5%, hot kernel >10% | Enforced when gate 4 runs; unit, near-threshold, and red-team coverage |
| Test-anchor removal detection | gate 5 | Enforced only when a candidate binary/anchor data is provided; otherwise production `evaluate()` reports gate 5 skipped |
| MCP exposes only READ_ONLY tools | `ToolClass`, discovery registry, exposure tests | Enforced; the eight agent wrappers are READ_ONLY |
| Execution input safety | root confinement, symlink resolution, metachar rejection, destructive-command denylist, compiler/profiler flag allowlists, safe subprocess environment, list-form subprocess calls | Enforced by execution tools and red-team/unit tests |
| Intent-first native-tool gate in backend TUIs | Claude/Gemini hooks; patched opencode hook; prompt/schema hints | Backend-dependent; opencode degrades to prompt-only with an upstream binary and an escape hatch exists |
| Canonical 14-attack red-team gate | attack registry, outcome files, runbook | Test-enforced when run; new freedom-specific attacks do not exist yet |
| ≥95% fixture parity | `tests/test_parity` | Test-enforced in air-gap mode; exact primary technique is a protected signal |
| Regression false-positive bound | proven/near-threshold suites | Test-enforced when run; current proven runner skips three gates |
| Audit dashboard | `scripts/exit_dashboard.py` | Not currently reliable because of stale paths/pending handling |
| Checked-in PR CI execution | top-level workflows | Not present for PerfXpert in this branch |

## Design

### Goals

1. Let each Layer-2 specialist express a small number of novel,
   workload-specific hypotheses beyond catalog IDs.
2. Preserve all existing intended caps: five tools, 400 fence lines,
   adjacent/downward handoff policy, immutable schemas, read-only MCP
   exposure, token/turn limits, and gate thresholds.
3. Make the deterministic core byte-stable across live and air-gap modes for
   the same deterministic inputs.
4. Make every creative proposal attributable and falsifiable.
5. Provide an explicit, human-reviewed path from proposal to proven YAML
   knowledge.
6. Fail closed: malformed, unsafe, unavailable, or over-budget creative
   output yields an empty exploratory channel, never loss or mutation of the
   vetted core.

### Non-goals

- Giving Layer-2 agents execution tools, file-write access, network access, or
  Layer-2-to-Layer-2 handoffs.
- Raising the five-tool, 400-line, 10-input-field, or five-output-field caps.
- Allowing a model to change a route, bottleneck verdict, diff arithmetic,
  gate threshold, gate order, gate input, or gate decision.
- Generating code patches or shell/profiler/compiler commands in the
  exploratory channel.
- Automatically writing generated ideas into a knowledge YAML.
- Promising deterministic exploratory text. Only the control plane is
  deterministic.
- Treating model memory as a source for numeric hardware facts or claimed
  speedups.

### Normative invariants

The implementation MUST satisfy all of the following:

1. `strict` is the built-in default.
2. `PERFXPERT_AIRGAP=1` forces the effective tier to `strict`, regardless of
   configuration.
3. `techniques`, specialist confidence/citations derived from the catalog,
   Diff wall/kernel deltas, and Diff verdict are deterministic functions of
   trusted inputs and knowledge files. A live model cannot replace them.
4. Exploratory proposals are present only in a separate
   `exploratory_proposals` channel. No code path aliases or concatenates that
   list into `techniques` or `recommendations`.
5. Recommendation ranking, deduplication, prediction attachment, task
   selection, patch generation, and correctness-gate input construction
   consume the vetted lane only.
6. A proposal cannot contain an executable command, patch, compiler flag,
   profiler argument list, or filesystem path to edit.
7. Every accepted proposal has at least one trusted evidence reference, at
   least one measurable expected direction, at least one verification metric
   or counter, explicit failure modes, and a bounded confidence.
8. Model-supplied proposal IDs, status, provider/model identity, fence digest,
   catalog digest, and trace fingerprint are ignored. The runtime stamps
   those fields.
9. The runtime returns at most three proposals per specialist invocation and
   uses a sub-budget inside the existing specialist `token_budget`; it does
   not increase the 3072-token specialist budget.
10. A creative-call failure never fails the deterministic specialist result.
11. No new tool or handoff is added to a specialist as part of this RFC.
12. Promotion to proven knowledge requires human review and measured
   before/after evidence; the model cannot promote itself.

### Public-facing change

#### Configuration

Add a validated configuration field:

```yaml
agent_creativity: strict  # strict | exploratory
```

The environment equivalent is:

```text
PERFXPERT_AGENT_CREATIVITY=strict|exploratory
```

Configuration is a deployment/session ceiling, not an LLM-controlled MCP
argument. The initial MCP agent tools MUST NOT expose a parameter that lets
the calling model elevate itself from `strict` to `exploratory`. Python
embedders may explicitly construct a session with a lower or equal tier, but
may not exceed the configured ceiling. Unknown values fail configuration
validation.

The effective policy is:

| Configured maximum | Air-gap | Agent capability | Effective tier |
|---|---:|---|---|
| `strict` | either | any | `strict` |
| `exploratory` | true | additive exploration | `strict` |
| `exploratory` | false | catalog-only | `strict` |
| `exploratory` | false | additive exploration | `exploratory` |

The `Agent` declaration gains a capability value with a safe default:

- `catalog_only` — no model-generated open-set channel.
- `additive_exploration` — Layer-2 only; permits the bounded proposal pass
  when the session maximum allows it.

Construction rejects `additive_exploration` on Layers 0 and 1. The four
existing Layer-2 specialists opt into the capability, while the runtime
default still keeps it disabled.

#### Output contract

Compute, Memory, and Latency gain one top-level output field, remaining below
the five-field cap:

```json
{
  "techniques": ["unchanged deterministic vetted entries"],
  "confidence": 0.6,
  "citations": ["deterministic catalog citations"],
  "exploratory_proposals": [
    {
      "proposal_id": "pxp-exp-<content hash>",
      "status": "exploratory",
      "title": "Short conceptual technique name",
      "specialist": "memory",
      "target_kernel": "kernel name from trusted input",
      "hypothesis": "Why this may help this measured workload",
      "mechanism": "The expected hardware/software mechanism",
      "evidence": [
        {
          "kind": "tool",
          "ref": "unified_memory.analyze_paging",
          "observation": "paging_events is non-zero"
        }
      ],
      "expected_effects": [
        {
          "metric": "page_faults",
          "direction": "decrease"
        }
      ],
      "verification": {
        "metrics": ["page_faults", "wall_time", "per_kernel_runtime"],
        "success_criteria": [
          {
            "metric": "page_faults",
            "op": "<",
            "baseline": "measured"
          }
        ],
        "requires_full_gate_cascade": true
      },
      "assumptions": ["Explicit assumption"],
      "failure_modes": ["How the proposal could regress or be inapplicable"],
      "confidence": 0.4,
      "provenance": {
        "provider": "runtime-stamped",
        "model": "runtime-stamped",
        "trace_fingerprint": "runtime-stamped",
        "fence_sha256": "runtime-stamped",
        "catalog_sha256": "runtime-stamped"
      }
    }
  ]
}
```

This example is illustrative; the implementation uses nested frozen Pydantic
models, `extra="forbid"`, string/list length bounds, a maximum of three items,
and typed literals/operators. The LLM emits only an
`ExploratoryProposalDraft`. The runtime validates the draft, resolves evidence
references against a trusted manifest, checks the target against known
kernels, strips any fields it owns, computes the ID, and constructs the final
`ExploratoryProposal`.

Exploratory confidence is limited to `0.0..0.5`. This is not a claim that all
novel ideas are unlikely; it prevents an unmeasured proposal from visually or
numerically impersonating a proven recommendation. Lane separation, not
confidence arithmetic, remains the primary safety mechanism.

Trace-Diff is already at the five-field cap. It retains the public and
internal `kernel_deltas` dictionary field and adds an
`exploratory_proposals` key beside `regressions` and `improvements`, with a
model validator enforcing the three allowed keys and validating proposal
items. The MCP wrapper continues flattening regressions and improvements and
also exposes a top-level `exploratory_proposals` key. The deterministic Diff
verdict and deltas can no longer be replaced by the model.

Recommendation gains a fourth top-level field:

```text
exploratory_proposals: List[ExploratoryProposal] = []
```

It propagates this lane separately and deduplicates it only by server-generated
`proposal_id`. It does not compare exploratory confidence with vetted impact,
does not pass proposals to `_predict_attach.py`, and does not include them in
`seen_recommendation_hashes`.

The first rollout exposes proposals from direct specialist and Recommendation
Python/MCP calls. Root and the batch `analyze` formatters are not changed until
their actual orchestration is aligned with the documented hierarchy. This
avoids hiding a new contract inside `RootOutput.metadata` and avoids implying
that Root currently executes Recommendation.

### Internal change

#### 1. Freeze the deterministic core first

For Compute, Memory, and Latency:

1. Fetch and architecture-filter the proven catalog.
2. Rank it with the existing deterministic rules.
3. Attach only catalog-backed impact predictions.
4. Derive vetted citations and confidence deterministically.
5. Store this result before any model call.
6. In `strict` mode, return immediately.
7. In `exploratory` mode, run one additive proposal pass and append only to
   `exploratory_proposals`.

For Diff:

1. Call `trace_diff.diff_runs()` once.
2. Calculate wall time, per-kernel deltas, and verdict deterministically.
3. Permit the model to add a narrative and exploratory causal hypotheses, but
   not to replace arithmetic or verdict.

This intentionally changes one current live behavior: a model can no longer
replace or empty the vetted technique list. Existing tests that script
model-controlled `techniques` must be rewritten to script the exploratory
draft and assert that the vetted list is unchanged. This is a strengthening
of the parity invariant, not a relaxation.

#### 2. Build a trusted evidence envelope

Open-set reasoning without evidence is speculation. Add optional
`database_path` fields to Recommendation and the three single-run specialist
inputs (all remain under the ten-field cap), and correctly thread existing
`counter_data`, `memcpy_data`, `source_hints`, and average-duration values.

Before the model call, each specialist builds a deterministic evidence
envelope from:

- its validated input model;
- the already-fetched catalog and a denylist of its normalized IDs/titles;
- outputs from its existing read-only tool allowlist when the required
  trusted arguments are available;
- a manifest of valid kernel names, metric names, counter names, tool names,
  and knowledge references;
- trace, fence, and catalog digests.

The evidence envelope is internal context, not an untyped extension to the
public input model. `run_agent()` should distinguish validated user input from
trusted runtime context so `extra="forbid"` remains meaningful.

The model may use general model knowledge to form a mechanism hypothesis, but
it may not represent that knowledge as an observed fact or source citation.
Such reasoning is tagged `model_hypothesis`, cannot contain a numeric speedup,
and must name a measurement that could disprove it.

#### 3. Validate model output at the framework boundary

The framework gains a model-response schema distinct from the final public
handoff schema. For ordinary agents it may default to the existing
`output_schema`; specialists use a strict creative-draft schema for the
augmentation call.

The live SDK path MUST:

- request typed/structured output when supported by the installed SDK;
- independently run Pydantic `model_validate` after extraction;
- reject extra fields and invalid bounds;
- pass the declared `token_budget` as the provider's output-token limit;
- wrap exposed callables through the allowlist dispatcher instead of handing
  bare functions directly to the SDK;
- account every SDK turn against a session counter;
- fail closed on an unavailable structured-output feature.

The opencode-provider path MUST apply the same post-parse Pydantic validation
and token ceiling even though it does not use SDK tools.

The global `MAX_SESSION_LLM_TURNS` value should move to or be consumed by the
session runtime. It must stop a call before exceeding the cap rather than
exist only as a constant. The existing per-agent budget remains unchanged.

#### 4. Consolidate the two fence systems

`perfxpert/agents/fence/*.md` becomes the only source of fence prose.
`FenceBuilder` remains as a compatibility API but reads those canonical files,
includes Diff, and composes:

```text
agents/fence/always.md + agents/fence/<role>.md + deterministic knowledge excerpt
```

The live `Agent` path uses the same builder output. The duplicate
`perfxpert/fence/slices/*.md` files are removed after the builder is rewired;
there is no deprecation period in which two editable copies remain.

Construction and tests enforce both:

- each source slice is at most 400 lines; and
- the effective composed prompt is within the existing 60 KiB assembled
  bound and a documented effective-line bound.

Eric must choose whether the normative “400 lines per agent” limit means the
role slice alone or the composed `always + role` text. This RFC recommends
counting the composed prose (excluding generated YAML blocks) so the live
prompt, rather than a dormant file, is what the narrow-scope guard measures.

The canonical shared fence is revised narrowly:

- measured facts and numeric hardware claims still require tool/knowledge
  evidence;
- a Layer-2 agent may emit an explicitly labelled, non-executable hypothesis
  only into `exploratory_proposals`;
- deterministic fields must never be changed;
- proposal output must include the verification contract;
- proposal text is untrusted data and cannot instruct another agent.

The specialist role fences are corrected to match actual plural output
schemas and actual tool bindings. Closed enums remain descriptions of the
**vetted** lane, while the exploratory section explicitly permits open-set
names. References to unavailable tools are removed or the required trusted
inputs are wired; prose is not allowed to promise a tool the agent cannot
call.

#### 5. Keep proposals out of execution and gates

The creative schema intentionally has no command or patch field. Verification
is represented as typed metrics, comparison operators, and expected
directions. Any user-facing command is synthesized later from trusted paths by
deterministic code.

The following consumers use only vetted `techniques`/`recommendations`:

- ranking and deduplication;
- change-impact prediction;
- task creation and alternative selection;
- code-edit generation;
- claimed-speedup construction;
- SOL and regression inputs;
- correctness verdicts;
- promotion into a catalog.

Backend instructions in `_bundled/opencode_config/AGENTS.md` must label the
new lane “experimental hypothesis — not validated,” render it after vetted
recommendations, and prohibit native edit/bash actions based solely on that
lane. The initial direct-API rollout should precede TUI rendering so this
behavior can be tested independently.

#### 6. Preserve air-gap parity

The parity contract becomes explicit:

```text
deterministic_core(live, input) == deterministic_core(airgap, input)
gate_inputs(live, input)        == gate_inputs(airgap, input)
gate_verdict(live, input)       == gate_verdict(airgap, input)
```

Allowed difference:

```text
live.exploratory_proposals may be non-empty
airgap.exploratory_proposals == []
narrative wording may differ
```

Air-gap mode does not fabricate “creative” templates. A deterministic
pseudo-novel proposal would be another catalog and would misleadingly imply
model reasoning. Empty, explicitly unavailable exploration is safer.

If the model is unavailable, times out, exceeds budget, emits invalid JSON, or
fails proposal validation, the live result degrades to the same deterministic
core with an empty exploratory lane. Provider failure of this optional pass
must not erase a useful catalog result.

#### 7. Proposal promotion workflow

Promotion has three one-way states:

1. **Exploratory** — model-generated, output-only, unmeasured, never executed
   automatically.
2. **Candidate** — a human explicitly selects the proposal and creates a
   review artifact containing the proposal ID, sanitized provenance, expected
   metrics, source anchor (if a code change is proposed), test plan, and
   before/after capture plan.
3. **Proven** — a maintainer-reviewed entry in
   `knowledge/proven_optimizations.yaml`, backed by reproducible fixtures and
   the required correctness evidence. Only this state is returned in the
   vetted catalog.

No agent or MCP call writes a proposal to the repository. A later explicit
CLI command, not exposed over MCP, may scaffold
`.perfxpert/proposals/<proposal_id>.yaml` after user confirmation. The
scaffold is a local draft, not knowledge and not an allowlisted technique.

Promotion requires:

- a stable proposal ID and specialist category;
- no raw user prompt, secret, absolute path, or proprietary kernel name in
  committed provenance;
- a concrete source/implementation description;
- measured baseline and candidate databases;
- output-correctness evidence;
- compile/run success;
- SOL sanity;
- numeric/bitwise comparison;
- regression comparison;
- test-anchor results;
- failure modes and architecture applicability;
- a citation or in-house experiment ID;
- normal knowledge-schema and reviewer approval.

After passing review, the PR updates the relevant set:

- `knowledge/proven_optimizations.yaml`;
- its JSON schema only if a new optional `origin` record is adopted;
- before/after fixtures and description;
- the specialist list and `_ENTRY_SPECS` in the technique-catalog loader;
- `change_impact_models.yaml` only when enough evidence exists to publish a
  conservative numeric prediction.

The current proven runner's three skipped gates are insufficient for this
promotion contract. Either a full promotion runner must be added, or the
documentation must explicitly define which evidence is validated out of band.
Creative promotion must not inherit the current “all five” wording without
making it true.

### Backward compatibility

- Default behavior remains `strict`; no existing user is opted into model
  creativity.
- Existing required output fields remain present and keep their meaning.
- New proposal fields are additive and default to an empty list.
- No tool or handoff count changes.
- No MCP execution exposure changes; the MCP tool count remains 56 if the
  optional scaffold stays CLI-only.
- Existing air-gap fixture technique order remains unchanged.
- A live specialist's vetted list becomes deterministic rather than
  model-replaceable. This is a deliberate safety-compatible behavior change
  and requires changing tests that currently treat live model techniques as
  authoritative.
- Trace-Diff no longer accepts a model override of verdict/deltas. That is a
  bug fix relative to its documented contract.
- Rollback is immediate by forcing
  `PERFXPERT_AGENT_CREATIVITY=strict`. Because the schema field is additive,
  the field need not be removed during rollback; it can remain empty.

### Decision trade-offs

| Decision | Benefit | Cost / rejected alternative |
|---|---|---|
| Separate exploratory lane | Strong noninterference and clear UX | Downstream consumers must render two lanes |
| Deterministic vetted core in live mode | Stronger air-gap parity and reproducibility | Removes live LLM re-ranking of catalog entries |
| Two tiers only | Auditable configuration and safe default | Less per-agent tuning than numeric creativity levels |
| Maximum three proposals within existing budget | Bounds cost and review load | May omit a useful fourth idea |
| Confidence capped at 0.5 | Prevents unverified ideas impersonating proven entries | Confidence is intentionally conservative |
| No patch/command fields | Greatly reduces injection-to-execution risk | User or later deterministic tooling must translate an accepted idea into action |
| No new tools or handoffs | Preserves attack surface and narrow-scope invariants | Creativity is limited to current evidence sources |
| Empty air-gap exploratory lane | Honest semantics and simple parity | Air-gap users do not receive novel hypotheses |
| Human-only promotion | Prevents knowledge poisoning and self-modification | Promotion is slower |
| Canonical live fence source | Removes policy drift | Requires deleting/reworking a currently tested compatibility package |

### Risks and mitigations

1. **Lane confusion:** a consumer may merge both lists.
   - Use different field names and Pydantic types.
   - Add tests that fail on concatenation or ranking across lanes.
   - Stamp status server-side and render a mandatory warning.
2. **Prompt/command injection in proposal prose:**
   - No executable fields; reject control characters, prompt-boundary markers,
     destructive-command patterns, and overlong text.
   - Treat all proposal strings as escaped data in formatters and backend
     prompts.
3. **Hallucinated evidence or counters:**
   - Evidence references must resolve against the runtime manifest.
   - Counter/metric names must be known or be explicitly typed as a new
     measurement request, never as observed.
4. **Numeric reward hacking:**
   - Draft schema does not accept claimed speedup.
   - Expected effects are directional unless a deterministic tool supplied a
     measured bound.
   - Existing SOL checks still apply after implementation.
5. **Catalog shadowing:**
   - Deterministically normalize IDs/titles and reject exact catalog
     duplicates from the exploratory lane.
   - Do not use semantic similarity as a security decision because it would
     itself be nondeterministic.
6. **Provider variation:**
   - Stamp provider/model/fence/catalog fingerprints.
   - Do not require proposal equality across providers.
7. **Optional LLM failure breaking analysis:**
   - Build the complete deterministic result first and catch augmentation
     failure locally.
8. **TUI auto-application:**
   - Stage direct API before TUI display.
   - Add backend contract tests and require explicit user selection before a
     proposal becomes a candidate.
9. **Privacy leakage in promotion artifacts:**
   - Keep raw traces/prompts local; commit only sanitized IDs and reproducible
     synthetic or approved fixtures.
10. **False confidence in the five-gate story:**
    - Do not advertise automatic protection until a production call site
      exists.
    - Require the promotion workflow to invoke or supply evidence for every
      named gate.

## Phased implementation plan

Each phase lands as a separate reviewable PR and keeps
`PERFXPERT_AGENT_CREATIVITY=strict` until Phase 11C is complete. “Phase 11” is
proposed because the current tree reserves that number for durable prediction
state; Eric must confirm the release numbering.

### Phase 11A — Make guardrails executable

No creative output yet.

Modify:

- `perfxpert/agents/framework.py`
  - validate model output against a declared response schema;
  - enforce `token_budget`;
  - route SDK tool wrappers through the allowlist dispatcher;
  - separate validated input from trusted runtime context;
  - recognize a session turn-budget object;
  - extend SDK-import isolation to the canonical `agents` module.
- `perfxpert/agents/runtime.py`
  - own and enforce the per-session turn counter;
  - pass an immutable run policy to `run_agent()`.
- `perfxpert/agents/analysis.py`
  - make the rule verdict and collected facts authoritative in live mode;
  - limit any model contribution to a separately typed advisory note so
    Recommendation routing cannot diverge from the air-gap classifier.
- `perfxpert/fence/_builder.py`
  - read canonical `agents/fence` files and add Diff;
  - expose one deterministic composition path.
- `perfxpert/fence/slices/*.md`
  - remove after the builder is rewired, eliminating duplicate policy text.
- `tests/test_agents/test_framework.py`
  - add invalid-output, extra-field, token-budget, turn-budget, and wrapped
    allowlist tests.
- `tests/test_agents/test_no_sdk_import_leak.py`
  - detect `import agents` and `from agents ...`.
- `tests/test_agents/test_schema_field_caps.py`,
  `test_fence_size_guardrail.py`, and
  `test_tool_allowlist_guardrail.py`
  - done — these now derive from `AGENT_BUILDERS`; see Implementation status.
- `tests/test_agents/test_root.py`
  - make fence/tool alignment detect tool-like references outside the
    allowlist section or replace the fragile Markdown parser with canonical
    metadata.
- `tests/test_fence/test_builder.py`,
  `test_determinism.py`, and `test_filters.py`
  - assert the live agent prompt is the builder result and contains the shared
    fence exactly once.
- `scripts/exit_dashboard.py`
  - invoke real test paths, consume actual air-gap results, and fail closed on
    missing PR-lane metrics.
- `docs/audit_gate_runbook.md`
  - describe the corrected dashboard inputs.
- `.github/workflows/perfxpert-ci.yml`
  - add a path-filtered, CPU-only PR job for unit, agent, runtime, fence,
    parity, red-team, regression-gate, and audit-dashboard checks. Do not
    inspect or run the vendored `experimental/python/perfxpert/opencode/`
    test tree.

Exit criteria:

- all claimed framework guardrails have a live-path test;
- all eight agents are in every narrow-scope inventory;
- the effective live prompt includes `always.md`;
- the dashboard is GO or explicitly blocked, never silently pending.

### Phase 11B — Add inert additive contracts

Add:

- `perfxpert/agents/creativity.py`
  - policy resolution, proposal draft-to-final validation, trusted evidence
    manifest, deterministic proposal IDs, normalization/deduplication, and
    provenance stamping.
- `tests/test_agents/test_creativity_policy.py`
  - tier lattice, default/air-gap clamping, Layer-2-only capability, malformed
    config, and server-owned fields.
- `tests/test_agents/test_exploratory_proposals.py`
  - nested schema bounds, evidence resolution, confidence cap, item cap,
    duplicate rejection, and no-command contract.
- `tests/test_agents/test_diff_specialist.py`
  - fill the current Diff isolation-test gap before adding creative behavior.

Modify:

- `perfxpert/config/_config.py` and `perfxpert/config/__init__.py`
  - add the validated `agent_creativity` maximum and environment mapping.
- `perfxpert/agents/schemas.py`
  - add frozen nested draft/final/evidence/verification/provenance models;
  - add empty `exploratory_proposals` fields to Compute/Memory/Latency and
    Recommendation outputs;
  - validate the nested Diff key while retaining five top-level fields;
  - add optional `database_path` to Recommendation and single-run specialist
    inputs.
- `perfxpert/agents/framework.py`
  - add `catalog_only` / `additive_exploration` capability with construction
    validation.
- `perfxpert/agents/runtime.py`
  - resolve configured, requested, capability, and air-gap tiers.
- `perfxpert/agents/{compute,memory,latency,diff}_specialist.py`
  - populate the new field with `[]` only; no model creativity yet.
- `perfxpert/agents/recommendation.py`
  - propagate an empty separate lane and thread trusted database/evidence
    fields.
- `perfxpert/tools/agents/{compute,memory,latency,diff,recommendation}.py`
  - serialize and document the additive fields without adding a self-elevation
    MCP argument.
- `tests/test_config/test_config.py`,
  `tests/test_agents/test_schemas.py`,
  `tests/test_agents/test_runtime.py`, and existing specialist tests
  - assert strict defaults and backward-compatible construction.

Exit criteria:

- default and air-gap snapshots preserve existing vetted outputs;
- every public wrapper returns the additive field;
- no model response can populate it yet.

### Phase 11C — Enable bounded exploration for direct calls

Modify:

- `perfxpert/agents/{compute,memory,latency,diff}_specialist.py`
  - build deterministic core/evidence first;
  - perform at most one creative augmentation call;
  - validate drafts independently;
  - make provider failure return the untouched core;
  - forbid model replacement of vetted fields (already done for Diff
    arithmetic/verdict — see Implementation status).
- `perfxpert/agents/recommendation.py`
  - keep separate propagation/deduplication and never rank across lanes.
- `perfxpert/agents/fence/always.md`
  - permit labelled hypotheses only in the exploratory channel while
    preserving evidence and non-execution rules.
- `perfxpert/agents/fence/{compute,memory,latency,diff}_specialist.md`
  - correct real tool/output contracts and add the bounded divergent-reasoning,
    self-critique, evidence, and falsification instructions.
- `perfxpert/agents/fence/recommendation.md`
  - define strict two-lane handling.
- `tests/test_agents/test_{compute,memory,latency,diff}_specialist.py`
  - assert deterministic-core equality, open-set acceptance only in the new
    lane, maximum count, invalid-proposal dropping, and provider-failure
    fallback.
- `tests/test_agents/test_recommendation.py`
  - assert no mixing, cross-lane ranking, prediction attachment, or vetted
    hash pollution.

Change existing test expectations:

- Replace `test_compute_specialist_ranks_techniques_llm_mode` with a test that
  the model cannot replace deterministic techniques and can only add a valid
  exploratory draft.
- Any Memory/Latency fake response that puts arbitrary content in
  `techniques` moves that content to the draft schema.
- A Diff regression test proving a model-supplied verdict/delta override is
  ignored already exists (`tests/test_agents/test_diff_specialist.py`); extend
  it to cover the exploratory lane rather than adding a new file.

Exit criteria:

- direct Python/MCP specialist and Recommendation calls can return novel,
  typed proposals when the server is explicitly configured exploratory;
- strict/air-gap vetted fields remain byte-identical.

### Phase 11D — Harden external surfaces and audit gates

Modify:

- `perfxpert/_bundled/opencode_config/AGENTS.md`
  - document the two lanes, warning label, no-auto-apply rule, explicit user
    selection, and full verification requirement.
- backend prompt adapters under `perfxpert/cli/_backend/`
  - carry the same two-lane contract where they stage agent instructions.
- backend gate-hook tests under `tests/test_cli/test_backend/`
  - assert exploratory output alone never authorizes a native execution step.
- `tests/test_integration/test_airgap_parity.py`
  - compare all deterministic specialist fields and Diff verdict/deltas, not
    only Root route/Analysis class.
- `tests/test_parity/*`
  - retain the ≥95% fixture contract and assert default strict primary
    techniques do not move.
- `tests/test_red_team/attack_registry.py`
  - add two RFC-authorized attacks:
    `exploratory_lane_confusion` and
    `exploratory_proposal_command_injection`.
- New red-team tests under `tests/test_red_team/`
  - attempt to overwrite vetted techniques/verdicts, self-stamp `proven`,
    forge provenance, smuggle destructive commands/patches, reference unknown
    evidence, and inject prompt-boundary markers.
- `scripts/exit_dashboard.py` and `docs/audit_gate_runbook.md`
  - raise the normative defeated-attack count from 14 to 16.
- `docs/architecture/agent-hierarchy.md`,
  `docs/architecture/gate-cascade.md`, `docs/architecture.md`,
  `docs/contributing/agents.md`, `docs/contributing/schemas.md`,
  `docs/guides/python-api.md`, `docs/guides/agentic-mode.md`,
  `CONTRIBUTING.md`, and `CHANGELOG.md`
  - document actual orchestration, effective fence composition, two-lane
    contract, current gate-interposition scope, configuration, and rollout.

The 14-to-16 red-team expectation is the only normative count change. It
strengthens the gate and is justified here because the audit runbook requires
an RFC before adding a new attack class. No existing attack or threshold is
weakened.

Exit criteria:

- 16/16 red-team outcomes;
- 100% deterministic-core air-gap parity;
- existing fixture parity remains at least 95%;
- no execution authorization from exploratory output;
- checked-in PR CI is green.

### Phase 11E — Add human promotion tooling

Add:

- `perfxpert/cli/proposals.py`
  - explicit, non-MCP `validate`, `show`, and `scaffold` operations for a
    user-supplied proposal JSON document;
  - write only under a confined `.perfxpert/proposals/` root after explicit
    invocation.
- `tests/test_cli/test_proposals.py`
  - path confinement, schema validation, privacy stripping, idempotence, and
    no repository-knowledge mutation.
- `tests/test_integration/test_proposal_promotion.py`
  - candidate evidence bundle through the chosen full-gate promotion runner.

Modify:

- `perfxpert/__main__.py`
  - register the explicit proposal subcommand.
- `perfxpert/knowledge/_schemas/proven_optimizations.schema.json`
  - optionally accept a sanitized `origin` object containing only proposal ID
    and non-sensitive generation digests.
- `docs/contributing/proven_optimizations.md` and
  `docs/contributing/knowledge.md`
  - document exploratory → candidate → proven review.
- `tests/test_knowledge/test_proven_optimizations.py` and
  `tests/test_regression_gate/proven_optimization_runner.py`
  - enforce the human promotion evidence contract and remove or accurately
    document skipped gates.

Exit criteria:

- generated output cannot mutate knowledge;
- a human can create a reviewable candidate without copying untrusted free
  text into an executable path;
- only a normal reviewed PR can make a technique vetted.

## Alternatives considered

### 1. Remove the fence enums and keep one `techniques` list

Rejected. The Python schema is already open enough to do this, which is the
problem: a novel item is indistinguishable from proven knowledge and can flow
into ranking, prediction, task, and execution-oriented consumers. Labelling
inside each dictionary is weaker than a separate typed lane and easier for a
consumer to ignore.

### 2. Increase the five-tool cap or allow Layer-2 handoffs

Rejected. Current specialists already have enough read-only evidence sources
for an advisory hypothesis once database/evidence inputs are correctly
threaded. More tools increase attack surface and prompt complexity.
Layer-2-to-Layer-2 handoffs would blur ownership and do not create creativity;
returning a proposal to the parent is sufficient.

### 3. Let live LLM output drive vetted ranking or gate decisions

Rejected. Seeding, temperature limits, majority voting, or repeated sampling
do not make a model decision reproducible or air-gap equivalent. Gate and
vetted-control decisions must remain deterministic; only an additive advisory
lane may vary.

### 4. Expand YAML only

Rejected as the complete solution. Catalog growth remains valuable and is the
destination of promotion, but it cannot surface a workload-specific idea
before a maintainer has encoded it. It also does not satisfy the request to
build creative reasoning into the workflow.

### 5. Generate deterministic “creative” proposals in air-gap mode

Rejected. If an idea is deterministic and maintained, it belongs in a
catalog. Labelling a template as creative would be misleading and would add a
second unreviewed rule catalog.

### 6. Automatically persist or promote high-confidence proposals

Rejected. It would make a read-only MCP call stateful, enable knowledge
poisoning, create privacy risks, and let the model influence its future
allowlist. Confidence is not evidence. Promotion remains an explicit human
and CI workflow.

### 7. Add a numeric `creativity_level` from 0 to 3

Rejected for the first release. Numeric levels imply meaningful gradations
that do not yet have distinct, testable security contracts. `strict` and
`exploratory` are sufficient. A future RFC may add a research tier after
measured use.

## Unresolved questions

Eric, as maintainer, must settle these before acceptance:

1. **Live vetted behavior:** approve the recommended stronger invariant that
   live models can no longer replace/re-rank `techniques`, or explicitly
   define which non-gate fields may remain nondeterministic.
2. **Rollout surface:** should Phase 11C ship direct specialist and
   Recommendation APIs only, or may the backend TUIs display proposals in the
   same release? This RFC recommends direct API first.
3. **Execution boundary:** is an explicit user confirmation sufficient before
   a TUI acts on a proposal, or must the native-tool gate track an accepted
   proposal ID mechanically?
4. **Gate integration:** must the five-gate cascade be wired as automatic
   production middleware before any exploratory proposal can become a
   candidate? This RFC requires it for promotion, but read-only proposal
   generation can ship independently.
5. **Promotion evidence:** should the proven corpus continue allowing
   compile/bitwise/anchor evidence to be validated out of band, or should the
   runner be changed to execute all five? The docs and implementation must
   agree.
6. **Fence cap interpretation:** does 400 lines apply to each source slice or
   the effective composed prose? This RFC recommends effective composed prose,
   excluding generated YAML.
7. **Fence compatibility:** may the duplicate `perfxpert/fence/slices/` files
   be removed in Phase 11A while retaining the `FenceBuilder` API, or is a
   deprecation release required?
8. **Diff wire shape:** approve adding `exploratory_proposals` under internal
   `kernel_deltas` and flattening it publicly, rather than breaking the
   five-field cap.
9. **Proposal limits:** approve maximum three proposals and the 0.5 confidence
   ceiling, or choose stricter values based on pilot UX.
10. **Persistence/privacy:** should Phase 11E ship a local scaffold at all, and
    may a proven entry retain sanitized origin digests?
11. **Phase number:** current comments reserve Phase 11 for durable prediction
    state. Should this work become Phase 11 (sharing that event/provenance
    foundation) or a later numbered phase?
12. **Reviewer threshold:** repository docs disagree between two and three
    reviewers for shared-fence architectural changes. This RFC recommends the
    stricter three-core-maintainer threshold and a two-week discussion.

## Test plan

### Unit

- Construction rejects creative capability on Layer 0/1.
- Config defaults strict, rejects unknown values, respects precedence, and
  clamps air-gap to strict.
- All eight agents satisfy tool, fence, and schema caps.
- Effective live fences contain the shared slice exactly once and are
  deterministic.
- Framework validates model input/output, forbids extras, enforces token and
  session-turn budgets, and dispatches only allowed tools.
- Draft validation rejects missing evidence/verification/failure modes,
  unknown targets/references, more than three items, confidence above 0.5,
  model-owned status/provenance, commands, patches, paths, prompt markers, and
  overlong fields.
- Proposal IDs are deterministic for identical normalized content and trace
  fingerprint.
- Exact catalog duplicates are rejected from the exploratory lane.
- Strict and provider-failure paths return a complete deterministic result.
- Diff ignores model-supplied arithmetic and verdict.
- Recommendation never mixes lanes or attaches catalog predictions to novel
  proposals.

### Integration

- For fixed fixtures, compare live-mocked and air-gap deterministic fields for
  all four specialists.
- Assert exploratory mode changes only `exploratory_proposals` and permitted
  narrative.
- Assert a malformed creative response cannot remove vetted output.
- Assert direct Python and MCP wrappers serialize the same additive shape.
- Assert MCP remains read-only and its tool count does not change.
- Assert backend rendering escapes proposal text and labels it unvalidated.
- Assert a proposal cannot authorize a native execution tool without the
  maintainer-selected confirmation policy.
- Assert promotion invokes or provides evidence for every required gate.

### Red-team

Add two normative attacks:

1. `exploratory_lane_confusion`: attempt to replace vetted techniques/diff
   verdict, self-label as proven, forge a catalog ID, or smuggle an
   exploratory item into Recommendation's vetted list.
2. `exploratory_proposal_command_injection`: inject shell commands, patches,
   compiler/profiler flags, path traversal, prompt-boundary tokens, forged
   evidence, and oversized nested content.

Extend the existing LLM-unavailable parity attack to compare deterministic
specialist core fields. Keep all existing 14 attacks unchanged.

### Regression and parity

- Existing primary bottleneck/type/technique fixture signals remain unchanged
  in default strict mode.
- Agreement remains at least 95%.
- Proven-optimization false-positive rate remains at most 5%.
- Near-threshold positive tests remain passing.
- Gate constants and thresholds do not change.
- Air-gap deterministic-core identity is 100%.

### Benchmark impact

- `strict`: no extra LLM call; after deterministic-core cleanup it may be
  cheaper than the current live specialist path.
- `exploratory`: at most one specialist augmentation run, bounded by the
  configured per-run turn limit (default ten), the enforced 100-turn session
  ceiling, and at most 3072 output tokens total; the proposal sub-budget
  should target at most 1024 tokens.
- No additional tool slot, process execution, network path, or GPU benchmark
  is introduced.
- Record provider latency, validation-drop rate, duplicate rate, proposal
  count, explicit user-selection rate, and eventual gate-pass/promotion rate.
  Do not record raw prompts or proprietary kernel names by default.

### Existing expectations intentionally changed

The following changes are red flags and require explicit reviewer approval,
but they strengthen rather than weaken safety:

- live fake-provider techniques no longer override deterministic catalog
  techniques;
- live Diff verdict/deltas no longer override deterministic arithmetic
  (already landed; no existing test expected the old behaviour);
- shared `always.md` becomes part of the actual prompt;
- all eight agents, including Diff, enter every cap/allowlist test;
- red-team required count increases from 14 to 16;
- the audit dashboard fails closed on missing PR-lane evidence.

No existing gate threshold, tool cap, fence cap, schema cap, MCP class rule, or
attack expectation is relaxed.

## Migration

1. Merge Phase 11A with no creativity config or output change.
2. Merge Phase 11B with additive empty fields and strict default.
3. Merge Phase 11C behind
   `PERFXPERT_AGENT_CREATIVITY=exploratory`; keep release notes marked
   experimental.
4. Merge Phase 11D before enabling TUI display.
5. Observe at least one release/pilot cycle before Phase 11E promotion tooling
   is enabled.

Rollback:

- set `PERFXPERT_AGENT_CREATIVITY=strict` globally;
- leave additive fields empty so clients do not face a schema rollback;
- retain deterministic core and guardrail hardening;
- disable TUI rendering independently if UX or injection findings appear;
- do not roll back new red-team cases or restored CI coverage.

## Prior art

- PerfXpert's own deterministic gate middleware separates measured decisions
  from model narrative.
- `proven_optimizations.yaml` already models a reviewed promotion destination
  with citations, preconditions, failure modes, and fixture pairs.
- Feature-flagged advisory/control-plane separation is a common safety pattern
  for recommendation systems: uncertain candidates are generated broadly,
  but only a deterministic or human-reviewed lane may trigger action.
- Content-addressed proposal IDs and server-stamped provenance follow standard
  supply-chain traceability practice without treating model self-report as
  trustworthy.

## Approval checklist (fill in at merge time)

- [ ] Eric / project lead
- [ ] Performance-domain core maintainer
- [ ] Agents/LLM-domain core maintainer
- [ ] Security-focused reviewer
- [ ] CI green
- [ ] Two-week architectural discussion elapsed
- [ ] Live vetted-core policy resolved
- [ ] Gate-promotion evidence policy resolved
