# Bounded Specialist Creativity

This guide explains why PerfXpert has an exploratory proposal channel, how
that channel is kept separate from trusted recommendations, and where the
enforcement lives. It assumes familiarity with GPUs and Python, but not with
PerfXpert.

To show the feature to someone rather than explain it, see
[Demoing bounded specialist creativity](demoing-bounded-creativity.md), which
walks through a scripted-model demo that needs no API key.

## 1. Orientation

PerfXpert analyzes AMD ROCm workloads. It turns profiling data, primarily
`rocprofv3` databases, into bottleneck classifications and optimization
advice. Its trusted path is deliberately conventional: deterministic rules
interpret measured data, then rank techniques from reviewed YAML knowledge
catalogs.

The agent hierarchy is an ownership model. Layer 0 Root classifies intent;
Layer 1 Analysis, Recommendation, and Correctness own classification,
strategy, and accept/revert decisions; Layer 2 Compute, Memory, Latency, and
Diff specialists own narrow performance domains. These agents are also
directly callable. The current Python runtime is not a literal SDK handoff
graph: Recommendation dispatches Compute, Memory, or Latency as Python
functions, while Diff is normally called directly. See
`perfxpert/agents/recommendation.py:121-210`.

Do not confuse these three agent *layers* with the two creativity *tiers*
introduced below.

## 2. The problem this feature solves

The catalog is a closed set: a specialist can select a technique only after a
maintainer has encoded it and supplied the evidence expected of trusted
knowledge. This makes results reproducible and reviewable, but it leaves no
place for a workload-specific idea that is not yet in the catalog.

An LLM has the opposite properties. It can connect unusual signals and form a
new hypothesis, but it can also invent a benchmark, cite a tool it never
called, overstate confidence, or present an idea as established fact. Simply
letting the model “suggest optimizations” inside the normal `techniques` list
would erase the distinction. Downstream code could rank the invention, attach
an impact prediction, turn it into a task, or present it as advice.

RFC 0001 therefore draws the boundary between a deterministic control plane
and a nondeterministic advisory plane. Creativity is useful beside routing,
catalog selection, measured arithmetic, and correctness decisions, but not
inside them.

## 3. The two-lane design

The **vetted lane** carries `techniques` from a specialist and
`recommendations` from Recommendation. The runtime owns this lane. It loads
and ranks catalog entries, attaches only catalog-backed predictions, and sets
the specialist confidence and citations without consulting model output. In
the current Compute, Memory, and Latency runners, confidence is fixed at
`0.6` and the top-level citations list is empty. Diff's analogous trusted
fields are its wall-time delta, per-kernel deltas, verdict, and confidence.

The **exploratory lane** carries `exploratory_proposals`. The model owns only
the draft's semantic content: title, hypothesis, mechanism, optional target
kernel, evidence claims, expected effects, verification plan, assumptions,
failure modes, and confidence. The runtime decides whether that draft is
admissible and owns the final `proposal_id`, `status`, `specialist`, and
`provenance`.

The lanes must never be concatenated, jointly ranked, or deduplicated against
one another. Recommendation transports proposals in a separate field and
deduplicates them only by runtime-generated ID
(`perfxpert/agents/recommendation.py:200-210`). Diff is already at the
five-top-level-field schema cap, so its internal lane is nested under
`kernel_deltas`; the public wrapper flattens it again. This odd shape preserves
the existing cap without creating an unvalidated sixth field.

The central invariant is **air-gap parity**. Given the same trusted inputs,
live and air-gapped runs must produce the same route, classification, vetted
techniques, ranking, measured Diff values, and gate inputs and verdicts.
Narrative wording may differ, and a live exploratory lane may be non-empty;
an air-gapped exploratory lane is always empty. PerfXpert does not manufacture
deterministic “creative” templates in air-gap mode: a maintained,
deterministic idea belongs in a catalog.

Parity makes the design auditable. A test can give the live model hostile
values for every trusted field and compare the result with air-gap output. If
the deterministic fields remain identical, the claim is enforced by code
rather than by prompt wording. If proposal validation fails, the complete
vetted result still returns with an empty proposal lane.

## 4. How it is implemented

### Policy resolution and capability

There are two creativity tiers: `strict`, the default, and `exploratory`.
Exploration is enabled only when all three independent conditions permit it:

1. Configuration sets the deployment ceiling to `exploratory`, through
   `agent_creativity` or `PERFXPERT_AGENT_CREATIVITY`.
2. The run is live, not air-gapped.
3. The agent declares
   `AgentCapability.ADDITIVE_EXPLORATION`.

`resolve_tier()` implements this three-way AND
(`perfxpert/agents/creativity.py:51-80`). The ceiling is configuration, not an
MCP argument, so a calling model has no handle with which to raise its own
permission.

`AgentCapability` defaults to `CATALOG_ONLY`. Agent construction rejects
`ADDITIVE_EXPLORATION` on Layers 0 and 1 because an open-set output there
could steer routing, classification, or acceptance. Only the four Layer 2
specialists opt in; see `perfxpert/agents/framework.py:108-116` and
`perfxpert/agents/framework.py:206-266`.

Compute, Memory, and Latency build the vetted result before considering a
provider. Under `strict` they return immediately, because every field they
publish is runtime-owned. Under `exploratory` they make one bounded model
call, then use only its proposal drafts. Memory is the clearest representative
at `perfxpert/agents/memory_specialist.py:83-138`. Diff may still use a model
for narrative in a live strict session; `strict` disables its exploratory
lane, not all ordinary LLM-assisted prose.

### The framework boundary

Every model-backed agent invocation enters through `run_agent()`
(`perfxpert/agents/framework.py:853-895`). This is where air-gap mode prevents
the provider call, `RunPolicy` charges the session turn budget, tool wrappers
re-check the agent allowlist, and structured output is validated.
`PERFXPERT_AIRGAP=1` is treated as a floor at this boundary: an explicit
`airgap=False` cannot authorize a network call.

All handoff models inherit a frozen Pydantic base with `extra="forbid"`
(`perfxpert/agents/schemas.py:20-23`). Framework validation creates an
all-optional mirror of the declared output schema. Fields are optional because
agents merge model output over an already-complete deterministic result, but
unknown keys and wrong types reject the whole structured response. The logic
is in `perfxpert/agents/framework.py:306-363`.

Proposal fields need a deliberate exception. The public output declares final
`ExploratoryProposal` objects, while the model must emit drafts. Fields marked
`model_supplies: "draft"` are therefore relaxed to `Optional[Any]` in the
partial mirror. This does not trust their contents; it merely postpones nested
validation. `build_proposals()` then parses each item as the strict
`ExploratoryProposalDraft`, whose inherited `extra="forbid"` rejects forged
runtime fields and malformed shapes.

### Evidence and proposal construction

An `EvidenceManifest` is an allowlist of evidence references for one run. It
contains declared tools that the SDK run record says were actually called,
kernel names supplied from trusted measured inputs, and known catalog entry
names. `manifest_from_run()` maps the SDK's sanitized tool names back to their
declared dotted names and ignores anything not declared
(`perfxpert/agents/creativity.py:294-320`). Agent construction also rejects
two tool names that would collapse to the same sanitized spelling.

Every accepted proposal must cite at least one manifest entry. Every evidence
`ref` must match exactly, and an optional `target_kernel` must be among the
measured kernels. A model cannot manufacture evidence by adding `tool_calls`
to its JSON because tool-call records come from the framework, outside the
structured response. The manifest proves that a source was available; it
does not independently prove that the model's free-form `observation`
accurately summarizes the source.

The schema split is at `perfxpert/agents/schemas.py:40-118`. After draft
validation, the runtime computes a content-addressed `pxp-exp-...` ID from
specialist, title, hypothesis, mechanism, and target, stamps the constant
status `exploratory`, and constructs the final frozen model
(`perfxpert/agents/creativity.py:125-198`). At most three drafts are considered
per specialist, and duplicate IDs collapse. Dedupe intentionally does not use
title or confidence, because model-controlled similarity could let one
proposal suppress another.

Exploratory confidence has a hard `0.0..0.5` ceiling in both the draft schema
and runtime validation. A draft claiming `0.95` is rejected, not silently
clamped to `0.5`. Clamping would preserve a proposal whose author overstated
its evidence and make it indistinguishable from one that respected the
contract.

The schema has no command, patch, compiler-flag, or edit-path field. Proposal
prose is still untrusted data: the bundled orchestrator contract requires it
to be displayed separately and forbids acting on it automatically
(`perfxpert/_bundled/opencode_config/AGENTS.md:139-168`).

Four current implementation limits are important when reading the RFC as a
design of record. First, `expected_effects`, verification metrics, assumptions,
and failure modes currently default to empty lists; the fences require them,
but proposal construction does not yet enforce that they are non-empty.
Second, the runtime owns the complete provenance object but currently
populates the provider. `run_agent()` does not carry a model identifier into
proposal construction, and the trace, fence, and catalog digests retain empty
defaults. Third, assumptions and failure-mode strings are not individually
length-bounded, and proposal prose is not scanned for command text or
prompt-boundary markers. Fourth, the exploratory specialist runners do not
catch provider exceptions locally, so a provider failure can still abort an
exploratory run rather than returning the vetted core with an empty lane. The
structural non-execution rule and orchestrator policy still apply, but these
are gaps relative to the stronger RFC wording.

### Human promotion

`perfxpert proposals list|show|promote` reads a saved agent result. `list`
summarizes proposals, and `show` keeps evidence, assumptions, and failure
modes attached. `promote` emits a candidate YAML skeleton; it never appends to
the knowledge catalog.

The skeleton deliberately omits four required fields that can exist only
after an experiment: `measured_speedup_range`, `source_citation`,
`preconditions`, and `fixture_pair`. They appear only as comments, so the
generated entry fails catalog-schema validation. Type-valid placeholders such
as `"TODO"` or `[1.0, 1.0]` would be unsafe because an untouched skeleton
could pass CI.

Model prose is placed into a Python dictionary and serialized with
`yaml.safe_dump`, then reparsed to verify that none of the withheld fields was
injected. Output to the knowledge tree, a symlink, or an existing file is
refused (`perfxpert/cli/proposals_cmd.py:252-397`). A human must run the
before/after experiment, complete all four fields, satisfy the correctness
evidence, and submit the normal reviewed catalog change.

## 5. What changed on this branch

The 14 commits form three themes rather than 14 independent features.

### Guardrails that previously existed only on paper

Before adding creativity, the branch made Trace-Diff arithmetic and verdicts
model-independent and put Diff into the canonical agent inventory. Phase 11A
then made model-output validation, token budgets, the session turn budget, and
live tool-allowlist dispatch real framework behavior. It consolidated the two
drifting fence systems so `always.md` reaches every live prompt, made Analysis
classification deterministic, repaired the audit dashboard, and added a
path-filtered CPU-only CI workflow.

These changes matter because a new “safe” lane cannot rely on surrounding
controls that are only comments or tests nobody runs.

### The feature itself, Phases 11B through 11E

Phase 11B added the inert contract: creativity configuration, capability
lattice, draft/final schemas, evidence manifest, IDs, and empty additive
fields. Phase 11C enabled proposal construction while freezing catalog
techniques, confidence, and citations in both modes; Recommendation gained
separate propagation and ID-only dedupe. Phase 11D added hostile parity tests,
raised the red-team inventory from 14 to 16 attacks, and documented the
two-lane presentation rule for backend orchestrators. Phase 11E added the
review and intentionally incomplete promotion skeleton, plus an optional
catalog `origin` record.

### Correctness fixes found while validating the design

- **Model-controlled trusted fields.** Analysis and later Root had accepted
  model classification or advice; Diff accepted model confidence; Correctness
  accepted an invented alternative technique. Those paths could change
  routing or place unproven content where consumers expected rules and
  measurements. The runtime now owns all of them.
- **Promotion YAML injection.** Promotion originally assembled YAML with
  strings, so model prose could escape its scalar and inject the four
  “measured” fields. Safe serialization, reparsing, and output-path refusal
  prevent an unmeasured proposal from becoming schema-valid or writing
  directly into knowledge.
- **Unwired Diff lane.** Diff declared the capability and documented proposals
  but never extracted them from model output. The nested `field_path` wiring
  now makes the lane real while preserving the five-field cap.
- **Air-gap environment clamp.** A call-site `airgap=False` could override
  `PERFXPERT_AIRGAP=1` at the framework boundary and reach a provider. The
  environment is now authoritative there.
- **Strict-tier provider call.** Compute, Memory, and Latency called a model
  even though strict mode discarded its entire response. Early return removes
  wasted latency and prevents an irrelevant provider failure from destroying
  a valid catalog result.
- **Evidence-name ambiguity and review robustness.** Agent construction now
  rejects tool names that become identical after SDK sanitization. Proposal
  review also treats malformed saved lanes or missing IDs as absent instead
  of crashing.
- **CI correctness.** The new workflow initially could not install because the
  bundled TUI build needed Bun and an uninitialized submodule. It now skips
  that unused build, installs the LiteLLM extra so provider routing is truly
  exercised, and uses SDK test doubles that accept enforced model settings.
  The branch's six CI lanes then passed on Python 3.10 and 3.12.

## 6. Reading map

Read these in order:

1. `docs/architecture/agent-hierarchy.md` gives the agent vocabulary and the
   intended ownership boundaries; keep the direct-call caveat from Section 1
   in mind.
2. `docs/rfcs/0001-bounded-specialist-creativity.md` is the design and threat
   model, including rejected alternatives and the promotion rationale.
3. `perfxpert/agents/schemas.py` shows exactly what a model may draft and what
   only the runtime may place in a final proposal.
4. `perfxpert/agents/framework.py` shows the common execution boundary,
   capability check, air-gap clamp, budgets, tool wrappers, and partial output
   validation.
5. `perfxpert/agents/creativity.py` contains tier resolution, evidence
   matching, confidence rejection, proposal IDs, construction, and dedupe.
6. `perfxpert/agents/memory_specialist.py` is the clearest end-to-end example;
   compare Compute and Latency, then read Diff for its nested wire shape.
7. `perfxpert/agents/recommendation.py` shows that proposals are transported
   beside recommendations and never enter vetted dedupe.
8. `perfxpert/agents/fence/*.md` shows the constraints presented to each model;
   `always.md` is composed with one role slice.
9. `perfxpert/_bundled/opencode_config/AGENTS.md` shows the final consumer's
   obligation to label, separate, and never auto-apply proposals.
10. `perfxpert/cli/proposals_cmd.py` completes the path from saved proposal to
    a human-owned, still-invalid promotion skeleton.
