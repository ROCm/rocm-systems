# Bounded Specialist Creativity

How PerfXpert lets a language model suggest an optimization without ever
letting that suggestion pass as measured advice.

Read the first two sections for the idea. Read the rest when you need to
change the code. To *show* the feature to someone instead of explaining it,
see [Demoing bounded specialist creativity](demoing-bounded-creativity.md) —
a scripted walkthrough that needs no API key.

## The short version

PerfXpert reads a GPU profile and tells you how to make your code faster. Its
advice comes from a **catalog**: a hand-maintained list of optimizations that
someone actually benchmarked. Each entry records what it changes, when it
applies, and the speedup that was measured. That is why the advice is worth
acting on — every recommendation traces back to a real experiment.

The catalog's strength is also its limit. If your workload has a problem
nobody has catalogued, PerfXpert has nothing useful to say about it.

A language model could fill that gap. It can look at an unusual profile and
suggest something genuinely new. It can also invent a speedup number, cite a
benchmark that never existed, or state a guess with complete confidence.

So PerfXpert keeps **two lists, and never lets them touch**:

- **Recommendations** come only from the catalog. A model cannot add to this
  list, reorder it, or reword it. It is identical whether or not a model was
  involved at all.
- **Exploratory proposals** are where model ideas go. Each is labelled
  unproven, capped in confidence, and tied to evidence from your actual run.

Everything else in this guide is the machinery that keeps that separation
honest.

> **Analogy.** A medical journal prints peer-reviewed findings and,
> separately, letters proposing hypotheses. This feature is the editorial rule
> that a hypothesis can never be typeset into the findings section — plus the
> checks that stop a letter from citing a study nobody ran.

## Why the separation has to be enforced, not just intended

Suppose the model returns this for a memory-bound kernel:

> Use `__builtin_nontemporal_store` here. Measured 2.1x on MI300X.

Nobody measured 2.1x. The model produced a plausible-looking number, and the
word "measured" is doing work it has not earned. An engineer reads it, spends
three days, gets nothing — and now distrusts every recommendation PerfXpert
makes, including the ones that were real.

The catalog's value is not that its advice is clever. It is that every number
in it came from an experiment. A single fabricated number contaminates the
whole list, which is why the boundary is enforced in code rather than
requested in a prompt.

## The five rules

Each rule exists because of a specific way the separation could fail.

**1. It is off by default.** Model proposals require the `exploratory` tier,
which is not the default. In the default `strict` tier a specialist does not
call the model at all — so this costs nothing until you ask for it.

**2. An idea may only cite what actually happened.** Before a proposal is
accepted, the runtime checks every reference in it against a list of the tools
that really ran and the kernels that were really measured in *your* analysis.
A proposal citing a profiler that never ran, or a kernel nobody profiled, is
thrown away. Crucially, that list is built from the framework's own record —
not from anything the model says — so the model cannot vouch for itself.

**3. Confidence is capped at 0.5, and cheating is rejected rather than
clipped.** A proposal claiming 0.95 is discarded, not quietly lowered to 0.5.
Clipping would leave a dishonest proposal looking exactly like an honest one.

**4. The model does not get to label its own idea.** The identifier, the
`exploratory` status, which specialist produced it, and its provenance are all
stamped by the runtime. The model fills in a *draft* form that has no fields
for any of them, so it cannot promote itself to "recommended".

**5. Promoting an idea into the catalog requires a human and a real
experiment.** `perfxpert proposals promote` writes a catalog entry for you and
deliberately leaves four fields blank — the measured speedup, the citation,
the preconditions, and the before/after fixture pair. Those only exist once
someone runs the experiment. The entry fails validation until they do, so an
untouched skeleton can never slip into the catalog.

## The one property that ties it together: air-gap parity

PerfXpert can run fully air-gapped, with no model in the loop. The invariant
is that **live and air-gapped runs produce the same measured output** — same
classification, same ranked techniques, same numbers. Only the narrative
wording differs, and only the exploratory list may be non-empty (air-gapped,
it is always empty).

This is what makes the whole design checkable. A test can hand a live model
hostile values for every trusted field and compare the result against the
air-gapped run. If the measured fields are identical, the separation is
enforced by code, not by prompt wording. That test is real:
`tests/test_integration/test_airgap_parity.py`.

---

The rest of this guide is implementation detail.

## How it works in the code

### Where PerfXpert's agents sit

PerfXpert analyzes AMD ROCm workloads, mostly from `rocprofv3` trace
databases. Its agents form three layers: Root classifies what you asked for;
Analysis, Recommendation, and Correctness own classification, strategy, and
accept/revert decisions; and four specialists — Compute, Memory, Latency,
Diff — own narrow performance domains.

Only the specialists may propose. Root and the middle layer route and decide,
and an open-ended suggestion there could change *what gets measured*. A
suggestion from a specialist can only ever be additive.

One caveat while reading: the layer diagram describes ownership, not literal
SDK handoffs. Recommendation calls Compute, Memory, and Latency as ordinary
Python functions, and Diff is normally invoked directly
(`perfxpert/agents/recommendation.py:121-210`).

### Turning exploration on

Three independent conditions must *all* permit it, or the tier is `strict`:

1. Configuration allows it (`agent_creativity` or
   `PERFXPERT_AGENT_CREATIVITY`).
2. The run is live, not air-gapped.
3. The agent itself declares `AgentCapability.ADDITIVE_EXPLORATION`.

`resolve_tier()` is that three-way AND
(`perfxpert/agents/creativity.py:51-80`). The setting is deployment
configuration, not a request parameter, so a calling model has no handle with
which to raise its own permissions. Agent construction refuses the capability
on Root and the middle layer outright
(`perfxpert/agents/framework.py:206-266`).

Compute, Memory, and Latency build their full catalog answer *before*
considering a model, and under `strict` they return right there. Memory is the
clearest example (`perfxpert/agents/memory_specialist.py:83-138`). Diff still
calls a model under `strict` because it genuinely uses one for narrative
prose; `strict` disables its proposal lane, not its writing.

### What the model is allowed to send back

Every agent output is validated against a schema that forbids unknown fields
(`perfxpert/agents/schemas.py:20-23`). Anything unexpected rejects the entire
response, which is why the model cannot smuggle in its own tool-call record.

Proposals need one deliberate exception. The public output type contains
finished proposals, but the model may only send *drafts*. So draft fields are
relaxed during the first validation pass and then parsed strictly as
`ExploratoryProposalDraft`, whose own schema rejects any runtime-owned field
(`perfxpert/agents/framework.py:306-363`).

### Checking the evidence

An `EvidenceManifest` is the allowlist described in rule 2: tools the run
record says were actually called, kernel names from trusted measured input,
and known catalog entries (`perfxpert/agents/creativity.py:294-320`). Every
proposal must cite at least one entry, every reference must match exactly, and
any named target kernel must be among the measured ones.

The manifest proves a source *was available*. It does not prove the model's
prose about that source is accurate — that is what the confidence cap and the
human promotion step are for.

The runtime then computes a content-addressed `pxp-exp-...` identifier from
the specialist, title, hypothesis, mechanism, and target
(`perfxpert/agents/creativity.py:125-198`). At most three proposals survive
per specialist, and duplicates collapse by identifier only — never by title or
confidence, since those are model-controlled and could be used to suppress a
competing proposal.

The identifier deliberately excludes the evidence, so the same idea keeps a
stable identity across runs and can accumulate support. Including per-run
evidence would mint a new identifier every time and break deduplication.

### The promotion path

`perfxpert proposals list|show|promote` reads a saved result. `promote` builds
the entry with `yaml.safe_dump` and then re-parses its own output to confirm
none of the four withheld fields appeared — model prose once escaped its YAML
scalar and injected them. It also refuses to write into the knowledge tree,
through a symlink, or over an existing file
(`perfxpert/cli/proposals_cmd.py:252-397`).

The withheld fields are left as comments rather than plausible placeholders,
because a placeholder like `[1.0, 1.0]` would let an untouched skeleton pass
CI.

## What changed on this branch

Sixteen commits, in three groups.

**Guardrails that previously existed only on paper.** Before adding anything
creative, the branch made Trace-Diff arithmetic model-independent, then made
output validation, token budgets, session turn budgets, and tool allowlists
into real runtime behavior. It merged two drifting prompt-fence systems so the
shared safety preamble reaches every agent, made Analysis classification
deterministic, and added a CPU-only CI lane. A new "safe" lane cannot rest on
controls that are only comments.

**The feature itself.** The contract landed inert first — configuration,
capabilities, draft and final schemas, the evidence manifest, identifiers —
then proposal construction was switched on while catalog techniques,
confidence, and citations were frozen in both modes. Recommendation carries
proposals in a separate field and deduplicates them by identifier only.
Hostile parity tests and the review-and-promotion command came last.

**Fixes found while validating it.**

- *Model-controlled trusted fields.* Diff accepted a model-supplied
  confidence; Correctness accepted an invented alternative technique; Root and
  Analysis accepted model classification. Each let unproven content sit where
  consumers expected a measurement. All are now runtime-owned.
- *Promotion YAML injection.* The entry was assembled by string
  concatenation, so model prose could break out of its field and inject the
  four "measured" keys. Now serialized, re-parsed, and path-guarded.
- *Diff's lane was never wired.* Diff declared the capability and documented
  proposals but never extracted them. Now nested under `kernel_deltas` to stay
  inside its five-field schema cap.
- *Air-gap environment clamp.* An explicit `airgap=False` could override
  `PERFXPERT_AIRGAP=1` and reach a provider. The environment now wins.
- *Strict tier still paid for a model call.* Three specialists called a model
  and discarded the entire response. They now return before the call.
- *Ambiguous tool names.* Two tools whose names collapse to the same spelling
  after SDK sanitization are now rejected at construction, since the evidence
  manifest matches on those names.

## Where the code is currently weaker than the RFC

Worth knowing before you rely on a guarantee the RFC states:

- Expected effects, verification metrics, assumptions, and failure modes
  default to empty lists. The prompts require them; construction does not yet
  enforce that they are non-empty.
- Provenance is only partly populated. The provider is recorded; the model
  identity and the trace, fence, and catalog digests keep empty defaults.
- Proposal prose is not scanned for command text or prompt-boundary markers,
  and free-text fields are not individually length-bounded. The structural
  protection — no schema field can carry a command, and the orchestrator
  contract forbids acting on proposal text
  (`perfxpert/_bundled/opencode_config/AGENTS.md:139-168`) — still holds.
- A provider error during an exploratory run propagates instead of returning
  the catalog answer with an empty lane, even though that answer was already
  computed. A transient rate limit therefore discards a good result.
- `build_session()` still lets an explicit `airgap=False` beat
  `PERFXPERT_AIRGAP=1` (`perfxpert/agents/runtime.py:338`), unlike the
  framework boundary. No provider call escapes, because the framework floor
  holds, but the session records itself as live.

## Reading map

1. `docs/rfcs/0001-bounded-specialist-creativity.md` — the design and threat
   model, including the alternatives that were rejected.
2. `perfxpert/agents/schemas.py` — exactly what a model may draft, and what
   only the runtime may set.
3. `perfxpert/agents/creativity.py` — tier resolution, evidence matching,
   confidence rejection, identifiers, construction, dedupe.
4. `perfxpert/agents/framework.py` — the shared execution boundary: capability
   checks, the air-gap clamp, budgets, tool wrappers, output validation.
5. `perfxpert/agents/memory_specialist.py` — the clearest end-to-end example.
   Then compare Diff for its nested shape.
6. `perfxpert/agents/recommendation.py` — proposals travelling beside
   recommendations and never entering vetted dedupe.
7. `perfxpert/cli/proposals_cmd.py` — saved proposal to deliberately
   incomplete catalog skeleton.
8. `tests/test_integration/test_airgap_parity.py` — the parity invariant,
   asserted against a model trying to break it.
