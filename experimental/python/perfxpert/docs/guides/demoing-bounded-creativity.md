# Demoing Bounded Specialist Creativity

A 10-minute walkthrough you can give another engineer to show what the
exploratory-proposal feature does and why it is safe. It needs **no API
key, no GPU, and no network** — the demo scripts a hostile model in
place of a real provider, so it produces the same output every time and
runs anywhere.

For what the feature *is*, read
[Bounded specialist creativity](bounded-specialist-creativity.md) first.
This guide is only about showing it.

Cross-links:
- [Agentic mode](agentic-mode.md) — air-gap vs LLM-enabled
- [Agent hierarchy](../architecture/agent-hierarchy.md) — who may explore
- [RFC 0001](../rfcs/0001-bounded-specialist-creativity.md) — the design

## The one-sentence pitch

Open with the tension, because everything else follows from it:

> PerfXpert recommends GPU optimizations that engineers actually act on,
> so its advice comes from a catalog of techniques someone measured. We
> want the model to be able to suggest something *not* in that catalog —
> without it ever being able to pass that suggestion off as measured.

Then the answer: two lanes that never merge. Recommendations stay
catalog-only and deterministic. Model ideas arrive in a separate field,
labelled unproven, bound to evidence from the run, and capped in
confidence.

## Setup

```bash
# SKIP-SAMPLE — demo setup; run from the perfxpert project root
cd experimental/python/perfxpert
python scripts/demo_bounded_creativity.py
```

That is the whole setup. The script monkeypatches the framework's
provider call with a canned response, so "the model" is whatever the
demo tells it to be. It is hostile at every step: it invents techniques,
claims perfect confidence, cites tools it never called, and forges its
own tool-call record.

Run it once yourself before presenting. It takes about a second.

## The walkthrough

The script prints six steps. Here is what to say at each.

### Step 1 — The default configuration ignores the model entirely

The model returns `invented_technique` at 0.99 confidence with a made-up
citation. The output contains the two real catalog techniques, the
deterministic confidence, no citations, and no proposals.

Say: *this is the default, and it is not "we filtered the model out"
— under the default `strict` tier the specialist never calls the model
at all.* That matters for cost and latency, not just safety.

### Step 2 — Turning exploration on

Set `PERFXPERT_AGENT_CREATIVITY=exploratory` and the same hostile model
now yields a proposal. Point at three things in the printed record:

- The vetted lane is **identical to step 1**. Enabling exploration did
  not change one word of the advice.
- The proposal carries `status: exploratory` and a confidence of 0.4
  against a hard ceiling of 0.5. It cannot be mistaken for advice.
- `proposal_id` was assigned by the runtime, not the model. It is a hash
  of the proposal's content, which is what lets the same idea be
  recognized across runs and promoted later.

### Step 3 — Evidence binding (the core of the demo)

Four hostile proposals, four rejections, vetted lane untouched each
time:

| The model tries to | Why it fails |
| --- | --- |
| Cite a tool the run never called | Manifest is built from tools actually invoked |
| Target a kernel nobody measured | Manifest lists only measured kernels |
| Claim 0.95 confidence | Over the ceiling — rejected, not clamped |
| Forge its own `tool_calls` record | Field is runtime-owned; output rejected |

Two points are worth dwelling on.

**The manifest is built from what the SDK observed, not what the model
reports.** The model cannot vouch for itself. This is the difference
between "the model says it called the profiler" and "the runtime
recorded a profiler call."

**Over-confidence is rejected, not clamped.** If it were clamped, a
proposal that tried to cheat would end up looking exactly like an honest
one. Refusing preserves the signal that something went wrong.

On the fourth case, the framework's own rejection appears on stderr
mid-demo:

```
framework: discarding MemoryTechniquesSpecialist output that failed
MemorySpecialistOutput validation: tool_calls Extra inputs are not permitted
```

That log line is the demo. The model tried to write the evidence record
it would later be judged against, and schema validation threw away the
entire response.

### Step 4 — Air-gap parity

The same call is made live and air-gapped. Every deterministic field
matches. The only difference is that air-gap returns zero proposals.

Say: *this is the invariant that makes the whole thing auditable. If a
customer runs PerfXpert on a closed network with no model at all, they
get the same measured findings. The exploratory lane is the single
permitted divergence, and it is additive.*

If someone asks how that is enforced rather than merely intended, this
is a good moment for the test suite (below).

### Steps 5 and 6 — Review and promotion

The script saves a real result to `/tmp/perfxpert_demo_result.json` and
prints three commands. Run them live:

```bash
# SKIP-SAMPLE — requires the demo script to have run first
perfxpert proposals list /tmp/perfxpert_demo_result.json
perfxpert proposals show /tmp/perfxpert_demo_result.json <proposal-id>
perfxpert proposals promote /tmp/perfxpert_demo_result.json <proposal-id> --promoted-by you
```

`list` and `show` are the review surface. Note the framing in the header
— "hypotheses, not recommendations" — and that `show` prints the
evidence the proposal is bound to.

`promote` is the interesting one, and it is where people are usually
surprised. It emits a catalog entry skeleton that **deliberately fails
catalog validation**:

```
This entry is incomplete by design: measured_speedup_range,
source_citation, preconditions, fixture_pair still need real measurements.
```

Say: *promotion cannot be automated, because the four fields it withholds
only exist once a human has actually run the experiment and produced a
before/after fixture pair. The tool scaffolds the entry and then stops.
That is the point at which an unmeasured idea would otherwise quietly
become measured advice.*

Step 6 then shows a proposal whose `mechanism` text is crafted to break
out of its YAML field and add sibling keys claiming a measured speedup.
The output shows the entry's keys — the forged fields are absent, and
the hostile text is inert inside a quoted string. The entry is
serialized rather than concatenated, and a fail-closed check re-parses
the result and refuses to emit if any withheld field reappeared.

## Showing that it is enforced, not just intended

Everything above is a demonstration. If your audience wants the
guarantee, the properties are encoded as tests:

```bash
# SKIP-SAMPLE — run from the perfxpert project root
pytest tests/test_agents/test_specialist_lanes.py \
       tests/test_agents/test_lane_boundary_probes.py \
       tests/test_integration/test_airgap_parity.py \
       tests/test_cli/test_proposals_cmd.py -q
```

99 tests, about a second. Worth naming what is in there: the parity
suite runs specialists under a deliberately hostile model and asserts
the vetted lane is byte-identical to air-gap; the boundary probes assert
the model cannot reach runtime-owned fields; the promotion tests cover
five distinct YAML injection vectors.

A good closing line: *the demo shows the model failing to get through.
The tests are what keep it failing.*

## Questions you should expect

**"What if the model's idea is actually good?"** Then someone runs the
experiment, and `promote` scaffolds the catalog entry. The feature is not
meant to filter good ideas out; it is meant to keep unmeasured ideas
labelled as unmeasured until measurement happens.

**"Why cap confidence at 0.5 instead of clamping to it?"** Clamping
makes a dishonest proposal indistinguishable from an honest one. The cap
exists to bound how much weight the lane can ever carry, and rejection
keeps a violation visible.

**"Why can only Layer 2 specialists explore?"** Layers 0 and 1 route and
rank. If a routing decision could be model-influenced, the deterministic
core would move, and air-gap parity would break. Specialists are the
only place where an additive suggestion cannot change what was measured.

**"Isn't the proposal ID hash missing the evidence?"** Yes, deliberately.
The ID covers the idea, so the same hypothesis is recognizable across
runs and can accumulate support. Including per-run evidence would make
every run produce a new ID and break deduplication.

**"What stops someone shipping with exploration on by default?"** It is
off by default, it requires a live session, and it requires the agent to
declare the capability. All three must independently permit it, and
`PERFXPERT_AIRGAP=1` overrides the config entirely rather than acting as
a default.

## If you have only two minutes

Run the script and show step 3 and step 4. The hostile model failing
four times while the advice never moves, and live output matching
air-gap exactly, are the whole idea.
