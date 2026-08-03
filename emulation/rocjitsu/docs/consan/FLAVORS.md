# ConSan flavors and engines

ConSan exposes the SuperCollider flavor and three engines under the MOI flavor.
They are not four independent implementations at the same abstraction level:

```text
ConSan
├── SuperCollider flavor
└── MOI flavor
    ├── Record/Replay engine
    ├── Sampled engine
    └── Inline Shadow engine
```

SuperCollider asks whether a repeated observation of an LDS access changed. It
implements the core delay-and-redundant-observation technique introduced by
Stephenson et al. in the NVIDIA Research paper
[“SuperCollider: Scalable and Effective Data Race Detection for
CUDA”](https://research.nvidia.com/publication/2026-06_supercollider-scalable-and-effective-data-race-detection-cuda)
(PLDI 2026). ConSan's AMD final-ISA rewriting, reporting, and current semantic
contract are its own; it does not inherit every implementation detail or claim
from the CUDA system.

MOI stands for **Memory-Ordering Instrumentation**. Its three engines share a
structured model of accesses, owners, epochs, barriers, and selected atomic
ordering events. They differ mainly in **when** they decide that two accesses
conflict and **how much** execution evidence they retain.

The SuperCollider flavor and all three MOI engines instrument final native GPU
code. In an ordinary run, the HSA tool automatically inventories every
admitted supported site, allocates registers and reports, patches the code
object, and loads the replacement. The user does not select individual sites
or registers.

## One execution, four data paths

In this document, **deferred evidence** means device-resident report state that
survives the instrumented kernel and is consumed later by the host. **Deferred
analysis** means analysis that the host performs when consuming that state; it
does not mean a background device task. Today, automatic report buffers are
consumed by the HSA tool during report teardown.

| Flavor or engine | Inside the device kernel | Deferred device-resident evidence | Host consumption or deferred analysis |
| --- | --- | --- | --- |
| **SuperCollider** | Access, delay, repeat or read back, and compare values. | Sticky mismatch marker. | Collect the marker and report value instability; do not reconstruct a race. |
| **MOI Record/Replay** | Publish bounded access and synchronization records. | Bounded event snapshot. | Replay the snapshot through the MOI shadow and ordering model. |
| **MOI Sampled** | Select dynamic instances and publish bounded causal windows. | Bounded sampled windows. | Scan the retained windows and report sampled conflicts, statistical misses, saturation, or lost evidence. |
| **MOI Inline Shadow** | Update exact shadow state and evaluate the supported-form conflict predicate immediately. | Bounded diagnostics for conflicts already decided on the device. | Collect and summarize the attributed diagnostics. |

Each row is one complete execution path. Shared host setup before execution is
described below rather than repeated in every row.

## Side-by-side comparison

| Phase or property | SuperCollider | MOI Record/Replay | MOI Sampled | MOI Inline Shadow |
| --- | --- | --- | --- | --- |
| Core question | Did a repeated/read-back value change after a delay? | Do the retained events form a conflict under the MOI model? | Does the selected subset of dynamic events expose a conflict? | Does this access conflict with the prior shadow state right now? |
| Inside the device kernel | Preserve the original access, delay, repeat a load or read back a store, compare values, and set a mismatch marker. | Publish bounded access records and supported barrier, atomic, and fence evidence. It does not ordinarily decide access conflicts in the kernel. | Apply deterministic runtime selection and publish compact watchpoints plus synchronization metadata for selected causal windows. | Atomically publish/read exact shadow cells, update owner/epoch/order state, evaluate the conflict predicate, and emit a bounded attributed diagnostic. |
| Deferred work | No conflict reconstruction. Only a sticky result waits to be collected. | The main race analysis is deferred: bounded device records wait for host replay. | The ordinary conflict scan is deferred. Only selected windows exist to be scanned. An expert-only immediate adjacent-range GPU check is available but is not part of the ordinary engine behavior. | No main conflict analysis is deferred. Diagnostics wait for collection, but the device already made the supported-form conflict decision. |
| Host after execution | Read the automatic marker and report measured value instability. | Replay visible records through exact-shadow and supported ordering semantics; emit attributed replay conflicts or an honest bounded-snapshot miss. | Scan retained, generation-qualified windows; emit sampled conflicts, saturation/loss information, or an honest statistical miss. | Read device-produced diagnostic records and summarize attribution, capacity, and completeness. |
| Where the useful decision happens | Device value comparison. This is not a happens-before decision. | Host replay. | Host scan by default; optional narrow GPU signal for experiments. | Device conflict check. |
| Synchronization model | None for detection. Barrier/atomic **fault injection** can compose with the value perturbation, but SuperCollider does not reconstruct ordering. | Supported barriers and selected atomics/fences are recorded and modeled during replay. | Supported synchronization metadata is associated with the same sampled causal identity and interpreted during the deferred scan. | Supported barriers advance device epoch state; selected release/acquire atomics use bounded device ordering tables. |
| Retained evidence | One sticky mismatch signal per relevant code object in ordinary automatic mode. | A bounded snapshot of static site slots and synchronization records; optional expert per-lane dynamic append is also bounded. | Bounded immutable watchpoint banks and paired synchronization metadata for selected windows. | Exact shadow slots for admitted LDS cells, bounded ordering state, and first-N structured diagnostics. |
| Main strength | Smallest and easiest perturbation/value-instability signal. | Clearest reference/debug semantics and easiest engine for inspecting the modeled event history. | Broad statistical coverage with bounded retained state and automatic sampling settings. | Strongest supported-form attribution without retaining an exhaustive event trace. |
| Fundamental limitation | A mismatch is not proof of a race, and same-value races can be invisible. | Each automatic static site retains first-light evidence in a bounded 4 dispatch × 4 wave-owner grid, not an exhaustive trace. Dispatch collision or an owner beyond the grid is explicit dynamic-incomplete saturation; caller-owned size-derived buffers retain one bank, while still using the exact entry-captured x/y/z identity. | Sampling can miss races; a clean run is intentionally inconclusive. | Device shadowing and ordering cost more, diagnostics are bounded, and unsupported instruction/order forms remain outside the claim. |

## What is common, and what is not

### Shared host work before execution

Before any instrumented kernel runs, the flavor and all three engines use the
same broad pipeline:

1. The HSA tools hook intercepts a final native code object.
2. ConSan decodes and inventories sites, kernel ownership, resources, and
   synchronization shapes relevant to the selected flavor and engine.
3. The hook plans exact automatic report storage under fixed safety ceilings.
4. Register allocation selects dead or fresh registers and uses the reusable
   spilling backend when required.
5. Patching is transactional: unsupported placement or validation returns no
   partially transformed image.
6. The runtime loads the validated replacement code object.

This common setup explains why “host-side” does not mean “host replay.” Host
replay is specific to Record/Replay and, in a lower-fidelity sampled form, to
Sampled. SuperCollider and Inline Shadow still need substantial host setup and
report collection even though their useful decision occurs on the GPU.

### Current Sampled performance limitation (TODO)

Sampled's runtime selector is active and honors
`RJ_CONSAN_MOI_RUNTIME_SAMPLE_STRIDE` (whose ordinary default is 16,384), but
the current implementation does not reduce execution overhead in proportion
to that stride. Every eligible static site is still analyzed and patched, and
every dynamic LDS access still enters the shared dispatcher, resolves its
static site, enters the site's instrumentation cave, and computes the runtime
sampling predicate. Only the subsequent watchpoint/evidence body is skipped
for an unselected workgroup. Code-object transformation, report planning, and
report allocation are likewise paid independently of the runtime sample rate.

A controlled gfx1250 experiment showed essentially unchanged test time as the
runtime stride increased from 1 to 16,384, even though retained evidence fell
from nine visible events to zero. Record/Replay took approximately the same
time. Thus this is not an accidental 100% sampling configuration: selection
works, but fixed per-site and per-access costs currently dominate. Whole-process
overhead measurements amplify the setup portion of this problem for short
workloads.

The gfx1250 fallback selector also currently combines workgroup coordinates
with only the low bits of a code-object-reader identity, rather than a
launch-varying dispatch identity. Repeated small dispatches can therefore keep
selecting the same workgroup or select no workgroup at all instead of rotating
coverage between launches.

The intended follow-up is to compute a launch-varying sampling decision once
per workgroup, retain a uniform selected/not-selected predicate, and test that
predicate before the shared per-access dispatch path. Until then, Sampled
should be chosen for its bounded retained evidence and statistical coverage,
not on the assumption that a large runtime stride guarantees correspondingly
lower execution overhead.

### Different meanings of a diagnostic

- **SuperCollider** reports a changed redundant observation. It deliberately
  does not name a racing peer or prove a missing happens-before edge.
- **Record/Replay** reports a conflict reconstructed from the bounded records
  that remained visible to the host model.
- **Sampled** reports a conflict found in retained sampled windows. A miss says
  only that this run did not retain a conflicting pair.
- **Inline Shadow** reports a conflict evaluated on the device against the
  previous exact shadow entry for a supported cell and ordering form.

“Exact” in Inline Shadow and the Record/Replay model means exact within the
declared supported forms and retained capacities. It is not a claim that an
unsupported site or dropped diagnostic was race-free. Coverage, saturation,
overflow, and dynamic-completeness reports describe whether the analysis ran as
intended.

## Choosing a flavor and engine

Loading the hook currently selects **MOI Record/Replay** by default. It remains
the inspectable expert engine for synchronization history. On gfx1201, however,
the [empirical study](GFX1201_EMPIRICAL_STUDY.md) recommends **MOI Sampled** for
ordinary barrier/LDS triage and Record/Replay for expert atomic order/scope
investigations. Other targets require their own empirical qualification.

Record/Replay's inspectable role is not a completeness claim. It retains a
bounded first-light snapshot: each static slot is claimed once for a loaded
code object, and generation filtering compares only records from the same
hardware dispatch. Consequently, complete static patch coverage does not imply
complete dynamic-history retention or fault-detection sensitivity; a clean
replay remains inconclusive.

- Choose **MOI Inline Shadow** when the strongest current supported-form,
  immediate device-side attribution matters more than its higher device work.
- Choose **MOI Sampled** for broad statistical runs where lower retained
  state matters more than detecting every individual manifestation. Sweep
  multiple runtime offsets when confidence matters.
- Choose the **SuperCollider flavor** for its complementary
  perturbation/value-instability signal, especially when structured causality
  is unnecessary or MOI's bounded evidence misses the manifestation.

Commands and ordinary no-tuning defaults are in [USAGE.md](USAGE.md), while
[DESIGN.md](DESIGN.md) defines the detailed
algorithms, report layouts, instruction coverage, and semantic boundaries.

The pseudocode in the gfx1201 empirical study is the canonical audience-facing
explainer for the four probe shapes. This file remains the detailed phase and
data-path comparison; future target studies should link to those explanations
rather than copy them.
