# ConSan modes

ConSan exposes four modes. SuperCollider is a separate analysis flavor; the
other three share Memory-Ordering Instrumentation (MOI):

```text
ConSan
├── SuperCollider
└── MOI
    ├── Record/Replay
    ├── Sampled
    └── Inline Shadow
```

All four instrument final native AMDGPU code. They share program inventory,
ownership analysis, target profiles, resource planning, placement, final
validation, HSA loading, and bounded report lifecycle. They differ in what the
device records and where the useful diagnostic decision happens.

SuperCollider implements the core delay-and-redundant-observation technique
introduced by Stephenson et al. in
[“SuperCollider: Scalable and Effective Data Race Detection for CUDA”](https://research.nvidia.com/publication/2026-06_supercollider-scalable-and-effective-data-race-detection-cuda)
(PLDI 2026). ConSan's AMD final-ISA rewriting and semantic/reporting contracts
are its own.

## The four data paths

| Mode | Device work | Retained evidence | Host work | Meaning of a positive result |
| --- | --- | --- | --- | --- |
| **SuperCollider** | Preserve an LDS access, delay, repeat/read back, compare. | Sticky mismatch marker. | Collect and report the marker. | A redundant observation changed; this is value-instability evidence, not a happens-before proof. |
| **Record/Replay** | Publish bounded access, barrier, atomic, and fence records. | Bounded event snapshot keyed by exact dispatch/workgroup/owner/site identity. | Replay records through the MOI shadow and ordering model. | The retained records form an attributed conflict under that model. |
| **Sampled** | Select dynamic instances and publish bounded causal windows plus associated synchronization metadata. | Immutable sampled watchpoint banks and ordering state. | Scan retained causal windows. | A selected window exposes an attributed conflict. |
| **Inline Shadow** | Update exact shadow/order state and evaluate supported conflicts immediately. | Shadow/order state and first-N attributed diagnostics. | Collect and summarize device decisions. | The GPU found a supported-form conflict against prior shadow state. |

“Deferred” means report state survives the kernel until the host retires it.
It does not mean that a background GPU analysis runs later. Record/Replay and
Sampled defer their principal analysis to the host. SuperCollider and Inline
Shadow decide their principal signal on the device and defer only collection.

## Semantic differences

| Property | SuperCollider | Record/Replay | Sampled | Inline Shadow |
| --- | --- | --- | --- | --- |
| Core question | Did a repeated/read-back value change? | Do retained events conflict under the MOI order model? | Does a selected causal window expose a conflict? | Does this access conflict with the current shadow now? |
| LDS accesses | Redundant value observation. | Bounded access publication. | Selected watchpoint publication. | Exact admitted four-byte-cell shadow update. |
| Barriers | No detection ordering; may be a mutation/perturbation point. | Record and coalesce supported barrier epochs on host. | Associate qualified barrier metadata with selected windows. | Execute guest barrier, then advance device epoch. |
| Atomics/fences | No causal detection; may be a mutation/perturbation point. | Retain selected release/acquire/address evidence for host replay. | Associate qualified ordering metadata with selected access windows. | Use bounded address-scoped release/acquire state on device. |
| Main strength | Small, complementary instability signal. | Most inspectable modeled history. | Bounded statistical campaigns. | Strongest immediate supported-form attribution. |
| Fundamental limit | Legal interleavings can change values; same-value races can hide. | Bounded snapshot can lose the manifestation. | Sampling can miss the manifestation. | Supported ISA/order forms and diagnostic/order capacity are bounded. |

SuperCollider deliberately does not reconstruct causality. The MOI modes share
normalized owner, epoch, access, barrier, and atomic/fence semantics, but each
owns its report ABI and conflict model. “Exact” always means exact within the
declared supported forms and retained capacities—not proof about excluded ISA
forms or dropped evidence.

## Shared setup before execution

Before an instrumented kernel runs, every mode uses the same broad pipeline:

1. Intercept and identify the native code object.
2. Run waitcheck on the original object.
3. Inventory sites, owners, resources, and synchronization shapes.
4. Apply target-neutral semantic policy for the selected mode.
5. Plan exact automatic evidence storage.
6. Allocate registers/private state and resolve native placement.
7. Validate the complete replacement independently.
8. Load and bind the replacement to its executable and kernel symbols.

This fixed work is why host-side replay and total host overhead are different
concepts. Every mode needs host transformation and report lifecycle; only
Record/Replay and Sampled perform the main conflict analysis there.

## Sampling behavior

The `sampled` mode uses a default runtime stride of 256 and offset zero. All
eligible static sites remain inventoried and represented in the transform.
Runtime selection controls which dynamic instances enter sampled evidence.

When the owner has the required entry-captured identity and scalar resources,
ConSan places a uniform dispatch/workgroup gate before the shared per-access
evidence body. Owners that need private workgroup identity or a compact
scalar-spill layout use an in-body workgroup selector instead. Selected
workgroups then apply the per-cell selector needed to preserve cross-wave
evidence for a chosen LDS cell.

Increasing the runtime stride does not make transformation, static patching,
or report planning proportionally cheaper. Even at execution time, fallback
placement and shared routing can leave fixed per-access costs. Treat stride as
an evidence-selection control, not as a promised overhead divisor.

Record/Replay also uses runtime workgroup selection (default stride 65,536) to
bound its snapshot. Inline Shadow's default stride is one.

## Interpreting diagnostics

- A **SuperCollider** diagnostic says a redundant observation changed. It does
  not name a racing peer or prove missing happens-before.
- A **Record/Replay** diagnostic is reconstructed from records that remained
  visible in the bounded snapshot.
- A **Sampled** diagnostic comes from a retained causal window. A statistical
  miss says only that this execution/offset did not retain a conflicting pair.
- An **Inline Shadow** diagnostic was evaluated on device against prior exact
  shadow state for an admitted cell and ordering form.

Always read the diagnostic together with static coverage, dynamic completeness,
overflow/saturation, report trust, and the program's own correctness result.
A timeout, signal, bad output, or GPU reset is not by itself a ConSan
diagnostic.

## Choosing a mode

The coded default is **Record/Replay**. It is a useful inspectable starting
point, not a claim of exhaustive history.

- Choose **Inline Shadow** when immediate supported-form attribution matters
  more than device cost.
- Choose **Sampled** for repeated statistical campaigns where bounded retained
  state matters; sweep runtime offsets when confidence matters.
- Choose **Record/Replay** when inspecting synchronization history or the host
  ordering model is central.
- Choose **SuperCollider** for complementary value-instability evidence,
  particularly when causal attribution is unnecessary.

Commands and defaults are in [USAGE.md](USAGE.md). Implementation ownership and
extension rules are in [DESIGN.md](DESIGN.md).
