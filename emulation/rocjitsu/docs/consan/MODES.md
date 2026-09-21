# ConSan modes

ConSan diagnoses races from sampled memory accesses and bounded causal
evidence. Its alternative **SuperCollider** mode checks value instability.

Both instrument final native AMDGPU code. They share program inventory,
ownership analysis, target profiles, resource planning, placement, final
validation, HSA loading, and bounded report lifecycle.

SuperCollider implements the core delay-and-redundant-observation technique
introduced by Stephenson et al. in
[“SuperCollider: Scalable and Effective Data Race Detection for CUDA”](https://research.nvidia.com/publication/2026-06_supercollider-scalable-and-effective-data-race-detection-cuda)
(PLDI 2026). ConSan's AMD final-ISA rewriting and semantic/reporting contracts
are its own.

## Data paths

| Mode | Device work | Retained evidence | Host work | Meaning of a positive result |
| --- | --- | --- | --- | --- |
| **ConSan** | Select dynamic accesses and publish causal windows with associated synchronization metadata; check supported same-instruction collisions. | Bounded watchpoint banks, ordering metadata, and immediate-conflict evidence. | Analyze retained windows and report conflicts. | Selected evidence exposes an attributed conflict under the ConSan ordering model. |
| **SuperCollider** | Preserve an LDS access, delay, repeat/read back, and compare. | Sticky mismatch marker. | Collect and report the marker. | A redundant observation changed; this is value-instability evidence, not a happens-before proof. |

Report state survives kernel execution until the host retires it. ConSan
performs causal-window analysis on the host. SuperCollider evaluates its
mismatch signal on the device and defers collection to the host.

## Semantic differences

| Property | ConSan | SuperCollider |
| --- | --- | --- |
| Core question | Does retained causal evidence expose a conflict? | Did a repeated/read-back value change? |
| LDS accesses | Selected watchpoint publication and supported intra-instruction collision checks. | Redundant value observation. |
| Barriers | Associate qualified barrier epochs with selected windows. | No detection ordering; may be a mutation/perturbation point. |
| Atomics/fences | Associate qualified release/acquire ordering with selected access windows. | No causal detection; may be a mutation/perturbation point. |
| Main strength | Attributed conflicts with adjustable sampling overhead. | A complementary instability signal. |
| Fundamental limit | Sampling, bounded retention, and unsupported forms can hide conflicts. | Legal interleavings can change values; same-value races can hide. |

ConSan retains exact workgroup coordinates and normalized owner, epoch,
access, and synchronization facts for its selected evidence. This does not
make the compact launch fingerprint an injective queue-instance/dispatch
identity; see [VALIDATION.md](validation/VALIDATION.md).

## Shared setup before execution

Before an admitted instrumented kernel runs, every mode uses the same broad
pipeline (a bounded kernel allowlist may exclude a code object before step 2):

1. Intercept and identify the native code object.
2. Run waitcheck on the original object.
3. Inventory sites, owners, resources, and synchronization shapes.
4. Apply target-neutral semantic policy for the selected mode.
5. Plan exact automatic evidence storage.
6. Allocate registers/private state and resolve native placement.
7. Validate the complete replacement independently.
8. Load and bind the replacement to its executable and kernel symbols.

Both modes incur host transformation and report-lifecycle costs. These costs
are distinct from ConSan's host conflict analysis.

## Sampling behavior

The `default` mode uses a default runtime stride of 256 and offset zero. All
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

Use `RJ_CONSAN_PRESET=higher` for small reproductions and `max`
to remove workgroup and cell sampling. Bounded retention still applies at
`max`. See [USAGE.md](USAGE.md#presets) for preset values and overrides.

## Interpreting diagnostics

A **ConSan** diagnostic identifies a conflict in selected evidence. A clean
report does not prove race freedom: the execution may not manifest the race,
or sampling and retention may omit a conflicting pair.

A **SuperCollider** diagnostic says a redundant observation changed. It does
not name a racing peer or prove missing happens-before.

Read diagnostics together with static coverage, dynamic completeness,
overflow/saturation, report trust, and the program's own correctness result.
A timeout, signal, bad output, or GPU reset is not by itself a ConSan diagnostic.

## Choosing a mode

Use **ConSan** for causal diagnostics and repeated statistical campaigns;
adjust its preset or selection offsets to trade recording cost for coverage.
Use **SuperCollider** for complementary value-instability evidence.

Commands and defaults are in [USAGE.md](USAGE.md). Implementation ownership and
extension rules are in [DESIGN.md](DESIGN.md).
