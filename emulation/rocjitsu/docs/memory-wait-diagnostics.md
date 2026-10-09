# Memory wait diagnostics

Memory-result diagnostics are enabled by default. Functional execution computes
memory results eagerly; a core, per-wave scoreboard separately tracks whether a
register result is known complete.
Reading or overwriting an outstanding destination prints a `memory-wait` warning
with the issuing PC, consuming PC, register, counter, and sufficient wait
threshold. Execution continues with the eager value. The first conflicting access
reports the producer and recovers its readiness to avoid cascading warnings. Each CU
prints at most 16 warnings during its lifetime, then one suppression notice.
This budget is per CU, so N active CUs can print up to 17 × N lines for the
same kernel. The budget stays local so a faulty kernel on one CU does not hide
diagnostics from another CU or device. Repeated sites can exhaust this budget
and hide distinct later hazards; the printed set is not exhaustive. Per-CU
diagnostic counts continue after printing is suppressed. Fix reported waits and
rerun to expose later sites. For an unfiltered investigation, a host debugger
breakpoint at `ComputeUnitCore::report_memory_wait` observes every reported
hazard before the output limit is applied.

Registers in diagnostics use the scoreboard's physical numbering. On targets
with an accumulator bank, `v256+n` names `aN` (for example, `v288` is `a32`).

If warnings appear incorrect or checking slows a workload, disable both ordinary
and XCNT checks for a launch without editing the config or rebuilding:

```sh
rocjitsu --config /path/to/config.json --wait-checking=off -- <command>
```

With Mirage, use `mirage run --wait-checking=off -- <command>`. The flag works
with profiles or `--config` and composes with `--cpu-thread-budget`.

Use `--wait-checking=on` for ordinary checks only, or `--wait-checking=all` to include
gfx1250 XCNT. These modes override both per-CU settings on every GPU. Omitting the
flag preserves JSON settings and defaults. See [configuration](configuration.md#memory-wait-diagnostics)
for persistent global overrides and launch-mode restrictions.

Without a global override, memory-result diagnostics default to
`memory_wait_diagnostics=warn`. To disable them per CU, add this entry to each
`compute_unit` node's `config` array:

```json
{"key": "memory_wait_diagnostics", "value": "off"}
```

`warn` and `off` are the accepted values. Invalid values reject the configuration.
This is a core simulator feature and requires no plugin. XCNT replay-source
diagnostics remain independently disabled by default, as described below.

## XCNT replay-source diagnostics

On gfx1250, the core also warns when a memory instruction's source register is
overwritten before address translation is known complete. Hardware may need that
original value for XNACK replay. This checks scalar address operands and VMEM
address/data operands, including EXEC and dynamically selected VGPR banks. Reading
a protected source again is allowed. `xcnt-wait` warnings have their own limit of
16 per CU and do not consume the ordinary `memory-wait` warning budget.

XCNT tracking and warnings default to `off`. Without a global `wait_checking`
override, enable them independently of memory-result checks by adding this entry
to each `compute_unit` node's `config` array:

```json
{"key": "xcnt_diagnostics", "value": "warn"}
```

The accepted values are `warn` and `off`; invalid values reject the configuration.
This setting has no effect on other architectures. It controls diagnostics, not
hardware XNACK support. A global `wait_checking` override takes precedence over
both per-CU keys. Without one, setting `memory_wait_diagnostics=off` leaves an
explicitly enabled XCNT check active. Use `--wait-checking=off` to disable both.

VMEM coverage is qualified for `MODE.REPLAY_MODE=1` (bit 25), the multi-group mode
selected by LLVM at kernel entry. VMEM replay sources in single-group mode are
not checked; scalar replay sources are checked in either mode. This is an explicit
coverage boundary, not a claim that single-group kernels are replay-safe.

Within the qualified model, X waits retire an ordered VMEM queue prefix. SMEM
translation is unordered and requires a zero wait. A zero KM wait also releases
SMEM sources. VMEM completion waits map completed instructions back to their X
queue positions and release the proven translation prefix, including mixed loads
and stores. Completion counters retain their independent positions.
The checker accounts for implicit drains at branches, register-control operations,
messages and barriers, SMEM/VMEM transitions, and ordered VMEM destination reuse.
It protects source dwords and executed vector lanes, without retaining payloads or
simulating faults. Warnings describe a possible replay failure, even when the eager
execution's numerical results are correct. Atomic replay ordering and memory
visibility are outside this register-source check.

## Coverage

The checker tracks scalar and vector register results from both memory pipelines and
inline producers, including DS permutations and message returns. Store operands are
checked when they consume a pending result. The issuer checks each instruction
before execution, using decoded register numbers and the wave's current EXEC,
register-bank and addressing state. Counter-only producers
contribute to partial waits even without a register result. Implicit SCC consumers and
overwrites (including HWREG aliases), M0 and EXEC message results, wave-sized VCC
accesses, and private address uses of FLAT_SCRATCH are checked too. Preserved scalar
halves and inactive vector lanes are not treated as consumed inputs. EXEC is consumed
even when its eagerly computed value disables all vector lanes.

The shared target model handles CDNA1 through CDNA5 and RDNA1 through RDNA4, including
RDNA3.5. Each ordered completion class retains the suffix requested by a wait.
Scalar memory results require a zero wait. A mixed counter can prove an ordered
result ready only using younger operations in that same completion class. The
[counter coverage document](memory-wait-counter-coverage.md) details producer families, multi-counter operations,
instruction units, and exceptions.

Finite counter capacity also proves completion. For example, admitting a 64th
operation to a six-bit counter forces the oldest of 63 outstanding operations in
one ordered class to complete. Admission is checked before the new instruction
reads its registers. Counter-only operations count, but unordered younger operations
cannot establish FIFO completion of an older result. This does not infer readiness
of scalar-memory, GDS, or legacy CDNA FLAT results from backpressure.

Completion dependencies survive ordinary branches and CU scheduling; branches
implicitly drain XCNT. State is reset when a wave slot is freed and is not serialized
in checkpoints. The checker does not compare memory addresses or check overlapping
LDS accesses, including accesses from lanes of the same wave. It does not validate
store visibility, source lifetime beyond the qualified XCNT checks above, or
communication between waves.
A warning describes a possible missing wait under
delayed completion or replay even though eager execution already has a value. The
feature does not model GPU latency.

## False negatives and false positives

A clean run does not prove that all required waits are present, even within one
wave. Memory effects execute eagerly, so a global or LDS write followed by a read
can appear correct even when a memory-ordering wait is missing. Direct/async LDS
transfers, tensor DMA footprints, cache coherence and accesses from other waves are
outside this register-dependency check. Not every store/read pair needs a wait:
ordinary same-wave DS accesses are ordered, and
CDNA5 VMEM stores and loads to the same global address stay ordered too.

Warnings can also be false positives. The checker does not model instruction
latency or distance: it cannot recognize a dependency made safe by
architecturally sufficient instruction spacing. Counter backpressure is modeled
only where the target's counter capacity and completion ordering prove readiness;
unqualified completion classes remain conservative. The checker does not simulate
hardware occupancy or select a latency at which memory becomes visible.

These are known limits, not an exhaustive list. Investigate warnings against the
target's ordering rules; use `--wait-checking=off` to suppress both warning classes
when needed. Plugins may overlap with these checks and additionally validate
hazards such as memory visibility and communication between waves.

## Access planning and cost

The scoreboard retains register dependencies, not memory payloads or cached access
plans. Ordinary elementwise instructions probe decoded register numbers directly.
A 1,280-byte shadow per wave covers physical VGPRs 0–1023 and scalar/special
registers. Separate result and replay bits share each byte. A negative probe skips
the detailed checker; a possible overlap checks the pending records' lane and byte
masks. The shadow is conservative across overlapping records and counter retirement.

The wave stores VGPR bank selection in one byte. An issued dependency records the
physical register number, so a later bank change cannot redirect that dependency.
VCC is wave state with separate low/high word identities. Unordered scalar memory
results require a zero wait; they cannot be made ready by an unrelated partial wait.

Exceptional instructions use shared addressing and lane-selection helpers before
execution. Relative registers and permutations can require M0 or selector values;
FLAT domain selection requires address values. Those control inputs are checked
before being read for planning. Planning does not fire execution-plugin callbacks.
The checks and producer registration precede execution lane loops and async MMA
publication. Only the issuing thread accesses the scoreboard: register accessors,
MMA workers, observer snapshots and memory writeback do not check it.

Lane masks distinguish divergent consumers and pending results across waterfall
iterations. Qualified sub-dword forms also distinguish consumed and written bytes,
without treating preservation of the other bytes as a read. Other sub-dword forms
can conservatively check a whole register. Encoded memory address/data sources keep
their dependencies even when an invalid descriptor or an out-of-bounds address
suppresses the memory effect.

Detailed records are allocated on the first tracked issue and reused across wave
slot activations. Retirement clears affected shadow bytes and restores overlapping
live records. Diagnostic formatting occurs only on a hazard. With both settings
disabled, the issuer skips the checker; execution-plugin callbacks remain independent.
