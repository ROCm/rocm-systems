# Memory wait diagnostics

Functional execution computes memory results eagerly. A core, per-wave scoreboard
separately tracks whether a register result or LDS access is known complete.
Reading or overwriting an outstanding destination, or conflicting with an unordered
LDS access, prints a `memory-wait` warning with the issuing PC, consuming PC,
register or LDS address, counter, and sufficient wait
threshold. Execution continues with the eager value. The first conflicting access
reports the producer and recovers its readiness to avoid cascading warnings. Each CU
prints at most 16 warnings during its lifetime.

Diagnostics default to `warn`. To disable tracking and messages, add this entry to each
`compute_unit` node's `config` array:

```json
{"key": "memory_wait_diagnostics", "value": "off"}
```

`warn` and `off` are the accepted values. Invalid values reject the configuration. This
is a core simulator feature and requires no plugin.

## XCNT replay-source diagnostics

On gfx1250, the core also warns when a memory instruction's source register is
overwritten before address translation is known complete. Hardware may need that
original value for XNACK replay. This checks scalar address operands and VMEM
address/data operands, including EXEC and dynamically selected VGPR banks. Reading
a protected source again is allowed. `xcnt-wait` warnings have their own limit of
16 per CU and do not consume the ordinary `memory-wait` warning budget.

XCNT tracking and warnings are enabled by the same `memory_wait_diagnostics`
setting as the completion checks above. Setting it to `off` disables both.
The setting controls diagnostics, not hardware XNACK support.

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
checked when they consume a pending result. Register checks use executed physical
accesses, including lane masks, byte masks and high VGPRs. Counter-only producers
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

LDS checks compare executed byte ranges across lanes, including DS dual accesses,
transpose request lanes, direct-to-LDS loads, and gfx1250 async LDS loads/stores.
They diagnose read/write, write/read and write/write conflicts across these paths.
Two gfx1250 async loads can also write overlapping LDS bytes out of order, despite
their in-order completion notifications. Ordinary same-wave DS operations stay
ordered and do not need a memory-order warning. Disjoint ranges, read/read pairs,
inactive lanes, and rejected out-of-range async accesses do not conflict. For cluster
multicast, only the issuing workgroup's selected destination is checked.

Completion dependencies survive ordinary branches and CU scheduling; branches
implicitly drain XCNT. State is reset when a wave slot is freed and is not serialized
in checkpoints. The checker does not validate general store visibility, source
lifetime beyond the qualified XCNT checks above, or communication between waves.
A warning describes a possible missing wait under
delayed completion or replay even though eager execution already has a value. The
feature does not model GPU latency.

## False negatives and false positives

A clean run does not prove that all required waits are present, even within one
wave. The LDS checks above catch dependencies hidden by eager transfers, but tensor
DMA footprints, same-path legacy direct-load write ordering, global/scalar cache
coherence and accesses from other waves are outside their coverage. Not every
store/read pair needs a wait: ordinary same-wave DS accesses are ordered, and
CDNA5 VMEM stores and loads to the same global address stay ordered too.

Warnings can also be false positives. The checker does not model instruction
latency or distance: it cannot recognize a dependency made safe by
architecturally sufficient instruction spacing. Counter backpressure is modeled
only where the target's counter capacity and completion ordering prove readiness;
unqualified completion classes remain conservative. The checker does not simulate
hardware occupancy or select a latency at which memory becomes visible.

These are known limits, not an exhaustive list. Investigate warnings against the
target's ordering rules; `memory_wait_diagnostics=off` suppresses both classes of
warning when needed. Plugins may overlap with these checks and additionally
validate hazards such as memory visibility and communication between waves.

## Cost

The scoreboard retains dependency records and coalesced LDS byte ranges, not memory
payloads or decoded instructions. LDS ranges are built only for instructions that
access LDS. A byte of shadow state per register rejects unrelated accesses before any
thread-local lookup. The 1,280-byte shadow is present in each wave slot, including when
tracking is disabled. Detailed records are allocated on first tracked memory issue and
reused across wave slot activations. Retirement clears affected shadow bytes and
restores overlapping live records. Formatting occurs only when reporting a hazard.

When tracking is disabled, a flag fixed at wavefront construction skips register
indexing and shadow loads. The CU also skips register notifications and their region
loops when diagnostics are off and no execution plugins are attached. Attaching a
plugin preserves its register callbacks even with wait checking disabled.

MMA helpers do not access the scoreboard concurrently. The issuer checks an eligible
MMA's operands before publishing it; other executed register accesses are checked within
the issuing thread's instruction scope. Observer snapshots and memory writeback do not
count as instruction consumers.

The slow register check is deliberately out of line: otherwise the compiler can hoist
TLS lookup ahead of the shadow test. Empty instruction scopes skip TLS installation.
Separate result and replay-source bits share the existing shadow byte. Reads ignore
the replay bit, keeping reads of replay-only sources out of the detailed checker.
Relaxed atomic shadow bytes let MMA helpers probe the filter without racing the issuer;
only the issuer accesses dependency records.
