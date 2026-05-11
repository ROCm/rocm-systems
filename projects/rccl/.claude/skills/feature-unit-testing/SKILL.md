---
name: feature-unit-testing
description: Use when writing, planning, or improving unit tests for low-level transport or systems code — especially when reasoning about branch coverage, test gaps, identifying which uncovered paths are worth pursuing, or deciding when a feature's test suite is ready to merge.
---

# Feature-Centric Unit Testing

## Overview

One test = one transport feature. Tests live at the internal API boundary, not at the
system level (`ncclAllReduce`). This localizes failures immediately: broken test → broken
feature, no stack archaeology. The suite is the acceptance criterion for every PR.

**Central discipline:** measure coverage first, plan second. Verify assumptions against
real data before writing a single test.

---

## Feature Testing Lifecycle

```
Feature spec / integration plan
         │
         ▼
  DOMAIN 1 — Requirements & Planning
  • Write test plan at spec time (before any code)
  • Hardware scope: which cluster, NIC model, GDR, AINIC, QP count
  • Agree on coverage tier as merge acceptance criterion
         │
         ▼
  DOMAIN 2 — Basic Scenarios
  • Happy path, functional correctness, data integrity
  • Whitebox: direct internal API calls, scheduler inspection
  • Typically reaches ~50% line coverage
         │
         ▼
  DOMAIN 3 — Bottleneck & Edge Case Discovery
  • Fault injection, stress, boundary sizes, concurrent connections
  • Parametric sweeps, async data mutation, multidirectional patterns
  • Pushes coverage from 50% toward 70–90%+
         │
         ▼
  ACCEPTANCE — Coverage Measurement
  • Measure → classify gap → add tests → re-measure
  • below target ──► identify gap ──► add tests ──► re-measure
  • at target    ──► merge
```

**TDD analogy:** test plan = "red" phase written before code exists; Domain 2+3 = "green";
coverage measurement = objective acceptance signal.

---

## Coverage Acceptance Tiers

Prefer **branch coverage** as the primary merge criterion. Line coverage is a secondary
signal; function coverage is informational only. See the *Why Branch Coverage* section.

| Tier | Line coverage | Branch coverage | When to target |
|------|:------------:|:---------------:|----------------|
| **Basic** | ≥ 50% | ≥ 35% | New feature, first PR — core path exercised |
| **Standard** | ≥ 70% | ≥ 50% | Feature complete — error paths and main branches covered |
| **Thorough** | ≥ 90% | ≥ 65% | Stable, high-impact — fault injection required |
| **Critical** | ≥ 95% | ≥ 80% | Safety-critical: fatal-error handling, data integrity |

**Cost of moving between tiers:** Standard→Thorough (branch) requires fault injection and
parametric sweeps. Thorough→Critical is expensive — hardware-specific paths and rare races;
calculate the realistic branch ceiling for your cluster before committing to this tier.

---

## Test Plan as Feature Contract

Write the plan **before** implementation. It must answer:
- What scenarios are covered, what are explicitly out of scope
- Which edge cases and error paths are targeted
- Hardware dependencies (RoCE vs IB, GDR, AINIC, QP count, cluster)
- Which coverage tier is the merge bar

The plan becomes the PR description's Test Plan section — reviewable alongside code.
Hardware scope assessment at planning time prevents discovering a test requires
unavailable hardware only after it is written.

---

## Commit Convention

**One commit per test.** Each test gets its own atomic commit so reviewers can
evaluate test intent and scope individually, and bisect targets a single test on regression.

**Title format:** `<subsystem>: add <test name> test` (lowercase, no brackets)

```
net-ib: add test infrastructure (StressTests harness)   ← CMakeLists + base fixture
net-ib: add InvalidRecvCount test                       ← repeated ×N (test name verbatim)
net-ib: update feature-unit-testing skill               ← skill update last
```

**Body (required):** one short paragraph — what the test does, what path it covers, BRDA ref:

```
Call ncclIbIrecv with n=9 (> NCCL_NET_IB_MAX_RECVS=8). Verifies the early-return
branch returns ncclInternalError without crashing (net_ib.cc:2731, BRDA:2731,0,1).
```

**What to NOT commit:**
- Test plans (`TEST_PLAN.md`) — planning artifact, not part of the test suite
- Coverage shell scripts (`run_*.sh`, `merge_coverage.sh`) — operational, not source
- Profraw / profdata files — build artifacts

---

## Domain 2: Basic Scenario Patterns

### Whitebox testing — bypass the public API

Call internal APIs directly. No `ncclCommInit` overhead. Example:
```cpp
// Direct internal call — not through ncclAllReduce
IbCastIsend(sendComm, buf, size, tag, mh, nullptr, &req);

// Read scheduler state directly to verify internal invariants
ncclIbCastGetSchedState(sendComm, &state);
EXPECT_GT(state.activeQpTokens[0], 0);
```

Whitebox lets a test assert that WRR correctly redistributed tokens when one link is
slow — something a blackbox allreduce test cannot verify.

### Structural patterns (MPI transport tests)

**Double-barrier around every send/recv iteration:**
```cpp
// rank 0: post recv  |  rank 1: post send
MPI_Barrier(MPI_COMM_WORLD);   // after both sides post
// both: wait for completion
MPI_Barrier(MPI_COMM_WORLD);   // after both sides complete
```
Without the second barrier, one rank reuses request slots before the other finishes.

**Retry loop for NULL send request (FIFO backpressure is normal):**
```cpp
do {
    ASSERT_EQ(net_->isend(sendComm, data, size, tag, mh, nullptr, &req), ncclSuccess);
    if (req) break;
    usleep(10000);
} while (true);
// Never ASSERT_NE(req, nullptr) immediately after isend.
```

**RAII guard declaration order:**
```cpp
NetConnectionGuard connGuard(net_);          // 1st — connection
auto bufGuard = makeHostBufferAutoGuard(…);  // 2nd — buffers
NetMHandleGuard mhGuard(mh, …);             // 3rd — memory registration
// Destruction runs in reverse: MR deregistered before connection closed.
```

**Non-fatal checks in multi-rank helpers** (fan-in, fan-out, all-to-all):
```cpp
// Use EXPECT_ (non-fatal), not ASSERT_ (fatal), before any MPI_Barrier.
// A fatal assert in rank 0 leaves all other ranks hanging at the barrier.
EXPECT_EQ(PostRecv(…), ncclSuccess);
// …
MPI_Barrier(MPI_COMM_WORLD);  // unconditional — always reached
```

---

## Domain 3: Fault Injection

**Deliberately break one component — verify the system response.**

Compile-guarded (`-DENABLE_FAULT_INJECTION=ON`), zero cost in production:
```cpp
ncclIbCastFaultSetQpDelay(comm, qpIdx, delayUs);   // artificial latency on one QP
ncclIbCastFaultSetQpError(comm, qpIdx, true);       // force ncclSystemError
ncclIbCastFaultClear(comm);                         // reset for recovery test
```

**Test categories fault injection enables:**
- Fatal event propagation — all QPs faulted simultaneously
- Single link failure — one QP faulted, WRR steered away deterministically
- Scheduler rebalancing — artificial delay → WRR redistributes tokens
- Data integrity under delay — sends complete correctly despite latency
- Recovery — fault → clear → fresh connection completes cleanly

**When fault injection is the only option:** `wc.status != IBV_WC_SUCCESS` paths in
`ncclIbTest`, `fatalErrorCount` threshold branches, async error processing — all show
zero hits in coverage until a fault injection hook exists.

**Without a compile-time hook:** use LD_PRELOAD to intercept at the dynamic linker level:
```c
// libibverbs_mock.so — intercepts ibv_poll_cq, injects bad WC after N calls
int ibv_poll_cq(struct ibv_cq *cq, int num, struct ibv_wc *wc) {
    static int calls = 0;
    if (++calls == INJECT_AT) { wc->status = IBV_WC_REM_ACCESS_ERR; return 1; }
    return real_ibv_poll_cq(cq, num, wc);
}
```

---

## Domain 3: Stress Testing

**Resource leak detection:**
```
100 × (listen → connect → transfer → close)
Assert QP / PD / CQ / MR counts return to baseline after every cycle.
```

**Transfer size coverage:**
- Minimum: 1-byte messages — exposes framing bugs invisible at larger sizes
- Non-powers-of-two: 3, 7, 1023, 4097, 65537 — alignment and boundary conditions
- Maximum: up to 64 MB — exercises large MR registration and buffer management

**Multidirectional patterns:** allgather, alltoall, hypercube topologies; bidirectional
simultaneous send/recv on the same connection; many concurrent connections open
simultaneously (stress QP allocation limits).

**Async data mutation:** mutate the send buffer *after* posting the send. Verifies the
NIC captured data before mutation — detects use-after-post bugs in buffer registration.

---

## Domain 3: Parametric Configuration Matrix

One test body swept over a grid of env-var combinations:
```
WRR on/off × QPS count × split threshold × adaptive routing threshold
```
Surfaces interactions between features that per-feature tests miss. Implement with
`GTEST_SKIP()` when a required env var is absent:
```cpp
const char* v = getenv("NCCL_IB_QPS_PER_CONNECTION");
if (!v) GTEST_SKIP() << "Set NCCL_IB_QPS_PER_CONNECTION to run this test";
```

---

## Why Branch Coverage, Not Lines or Functions

**Line coverage lies. Branch coverage tells the truth.**

A line is "covered" if it executed at least once — regardless of which path through it was taken.
A function is "covered" if it was called — regardless of what happened inside.
A branch is covered only when **both sides** of a conditional have been exercised.

```
// Line coverage: 100% (line executes in every test)
// Branch coverage: 50% (error path never taken)
if (result != ncclSuccess) goto fail;   // ← "fail" branch: count = 0
```

**Real numbers from net_ib.cc** (same binary, same test run):

```
Metric          Before stress tests    After 14 stress tests
────────────────────────────────────────────────────────────
Line coverage       71.7%                  71.9%   Δ +0.2 pp
Branch coverage     40.6%                  41.6%   Δ +1.0 pp
```

Line coverage looks healthy at 71%. Branch coverage at 40% reveals that roughly
**6 in 10 decision points** are only partially tested. For transport code where the
uncovered side is usually the error path or the hardware-specific path, this gap
directly maps to bugs that tests cannot catch.

**Functions coverage is the least informative metric for systems code.** A function
called once with the happy-path arguments appears fully covered even if it contains
20 conditional branches that were never exercised.

### Branch coverage in practice

**Use `--branch-coverage` in genhtml:**
```bash
genhtml --branch-coverage merged.lcov.info -o html/
# Without this flag, the HTML report does not show per-branch hit counts.
```

**lcov vs llvm-cov branch counts differ:**
- lcov (BRDA parsing): 1821 branches for net_ib.cc
- `llvm-cov report`: 1954 branches (includes C++ exception pseudo-branches)

Use one tool consistently within a comparison. Mixing denominators makes deltas
meaningless. lcov/genhtml is preferred for branch-level HTML drill-down.

**Read BRDA lines, not DA lines, when hunting gaps:**
```
DA:1234,5        ← line 1234 executed 5 times (line coverage)
BRDA:1234,0,0,5  ← block 0, branch 0: taken 5 times (covered)
BRDA:1234,0,1,0  ← block 0, branch 1: taken 0 times ← THIS IS THE GAP
```
A line with `DA` count > 0 but a `BRDA` count of 0 on one arm is the exact pattern
of "line covered, branch not covered" — the most common source of false confidence.

**Agreement on branch coverage tier at planning time** (not line coverage):

| Tier | Branch coverage | Notes |
|------|:--------------:|-------|
| Basic | ≥ 35% | Minimal — only happy path exercised |
| Standard | ≥ 50% | Main error paths and env-var configs covered |
| Thorough | ≥ 65% | Fault injection required to proceed further |
| Critical | ≥ 80% | Reserved for safety-critical subsystems |

Branch coverage ceilings are lower than line ceilings. ~155 branches in
`ncclIbConnect`/`ncclIbAccept` state machine require cross-host runs; ~25 require GDR
hardware. Establish a **realistic ceiling** (branches reachable on your hardware) before
setting a target — otherwise the target is unachievable by design.

---

## Coverage Analysis Workflow

> **STOP — mandatory before writing any test to cover a gap:**
> 1. Export BRDA data and find the exact branch at the target line.
> 2. Confirm the count field is `0`. If it is `> 0`, the branch is already covered — do not write a test.
> 3. Read ±10 lines of source context to classify the branch (Trivial / Structural / Hardware / Fault injection / Dead).
> 4. Only then write or run a test.
>
> Skipping this check is the single most common cause of zero-delta test additions.

### 1. Get the raw BRDA data

```bash
llvm-profdata merge -sparse *.profraw -o merged.profdata
llvm-cov export -format=lcov -instr-profile=merged.profdata ./binary \
    -sources src/your_file.cc > merged.lcov.info
genhtml --branch-coverage merged.lcov.info -o html/

# Extract uncovered branches (count == 0)
awk -F: '/^SF:.*your_file/{found=1} found && /^BRDA:/ && $NF==0 \
    {print} /^end_of_record/{found=0}' merged.lcov.info | sort -t: -k2 -n
```

### 2. Read source context around every uncovered line

Read ±10 lines. A branch at a given line may be a trivial env-var check or a
hardware-only fault path — the line number tells you nothing without context.

### 3. Verify existing tests before claiming a gap

**Anti-pattern:** "All tests call `PostRecv(n=1)`, so the `nreqs > 1` path is uncovered."

**What actually happened:** `MultiRecv` and `MultiRecvShuffled` already called
`PostRecv(n=8)` with shuffled tag order, covering the FIFO slot scan for r > 0.
The BRDA count showed > 0. The assumption was wrong.

**Rule:** Before adding a test to cover branch X, check the BRDA count for that exact
line in the merged profile. If count > 0, it is already covered.

---

## Branch Classification

Classify every uncovered branch before deciding whether to test it:

| Class | Description | Technique |
|-------|-------------|-----------|
| **Trivial** | One env var or API param change | Set `NCCL_IB_X=1`, call with `n=9` |
| **Structural** | State the harness can't easily create | Cross-host run, `tc qdisc netem delay` |
| **Hardware** | Needs specific HW (GDR, >1 NIC, non-loopback) | Different node, or `GTEST_SKIP()` |
| **Fault injection** | Needs a mock/shim returning errors | LD_PRELOAD, compile-time hook |
| **Dead/unreachable** | Kernel or driver guarantees make it impossible | Document and skip |

**Ceiling calculation:** count only Trivial + Structural (with infra) + Fault injection.
Never include Dead branches in projected delta.

**What moves coverage in net_ib.cc (measured):**
- Multi-QP split (`NCCL_IB_SPLIT_DATA_ON_QPS=1`) — split alignment branches, QP distribution
- MR cache stress (many `regMr`/`deregMr` cycles) — binary-search insert/expand/deref
- Shuffled multi-recv (`PostRecv(n=8)` + non-FIFO send order) — FIFO slot scan for r > 0

**Structural ceiling without special infra (~155 branches):** `ncclIbConnect`/`ncclIbAccept`
state machine — only reachable when `ncclSocketConnect` returns before socket ready.
On loopback this never happens. Requires cross-host (`srun -N 2`) or `tc qdisc netem`.

---

## Hardware Capability Gating

```cpp
// Graceful SKIP (not FAIL) when hardware feature absent
if (!getenv("NCCL_IB_GDR_LEVEL") || atoi(getenv("NCCL_IB_GDR_LEVEL")) == 0)
    GTEST_SKIP() << "GDR not enabled on this node";
```

Makes the same test suite portable across clusters. Hardware-specific gaps are
**documented**, not faked. A SKIP that explains its reason is useful; a disabled test
with no explanation is technical debt.

---

## AI Assistance at Each Phase

| Phase | AI role |
|-------|---------|
| Requirements | Identify corner cases from spec; flag hardware dependencies; draft test plan |
| Domain 2 | Generate test boilerplate, MPI rank structure, buffer helpers |
| Domain 3 | Propose non-obvious parameter combinations; identify race-prone paths |
| Gap analysis | Map uncovered functions to test categories; suggest fault injection targets |
| Iteration | Generate new tests for specific uncovered branches; validate fixes |

**AI proposes, engineer validates** — especially for hardware-specific behaviour and
expected error codes. AI's highest value is at requirements and gap-analysis phases:
"what are we missing?" rather than "how do we write this loop?".

---

## MPI + SLURM Execution (this codebase)

```bash
srun --nodelist=<NODE> --gres=gpu:<N> \
  mpirun -np <N> --oversubscribe \
  --mca plm_rsh_agent srun \
  --mca oob_tcp_if_include eno8303 \
  --mca btl_tcp_if_include eno8303 \
  --mca btl ^openib \
  --bind-to none \
  -x LLVM_PROFILE_FILE=/path/to/%p.profraw \
  -x LD_LIBRARY_PATH=<build>:/opt/rocm/lib \
  <build>/test/rccl-UnitTestsMPI --gtest_filter='NetIbMPITest.X'
```

- `--gres=gpu:N` **required** — without it `/dev/kfd` absent, HIP fails, tests report SKIPPED
- `%p` in `LLVM_PROFILE_FILE` → one `.profraw` per MPI rank
- `--mca plm_rsh_agent srun` replaces SSH for process launch on SLURM nodes
- `--mca btl ^openib` disables legacy OpenMPI IB verbs (use TCP for MPI control plane)

### Background builds and cron monitoring

Builds and long test runs block the conversation when run in the foreground.
Use `sbatch` (not `srun`) so the job runs detached and writes output to an NFS log file.
Then set a `CronCreate` job to poll for completion.

```bash
# 1. Submit build as sbatch, output to NFS log
BUILDLOG=<build_dir>/build_$(date +%Y%m%d_%H%M%S).log
sbatch --nodelist=<NODE> --gres=gpu:1 --cpus-per-task=64 \
  --output="$BUILDLOG" \
  --wrap='cd <build_dir> && cmake --build . --target rccl-UnitTestsMPI -- -j64'
# → prints "Submitted batch job <JOBID>"

# 2. Create cron to monitor (every 2 min)
CronCreate(
  cron="*/2 * * * *",
  prompt="Check job <JOBID>: run squeue --me. If gone, read last 30 lines of $BUILDLOG
          and report result to user. If still running, tail -3 log for errors and note
          'still building, N min elapsed' silently.",
  recurring=true
)

# 3. When done, CronDelete the monitoring job
```

**Always specify `-j64` (or `-- -j64` after `cmake --build`)** — SLURM allocates 1 CPU by
default; without `--cpus-per-task=64` and `-j64`, `$(nproc)` returns 1 and the build
serialises. The `-- -j64` form passes the flag through cmake to the underlying make/ninja.

**Same pattern for test runs** — any `mpirun`/`srun` that takes > 30 s:
```bash
sbatch --nodelist=<NODE> --gres=gpu:<N> --cpus-per-task=8 \
  --output=<logfile> \
  --wrap='mpirun -np <N> ... rccl-UnitTestsMPI --gtest_filter=...'
```

---

## Common Mistakes

| Mistake | Consequence | Fix |
|---------|-------------|-----|
| Assume branch uncovered without reading BRDA | Write redundant test | Check `count` in lcov first |
| `ASSERT_NE(req, nullptr)` after `isend` | Spurious failures | Use retry loop; NULL is normal |
| `ASSERT_` before `MPI_Barrier` in multi-rank helper | Deadlock on failure | `EXPECT_` + unconditional barrier |
| Forget `--gres=gpu:N` in srun | All tests SKIPPED silently | Always include GPU resource request |
| Mix profraws from different builds | Corrupt coverage data | One build dir per coverage run |
| Wrong binary path in srun/mpirun | "No such file" on remote node | Binary is in `build/test/`, not `build/`; NFS not mounted on all nodes |
| `llvm-profdata` version mismatch | "unsupported profile format" | Use `/opt/rocm-X.Y.Z/lib/llvm/bin/llvm-profdata` matching the compiler |
| Run tests expecting to cover already-covered paths | +0 delta, wasted effort | Check BRDA count field before writing — if >0, path is already covered |
| Count structural/dead branches in projected delta | Overestimate ceiling | Classify before projecting |
| Read line number without source context | Wrong technique chosen | Always read ±10 lines |
| Write tests before test plan | No merge criterion | Plan first, including hardware scope |

---

## Key Principles (summary)

| Principle | Applied as |
|-----------|------------|
| Feature-centric | One test = one transport feature |
| Plan first | Written at spec time, part of integration plan |
| TDD analogy | Coverage tier agreed upfront; measure → iterate until met |
| Whitebox | Direct internal API + inspect scheduler state |
| Fault injection | Controlled failures; compile-guarded, zero production cost |
| Stress | Resource leaks, extreme sizes, concurrent, async mutation |
| Coverage-driven | Branch coverage primary; tiered acceptance; realistic ceiling per cluster |
| Portable | Hardware gaps documented, graceful SKIP |
| AI-assisted | Agents at each phase; engineer validates |

---

## Living Document

Updated after each testing iteration. Add new patterns and anti-patterns here as they emerge.

**Current baseline** (2026-05-07, `build_debug_fault_inject`, net_ib.cc, 1821 lcov branches excluding exception pseudobranches):

| Milestone | Branch | Line | Notes |
|-----------|-------:|-----:|-------|
| Before stress tests | 40.6% (740/1821) | 71.7% | 18 GeneralTests |
| After 14 stress tests | 41.6% (757/1821) | 71.9% | +17 branches |
| After MultiRecv (standalone) | ~42.5% (774/1821) | — | nreqs>1 path covered: BRDA:2387,0,1,255 |
| After SendSizeClamping (E2) | 42.5% (773+1=774/1821) | — | BRDA:2544,0,0 — size clamping |
| After NullCommClose (E3) | 42.6% (776/1821) | 72.7% | BRDA:3064,3084,3114 — NULL comm guards |
| After SetNetAttrNoOp (E5) | 42.6% (776/1821) | 72.9% | FNDA:2 ncclIbSetNetAttr — 0 branch delta |
| After NCCL_IB_SL=2 + NCCL_IB_TC=5 | 42.7% (778/1821) | — | BRDA:1716,0,0 + 1717,0,0 |
| After tc netem delay 200ms (loopback) | 42.8% (780/1821) | — | BRDA:1566,0,0 (SendDevList reentry) |
| After NCCL_IB_ECE_ENABLE=0 | 42.9% (781/1821) | — | BRDA:1652,0,1 — ECE disabled path |
| After NCCL_IB_MERGE_VFS=0 | 43.0% (783/1821) | — | BRDA:492,0,1 + 512,0,1 — VFS merge disabled |
| After RCCL_FORCE_ENABLE_DMABUF=1 | 43.8% (797/1821) | 74.1% | misc init branches |
| After 4-rank loopback (k13-41 single node) | 43.88% (799/1821) | 74.4% | FanIn/FanOut/AllToAll/Bidirectional/LongRunning |
| After env-var batch 2 (RELAXED_ORDERING, INLINE, TIMEOUT, FIFO_TC, AR=0, HCA, DISABLE) | **44.54% (811/1821)** | **74.5%** | +14 branches, +8 lines |

**Total tests in StressTests.cpp: 27** (14 stress + 13 branch-coverage)

**Reference files on k13-41:** `~/coverage/dmabuf/with_dmabuf_netib.lcov.info` — current best.

**Note on denominators**: genhtml `--branch-coverage` shows 1821 (exception pseudobranches excluded); raw lcov export has 1954. Use consistent tool.

**Resolved puzzle** (documented 2026-05-07): `BRDA:2387,0,1` (nreqs>1 in `ncclIbMultiSend`) showed 0 in `full_combined.profdata` because that profdata was built from runs that didn't include a proper `MultiRecv` execution. Running `MultiRecv` standalone showed `BRDA:2387,0,1,255`. The branch IS covered — the old profdata was incomplete.

**Coverage ceiling analysis** (1821 total branches, 797 covered):
- ~30 uncovered: State machine mid-transitions (`ncclIbConnect` L1567,1569,1571; `ncclIbAccept` L1881,1883,1885) — require real cross-host run or `tc netem` on recv path
- ~100 uncovered: RoCE/GID selection paths (L408-479) — require RoCE hardware  
- ~50 uncovered: GDR/DMA-buf paths (L2035-2177) — require GDR kernel module
- ~30 uncovered: WC error / fatal-error paths (L2916,2921,2968) — require LD_PRELOAD shim
- ~600 uncovered: NCCLCHECK error fallback branches — require injected failures
- Realistic ceiling without LD_PRELOAD or cross-host or GDR: **~44-45%**

**Env-var technique summary** — run existing tests with these env vars to cover new branches:
- `NCCL_IB_SL=N NCCL_IB_TC=N` — covers ternary in ncclIbConnect L1715-1716
- `NCCL_IB_ECE_ENABLE=0` — covers ECE disabled path in connect loop
- `NCCL_IB_MERGE_VFS=0` — covers VFS merge disabled paths in init
- `RCCL_FORCE_ENABLE_DMABUF=1` — covers forceDmaBuf branch in accept
- `sudo tc qdisc add dev lo root netem delay Xms` + remove after — covers SendDevList reentry on loopback
- Each env-var run should be merged with the existing profdata, not treated as a standalone baseline.
