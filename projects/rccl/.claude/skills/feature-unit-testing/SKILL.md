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

## Domain 1: Requirements & Planning

Before writing any test code, create a test plan that defines:

1. **Feature scope** — which internal API surface is being tested, what is explicitly out of scope
2. **Scenario inventory** — happy path, error paths, edge cases, concurrency patterns
3. **Hardware scope** — which NIC model, how many ports, GDR/DMA-buf availability, single-node
   vs cross-host. Discovering that a test requires unavailable hardware after writing it wastes effort.
4. **Coverage tier** — agree on a branch coverage target (Basic/Standard/Thorough/Critical) as
   the merge acceptance criterion. This is the "red" phase in TDD: the bar is set before code exists.

The plan becomes the PR description's Test Plan section — reviewable alongside code.
It is a planning artifact — do not commit it as a file; keep it in the PR body.

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

**Timeout in poll loops (never spin forever):**
```cpp
// Poll loops that wait for async completion MUST have a timeout.
// An infinite loop masks the real bug (e.g. unroutable NIC) as "test hung".
auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(30);
while (!comm && std::chrono::steady_clock::now() < deadline) {
    ncclResult_t r = AcceptConnection(listenComm, &comm);
    if (r != ncclSuccess) break;
}
if (!comm) { /* handle timeout — GTEST_SKIP or fail with diagnostic */ }
```
Track the return value of each poll iteration separately from the NULL-comm check.
A loop that only checks `!comm` cannot distinguish "still in progress" from "API returned
an error 10 iterations ago and will never succeed".

**GTEST_SKIP() synchronization across ranks:**
```cpp
// When one rank detects a condition requiring SKIP (e.g. QP connect failed),
// it MUST broadcast this to all ranks BEFORE calling GTEST_SKIP().
// GTEST_SKIP() does a return — if rank 0 returns, rank 1 hangs on MPI_Recv.
int skipFlag = (connectFailed ? 1 : 0);
MPI_Allreduce(MPI_IN_PLACE, &skipFlag, 1, MPI_INT, MPI_MAX, MPI_COMM_WORLD);
if (skipFlag) {
    GTEST_SKIP() << "QP connect failed on at least one rank";
}
```
Without the Allreduce, a unilateral `GTEST_SKIP()` in one rank leaves the other
rank blocked forever at its next MPI call or poll loop.

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

**Example from net_ib.cc** (single MI300x node, Mellanox CX-7 — numbers will differ on other hardware):

```
Metric          Before stress tests    After 14 stress tests
────────────────────────────────────────────────────────────
Line coverage       ~72%                   ~72%    Δ +0.2 pp
Branch coverage     ~41%                   ~42%    Δ +1.0 pp
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

Agree on a branch coverage tier at planning time (see *Coverage Acceptance Tiers* above).
Branch coverage ceilings are hardware-dependent — establish a **realistic ceiling**
(branches reachable on your hardware) before setting a target.

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

**What moves coverage in net_ib.cc (measured on one cluster — verify on yours):**
- Multi-QP split (`NCCL_IB_SPLIT_DATA_ON_QPS=1`) — split alignment branches, QP distribution
- MR cache stress (many `regMr`/`deregMr` cycles) — binary-search insert/expand/deref
- Shuffled multi-recv (`PostRecv(n=8)` + non-FIFO send order) — FIFO slot scan for r > 0

**Structural ceiling without special infra:** `ncclIbConnect`/`ncclIbAccept`
state machine branches — only reachable when `ncclSocketConnect` returns before socket
ready. On loopback this never happens. Requires cross-host (`srun -N 2`) or `tc qdisc netem`.
The exact count depends on the source version; use BRDA analysis to find these in your build.

---

## Hardware Capability Gating

**Env-var gating:**
```cpp
// Graceful SKIP (not FAIL) when hardware feature absent
if (!getenv("NCCL_IB_GDR_LEVEL") || atoi(getenv("NCCL_IB_GDR_LEVEL")) == 0)
    GTEST_SKIP() << "GDR not enabled on this node";
```

**Network topology gating (routable GID check):**

A NIC with only link-local GIDs (`fe80::`) or all-zero GIDs will pass `ibv_modify_qp`
INIT→RTR without error — RoCE QP setup does not validate routing. But RDMA packets
are silently dropped cross-node: no CQE, no error, just a timeout. This is a common
source of "recv timed out" failures that look like bugs but are hardware topology.

Gate on routable GIDs by reading `/sys/class/infiniband/<dev>/ports/1/gids`:
```cpp
// Check that the NIC has at least one routable GID (IPv4-mapped or global IPv6).
// Skip test if the NIC can set up QPs but cannot actually deliver RDMA traffic.
static bool HasRoutableGid(const char* devName) {
    // Read /sys/class/infiniband/<devName>/ports/1/gids/<index>
    // A GID is routable if it is NOT all-zero and NOT fe80:: (link-local).
    // IPv4-mapped: 0000:0000:0000:0000:0000:ffff:xxxx:xxxx — routable.
    ...
}
if (!HasRoutableGid(props.name))
    GTEST_SKIP() << devName << " has no routable GID (link-local only)";
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

MPI/SLURM flags are **cluster-specific** — interface names, MCA parameters, GPU resource
requests, and HPC-X paths differ across environments. The example below is a template;
adapt to your cluster.

**Reference scripts** (actual working configurations):
- `test/transport/NetIbMPI/configs/` — runner configs per cluster/NIC combination
- Build & test runner scripts used during development (check `~/run_*.sh` on the node)

```bash
# Template — adapt interface, MCA, and path values to your cluster
sbatch --nodes=2 --ntasks-per-node=1 --exclusive --time=00:10:00 \
  --wrap='mpirun -np 2 --host "$NODE1,$NODE2" \
  --mca pml ob1 --mca btl tcp,self --mca btl_tcp_if_include <IFACE> \
  --mca coll_hcoll_enable 0 --mca coll_ucc_enable 0 \
  --bind-to none \
  -x LD_LIBRARY_PATH=<build>:/opt/rocm/lib \
  -x LLVM_PROFILE_FILE=/path/to/%p.profraw \
  <build>/test/rccl-UnitTestsMPI --gtest_filter="NetIbMPITest.X"'
```

- Interface name (`<IFACE>`) varies: `eth0`, `eno8303`, etc. — check `ip link` on the node
- `--gres=gpu:N` is required only for tests that use GPU/HIP; net-ib MPI transport tests
  do not need it — they test the IB verbs layer directly
- `%p` in `LLVM_PROFILE_FILE` → one `.profraw` per MPI rank
- `--mca pml ob1 --mca btl tcp,self` — use TCP for MPI control plane, not IB verbs
- HPC-X paths (`OPAL_PREFIX`, `LD_PRELOAD` for UCX) depend on the installed version

### Background builds and cron monitoring

Builds and long test runs block the conversation when run in the foreground.
Use `sbatch` (not `srun`) so the job runs detached and writes output to an NFS log file.
Then set a `CronCreate` job to poll for completion.

```bash
# 1. Submit build as sbatch, output to NFS log
BUILDLOG=<build_dir>/build_$(date +%Y%m%d_%H%M%S).log
sbatch --nodelist=<NODE> --cpus-per-task=64 \
  --output="$BUILDLOG" \
  --wrap='cd <build_dir> && cmake --build . --target rccl-UnitTestsMPI -- -j64'
# → prints "Submitted batch job <JOBID>"
```

Then create a Claude `CronCreate` tool call (not a shell command) to monitor:
```
CronCreate(
  cron="*/2 * * * *",
  prompt="Check job <JOBID>: run squeue --me. If gone, read last 30 lines of $BUILDLOG
          and report result to user. If still running, tail -3 log for errors and note
          'still building, N min elapsed' silently.",
  recurring=true
)
```

When the job completes, `CronDelete` the monitoring job.

**Always specify `-j64` (or `-- -j64` after `cmake --build`)** — SLURM allocates 1 CPU by
default; without `--cpus-per-task=64` and `-j64`, `$(nproc)` returns 1 and the build
serialises. The `-- -j64` form passes the flag through cmake to the underlying make/ninja.

**Same pattern for test runs** — any `mpirun`/`srun` that takes > 30 s:
```bash
sbatch --nodes=2 --ntasks-per-node=1 --exclusive --time=00:10:00 \
  --output=<logfile> \
  --wrap='mpirun -np 2 ... rccl-UnitTestsMPI --gtest_filter=...'
# Add --gres=gpu:<N> only for GPU-using tests.
```

---

## Common Mistakes

| Mistake | Consequence | Fix |
|---------|-------------|-----|
| Assume branch uncovered without reading BRDA | Write redundant test | Check `count` in lcov first |
| `ASSERT_NE(req, nullptr)` after `isend` | Spurious failures | Use retry loop; NULL is normal |
| `ASSERT_` before `MPI_Barrier` in multi-rank helper | Deadlock on failure | `EXPECT_` + unconditional barrier |
| Forget `--gres=gpu:N` in srun (GPU tests) | GPU tests SKIPPED silently | Include GPU resource request for HIP-using tests; net-ib MPI tests don't need it |
| Mix profraws from different builds | Corrupt coverage data | One build dir per coverage run |
| Wrong binary path in srun/mpirun | "No such file" on remote node | Binary is in `build/test/`, not `build/`; NFS not mounted on all nodes |
| `llvm-profdata` version mismatch | "unsupported profile format" | Use `/opt/rocm-X.Y.Z/lib/llvm/bin/llvm-profdata` matching the compiler |
| Run tests expecting to cover already-covered paths | +0 delta, wasted effort | Check BRDA count field before writing — if >0, path is already covered |
| Count structural/dead branches in projected delta | Overestimate ceiling | Classify before projecting |
| Read line number without source context | Wrong technique chosen | Always read ±10 lines |
| `GTEST_SKIP()` in one rank without MPI sync | Other rank hangs at next barrier/recv | `MPI_Allreduce` a skip flag across all ranks before any `GTEST_SKIP()` |
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

**Coverage numbers below are approximate.** Exact values depend on hardware (NIC model,
number of ports, GDR/DMA-buf support), cluster topology (single-node loopback vs cross-host),
compiler version (affects branch count — C++ exception pseudobranches inflate the denominator),
and which env-var combinations were exercised. Re-measure on your hardware; do not treat
these numbers as universal targets.

**Approximate baseline progression** (net_ib.cc, ~1800 lcov branches, single MI300x node with Mellanox CX-7):

| Phase | Branch (approx) | Line (approx) | What moved the needle |
|-------|:---------------:|:--------------:|----------------------|
| GeneralTests only | ~40% | ~72% | 18 basic functional tests |
| + 14 stress tests | ~42% | ~72% | Resource churn, multi-connection, bidirectional |
| + branch-coverage tests | ~43% | ~73% | Targeted BRDA gaps: size clamping, NULL comm, nreqs>1 |
| + env-var sweeps | ~44–45% | ~74–75% | MERGE_VFS, ECE, SL/TC, DMABUF, RELAXED_ORDERING, etc. |
| + multi-rank (4-rank loopback) | ~44–45% | ~74–75% | FanIn/FanOut/AllToAll — diminishing returns on loopback |

**Total tests:** ~27 in StressTests.cpp (14 stress + 13 branch-coverage) + GeneralTests.

**Note on denominators**: genhtml `--branch-coverage` shows ~1821 branches (exception
pseudobranches excluded); raw `llvm-cov export` shows ~1954. Use one tool consistently
within a comparison — mixing denominators makes deltas meaningless.

**Resolved puzzle**: Always verify BRDA counts against a profdata that actually includes the
relevant test execution. A branch showing `count=0` may just mean the profdata was built
from an incomplete run — not that the branch is uncovered.

### Hardware-dependent coverage ceiling

The coverage ceiling is determined by what hardware and infrastructure are available.
Most uncovered branches fall into categories that require specific capabilities:

| Category | Approx branches | Requirement |
|----------|:--------------:|-------------|
| Connect/Accept state machine mid-transitions | ~30 | Cross-host run or `tc netem` latency injection |
| RoCE/GID selection paths | ~100 | Multiple RoCE port types or IB fabric |
| GDR/DMA-buf paths | ~50 | GDR kernel module + GPU direct RDMA support |
| WC error / fatal-error paths | ~30 | LD_PRELOAD shim to inject bad work completions |
| NCCLCHECK error fallback branches | ~600 | Systematic fault injection at each call site |

**Realistic ceiling without LD_PRELOAD shims or GDR hardware: ~44–45%.** This is not a
quality problem — it reflects that ~55% of branches are error/fault/hardware paths that
cannot be reached without special infrastructure. Establish the ceiling for your specific
hardware before setting a target; otherwise the target is unachievable by design.

**Env-var technique** — run existing tests with specific env vars to cover additional branches
without writing new tests:
- `NCCL_IB_SL=N NCCL_IB_TC=N` — SL/TC configuration branches
- `NCCL_IB_ECE_ENABLE=0` — ECE disabled path in connect loop
- `NCCL_IB_MERGE_VFS=0` — VFS merge disabled paths in init
- `RCCL_FORCE_ENABLE_DMABUF=1` — forceDmaBuf branch in accept
- `sudo tc qdisc add dev lo root netem delay Xms` — SendDevList reentry on loopback
- Each env-var run should be merged with the existing profdata, not treated as standalone.
