# RCCL Branch Naming and Commit Message Conventions

This document describes the branch naming and commit message conventions
observed in the `projects/rccl` portion of the `rocm-systems` monorepo.
It is derived from analysis of live branch names and ~2,000 non-merge
commits that touch `projects/rccl`.

---

## Branch Naming Conventions

### 1. Release / Drop Branches

Release integration branches live under the `rccl/` namespace and use a
`YYYY-MM` date suffix:

```
rccl/drop_<YYYY-MM>
```

Examples:
- `rccl/drop_2025-11`
- `rccl/drop_2026-02`
- `rccl/drop_2026-03`
- `rccl/drop_2026-04`

These represent periodic integration snapshots and are the primary long-lived
named branches for the project (alongside `develop`).

### 2. Personal / Feature Branches — `users/<username>/<topic>`

The canonical pattern for developer work branches is:

```
users/<username>/<topic>
```

The `<topic>` portion is typically a short, hyphen-separated description of the
work. For RCCL-specific changes, contributors often include `rccl` in the topic
to make the scope clear:

```
users/arravikum/rccl-ainic-multinode-tests
users/arozanov/rccl-nic-fusion-threshold
users/ilkosare/rccl-new-net-ib-tests-NIC-fusion-with-perf
users/nikhil-nunna/rccl-comm-traffic-class-test
users/speriasw/rccl-device-api-tests
users/sstamenk/rccl-rdna-targets
users/KawtharShafie/rccl-ll128-support
users/shwetjha/fast_init
users/yiltan/alltoallv-large-msg
users/yiltan/ipc-topology
users/thomas-huber/DAG-P2P-fix2-proxy-budget
```

#### Sub-categorised personal branches

When a branch belongs to a well-defined category (`fix`, `improve`, etc.),
contributors sometimes add an additional path segment:

```
users/nileshnegi/rccl/fix/static-build
users/nileshnegi/rccl/fix/update-cmake-toolchain
users/nileshnegi/rccl/fix/ut-dtypes
users/nileshnegi/rccl/improve-rcclras
users/nileshnegi/rccl/rccl-tests/update-rccl-float8-header
```

#### Ticket-keyed personal branches

Some contributors name their personal branch after the ticket being addressed:

```
users/thomas-huber/AICOMRCCL-637
users/tpatinya/aicomrccl-96-destroy-comm-pending
users/mustafabar/AICOMRCCL-679-mnnvl_initial_support
```

### 3. rccl-tests Branches

Branches that exist solely to work on the rccl-tests sub-component use either
the `rccl-tests/` namespace or include `rccl-tests` in the topic:

```
rccl-tests/ainic-collector
rccl-tests-ainic-collector-local
users/nilenegi/rccl-tests/improve-UT
users/arravikum/rccl-tests-ci-enable
users/jayhawk-commits/install-rccl-tests-test-runner
```

### 4. net-ib Branches

Branches focused on the InfiniBand network transport layer use descriptive
topics without a special prefix:

```
users/ilkosare/rccl-new-net-ib-tests-NIC-fusion-with-perf
users/ilkosare/rccl-new-net-ib-multirecv-tests
users/ilkosare/net-ib-fault-injection-tests
```

### 5. Monorepo Infrastructure Branches

These are maintained by the repository infrastructure tooling:

| Pattern | Purpose |
|---|---|
| `filtered/rccl` | History produced by `git filter-repo` |
| `filtered/rccl-tests` | As above, for rccl-tests |
| `preserved/rccl` | Snapshot preserved before a re-import |
| `preserved/rccl-tests` | As above, for rccl-tests |
| `external/rccl-tests/cpp` | Vendored external dependency |
| `import/develop/<upstream>_rccl/<branch>` | Upstream PR imports |
| `dependabot/pip/projects/rccl/docs/sphinx/<package>-<version>` | Dependabot doc-dep updates |

### 6. Naming Style Notes

| Element | Convention |
|---|---|
| Word separator | Hyphens (`-`) strongly preferred; underscores (`_`) occasionally seen in older branches |
| Case | Generally lowercase; ticket IDs and acronyms (AICOMRCCL, ROCM, RCCL) may be uppercase |
| Length | Aim for concise but self-explanatory topics (3–6 words) |
| Ticket reference | Include the ticket ID when one exists, either as prefix or in the topic slug |

---

## Commit Message Conventions

### Primary Style — Square-Bracket Project Tag

The dominant convention for commits that touch `projects/rccl` is a square-bracket
project tag followed by an imperative description:

```
[RCCL] <Short imperative description>
```

This is by far the most common pattern (~47 of the most recent 200
RCCL-touching commits). The description should:

- Start with an imperative verb (Fix, Add, Enable, Guard, Refactor…)
- Be concise — ideally ≤ 72 characters for the whole subject line
- Not end with a period

**Examples:**
```
[RCCL] Fix incorrect ifdef guard that disabled IPC registration
[RCCL] Guard directBuff usage in waitPeer when no IPC registration
[RCCL] Add log message after P2P preconnect is completed
[RCCL] Strix-Halo tuning
[RCCL] Parallel build cleaned up v1
[RCCL] use 256 threads per block on gfx950
[RCCL] Fix build warnings: format specifiers and unused variables
[RCCL] MPI log assertion helpers, doc renames, and test runner gtest / multi-node behavior
```

#### Sub-feature tags

When the change is confined to a named sub-feature or sub-component, a second
bracketed tag is appended:

```
[RCCL][SYMMETRIC] Initial support for symmetric memory kernels
[RCCL][Tuning] Add built-in CSV tuner for runtime algo/proto/channel selection without rebuilds
[RCCL][Profiler Plugin] Fix kernelCh StopGpuClk and stopTs not recorded for channels > 0
[RCCL][NAVI3X] Fix UT hang
[RCCL][CUMEM] Fix cuMem multi-process runs
[RCCL][CUMEM] Gate cuMem support with Linux Kernel version
[RCCL][Tuner Plugin] Enable tuning of RCCL tuning constants
```

#### rccl-tests tag

Changes that exclusively modify rccl-tests use `[RCCL-TESTS]` instead of
`[RCCL]`:

```
[RCCL-TESTS] Add MPI_Barrier before finalize
```

Combined changes that touch both may use:

```
[RCCL] [RCCL-Tests] Add gfx1250 target
```

### Component-Prefix Style — `net-ib:`

Commits scoped to the InfiniBand network transport use a lowercase component
prefix followed by a colon and a space:

```
net-ib: <description>
```

**Examples:**
```
net-ib: add RapidConnectDisconnect test to NetIbMPITests
net-ib: add MultipleOutstandingSendRecv test to NetIbMPITests
net-ib: add SendRecvDifferentMemoryTypes test to NetIbMPITests
net-ib: add helper to merge NIC devices in tests
net-ib: tests simplification
net-ib: add new tests to test runner configs
```

### Monorepo-Path Prefix Style — `projects/rccl:` / `projects/RCCL:`

Some commits use the repository sub-path as a namespace, making the scope
explicit in a monorepo context:

```
projects/rccl: Add runtime QP tracking with atomic counters in net_ib and net_ib_cast
projects/RCCL: net_ib_rocm/net_ib_cast: fixed CTS-offload corner cases.
projects/RCCL: AINIC: respect autodetection in Connect functions
```

Note: `RCCL` (uppercase) and `rccl` (lowercase) both appear; prefer lowercase
for consistency.

### Ticket-Only Style

When the change is driven entirely by a tracked issue, the ticket ID can stand
alone as the prefix, with or without brackets:

#### With brackets (preferred modern style):
```
[AICOMRCCL-350] Allow for advanced tuning of P2P channels/part mapping via RCCL_P2P_SHIFT_SIZE
[AICOMRCCL-355] Enable threshold-based p2p-batching
[AICOMRCCL-633] - Fixed warnings in tests
[AICOMRCCL-697] Add --enable-mpi-tests and --cmake-options to install.sh
[ROCM-1722] [ROCM-1721] fix memory leaks
[ROCM-21076][RCCL][MI350] Fixing the rccl tuning parameters for large scale AR on MI350
[AICOMRCCL-98] [ROCM-1974] Enable user buffer and graph registration feature
```

#### Without brackets (older style — still seen, but avoid for new commits):
```
AICOMRCCL-654 cmake changes to create patched file in build staging area
AICOMRCCL-656 fix memory leak in ncclCommInitRankFunc
AICOMRCCL-708 fix rccl unit test failures on mi300a
ROCM-2357 handle net ib request failures
```

When both a ticket and `[RCCL]` are applicable, combine them:
```
[ROCM-21076][RCCL][MI350] Fixing the rccl tuning parameters for large scale AR on MI350
[AICOMRCCL-98] [ROCM-1974] Enable user buffer and graph registration feature
```

### RCCL: Colon Style

A small number of commits use `RCCL:` (uppercase, no brackets) as a prefix.
This style is valid but less common than the bracketed form:

```
RCCL: Sync ibv_qp_init_attr_ex and related structs with latest IB verbs provider headers in ibvcore.h
RCCL: fix ProcessIsolatedRegisterTests for NCCL_LOCAL_REGISTER default
RCCL: add NCCL CMake alias shim layer
```

### CI-Only Changes

Commits that only affect CI configuration use a `CI:` prefix:

```
CI: decrease precheckin and extended test timeouts
```

### Automated / Bot Commits

These are generated automatically and require no manual intervention:

| Pattern | Source |
|---|---|
| `build(deps): bump <package> from X to Y in /projects/rccl/...` | Dependabot (conventional commits style) |
| `Bump <package> from X to Y in /projects/rccl/...` | Older Dependabot style |
| `Revert "<original subject>"` | Standard git revert |

---

## Summary Table

| Style | Pattern | When to use |
|---|---|---|
| **Primary** | `[RCCL] <description>` | General RCCL source / build / doc changes |
| **Sub-feature** | `[RCCL][SubFeature] <description>` | Changes scoped to a named subsystem (SYMMETRIC, Tuning, CUMEM, …) |
| **rccl-tests** | `[RCCL-TESTS] <description>` | Changes only in rccl-tests |
| **net-ib** | `net-ib: <description>` | Changes only in the InfiniBand transport layer |
| **Ticket + RCCL** | `[TICKET-NNN][RCCL] <description>` | Ticket-driven changes; include both for traceability |
| **Ticket only** | `[TICKET-NNN] <description>` | Ticket-driven, project tag redundant from context |
| **CI** | `CI: <description>` | CI-only configuration changes |
| **Monorepo path** | `projects/rccl: <description>` | Used by some maintainers to be explicit about monorepo location |

---

## Quick Reference

```
# Personal branch
users/<your-username>/rccl-<short-topic>
users/<your-username>/rccl/fix/<short-topic>

# Regular commit
[RCCL] Add support for foo on gfx1250

# Commit scoped to a sub-feature
[RCCL][SYMMETRIC] Fix kernel launch race condition

# Commit scoped to net-ib only
net-ib: add FooBar test to NetIbMPITests

# Ticket-driven commit
[AICOMRCCL-123][RCCL] Enable bar for multi-node topologies

# rccl-tests only
[RCCL-TESTS] Add MPI barrier before teardown
```
