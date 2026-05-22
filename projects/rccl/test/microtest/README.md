# `test/microtest` — microtests for RCCL internals

`rccl-UnitTestsMicro` is a **microtest** binary: tests that are small,
fast, branchy, and run in isolation from real RCCL plumbing — no
`librccl.so`, no GPU, no proxy threads, no network.

We use "microtest" in the sense GeePaw Hill describes:

> *"A microtest is a small, fast, precise, easy-to-invoke/read/write/debug
> chunk of code that exercises a single particular path through another
> chunk of code containing the branching logic from my shipping app."*
>
> — GeePaw Hill, [Microtest TDD: More Definition][gpwh-microtest]

[gpwh-microtest]: https://www.geepawhill.org/2020/06/12/microtest-tdd-more-definition/

Concretely, this binary compiles selected RCCL source files
**directly** into the test executable instead of reaching them through
`librccl.so`. The goal is hermetic, GPU-less, fast coverage of
internal logic in those files — including `static`-linked helpers that
aren't reachable any other way.

This document is the standing record of:

- why this binary exists alongside `rccl-UnitTests`,
- the tradeoffs of the direct-compile approach,
- the layered scaffolding that makes it actually link,
- how to add tests incrementally and watch branch coverage grow,
- how to deal with each category of dependency that crops up.

If you just want to *run* it, jump to [Running and rebuilding](#running-and-rebuilding).


## Why a separate test binary

The existing `rccl-UnitTests` binary links against `librccl.so`. That
works well for tests that exercise the public API and can tolerate
running a real communicator on real GPUs. It is **not** well suited
to:

- Covering `static` helper functions, which have no external symbol
  to call.
- Covering individual failure branches that need a specific dependency
  (the proxy layer, the HIP driver API, the topology graph) to return
  a specific failure.
- Running in CI containers that have no GPUs, no `/dev/kfd`, and no
  network fabric.

`rccl-UnitTestsMicro` addresses all three by:

1. **`#include`-ing the unit-under-test `.cc` file** from the test TU,
   so `static` symbols are visible to tests.
2. **Linking the test binary against gtest only — not `librccl.so`** —
   so we can provide our own definitions for every external symbol the
   `.cc` references.
3. **Stubbing those external symbols** in `fakes/`, defaulting to
   "return failure loudly" so tests that accidentally exercise an
   un-faked path fail fast.

The first file under test is `src/transport/p2p.cc`, with
`ipcRegisterBuffer` as the initial focus.


## Tradeoffs

### Pros

- **Real seam control.** Each external function (proxy calls, HIP
  driver entry points, registration helpers, topology checks) becomes
  a function you define yourself in `fakes/`. You can drive failure
  paths deterministically that would be near-impossible to provoke
  against a real `librccl.so`.
- **Fast and hermetic.** No HIP init, no `hipSetDevice`, no proxy
  threads, no network. Whole binary runs in milliseconds.
- **No GPU required.** Runs in any CI container.
- **Static-symbol access.** `#include`-ing the `.cc` exposes every
  internal helper directly.

### Cons

- **The stubs *are* the test.** If your stub for
  `ncclProxyCallBlocking` doesn't match the real call's semantics
  (ownership, blocking, error conventions), your tests will pass and
  production can still break. Treat the fakes file as part of the
  contract under test.
- **Stub maintenance is a tax forever.** Every time someone adds a
  new call inside the unit under test (or anything reachable through
  its headers), the test binary fails to link until a stub is added.
- **You cannot test things the substituted layer hides.** This is
  unit-test coverage, not integration coverage. Keep the existing
  `librccl.so`-linked tests for end-to-end behaviour.
- **`static` and `#include "x.cc"` is unusual.** It's standard C++,
  but readers will need a moment to orient. The header comment in
  `p2p-test.cc` explains why.


## Architecture

```
test/microtest/
├── CMakeLists.txt          # defines rccl-UnitTestsMicro target
├── p2p-test.cc             # gtest TU; #include's the hipified p2p.cc
├── fakes/
│   └── p2p_fakes.cc        # minimal stubs for everything p2p.cc needs
├── coverage.sh             # llvm-cov wrapper (see Coverage section)
└── README.md               # this file
```

Compile graph for `rccl-UnitTestsMicro`:

```
   gtest main (common/main_altrsmi.cpp)
              │
              ▼
   p2p-test.cc  ── #include ──▶  build/release/hipify/src/transport/p2p.cc
              │
              │ link
              ▼
   fakes/p2p_fakes.cc  (definitions for every external symbol p2p.cc references)

   (no libamdhip64 calls at runtime — only at link-time, satisfied by hip::host)
   (no librccl.so — deliberately)
```

The hipified path matters: every RCCL `.cc` goes through a `hipify`
pass at build time that rewrites `cudaXxx` → `hipXxx`,
`CUdeviceptr` → `hipDeviceptr_t`, etc. We compile the
**post-hipify** copy from `build/release/hipify/src/transport/p2p.cc`,
not the pre-hipify source from `src/transport/p2p.cc`. The path is
passed to the test TU as a `-DP2P_CC_PATH="..."` preprocessor define
from CMake so the `#include` stays portable across build directories.

The CMake target also has an `add_dependencies(rccl-UnitTestsMicro rccl)`
edge — **not** to link against librccl, but to ensure the hipify step
has run before we try to compile.


## The fakes layer

`fakes/p2p_fakes.cc` provides definitions for every external symbol
that `p2p.cc` references but does not define. Symbols fall into four
buckets:

| Bucket | Examples | Default behaviour |
|---|---|---|
| **A. Globals** | `allocTracker[]`, `ncclCuMemHandleType`, `ncclDebugLevel`, `ncclDebugMask`, `ncclDebugNoWarn` | Zero-initialised or sensible default. |
| **B. Logging / params** | `ncclDebugLog(...)`, `ncclLoadParam(...)` | `ncclDebugLog` prints to stderr with `[fake]` prefix (visible during test runs); `ncclLoadParam` is a no-op, so every `NCCL_PARAM(...)` returns its compile-time default. |
| **C. Controllable seams** | `ncclProxyCallBlocking`, `ncclProxyConnect`, `ncclProxyClientGetFdBlocking`, `ncclProxyClientQueryFdBlocking`, `ncclCuMemEnable`, `ncclRegLocalIsValid`, `ncclShmImportShareableBuffer`, `ncclShmIpcClose`, `ncclShmAllocateShareableBuffer`, `ncclCommGraphRegister`, `ncclCommGraphDeregister`, `ncclStrongStream*`, `ncclStreamWaitStream` | Return `ncclSystemError` or a benign success. Tests that don't expect to reach these branches *want* the failure to surface. |
| **D. Topology / arch helpers** | `ncclTopoCheckP2p`, `ncclTopoCheckNet`, `ncclTopoGetLinkType`, `IsArchMatch`, `busIdToInt64`, `getBusId` | Return "nothing supported" / `0`. |

**Philosophy:** the fakes are intentionally minimal and intentionally
strict. A test that wants to drive one of the bucket-C functions to a
specific value should not edit the default stub — it should promote
that symbol to a function-pointer hook (see [Adding more
controllable seams](#adding-more-controllable-seams) below).


## Adding a new test

Tests live in `p2p-test.cc`. The recipe:

1. **Pick the branch you want to cover.** Open the coverage report
   (see below) scoped to the function under test and find a branch
   with `False: 0` or `True: 0`.
2. **Identify the minimum state needed to reach it.** Most branches
   in `ipcRegisterBuffer` are gated on:
   - Whether `regRecord` is null.
   - Whether `regRecord->ipcInfos[peerLocalRank]` is null (reuse vs.
     register).
   - The value of `type` (`NCCL_IPC_SENDRECV` vs `NCCL_IPC_COLLECTIVE`).
   - Whether `regRecord->regIpcAddrs.devPeerRmtAddrs` is null
     (triggers the strong-stream allocation block).
   - Failure injection through the bucket-C fakes.
3. **Build a minimal `ncclComm` / `ncclReg` on the stack.** Don't
   call any real RCCL initialisation. Zero-initialise with `{}` and
   populate only the fields the branch reads. Use sentinel
   "0xdead..." values for fields that the function should overwrite,
   so an accidental no-op is detected.
4. **Call the unit under test directly.** Because the `.cc` is
   `#include`-d, `static` helpers are in scope.
5. **Assert every output.** Including the ones you expect to be
   zeroed — a regression that "succeeds but leaves outputs uninitialised"
   is one of the kinds of bug this test surface is uniquely good at
   catching.

The existing tests in `p2p-test.cc` are the working templates:

| Test | Branch covered | Setup highlights |
|---|---|---|
| `NullRegRecordIsNoOp` | `regRecord == nullptr` | Bare `ncclComm{}` is enough. |
| `SendrecvReusesExistingIpcInfo` | Reuse + `NCCL_IPC_SENDRECV` post-loop | Hand-built `ipcInfos[peerLocalRank]` and `hostPeerRmtAddrs[]`. |
| `CollectiveReuseReturnsDevicePeerAddrTable` | Reuse + `NCCL_IPC_COLLECTIVE` post-loop with table pre-populated | Adds a pre-built `devPeerRmtAddrs[]` so the strong-stream block stays skipped. |


## Adding more controllable seams

The fakes today return constants. When a test needs to drive one of
them to a specific value (for instance, fake
`ncclProxyCallBlocking` returning a canned `rmtRegAddr` so the
new-registration happy path can be tested), the recommended pattern
is:

1. In `fakes/p2p_fakes.cc`, add a `std::function`-typed hook with a
   default that matches the current constant behaviour:
   ```cpp
   std::function<ncclResult_t(ncclComm*, ncclProxyConnector*, int,
                              void*, int, void*, int)>
       g_proxyCallBlocking = [](auto...) { return ncclSystemError; };

   ncclResult_t ncclProxyCallBlocking(ncclComm* c, ncclProxyConnector* p,
                                      int t, void* req, int rs,
                                      void* resp, int rsz) {
       return g_proxyCallBlocking(c, p, t, req, rs, resp, rsz);
   }
   ```
2. Expose the hook from a small `fakes/p2p_fakes.h` so tests can
   install per-test behaviour in a gtest fixture's `SetUp` / `TearDown`.
3. Reset the hook to its default in `TearDown` so tests don't
   contaminate each other.

This is preferable to e.g. `LD_PRELOAD` or `--wrap` because the seam
is explicit, greppable, and visible in code review.


## Dealing with each kind of dependency

When the link fails with `undefined symbol: foo`, find `foo` and
triage it into the right bucket:

- **It's a global variable (`extern int foo;`)** → add a definition
  to `fakes/p2p_fakes.cc`. Use a sensible default (usually zero).
- **It's a logging or env-param helper** → already covered by the
  no-op `ncclDebugLog` / `ncclLoadParam`. If a new logging primitive
  appears, follow the same pattern.
- **It's a `ncclProxy*` / `ncclShm*` / `ncclCommGraph*` / `ncclTopo*`
  function** → add a return-failure stub. If a future test will need
  to drive it, plan for the function-pointer-hook upgrade.
- **It's a HIP driver / `cuMem*` / `hipMem*` symbol** → first try
  linking against `hip::host` (already in `RCCL_COMMON_LINK_LIBS`);
  most `hipMem*` host-runtime entry points resolve from there. If
  the symbol isn't in the host runtime — typically a CUDA-driver-API
  shim that the real RCCL resolves through `dlsym` on `libcuda.so`
  at runtime (`cuMemGetAddressRange`, `cuPointerGetAttribute`,
  `cuMemCreate`, `cuMemExportToShareableHandle`, …) — define your
  own version in `fakes/p2p_fakes.cc`. Use the same signature the
  header declares and have it return a failure code (or a canned
  success) by default; this is just another bucket-C seam and gets
  the function-pointer-hook treatment when a test needs to drive it.
  Don't try to link `libcuda.so` into this binary — the whole point
  is that it runs on machines with no driver.
- **It's a HIP kernel launch** → you almost certainly don't want to
  test the path that launches it from this binary. Refactor the test
  to avoid the branch, or split the kernel-launching code into a
  function that can itself be stubbed.


## Coverage

`rccl-UnitTestsMicro` supports llvm source-based coverage via a
**separate** cmake option from the rest of the tree:

- `ENABLE_CODE_COVERAGE` (top-level) — gated on Debug builds because
  it instruments `librccl.so` and needs Debug-only internal symbol
  visibility.
- `ENABLE_MICROTEST_COVERAGE` (this directory) — independent knob, works
  in any build type. The microtest binary doesn't link `librccl.so`, so it
  has full symbol visibility regardless of build type.

Enable at configure time (see [Running and rebuilding](#running-and-rebuilding)
for the canonical build commands). The short version: pass
`--cmake-options "-DENABLE_MICROTEST_COVERAGE=ON"` to `./install.sh` on
the initial build, then iterate with `make` in `build/release`.

Render a report:

```bash
# File totals only (whole p2p.cc):
./test/microtest/coverage.sh

# File totals + annotated source for one function, with inline branch counts:
FUNC=ipcRegisterBuffer ./test/microtest/coverage.sh

# Same, plus an HTML report:
FUNC=ipcRegisterBuffer ./test/microtest/coverage.sh --html build/release/test/microtest/coverage/html
# Then open: build/release/test/microtest/coverage/html/index.html
```

The annotated text output puts a block under every conditional:

```
|  Branch (897:21): [True: 2, False: 2]
```

To find branches that are still uncovered, grep:

```bash
FUNC=ipcRegisterBuffer ./test/microtest/coverage.sh | grep -E "True: 0|False: 0"
```

Each match is a candidate for the next test.

### Coverage-driven workflow

The intended iteration loop for this directory:

1. Run `FUNC=<name> ./test/microtest/coverage.sh` and find an uncovered
   branch.
2. Trace what state would have to exist for control flow to reach it.
3. Add a new `TEST()` that constructs that state.
4. Rebuild, re-run coverage, confirm the branch is now hit.
5. Commit, with the coverage delta in the commit message
   (e.g. "branch coverage 0.97% → 1.4%").


## Running and rebuilding

RCCL's canonical build entry point is `./install.sh` (never `cmake`
directly). The two-phase pattern for this directory is: one full
`install.sh` to configure + build everything, then a tight
`make`-only inner loop for every subsequent edit to `p2p-test.cc` or
`fakes/p2p_fakes.cc`.

### Initial (one-time) build

Local-arch (`-l`), with tests (`-t`) and microtest coverage
instrumentation wired in via `--cmake-options`:

```bash
./install.sh -l -t -j $(nproc) \
    --cmake-options "-DENABLE_MICROTEST_COVERAGE=ON"
```

`-l` is non-negotiable for dev builds — a cross-arch build is 30+
minutes. `-t` is required: without `BUILD_TESTS=ON` the
`test/microtest/CMakeLists.txt` guard short-circuits and
`rccl-UnitTestsMicro` is never produced. `install.sh` wipes
`build/release` first; that's the safety guarantee.

### Tight inner loop (after editing a test or a fake)

```bash
cd build/release
make -j $(nproc) rccl-UnitTestsMicro
```

This rebuilds only what changed. Don't re-run `./install.sh` for
incrementals — it would wipe the whole tree and force a full
reconfigure.

### Run

```bash
# All tests:
./build/release/test/microtest/rccl-UnitTestsMicro

# One test:
./build/release/test/microtest/rccl-UnitTestsMicro \
    --gtest_filter='IpcRegisterBuffer.NullRegRecordIsNoOp'

# Coverage (see Coverage section for FUNC/--html options):
./test/microtest/coverage.sh
```

### When to drop back to a full `./install.sh`

- You toggled `-DENABLE_MICROTEST_COVERAGE` (or any other
  configure-time cmake flag).
- You switched git branches or pulled commits that touch
  `CMakeLists.txt`, `install.sh`, or `src/device/generate.py`.
- You see linker errors that don't make sense — stale objects from a
  previous configure are in play.


## What this binary is not

- **Not** a replacement for `rccl-UnitTests` or the MPI test binary.
  Those exercise real RCCL behaviour end-to-end and remain authoritative.
- **Not** a place for tests that need real HIP, real GPUs, or any
  multi-process orchestration. Those belong in the existing test
  binaries.
- **Not** a microbenchmark harness. The binary is meant to be cheap
  to run, not a perf bed.

When in doubt: if your test cares about whether something *actually
works on hardware*, it belongs elsewhere. If your test cares about
whether one specific branch in one specific function does the right
bookkeeping, it belongs here.
