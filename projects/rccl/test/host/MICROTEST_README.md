# `test/host` — microtests for RCCL internals

`rccl-UnitTestsMicro` is for host only testing: tests that give feedback quickly, support future code changes, help predict release success, and minimize maintence burden. They run in isolation from real RCCL plumbing — no
`librccl.so`, no GPU, no proxy threads, no network.

Here "microtest" is defined by GeePaw Hill:

> *"A microtest is a small, fast, precise, easy-to-invoke/read/write/debug
> chunk of code that exercises a single particular path through another
> chunk of code containing the branching logic from my shipping app."*
>
> — GeePaw Hill, [Microtest TDD: More Definition][gpwh-microtest]

[gpwh-microtest]: https://www.geepawhill.org/2020/06/12/microtest-tdd-more-definition/

Concretely, this binary compiles selected RCCL source files
**directly** into the test executable instead of reaching them through
`librccl.so`. The goal is host only, fast coverage of
internal logic in those files — including `static`-linked helpers that
aren't reachable any other way.

This document is the standing record of:

- why this binary exists alongside `rccl-UnitTests`,
- the tradeoffs of the direct-compile approach,
- the layered scaffolding that makes it actually link,
- how to add tests incrementally and watch branch coverage grow,
- how to deal with each category of dependency that crops up.

If you just want to *run* it, jump to [Running and rebuilding](#running-and-rebuilding).


## Units under test

Each production `.cc` gets its own binary so that `#include`-ing it cannot
collide with another unit's file-scope state (`static` globals, `std::once_flag`,
translation-unit anonymous namespaces):

- **`rccl-UnitTestsMicro`** — `p2p.cc` (via `P2P_CC_PATH`); suites `P2pMicrotest.*`,
  `FreshRegistration*`.
- **`rccl-UnitTestsMicroInit`** (+ **`-uncached`**) — `init.cc` (via `INIT_CC_PATH`);
  suites `InitMicrotest.*`, `InitMicrotestIsolated.*`. The `-uncached` variant adds
  `HIP_HOST_UNCACHED_MEMORY`/`HIP_UNCACHED_MEMORY` to cover the alternate host-alloc
  arm. init.cc compiles the *real* `argcheck.cc`/`archinfo.cc`/`utils.cc` ("oracle"
  TUs) from the hipify tree rather than stubbing them; `--gc-sections` drops the
  deep-path symbols the tests never reach. See `test_categories_micro_init.yaml`.

Everything below (seams, fakes, coverage) applies to both; the concrete examples
use `p2p.cc`.


## Why a separate test binary

The existing `rccl-UnitTests` binary links against `librccl.so`. That
works well for tests that exercise the public API and can tolerate
running a real communicator on real GPUs. It is not well suited
to:

- Covering `static` helper functions, which have no external symbol
  to call.
- Covering individual failure branches that need a specific dependency
  (the proxy layer, the HIP driver API, the topology graph) to return
  a specific failure.

`rccl-UnitTestsMicro` addresses all three by:

1. **`#include`-ing the unit-under-test `.cc` file** from the test TU,
   so `static` symbols are visible to tests.
2. **Linking the test binary against gtest only — not `librccl.so`** —
   so we can provide our own definitions for every external symbol the
   `.cc` references.
3. **Stubbing those external symbols** in `fakes/`, defaulting to
   "return failure loudly" so tests that accidentally exercise an
   un-faked path fail fast.


## Tradeoffs

### Pros

- Real seam control: Each external function becomes
  a function you can control by defining a test double that behaves however you
  need it to to reach the code you are trying to test.
- Fast: No HIP init, no `hipSetDevice`, no proxy
  threads, no network. Whole binary runs in milliseconds.
- No GPU required
- Static-symbol access: `#include`-ing the `.cc` exposes every
  internal helper directly.

### Cons

- Test Double drift and maintenance: the test doubles need to match
  the API of the actual symbol. Drift can at least be detected using
  a `static_assert`.
- Cannot test things the substituted layer hides: This is
  unit-test coverage, not integration coverage. Keep the existing
  `librccl.so`-linked tests for end-to-end behaviour.
- **`static` and `#include "x.cc"` is unusual.** It's standard C++,
  but readers will need a moment to orient.

## Adding a new test

The unit-under-test is the production `.cc` that the test TU `#include`s via a
build-time path macro (e.g. `P2P_CC_PATH` → the hipified `p2p.cc`). To add a
test:

1. **Pick the unit.** If it lives in a `.cc` that is already `#include`d
   (currently `p2p.cc` or `init.cc`), skip to step 3. Otherwise add a new path
   macro in `CMakeLists.txt` (mirror `P2P_CC_PATH`/`INIT_CC_PATH`) pointing at the
   hipified copy, and `#include` it from the test TU *after* the fakes/macro shims
   are in scope. A new unit generally warrants its own binary (see
   [Units under test](#units-under-test)) so its file-scope state stays isolated.
2. **Register the source.** Add the test `.cc` to the target's source list in
   `test/host/CMakeLists.txt` — both the in-build source list and the standalone
   list for that target. If you add a new gtest suite, add its pattern to the
   target's `test/test_categories_micro*.yaml` so CTest runs it.
3. **Write the `TEST` / fixture.** Use a fixture whose `TearDown()` calls
   `ResetP2pFakes()` so hooks do not leak between tests. Install per-test
   behaviour by overwriting a `std::function` hook (see the `ScopedHook`
   helper in `p2p-test.cc`) rather than editing a fake's default.
4. **Only exercise faked seams.** Every external symbol the `#include`d `.cc`
   reaches must be satisfied by `fakes/`: a missing symbol surfaces as a
   link error, a wrongly
   defaulted hook as an unexpected call. Add or override the seam as needed
   (see "Adding more controllable seams" below).
5. **Build and run** `rccl-UnitTestsMicro` (or `ctest -R rccl-UnitTestsMicro`),
   then re-render coverage (see the Coverage section) to confirm the new branch
   is covered.

> **Moving an existing test into this directory?** Do not assume its link
> dependencies carry over. A test previously built against
> `RCCL_COMMON_LINK_LIBS` (the ordinary, *runtime-linked* RCCL test targets such
> as `rccl-UnitTests`) may have obtained HIP host-runtime symbols through
> `hip::host`; `rccl-UnitTestsMicro` intentionally links neither `hip::host`,
> `libamdhip64.so`, nor `librccl.so`. Replace those dependencies with
> fakes/seams, or keep the test in a runtime-linked RCCL target if exercising
> the real HIP runtime is part of what it validates. (The other host-only suite,
> `rccl-HostUnitTests`, is likewise hermetic and does not provide `hip::host`
> either — "host-only" means *where code runs*; the microtest additionally means
> *no HIP-runtime linkage*.)

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
- **It's a `cuMem*` / `hipMem*` symbol** → first identify which test
  model you are extending; the two treat the HIP runtime differently:
  - **Ordinary RCCL unit-test targets** link `RCCL_COMMON_LINK_LIBS`,
    which already includes `hip::host`, so most `hipMem*` host-runtime
    entry points resolve from there. Tests imported from those suites may
    legitimately rely on HIP host-runtime symbols when exercising the real
    runtime is the point of the test.
  - **`rccl-UnitTestsMicro`** deliberately does **not** link `hip::host`,
    `libamdhip64.so`, or `librccl.so`. For this target, add a fake or
    hookable seam in `fakes/` with the exact signature the HIP headers
    declare — do **not** add `hip::host` merely to resolve an undefined
    symbol. An unresolved HIP symbol is precisely the mechanism that
    surfaces an unfaked dependency.

  Either way, CUDA-driver-API shims that the real RCCL resolves
  dynamically via `dlsym` on `libcuda.so` (`cuMemGetAddressRange`,
  `cuPointerGetAttribute`, `cuMemCreate`, `cuMemExportToShareableHandle`,
  …) are never ordinary HIP host-runtime symbols, so under
  `rccl-UnitTestsMicro` they always need an explicit definition in
  `fakes/p2p_fakes.cc`: use the signature the header declares and return a
  failure code (or a canned success) by default — another bucket-C seam
  that gets the function-pointer-hook treatment when a test needs to
  drive it.
- **It's a HIP kernel launch** → you almost certainly don't want to
  test the path that launches it from this binary. Refactor the test
  to avoid the branch, or split the kernel-launching code into a
  function that can itself be stubbed.


## Coverage

`rccl-UnitTestsMicro` always builds with llvm source-based coverage
(`-fprofile-instr-generate -fcoverage-mapping`). Render a report with ROCm's
llvm tooling directly. Scope it to the unit under test -- the file compiled in
via `P2P_CC_PATH`, i.e. the unroll-transformed
`hipify/src/transport/p2p_tmp.cc` (there is no plain `p2p.cc` in the hipify
tree; scoping to a non-existent file makes `llvm-cov` silently fall back to
whole-binary totals).

```bash
BD=build/release                        # or build/debug, or a standalone build dir
BIN=$BD/test/host/rccl-UnitTestsMicro
SRC=$BD/hipify/src/transport/p2p_tmp.cc
LLVM=/opt/rocm/llvm/bin                 # ROCm's llvm-cov matches the build clang

# 1. Run the instrumented binary, capturing a raw profile.
LLVM_PROFILE_FILE=micro.profraw "$BIN"

# 2. Index it.
"$LLVM/llvm-profdata" merge -sparse micro.profraw -o micro.profdata

# 3a. File/branch totals for the unit under test:
"$LLVM/llvm-cov" report "$BIN" -instr-profile=micro.profdata \
    --show-branch-summary --show-region-summary "$SRC"

# 3b. Annotated source for one function, with inline branch counts
#     (each conditional prints e.g. `Branch (897:21): [True: 2, False: 2]`;
#      grep the output for `True: 0|False: 0` to find uncovered branches):
"$LLVM/llvm-cov" show "$BIN" -instr-profile=micro.profdata \
    --name=ipcRegisterBuffer --show-branches=count "$SRC"

# 3c. HTML report (open cov-html/index.html):
"$LLVM/llvm-cov" show "$BIN" -instr-profile=micro.profdata \
    -format=html -output-dir=cov-html --show-branches=count "$SRC"
```

On OCI compute nodes `llvm-cov`/`llvm-profdata` are unavailable: run step 1
there to produce `micro.profraw`, copy it plus the binary to a host that has
`llvm-cov` (matching the build clang's major version), and render there.

### Coverage-driven workflow

The intended iteration loop for this directory:

1. Render the annotated source (step 3b) and find an uncovered branch
   (`True: 0` / `False: 0`).
2. Trace what state would have to exist for control flow to reach it.
3. Add a new `TEST()` that constructs that state.
4. Rebuild, re-render coverage, confirm the branch is now hit.
5. Commit, noting in the message which branch the new test covers.


## Running and rebuilding

RCCL's canonical build entry point is `./install.sh` (never `cmake`
directly). The two-phase pattern for this directory is: one full
`install.sh` to configure + build everything, then a tight
`make`-only inner loop for every subsequent edit to `p2p-test.cc` or
`fakes/p2p_fakes.cc`.

### Initial (one-time) build

Local-arch (`-l`), with tests (`-t`):

```bash
./install.sh -l -t -j $(nproc)
```

### Tight inner loop (after editing a test or a fake)

```bash
cd build/release
make -j $(nproc) rccl-UnitTestsMicro
```

### Run

```bash
# All tests:
./build/release/test/host/rccl-UnitTestsMicro

# One test:
./build/release/test/host/rccl-UnitTestsMicro \
    --gtest_filter='P2pMicrotest.IpcRegisterBuffer_NullRegRecordIsNoOp'

# Coverage: see the Coverage section (run under LLVM_PROFILE_FILE, then llvm-cov).
```

## Standalone host-only build (no full librccl build)

`test/host/CMakeLists.txt` is dual-mode. Alongside the in-RCCL-build target
above (`./install.sh -t`, wired via `add_subdirectory(host)`), the same file
can be configured **directly** to build the host binaries — `rccl-HostUnitTests`
and `rccl-UnitTestsMicro` — **without configuring/building all of librccl**. It
compiles just the tests + fakes + the hipified unit-under-test sources.

**ROCm is a prerequisite.** Per epic AICOMRCCL-1661 ("ROCm toolchain is
available"), this build uses `hipcc` in host-only mode (`--offload-host-only`)
against the **real ROCm headers**. There is no CPU-only / g++ path and no stubbed
`<hip/*>` / `<hsa/*>` / `<cuda*>` headers. It links **gtest + fmt only** and
passes `-no-hip-rt`, so it links **neither `librccl.so` nor the HIP runtime** —
every HIP symbol the tests reach is provided by `fakes/`.

```bash
cd projects/rccl/test/host
cmake -B build -DCMAKE_BUILD_TYPE=Release \
      -DRCCL_BUILD_DIR=/path/to/projects/rccl/build/release
cmake --build build -j"$(nproc)"
./build/rccl-UnitTestsMicro          # p2p tests, ldd shows no HIP/ROCm/HSA/RCCL
./build/rccl-UnitTestsMicroInit      # init.cc tests
./build/rccl-UnitTestsMicroInit-uncached
./build/rccl-HostUnitTests
```

Disable coverage instrumentation for the standalone micro-test with
`-DMICRO_COVERAGE=OFF`.
