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

Units sharing a binary must be `#include`d from *different* test TUs and must not
export colliding non-`static` symbols; otherwise a unit needs its own binary:

- **`rccl-UnitTestsMicro`** — one unit per test TU:
  - `p2p.cc` (`P2P_CC_PATH`, from `p2p-test.cc`); suites `P2pMicrotest.*`,
    `FreshRegistration*`.
  - `rma/rma_proxy_progress.cc` (`RMA_PROXY_PROGRESS_CC_PATH`, from
    `rma-proxy-progress-test.cc`); suite `RmaProxyProgressTest.*`.
  - `devcomm/devcomm_v22902.cc` + `devcomm/devcomm_v22907.cc`
    (`DEVCOMM_V22902_CC_PATH` / `DEVCOMM_V22907_CC_PATH`, both from
    `devcomm-test.cc`); suites `Devcomm*`. `devcomm/devcomm_v23000.cc` is not
    covered yet.
- **`rccl-UnitTestsMicroEnqueue`** — `enqueue.cc` (via `ENQUEUE_CC_PATH`); suite
  `EnqueueMicrotest.*`. All tests live in `enqueue-test.cc`, grouped by unit under
  test; several fixtures are reused by later groups, so the order within the file
  matters. `enqueue.cc:28` pulls in the device header `src/device/common.h`,
  which cannot compile host-only; the TU pre-sets that header's include guard and
  supplies the six `ncclDevKernel_Generic_N` kernels as host surrogates (their
  addresses are stored in a table, never launched). Two shared `nccl_stubs.cc`
  entries are omitted for this target via `RCCL_STUBS_OMIT_<symbol>` macros
  because `enqueue.cc` defines them itself. See
  `test_categories_micro_enqueue.yaml`.
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

1. **Pick the unit.** If it lives in a `.cc` that is already `#include`d (see
   [Units under test](#units-under-test)), skip to step 3. Otherwise add a new path
   macro in `CMakeLists.txt` (mirror `P2P_CC_PATH`/`INIT_CC_PATH`) pointing at the
   hipified copy, and `#include` it from the test TU *after* the fakes/macro shims
   are in scope. A new unit generally warrants its own binary (see
   [Units under test](#units-under-test)) so its file-scope state stays isolated.
2. **Register the source.** Add the test `.cc` to the target's source list in
   `test/host/CMakeLists.txt` (`RCCL_MICRO_TEST_SOURCES` for
   `rccl-UnitTestsMicro`). If you add a new gtest suite, add its pattern to the
   target's `test/test_categories_micro*.yaml` so CTest runs it.
3. **Write the `TEST` / fixture.** Use a fixture whose `TearDown()` calls the
   unit's reset entry point (`ResetP2pFakes()`, `ResetEnqueueFakes()`, ...) so
   hooks do not leak between tests. Install per-test behaviour by overwriting a
   `std::function` hook rather than editing a fake's default. Prefer the
   `ScopedHook` helper in `ScopedHook.h` (see
   [Installing per-test behaviour with `ScopedHook`](#installing-per-test-behaviour-with-scopedhook)),
   which installs the hook, counts calls, and restores the previous behaviour
   automatically on scope exit.
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

## Where a fake belongs

**A fakes file is named after the production TU that DEFINES the symbol, never
after the test target that first needed it.** A fake can only ever replace an
*external*, so "the unit under test owns it" is never the reason a symbol is in
a file — if it were owned by the UUT there would be nothing to fake. Filing by
target instead produces the same symbol faked three times in three files, each
slightly weaker than the others, which is what `rccl::Recorder` and `ncclGetEnv`
had become before this map existed.

| Production TU | Fakes file |
|---|---|
| `src/ce_coll.cc` | `fakes/ce_fakes.cc` |
| `src/collectives.cc` | `fakes/collectives_fakes.cc` |
| `src/dev_runtime.cc` | `fakes/dev_runtime_fakes.cc` |
| `src/graph/tuning.cc`, `src/graph/connect.cc` params | `fakes/tuning_fakes.cc` |
| `src/init.cc` comm lifecycle | `fakes/comm_fakes.cc` |
| `src/misc/param.cc` + `getenv` interposition | `fakes/env_fakes.cc` |
| `src/misc/strongstream.cc` | `fakes/strongstream_stubs.cc` |
| `src/misc/utils.cc` | `fakes/utils_fakes.cc` |
| `src/os/linux.cc` | `fakes/os_fakes.cc` |
| `src/proxy.cc` | `fakes/proxy_fakes.cc` |
| `src/rccl_wrap.cc` | `fakes/rccl_wrap_fakes.cc` |
| `src/recorder.cc` | `fakes/recorder_fakes.cc` |
| `src/register/*.cc` | `fakes/register_stubs.cc` |
| `src/scheduler/*.cc` and the deep launch paths | `fakes/sched_stubs.cc` |
| `src/sym_kernels.cc` | `fakes/sym_kernels_fakes.cc` |
| `src/transport/*` | `fakes/transport_stubs.cc` |
| core/lifecycle floor + data symbols | `fakes/nccl_stubs.cc` |
| reusable `nccl*` seams | `fakes/nccl_fakes.cc` |
| HIP runtime | `fakes/hip_fakes.cc` |

**`_fakes.cc` versus `_stubs.cc`.** The suffix records what the file mostly *is*, not a rule the
build enforces: `_fakes` for a file whose point is controllable seams, `_stubs` for one whose point
is a fail-loud floor. Several files are honestly both — `transport_stubs.cc` is a floor that also
owns one driven seam, and `os_fakes.cc` holds working implementations with no seam at all. Do not
read the suffix as a guarantee; read the file's header comment, which states what it owns. If you
add a file, pick the suffix matching its majority content and say so at the top.

**`// UNDRIVEN`.** A seam carrying this marker is declared so the binary links and so an accidental
call is visible, NOT because its path is covered. It is a link-floor entry with a chosen default,
and that default silently selects which production arm runs. Driving a seam means deleting its
marker. The marker travels with the declaration rather than a block comment so it cannot drift from
what it describes. Call *counters* do not take the marker unless the counter itself is unread.

Three things do NOT follow the TU-per-file rule, deliberately:

- `fakes/collective_stubs.cc` is a fail-loud floor for the collective *launch*
  pipeline (`ncclLaunchKernel` and friends), which `enqueue.cc` itself defines.
  It therefore cannot link into the enqueue target and stays target-shaped.
- `ncclStrongStreamAcquire` / `Release` stay in `nccl_fakes.cc` rather than
  `strongstream_stubs.cc`: they carry `ASSERT_HOOK_MATCHES_PROD` drift
  assertions and moving those is a larger change.
- `src/os/*.cc` is only partly consolidated. `os_fakes.cc` owns the `linux.cc`
  allocation shims, but `ncclOsCpuCount`, `ncclOsSetAffinity` and
  `ncclOsTopoGetStrFromSys` are still split between `collective_stubs.cc` and
  `nccl_stubs.cc`. That predates this map; the row below is where they *should*
  live, not where all of them do.

`<uut>_fakes.h` (e.g. `enqueue_fakes.h`) is an aggregation header: it includes
the per-TU headers that unit's tests use and declares the `Reset<Uut>Fakes()`
that chains their per-TU resets. It defines no seams itself.

**`RCCL_STUBS_OMIT_<symbol>` is only for a symbol the unit under test defines**,
where the omission exists purely to avoid a duplicate at link time. If a target
instead needs a real *value* where the shared floor aborts, that symbol wants a
seam in its owning TU's fakes file, which serves every target at once. Reaching
for an omit macro plus a private replacement file is how `rcclUseAinic` ended up
faked in two places.

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

### Factor each default into a named `Default*` function

A hook's default behaviour is needed in two places — the hook's
initialiser and the fakes file's `Reset*()` function. Do **not** write
the lambda body out twice; the two copies drift. Instead put each
default in a named free function prefixed `Default` and reference it
from both. `fakes/hip_fakes.cc` and `fakes/rma_fakes.cc` follow this
pattern:

```cpp
// One definition of the behaviour...
static ncclResult_t DefaultRmaDestroyDesc(struct ncclComm*,
                                          struct ncclRmaProxyDesc** desc) {
    *desc = nullptr;
    return ncclSuccess;
}

// ...used for the hook's initial value...
std::function<ncclResult_t(struct ncclComm*, struct ncclRmaProxyDesc**)>
    g_rmaDestroyDesc = DefaultRmaDestroyDesc;

// ...and reused by the reset, no duplicated body.
void ResetRmaFakes() {
    g_rmaDestroyDesc = DefaultRmaDestroyDesc;
}
```

### Installing per-test behaviour with `ScopedHook`

Once a seam is a `std::function` hook, install per-test behaviour with the
`ScopedHook` RAII helper (`test/host/ScopedHook.h`) instead of assigning the
global directly and remembering to reset it. `ScopedHook` does three things:

1. installs the test's behaviour on construction,
2. counts calls automatically via its `.calls` member, and
3. restores the previous behaviour in its destructor — so a hook can't leak
   into the next test even if you forget the fixture's `TearDown` reset.

Class template argument deduction picks up the signature from the hook
variable, so call sites don't spell it out:

```cpp
#include "ScopedHook.h"

TEST_F(P2pMicrotest, IpcRegisterBuffer_UsesBaseAddr) {
    ScopedHook memGet(g_hipMemGetAddressRange,
        [&](hipDeviceptr_t* pb, std::size_t* ps, hipDeviceptr_t) {
            if (pb) *pb = /* canned base addr */;
            if (ps) *ps = /* canned size */;
            return hipSuccess;
        });

    // ... call the unit under test ...

    EXPECT_EQ(memGet.calls, 1);
    // memGet's destructor restores g_hipMemGetAddressRange here.
}
```

`ScopedHook` is intentionally non-copyable and non-movable (the installed
lambda captures `this` to bump the counter), so declare it as a local; C++17
guaranteed copy elision lets helper factories still return it by value.

### Naming a reusable seam behaviour with a lambda factory

When the *same* seam behaviour is needed by many tests — e.g. "make this param
return non-zero so control reaches branch X" — don't copy-paste the lambda into
every test. Wrap it in a small **lambda-factory helper**: a function that
*returns a lambda* whose signature matches the seam. This keeps the "what value
must this seam return, and why" knowledge in one named, commented place.

```cpp
// Returns a hook lambda that reports the buffer as legacy-IPC-capable, so the
// `else if (legacyIpcCap)` arm of ipcRegisterBuffer is reachable.
auto ForceLegacyIpcCapable()
{
    return [](void* data, hipPointer_attribute attribute,
              hipDeviceptr_t) -> hipError_t {
        if (data && attribute == HIP_POINTER_ATTRIBUTE_IS_LEGACY_HIP_IPC_CAPABLE)
            *static_cast<int*>(data) = 1;
        return hipSuccess;
    };
}
```

Return a **plain lambda, not a pre-wrapped `ScopedHook`**, so call sites stay
free to compose the recipe into whichever installation form they need:

```cpp
// As a ScopedHook local (CTAD deduces the signature from the hook variable):
ScopedHook pointerAttr(g_hipPointerGetAttribute, ForceLegacyIpcCapable());

// Or into an optional/emplace form owned by a shared fixture:
pointerAttr.emplace(g_hipPointerGetAttribute, ForceLegacyIpcCapable());
```

Name the factory after the *state it forces* (`ForceLegacyIpcCapable`,
`ForceLegacyCudaRegister`), not after the seam it drives — the point is that a
test reads as "force this precondition, then call the unit under test". Where a
single branch depends on HIP version (only one arm is live per build, but the
test can't know which at authoring time), pair the factories that drive each
version's arm so the test passes regardless of the toolchain — see
`ForceLegacyCudaRegister` + `ForceLegacyIpcCapable` in `p2p-test.cc`.

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
can be configured **directly** to build every host binary — `rccl-HostUnitTests`,
`rccl-UnitTestsMicro`, `rccl-UnitTestsMicroInit[-uncached]` and
`rccl-UnitTestsMicroEnqueue[-devlinker]` — **without configuring/building all of
librccl**. It compiles just the tests + fakes + the hipified unit-under-test
sources.

Two of those names are preprocessor variants, not duplicates. `init.cc` gates
part of its allocation path on `HIP_*_UNCACHED_MEMORY` and `enqueue.cc` gates
`rcclShmemDynamicSize` on `RCCL_DEVICE_LINKER`, both at the **preprocessor**, so
one compile can only ever reach one arm. The in-RCCL-build path inherits
`RCCL_DEVICE_LINKER` from the `rccl` target's compile definitions
(`ENABLE_DEVICE_LINKER` defaults ON, so the device-linker arm is the one that
ships); this standalone project has no `rccl` target to inherit from, which is
why it builds the `-devlinker` variant explicitly.

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
./build/rccl-UnitTestsMicroEnqueue            # enqueue.cc tests
./build/rccl-UnitTestsMicroEnqueue-devlinker  # same, RCCL_DEVICE_LINKER arm
./build/rccl-HostUnitTests
```

Disable coverage instrumentation for the standalone host-only test binaries
with `-DHOST_TEST_COVERAGE=OFF`.
