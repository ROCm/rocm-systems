# `test/microtest` — microtests for RCCL internals

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

TODO REPLACE DELETED SLOP

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
- **It's a `cuMem*` / `hipMem*` symbol** → first try
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
- **It's a HIP kernel launch** → you almost certainly don't want to
  test the path that launches it from this binary. Refactor the test
  to avoid the branch, or split the kernel-launching code into a
  function that can itself be stubbed.


## Coverage

`rccl-UnitTestsMicro` always builds with llvm source-based coverage.

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
./build/release/test/microtest/rccl-UnitTestsMicro

# One test:
./build/release/test/microtest/rccl-UnitTestsMicro \
    --gtest_filter='IpcRegisterBuffer.NullRegRecordIsNoOp'

# Coverage (see Coverage section for FUNC/--html options):
./test/microtest/coverage.sh
```
