# `test/host` — microtests for RCCL internals

The `rccl-UnitTestsMicro*` binaries are for host only testing: tests that give feedback quickly, support future code changes, help predict release success, and minimize maintence burden. They run in isolation from real RCCL plumbing — no
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

## Naming a new microtest

One unit under test fixes every other name, mechanically, so nothing has to be
invented per binary. For a unit at `src/<dirs>/<stem>.cc`:

| Thing | Rule | `src/gin/gin_host.cc` |
|---|---|---|
| Path macro | `<STEM>_CC_PATH`, `<STEM>` upper-snake, prefixed with the parent directory when taken (`net_ib/connect.cc` → `NET_IB_CONNECT_CC_PATH`) | `GIN_HOST_CC_PATH` |
| Test TU | `test/host/<dirs>/<stem>-test.cc`, `_` → `-`; prefix the stem with the parent directory if the basename is taken under the same top directory (`net_ib/p2p.cc` → `net-ib-p2p-test.cc`) | `test/host/gin/gin-host-test.cc` |
| Binary | `rccl-UnitTestsMicro<Dirs><TestStem>` UpperCamel; `<TestStem>` is the TU's basename less `-test.cc` and less leading words already in `<Dirs>` | `rccl-UnitTestsMicroGinHost` |
| gtest suites | `<Stem>…Microtest` — every suite in the TU starts with `<Stem>` | `GinHostProxyAffinityMicrotest` |
| CTest categories | `test/test_categories_micro_<top>.yaml`, `<top>` the first of `<dirs>`, shared by its binaries | `test/test_categories_micro_gin.yaml` |
| JUnit XML | `host_tests_micro_<binary suffix, snake_case>.xml` (`run_host_tests.sh` derives it from the binary) | `host_tests_micro_gin_host.xml` |

Because every suite starts with its unit's `<Stem>`, the directory's yaml needs
one pattern per unit, and fewer when one stem prefixes another (gin's
`GinHost*` also matches `GinHostProxy*`). A new suite in the same TU is picked
up without editing it — gtest's `*` does not cross the `.`, so a per-suite list
silently drops any suite someone forgets to add.

A new binary is named once, in its `rccl_add_micro_binary()` call in
`test/host/CMakeLists.txt`, or its directory wrapper's call in
`<dir>/host_tests.cmake` (e.g. `rccl_add_transport_micro_binary()`). The
standalone configure writes every binary to `host_test_binaries.txt` in the build
directory, which `run_host_tests.sh` runs, and `lib/test_executor.py` passes every
`rccl-UnitTestsMicro*` binary to llvm-cov as `--object`.

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

The microtests address both by:

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

1. **Pick the unit.** If a test file already `#include`s its `.cc`, skip to step 3.
   Otherwise add a new path macro beside its binary (mirror `P2P_CC_PATH`/`INIT_CC_PATH`)
   pointing at the hipified copy, and `#include` it from the test TU *after* the
   fakes/macro shims are in scope. A new unit gets its own test file and binary, so its
   file-scope state stays isolated.
   The unit's path fixes the test file, binary, suite and yaml names — see
   [Naming a new microtest](#naming-a-new-microtest).
2. **Register the source.** Add the test to its binary's
   `rccl_add_unit_micro_binary()` call (`test/host/CMakeLists.txt` for root tests,
   else `<dir>/host_tests.cmake`), which both build paths share. If you add a new
   gtest suite, add its pattern to the target's `test/test_categories_micro*.yaml`
   so CTest runs it (a suite named for its binary's stem is already matched). A new
   *binary* is named once, as [Naming a new microtest](#naming-a-new-microtest)
   describes.
3. **Write the `TEST` / fixture.** Use a fixture whose `TearDown()` calls the
   unit's reset entry point (`ResetP2pFakes()`, `ResetInitFakes()`, ...) so
   hooks do not leak between tests. Install per-test behaviour by overwriting a
   `std::function` hook rather than editing a fake's default. Prefer the
   `ScopedHook` helper in `ScopedHook.h` (see
   [Installing per-test behaviour with `ScopedHook`](#installing-per-test-behaviour-with-scopedhook)),
   which installs the hook, counts calls, and restores the previous behaviour
   automatically on scope exit. Reach for the other
   [shared test-side helpers](#shared-test-side-helpers) before hand-rolling a
   failure counter, a wait loop, or a scope guard.
   A death test takes `DEATH_BY_SEGV`, `DisableCoreDumpsForExpectedCrash()` and
   `ScopedDeathTestStyle` from `ExpectedCrash.h`. If the crash needs a NULL, hide it
   from the optimizer with `asm volatile("" : "+r"(p))`, or it may drop the crash.
4. **Only exercise faked seams.** Every external symbol the `#include`d `.cc`
   reaches must be satisfied by `fakes/`: a missing symbol surfaces as a
   link error, a wrongly
   defaulted hook as an unexpected call. Add or override the seam as needed
   (see "Adding more controllable seams" below).
5. **Build and run** the test's binary (or `ctest -R <binary>`),
   then re-render coverage (see the Coverage section) to confirm the new branch
   is covered.

> **Moving an existing test into this directory?** Do not assume its link
> dependencies carry over. A test previously built against
> `RCCL_COMMON_LINK_LIBS` (the ordinary, *runtime-linked* RCCL test targets such
> as `rccl-UnitTests`) may have obtained HIP host-runtime symbols through
> `hip::host`; the `rccl-UnitTestsMicro*` binaries intentionally link neither `hip::host`,
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

`src/graph/search.cc` has two rows for the same reason as `src/rccl_wrap.cc` below:
the targets that compile it (`rccl-UnitTestsMicroGraphSearch`, which tests it, and the
other `src/graph/` binaries that link it, which only need it to collaborate with
`paths.cc`) take its dependencies from `topo_stubs.cc` or `graph_fakes.cc`, while
every other target fakes its own entry points (`search_fakes.cc`). The two never define
the same symbol, so no target needs an `RCCL_STUBS_OMIT_` macro for them.

`src/rccl_wrap.cc` has two rows because it is both a unit under test
(`rccl-UnitTestsMicroWrap` compiles it and fakes its dependencies) and a
dependency of another unit (`rccl-UnitTestsMicroEnqueue` doesn't compile it,
so it fakes the file's own entry points). The two never define the same
symbol.

| Production TU | Fakes file |
|---|---|
| `src/algorithms/dda/*.cc` (targets that do not compile the real file) | `fakes/dda_fakes.cc` |
| `src/allocator.cc` | `fakes/allocator_fakes.cc` |
| `src/bootstrap.cc` | `fakes/bootstrap_stubs.cc` |
| externals `src/bootstrap.cc` reaches (`rccl-UnitTestsMicroBootstrap`, which compiles it) | `fakes/bootstrap_micro_fakes.cc` |
| `src/ce_coll.cc` (targets that do not compile the real file) | `fakes/ce_fakes.cc` |
| `src/channel.cc` (targets that do not compile the real file) | `fakes/channel_fakes.{h,cc}` |
| `src/collectives.cc` | `fakes/collectives_fakes.cc` |
| `src/debug.cc` (`ncclDebugLog` recorder for binaries that cannot link `nccl_fakes.cc`, which owns it everywhere else) | `fakes/debug_fakes.cc` |
| `src/dev_runtime.cc` (targets that do not compile the real file) | `fakes/dev_runtime_fakes.cc` |
| `src/diagnostics.cc` (`ncclDiagChildRun*` external-tool runners) | `fakes/diagnostics_fakes.cc` |
| externals `src/dev_runtime.cc` reaches (the binaries that compile it, e.g. `rccl-UnitTestsMicroDevRuntime`) | `fakes/dev_runtime_micro_fakes.cc` |
| externals the devcomm compat shims reach | `fakes/devcomm_fakes.cc` |
| `src/diagnostics/device/p2p.cu` (`ncclDiagP2p*` kernel launchers) | `fakes/diagnostics_p2p_device_fakes.cc` |
| `src/graph/*.cc` (topo, paths, connect, rome consensus) and `search.cc`'s *dependencies* | `fakes/topo_stubs.cc` |
| `src/graph/search.cc`'s own entry points (targets that don't compile the real file, i.e. all but the `src/graph/` binaries that link it) | `fakes/search_fakes.cc` |
| `src/graph/tuning.cc`, `src/graph/connect.cc` params | `fakes/tuning_fakes.cc` |
| everything `src/graph/*.cc` calls outside `src/graph` (the `graph/` binaries except `rccl-UnitTestsMicroGraph{Dev,RcclGraphGen,Trees}`) | `fakes/graph_fakes.cc` |
| `src/group.cc` | `fakes/group_fakes.cc` |
| `src/config/algorithm_*.cc` | compiled real (no fakes file) |
| `src/config/collconfig.cc` (targets that do not compile the real file) | `fakes/collconfig_fakes.cc` |
| `src/debug.cc` (`ncclDebugLog`, `ncclDebugLevelMask`, `ncclSetThreadName`, ...) | `fakes/nccl_fakes.cc` |
| `src/enqueue/enqueue.cc`'s own symbols (targets that do not compile the real file) | `fakes/enqueue_fakes.cc` |
| `src/enqueue/mgmt_task_enq.cc` (targets that do not compile the real file) | `fakes/mgmt_task_enq_fakes.cc` |
| `src/init.cc` comm lifecycle, CTA/channel params, `allocTracker` | `fakes/comm_fakes.cc`; `rccl-UnitTestsMicroRmaProxy` keeps a local `allocTracker` because that isolated target does not link the broader comm fake |
| `src/init_nvtx.cc` | `fakes/init_nvtx_fakes.cc` |
| `src/mem_manager.cc` | `fakes/mem_manager_fakes.cc`; `ncclMemTrack*` / `ncclMemUntrack*` in `fakes/nccl_fakes.cc` |
| externals `src/mem_manager.cc` reaches (`rccl-UnitTestsMicroMemManager`, which compiles it) | `fakes/mem_manager_micro_fakes.cc` |
| `src/misc/amdsmi_wrap.cc` | `fakes/amdsmi_fakes.cc` |
| `src/misc/api_trace.cc` (`NCCL_API` dispatch) | `fakes/api_trace_fakes.cc` |
| `src/misc/argcheck.cc` | `fakes/argcheck_fakes.cc` |
| `src/misc/gdr_peermem.cc` | `fakes/gdr_peermem_fakes.cc` |
| `src/misc/gdr_probe.cc` (targets that do not compile the real file) | `fakes/gdr_probe_fakes.cc` |
| file-static RMA helpers in `src/include/gdrwrap.h` | `fakes/gdrwrap_fakes.{h,cc}` via `fakes/gdrwrap_seam.h` / `fakes/gdrwrap_seam_undef.h` |
| HSA runtime diagnostics | `fakes/hsa_fakes.cc` |
| `src/misc/ibvwrap.cc` (verbs wrappers) | `fakes/ibvwrap_fakes.cc` |
| libibverbs provider ops behind the `ibvwrap.h` inline post/poll wrappers | `IbVerbsScene.h` (one fake device per test) |
| `src/misc/ionicdvwrap.cc` | `fakes/ionicdvwrap_fakes.cc` |
| ibverbs provider dv libraries (`libionic.so`, `libmlx5.so`) | `fakes/dv_provider_fake.cc` |
| `src/misc/kernel_config.cc` | `fakes/kernel_config_fakes.cc` |
| `src/misc/mlx5dvwrap.cc` | `fakes/mlx5dvwrap_fakes.cc` |
| `src/misc/param.cc` + `getenv` interposition | `fakes/env_fakes.cc` |
| `src/misc/rocmwrap.cc` | `fakes/rocmwrap_fakes.cc` |
| `src/misc/roctx.cc` (`roctx_scoped_range_in`, recorded) | `fakes/roctx_fakes.cc` |
| `src/misc/socket.cc` | `fakes/socket_fakes.cc` |
| `src/misc/strongstream.cc` | `fakes/strongstream_stubs.cc` |
| `src/misc/utils.cc` | `fakes/utils_fakes.cc` |
| `src/nccl_device/core.cc` host `ncclTeam*` accessors | `fakes/nccl_device_core_fakes.cc` (except `ncclTeamLsa`; see below) |
| `src/os/linux.cc` (including socket close) | `fakes/os_fakes.cc` |
| `src/plugin/env.cc` | `fakes/env_plugin_fakes.cc` |
| `src/plugin/profiler.cc` (`ncclProfilerProxyTraceDumpIfAny`, recorded) | `fakes/profiler_fakes.cc` |
| `src/plugin/gin.cc`, `src/gin/gin_host.cc` (targets that do not compile the real file) | `fakes/gin_fakes.cc` |
| `src/plugin/profiler.cc` (`ncclProfilerCallback`, until it has its own fakes file) | `fakes/nccl_fakes.cc` |
| `src/proxy.cc` | `fakes/proxy_fakes.cc` |
| externals `src/proxy.cc` reaches (`rccl-UnitTestsMicroProxy`, which compiles it) | `fakes/proxy_micro_fakes.cc` |
| `src/ras/ras_param.cc` (seconds and nanoseconds timeout scaling) | `fakes/ras_param_fakes.{h,cc}` |
| `src/rccl_arch_thresholds.cc` | `fakes/rccl_arch_thresholds_fakes.cc` |
| `src/rccl_wrap.cc`'s own public entry points (targets that don't compile the real file, e.g. `rccl-UnitTestsMicroEnqueue`) | `fakes/rccl_wrap_fakes.cc` |
| `src/rccl_wrap.cc`'s dependencies (`rccl-UnitTestsMicroWrap`, which compiles the real file and tests it directly) | `fakes/wrap_fakes.cc` |
| `src/recorder.cc` | `fakes/recorder_fakes.cc` |
| `src/register/*.cc` (targets that do not compile the real files) | `fakes/register_stubs.cc` |
| `src/register/register.cc`'s `ncclParamLocalRegister` (`rccl-UnitTestsMicroRegister{CollReg,SendrecvReg}`) | `fakes/register_fakes.cc` |
| `src/rma/*.cc` | `fakes/rma_fakes.cc` |
| `src/scheduler/*.cc`'s own public entry points (targets that don't compile the real files, e.g. `rccl-UnitTestsMicroEnqueue`) and the deep launch paths | `fakes/sched_stubs.cc` |
| `src/sym_kernels.cc` | `fakes/sym_kernels_fakes.cc` |
| `src/transport/net_ib/common.cc` (`ncclNetIb`, targets that do not compile the real file) | `fakes/transport_net_ib_common_fakes.cc` |
| `src/transport/net_ib/connect.cc` (`ncclParamIbQpsPerConn`, `ncclIbGidInfoQuery`, GDR flush-flag support and enable; listen, connect, accept and close fail loudly) | `fakes/transport_net_ib_fakes.{h,cc}` |
| `src/transport/net_ib/connect.cc`'s `rcclNetP2pPolicy` and GDR flush-flag accessors (targets that do not compile the real file) | `fakes/transport_net_ib_connect_net_fakes.cc` |
| `src/transport/net_ib/p2p.cc` | `fakes/transport_net_ib_p2p_fakes.{h,cc}` |
| `src/transport/net_ib/p2p_resiliency.cc` | `fakes/transport_net_ib_resiliency_fakes.{h,cc}` |
| `src/transport/net_ib/p2p_resiliency_recovery.cc` | `fakes/transport_net_ib_recovery_fakes.{h,cc}` |
| `src/transport/net_ib/wqe_lat_mon.cc` | `fakes/transport_net_ib_wqe_lat_fakes.{h,cc}` |
| `src/transport/net_ib_cast/capability.cc` | `fakes/net_ib_cast_capability_fakes.cc` |
| `src/transport/net_ib_cast/common.cc` (targets that do not compile the real file) | `fakes/net_ib_cast_common_fakes.cc` |
| `src/transport/net_ib_cast/{connect,gdr,multiplane,p2p,p2p_resiliency,p2p_resiliency_recovery,qp_sharing,scheduler}.cc` (targets that do not compile the real file) | `fakes/net_ib_cast_<file>_fakes.cc` |
| `src/transport/net_ib_cast/init.cc` (targets that do not compile the real file) | `fakes/net_ib_cast_init_fakes.cc` |
| `src/transport/net_ib_cast/net_ib_ops_fault.cc` | `fakes/net_ib_ops_fault_fakes.cc` |
| `src/transport/net_socket.cc` (targets that do not compile the real file) | `fakes/transport_net_socket_fakes.{h,cc}` |
| `src/transport/net_socket/common.cc` (targets that do not compile the real file) | `fakes/transport_net_socket_common_fakes.{h,cc}` |
| generated `sym_kernels_host.cc` (targets that do not compile it) | `fakes/sym_kernels_index_fakes.cc` |
| rocSHMEM host API | `fakes/rocshmem_fakes.cc` |
| `src/transport/p2p.cc` shareable-buffer entry points (`rccl-UnitTestsMicroDiagnosticsP2p`, `rccl-UnitTestsMicroTransportNet`) | `fakes/transport_p2p_fakes.cc` |
| `src/transport/p2p.cc` registration entry points, macro-renamed via `fakes/transport_p2p_reg_seam.h` (the `rccl-UnitTestsMicroRegister*` binaries) | `fakes/transport_p2p_reg_fakes.cc` |
| `src/plugin/plugin_open.cc` library registry and `src/plugin/net/net_v*.cc` getters (targets that do not compile the real files) | `fakes/plugin_stubs.cc` |
| selected entry points from `src/{transport,proxy}.cc`, `src/transport/{generic,coll_net,net,nvls}.cc`, `src/{graph/paths,tuning/tuning,plugin/net}.cc` (targets that do not compile the real files) | `fakes/transport_stubs.cc` |
| libc (`gethostname`, `dladdr`), process-wide for units linking libc directly | `fakes/libc_interposers.cc` |
| libc macro-renamed in one unit via `fakes/libc_seam.h` (sockets/stdio/exit for `ras/client.cc` and `ras/client_support.cc`; `gethostname`/`access` for `diagnostics/ib_write_bw.cc`; heap for `graph/rccl_graph_gen.cc`; files, `ioctl`, `atexit`, clock, aligned allocation, thread start and sleep for `transport/net_telemetry.cc`; `opendir` for `misc/gdr_peermem.cc`; `open`/`read`/`close` for `transport/net_socket.cc`; `clock_gettime` for `transport/net_ib/wqe_lat_mon.cc` and `transport/net_ib_cast/scheduler.cc`; the OS surface of `os/linux*.cc` and `misc/{socket,signals}.cc`) | `fakes/libc_fakes.cc` |
| libdl macro-renamed in one unit via `fakes/dl_seam.h` (the `misc/*symbols.cc` builders and `os/linux.cc`) | `fakes/dl_fakes.cc` |
| `src/init.cc`'s `SetCpuStackSize` param (`rccl-UnitTestsMicroOs<Unit>`, which cannot link `init_fakes.cc`) | `fakes/linux_fakes.cc` |
| reusable RAS poll-entry, message, and diagnostics-context seams | `fakes/ras_fakes.cc` |
| `src/ras/ras.cc` communicator registry (`ncclComms`, `nNcclComms`, `ncclCommsMutex`, `ncclCommsSorted`) | `fakes/ras_comms_fakes.cc` |
| core/lifecycle floor + data symbols | `fakes/nccl_stubs.cc` |
| reusable `nccl*` seams | `fakes/nccl_fakes.cc` |
| HIP runtime | `fakes/hip_fakes.cc` |

**`_fakes.cc` versus `_stubs.cc`.** The suffix records what the file mostly *is*, not a rule the
build enforces: `_fakes` for a file whose point is controllable seams, `_stubs` for one whose point
is a fail-loud floor. Several files are honestly both — `transport_stubs.cc` is a floor that also
owns one driven seam, and `os_fakes.cc` holds working implementations with no seam at all. Do not
read the suffix as a guarantee; read the file's header comment, which states what it owns. If you
add a file, pick the suffix matching its majority content and say so at the top.

**Recorders a production thread can reach.** Most fakes are touched only by the test's own thread,
so their counters and recording vectors are plain `int`s and `std::vector`s. A fake reached from a
thread the unit under test spawns needs more: `os_fakes.cc`'s affinity pair is the current example
(`gin_host.cc`'s proxy progress workers call `ncclOsCpuCount` / `ncclOsSetAffinity` from their own
threads), so its counter is a `std::atomic<int>` and its mask vectors are written under
`g_ncclOsAffinityMutex` and read through `ncclOs*MasksSnapshot()`. If you seam something a
spawned thread reaches, do the same, and keep a direct-read path for the single-threaded suites
rather than churning them. Note that joining the worker and reading safely are two different
problems: the join is what makes the write have happened, the snapshot is what makes the read
safe.

**`// UNDRIVEN`.** A seam carrying this marker is declared so the binary links and so an accidental
call is visible, NOT because its path is covered. It is a link-floor entry with a chosen default,
and that default silently selects which production arm runs. Driving a seam means deleting its
marker. The marker travels with the declaration rather than a block comment so it cannot drift from
what it describes. Call *counters* do not take the marker unless the counter itself is unread.

These do NOT follow the TU-per-file rule, deliberately:

- `fakes/collective_stubs.cc` is a fail-loud floor for the collective *launch*
  pipeline (`ncclLaunchKernel` and friends), which `enqueue.cc` itself defines.
  It therefore cannot link into the enqueue target and stays target-shaped.
  One symbol in it, `ncclArgsGlobalCheck`, is a controllable hook
  (`g_ncclArgsGlobalCheck` in `fakes/collective_stubs.h`) rather than a hard
  `::abort()`, for `group-test.cc`'s `ArgsGlobalCheckTest`; everything else in
  the file is still the same fail-loud floor described above.
- `ncclStrongStreamAcquire` / `Release` stay in `nccl_fakes.cc` rather than
  `strongstream_stubs.cc`: they carry `ASSERT_HOOK_MATCHES_PROD` drift
  assertions and moving those is a larger change.
- `transport_stubs.cc` and `collective_stubs.cc` both provide several fail-loud
  collective-transport setup symbols. `rccl-UnitTestsMicroGroup` needs both floors,
  so `RCCL_TRANSPORT_STUBS_OMIT_COLLECTIVE_FLOOR` omits the transport copy in
  that target to avoid fakes-versus-fakes duplicate definitions.
  `ncclTransportP2pSetup` is no longer one of them: it is a single seam in
  `transport_stubs.cc`, the file that owns the subsystem. `ncclCeInit` moved the
  same way, to `ce_fakes.cc`.
- `rcclParamIntraGraphGen` stays in `fakes/init_fakes.cc` because its owner
  (`graph/rccl_graph_gen.cc:34`) has no fakes file at all. `rcclEffectiveP2pBatchEnable`
  did have one and moved to `fakes/enqueue_fakes.cc`.
- `ncclTeamLsa` stays in `fakes/devcomm_fakes.cc` rather than moving to
  `fakes/nccl_device_core_fakes.cc` with the other `src/nccl_device/core.cc`
  accessors: its default silently satisfies the `nRanks ==
  comm->nRanks` comparison every `devcomm/devcomm-test.cc` filter makes, and the trap
  comment warning about that is what keeps those tests honest. Moving the hook
  without moving the readers that depend on the warning is how it would get lost.

An aggregation header includes the per-TU headers a unit's tests use and
declares the reset that chains their per-TU resets, and defines no seams itself.
`enqueue_test_deps.h` is the only one: it is named for that role rather than
`<uut>_fakes.h`, because `enqueue_fakes.h` now holds `enqueue.cc`'s own seams.

**`RCCL_STUBS_OMIT_<symbol>` is only for a symbol the unit under test defines**,
where the omission exists purely to avoid a duplicate at link time. If a target
instead needs a real *value* where the shared floor aborts, that symbol wants a
seam in its owning TU's fakes file, which serves every target at once. Reaching
for an omit macro plus a private replacement file is how `rcclUseAinic` ended up
faked in two places.

## Adding more controllable seams

When a test needs to drive a fake that still returns a constant, turn it
into a hook. `g_proxyCallBlocking` in `fakes/nccl_fakes.cc` is an example:

1. In the fakes file that owns that module's seams (`fakes/nccl_fakes.cc`
   for `nccl*` symbols, `fakes/hip_fakes.cc` for HIP runtime symbols), add
   a `std::function`-typed hook with a default that matches the current
   constant behaviour:
   ```cpp
   static ncclResult_t DefaultProxyCallBlocking(ncclComm*, ncclProxyConnector*, int,
                                                void*, int, void*, int) {
       return ncclSystemError;
   }
   std::function<ncclResult_t(ncclComm*, ncclProxyConnector*, int,
                              void*, int, void*, int)>
       g_proxyCallBlocking = DefaultProxyCallBlocking;

   ncclResult_t ncclProxyCallBlocking(ncclComm* c, ncclProxyConnector* p,
                                      int t, void* req, int rs,
                                      void* resp, int rsz) {
       return g_proxyCallBlocking(c, p, t, req, rs, resp, rsz);
   }
   ```
2. Expose the hook from the matching header (`fakes/nccl_fakes.h`,
   `fakes/hip_fakes.h`) so tests can install per-test behaviour in a
   gtest fixture's `SetUp` / `TearDown`.
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
`ForceLegacyCudaRegister` in `transport/p2p-test.cc` +
`ForceLegacyIpcCapable` in `fakes/hip_fakes.h`.

## Shared test-side helpers

The scaffolding a test needs on *its* side of the seam — scripting a fake's
failure, waiting for a worker thread, cleaning up on the fatal-assert path — is
the same in every binary, so it lives in a header next to the tests rather than
being re-invented per TU. Each is one small header named after the type it
declares:

| Header | Use it for |
|---|---|
| `ScopedHook.h` | Install behaviour on a `std::function` seam, count the calls, restore the previous behaviour on scope exit. See [Installing per-test behaviour with `ScopedHook`](#installing-per-test-behaviour-with-scopedhook). |
| `ScriptedFailure.h` | "Fail call N of this entry point." A fake holds one per entry point and asks `at(++calls)` what to return; the test writes `fake_.failConnect = {ncclSystemError, 2}`. Replaces the `g_fooCalls` + `g_failFooCall` pair that otherwise grows per seam. |
| `WaitUntil.h` | `waitUntil(pred)` — bounded poll for something another thread will do, returning false when the budget elapses so the test fails with its own message instead of hanging. Never `sleep_for` and hope. |
| `JoinThreadsOnExit.h` | Spawn worker threads and stop/join them on every exit from the scope. The stop action is a callback, because what ends the worker's loop belongs to the unit under test. |
| `ScopeExit.h` | The one-off cleanup that has no purpose-built RAII type. A fatal `ASSERT_*` returns immediately, so cleanup written after it does not run. |

When a test grows a helper that is not specific to its unit under test, promote
it here rather than leaving it in the TU — a per-TU copy is how four tests end
up with four slightly different "wait for the thread" loops. Helpers that *are*
specific to one unit stay with it (`enqueue/task_prep/TaskPrepScene.h` for the task-prep scene,
`fakes/ras_*_test_support.h` for the RAS fixtures).

## Dealing with each kind of dependency

When the link fails with `undefined symbol: foo`, find `foo` and
triage it into the right bucket:

- **It's a global variable (`extern int foo;`)** → add a definition
  to its owning TU's fakes file (e.g. `allocTracker`, from `src/init.cc`, is in
  `fakes/comm_fakes.cc`). If it has no owning TU, define it in the test. Use a
  sensible default (usually zero).
- **It's a plain function the module references but doesn't define**
  → add a definition returning a sensible default to its owning TU's
  fakes file (e.g. `busIdToInt64` / `getBusId` go in
  `fakes/utils_fakes.cc`, since `src/misc/utils.cc` owns them). Only
  when the symbol has no owning TU does it belong in the test itself.
- **It's a logging or env-param helper** → already covered by
  `ncclDebugLog` (prints to stderr; a test that asserts on what was logged
  hooks `g_ncclDebugLog`) / `ncclLoadParam` (parses the value
  `SetMicroEnv` scripted for that name, and the param's compile-time default
  when the name is unmapped). If a new logging primitive
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
  - **`rccl-UnitTestsMicro*`** deliberately do **not** link `hip::host`,
    `libamdhip64.so`, or `librccl.so`. For these targets, add a fake or
    hookable seam in `fakes/` with the exact signature the HIP headers
    declare — do **not** add `hip::host` merely to resolve an undefined
    symbol. An unresolved HIP symbol is precisely the mechanism that
    surfaces an unfaked dependency.

  Either way, RCCL's `cuMem*` / `cuPointer*` calls are hipified to
  `hipMem*` / `hipPointer*` and called directly, so under
  the `rccl-UnitTestsMicro*` binaries each needs an explicit definition in
  `fakes/hip_fakes.cc`: use the signature the header declares and return a
  failure code (or a canned success) by default.
- **It's a HIP kernel launch** → you almost certainly don't want to
  test the path that launches it from this binary. Refactor the test
  to avoid the branch, or split the kernel-launching code into a
  function that can itself be stubbed.


## Coverage

`rccl-UnitTestsMicroTransportP2p` builds with llvm source-based coverage
(`-fprofile-instr-generate -fcoverage-mapping`) only when coverage is requested
with `-DENABLE_CODE_COVERAGE=ON`. Render a report with ROCm's
llvm tooling directly. Scope it to the unit under test -- the file compiled in
via `P2P_CC_PATH`, i.e. the unroll-transformed
`hipify/src/transport/p2p_tmp.cc` (there is no plain `p2p.cc` in the hipify
tree; scoping to a non-existent file makes `llvm-cov` silently fall back to
whole-binary totals).

```bash
BD=build/release                        # or build/debug, or a standalone build dir
BIN=$BD/test/host/rccl-UnitTestsMicroTransportP2p
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
`make`-only inner loop for every subsequent edit to
`transport/p2p-test.cc` or the `fakes/*.cc` it links against.

### Initial (one-time) build

Local-arch (`-l`), with tests (`-t`):

```bash
./install.sh -l -t -j $(nproc)
```

### Tight inner loop (after editing a test or a fake)

```bash
cd build/release
make -j $(nproc) rccl-UnitTestsMicroTransportP2p
```

### Run

```bash
# All tests:
./build/release/test/host/rccl-UnitTestsMicroTransportP2p

# One test:
./build/release/test/host/rccl-UnitTestsMicroTransportP2p \
    --gtest_filter='P2pMicrotest.IpcRegisterBuffer_NullRegRecordIsNoOp'

# Coverage: see the Coverage section (run under LLVM_PROFILE_FILE, then llvm-cov).
```

### `rccl-UnitTestsHostAll` — one front door for every host binary

The host tree builds many small binaries on purpose (see
[Why a separate test binary](#why-a-separate-test-binary)), and every new unit
under test tends to add another. Anything outside this directory that wants to
*run* them therefore ends up keeping its own copy of the binary list, and those
copies drift: a new binary silently goes unrun, and its coverage silently goes
unreported.

`rccl-UnitTestsHostAll` (`host_test_dispatch.cc`) removes the lists. It is not
a test binary; it is a gtest-shaped front door that discovers its
`rccl-UnitTestsMicro*` siblings **at run time** and forks the ones that own a
selected case:

```bash
# Everything, as one report:
./build/release/test/host/rccl-UnitTestsHostAll --gtest_output=xml:host.xml

# What it can run (the union of every binary's cases):
./build/release/test/host/rccl-UnitTestsHostAll --gtest_list_tests

# One suite, wherever it lives:
./build/release/test/host/rccl-UnitTestsHostAll --gtest_filter='*RmaCeInitTest.*'
```

Case names are qualified with the owning binary's suffix — 
`MicroInitUncached/InitMicrotest.Case` — because the multi-arm binaries
(`-uncached`, `-faultinj`, `-devlinker`, `-roctx`) publish identical
`Suite.Case` names and would otherwise collide into one identity when they all
report through one front door. The dashes are folded away (`MicroInitUncached`,
not `MicroInit-uncached`) because gtest splits a `--gtest_filter` at its first
`-`, so a qualifier carrying one could never be named by a positive pattern.

The qualifier is part of the name a filter has to match, so a pattern written
against one binary does **not** carry over unchanged: `P2pMicrotest.*` (as in
`test_categories_micro.yaml`) matches nothing here, because every published name
begins `<Qualifier>/`. Lead with `*` to mean "in whichever binary owns it", or
name the qualifier to mean one arm:

```bash
# Wherever it lives:
--gtest_filter='*P2pMicrotest.*'
# Only the uncached arm's copy:
--gtest_filter='MicroInitUncached/InitMicrotest.*'
```

A filter that selects nothing is not an error — gtest exits 0 for one too, and
the shared categories yaml depends on that, since each binary is handed every
category's patterns and most match none of them. The front door prints the
qualification rule when it happens, so a pattern that silently selected nothing
says why.

The filter itself is resolved by **gtest**, not by the dispatcher: it registers
the qualified names as empty tests and reads back `should_run()`. gtest exposes
no matcher any other way (`UnitTestOptions::FilterMatchesTest` is declared only
in `src/gtest-internal-inl.h`, which is not installed), and a reimplementation
drifts — the first one here selected `DISABLED_` cases that gtest would not have
run.

Each child is then handed an **exact `:`-joined list of its own case names**,
never the caller's pattern: the pattern is written against qualified names, and
in a multi-arm binary (`rccl-UnitTestsMicroInit*`) it would also match the
identically named cases of the arm it was meant to exclude. Two exceptions keep
that list manageable — when every case a binary could run is selected it is sent
`*` instead, which is shorter in the logs and is why its `DISABLED_` cases do
not force a list; and a list longer than ~60,000 characters is split across
several invocations of that binary, because the kernel caps one argv entry at
`MAX_ARG_STRLEN` (128 KiB) and rejects a longer `exec` outright with `E2BIG`,
which the biggest binary's 2,199 case names would exceed.

### What the children inherit

gtest configures itself from `GTEST_*` variables as well as from flags, and that
is dangerous in a dispatcher, because this process reads them too. Every such
variable is one of two things:

- **Consumed** — the dispatcher acts on it and re-expresses the decision as an
  explicit per-child flag, so the children never see the variable. A child that
  saw it would decide the same question a *second* time, over a case list that
  has already been narrowed, and exit 0 having run a fraction of it.
  `GTEST_FILTER` (resolved over the qualified names), `GTEST_OUTPUT` (becomes
  the merged report), `GTEST_ALSO_RUN_DISABLED_TESTS` (decides the selection)
  and the three `GTEST_*SHARD*` variables (sharding happens once) are consumed.
- **Forwarded** — it means the same thing in a child and changes nothing about
  which cases run: `GTEST_COLOR`, `GTEST_BRIEF`, `GTEST_REPEAT`,
  `GTEST_DEATH_TEST_STYLE` and the like.

An **unknown** `GTEST_*` variable is consumed and named on stderr. That is the
safe default of the two: a variable a future gtest adds costs a redundant strip
and a line of output, where inheriting it could silently cost coverage. Nothing
here has to be updated for gtest to stay *correct* — the table in
`host_test_dispatch.cc` only buys back the forwarding of variables we have
decided are harmless. Non-`GTEST_` variables are never touched.

The strip happens **in each child, between `fork()` and `exec()`**, not in the
dispatcher. Two reasons: the dispatcher goes on using the originals after the
first fork (it shards over them), and putting it at the fork sites means there
is no ordering in `main()` to get right and a fork site added later inherits the
guarantee by construction.

For coverage, each child inherits `LLVM_PROFILE_FILE` and drops its own
`.profraw` as usual. What a harness cannot know is which binaries those
profiles came from — `llvm-cov` needs an `--object` per binary and silently
discards a profile whose object is missing — so the dispatcher appends every
binary it ran to `coverage-objects.txt` in the profile directory. That file,
not a list maintained in the harness, is the object set.

Adding a host test binary therefore needs no change outside this directory.

`run_host_tests.sh` keeps driving the binaries individually: it writes each
one's profiles into its own `coverage/<binary>/` subdirectory to produce the
per-binary reports CI publishes, which a single shared profile directory could
not reconstruct.

## Standalone host-only build (no full librccl build)

`test/host/CMakeLists.txt` is dual-mode. Alongside the in-RCCL-build target
above (`./install.sh -t`, wired via `add_subdirectory(host)`), the same file
can be configured **directly** to build every host binary (`host_test_binaries.txt` in the
build directory lists them) — **without configuring/building all of
librccl**. It compiles just the tests + fakes + the hipified unit-under-test
sources.

The `-uncached`, `-faultinj`, `-devlinker` and `-roctx` binaries are preprocessor
variants, not duplicates. `init.cc` gates
part of its allocation path on `HIP_*_UNCACHED_MEMORY` and its fault-mask blocks
on `ENABLE_FAULT_INJECTION`, and `enqueue.cc` gates `rcclShmemDynamicSize` on
`RCCL_DEVICE_LINKER`, and `roctx.cc` gates roctracer on `ROCTX_ENABLE`, all at
the **preprocessor**, so one compile can only ever reach one arm. The in-RCCL-build path inherits `RCCL_DEVICE_LINKER` from the
`rccl` target's compile definitions (`ENABLE_DEVICE_LINKER` defaults ON, so the
device-linker arm is the one that ships); this standalone project has no `rccl`
target to inherit from, which is why it builds the `-devlinker` variant
explicitly, and likewise `-roctx`. `ENABLE_FAULT_INJECTION` was the same inheritance in reverse -
defined in-tree, absent standalone, so the two paths tested different code - and
is now stripped from the inherited list and set per variant in both paths, so the
`-faultinj` binary is the only one that has it either way.

**ROCm is a prerequisite.** Per epic AICOMRCCL-1661 ("ROCm toolchain is
available"), this build uses `hipcc` in host-only mode (`--offload-host-only`)
against the **real ROCm headers**. There is no CPU-only / g++ path and no stubbed
`<hip/*>` / `<hsa/*>` / `<cuda*>` headers. It links **gtest, fmt and pthread** and
passes `-no-hip-rt`, so it links **neither `librccl.so` nor the HIP runtime** —
every HIP symbol the tests reach is provided by `fakes/` (and, for
`rccl-HostUnitTests`, `wrapper_link_stubs.cpp` / `hip_prop_r0600_stub.cpp`).

```bash
cd projects/rccl/test/host
cmake -B build -DCMAKE_BUILD_TYPE=Release \
      -DRCCL_BUILD_DIR=/path/to/projects/rccl/build/release
cmake --build build -j"$(nproc)"
./build/rccl-UnitTestsMicroBootstrap   # one binary per test file; build/host_test_binaries.txt lists them
```

Coverage instrumentation for the standalone host-only test binaries is OFF by
default; enable it with `-DENABLE_CODE_COVERAGE=ON` (`run_host_tests.sh
configure` does this).


### Peers review follow-up

The peers suite is `ras/ras-peers-test.cc`, with the `RAS_PEERS_CC_PATH` source
definition. `fakes/ras_net_fakes.{h,cc}` owns connection lookup/creation,
disconnect, and fallback hooks. `fakes/ras_client_support_fakes.{h,cc}` owns
event notification and synthetic host/device rendering. Both the peers and
RAS fixtures reset/configure the shared hooks, while the RAS unit's real
message implementation stays separate from the peers message double in
`fakes/ras_message_fakes.{h,cc}`. Explicit `RasMessageTest*` wrapper names
avoid colliding with the existing `RasTestMsg*` adapter API in other PRs.

`fakes/ras_message_test_support.h` provides move-only message ownership;
callers supply the allocation/free pair matching their unit. Received peers
updates and RAS send-queue fixtures use it so fatal assertions cannot leak
messages. Rank sorting, merging, insertion, duplicate/empty filtering, and
device-bit diffs are exercised through `rasLocalHandleAddRanks`, which owns
its incoming rank array. Tests assert the resulting registry/hash and sent
diffs rather than a temporary conversion buffer. The private debug formatter
has a separate test; synthetic `<host>`/`<devs>` output prevents these tests
from accidentally asserting a copy of client-support formatting logic.

The dead-only and live-only update cases independently pin checksum state
when one array is omitted. Fallback tests find a healthy remote process after
an unusable process on the same node, and previous-link failure uses three
peers so the next and previous destinations differ.

Review follow-up validation: 56 peers tests and 71 RAS tests pass together
for 20 shuffled iterations. The peers translation unit and the new shared
fakes are checked with ASan and UBSan; the RAS translation unit uses ASan
only because its existing unknown-message fixture contains an out-of-range
enum value. All 56 peers tests also pass with leak detection enabled.
Three fresh targeted mutations are killed: removing either sent-hash guard
or making the remote fallback-index update unconditional. The mutation
baseline passes. These are focused local runs, not a fresh full-binary CI
run or coverage measurement.

### Shared RAS diagnostic fixtures

`fakes/ras_registry_test_support.h` owns test installation/reset of the
`ras.cc` communicator registry. Both operations hold `ncclCommsMutex` and
clear `ncclCommsSorted`; installation validates allocation before replacing
the old registry and preserves explicit vacant slots. The RAS and diagnostics
suites use the same helper. `fakes/ras_diagnostics_test_support.h` shares the
owned communicator and recording reporter used by the diagnostic suites.

The GPU diagnostic suite exercises payload filling through the public collectors
and owns returned records with RAII. Each summarizer has an over-capacity rank
case; together they exercise all seven bounded rank arrays and check truncated
rank sets with total counts. ECC maxima are checked with the largest values on
an earlier rank, and incomplete-report fixtures include hexadecimal hashes
larger than nine.

`SetMicroEnviron` in `fakes/env_fakes.{h,cc}` replaces the scripted environment
for enumeration tests, preserving input order and malformed entries while
making valid entries visible through `micro_getenv` and `getenv`. While that
override is active, `SetMicroEnv` and `SetMicroEnvAbsent` also update `environ`.
`ClearMicroEnv`/`ResetEnvFakes` restore the original process environment and
clear the lookup map. Map-only tests retain the existing fallback behavior
unless they opt into the enumeration override.

`DISABLED_Summarize_HashCollisionsReportFullCommIdentity` tracks AICOMRCCL-2743.
The environment reporters currently omit host and process hashes, so distinct
communicators can appear identical. The disabled regression specifies the full
identity for consistent, mismatch, and mismatch-summary reports. Enable it
with the separate production fix; enabled tests still pin correct grouping.
