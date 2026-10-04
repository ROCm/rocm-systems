# Design: faking external interfaces in amd-smi C++ unit tests

This document works through one concrete case — RAS block-enabled state and the
`ras/features` sysfs read — as the first application of a general, reusable,
two-layer pattern for testing amd-smi's real public API without real hardware.

## Motivation

`amdsmi_get_gpu_total_ecc_count()` had three bugs: it re-read RAS state up to 39
times per call (one read of `.../ras/features` per RAS block), which could trip
host-side RAS read throttling on SR-IOV guests; a failed mask read was silently
reported as a zero-error GPU instead of a real failure; and the per-block
accumulation loop only ever added to `*ec` (`+=`), so a caller passing in a
struct that wasn't already zeroed (or reused across calls) got a result
contaminated by whatever was already there, even on `AMDSMI_STATUS_SUCCESS`.
Fixing this exposed a broader gap: there was no way to unit-test amd-smi's public
API surface without real GPU hardware. This document is both the fix's regression
coverage and the general pattern for closing that gap elsewhere.

## Design goals

- Tests call the **real, unmodified public API** (e.g. `amdsmi_get_gpu_total_ecc_count()`)
  directly wherever possible, not an internal helper one layer down — the closer a
  test drives the actual entry point users call, the more of the real code path it
  covers.
- No environment-variable overrides. Any test-only redirection of paths or state
  must not live in a form that's checked in always-shipped, often-privileged
  production code. See "Why this is safe to ship" below for the mechanism actually
  used instead.
- Every override is RAII-guarded, never a bare "set and forget" setter — the only
  way to change any faked value is to construct a guard, which always restores the
  real value on scope exit, even on early test failure (`gtest`'s `ASSERT_*` return
  early rather than throwing, so destructors still run). This removes the "forgot
  to reset in TearDown" bug class by construction.
- The pattern generalizes across amd-smi's processor types (GPU, CPU, NIC, switch,
  AI-NIC), not just this one RAS fix. See "The two-layer pattern" below.

## The two-layer pattern

Every amd-smi public API call reaches its target processor through the same
dispatch: `get_gpu_device_from_handle()` (or its CPU/NIC/switch equivalents) calls
`AMDSmiSystem::handle_to_processor()`, which looks up the handle by pointer identity
in one of `AMDSmiSystem`'s private sets (`processors_`, `nic_processors_`,
`switch_processors_`, `ainic_processors_`). This holds **regardless of processor
type** — CPU sockets/cores are inserted into the exact same `processors_` set as
GPUs.

That gives a clean two-layer split for any future device-type test:

1. **Universal layer — build once, reuse everywhere.** A single seam on
   `AMDSmiSystem` registers/unregisters a synthetic processor into the correct set
   for the scope of a test (RAII-guarded, per the design goals above). The correct
   set is chosen from the processor's own `get_processor_type()` -- the same
   `amdsmi_processor_type_t` value the processor was already constructed with, not
   a separate parameter the test has to supply and keep in sync. Any future CPU,
   NIC, switch, or AI-NIC unit test reuses this exact mechanism with zero new
   registration code.
2. **Backend-specific layer — investigate per device type.** Once a fake processor
   is reachable through the public dispatch, whatever that processor's own methods
   touch underneath varies by type, and needs its own short investigation before
   assuming a seam is needed (or already exists):
   - **GPU**: `AMDSmiGPUDevice::get_mutex()` resolves through
     `RocmSMI::getInstance().devices()[gpu_id_]`. That vector is already public and
     mutable (`RocmSMI::devices()`), and `rocm_smi::Device`'s constructor is
     confirmed to only create a real named shared-memory mutex — no other hardware
     access — so a test can push a real, hardware-inert fake `Device` there
     directly. **No new seam was needed for this part at all.** Sysfs reads below
     that (like `ras/features`) still need their own small override — see "The RAS
     sysfs seam" below. The same reasoning extends to per-block RAS/ECC counter
     reads (`rsmi_dev_ecc_count_get()` -> `Device::openSysfsFileStream()`), which
     also build their path from the same `Device::path_` field a test already
     controls directly, rather than a separate hardcoded literal.
   - **CPU**: confirmed `AMDSmiProcessor` (the plain base class CPU processors use
     directly) has no `get_mutex()` at all, so CPU never hits the mutex/`RocmSMI`
     layer. Its data instead comes from ESMI (`esmi_init()`/`esmi_*` calls, a
     separate library for AMD CPU/HSMP telemetry). A future CPU unit test would
     need its own ESMI-call seam (same getter/setter-behind-a-guard shape as the
     RAS one below) — not something this work builds automatically.
   - **NIC / switch / AI-NIC**: already has its own seam,
     `nic_set_info_getters_for_testing()` (a struct-of-function-pointers swap, a
     different shape than the sysfs-root getter/setter used here, but the same
     RAII-guard discipline). One detail worth checking before relying on it for a
     mutex-adjacent test: `AMDSmiNICDevice::get_mutex()` resolves through the same
     `amd::smi::GetMutex()` helper as GPU, which indexes `RocmSMI::devices()` (the
     GPU vector) rather than `nic_devices()`/`switch_devices()`. Not chased down
     further here — flagged for whoever next writes a NIC test that needs a mutex
     to resolve.

## The RAS sysfs seam (this fix's concrete instance of layer 2)

Reusable test infrastructure (not RAS-specific) lives in
`tests/amd_smi_test/test_fixture_utils.h`/`.cc`:
- `FakeSysfsTree`: builds a real, temporary, hardware-inert file tree.
- A generic, header-only `ScopedOverride<T>` template: takes a getter, a setter, and
  a new value. Its constructor saves the current value and installs the new one;
  its destructor restores the saved value. Being fully templated/inline, it never
  crosses the shared-library ABI boundary, so it carries no export/visibility
  concerns of its own — reused by both the layer-1 processor-registration guard and
  this layer-2 sysfs-root guard, and available for any future interface (ioctl,
  libdrm loader, an ESMI seam for CPU, ...) without duplicating guard boilerplate.
- `ScopedProcessorRegistration`: the RAII wrapper around the layer-1
  `AMDSmiSystem::register_processor_for_testing()`/`unregister_processor_for_testing()`
  seam.
- `ScopedRocmSmiDevice` and `ScopedAmdsmiInit`: reusable by any GPU unit test
  needing a fake `rocm_smi::Device` or a no-hardware-discovery `amdsmi_init()`, not
  just this RAS fix.

The one piece that must live in production code (compiled into
`libamd_smi`/`amd_smi_static`, since production code calls the getter) stays in
`amd_smi_test_overrides.h`/`.cc`:
- The RAS-specific primitive: `smi_amdgpu_sysfs_drm_root()` (getter, called by
  production code) and `smi_amdgpu_set_sysfs_drm_root_for_testing()` (setter — only
  ever called by the guard). Both intentionally not `amdsmi_`-prefixed.

Each test uses the `FakeSysfsTree` fixture helper to build a real, temporary,
hardware-inert file tree, and constructs a real `AMDSmiDrm` + `AMDSmiGPUDevice`
pointing at it. Combined with the layer-1 processor-registration guard and the
layer-2 sysfs-root guard, a test can call the real, unmodified public
`amdsmi_get_gpu_total_ecc_count()` directly and observe its actual failure-handling
behavior — the thing this fix changed — with no real hardware involved.

## Why this is safe to ship (verified, not assumed)

- Every setter/registration method introduced by this pattern is not
  `amdsmi_`-prefixed, so `libamd_smi.so`'s linker version script — which only
  exports the `amdsmi_*` symbol glob — excludes it from the shared library's
  dynamic symbol table entirely. No external caller (the `amd-smi` CLI, Python
  `ctypes` bindings, any customer application linking the `.so`) can resolve or
  call it. This is a linker-level guarantee, not a convention — confirmed via the
  explicit comment in `src/CMakeLists.txt` describing which symbols the version
  script's `amdsmi_*` glob does and doesn't match.
- The `amdsmitst` gtest binary does not link the shared `.so` at all — it links a
  separate static archive built from the same sources, never installed/packaged. A
  static archive has no symbol-visibility restriction, so the test binary can call
  these methods directly.
- **Scope boundary, stated plainly**: `unit/` and `functional/` test sources compile
  into that same `amdsmitst` binary today. The real guarantee is "unreachable
  outside the test binary" — not "unreachable from functional/ tests specifically."
  A harder split would need a new build target; out of scope here.

## How to reuse this for a new device type or a new fake interface

1. **Register your processor** using the existing layer-1 seam on `AMDSmiSystem` —
   no new code needed for this step, regardless of processor type.
2. **Find out what your processor's methods actually touch** before assuming you
   need a new seam. Check: does it go through `get_mutex()`/`RocmSMI`? Does it read
   a fixed sysfs path? Does it call into ESMI, a vendor NIC library, or something
   else entirely? The answer differs per type (see the bullets above) — don't
   assume the GPU/RAS answer transfers.
3. **If a seam already exists for that backend** (e.g. `nic_info_getters_t` for
   NIC), reuse it. **If not**, add a minimal getter + setter pair (not
   `amdsmi_`-prefixed) and wrap it with the same `ScopedOverride<T>` template used
   here — don't write a new bespoke guard type.
4. **Write the test against the real public API function**, not an internal
   helper, using real temp-file/in-memory fixtures for whatever your seam
   redirects (see `FakeSysfsTree` for a reusable fixture-directory helper). Keep
   the guard-only discipline: no bare setters called directly from test bodies.
5. **Only fake as much of the call graph as your test needs.** For example, this
   fix's regression only needs the RAS mask read to fail — the early return happens
   before the per-block ECC-count loop runs, so that loop's own backend
   dependencies never needed faking here. Don't over-build fixtures for code paths
   your specific test doesn't reach.

## Monitoring file operations in tests

Some regressions can only be caught by observing *how many times* (or whether)
production code actually touched a file — not just by asserting on the
function's return value. `ScopedFileWatch` (`test_fixture_utils.h`) covers
this: an RAII wrapper around a Linux `inotify(7)` watch on one
already-existing file path (construct it right after
`FakeSysfsTree::WriteFile()`). It tallies operations — `OpenCount()`,
`WriteCount()`, or `Count(mask)` for any other `IN_*` flag — performed by
*any* process on that path, including a forked-and-privilege-dropped child
(see `RAS_FEATURES__PermissionDeniedIsNotSupported`), with **zero changes to
the production code under test**. Check `IsAvailable()` before trusting a
count — it's `false` if the watch couldn't be set up (e.g. the target path
didn't exist yet), and a test should `GTEST_SKIP()` or skip just that
assertion rather than treat a failed-to-construct watch as "zero events."

### Constraint: construct the watch before locking the file down

`inotify_add_watch(2)` requires read permission on the target file only at
construction time, not on every later event (verified empirically: a watch
attached while a file is still readable keeps observing that inode even
after a later `chmod()` revokes read access). `RAS_FEATURES__PermissionDeniedIsNotSupported`
constructs its watch right after `FakeSysfsTree::WriteFile()` but *before*
`FakeSysfsTree::SetPermissions(..., 0000)`, so `IsAvailable()` stays true
regardless of whether the test binary runs as root — keeping the
zero-successful-opens assertion meaningful either way.

### Limitation: inotify can't exactly count back-to-back identical events

Per `man 7 inotify`: "successive identical events are coalesced into a
single event if the older event has not yet been read ... an application
can't use inotify to reliably count file events." The kernel only compares
a new event against the single most-recently-queued (still-unread) entry
for that watch — not the whole queue — so a run of identical events
collapses to one:

```
watch mask = IN_OPEN only:
real events:  OPEN  OPEN  OPEN  OPEN  ...        (39 times)
queue state:  [OPEN] [OPEN] [OPEN] [OPEN]        <- each new OPEN matches the
                                                     last queued OPEN -> merged
final queue:  [OPEN]                             <- only 1 survives
```

This bit `RAS_FEATURES__ReadsMaskExactlyOncePerCall`: watching `IN_OPEN`
alone, `OpenCount()` reported `1` even though the file was genuinely
`open()`'d 39 times in a row (confirmed via `strace -f -e trace=openat`) —
a false negative that would have silently let the regression it guards
against pass.

**Fix:** also watch a second event type that naturally interleaves with the
first (here, `IN_ACCESS` alongside `IN_OPEN`, since each `std::ifstream`
open is followed by a real `read()`). Adjacent queued events then alternate
types and never match, so nothing coalesces:

```
watch mask = IN_OPEN | IN_ACCESS:
real events:  OPEN ACCESS OPEN ACCESS OPEN ACCESS ...   (39 of each)
queue state:  [OPEN] [OPEN,ACCESS] [OPEN,ACCESS,OPEN] ...
              each new entry compared only to the PREVIOUS one -> never
              identical (OPEN != ACCESS) -> nothing merges
final queue:  [OPEN, ACCESS, OPEN, ACCESS, ..., OPEN, ACCESS]  <- all 78 survive
```

`Count(IN_OPEN)` still sums only the `IN_OPEN` entries (39), ignoring
`IN_ACCESS`. Only matters for *exact* repeat-count assertions in a tight
loop; the zero-vs-nonzero checks elsewhere in this file are unaffected
(zero real events still means zero coalesced events).

### Alternatives considered for monitoring file operations

- **A single bespoke global counter** incremented inside the one function
  under test — simplest, but can't separate multiple devices and needs a new
  copy-pasted counter for every future function someone wants to monitor.
- **A path-keyed in-process seam** (a `std::map<path, count>` + a
  `record_...()` call added at each production read/write site) — reusable as
  a *pattern*, and able to count failed attempts (unlike inotify), but still
  requires manually instrumenting every new call site, and its in-process
  counting state doesn't survive `fork()` (a forked child's increments live
  in its own copy-on-write memory, invisible to the parent afterward) —
  a real problem given the fork-based privilege-drop pattern already used in
  this file.
- **A centralized "open" wrapper / dependency-injected stream factory** that
  every read/write site is migrated onto — general once migrated, but a much
  bigger upfront diff, and still bypassable by a future author writing a raw
  `std::ifstream`/`std::ofstream` directly.
- **`stat()`/atime-based detection** — zero source changes, but unreliable:
  `relatime` (the default mount option on most modern Linux, including CI)
  only updates atime once per day or on an mtime change, so it can't reliably
  distinguish one read from many in a fast unit test.
- **`LD_PRELOAD` syscall interposition** — even more fully external than
  inotify, but scopes to the *entire test binary process*, not one test case,
  and depends on libstdc++'s unspecified internal use of libc file APIs; a new
  category of build/test infrastructure this suite has no other precedent for.

inotify avoids all of these: it requires no production-code changes, keys
naturally per-device (one watch per resolved path), and observes operations
performed by any process against the watched inode — including a child that
dropped privileges after `fork()` — since the watch lives in the kernel, not
in any process's memory.

## Explicitly out of scope for this fix

- Migrating the other ~12 `"/sys/class/drm/" + device->get_gpu_path()` call sites in
  `amd_smi_utils.cc` to the sysfs-root seam. Structure supports it; not done now.
- Full success-path coverage of `amdsmi_get_gpu_total_ecc_count()` through the
  public API for every RAS block. One block (UMC) is faked end-to-end —
  `RAS_FEATURES__ValidMaskOverwritesStaleOutput` writes a real `ras/umc_err_count`
  fixture to prove accumulation happens correctly and isn't contaminated by a
  stale/un-zeroed `*ec` — but the other ~20 blocks' `amdsmi_get_gpu_ecc_count()`
  dependencies remain unfaked; not needed beyond that one proof case.
- Actually adding CPU/NIC/switch/AI-NIC unit tests using the layer-1 seam — it's
  built and documented as reusable, but no such tests are added here.
- A dual-language shared fake-sysfs mechanism. A true one would need a second,
  test-only-exported shared library variant never shipped to customers — a
  separate, larger project. `tests/python/functional/gpu/test_ras.py`'s
  `test_get_gpu_total_ecc_count` was tightened to assert specific accepted
  statuses (`expect_status`/`status_sweep`) instead of the blanket
  `Test_API_Per_GPU`, and a `test_get_gpu_total_ecc_count_rejects_non_handle`
  test was added, but both still run against real hardware with no way to inject
  a malformed mask. A `unittest.mock.patch()`-based test that forces the
  underlying ctypes call to fail remains a reasonable cheaper alternative for
  that specific gap, deferred to a follow-up.

## Alternatives considered (brief)

- **Extract a pure `std::istream`-based parser**, testing only the line-parsing
  logic in isolation (matches the existing `smi_amdgpu_parse_od_clk_range`
  convention for pp_od_clk_voltage). Rejected as the primary approach: it exercises
  parsing only, not the real function's path construction, file I/O, or the public
  API's actual failure-handling behavior.
- **Environment-variable override** for the sysfs root. Rejected: the redirect
  logic would need to live in always-shipped production code, checked on every
  real call, in a library that often runs with elevated privileges to read GPU
  sysfs — a known hardening anti-pattern even when the intent is test-only.
