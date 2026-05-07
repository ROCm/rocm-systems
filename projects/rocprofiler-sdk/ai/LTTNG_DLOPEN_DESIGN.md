# Plan: Load liblttng-ust on demand via dlopen, fall back to no-op when absent

> **Status:** Design proposal. Not yet implemented.
> Companion to `HIGH_LEVEL_DESIGN_SUMMARY.md` (§"Parallel track: LTTng-UST emit-and-subscribe transport").

This document plans a follow-up evolution of the LTTng-UST producer-side
instrumentation that's already shipped in PRs #5475 + #5513: replace the
hard runtime dependency on `liblttng-ust.so.1` with a runtime `dlopen()`
that tries, in priority order:

1. An already-loaded `liblttng-ust.so.1` in the process (`RTLD_NOLOAD`).
2. A system install (default loader search path).
3. The vendored copy we ship at `/opt/rocm/lib/liblttng-ust.so.1`.
4. Fall back to no-op tracing if all three fail.

The producer .so files (`libamdhip64.so`, `libhsa-runtime64.so`) would
no longer have a `DT_NEEDED` entry on `liblttng-ust.so.1`, and would
load cleanly on systems where LTTng is not installed.

---

## 1. Goals and non-goals

**Goals:**

1. **Soft runtime dependency on liblttng-ust.** ROCm runtimes load
   cleanly on systems with no LTTng installation; tracing simply
   isn't available. Customers in restricted environments (no autotools
   build, no liblttng-ust package, no permission to install one) are
   not blocked from running ROCm.
2. **Honor existing process-wide liblttng-ust.** If another component
   in the process (a profiler, a tracer harness, the application
   itself) has already loaded liblttng-ust, use that copy — do not
   double-load, do not allocate independent state.
3. **Honor system installs.** Customers running on a distro with
   `liblttng-ust-dev` installed get tracing via the distro's copy; we
   do not silently override their version.
4. **Ship a vendored fallback.** When neither already-loaded nor
   system install is present, fall back to the vendored copy at
   `/opt/rocm/lib/liblttng-ust.so.1` (still shipped per PR #5475's
   vendoring decision).
5. **Tracepoints compiled in to our `.so`.** No separate
   `librocm_hip_tp.so` or `librocm_hsa_tp.so`. The tracepoint
   provider definitions and probe code live in
   `libamdhip64.so` / `libhsa-runtime64.so` directly (Scenario 1 of
   the LTTng v2.15 build matrix), but the runtime liblttng-ust
   dependency is dlopen-resolved (Scenario 5).
6. **Lazy init on first tracepoint touch.** The first
   `lttng_ust_tracepoint(...)` call site to fire triggers the
   dlopen + register cascade. No-cost at process startup if nothing
   ever touches a tracepoint.

**Non-goals:**

1. **Replacing the vendored build.** The vendored LTTng-UST 2.13.7 +
   userspace-rcu 0.14.0 stay as build artifacts; this work changes
   only how `libamdhip64.so` and `libhsa-runtime64.so` resolve them
   at runtime.
2. **Replacing the consumer-side tooling.** `lttng-sessiond`,
   `babeltrace2`, `lttng` CLI are operator-side and unaffected by
   this work.
3. **Changing the tracepoint event schema or the curated-args
   subset.** Schema v3 (vpid+vtid+ts join, no in-band corr_id) and
   the 73 HIP + 10 HSA curated APIs are unchanged.
4. **Loading liblttng-ust eagerly at .so constructor time.** Eager
   load defeats the soft-dependency property. Lazy on first
   tracepoint is the trigger.
5. **Replacing the firmware-ring or KFD work.** Different layer;
   independent.

---

## 2. Background — what the LTTng v2.15 docs describe

The LTTng v2.15 documentation enumerates 5 distinct linking scenarios
for combining a tracepoint provider with an application. Each one is
some combination of "where the provider code lives" × "how the
application is linked":

| # | Provider lives where | App link line | Runtime requires |
|---|---|---|---|
| 1 | Statically inside the application binary | `-llttng-ust -ldl` | `liblttng-ust.so` |
| 2 | Static archive (.a) inside the application | `-llttng-ust -ldl` | `liblttng-ust.so` |
| 3 | Separate `libtpp.so`; app links it explicitly | `-ltpp -ldl` (liblttng-ust transitive) | `libtpp.so` + `liblttng-ust.so` |
| 4 | Separate `libtpp.so`; LD_PRELOAD-injected | **`-ldl` only** | None (works with or without preload) |
| 5 | Separate `libtpp.so`; app `dlopen()`s it | **`-ldl` only** | None (works with or without dlopen) |

Today's PR #5475 implementation is **Scenario 1**: the tracepoint
providers (`rocm_hip` and `rocm_hsa`) are compiled into their respective
runtime .so files, and the .so link line includes `-llttng-ust`. This
gives the simplest build but the strongest runtime requirement: the
producer .so will refuse to load on systems without `liblttng-ust.so.1`.

**The crucial observation from Scenarios 4 and 5:** when you compile
the tracepoint call sites with `LTTNG_UST_TRACEPOINT_PROBE_DYNAMIC_LINKAGE`
defined, the `lttng_ust_tracepoint(...)` macro expands to code that
does **not** reference any liblttng-ust symbol at link time. The
application can be built with **just `-ldl`** in its link line (no
`-llttng-ust`). The tracepoint call sites remain safe to invoke
whether or not the provider DSO (and hence liblttng-ust) is present
at runtime.

Verbatim from the docs (§ Scenario 5):

> "The instrumented application dynamically loads the tracepoint
> provider package shared object."
> Build commands:
>   `gcc -I. -fpic -c tpp.c`
>   `gcc -shared -o libtpp.so tpp.o -llttng-ust -ldl`
>   *(in app.c, before `#include "tpp.h"`):*
>     `#define LTTNG_UST_TRACEPOINT_DEFINE`
>     `#define LTTNG_UST_TRACEPOINT_PROBE_DYNAMIC_LINKAGE`
>   `gcc -c app.c`
>   `gcc -o app app.o -ldl`              # <-- no -llttng-ust, no -ltpp
> Run: `./app`

What the v2.15 docs do **not** describe is the combination we want:
**provider compiled into our `.so` (Scenario 1's "in our binary") +
liblttng-ust dlopen-loaded on demand (Scenarios 4/5's "no `-llttng-ust`
in link line")**. This is a synthesis of the documented scenarios, not
a directly documented pattern. Section 4 below addresses what that
combination requires.

A related building block from the LTTng API surface, not documented in
v2.15 but described in older lttng-ust documentation:

> `lttng_ust_loaded` — a weak symbol set to 1 by liblttng-ust's library
> constructor. Application code can declare
> `int lttng_ust_loaded __attribute__((weak));` and test it at runtime
> to detect whether liblttng-ust has been loaded into the process.

This gives us a clean way to detect the "process already has
liblttng-ust" case without requiring our own bookkeeping.

---

## 3. Why this is worth doing

Today (PR #5475), even when LTTng tracing is not in use, every ROCm
deployment carries:

1. A hard `DT_NEEDED` entry on `liblttng-ust.so.1` from
   `libamdhip64.so` and `libhsa-runtime64.so`. Loaders without
   the file at runtime refuse to start the application.
2. The vendored `liblttng-ust.so.1` (and friends) installed under
   `/opt/rocm/lib/`, ~120 KiB on disk.
3. Per-tracepoint OFF cost of ~5 ns (one atomic load + branch) on
   every tracepoint call site, even when nobody is recording.

The dlopen-on-demand design changes (1) from hard to soft. (2) is
unchanged — we still ship the vendored copy as a fallback. (3) is
expected to be approximately the same when liblttng-ust is loaded;
when liblttng-ust is **not** loaded, the per-tracepoint cost drops
to one branch on a process-global state byte (no-op stub).

The soft dependency matters for several real customer scenarios:

* **Restricted minimal containers** that intentionally exclude
  optional libraries.
* **Pre-existing customer LTTng installations** at versions different
  from the vendored 2.13.7 — runtime can use whichever copy is in the
  process without our build needing to know about every customer
  version matrix.
* **Air-gapped / signed-package environments** where customers cannot
  install autotools-built artifacts and the only acceptable
  liblttng-ust is the distro's signed package.
* **Customers explicitly want to use their own LTTng tooling** — they
  expect events to flow through their existing observability stack
  rather than through a vendor-shipped copy.

---

## 4. Architecture

### 4.1 Build-time changes

**Producer .so link lines:** drop `-llttng-ust`. Add `-ldl` if not
already present. Both `libamdhip64.so` and `libhsa-runtime64.so` end
up with no `DT_NEEDED` on `liblttng-ust.so.1`.

**Tracepoint provider TUs:** add `LTTNG_UST_TRACEPOINT_PROBE_DYNAMIC_LINKAGE`
define alongside the existing `LTTNG_UST_TRACEPOINT_DEFINE` and
`LTTNG_UST_TRACEPOINT_CREATE_PROBES` defines.

The current pattern in `projects/clr/hipamd/src/lttng/rocm_hip_tp.cpp`
is approximately:

```c
#define LTTNG_UST_TRACEPOINT_CREATE_PROBES
#define LTTNG_UST_TRACEPOINT_DEFINE
#include "rocm_hip_tp.h"
```

The new pattern would add:

```c
#define LTTNG_UST_TRACEPOINT_CREATE_PROBES
#define LTTNG_UST_TRACEPOINT_DEFINE
#define LTTNG_UST_TRACEPOINT_PROBE_DYNAMIC_LINKAGE
#include "rocm_hip_tp.h"
```

**Open question to resolve in spike (§7):** does
`LTTNG_UST_TRACEPOINT_CREATE_PROBES` produce code that compiles cleanly
without `-llttng-ust` on the link line? The `CREATE_PROBES` expansion
generates the per-event probe functions, which call into liblttng-ust
internals to perform serialization. If those calls are unresolved at
link time, we have three options:

1. **Weak symbol annotations** — declare every liblttng-ust function
   the probes call as `__attribute__((weak))` so undefined references
   resolve to NULL. Would require source-level changes in our build,
   not in liblttng-ust itself; verify whether this is feasible
   without forking liblttng-ust headers.
2. **dlsym + thunk layer** — at first tracepoint touch, dlsym every
   liblttng-ust symbol the probes need into a function-pointer table;
   probe code calls the table. Most invasive but least dependent on
   linker / ABI details.
3. **Defer probe registration only, not probe compilation** — probe
   bodies compile with normal `-llttng-ust` references but a
   weak-undefined `liblttng-ust.so.1` allows them to remain undefined
   at process load. Requires `--unresolved-symbols=ignore-in-shared-libs`
   or equivalent linker flags. Cleanest if it works.

The Phase 0 spike must determine which of these three is viable.

### 4.2 Runtime changes — the loader shim

A new component, `shared/lttng/rocm_lttng_loader.{h,cpp}` (or a
per-provider variant), provides the dlopen orchestration. Public
surface is small:

```c
// One-time init; idempotent. Safe to call from multiple threads.
// Returns true if liblttng-ust is loaded and providers are registered.
bool rocm_lttng_init();

// Cheap probe — single relaxed atomic load. Used by tracepoint stubs.
bool rocm_lttng_is_active();
```

Internal state machine:

```
NOT_INITIALIZED ──────► INIT_IN_PROGRESS ──────► LOADED
                              │
                              ├──────► NOT_AVAILABLE
                              │
                              └──────► REGISTRATION_FAILED
```

Initialization sequence (one shot, guarded by atomic CAS):

```c
// Step 1: detect already-loaded liblttng-ust.
//
// The lttng_ust_loaded weak symbol, when liblttng-ust is in the
// process, is set to 1 by its constructor. If our own .so was
// loaded after some other liblttng-ust-using component, this gives
// us a free signal.
extern int lttng_ust_loaded __attribute__((weak));

void* h = nullptr;
if (&lttng_ust_loaded != nullptr && lttng_ust_loaded != 0) {
    // liblttng-ust is already in the process. Get a handle to it
    // without re-loading.
    h = dlopen("liblttng-ust.so.1", RTLD_NOLOAD | RTLD_NOW);
}

// Step 2: try system install (default loader search).
if (!h) {
    h = dlopen("liblttng-ust.so.1", RTLD_NOW | RTLD_GLOBAL);
}

// Step 3: try our vendored fallback at a known absolute path.
if (!h) {
    h = dlopen("/opt/rocm/lib/liblttng-ust.so.1", RTLD_NOW | RTLD_GLOBAL);
}

// Step 4: give up gracefully.
if (!h) {
    state.store(NOT_AVAILABLE, memory_order_release);
    return false;
}

// Step 5: register our in-binary providers with the now-loaded
// liblttng-ust. See §4.3 for what this entails.
if (!register_providers(h)) {
    state.store(REGISTRATION_FAILED, memory_order_release);
    return false;
}

state.store(LOADED, memory_order_release);
return true;
```

`RTLD_GLOBAL` is required so subsequent dlopen calls (or the loader's
search for liblttng-ust-tracepoint.so / liburcu) can resolve symbols
from the loaded copy.

`RTLD_NOLOAD | RTLD_NOW` on Step 1 returns a handle to an already-loaded
copy without changing reference counts or load behavior. If liblttng-ust
is not loaded, returns NULL and we fall through to Step 2.

**Failure modes — all silent:** none of the dlopen failures emit
diagnostics. ROCm-side opt-in for verbose loader diagnostics could be
gated on an env var (`ROCM_LTTNG_VERBOSE=1`) for ops-debugging; default
quiet.

### 4.3 Provider registration after dlopen

Normal LTTng tracepoint providers are packaged as separate .so files;
the provider's library constructor calls into liblttng-ust at dlopen
time to register the provider. With the providers compiled into our
.so, our .so's constructor runs at process startup — long before
liblttng-ust may be loaded.

We need a deferred registration path. Approximate options:

**Option A — Deferred constructor pattern.** Compile providers with
`CREATE_PROBES` as today, but place the LTTng-UST initialization call
behind a runtime gate that fires from `rocm_lttng_init()` instead of
from a `__attribute__((constructor))`. This requires understanding the
internal LTTng-UST symbol(s) involved in provider registration and
calling them via dlsym after dlopen.

**Option B — Late constructor.** Implement a function in our .so that
mimics what liblttng-ust's `lttng_ust_init_tracepoint()` (or whatever
the real entry point is named) does: walks the tracepoint provider
descriptor table, calls `lttng_ust_register_tracepoint_provider()` for
each. Requires more LTTng-UST internal API archaeology.

**Option C — Probe shim hand-off.** Define the providers in a small
section of our .so, but defer ALL liblttng-ust calls via a function
pointer table. At `rocm_lttng_init()`, dlsym every required entry
point into the table, then walk the descriptor list and register.

The Phase 0 spike must identify the actual LTTng-UST symbol name(s)
required for in-binary provider registration. The
`lttng-ust/include/lttng/tracepoint.h` and `tracepoint-event-impl.h`
headers are the source of truth; we may need to copy small portions
of their constructor logic into our shim or invoke them via dlsym
after liblttng-ust loads.

### 4.4 Tracepoint call-site stubs

Today (PR #5475), each `lttng_ust_tracepoint(...)` call expands to
an inline check of a per-tracepoint `state` byte that lives in the
tracepoint provider's data section. With providers in our .so this
state byte is in our address space; access is one atomic load + one
likely branch.

After this work, the stub needs to additionally honor the loader
state: when liblttng-ust is not loaded, no events are recorded. The
docs say this is automatic when `LTTNG_UST_TRACEPOINT_PROBE_DYNAMIC_LINKAGE`
is defined ("the call-site is safe to invoke when the provider DSO is
absent"), but for our case the provider IS in our .so — we may need
to wrap each tracepoint with our own enabled-check macro:

```c
#define ROCM_TRACE(provider, event, ...) \
    do { if (rocm_lttng_is_active()) { \
        lttng_ust_tracepoint(provider, event, __VA_ARGS__); \
    } } while (0)
```

This adds one branch per tracepoint call (two branches total when
LTTng is loaded — our outer check + LTTng's per-tracepoint enabled
flag — but when LTTng is unloaded we skip the inner check entirely
and the outer check is the only cost).

The Phase 0 spike measures whether this is actually cheaper than the
current pattern in the "loaded but no session" case — if the LTTng
per-tracepoint check is already ~5 ns and our outer check adds ~1 ns,
the total ~6 ns is a regression vs today's ~5 ns. Acceptable if it's
the price of soft-dep, but worth knowing.

### 4.5 Init trigger — lazy on first tracepoint

The first call to `rocm_lttng_is_active()` (or directly to a tracepoint
stub) triggers `rocm_lttng_init()`. Implementation:

```c
inline bool rocm_lttng_is_active() {
    auto s = state.load(memory_order_acquire);
    if (LIKELY(s == NOT_AVAILABLE || s == REGISTRATION_FAILED))
        return false;
    if (LIKELY(s == LOADED))
        return true;
    return rocm_lttng_init_slow_path();
}
```

The `LIKELY` annotations bias the branch predictor toward the
already-resolved cases (`NOT_AVAILABLE` for a customer with no LTTng;
`LOADED` for a customer with one). The `INIT_IN_PROGRESS` and
`NOT_INITIALIZED` paths fall through to `rocm_lttng_init_slow_path()`
which holds a mutex, performs the dlopen sequence, and updates state.

**Race window:** between process startup and the first tracepoint
touch, no liblttng-ust loading happens. A pathological case is a
customer who starts `lttng-sessiond` then immediately runs a HIP
program that does no ROCm-side work that would touch a tracepoint —
no events get captured because we never initialized. In practice, any
HIP program calls `hipMalloc` / `hipDeviceSynchronize` / etc. early,
and every HIP wrapper has a tracepoint, so this race is theoretical
for most programs.

**Alternative trigger:** explicit env var (`ROCM_LTTNG_UST=1`) forces
eager init at .so constructor time. Could be supported alongside the
lazy path for ops-driven workflows where init timing matters. **Not in
scope for the initial change** — lazy-only for simplicity; revisit if
operators ask for it.

---

## 5. Vendoring strategy — Model A (keep vendored as fallback)

PR #5475 ships LTTng-UST 2.13.7 + userspace-rcu 0.14.0 in
`/opt/rocm/lib/` flat. After this dlopen work, that copy becomes the
**third-priority fallback**: used only when neither already-loaded nor
system install is available.

**Why keep it:** dissolves the "what if the customer's distro doesn't
have liblttng-ust?" question entirely. RHEL 9 and Ubuntu 22.04 LTS —
both first-class ROCm targets — do not ship liblttng-ust by default;
without the vendored fallback, customers on those distros would see
no tracing unless they install liblttng-ust separately. With the
fallback, tracing Just Works on every supported distro.

**Cost of keeping it:** ~120 KiB on disk under `/opt/rocm/lib/`,
build-time autotools dependency for the vendored bootstrap. Both are
already paid by PR #5475 today.

**Eventual revisit:** once rocprofiler-sdk's CTF consumer ships and
customers adopt the new flow, measure how often the vendored fallback
actually gets used (e.g., add an opt-in event in the loader shim that
records which path was taken). If the fallback never fires in practice,
make it a build-time option (Model C) and let downstream packagers
decide.

---

## 6. Phased implementation plan

### Phase 0 — Spike (~1 day)

A small standalone test program that exercises the architectural
unknowns:

1. Build a tracepoint provider compiled into a single executable, with
   `LTTNG_UST_TRACEPOINT_PROBE_DYNAMIC_LINKAGE` defined.
2. Link without `-llttng-ust`. Verify it links.
3. Run the executable with no `liblttng-ust.so` available (e.g., in a
   `LD_LIBRARY_PATH=` empty env or chroot). Verify it runs without
   crashing and emits no events.
4. Add the dlopen shim. Run again with liblttng-ust available. Verify
   events flow.
5. Identify the exact LTTng-UST entry point(s) needed for in-binary
   provider registration after dlopen. Document.
6. Microbenchmark: compare per-tracepoint cost in three states:
   (a) liblttng-ust never loaded; (b) loaded, no session; (c) loaded,
   session active.

**Spike output:** thumbs-up or thumbs-down on architectural feasibility.

If thumbs-down, the design fails and we keep the current hard-link
model. If thumbs-up, proceed to Phase 1.

### Phase 1 — Producer-side conversion (~3-5 days)

1. Add `shared/lttng/rocm_lttng_loader.{h,cpp}` (or per-provider
   variant). Includes the dlopen sequence, the provider registration
   logic identified in Phase 0, the lazy-init state machine.
2. Update tracepoint provider TUs in HIP CLR + ROCr to add
   `LTTNG_UST_TRACEPOINT_PROBE_DYNAMIC_LINKAGE`.
3. Update CMake: drop `-llttng-ust` from producer .so link lines.
   Drop the `liblttng_ust_vendored` PRE_LINK staging step's runtime
   .so copy if no longer needed (the build-time .so is still needed
   for the linker to satisfy `LTTNG_UST_TRACEPOINT_PROBE_DYNAMIC_LINKAGE`'s
   weak references — verify in spike).
4. Wrap tracepoint call sites with the `ROCM_TRACE(...)` macro that
   adds the loader-state check.
5. Update tests: existing trace-flow tests should still pass when
   liblttng-ust is available; add a new test that strips
   `liblttng-ust.so` from `LD_LIBRARY_PATH` and verifies the runtime
   loads + runs cleanly with no crashes and no events.

### Phase 2 — Validation (~2 days)

Test matrix on `bewelton_lttng` container:

| Scenario | Setup | Expected |
|---|---|---|
| **No liblttng-ust** | Strip `/opt/rocm/lib/liblttng-ust*`, no system install | Runtime loads; HIP program runs; 0 events captured; no error spam |
| **System install only** | `apt install liblttng-ust1` (or distro equiv); strip vendored copy | Runtime loads; events flow via system copy |
| **Vendored only** | Vendored copy at `/opt/rocm/lib/`; no system copy | Runtime loads; events flow via vendored copy |
| **Already loaded** | LD_PRELOAD a wrapper that loads liblttng-ust before our .so | Runtime detects already-loaded; uses that copy; events flow |
| **System + vendored both present** | Both available | Runtime picks system (priority 2 beats priority 3); events flow via system |
| **Concurrent threads first-tracepoint race** | Multi-threaded program launches HIP work on N threads simultaneously | One thread wins the init CAS; others wait; init completes once; events flow |

GraphBench overhead measurement (12 reps × 4 configs):

| Config | Expected vs current PR #5475 |
|---|---|
| LTTng never loaded (no session, no liblttng-ust) | ~Same as current "lttng_off" or slightly cheaper (no DT_NEEDED resolution at startup) |
| LTTng loaded, no session | Same or +1 branch per tracepoint |
| LTTng loaded, generic session (14 events) | Same as current +0.6% |
| LTTng loaded, full curated session (97 events) | Same as current +0.9% |

### Phase 3 — Rollout decision (~1 day)

Land the change behind a build-time flag (`ROCM_LTTNG_DLOPEN=ON`)
defaulted to OFF for the first release; flip default to ON after one
release of bake time. Operators who hit dlopen-related issues can
revert per-build via the flag. Once dlopen-mode is the only path,
remove the static-link option in a subsequent release.

---

## 7. Open technical questions to resolve in Phase 0

These are the real unknowns. Each has a fallback path documented in
case the answer makes the design infeasible.

1. **Does `LTTNG_UST_TRACEPOINT_CREATE_PROBES` produce code that
   compiles + links without `-llttng-ust`?** Probe code calls
   liblttng-ust internals; if those references are unresolved at .so
   link time, we need either weak-symbol annotations, a function
   pointer thunk layer, or ld flags like
   `--unresolved-symbols=ignore-in-shared-libs`.
   **Fallback if answer is "no clean way":** keep `-llttng-ust` in
   link line but mark it as `--as-needed --no-as-needed=lttng-ust`?
   Or accept the architecture is not workable and stop the project.

2. **What is the exact LTTng-UST entry point for in-binary provider
   registration after dlopen?** Normally provider registration runs
   from the provider .so's library constructor; with the provider in
   our .so we need to invoke this manually after dlopen succeeds.
   **Fallback if no clean public entry point:** ship the providers as
   tiny separate .so files (`librocm_hip_tp.so`, `librocm_hsa_tp.so`)
   that we dlopen ourselves, with the rest of our .so compiled
   without LTTng knowledge — partial regression vs the "all in our
   .so" goal but architecturally clean.

3. **Does the `lttng_ust_loaded` weak symbol exist in LTTng 2.13?**
   We vendor 2.13.7. The web search referenced the symbol but
   without a version pin. Verify it's present in the version we
   ship and in the system versions we expect to encounter.
   **Fallback if absent:** use our own bookkeeping (set a global
   from our shim after a successful dlopen), accepting we might
   double-load if another component already loaded liblttng-ust
   (RTLD_NOLOAD prevents double-loading at the kernel level, but
   we'd waste the dlopen syscall).

4. **`RTLD_GLOBAL` interactions.** If we dlopen liblttng-ust with
   `RTLD_GLOBAL`, its symbols become available to subsequent dlopen'd
   libraries that may need them (e.g., a customer's profiling tool).
   This is the correct flag, but verify it doesn't introduce ordering
   problems with already-loaded copies that may have been opened with
   `RTLD_LOCAL` by some other component.

5. **liburcu dependency.** liblttng-ust depends on liburcu. When we
   dlopen liblttng-ust, the dynamic linker resolves liburcu via
   normal search paths. If liburcu is not present (e.g., on a stripped
   container) the dlopen fails. We may need to dlopen liburcu first,
   in priority order (already-loaded / system / vendored), before
   dlopening liblttng-ust. Spike validates this.

6. **Per-tracepoint cost regression.** Current PR #5475 cost is one
   atomic load + branch (the per-tracepoint enabled flag). New design
   adds an outer check on the loader state. Two branches when LTTng
   is loaded vs one today. Verify the regression is in the noise and
   not measurable on GraphBench.

7. **`liblttng-ust-tracepoint.so`** — there's a separate small library
   that provides `tracepoint(...)` symbol resolution helpers. Investigate
   whether it's also auto-loaded by liblttng-ust at dlopen time, or
   whether we need to dlopen it ourselves.

---

## 8. Risks

| Risk | Severity | Mitigation |
|---|---|---|
| The combination "provider in our .so + link without `-llttng-ust` + dlopen on demand" is not a documented LTTng pattern; may not work without source-level patches to LTTng-UST or non-trivial registration glue | **HIGH** | Phase 0 spike. Hard gate on feasibility. If spike fails, abort the design. |
| LTTng-UST internal API for "register this in-binary provider after the fact" may not be a stable public symbol; may need dlsym of internal entries | **MEDIUM** | Phase 0 identifies. If too brittle, fall back to ship-as-separate-.so as documented in §7 question 2. |
| ABI / version compatibility: vendored 2.13.7 vs system 2.14 / 2.15. dlopen may pick up an incompatible version that crashes when our probes try to call into it | **MEDIUM** | Pin to ABI version `liblttng-ust.so.1` already (does so today). Add a post-dlopen ABI probe via dlsym on a known symbol; refuse load if ABI mismatch detected. |
| Per-tracepoint OFF cost might not actually drop materially in the "no liblttng-ust" case vs today's "loaded but session disabled" case | **LOW** | Phase 0 microbenchmark. If the difference is in the noise, the architectural value is in the **runtime dependency story** (no liblttng-ust required) rather than performance — still worth doing but framing changes. |
| Registration timing race: customer starts `lttng-sessiond` after our .so loads but before any tracepoint fires. Init-on-first-touch handles this; an eager-init env var is the escape valve if needed | **LOW** | Document. Ship lazy-only initially; add eager opt-in if operators ask. |
| liburcu missing in stripped containers means even a working dlopen of liblttng-ust fails | **LOW** | Spike validates. If liburcu absence is common we add a similar priority-order dlopen for liburcu. |
| Vendored copy never actually gets used in practice (defeats Model A's value) | **LOW** | Add an opt-in counter event in the loader shim that records which path was taken. Revisit Model A → Model C decision after one release of usage data. |

---

## 9. Alternatives considered

### Alt 1 — Status quo (hard link to liblttng-ust)

What PR #5475 ships today. Strongest dependency, simplest build. The
explicit reason this design exists is that "strongest dependency" is
the property we want to weaken.

### Alt 2 — Provider as separate `.so`, app dlopen's it (Scenario 5 verbatim)

Ship `librocm_hip_tp.so` and `librocm_hsa_tp.so` as separate
artifacts. Our .so dlopens them at first tracepoint touch. Works
exactly per the LTTng v2.15 docs without any synthesis. Drawbacks:

* Two extra .so files in `/opt/rocm/lib/`.
* Lifecycle management of the provider .so files (who installs them,
  how do they get found, package dependency declarations, etc.).
* Customer-facing surface area increases — more files to inspect, more
  symbols to chase in `ldd` / `lsof` output.

This is the **fallback** if Phase 0 shows the "providers in our .so"
combination doesn't work cleanly.

### Alt 3 — LD_PRELOAD-based activation (Scenario 4)

Customers explicitly preload `librocm_hip_tp.so` to enable tracing.
Requires customer-side cooperation; defeats the "tracing just works
when a session is active" UX. Not viable as a default but could
coexist with the dlopen path.

### Alt 4 — Migrate to Linux `user_events` (kernel 6.4+)

Reuses the existing tracepoint instrumentation but swaps the backend
to the kernel's `user_events` ABI. Removes the liblttng-ust dependency
entirely. Discussed in `TRACING_DELIVERY_RESEARCH.md`; held back by
distro reach (RHEL 9 / Ubuntu 22.04 LTS lack the kernel). The dlopen
work proposed here is independent of the eventual `user_events`
migration; both can coexist in a future v2 design where the per-event
backend is selectable.

### Alt 5 — Roll our own tracing transport

Discussed and rejected in `TRACING_DELIVERY_RESEARCH.md` § "Rolling
our own (custom shared-memory transport)". The reasons that rejected
it then still apply.

---

## 10. Out of scope

* Changing the tracepoint event schema or curated-args coverage.
* The rocprofiler-sdk consumer-side translator (CTF →
  `rocprofiler_*_record_t`) — separate planned PR.
* The firmware-ring + KFD work — different layer.
* Cross-distro CI matrix — orthogonal; address separately.
* Windows behavior — already OFF by default; unchanged.
