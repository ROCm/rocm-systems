# LTTng-UST Producer-Side Spike (Phase 0)

This directory contains a hello-world LTTng-UST tracepoint provider used
to validate the toolchain and discover host-environment surprises before
touching the HIP and HSA runtimes.

See `/home/bewelton/ai/2026-04-24-lttng-ust-producer-side-plan.md` for
the full implementation plan; this document captures only the Phase 0
findings that subsequent phases need to be aware of.

## Files

| File | Purpose |
|---|---|
| `spike_tp.h`               | Tracepoint provider header (`rocm_spike:hello`). |
| `spike_tp.cpp`             | Tracepoint provider package (TPP) translation unit. |
| `spike_main.cpp`           | Driver. Three modes: default (10 events, exit), `trace N M` (N events x M iters at 1 Hz), `sleep N` (10 events then sleep). |
| `restrictive.map`          | Conservative version script, mirrors the HSA-runtime pattern. |
| `restrictive_strict.map`   | Maximally-strict version script -- exports NOTHING. |
| `CMakeLists.txt`           | Builds executable + two restrictive .so flavors. |

## Build

In the dev container:

```bash
cmake -S projects/clr/hipamd/test/lttng/spike -B /tmp/lttng-spike-build
cmake --build /tmp/lttng-spike-build -j
```

Produces:
- `rocm_lttng_spike`              -- regular executable
- `librocm_lttng_spike_so.so`     -- restrictive build (whitelist version script)
- `librocm_lttng_spike_strict_so.so` -- maximally-strict build (no exports)

## Phase 0 Findings (must read before Phase 1 / 4)

### Finding 1: small `/dev/shm` requires explicit channel sizing

The dev container has `/dev/shm` of only 64 MiB. With 224 CPUs and
LTTng's default per-uid channel of `subbuf-size=524288 num-subbuf=4`,
channel creation fails on the consumer daemon with:

```
Error: ask_channel_creation consumer command failed
Error: Error creating UST channel "channel0" on the consumer daemon
```

The session is created and the event is enabled, but no data file is
written -- only the metadata file appears in the trace output dir, and
that metadata does not even mention the user's event class.

**Fix for Phase 0 / Phase 7 validation scripts:** explicitly create a
small channel before enabling events:

```bash
lttng enable-channel --userspace --subbuf-size=4096 --num-subbuf=2 chan_small
lttng enable-event --userspace --channel chan_small "rocm_spike:hello"
```

**Fix for production:** ensure `/dev/shm` is sized appropriately for the
per-uid channel layout the producer ships with. Phase 7 microbench will
need to commit a `/dev/shm` minimum-size guideline alongside the
producer-side architecture commitments.

### Finding 2: tracepoint provider registration does NOT require any dynamically exported symbols

The plan (Step 8) anticipated needing to whitelist `__tracepoint_*` and
`__tracepoint_provider_*` patterns in `hsacore.so.def` to keep the
provider registration working under HSA's restrictive
`-fvisibility=hidden + --version-script + -flto + --exclude-libs,ALL`
link.

**Empirical result:** both `librocm_lttng_spike_so.so` (whitelist map)
and `librocm_lttng_spike_strict_so.so` (zero exports) successfully
register `rocm_spike:hello` with `lttng-sessiond` when LD_PRELOADed
into `/bin/sleep`.

```
nm -D --defined-only /tmp/lttng-spike-build/librocm_lttng_spike_strict_so.so | wc -l
# 0
LD_PRELOAD=/tmp/lttng-spike-build/librocm_lttng_spike_strict_so.so /bin/sleep 30 &
lttng list --userspace
# PID: <pid> - Name: /bin/sleep
#       rocm_spike:hello (...)
```

**Reason:** the LTTng provider's `__attribute__((constructor))` is
invoked by glibc's `.init_array` at .so load time. The constructor's
address does not need to be visible in the dynamic symbol table for the
loader to call it. Registration with sessiond happens entirely via the
per-.so `__lttng_ust_tracepoints_ptrs` section, which lttng-ust
discovers via `__start_/__stop_` linker-defined symbols inside the
same .so (also internal to the .so, not requiring dynsym entries).

**Implication for Phase 4:** `hsacore.so.def` does NOT need any new
entries to support LTTng-UST. The existing export whitelist can stay
as-is. This is a meaningful simplification of the originally-planned
HSA build-system change.

### Finding 3: `dlmopen(LM_ID_NEWLM)` SEGFAULTs the LTTng provider constructor

When `librocm_lttng_spike_so.so` is loaded via
`dlmopen(LM_ID_NEWLM, ..., RTLD_NOW)` -- the loader pattern used by
HPCToolkit, Score-P, and other tools that isolate HIP/HSA into a
private link namespace -- the LTTng provider constructor crashes with
SIGSEGV inside glibc's `add_to_global_resize` (called from the
constructor's internal `dlopen("liblttng-ust-tracepoint.so.1")`).

```
#0  0x00007ffff7fd1efb in add_to_global_resize (new=...) at ./elf/dl-open.c:126
#1  0x00007ffff7fd2f10 in dl_open_worker_begin (...) at ./elf/dl-open.c:737
...
#5  0x00007ffff7fd2164 in _dl_open (file=0x... "liblttng-ust-tracepoint.so.1", ...)
...
#12 0x00007ffff7acf594 in ?? () from /lib/x86_64-linux-gnu/liblttng-ust-tracepoint.so.1
#13 0x00007ffff7fca71f in call_init (...)  -- constructor of the spike .so
...
#27 0x0000555555555230 in main ()         -- dlmopen call site
```

This is a known-bad interaction between `dlmopen` namespace isolation
and constructors that internally `dlopen` more libraries. Tested glibc
2.39 on Ubuntu 24.04.

**Action items for Phase 4 (HSA) and Phase 1 (HIP):**

1. The `__rocm_*_tp_init()` initialiser MUST be wrapped in a
   `dlopen(NULL, RTLD_NOLOAD)`-style introspection guard that detects
   whether the host runtime was loaded into a non-default link
   namespace, and SKIPS provider registration in that case (graceful
   no-op, all tracepoints inert).

2. Add the `ROCM_LTTNG_UST_DISABLE=1` env-var kill switch (already in
   the plan for Phase 1 Step 5) -- HPCToolkit users can set it to
   bypass the issue entirely until upstream glibc fixes the dlmopen +
   constructor-dlopen interaction.

3. Document this clearly in the user-facing release notes for the
   Phase 1 / Phase 4 PRs: "If you load libamdhip64.so or
   libhsa-runtime64.so via dlmopen(LM_ID_NEWLM), set
   ROCM_LTTNG_UST_DISABLE=1 to suppress LTTng provider registration."

The `RTLD_DEEPBIND` case (also tested) is unaffected -- the deepbind
loader does NOT trigger the `add_to_global_resize` crash because it
re-uses the default namespace.

### Finding 4: graceful no-op in containers without `lttng-sessiond`

Tested by running both `rocm_lttng_spike` and `LD_PRELOAD=...so /bin/true`
inside a fresh `docker run --rm ubuntu:22.04` with `liblttng-ust1`
installed but no `lttng-tools`, no `/var/run/lttng/`, and
`LTTNG_HOME=/nonexistent`.

Result: program exits 0 silently, no SIGSEGV, no hang, no error output
on stderr. This matches the requirement in the plan's Step 9(c).

## Step 9 outcomes summary

| Loader scenario             | Outcome              | Action |
|-----------------------------|----------------------|--------|
| Plain `LD_PRELOAD`          | Registers OK         | None   |
| `dlopen(... RTLD_DEEPBIND)` | Registers OK         | None   |
| `dlmopen(LM_ID_NEWLM, ...)` | **SEGFAULT in glibc**| Add link-namespace guard to `__rocm_*_tp_init()` (Phase 1 + Phase 4); document `ROCM_LTTNG_UST_DISABLE=1` workaround |
| Container no sessiond       | Exits 0 silently     | None   |
