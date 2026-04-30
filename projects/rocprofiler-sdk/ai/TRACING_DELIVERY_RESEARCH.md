# Userspace Tracing-Data Delivery Techniques — Research for rocprofiler-sdk

> Research output for a future rocprofiler-sdk design task: replacing the current
> callback-based tracing-data delivery (HIP/HSA → rocprofiler-sdk via
> `rocprofiler-register` function pointers) with a generic emit-and-subscribe
> mechanism. This document compares modern Linux userspace tracing-delivery
> techniques against two hard constraints:
>
> 1. **No `LD_PRELOAD`** — must support attaching to a process that did NOT
>    pre-arrange tracing.
> 2. **No runtime binary modification** — no text-segment patching, JIT code
>    rewriting, or in-place instruction overwrite when tracing is enabled.
>
> Author: research session, 2026-04-24
> Status: research complete; recommendation pending design review

---

## Status update — 2026-04-30

**The recommendation made by this document (LTTng-UST as primary) has
been carried out.** Producer-side instrumentation is complete and posted
as two stacked draft PRs:

| PR | Branch | Contents |
|---|---|---|
| **#5475** | `users/bewelton/lttng` | HIP CLR + ROCr LTTng-UST tracepoint providers; vendored LTTng-UST 2.13.7 + URCU 0.14.0 submodules; ~530 HIP + ~270 HSA wrappers migrated; 73 HIP + 10 HSA curated typed-args APIs; schema v3 (vpid+vtid+ts join, no in-band corr_id) |
| **#5513** | `users/bewelton/lttng-curated-verifier` | Stacked on #5475: libclang YAML↔header drift verifier; build-time symbol-coverage gate; DSL parser library; real-resource coverage harness; payload/invariant test suite; CI workflow |

**Headline measured numbers** (GraphBench, MI325X gfx942, 12 reps,
`/dev/shm` 64 MiB):

* Generic-only (14 event types): **+0.6% wall-time vs no-tracing baseline,
  0 drops**.
* Generic + curated (97 event types, ~10M events): **+0.9% wall-time, 0
  drops**.
* For comparison, `rocprofv3 --hsa-trace` at full fidelity is **+73%**
  on the same workload — roughly **80× more expensive** than LTTng-UST at
  full curated capture.

**Material design changes from the recommendation in this doc:**

1. **Distro dependency dissolved by vendoring.** Rather than depending on
   distro `liblttng-ust-dev` (the thing that originally made distro reach
   the headline risk on RHEL 9 / Ubuntu 22.04 LTS), LTTng-UST 2.13.7 and
   userspace-rcu 0.14.0 are vendored as git submodules under
   `projects/{clr,rocr-runtime}/external/`, built via `ExternalProject_Add`,
   and installed flat into `/opt/rocm/lib/`. No system package required at
   build time or runtime.
2. **In-band correlation IDs dropped (schema v3).** The original
   recommendation contemplated typed `correlation_id` fields on every event.
   Implementation work showed that vpid+vtid+timestamp from CTF channel
   contexts gives a strictly more correct join key (multi-process safe) and
   eliminates the entire `librocprofiler-register/correlation` ABI surface
   and ~530 LOC of producer-side correlation-stack maintenance.
3. **Curated parameter capture (Option C from QEMU/DPDK pattern).** The
   original recommendation outlined "typed tracepoint set" generically; the
   implementation chose a YAML DSL + Python codegen + libclang drift
   verifier pattern, with checked-in generated headers (default build needs
   no Python or libclang). Coverage: 73 HIP + 10 HSA APIs.

**Per-open-question resolution** is annotated inline in the "Open Questions"
section below.

**The body of this document is preserved as a reference** — the comparison
matrix and per-technique deep-dives remain a useful reference for
adversarial review of the chosen path and for any future revisit of the
transport decision.

### Update — 2026-04-30: LTTng-UST also delivers FW kernel-dispatch records (PR #5519)

The original recommendation framed LTTng-UST as the delivery mechanism
for **HIP/HSA API events**. Since then, the firmware-ring track
(`HIGH_LEVEL_DESIGN_SUMMARY.md`, `KNOWN_ISSUES.md`,
`FIRMWARE_RING_HYBRID_DESIGN.md`) has converged on the same transport:

* The HSA-resident drainer on `users/bewelton/lttng-kernel-ts`
  (PR #5519, draft) emits one
  `rocm_hsa:kernel_dispatch_record(queue_id, dispatch_idx, gpu_ts,
  record_type, corr_id, parent_corr_id)` LTTng tracepoint per FW
  dispatch record (one event per non-zero record; HSA does not
  interpret `record_type`).
* This event lands in the **same CTF channel** as
  `rocm_hip:hip_aql_kernel_dispatch_submit` and the curated
  `<api>_args` events from PR #5475.
* Schema v3's `(queue_id, write_idx)` cross-runtime join key is the
  same `(queue_id, dispatch_idx)` the FW writes into each record.
  Consumers join on this pair to recover the launching thread, the
  HIP API correlation, the workgroup/grid/segment sizes (already on
  the submit event), and the GPU-domain start/end timestamps (on the
  HSA event).
* Measured 169.3% combined-record capture rate on graphbench
  (~85% per record_type) with the per-queue drainer thread design.
  See `HIGH_LEVEL_DESIGN_SUMMARY.md` firmware-ring status table.

**Implication for this document.** The "schema v3 (no in-band
correlation IDs)" decision turned out to be the right one for a second
reason that was not visible at the time: it makes the FW-side records
joinable to the HIP-side events without either side knowing about the
other. If schema v3 had carried in-band correlation IDs, the HSA
drainer would have had to look them up — which would have re-introduced
the per-queue side-table machinery from
`FIRMWARE_RING_HYBRID_DESIGN.md` Sections 4.1–4.3 inside HSA, exactly
the coupling the side-table-free design avoids.

The rocprofiler-sdk consumer side — already noted as "not yet started"
in the Status update above — now has a slightly larger scope: it must
emit `KERNEL_DISPATCH_COMPLETE` records by joining
`rocm_hip:hip_aql_kernel_dispatch_submit` with
`rocm_hsa:kernel_dispatch_record` on `(queue_id, dispatch_idx)`,
in addition to translating the HIP/HSA API events into `*_API_*`
records. See `FIRMWARE_RING_HYBRID_DESIGN.md` §13.3 for the
consumer-side join mechanism, and §13.5 for the trade-off vs
bypassing LTTng and querying KFD directly.

---

# Why LTTng-UST was chosen over the alternatives

The original survey considered eleven candidate transports. This section
records — in current-status form — why LTTng-UST won the comparison and what
the disqualifying property of each alternative was. Per-technique mechanism
detail is preserved as an appendix below.

The decision was driven by four properties, in this order of priority:

1. **No `LD_PRELOAD`** — must support attaching to a process that did not
   pre-arrange tracing.
2. **No runtime binary modification** — no text-segment patching at attach
   time, no JIT code rewriting, no in-place instruction overwrite.
3. **Acceptable steady-state cost when nobody is subscribed** — one atomic
   load + branch per event budget.
4. **Late-attach without producer cooperation** — operator decides at any
   time to start tracing; producer doesn't need a restart, signal, or
   callback.

Five candidates fail at least one of these immediately: **uprobes**, **eBPF
uprobes**, and **USDT** all attach by writing `INT3` (0xCC) into the
producer's text segment when a consumer subscribes — they fail constraint 2.
**OpenTelemetry** has no true zero-cost off state — virtual dispatch and
attribute construction happen even with a NoopTracer, failing constraint 3.
**Chrome/Perfetto file** is an offline output format, not a live delivery
mechanism — fails constraint 4.

That leaves three candidates that survive the hard constraints:
LTTng-UST, io_uring channels, and rolling our own. Each is treated in
detail below.

## LTTng-UST (chosen)

| Property | Value |
|---|---|
| Hard constraint #1 (no LD_PRELOAD) | satisfied — linked at build, registers with `lttng-sessiond` at lib load |
| Hard constraint #2 (no binary mod) | satisfied — activation flips a per-CPU enabled byte; `.text` is never touched |
| OFF cost | ~5 ns (1 atomic load + 1 unlikely branch) |
| ON cost | ~100 ns per event (lock-free per-CPU shared-memory ring buffer) |
| Late-attach | yes, by design |
| Schema | strongly typed via CTF metadata; native versioning story |
| Tooling ecosystem | babeltrace2, Trace Compass, Perfetto (CTF ingest), Eclipse, custom CTF readers |
| Maturity | 10+ years upstream; production use at HFT firms, CERN, automotive, telecom |
| Distro reach | excellent on every ROCm-targeted distro (and: dissolved entirely by vendoring — see below) |
| Adopting cost | producer links `liblttng-ust`; tracepoint provider headers; consumer is `lttng-sessiond` + babeltrace2 |

**Implementation outcome** (see status banner above): producer-side
implementation is complete in PR #5475 + #5513. Measured GraphBench overhead
at full curated capture is +0.9% wall-time vs no-tracing baseline with zero
discarded events; the same workload under `rocprofv3 --hsa-trace` at full
fidelity is +73% (~80× more expensive). The original distro-portability
concern was sidestepped entirely by vendoring LTTng-UST 2.13.7 +
userspace-rcu 0.14.0 as git submodules under `projects/{clr,rocr-runtime}/`
— no system `liblttng-ust-dev` required at build time or runtime.

## OpenTelemetry SDK + OTLP exporters

**Why not chosen: no true zero-cost off state.** Even with a NoopTracer,
the API call still happens — virtual dispatch through the tracer, attribute
vector construction, sampler check. Putting an OTel call on the HIP
dispatch hot path adds always-on cost regardless of whether anyone is
collecting.

**Secondary concerns:**

* **Per-event cost when ON is hundreds of ns to low µs.** Span object
  construction, attribute serialization, BatchSpanProcessor enqueue — all
  on the producer thread. Heavyweight relative to LTTng-UST's ~100 ns.
* **Heavy dependency tree** (Protobuf, gRPC, Abseil) is a real
  distribution-portability headache. We avoid this with LTTng vendoring;
  vendoring OTel is much harder due to its dep tree size.
* **Wrong abstraction layer for our use case.** OTel is request-tracing
  oriented (spans with parent-child relationships, semantic conventions
  for HTTP/RPC/DB calls). HIP/HSA tracing is event-stream oriented. The
  semantic mismatch costs more than it saves.

**Where OTel still fits:** as an **export format** from rocprofiler-sdk
downstream of the in-process record consumer, for customers running
OTel-based observability stacks. AIM Engine (AMD's vLLM-on-ROCm sidecar)
already uses OTel for serving metrics. That's a reasonable place for it —
not on the HIP dispatch hot path.

## uprobes (kernel-installed user-space probes)

**Disqualified by hard constraint #2.** Arming a uprobe rewrites the
producer's in-memory text segment (kernel writes `INT3` (0xCC) at the probe
offset; the original instruction is preserved out-of-line for
single-stepping). The on-disk binary is preserved via copy-on-write but
the running process's text page is modified.

We chose this constraint deliberately: text-segment modification at attach
time interacts badly with self-relocating loaders, page-permission tooling,
and any future executable-integrity verification (signed code, secure
boot, IMA/EVM, container security policies). Permanently constraining
attach to "no text mod" rules out the entire uprobes family.

## eBPF uprobes (`bpf_program__attach_uprobe`)

**Same disqualification as raw uprobes.** The attach mechanism is identical
— INT3 in the producer's text, single-step out-of-line. The eBPF program
runs in the kernel handler instead of the bare ftrace path, but the text
modification is unchanged.

**Worth noting:** USDT-via-eBPF (`bpf_program__attach_usdt`) attaches to
compile-time-emitted NOPs in the producer. The NOPs were already in the
binary at compile time, but activation still flips them to INT3 via the
uprobes subsystem. **Same constraint violation.**

## USDT (Userspace Statically Defined Tracepoints) — DTrace-style probes

**Disqualified by hard constraint #2.** This is the closest in spirit to
what we want — compile-time static probes, generic emission, third-party
out-of-process subscribe via uprobes / bpftrace / SystemTap, zero runtime
library dep. **The disqualifying factor is the kernel's chosen
implementation** (uprobes/INT3): when a consumer attaches, the deliberately
inserted NOP slot gets overwritten with INT3.

If hard constraint #2 is renegotiated specifically for USDT — i.e.
"in-place patching of compile-time-reserved tracing slots is OK, but
patching of real code is not" — USDT comes back into play as a strong
contender. We did not pursue this renegotiation; LTTng-UST landed first
with measured-acceptable overhead. Worth a future conversation if specific
tooling needs (e.g., bpftrace-native subscribers without `liblttng`
dependency) emerge as a customer requirement.

## io_uring channels (`IORING_OP_MSG_RING` and proposed IPC)

**Why not chosen: late-attach discovery is unsolved on shipping kernels.**
A late-attaching consumer needs to obtain the producer's ring fd, which
requires either:

* (a) Pre-arranged out-of-band fd-passing via UNIX socket `SCM_RIGHTS` —
  implies LD_PRELOAD or pre-launch coordination, **violates constraint
  #1**.
* (b) `pidfd_getfd(2)` with `PTRACE_MODE_ATTACH_REALCREDS` — privileged.
* (c) The proposed io_uring IPC subsystem with named-channel discovery —
  not yet upstream in any shipping kernel.

None of these gives the "any tool subscribes by name from outside" property
we need.

**Strengths if used in a different role:** very low per-event cost
(~50–100 ns, comparable to LTTng-UST), shared-memory native, no syscall
per event. Could serve as a *secondary* channel between rocprofiler-sdk
and a downstream tool consumer once the primary HIP/HSA → tool channel is
solved by something else. Not the primary candidate today.

## perf_event_open(2) + tracepoints

Not a producer-side delivery mechanism on its own — it is a **consumer**
syscall that pairs with one of the actual producers (USDT via uprobes,
kernel tracepoints). For our problem it has nothing to add that
LTTng-UST + babeltrace2 doesn't already cover.

## Linux kernel tracepoints (`/sys/kernel/tracing`)

Wrong layer. These are static tracepoints compiled into the **kernel**
(KFD ioctls, AMDGPU driver). They are useful for a complementary
**kernel-side** trace stream (and KFD/AMDGPU could add more) but cannot
solve the **userspace** HIP/HSA → consumer delivery problem.

## Chrome trace / Perfetto file format

Wrong category. This is an **output format** for offline visualization,
not a delivery mechanism between a live producer and a live consumer.
rocprofv3 already supports `--output-format pftrace` as an export; that
role continues unchanged.

## Rolling our own (custom shared-memory transport)

**Why not chosen: cost-vs-benefit doesn't justify it.** This option was
considered seriously and rejected on the following grounds:

### What "our own" would have to provide

To match LTTng-UST's feature surface, a from-scratch tracing transport
would need every one of these subsystems:

* **Lock-free per-CPU shared-memory ring buffers** with atomic
  publish-on-commit semantics. (LTTng's implementation took years to
  shake out the ABA / sub-buffer-rotation / wraparound corner cases.)
* **RCU-protected per-tracepoint enabled flags** so the producer's
  hot-path check is one atomic load + branch and the
  enable/disable-from-outside mutation is wait-free for the producer.
* **Out-of-process atomic enable/disable protocol** with a session daemon
  the consumer talks to — including how the daemon discovers producers,
  how producers discover the daemon, and how the enable signal crosses
  the process boundary safely.
* **fd-passing for shm rings via UNIX domain sockets** (LTTng uses this
  to give the consumer daemon mmap access to the producer's per-CPU
  ring buffers without requiring root).
* **Sub-buffer rotation with backpressure / drop policy** (and the
  `--discard` vs `--overwrite` semantics that LTTng exposes — and an
  honest events-discarded counter so consumers know when they lost
  data).
* **Multi-consumer fanout** — N independent consumer sessions can
  subscribe to overlapping events from the same producer without each
  paying the full producer-side cost.
* **Live-streaming protocol** for cases where consumers want events as
  they happen (LTTng-live; analogous to `journalctl -f`).
* **Schema metadata + versioning** — without this, the consumer can't
  parse new event types or new fields on existing types without a
  coordinated producer/consumer release cycle.
* **File format library** for the offline case (event stream serialized
  to disk, readable later by a different tool than the live consumer).
* **Reader, viewer, indexer tools** — the Trace Compass / babeltrace2
  ecosystem, but written by us.

### Engineering cost

LTTng-UST is **~50,000 LOC** of producer + consumer code, plus the
~30,000 LOC of `userspace-rcu` it depends on, plus the ~200,000 LOC
of `babeltrace2` for the consumer-side reader. **All of this would have
to be re-implemented and re-validated by us** if we rolled our own.

The producer-side ring buffer alone is the kind of thing where every
non-obvious atomic-ordering bug ships as a customer-visible "we lost
events under contention" report years later. LTTng has been validated
in latency-critical applications (HFT, CERN ATLAS trigger system,
automotive ECUs) — that's the exact "low overhead, can't slow down"
profile rocprofiler-sdk needs, and someone else has paid the cost of
proving it works.

### Tooling ecosystem cost

LTTng emits **CTF (Common Trace Format)**, which is read by:

* `babeltrace2` — the canonical CTF reader/converter.
* **Trace Compass** — Eclipse-based GUI trace viewer with extensible
  analysis modules.
* **Perfetto** (which has a CTF ingest path) — Google's tracing UI,
  already a familiar tool to many ROCm customers via rocprofv3
  pftrace output.
* Custom CTF readers in Python, Rust, Go.

A custom format means **a custom toolchain**. Every tool integration
(crash dumpers, performance profilers, customer-internal observability
stacks) is paid for in cash. CTF gives us all of these for free.

### Customer-facing cost

Asking customers to **learn a new trace format** to consume rocprofiler
data — when their tooling already supports CTF (or they could trivially
add CTF support since open-source readers exist) — is a customer
training cost we'd have to keep paying forever.

### Maintenance cost

A custom transport becomes **another in-tree subsystem to maintain**
across distros, kernels, container baselines, and ASIC families. We
already have this matrix problem with KFD UAPI; adding a second
matrix-multiply for a tracing transport multiplies the validation
burden. LTTng-UST is upstream and someone else maintains its kernel /
userspace compatibility.

### Validation cost

Every "did we lose events?", "did the consumer crash safely?", "did
backpressure behave correctly under load?", "what happens when the
producer dies mid-write?", "what happens when the consumer's ring
buffer fills?" question has been answered for LTTng-UST already — by
people whose business depends on the answers being right. We do not
have an LTTng-UST validation team; we would have to build one to
maintain a custom transport.

### Summary

Rolling our own would deliver no property LTTng-UST doesn't already
deliver, would require thousands of LOC of producer + consumer code we
don't currently have headcount to maintain, and would force every
customer's tooling integration story to start from scratch. The cost
is high enough and the benefit is low enough that it was not seriously
pursued. The vendoring decision (LTTng-UST 2.13.7 + URCU 0.14.0 as git
submodules) captures essentially all of "rolling our own"'s
self-containment benefit at none of the engineering cost.

---

# Appendix: per-technique deep dives

The original survey produced a per-technique deep dive for each of the
eleven candidates. Those are preserved below as a reference for future
backend swaps and adversarial review of the chosen path. Each section
covers the technique's mechanism, producer / consumer requirements,
activation model, binary-modification behavior, OFF / ON overhead,
schema support, kernel/distro requirements, and (where applicable)
existing AMD/ROCm usage.

## 1. LTTng-UST (Linux Trace Toolkit Next Generation — User Space)

### A. Mechanism
Producer links `liblttng-ust`. Tracepoints are defined at compile time via
`LTTNG_UST_TRACEPOINT_EVENT()` macros in a tracepoint-provider header, with
typed fields. The macros expand to a function call that:
  1. checks a per-tracepoint, per-CPU enabled flag,
  2. if enabled, serializes the event into a per-CPU shared-memory
     sub-buffer (CTF format, lock-free producer side via the userspace RCU
     library `liburcu`).
The consumer is `lttng-sessiond` (session daemon, system or per-user) plus
`lttng-consumerd`, which mmap the same shared-memory ring buffers and stream
events to disk (CTF) or over the network (LTTng-live). Babeltrace 2 / Trace
Compass are the readers.

### B. Producer requirements
- Compile-time: include the generated tracepoint header, link
  `-llttng-ust -ldl`. Build a separate "tracepoint provider package" `.o`
  alongside the application code.
- The tracepoint definitions are statically declared. New event types require
  a code change and rebuild on the producer side.

### C. Consumer requirements
- `lttng-sessiond` must be running (system-wide as root, or per-user).
- The consumer side runs in **separate processes**. The instrumented
  application and the consumer communicate via shared memory + UNIX domain
  sockets (the shm fds are passed over the socket).
- Reader needs `liblttng-ctl` to control sessions; `babeltrace2` to read CTF
  output.

### D. Activation model — **YES, attach to running producer**
The application's `liblttng-ust` constructor registers with `lttng-sessiond`
on startup (via `LTTNG_HOME/.lttng/`). At any later point a tool can
`lttng create`, `lttng enable-event`, `lttng start` and the producer's
per-CPU enabled flags get flipped. **No code modification, no
LD_PRELOAD-equivalent, no restart needed.**

### E. Binary modification on activation — **NO**
Activation only flips a per-CPU enabled byte/word in the producer's address
space (which the producer reads on each tracepoint call). The producer's
`.text` is not modified.

### F. `LD_PRELOAD` requirement — **NO**
The producer must have been linked against `liblttng-ust` at build time.
Once it has been, no preload is needed. (LTTng does ship `liblttng-ust-dl`
and `liblttng-ust-libc-pthread-wrapper` as helpers usable via LD_PRELOAD,
but those are for instrumenting the dynamic linker / libc, not for the
core tracepoint mechanism.)

### G. Overhead when OFF
- 1 atomic load of the per-CPU enabled byte.
- 1 unlikely-branch.
- ~5 ns measured in upstream benchmarks.

### H. Overhead when ON
- Lock-free per-CPU ring buffer write of typed fields.
- ~100 ns per event measured in upstream benchmarks.
- Sub-buffers swapped atomically, consumer-daemon reads completed
  sub-buffers asynchronously.

### I. Data shape
**Strongly typed.** Each tracepoint is declared with `LTTNG_UST_TP_FIELDS()`
listing typed fields (integer, float, string, array, sequence, enum). The
schema is emitted as CTF metadata, separate from the event stream. Versioning
story: tracepoints can be deprecated/added; reader uses metadata to parse.

### J. Userspace stability
- `liblttng-ust` ABI was bumped at LTTng 2.13 (`lttng_ust_*` namespace) —
  recompile required.
- LTTng 2.12 → 2.13: applications need rebuild. Backwards-compat shim is
  available.
- CTF format is stable.
- `lttng-tools` CLI is stable across point releases.

### K. Multi-producer / multi-consumer
- **Multi-producer**: any process that linked `liblttng-ust` is a producer.
  Sessiond aggregates.
- **Multi-consumer**: a single sessiond's recording-session output goes to
  one set of consumers. Multiple sessions can target overlapping events
  (`buffer-shared` vs `buffer-uid` vs `buffer-pid` schemes), so two
  independent sessions can subscribe to the same producer.

### L. Kernel dependency
- LTTng-UST: **none kernel-side**. Pure userspace tracing.
  (Distinct from LTTng-modules, which is the kernel tracer.)
- Distros: Ubuntu 22.04+, Debian 12+, Fedora 37+, Arch (extra), Alpine 3.16+
  ship LTTng 2.13. RHEL/SLES via EfficiOS Enterprise Packages.
- Run-time dep on `liburcu` removed in LTTng 2.13 (`liburcu` only at build
  time now).

### M. AMD/ROCm-specific usage
**None documented.** AMD search confirms rocprofiler-sdk does not integrate
with LTTng. rocprof outputs to its own formats (rocpd SQLite, JSON, CSV,
PFTrace, OTF2). HPCToolkit and Score-P consume rocprofiler-sdk via its
existing callback API.

---

## 2. OpenTelemetry (OTel) SDK + OTLP exporters

### A. Mechanism
Producer links a language-specific OTel SDK (`opentelemetry-cpp` for C++).
Code calls `Tracer::StartSpan()`, `Span::SetAttribute()`, etc. Spans/metrics/
logs are processed by configured **processors** (BatchSpanProcessor,
SimpleSpanProcessor) and shipped to **exporters**. Exporters typically send
**OTLP** (gRPC or HTTP/protobuf) to an **OpenTelemetry Collector** running
out-of-process. The Collector then routes to backends (Jaeger, Prometheus,
Tempo, Grafana, vendor SaaS, etc.).

### B. Producer requirements
- Link `opentelemetry-cpp` (and Abseil, Protobuf, gRPC if using OTLP/gRPC
  exporter — significant dependency tree).
- Configure a tracer provider, exporter, sampler at startup.
- Make explicit OTel API calls on the hot path.

### C. Consumer requirements
- An OTel Collector or compatible OTLP receiver process.
- Consumer is **always out-of-process** (network or UNIX socket).
- Not privileged; just needs network reach.

### D. Activation model — partially YES
- The SDK is in-process. Consumer (Collector) can be started/stopped
  independently. If exporter has a queue and Collector is down, events are
  dropped or backpressure builds.
- However, the SDK is **always running** if linked: there is no clean "off"
  state where the producer pays zero cost. Sampling can reduce per-event
  cost but the API call itself is taken.

### E. Binary modification on activation — **NO**

### F. `LD_PRELOAD` requirement — **NO** (linked at build)

### G. Overhead when OFF
There is no true "off". Even with a NoopTracer, the API call still happens
(virtual dispatch, attribute construction) unless the producer wraps every
call site in a `static` config flag. The OTel C++ SDK has historically not
been zero-cost when "disabled".

### H. Overhead when ON
- Span creation: ~hundreds of ns to low µs (creates a Span object, attribute
  vectors).
- Export: serialized to protobuf, sent over gRPC/HTTP. Batched in the
  BatchSpanProcessor (default ~5s flush).
- Heavyweight relative to LTTng-UST.

### I. Data shape
Strongly typed via OTel semantic conventions; OTLP is protobuf-defined.
Versioning is part of the OTel spec lifecycle.

### J. Userspace stability
OTLP protocol is stable. C++ SDK is at 1.x and stable. Heavy dependencies
(Protobuf, gRPC, Abseil) are a real distribution-portability concern.

### K. Multi-producer / multi-consumer
- Multi-producer: trivially yes — any process that runs OTel is a producer
  shipping to a Collector.
- Multi-consumer: yes — Collector fans out to N exporters.

### L. Kernel dependency
None.

### M. AMD/ROCm-specific usage
- **No native OTLP exporter in rocprofiler-sdk.** Output formats are rocpd
  SQLite, CSV, JSON, Perfetto/PFTrace, OTF2.
- AIM Engine (AMD inference/serving stack) uses OpenTelemetry as a sidecar
  collector for vLLM metrics — this is upstream OTel use, not in
  rocprofiler-sdk.

---

## 3. uprobes (kernel-installed user-space probes)

### A. Mechanism
The kernel uprobes subsystem (CONFIG_UPROBE_EVENTS) installs **software
breakpoints (INT3 / 0xCC on x86_64)** at a specified file:offset (or
function symbol). Probes are armed by writing to
`/sys/kernel/tracing/uprobe_events` or via `perf_event_open` with
`type=PERF_TYPE_UPROBE`. When a probed thread reaches the INT3, it traps,
the kernel handler runs (recording the event into ftrace ring buffer or
firing eBPF), and then the original instruction is single-stepped
out-of-line (XOL) before returning to userspace.

### A.1. Implementation detail (critical for our constraint)
> "The uprobe registration mechanisms record the current instruction, and
> then place a software breakpoint (on x86_64 an INT3) at the specified
> offset in each VMA associated with the specified inode."
>
> "On an x86 architecture when a uprobe is armed, the kernel will dynamically
> replace the target address with an INT3 (0xCC) instruction."

The kernel uses copy-on-write to avoid modifying the on-disk binary, but the
**in-memory text page is modified** when the probe is enabled.

### E. Binary modification on activation — **YES (FATAL)**
Arming a uprobe **rewrites the producer's text segment** (the in-memory
copy). This violates the second hard constraint. **Eliminated.**

### Other notes (for completeness)
- Activation: no LD_PRELOAD; tracefs ioctl from outside.
- Steady-state OFF cost: zero (text is unmodified).
- Consumer: tracefs reader, perf, bpftrace, libbpf.
- Data: unstructured; user must declare which registers/memory to fetch
  (`%ax`, `+8(%rdi):string`, etc.) — fragile across compiler versions and
  optimization levels.
- Kernel: 3.5+; widely shipped.

**Verdict: Eliminated by hard constraint #2.**

---

## 4. eBPF uprobes (`bpf_program__attach_uprobe`)

### A. Mechanism
Same physical attach mechanism as bare uprobes (INT3 in the producer's text)
but the handler is a verified eBPF program loaded into the kernel. The eBPF
program reads register/memory state from `pt_regs`, can call BPF helpers,
write into BPF maps or ring buffers (`BPF_RB_FORCE_WAKEUP`), and the
consumer reads from those maps from userspace.

The "attach" mechanics are identical to uprobes: the kernel installs an INT3
in the producer's text page when the probe is enabled.

### E. Binary modification on activation — **YES (FATAL)**

eBPF uprobes share the uprobes subsystem's INT3-injection model. Same
violation. **Eliminated** for the same reason as #3.

There is one nuance: **USDT-via-eBPF** (`bpf_program__attach_usdt`) attaches
to compile-time-emitted NOPs. The NOPs were already in the binary at compile
time; activation flips them to INT3 (or arms the uprobe at the NOP offset).
So USDT-via-eBPF still arms an INT3 at the NOP location. The text **is**
modified on enable, even though the slot was a benign NOP before.

**Verdict: Eliminated by hard constraint #2** for both raw uprobes and
USDT-via-eBPF that uses the uprobes subsystem under the hood.

---

## 5. USDT (Userspace Statically Defined Tracepoints) — DTrace-style probes

### A. Mechanism
Producer is compiled with `STAP_PROBEn()` / `DTRACE_PROBEn()` macros. Each
macro expands to:
  - A NOP instruction in the text (a literal `nop` byte).
  - An ELF note (`stapsdt`) recording the probe name, location, and the
    register/stack locations of the probe arguments.
  - Optionally, a "semaphore" (a 16-bit int in `.probes` data) the consumer
    flips to tell the producer "someone is listening" so the producer can
    skip expensive argument computation.

Consumer (SystemTap, bpftrace, perf, libbpf USDT helpers) reads the ELF
notes from the binary, finds the NOP locations, and arms a uprobe at each
NOP. When the probe fires, the consumer's handler runs.

### B. Producer requirements
- Compile-time changes: header include + `STAP_PROBEn()` macros at hot-path
  sites. Build dep: `systemtap-sdt-dev` (provides `<sys/sdt.h>`).
- Probe definitions can be in a `.d` file processed by the `dtrace -h`
  utility (this is just a header generator, not the DTrace runtime).
- Producer **does not link any tracing library**. The macros expand to
  inline assembly emitting a NOP and a section note.

### C. Consumer requirements
- Out-of-process (perf / bpftrace / SystemTap / libbpf).
- Needs uprobes available in the kernel.

### D. Activation model — YES
USDT probes are inert NOPs at runtime when nobody is attached. A consumer
attaches by reading the binary's ELF notes and arming uprobes at the NOP
offsets. **The application does not need to be relaunched.**

### E. Binary modification on activation — **YES when consumer attaches** (FATAL)
The NOP at the probe site is replaced with INT3 by the uprobes subsystem
when the consumer arms the probe. While the probe is inert when nobody is
attached (so on-CPU cost is exactly 1 NOP, ~zero), the act of subscribing
modifies the producer's text segment.

This violates hard constraint #2 unless we redefine the constraint as "no
binary modification *if no consumer is attached*" (which would be
satisfied) — but the user explicitly framed the constraint as "techniques
that rewrite text segments... at attach time are out".

**Verdict: Eliminated by hard constraint #2** (text modification happens at
the moment any consumer subscribes).

> Note: USDT is *very* close to what the user wants in spirit (compile-time
> static probes, generic emission, third-party subscribe out-of-process).
> The disqualifying factor is the kernel's chosen implementation
> (uprobes/INT3) — not the USDT model itself. If the constraint is
> renegotiated to "no in-place patching of code that *was not* a
> compile-time-reserved tracing slot", USDT comes back into play.

---

## 6. `perf_event_open(2)` + tracepoints

### A. Mechanism
A general-purpose syscall to create a file descriptor representing a
performance event. The event can be:
- Hardware (CPU cycles, cache misses)
- Software (PERF_COUNT_SW_*)
- Tracepoint (kernel tracepoints by `id`)
- Dynamic (kprobe/uprobe — see §3/§4)

Events are sampled into an mmap'd ring buffer or counted via `read()`.
Consumer (typically `perf` or libperf) opens the fd against a `pid` and
optional `cpu`.

### Notes for our problem
- Useful for *consuming* events from any source (kernel tracepoints,
  USDT-via-uprobes, etc.). Not itself a producer-side emit-data API.
- For our problem, perf_event_open is the **consumer** API that pairs
  with USDT (`perf probe`) or with kernel tracepoints. Not relevant on
  the producer side.

### Verdict
Not a delivery technique on its own; it is a consumer transport that
can be paired with kernel tracepoints. Mention but not a standalone
option.

---

## 7. Linux kernel tracepoints exported via `/sys/kernel/tracing`

### A. Mechanism
Static tracepoints compiled into the **kernel** (`TRACE_EVENT()` macros in
kernel source). Exposed via tracefs. Consumed by ftrace, perf, eBPF.

### Relevance to our problem
- Only useful if HIP/HSA's hooks are in **kernel** code paths (KFD ioctls,
  AMDGPU driver). The userspace HIP/HSA runtimes can't add kernel
  tracepoints.
- KFD/AMDGPU could add kernel tracepoints (some already exist in
  drivers/gpu/drm/amd/) for ioctl-level events but this gives only the
  kernel-side view, not the userspace runtime API events.

### Verdict
Useful for a complementary kernel-side trace stream (KFD ioctl tracing) but
does not solve the userspace HIP/HSA → rocprofiler-sdk delivery problem.

---

## 8. LTTng-modules vs LTTng-UST (clarification)

- **LTTng-modules**: kernel tracer. Out-of-tree kernel module. Replaces
  ftrace as a tracing backend in some setups. Not relevant to userspace
  delivery.
- **LTTng-UST**: userspace tracer (covered in detail in #1).

These are independent components; either can be used without the other.
LTTng-UST does not require LTTng-modules to be installed.

---

## 9. Chrome trace event format (file-based)

### A. Mechanism
Producer emits JSON or protobuf records to a file (or a Perfetto SDK shared
memory buffer). Consumer is offline — the file is loaded into Chrome
tracing UI or Perfetto UI for visualization.

### Relevance
This is a **trace output format**, not a delivery mechanism between a live
producer and a live consumer. rocprofv3 already supports `--output-format
pftrace` for offline visualization. Useful as a sink, not as the primary
HIP/HSA → rocprofiler-sdk channel.

### Verdict
Not a candidate for live delivery. Already supported as an output of
rocprofiler-sdk for visualization.

---

## 10. io_uring channels (`IORING_OP_MSG_RING` and proposed IPC)

### A. Mechanism
io_uring is a shared SQ/CQ ring between userspace and kernel. Since 5.18+,
`IORING_OP_MSG_RING` allows one io_uring instance to post a 64-bit `user_data`
message into another instance's CQ. Recently proposed IPC subsystem
(`IORING_REGISTER_IPC_CHANNEL_CREATE`, `IORING_OP_IPC_SEND/RECV`) extends
this to a more general N-process IPC channel.

### B. Producer requirements
- `io_uring_setup()` to get a ring fd at startup.
- `io_uring_prep_msg_ring()` + `io_uring_submit()` per event.
- Or shared-memory pool keyed by 64-bit user_data values.

### C. Consumer requirements
- Own io_uring instance.
- Permission to acquire the producer's ring fd. **Yama LSM blocks
  cross-process fd acquisition by default** without root or
  `PTRACE_MODE_ATTACH_REALCREDS`.

### D. Activation model — partially YES
The producer's ring is always present once created. A late-attaching
consumer needs to obtain the producer's ring fd, which requires either
(a) a pre-arranged out-of-band fd-passing (UNIX socket SCM_RIGHTS, which
implies LD_PRELOAD or pre-launch coordination — **violates constraint
#1**), (b) `pidfd_getfd(2)` with appropriate ptrace credentials
(privileged), or (c) the new IPC subsystem's named-channel discovery
(still upstream-only, not in any shipping kernel).

### E. Binary modification on activation — **NO**

### F. `LD_PRELOAD` requirement — borderline
For a generic late-attach without prior coordination, the consumer needs
privilege (CAP_SYS_PTRACE / pidfd_getfd) or a named channel that doesn't
exist yet in stable kernels.

### G. Overhead when OFF
The producer can't easily tell if anyone is consuming. It will still post to
its ring (which lives in shared memory). Cost is ~1 atomic write to the
producer's own SQ. If the SQ overflows because nobody is draining, the
producer needs an explicit drop policy.

### H. Overhead when ON
Very low — ~50–100 ns per submission, comparable to a function call. 20M
no-op messages/sec measured upstream.

### I. Data shape
Raw 64-bit user_data per message. For larger payloads, the producer
allocates from a shared-memory pool and passes the pool index as user_data.
**No built-in schema or versioning.**

### J. Userspace stability
io_uring API is stable; `MSG_RING` is stable since 5.18. The proposed IPC
subsystem is not yet stable.

### K. Multi-producer / multi-consumer
N→1 (multiple producers post to one consumer ring) is supported. 1→N
(one producer fans out to multiple consumers) requires the producer to know
about each consumer's ring — this **reintroduces the same coupling as the
current callback design**, defeating the purpose.

### L. Kernel dependency
- 5.18+ for `MSG_RING`.
- 6.0+ for full feature set.
- Proposed IPC subsystem: not yet upstream.

### M. AMD/ROCm-specific usage
None documented.

### Verdict
Conceptually attractive for in-process or pre-coordinated cross-process,
but the **late-attach generic-discovery story is weak** without the
upstream IPC subsystem. Not the right fit for "any tool can subscribe to
HIP/HSA without prior arrangement". Could be useful as a *secondary*
channel between rocprofiler-sdk and a tool consumer, after the primary
HIP/HSA → rocprofiler-sdk channel is solved.

---

# Open Questions / Things to Validate

1. **Real-world LTTng-UST overhead on the HIP hot path.** Upstream cites
   ~100 ns per event when ON, ~5 ns when OFF. The HIP dispatch hot path
   already has 1248 ns budget eaten by `pool::acquire` (per
   `AIPROFSDK-813-always-on-tracing.md`). Need to measure: does adding an
   LTTng-UST tracepoint per dispatch (kernel_object, grid, wg, signal,
   correlation_id) push us over the existing 2.95×–3.55× envelope?
   Hypothesis: no, because ~100 ns is small relative to 1248 ns.

   > **Status (2026-04-30): RESOLVED — measured.** GraphBench, MI325X gfx942,
   > 12 reps, `--discard --subbuf-size=65536 --num-subbuf=4`: **+0.6%
   > wall-time** for generic-only (14 event types), **+0.9%** for generic +
   > 83 typed `_args` events (97 event types, ~10M events captured), **0
   > discarded events** in either configuration. The hypothesis held —
   > tracepoint cost is well below the existing per-dispatch noise floor.
   > Same workload under `rocprofv3 --hsa-trace` at full fidelity: **+73%
   > (~80× more expensive)**.

2. **CTF-side correlation IDs.** rocprofiler-sdk's callback API gives
   typed correlation IDs (internal + external). LTTng's tracepoint fields
   handle this (it's just two u64 fields). Verify the consumer side
   (rocprofiler-sdk reading CTF) can join async-signal-handler-emitted
   completion records with API-boundary records efficiently.

   > **Status (2026-04-30): RESOLVED — by dropping in-band corr_id entirely
   > (schema v3).** Producers carry no correlation identifiers. Consumers
   > reconstruct enter/exit pairing and parent attribution by walking the
   > per-`(vpid, vtid)` event stream sorted by CTF event-header timestamp
   > using a LIFO stack. Cross-runtime parents (HIP → HSA on the same
   > thread) merge naturally because both providers share the same
   > per-thread call stack under this scheme. The `(queue_id, write_idx)`
   > fields on `hip_aql_kernel_dispatch_submit` serve as the natural join
   > key for the firmware-ring track's GPU completion records. ~530 LOC
   > and the entire `librocprofiler-register/correlation` ABI surface were
   > deleted as part of this decision. **Strictly more correct for
   > multi-process tracing than a per-process counter would be.**

3. **Multi-tool subscriber semantics.** Today, rocprofiler-sdk supports N
   tools subscribing to HIP/HSA events via its context API. With LTTng,
   each tool would either (a) be its own LTTng consumer with its own
   recording session (heavy), or (b) all flow through rocprofiler-sdk
   which subscribes once and re-fans-out. Option (b) preserves current
   semantics; (a) would be a behavior change. Pick one and commit.

   > **Status (2026-04-30): RESOLVED — option (a) selected.** LTTng-UST
   > natively supports N-consumer multiplex via `buffer-shared` /
   > `buffer-uid` / `buffer-pid` schemes; no rocprofiler-sdk re-fan-out
   > layer is needed. The behavior change vs today is intentional and
   > aligns with the "generic emit, anyone subscribes" goal of this work.

4. **How does rocprofiler-sdk consume CTF live?** LTTng-live (TCP
   protocol; networked CTF stream) is the obvious answer. Validate that
   rocprofiler-sdk's consumer thread can keep up with high-rate dispatch
   workloads (1000+ kernels/s sustained). Babeltrace 2 can also read live
   CTF buffers via shared-memory bridges.

   > **Status (2026-04-30): DEFERRED — separate planned PR.** The
   > rocprofiler-sdk consumer-side CTF→`rocprofiler_*_record_t` translator
   > is not yet started. Today, any tool that wants LTTng-sourced events
   > consumes via `lttng create` + `lttng enable-event` + `babeltrace2`
   > directly — no rocprofiler-sdk involvement required. The translator is
   > scoped roughly at "consume CTF live via libbabeltrace2 + LTTng-live
   > protocol; emit existing `rocprofiler_*_record_t` shapes" so existing
   > tools subscribing via the rocprofiler-sdk callback API see
   > LTTng-sourced events transparently.

5. **Build/packaging implications.** Adding `liblttng-ust` as a build dep
   for HIP/HSA touches the build matrix. Verify availability on:
   - Ubuntu 22.04 / 24.04: in distro repo (`liblttng-ust-dev`).
   - RHEL 9 / 10: via EfficiOS Enterprise Packages or EPEL.
   - SLES 15: EPEL or EfficiOS.
   - The internal AMD container baselines.

   > **Status (2026-04-30): RESOLVED — by vendoring.** LTTng-UST 2.13.7 and
   > userspace-rcu 0.14.0 are vendored as git submodules under
   > `projects/{clr,rocr-runtime}/external/`, built via
   > `ExternalProject_Add`, and installed flat into `/opt/rocm/lib/`. **No
   > system `liblttng-ust-dev` required at build time or runtime.** The
   > distro-reach question is dissolved for the primary backend. Build-side
   > the only new hard deps are autotools (`autoconf`, `automake`,
   > `libtool`, `libtool-bin`, `pkg-config`, `patchelf`); these are
   > universally available. Validated on TheRock manylinux container
   > (`bewelton_therock`).

6. **The "tracing of intercept queues" interaction.** Today HSA queue
   intercept is the cheapest at-fidelity option (`rtl_full` at 2.95×). If
   we move HIP/HSA to LTTng-UST emit, do we also rip out the queue
   intercept path or run both? Both have different fidelity properties
   for graphs.

   > **Status (2026-04-30): COEXIST.** The producer always emits
   > tracepoints regardless of whether anyone is subscribed. The
   > firmware-ring drainer (the kernel-dispatch-timestamp side of this
   > overall design) remains optional and orthogonal — different layer.
   > The legacy queue-intercept path inside rocprofiler-sdk is unchanged
   > by this PR; it can be removed in a later cleanup once the
   > rocprofiler-sdk LTTng consumer ships.

7. **Re-negotiate the USDT constraint?** Confirm whether the hard
   constraint really means "no binary mod ever" or "no binary mod of code
   that wasn't a deliberate tracing slot". If the latter, USDT (and
   USDT-via-eBPF for the consumer side) re-enters the candidate set as a
   strong contender — minimal producer changes, zero runtime library dep,
   eBPF/perf/SystemTap/bpftrace as ready-made consumers.

   > **Status (2026-04-30): NOT PURSUED.** LTTng-UST landed first with
   > acceptable measured overhead. The constraint stands as originally
   > stated. The USDT renegotiation can be revisited in a future v2 design
   > if specific tooling needs (e.g., bpftrace-native subscribers without
   > liblttng dependency) emerge as a customer requirement.

---

# Sources

- LTTng v2.13 documentation: https://lttng.org/docs/v2.13/
- Linux kernel uprobetracer: https://docs.kernel.org/trace/uprobetracer.html
- `perf_event_open(2)` man page: https://man7.org/linux/man-pages/man2/perf_event_open.2.html
- BPF iterators: https://docs.kernel.org/bpf/bpf_iterators.html
- OpenTelemetry components: https://opentelemetry.io/docs/concepts/components/
- Brendan Gregg, "Hacking Linux USDT with Ftrace" (2015): https://www.brendangregg.com/blog/2015-07-03/hacking-linux-usdt-ftrace.html
- AMD search results (no documented LTTng/USDT/OTel/eBPF integrations in rocprofiler-sdk; see this file's Notes section for inline citations)
- Local context: `~/ai/task_info/AIPROFSDK-813-always-on-tracing.md`,
  `~/ai/task_info/sideband-tracing-approaches.md`
