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

## Today's situation (baseline for comparison)

HIP and HSA runtimes today call into rocprofiler-sdk via function pointers
populated through `rocprofiler-register`. The runtimes know about
rocprofiler-sdk (via the shared API table contract); rocprofiler-sdk receives
data **synchronously on the producer's thread** during the dispatch hot path.
This gives full fidelity but couples the runtimes to the consumer and forces
producer-side overhead on every event regardless of whether anyone is
subscribed (the existence of the shared table is a non-zero cost — see
`AIPROFSDK-813-always-on-tracing.md` for measured 2.77×–3.55× overhead).

The desired property is: HIP/HSA **emit events generically** into a transport
that does not name a consumer, and rocprofiler-sdk (or any other tool)
**subscribes** to that transport. Steady-state cost when nobody is subscribed
should be ~one branch per event.

---

## TL;DR — Comparison Matrix

| Property | LTTng-UST | OpenTelemetry SDK | uprobes (kernel-installed) | eBPF uprobes | USDT (DTrace probes) | Linux `user_events` (6.4+) | `perf_event_open` + tracepoints | Chrome trace / Perfetto file | io_uring channels | Today's `rocprofiler-register` |
|---|---|---|---|---|---|---|---|---|---|---|
| Works without `LD_PRELOAD`? | **YES** (linked at build) | **YES** (linked at build) | **YES** (probes set externally) | **YES** | **YES** | **YES** (linked at build) | **YES** | **YES** (file-based) | **YES** | YES (rocprofiler-register linked into HIP/HSA) |
| Works without runtime binary modification? | **YES** (no text patch) | **YES** | **NO** — INT3 (0xCC) overwrite at attach time | **NO** — same INT3 attach mechanism | **YES** when nobody attached (NOP); **NO** when attached (NOP→INT3) | **YES** (no text patch — bit flip in registered word) | **YES** for software events; depends on probe type | **YES** | **YES** | YES |
| Producer-side compile-time changes | tracepoint provider macros + link `liblttng-ust` | OTel SDK calls + link OTel | none (probes set externally) | none, OR USDT macros for static probes | `STAP_PROBEn()` / `DTRACE_PROBE` macros, link `libstapsdt-dev` | `ioctl(DIAG_IOCSREG)` + `writev()` to a tracefs fd | none (consumer attaches externally) | call into Perfetto SDK or write JSON | `io_uring_setup()` + `IORING_OP_MSG_RING` | populate API table function pointers |
| Consumer attach model | sessiond (out-of-process); CTF reader | OTel collector (out-of-process); OTLP gRPC/HTTP | tracefs / perf / bpftrace (out-of-process) | bpftrace / BCC / libbpf reader (out-of-process) | bpftrace / SystemTap / perf (out-of-process) | tracefs / perf (out-of-process) | perf or libperf (out-of-process) | open the trace file | another io_uring (in or out of process) | in-process callback |
| Steady-state overhead when OFF | ~1 load + 1 branch per tracepoint (per-CPU enabled flag) | non-zero — usually a sampled-flag check; SDK is not a true zero-cost design | **0** (attach modifies code only when enabled) | **0** | ~1 NOP (literally compiled-in NOP) | ~1 load + 1 bit test of the registered enable_addr | depends on probe type | per-call cost (no on/off) | per-call cost | 1 load + 1 branch per event |
| Per-event overhead when ON (producer side) | ~100 ns (per-CPU ring buffer write, lock-free) | µs range (SDK + queueing + serialization) | µs (INT3 + kernel trap + kprobe handler + uprobe-XOL) | µs (same INT3 + eBPF program execution) | µs (NOP→INT3 trap path, same as uprobe) | ~100–300 ns (one writev() syscall) | µs (perf record path) | µs (file write or shmem proto serialization) | ~50–100 ns (shared-mem ring) | 100s of ns–µs (function call + correlation work) |
| Data shape / schema | strongly typed; CTF metadata; versioned | strongly typed; OTel semantic conventions | unstructured (raw register/memory values) | structured if eBPF program parses; otherwise raw | semi-typed (probe args declared in `.d` file) | strongly typed (registered struct schema) | semi-typed | typed (Perfetto protos) or untyped JSON | raw bytes (consumer interprets) | strongly typed (C structs in API table) |
| Kernel/distro requirements | none kernel-side for UST; userspace daemon | none | uprobes: 3.5+; widely shipped | eBPF: 4.18+ for libbpf; widely shipped on modern distros | none (NOPs are inert); consumer needs uprobes | **6.4+** with `CONFIG_USER_EVENTS=y` (NOT default on RHEL 9, Ubuntu 22.04) | 2.6.31+; widely shipped | none | 5.18+ for `MSG_RING`; 6.0+ for full feature; not default | n/a |
| Existing AMD/ROCm usage | none documented | none in rocprofiler-sdk; AIM Engine sidecar uses OTel for vLLM metrics only | none in rocprofiler-sdk | none in rocprofiler-sdk; GPUprobe uses on CUDA externally | none documented | none documented | none for HIP/HSA delivery; rocprofiler-sdk uses HSA queue intercept | rocprofv3 emits Perfetto/`pftrace` as a final output format | none | yes — current design |

The two columns that survive both hard constraints AND have a usable
producer-side cost when OFF are **LTTng-UST**, **`user_events`**, **USDT**
(when nobody attached), **OpenTelemetry**, and **io_uring channels**. The rest
fail at least one hard constraint or have unacceptable always-on cost.

---

# Per-technique deep dives

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
- Heavyweight relative to LTTng-UST or `user_events`.

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

## 6. Linux `user_events` (kernel 6.4+) — **the modern dark horse**

### A. Mechanism
A new tracefs ABI introduced upstream in Linux 6.4 (Beau Belgrave,
Microsoft). Producer:
  1. Opens `/sys/kernel/tracing/user_events_data`.
  2. Issues `ioctl(DIAG_IOCSREG, &user_reg)` to register an event. The
     `user_reg` struct includes:
     - `name_args`: event name + typed-field schema string
       (`"my_event u32 thread_id; u64 timestamp; char name[64]"`).
     - `enable_addr`: a 32- or 64-bit word in the *producer's own address
       space* that the kernel will atomically set/clear bits in.
     - `enable_bit`: which bit in that word reflects "someone is consuming".
  3. The kernel returns a `write_index` — an integer the producer must
     prepend to event payloads.

Producer hot-path: load the `enable_addr` word, test the bit, branch out if
zero, otherwise `writev(fd, &write_index, &payload)`.

Consumer attaches via standard tracefs:
`echo 1 > /sys/kernel/tracing/events/user_events/<name>/enable` or via
`perf record -e user_events:<name>` or via eBPF tracepoint program. When a
consumer enables, the kernel atomically sets the producer's bit; when the
last consumer detaches, the kernel clears it.

### B. Producer requirements
- One-time `open()` + `ioctl()` at startup per event.
- One `writev()` per event when enabled (or just a `write()`).
- No library link required (could be a vendored helper). Total LoC for a
  C wrapper: ~150 lines.

### C. Consumer requirements
- tracefs mounted (`/sys/kernel/tracing`); CAP_PERFMON if event registered
  with `USER_EVENT_REG_PERSIST`.
- Standard ftrace / perf / bpftrace. **Out-of-process.**

### D. Activation model — **YES, attach to running producer**
Producer registers events at startup. A consumer can attach hours later via
tracefs and the producer's enable bit gets flipped. Detach symmetric.

### E. Binary modification on activation — **NO**
Activation flips a bit in a registered word in the producer's data segment.
**The text segment is never touched.** This is the critical differentiator
from USDT/uprobes.

### F. `LD_PRELOAD` requirement — **NO**
Producer registers via syscall during normal startup. The consumer attaches
via syscall.

### G. Overhead when OFF
- 1 atomic load of the registered word.
- 1 bit test + unlikely branch.
- Comparable to LTTng-UST's per-CPU enabled flag check.

### H. Overhead when ON
- 1 `writev()` syscall per event with the registered struct.
- Syscall overhead (~100–300 ns on modern hardware) plus the kernel-side
  ftrace ring-buffer write.
- Higher per-event cost than LTTng-UST's per-CPU shared-memory write,
  because every event crosses the kernel boundary.
- Mitigated by event batching at the producer (combine multiple events into
  one writev) — supported by the ABI.

### I. Data shape
Strongly typed via the registration string. Schema includes basic types
(u8/u16/u32/u64, char, char[N], `__data_loc`, `struct mytype N`).
**Versioning**: `USER_EVENT_REG_MULTI_FORMAT` flag (since 6.x) allows the
same event name to coexist with multiple format versions, each tracked via
a unique-id suffix in tracefs (`event.<hex>`). This is a real schema
versioning story, unique among the kernel-side options.

### J. Userspace stability
Stable upstream ABI as of 6.4 with continued additions (multi-format,
persist flag). The `enable_bit`/`enable_addr` model is the second
iteration; the original 6.0–6.3 design used a shared mmap'd page and was
removed in favor of the registered-address-write model.

### K. Multi-producer / multi-consumer
- Multi-producer: any process can register events with the same name.
  With `USER_EVENT_REG_MULTI_FORMAT`, format collisions are handled per-id.
  Without it, two producers with the same name+schema share the same
  tracepoint.
- Multi-consumer: standard tracefs — N consumers can enable, refcounted by
  the kernel; the bit stays set until all detach.

### L. Kernel dependency
- **CONFIG_USER_EVENTS=y** required.
- **6.4+** for the registered-address API. The earlier shared-page API is
  not what tools target now.
- Distro reality (as of 2026):
  - Mainline / Arch / Fedora 39+: enabled.
  - Ubuntu 24.04 (kernel 6.8): enabled.
  - Ubuntu 22.04 (kernel 5.15): **NOT available**.
  - RHEL 9 (kernel 5.14): **NOT available**.
  - RHEL 10 (kernel 6.12+): expected to ship enabled.
  - SLES 15 SP6: depends on kernel revision.
  - Container images: must run on a host kernel with the feature.

This is the **single biggest practical risk** of `user_events`: ROCm
customers on RHEL 9 / Ubuntu 22.04 LTS / older SLES will not have it for
several more years.

### M. AMD/ROCm-specific usage
**None documented.** Search confirms no rocprofiler / rocprof / rdc usage of
user_events.

---

## 7. `perf_event_open(2)` + tracepoints

### A. Mechanism
A general-purpose syscall to create a file descriptor representing a
performance event. The event can be:
- Hardware (CPU cycles, cache misses)
- Software (PERF_COUNT_SW_*)
- Tracepoint (kernel tracepoints by `id`)
- Dynamic (kprobe/uprobe — see #3/#4)

Events are sampled into an mmap'd ring buffer or counted via `read()`.
Consumer (typically `perf` or libperf) opens the fd against a `pid` and
optional `cpu`.

### Notes for our problem
- Useful for *consuming* events from any source (kernel tracepoints,
  user_events, USDT-via-uprobes, etc.). Not itself a producer-side
  emit-data API.
- For our problem, perf_event_open is the **consumer** API that pairs with
  user_events (a tool can do `perf record -e user_events:my_event`) or
  with USDT (`perf probe`).

### Verdict
Not a delivery technique on its own; it is a consumer transport that can be
paired with user_events (the recommended candidate) or with kernel
tracepoints. Mention but not a standalone option.

---

## 8. Linux kernel tracepoints exported via `/sys/kernel/tracing`

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

## 9. LTTng-modules vs LTTng-UST (clarification)

- **LTTng-modules**: kernel tracer. Out-of-tree kernel module. Replaces
  ftrace as a tracing backend in some setups. Not relevant to userspace
  delivery.
- **LTTng-UST**: userspace tracer (covered in detail in #1).

These are independent components; either can be used without the other.
LTTng-UST does not require LTTng-modules to be installed.

---

## 10. Chrome trace event format (file-based)

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

## 11. io_uring channels (`IORING_OP_MSG_RING` and proposed IPC)

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

# Final Comparison Matrix (constraint-focused)

| Technique | No LD_PRELOAD | No binary mod on activate | OFF overhead acceptable | Schema support | Late-attach | Distro reach today |
|---|---|---|---|---|---|---|
| **LTTng-UST** | YES | **YES** | YES (~5 ns) | YES (CTF) | YES | Excellent (all major distros) |
| **`user_events` (Linux 6.4+)** | YES | **YES** | YES (~5 ns) | YES (typed schema + multi-format) | YES | Poor today (RHEL 10 / Ubuntu 24.04+ only) |
| **OpenTelemetry C++ SDK** | YES | YES | NO (no zero-cost off) | YES (semconv) | partial | Good (heavy deps) |
| **io_uring channels** | borderline | YES | YES | NO | borderline | Fair (5.18+) |
| USDT (DTrace-style) | YES | **NO** (text patched at attach) | YES (NOP-only) | partial | YES | Excellent |
| Raw uprobes | YES | **NO** (INT3 patch) | YES (zero) | NO (raw fetchargs) | YES | Excellent |
| eBPF uprobes | YES | **NO** (INT3 patch) | YES (zero) | YES (BPF map schema) | YES | Excellent |
| Chrome/Perfetto file | n/a | YES | n/a (offline) | YES | n/a | Already used as output |
| `perf_event_open` | n/a | depends on probe | depends | depends | YES | Excellent |
| Today's `rocprofiler-register` | YES | YES | NO (always-on cost) | YES (C structs) | YES (in-process) | n/a |

**Survives both hard constraints + acceptable when-OFF cost + late-attach:**
- LTTng-UST
- `user_events` (subject to kernel availability)
- io_uring channels (subject to discovery problem)

---

# Recommendation

## Primary: **LTTng-UST**

Rationale:
1. **Both hard constraints satisfied unambiguously.** No LD_PRELOAD, no
   text-segment modification at any point.
2. **Production-grade quality.** 10+ years of upstream development, used by
   high-frequency-trading firms, CERN, automotive, telecom — i.e., the
   exact "low overhead, can't slow down" profile rocprofiler-sdk needs.
3. **Strongly typed schema** (CTF) with versioning and language bindings.
   Important for the rocprofiler-sdk record contract (we want HIP_API_ID,
   correlation_id, kernel_object, grid/wg/segments — all typed).
4. **~5 ns overhead when OFF**; this matches the user's request for a
   technique where the producer pays ~one branch when nobody subscribes.
5. **Late-attach works** out of the box. The HIP/HSA-instrumented runtimes
   register tracepoints at load time; a tool runs `lttng enable-event`
   later and the per-CPU enable byte flips. No restart, no LD_PRELOAD.
6. **Already shipped on every major distro** that ROCm targets, and has been
   for many years.
7. **Out-of-process consumer model** is a perfect match for the user's
   architectural goal: HIP/HSA emit generically, *any* tool — including but
   not limited to rocprofiler-sdk — can subscribe by talking to
   `lttng-sessiond`.

Cost:
1. New build dep on `liblttng-ust` for HIP/HSA runtimes.
2. New runtime dep on `lttng-sessiond` for the *consumer* (not the
   producer). The producer just needs `liblttng-ust`.
3. CTF as the on-the-wire format — rocprofiler-sdk would consume CTF (via
   `libbabeltrace2` or by reading the CTF stream directly) instead of
   getting in-process callbacks.

## Fallback / future-track: **`user_events`**

Why not primary today:
- **Distro reach is too narrow.** RHEL 9 and Ubuntu 22.04 LTS — both first-
  class ROCm targets — do not ship a kernel new enough. This is a hard
  blocker for production-grade adoption right now.
- The 6.4 ABI itself is a clean design and arguably a *better* fit for our
  problem than LTTng-UST in the long run: zero new userspace dependencies
  on the producer side, kernel-managed enable bit, and a writev()-based
  data path that's trivial to get right.

When user_events becomes broadly available (RHEL 10, Ubuntu 24.04 LTS as
the floor — i.e., 2027–2028 in customer reality), it becomes a serious
candidate for a v2 design.

A pragmatic intermediate: design HIP/HSA's emit interface as an internal
abstraction with two backends — LTTng-UST (today) and user_events (future)
— picked at build time or runtime. The hot-path call site is the same
either way; the backend is a per-event function pointer set at init.

## What we are NOT recommending and why

- **OpenTelemetry**: too heavy for the hot path; no true zero-cost off
  state; protobuf/gRPC dependency tree is a distro-portability headache.
  Useful as an *export* format from rocprofiler-sdk, not as the HIP/HSA →
  rocprofiler-sdk channel.
- **uprobes / eBPF uprobes / USDT**: all violate hard constraint #2
  (text-segment modification at attach). USDT in particular is so close in
  spirit to what the user wants that this is worth raising as a
  re-negotiation: if "no binary mod" really means "no in-place patch of
  *real* code", USDT (which patches a deliberately-reserved NOP slot) might
  be acceptable, in which case it deserves a second look. As stated by the
  user, it's out.
- **io_uring channels**: late-attach discovery isn't a solved problem on
  shipping kernels. Reserve as a possible secondary channel.
- **Chrome/Perfetto file**: already used as an *output* format. Not a live
  delivery mechanism.
- **Kernel tracepoints**: only relevant for KFD/AMDGPU-side events; doesn't
  solve the userspace runtime → consumer problem.

---

# Open Questions / Things to Validate

1. **Real-world LTTng-UST overhead on the HIP hot path.** Upstream cites
   ~100 ns per event when ON, ~5 ns when OFF. The HIP dispatch hot path
   already has 1248 ns budget eaten by `pool::acquire` (per
   `AIPROFSDK-813-always-on-tracing.md`). Need to measure: does adding an
   LTTng-UST tracepoint per dispatch (kernel_object, grid, wg, signal,
   correlation_id) push us over the existing 2.95×–3.55× envelope?
   Hypothesis: no, because ~100 ns is small relative to 1248 ns.

2. **CTF-side correlation IDs.** rocprofiler-sdk's callback API gives
   typed correlation IDs (internal + external). LTTng's tracepoint fields
   handle this (it's just two u64 fields). Verify the consumer side
   (rocprofiler-sdk reading CTF) can join async-signal-handler-emitted
   completion records with API-boundary records efficiently.

3. **Multi-tool subscriber semantics.** Today, rocprofiler-sdk supports N
   tools subscribing to HIP/HSA events via its context API. With LTTng,
   each tool would either (a) be its own LTTng consumer with its own
   recording session (heavy), or (b) all flow through rocprofiler-sdk
   which subscribes once and re-fans-out. Option (b) preserves current
   semantics; (a) would be a behavior change. Pick one and commit.

4. **How does rocprofiler-sdk consume CTF live?** LTTng-live (TCP
   protocol; networked CTF stream) is the obvious answer. Validate that
   rocprofiler-sdk's consumer thread can keep up with high-rate dispatch
   workloads (1000+ kernels/s sustained). Babeltrace 2 can also read live
   CTF buffers via shared-memory bridges.

5. **Build/packaging implications.** Adding `liblttng-ust` as a build dep
   for HIP/HSA touches the build matrix. Verify availability on:
   - Ubuntu 22.04 / 24.04: in distro repo (`liblttng-ust-dev`).
   - RHEL 9 / 10: via EfficiOS Enterprise Packages or EPEL.
   - SLES 15: EPEL or EfficiOS.
   - The internal AMD container baselines.

6. **The "tracing of intercept queues" interaction.** Today HSA queue
   intercept is the cheapest at-fidelity option (`rtl_full` at 2.95×). If
   we move HIP/HSA to LTTng-UST emit, do we also rip out the queue
   intercept path or run both? Both have different fidelity properties
   for graphs.

7. **Re-negotiate the USDT constraint?** Confirm whether the hard
   constraint really means "no binary mod ever" or "no binary mod of code
   that wasn't a deliberate tracing slot". If the latter, USDT (and
   USDT-via-eBPF for the consumer side) re-enters the candidate set as a
   strong contender — minimal producer changes, zero runtime library dep,
   eBPF/perf/SystemTap/bpftrace as ready-made consumers.

8. **`user_events` migration path.** Lock in the design abstractions now
   so the future move to `user_events` is a backend swap rather than a
   redesign. The two transports have nearly identical *application-facing*
   contracts: typed event + per-event enable check + producer-side emit.

---

# Sources

- LTTng v2.13 documentation: https://lttng.org/docs/v2.13/
- Linux kernel `user_events`: https://docs.kernel.org/trace/user_events.html
- Linux kernel uprobetracer: https://docs.kernel.org/trace/uprobetracer.html
- `perf_event_open(2)` man page: https://man7.org/linux/man-pages/man2/perf_event_open.2.html
- BPF iterators: https://docs.kernel.org/bpf/bpf_iterators.html
- OpenTelemetry components: https://opentelemetry.io/docs/concepts/components/
- Brendan Gregg, "Hacking Linux USDT with Ftrace" (2015): https://www.brendangregg.com/blog/2015-07-03/hacking-linux-usdt-ftrace.html
- AMD search results (no documented LTTng/USDT/OTel/eBPF integrations in rocprofiler-sdk; see this file's Notes section for inline citations)
- Local context: `~/ai/task_info/AIPROFSDK-813-always-on-tracing.md`,
  `~/ai/task_info/sideband-tracing-approaches.md`
