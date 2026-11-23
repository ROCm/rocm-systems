# ROCr GPU core dumps: capture implementation and qualification

The five output-producing cases are now fixed. All seven corrected ROCr
core-dump cases pass against the Rust runtime on gfx1201 with the qualified
VM-clear driver correction. The historical investigation below explains the
original failures and the harness fixes; it is no longer the current status.

## Implemented fatal capture

The shared fatal path in `libhsa` captures real KFD state through `rocddi`.
It permanently inactivates retained compute queues, refreshes CWSR headers
through `GET_QUEUE_WAVE_STATE`, and writes ELF64 `ET_CORE` / `EM_AMDGPU`
with the version-1 `NT_AMDGPU_CORE_STATE` note. Runtime registration, KFD
version, GPU identity/apertures, queue IDs and submitted ring extents come
from their native owners. An allocation's page cover is kept separate from
the ring size used by the command processor and debugger.

The lifecycle registry retains queue, scratch, allocation, VMEM, committed
SVM and executable backing through emission. GPU virtual addresses label
`PT_LOAD` segments; `/proc/self/mem` reads their retained CPU aliases in
bounded chunks without forming Rust references to concurrently written
memory. CWSR's anonymous SVM storage is included. The reader opens a fresh
process-local descriptor for each capture. Loader rendezvous/maps, code-object destinations, and retained entry
trampolines are included. The original source ELF is a separate region at its
retained host address, matching `memory://` loader URIs and preserving metadata
that is absent from the relocated image. Staging/source bytes never substitute
for current device memory. Managed SVM commitment does not establish CPU
residency, so managed ranges have no reader: capture records their extents
without risking a fault that migrates pages back through a failed GPU. A region with no host reader is represented with
`p_filesz = 0` and its real `p_memsz`, not fabricated contents. Dumps are
therefore diagnostic snapshots, not an atomic snapshot of concurrent host or independent copy-engine
writes, or a claim to include private VRAM contents.

Policy is sampled at HSA initialization. The implementation handles the
existing disable flag, `%p/%i/%h/%e/%t/%%` substitutions, exclusive mode-0600
files, kernel-pattern GPU fallback and explicit pipe commands. File output
preflights the full ELF against the smaller sampled soft `RLIMIT_CORE` and
`RLIMIT_FSIZE` limits. An oversized capture or read/write failure removes the
new output and preserves fatal termination. Pipe output has a bounded
write/exit deadline and requires Linux pidfd process-group signaling. The
child sends its own pidfd before exec and waits for acknowledgement from a
concurrent receiver. The receiver retains cleanup ownership before approving
exec, so cleanup remains tied to the original handler group even if an
application child reaper removes its leader. A failed handoff never permits
handler execution. The bootstrap self-exits on rejection or its independent
monotonic timeout; without a received pidfd, it may remain a zombie until
application reaping or fatal parent exit. Pipe spawn runs outside HSA locks
so application `pthread_atfork` handlers may query HSA. Unsupported kernel
functionality or a missing descriptor handoff fails capture without falling
back to a reused numeric PID. Neither fatal
entry point writes to application stderr before capture or abort. The initial
default AQL diagnostic was removed during consolidation review because a full
stderr pipe could block the error worker indefinitely. A bounded subprocess
regression now exercises the default callback with ordinary and full stderr.

The default AQL callback and unhandled registered system-event paths share
this capture policy. Application callbacks retain priority. Automatic
process observation without application registration is a separate remaining
coverage gap; these seven cases reach the AQL fatal path.

## Capture verification

The three trap and four negative-queue callback regressions also pass after
capture integration. All seven dump cases pass serially in approximately 1.6–2.2 seconds, with only the
intended address-zero faults, no MES timeout/reset, and no remaining GPU
handles. The retained ContentIntegrity ELF is 39,175,232 bytes. Independent
parsing finds KFD 1.23, one registered GPU and queue, a submitted dispatch
(read index 0, write index 1, packet header `0x1402`), and a refreshed CWSR
with 52 control-stack bytes and 13,312 saved-wave bytes. These are actual
native records and memory, not placeholder ELF data.

Host tests cover CPU/GPU alias translation, unavailable regions, native note
layout, ring-vs-page extents, SVM/native CWSR storage, stop/query failures,
invalid wave-state sizes, effective registration, short output, quoting,
broken/stalled pipes, process-group cleanup after leader reaping, missing
pidfd handoff and fatal termination while application stderr is locked.
The complete native/libhsa/loader host run passes 658 tests with seven
existing ignored cases. Owner and curator review corrected ring extent,
post-loss CPU backing, VMEM enumeration, child identity and fork lock scope
before commit.

Artifacts under `target/rocr-test-fixes/` include `runs/core-writer-final-*`,
`kernel/core-writer-final-health.json`, the retained `*-integrity.gpu` core,
`core-all-host-tests.log`, `core-final-clippy.log`, and the final CTest log.

## PR review: retained executable regions and managed-memory policy

The revised writer retains three distinct loader-owned regions: current
executable backing, optional gfx1250 entry-trampoline backing, and the complete
source ELF at the host address advertised by a memory-backed loader URI.
Managed SVM ranges retain their virtual extent with `p_filesz = 0`, because
reading a committed page could migrate it from VRAM through a failed GPU.
That potential stall is established by the source path; it was not reproduced
by deliberately wedging this machine's GPU.

Fresh Release libhsa/rocddi/rocloader host tests pass 662 cases, with eight
ignored cases. New CPU tests distinguish relocated bytes from the original
ELF (including nonloaded metadata), retain the source across ownership moves,
keep a source readable when the device image has no CPU alias, and emit
managed extents without reading their unmapped addresses. The separately run
native trampoline-enumeration test passes on gfx1201 using real System
allocations. It submits no gfx1250 instructions; existing CPU trampoline-plan
tests cover the descriptor rewrite, and no gfx1250 hardware run is claimed.
Release CTest passes all 12 cases after rerunning the two sandbox-skipped fork
cases with native access. Warnings-denied workspace Clippy and rustfmt pass.

All seven corrected ROCr core-dump cases pass again, in 1.617–2.168 seconds,
on the unchanged loaded module `DE114CD2BB59945654162BC`. Per-case audits show
only intended address-zero faults, no reset/timeout, and no remaining GPU
handles. This run uses no preload. A separate bounded ContentIntegrity probe
redirects the public file-reader creation call to the public memory-reader
API while retaining its source mapping for the process lifetime. It also
passes. Following the saved `r_debug` and `link_map` in that real GPU core
resolves the `memory://` URI to all 4,392 source bytes, including all 15 ELF
section headers, in a separate `PT_LOAD`. The source SHA-256 matches the input
code object: `1514c517e44bcff69b6278e9720e5e93ad884b716d2db3d7fd1f4ec250174c62`.

Fresh run manifests and health checks are in
`target/rocr-test-fixes/runs/pr16293-capture/` and `pr16293-memory-reader/`.
Build, host/native tests, the retained memory-reader core, and its independent
inspection are recorded in `target/rocddi-pr16293-review/`. These targeted
runs validate the capture changes; they do not requalify the revised batching
kernel module or represent a new 133-case sweep.

## PR review: preflight file-size limits

The file-output path previously checked only `RLIMIT_CORE`. A smaller finite
`RLIMIT_FSIZE` can shorten a write and then deliver default-action `SIGXFSZ`
on the next `write_all` attempt, terminating before cleanup and the intended
SIGABRT. An isolated native reproduction exited on SIGXFSZ and retained a
256-byte partial file after trying to write 4,096 bytes.

File output now also samples `RLIMIT_FSIZE` and rejects a planned ELF larger
than the smaller limit before writing any bytes. The diagnostic identifies
that effective file limit as the minimum of the two limits. Existing exclusive
file creation and removal on failure remain intact. `RLIMIT_CORE=0`
suppression and explicit-pipe policy are unchanged. This preflight checks the
sampled values; it does not protect against concurrent resource-limit changes
during emission or alter process-wide signal dispositions.

A GPU-free subprocess regression exercises the production file-output helper,
ELF layout, preflight, and writer with real host-backed data. Each child uses
native `RLIMIT_FSIZE`, default unblocked SIGXFSZ, and a supplied CORE allowance.
The five cases cover FSIZE below the full ELF size, FSIZE zero, CORE below the
size, acceptance exactly at both limits, and successful pipe emission despite
FSIZE zero. Rejection cases leave no output file and reach the actual
`fatal_abort()` SIGABRT. The boundary case verifies the complete ELF and source
bytes. Resource limits and signal setup affect only the isolated children.

The revised Release host run passes 663 tests with eight ignored cases.
Warnings-denied workspace Clippy and rustfmt pass. All 12 CTest cases pass
across the sandbox run and native reruns of its two fork-capability skips.
All seven corrected ROCr capture cases also pass on gfx1201 with the unchanged
`DE114CD2BB59945654162BC` module and no preload. Their audits report only
intended address-zero faults, no reset or timeout, and no remaining GPU
handles. No new kernel module was loaded.

The new host/build/review evidence is under
`target/rocddi-pr16293-review/rlimit/`; GPU manifests, logs, and per-case health
checks are under `target/rocr-test-fixes/runs/pr16293-rlimit-capture/`.

## PR review: acknowledge pidfd ownership before handler execution

The pipe bootstrap previously handed its pidfd to the parent after allowing
the handler to execute. Descriptor exhaustion during `SCM_RIGHTS` receipt
can consume the message while discarding its descriptor with `MSG_CTRUNC`.
That left no safe identity for cleanup of the running handler or descendants.

The child now waits for explicit approval before exec. A scoped receiver
thread receives the pidfd, establishes an RAII cleanup owner, and only then
sends approval. Receipt must run concurrently because `Command::spawn` waits
for exec to finish. The owner follows the handler through acknowledgement,
spawn completion, stdin extraction, emission and exit checking; every failure
after receipt uses the original pidfd for group cleanup. Receipt failure
sends a best-effort rejection. Each side has a separate five-second monotonic
bootstrap deadline, with interrupted polls consuming the same time budget.

Bootstrap failure uses child-safe `_exit(127)` without entering arbitrary
handler code. It does not return a pre-exec error, which can make the standard
library assert while reaping a child already removed by an application reaper.
Without a received pidfd, the runtime does not use a numeric PID to signal or
reap; a failed bootstrap can remain a zombie until application reaping or
fatal parent exit. The guarantee is that failed handoff cannot execute a
handler or create its descendants, not that missing identity permits immediate
safe reaping. Application atfork callbacks still run outside HSA locks.

An isolated GPU-free regression waits for the real handoff message and lowers
the parent's soft `RLIMIT_NOFILE` to zero only around actual descriptor receipt.
It verifies `MSG_CTRUNC` rejection, bounded completion and absence of a handler
execution marker, both with ordinary SIGCHLD handling and with automatic child
reaping. Another case forces acknowledgement-send failure after real receipt
and checks cleanup before execution. Repeated signal interruption also leaves
the bootstrap deadline bounded. Successful consumption, stalled pipes,
descendant cleanup and already-reaped group leaders retain their coverage.

The revised Release libhsa/rocddi/rocloader host suite passes 664 tests with
eight existing ignored cases. All 12 CTest cases pass with native access,
including both fork cases. Warnings-denied workspace Clippy and rustfmt pass.
All seven corrected ROCr capture cases pass serially on gfx1201 in
1.620–2.172 seconds with unchanged module `DE114CD2BB59945654162BC` and no
preload. Per-case audits show only intended address-zero faults, no reset or
timeout and no remaining GPU handles. No kernel module was loaded for this
revision. These targeted capture runs do not replace the earlier full sweep
or qualify the pending revised kernel module.

Host, build and review evidence is under
`target/rocddi-pr16293-review/handoff/`. The GPU run's immutable library,
manifest, per-case logs and health checks are under
`target/rocr-test-fixes/runs/pr16293-handoff-capture/`.

## Historical investigation

`rocrtstFunc.GpuCoreDump_DefaultPattern` failed with libhsa revision
`b71be8828809c2dcd56f2e5f6e86d0007ace61cf` on gfx1201: the child dispatched a
null-pointer fault kernel, timed out, and produced no matching GPU dump.
Same-source C++ ROCr also failed on the current machine after a GPU Hang
notification. The [trap recovery finding](rocr-gfx1201-trap-recovery.md)
describes that shared-state limitation; it does not remove the independent
libhsa implementation gaps below.

## Missing libhsa behavior

The core-dump child supplies neither a queue error callback nor a system-event
handler. Originally `queue/event.rs::pending_callback` dropped notifications
for a null callback. Commit `2941a70fa28` supplies a default AQL handler that
prints the queue/status and aborts. Explicit application callbacks retain
priority, and only exact `HSA_ENABLE_QUEUE_FAULT_MESSAGE=0`, sampled at startup,
disables that default action. System-observer deferral and duplicate suppression
remain in front of callback selection.

This fixes the callback-free AQL hang, but `events.rs::SystemEvents` still
starts its process observer only on application registration. Reference ROCr
starts a process VM-fault handler and substitutes `Queue::DefaultErrorHandler`
for a missing queue callback. On applicable gfx12 systems without kernel dump
support, its VM-fault path attempts `dump_gpu_core` before aborting. The earlier
reference hardware-hang path aborted directly; that historical no-dump result
is not used to excuse the Rust implementation gap.

The Rust runtime has no GPU dump writer or `HSA_COREDUMP_PATTERN` handling.
Implementing ROCr-equivalent dump behavior still requires actual GPU state
capture, ELF output, file/pattern/error handling, and a default process observer
with generation-safe teardown. The AQL fallback only establishes default fatal
handling of queue-delivered faults; it does not implement that wider feature.

## Native capture work still required

The reference path uses cached runtime/device state and the existing KFD
`GET_QUEUE_WAVE_STATE` operation; it does not require debugger attachment.
This ioctl is already present in the vendored UAPI and the installed gfx12
kernel, but rocddi has no wrapper or queue-capture interface for it.

A valid implementation needs lifetime-bound native snapshots through rocddi:
real queue/GPU IDs, effective runtime registration, device/aperture metadata,
refreshed CWSR state, and retained diagnostic memory views. The existing
one-way queue inactivation retains backing and can support a fatal capture;
it is not a reversible debugger suspension API. Effective registration must
come from successful native enablement, rather than a requested setting.

The reference writer's render-node proc-maps filter cannot simply be copied.
rocddi normally uses anonymous SVM for CWSR, so that filter would omit its saved
wave state. USERPTR allocations can also have different host and GPU addresses:
ELF load addresses must use GPU VA while byte reads use the retained host alias.
SDMA copies are not an established post-fault diagnostic reader.

The dump coordinator must retain queue, allocation, scratch, and executable
owners against teardown; inactivate recorded compute queues; refresh and
validate wave-state sizes against retained CWSR backing; then emit real ELF
notes and load data. Failed capture must be reported as such. KFD calls and
backing access belong in rocddi; HSA policy, enumeration, naming, and ELF
formatting belong in libhsa. No additional kernel UAPI requirement has been
identified for this host, but these runtime mechanisms are not implemented by
the default-callback fix.

## Non-qualifying test passes

The original same-source `gpu_coredump.cc::RunFaultingKernelInChild` had three
outcome problems relevant to this report:

- Child setup failures exit with status 1, but the parent treats any nonzero
  termination as the expected fault. Disable/invalid-path cases can therefore
  pass solely because no dump was generated after failed setup. This occurred
  before the code-object profile fix.
- The local watchdog kills the child and returns its PID as if an expected
  fault occurred, again allowing absence-of-file checks to pass.
- A child that returns from the signal wait prints "Kernel completed without
  fault" and returns sentinel -2 through the parent. The wait's result is
  ignored, so that message does not prove fault-free completion. Callers
  return without checking dump behavior, and gtest reports success. Six
  post-profile cases took this path on the observed machine.

These are test outcome gaps, not proof of supported dump generation. The
local report runner marks printed setup errors/watchdog kills as false passes
and no-fault early returns as `not_exercised`. The corrected harness now records failures for these paths. It requires a
one-byte readiness marker immediately before packet publication and a reaped
child terminated by SIGABRT before evaluating file effects. The marker rejects
setup-time aborts; it does not independently prove GPU causality. Reference
ROCr's unhandled VM-fault path calls `std::abort()` even when the GPU dump
writer is disabled. This is a source-backed ROCr test expectation, rather
than an HSA-wide requirement for all implementations.

The parent uses a monotonic ten-second polling deadline and handles EINTR.
Timeout and wait failures remain failures after child cleanup; ECHILD cannot
trigger a kill of a PID the parent no longer owns. The readiness pipe is
nonblocking and close-on-exec. All seven callers clean their environment on
helper failure. The child retries signal waits whose returned value does not
satisfy the condition, as the HSA contract permits spurious returns. The final
blocking reap still depends on the kernel retiring a killed child; the outer
runner provides an additional process-group watchdog.

## Harness verification and corrected hardware comparison

The GNU C++17 rebuild passes. Owner, C++/POSIX, and curator reviews found no
remaining introduced issue. Host-only checks exercise the actual linked
harness, without calling real HSA initialization or touching GPU devices:

- All seven cases now fail on injected setup errors and clean up their
  environment. Synthetic exits, unexpected signals, setup-time SIGABRT,
  invalid readiness, timeout, fork/pipe failure, and wait failure are rejected.
- Readiness plus SIGABRT passes DisableFlag and InvalidPath, including an
  injected EINTR. DefaultPattern still fails its missing-file assertion; no
  fake dump file is used. Children are reaped, and ECHILD triggers no kill.
- Five before-fix controls demonstrate false passes for setup failure,
  normal completion, early SIGABRT, and timeout.
- Host-backed HSA objects exercise the actual child: readiness precedes the
  valid packet header, which precedes the doorbell. A spurious return of 1 is
  retried and the subsequent abort accepted; actual completion at 0 fails.
  The old binary instead exits after the first spurious return and passes.

There are 27 parent/caller injection checks and four host-backed child checks;
all meet their expected outcomes. Their logs/XML, binary hashes, build
commands, and injection sources are under `target/rocr-test-fixes/` in
`coredump-harness/`, `coredump-harness-build-commands.json`,
`coredump-outcome-injection.c`, and `coredump-fake-hsa.c`.

With the patched driver and corrected harness, the first bounded DisableFlag
comparison failed for Rust libhsa after the ten-second child watchdog
(10.379 s total) and passed for reference ROCr (1.618 s). After the default AQL
handler fix, Rust also passes in 1.618 s, with its new queue/status diagnostic.
This isolates missing default fault termination from the trap-page-table race.
Evidence is in `runs/coredump-harness-{rust,reference}-disable/`,
`runs/default-queue-rust-disable/`, and their kernel snapshots.

All seven core-dump cases were then exercised with the corrected harness and
runtime on the patched driver, without a preload:

| Case suffix | Result | Seconds | Observed behavior |
|---|---|---:|---|
| `DisableFlag` | Pass | 1.618 | Expected abort, no output |
| `InvalidPath` | Pass | 1.619 | Expected abort, no output |
| `DefaultPattern` | Fail | 1.619 | Expected abort, missing dump file |
| `CustomPattern` | Fail | 1.618 | Expected abort, missing dump file |
| `PatternSubstitution` | Fail | 1.669 | Expected abort, missing dump file |
| `ContentIntegrity` | Fail | 1.620 | Expected abort, missing dump file |
| `PipePattern` | Fail | 2.122 | Expected abort, missing pipe output |

The two absence-of-output passes do not establish that a writer attempted the
requested configuration. The five failing cases now reach their output
assertions instead of false-pass early returns. None times out. Serial
per-case kernel and handle audits show only the intended address-zero faults,
with no MES timeout/reset and no remaining GPU users. The seven explicit
trap/negative-queue callback regressions also pass after the default-handler
change; their kernel delta contains malformed-queue errors and workqueue
warnings, with no recovery failure.

A subsequent reference ROCr DefaultPattern control passes in 3.171 s and
produces a 161,559,936-byte GPU core file verified by the same corrected
harness. Its kernel delta contains only the intended address-zero fault, with
no remaining GPU users. This establishes that dump generation is compatible
with this gfx1201 system on the patched driver; the Rust failures are a missing
runtime feature. Evidence is in `runs/patched-reference-coredump-default/` and
`kernel/after-reference-coredump-default-kernel.log`.

The latest Rust evidence is in `runs/default-queue-*`,
`kernel/default-queue-coredump-health.json`, and `default-queue-{host,ctest,clippy}.log`.
There are 35 queue unit-test passes; the full CTest run has ten passes and two
expected skips. Workspace Clippy with warnings denied and rustfmt pass. Owner,
Rust, and curator reviews found no remaining introduced issues in the default
handler fix.

Workspace evidence is under
`target/rocr-test-fixes/runs/after-cache-permissions/`,
`runs/after-modern-profile/`, and `runs/reference-traps/`. Intentional fault
comparisons were subsequently authorized after earlier runs caused GPU hangs
and MES recovery failures. These cases were excluded from the final
nonfaulting run; the bounded patched-driver comparison is recorded above.
