# ROCr queue-event lifetime investigation

Date: 2026-10-16
Toga/agent: toga-6-shark52 / Codex
Verified unmodified develop revision: `6191d1527b29369b6558ff7cc246281704e22224`
Preserved candidate revision: `d68cd7890a2403d3527bb575309d09a5982c97f6`
Related PR: [ROCm/rocm-systems #13635](https://github.com/ROCm/rocm-systems/pull/13635)
Disposition: investigation archived on the candidate branch; PR closure requested, branch retained; no issue publication requested

This branch retains an experimental fix and the evidence for a queue-event lifetime problem. The investigation is complete for now at the user's request. The fix is not an accepted or merged ROCr implementation. The verified develop revision above was fetched and built during this investigation; this is a dated result, not a claim about all later develop revisions.

## Finding and original observation

The problem was first observed during a CPU-emulated gfx942 TensileLite race-detection campaign being prepared for nightly CI. Numerical and dispatch checks could finish successfully, followed by an intermittent rocJITsu `SET_EVENT_MISS` warning at shutdown. That original observation used CPU emulation, not physical gfx942 execution. The broader nightly work is preserved separately in Libraries draft [#13781](https://github.com/ROCm/rocm-libraries/pull/13781), related to the regular-GEMM workload in [#13687](https://github.com/ROCm/rocm-libraries/pull/13687) and sampled campaign in [#12734](https://github.com/ROCm/rocm-libraries/pull/12734).

The runtime symptom can also be exposed without rocJITsu, HIP, GEMM libraries, or GPU kernel dispatch. A native HSA program creates and destroys one empty application queue and then shuts down. When a separate helper pauses background KFD SET_EVENT calls for 100 ms, the real driver reports EINVAL for an event notification after that event was destroyed. All public HSA calls still return SUCCESS. The simple program exits 0; the error is an internal callback notification, not an application-level HSA failure.

The delay deliberately widens a possible scheduling interval before the real ioctl. It changes timing, forwards the actual arguments and return values, and does not inject an error or destroy events itself. These runs do not establish an ordinary native crash or the frequency of an uninstrumented native failure.

## Preferred reproducer and validity

Start with [repro/queue-lifecycle.cpp](repro/queue-lifecycle.cpp) and its separate [repro/delay-set-event.cpp](repro/delay-set-event.cpp). [Repro instructions](repro/README.md) explain portable build/run commands and the pinned public API contract. The workload queries the agent's queue size/type, supplies documented unspecified segment hints, creates an empty queue with a permitted null callback, destroys it, and balances HSA initialization/shutdown. It submits no packets, has no pending application work to wait for, performs all HSA operations on one application thread, and never accesses the queue after destruction.

The helper uses named KFD ioctl requests from Linux headers. It delays background SET_EVENT calls, forwards the real ioctl, preserves errno, logs completed DESTROY_EVENT requests and failed background SET_EVENT requests, and has no event map, overlap assertion, teardown destructor, or forced failing application exit. It assumes pointer-argument ioctls as used by this workload; it is not a general interposer for arbitrary applications.

Typical delayed output, omitting unrelated event destruction:

```text
Real KFD DESTROY_EVENT(event_id=3) returned 0
Real KFD SET_EVENT(event_id=3) returned -1, errno=22
All HSA calls returned SUCCESS
```

Event IDs can vary. The expected cleanup is that the runtime keeps an event valid until the callbacks using it finish, so no late SET_EVENT fails. The independent syscall trace confirms successful DESTROY_EVENT followed by actual SET_EVENT EINVAL on the callback thread. This strace version prints argument pointers; the helper's logs identify the event IDs. Plain and zero-delay traced controls have no failed SET_EVENT syscall.

## What was tested

| Evidence | Runtime / hardware | Result and limits |
| --- | --- | --- |
| Original workload | CPU-emulated gfx942 TensileLite | Intermittent shutdown SET_EVENT_MISS after numerical/dispatch checks; no original physical-hardware observation |
| Emulated 64-queue regression, 1 ms delay | Original artifact vs candidate `d68cd7890a2` | Original fails 3/3, candidate passes 3/3; [historical receipt](evidence/emulated-candidate-validation.json) |
| Native four-queue diagnostic, 100 ms delay | Earlier artifact vs candidate, physical gfx1201 | Original SET_EVENT EINVAL/overlap 3/3; candidate passes 3/3 with delay exercised; [receipt](evidence/native-earlier-comparison.json) |
| Native four-queue diagnostic, 100 ms delay | Installed ROCm 7.2.1, physical MI300X gfx942 | SET_EVENT EINVAL/overlap 3/3; candidate and current develop were not built on this worker |
| Native one-queue diagnostic | Unmodified develop `6191d1527b29`, physical gfx1201 | 100 ms failure 5/5, zero-delay control passes 5/5; matching-header repeat 3/3 each; [receipt](evidence/current-develop.json) |
| Preferred split simple reproducer | Same unmodified develop, physical gfx1201 | Plain 3/3 and zero-delay 3/3 exit 0 without event error; 100 ms 3/3 produce internal EINVAL and exit 0; [receipt](evidence/simple-results.json) |

Native current-develop environment: Ubuntu 24.04.5, Linux `7.0.0-31-generic`, GPU gfx1201 PCI `1002:7551`, `ROCR_VISIBLE_DEVICES=0`, `HSA_ENABLE_INTERRUPT=1`. Runtime and matching thunk were built from the clean pinned checkout; Release host build used upstream Clang 23.1.0, with retained ROCm Clang 24/llvm-objcopy import shims for runtime support kernels. Reproducers used AMD Clang 20 from installed ROCm 7.1 and headers from the pinned checkout. Runtime file SHA256: `ac53cf8603637cde6a233fda85ab9518f0ee9398ac42e25b6d56594c3ed80c39`. Commands, mappings, compiler selection and source hashes are retained in the evidence archive. The source did not need the experimental fix to reproduce the issue.

The candidate additionally passed the earlier 100-solution/400-case emulated TensileLite sample and 20-case installed bench sample with numerical/dispatch validation. These are historical candidate results against the earlier base. No same-current-head candidate-fix comparison was performed. The candidate's existing 1 ms/64-queue checked-in regression was qualified under emulation and did not reliably reproduce the native bug in our experiments; the 100 ms standalone diagnostic is the native evidence.

## Source explanation

The precise current-develop native failing callback was `AqlQueue::DynamicQueueEventsHandler<false>`, not `ExceptionHandler`. The late notification stack is AsyncEventsLoop → DynamicQueueEventsHandler → hsa_signal_store_screlease → thunk/ioctl. Event destruction occurs during hsa_shut_down → Runtime::Unload → GpuAgent::ReleaseResources → AqlQueue::~AqlQueue, while releasing the final internal queue after the application queue has already been destroyed.

The pinned source provides the following explanation:

- [DynamicQueueEventsHandler](https://github.com/ROCm/rocm-systems/blob/6191d1527b29369b6558ff7cc246281704e22224/projects/rocr-runtime/runtime/hsa-runtime/core/runtime/amd_aql_queue.cpp#L1383) publishes ERROR_HANDLER_DONE before its signal store completes.
- [InterruptSignal::StoreRelease](https://github.com/ROCm/rocm-systems/blob/6191d1527b29369b6558ff7cc246281704e22224/projects/rocr-runtime/runtime/hsa-runtime/core/runtime/interrupt_signal.cpp#L133) writes the signal value before notifying its event.
- [AqlQueue destruction](https://github.com/ROCm/rocm-systems/blob/6191d1527b29369b6558ff7cc246281704e22224/projects/rocr-runtime/runtime/hsa-runtime/core/runtime/amd_aql_queue.cpp#L397) can observe callback completion and release the shared event when the final queue disappears.
- [AsyncEventsLoop](https://github.com/ROCm/rocm-systems/blob/6191d1527b29369b6558ff7cc246281704e22224/projects/rocr-runtime/runtime/hsa-runtime/core/runtime/runtime.cpp#L2075) retains the signal until callback return, but the interrupt signal borrows the queue event through a raw pointer. Retaining the signal therefore does not retain the borrowed event.

The exception callback has analogous completion-before-final-notification ordering and was part of the earlier emulated analysis. It must not be substituted for the directly traced dynamic callback in the native finding. Raw event-pointer use after free before entering ioctl is plausible from source reasoning; no separate native memory fault was demonstrated.

## Preserved fix and evidence

Commit `d68cd7890a2403d3527bb575309d09a5982c97f6` retains shared queue-event ownership in interrupt signals. Queue destruction drops the queue holder; retained callback signals keep the event alive until their final use. A process-lifetime holder follows the existing queue lock/count lifetime to support late HIP static teardown; the event itself still releases after its final owner. This implementation remains an experimental proposal for maintainers to assess, with whole-suite ROCr and same-current-head validation outstanding. Existing whole-file formatting debt was documented rather than reformatted as part of the candidate.

The readable `diagnostics/` sources retain the earlier combined one-queue, four-queue and stack-tracing variants. Their exit 1 is an instrumentation assertion about overlapping destroy/write operations, unlike the preferred simple program. Their exact historical source versions and hashes are archived, so formatting of readable copies does not invalidate the prior receipts.

[evidence/investigation-text.tar.gz](evidence/investigation-text.tar.gz) contains the detailed textual sources, GDB scripts, logs, commands, build configuration, process mappings, receipts, experimental variants, historical local issue draft and original nightly context. [archive-manifest.json](evidence/archive-manifest.json) records original and archived file hashes. Unrelated formatting backups, generated header copies and binaries are omitted; headers come from the pinned source, and build/run instructions recreate the executables. Host-specific paths in the preserved text are replaced by explicit placeholders. Historical entries are dated snapshots and may describe superseded assumptions or publication state; this README states the final finding.

Unpack outside the source tree:

```sh
mkdir -p /path/to/investigation-output
tar -xzf evidence/investigation-text.tar.gz -C /path/to/investigation-output
```

No GitHub issue or gist has been created. The PR is being closed while retaining its branch and all candidate commits. No further runtime implementation, CI monitoring or issue publication is part of this completed investigation. A future investigation should refresh develop, repeat the preferred native controls, and validate a proposed fix on that same revision before deciding how to report or resolve the bug.
