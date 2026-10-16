# Small public-HSA workload with separate timing instrumentation

Date: 2026-10-16
Toga/agent: toga-6-shark52 / Codex
Repository revision: unmodified rocm-systems develop `6191d1527b29369b6558ff7cc246281704e22224`
Related PR/issue: Systems draft #13635; no issue created
Outcome: simpler reproducer verified locally; no network writes

`queue-lifecycle.cpp` is a small public-HSA program. It initializes the runtime, finds a GPU, queries that agent's queue size and type, creates one empty queue, destroys it, and shuts down. All HSA status values are checked. It has no timing code or internal assertions. The older combined source is retained in `../diagnostics/one-queue.cpp` and its exact historical version in the evidence archive; prefer this split version for a maintainer handoff.

## Why the workload is valid

The pinned public API documentation supplies the contract:

- [Queue creation](https://github.com/ROCm/rocm-systems/blob/6191d1527b29369b6558ff7cc246281704e22224/projects/rocr-runtime/runtime/hsa-runtime/inc/hsa.h#L2388) initializes packets to INVALID. This program submits none. It uses the agent's advertised minimum queue size and queue type, allows a null callback as documented, and passes `UINT32_MAX` for unspecified private/group segment hints.
- [Queue destruction](https://github.com/ROCm/rocm-systems/blob/6191d1527b29369b6558ff7cc246281704e22224/projects/rocr-runtime/runtime/hsa-runtime/inc/hsa.h#L2534) requires the application to finish pending operations if their results are needed. This empty queue has no submitted operations. The program never accesses it after destruction.
- Initialization and shutdown are balanced. The application makes every HSA call from its main thread, so there is no concurrent application access to the queue.

The intended result is successful API calls and safe cleanup of the runtime's internal resources. The concern is that a runtime callback still tries to notify a queue event after that event has been destroyed.

## What the helper does

`delay-set-event.cpp` is a separate Linux `LD_PRELOAD` helper. It checks the named `AMDKFD_IOC_SET_EVENT` request from `<linux/kfd_ioctl.h>`. For background-thread requests it sleeps for 100 ms by default, then forwards the original fd, request and argument to libc's real ioctl. It returns the real result and preserves errno. It also logs completed DESTROY_EVENT requests and failed background SET_EVENT requests, including their event IDs. It has no event map, overlap counter, destructor, or application failure assertion.

The delay permits the runtime's other thread to progress while a callback is paused before notifying an event, modeling an extended thread scheduling pause. It deliberately changes timing. It does not forge a syscall error, modify ioctl arguments, or destroy events itself. This is a workload-specific pointer-argument ioctl wrapper, not a general interposer for arbitrary applications. Run without the helper as the plain-workload control, and with `KFD_SET_EVENT_DELAY_US=0` as the forwarding/logging control.

## Portable build and run

Use the unmodified runtime build and headers from the pinned revision. For a standard installed runtime the include/library directories can instead point to its installation. The helper additionally uses standard Linux development headers and libdl.

```sh
ROCR_SOURCE=/path/to/rocm-systems/projects/rocr-runtime
ROCR_BUILD=/path/to/rocr-build
mkdir -p repro-include
ln -s "$ROCR_SOURCE/runtime/hsa-runtime/inc" repro-include/hsa

amdclang++ -std=c++17 -O2 -Wall -Wextra -Werror \
  -Irepro-include queue-lifecycle.cpp \
  -L"$ROCR_BUILD/rocr/lib" -lhsa-runtime64 -o queue-lifecycle
amdclang++ -std=c++17 -O2 -Wall -Wextra -Werror \
  -shared -fPIC delay-set-event.cpp -ldl -o delay-set-event.so

# Add matching runtime dependency directories here if your build needs them.
export LD_LIBRARY_PATH="$ROCR_BUILD/rocr/lib:$LD_LIBRARY_PATH"
env -u LD_PRELOAD ROCR_VISIBLE_DEVICES=0 HSA_ENABLE_INTERRUPT=1 ./queue-lifecycle
env LD_PRELOAD="$PWD/delay-set-event.so" KFD_SET_EVENT_DELAY_US=0 \
  ROCR_VISIBLE_DEVICES=0 HSA_ENABLE_INTERRUPT=1 ./queue-lifecycle
env LD_PRELOAD="$PWD/delay-set-event.so" KFD_SET_EVENT_DELAY_US=100000 \
  ROCR_VISIBLE_DEVICES=0 HSA_ENABLE_INTERRUPT=1 ./queue-lifecycle

# Independently observe the actual kernel results. Only the child loads the helper.
env -u LD_PRELOAD ROCR_VISIBLE_DEVICES=0 HSA_ENABLE_INTERRUPT=1 \
  strace -f -tt -yy -e trace=ioctl -o native-strace.log \
  -E "LD_PRELOAD=$PWD/delay-set-event.so" -E KFD_SET_EVENT_DELAY_US=100000 \
  ./queue-lifecycle
rg 'AMDKFD_IOC_DESTROY_EVENT|EINVAL' native-strace.log
```

The program returns 77 if no GPU is available and 1 if an HSA call fails. Otherwise it returns 0, including the observed delayed-write case. An internal SET_EVENT error is the symptom, not a program exit failure. A run with no such error does not establish that the race is absent; the helper widens a scheduling interval rather than enforcing an exact interleaving.

## Verified results

This split version was compiled with `/opt/rocm/bin/amdclang++` (AMD Clang 20, ROCm 7.1) against the pinned develop headers and the already-built unmodified Release runtime. Tests used physical Shark52 gfx1201 hardware, `ROCR_VISIBLE_DEVICES=0`, and `HSA_ENABLE_INTERRUPT=1`. No rocJITsu, HIP, GEMM library, or GPU kernel dispatch is involved. The minimal `LD_LIBRARY_PATH` selects only the locally built runtime directory; that runtime retains its own recorded dependency search paths.

| Mode | Runs | Internal SET_EVENT error | HSA calls / process exit |
| --- | --- | --- | --- |
| Plain workload, no helper | 3 | No error reported | SUCCESS / 0 |
| Helper, delay 0 | 3 | None | SUCCESS / 0 |
| Helper, delay 100 ms | 3 | EINVAL in each run | SUCCESS / 0 |

Representative delayed output, with unrelated event destruction lines omitted:

```text
Real KFD DESTROY_EVENT(event_id=3) returned 0
Real KFD SET_EVENT(event_id=3) returned -1, errno=22
All HSA calls returned SUCCESS
```

The independent trace records successful `AMDKFD_IOC_DESTROY_EVENT` on the main thread followed by `AMDKFD_IOC_SET_EVENT = -1 EINVAL` on the callback thread. This strace version prints request arguments as pointers; the helper's logs identify the same event ID. Additional traced plain and zero-delay controls show no failed SET_EVENT syscall. The earlier diagnostic version supplied callback/destruction stacks and overlap tracking; see `../evidence/investigation-text.tar.gz` and `../evidence/current-develop.json` for that separate evidence. The failing callback was `DynamicQueueEventsHandler<false>`; destruction happened at shutdown while releasing the final internal queue.

The old four-queue version also reproduced on physical gfx942 with installed ROCm 7.2.1. This simplified version has only been tested on gfx1201 with the pinned develop build. No native failure without injected timing, native crash, numerical corruption, or same-current-head proposed-fix comparison is established.

Exact local commands, source/runtime hashes and every run's output are in the evidence archive and `../evidence/simple-results.json`. The traced run is recorded in the archived `event-diagnosis/develop/simple/native-strace.log` and `../evidence/simple-delayed-output.log`.
