# rocprofv3-att-gdb

Trigger ATT from CPU breakpoints in an application or its descendants:

```sh
rocprofv3-att-gdb --start begin_iteration --stop end_iteration \
    --att-shader-engine-mask 0x3 --att-buffer-size 16777216 -d traces -- ./application
```

Use your usual rocprofv3 ATT parameters. No application changes, ROCTx library,
or separate multiprocess flag are required. The launcher and rocprofv3 must come
from a build containing the debugger control backend.

If A launches B and B launches GPU worker C, launch A with this command and name
the breakpoint in B. ROCgdb follows fork/exec and spawn descendants and resolves
pending symbols as their executables load. The first selected hit broadcasts to
all profiling processes registered in the session. Each process controls its own
ATT resources, including the existing lazy code-object allocation policy.

Add `--skip 100` to start on hit 101. The count spans threads, resolved locations,
and retained inferiors; it counts dispatches only when the chosen CPU location
executes once per dispatch. All owned start locations are removed before START.
The stop breakpoint is enabled only after the start and is restricted to the
triggering process. Start and stop can name the same location: stop then fires on
the next visit, including a visit by another thread in that process.

Use `--timeout 50ms` instead of, or together with, `--stop`. After all participants
acknowledge START and the debugger finishes continuing the triggering thread,
the coordinator computes one absolute monotonic deadline and sends it to each
worker. Setup and continuation do not consume the interval. Late delivery or
scheduling may extend it. A shared user breakpoint delays timer arming until
manual continuation; later user stops do not pause the timer. Explicit stop,
cancel, process exit, and failures can end capture earlier. This is a host interval,
not an exact GPU execution duration.

Without `--start`, select triggers interactively:

```text
$ rocprofv3-att-gdb -d traces -- ./application
(gdb) att arm --start begin_iteration --skip 100 --stop end_iteration
(gdb) run
(gdb) att status
```

You can also run first, press Ctrl+C, arm, and continue. Non-stop mode means
Ctrl+C stops the selected thread; other application and profiler threads remain
runnable. `att cancel` removes an armed trigger or stops an active capture.
Rearm after completion to capture another region. Unrelated user breakpoints
and subsequent user stops remain owned by the user.

`--batch` exits after the process tree and participants finish, with a nonzero
status for incomplete capture or application failure. A exiting does not discard
B/C. Without this flag, ROCgdb stays open. `--command 'GDB COMMAND'` can be repeated
to configure the debugger before execution. `--rocprofv3` and `--rocgdb` select
installations explicitly.

Participants must register before the selected hit; later processes join the
next capture. Use `att status` to list registered participants. Launchers that
replace the environment must preserve the profiling environment and internal
session variables. Fork+exec and `posix_spawn` are covered; raw fork without exec
after profiler/GPU-runtime initialization is not supported.

To add an independently running profiling process, print the session descriptor
with `att session`, then use it from another terminal:

```sh
rocprofv3 --att --att-debugger-session /tmp/rocprofv3-att-gdb-.../session.json \
    --attach PID -d traces [usual ATT parameters]
```

The normal rocprofv3 attach prerequisites apply. The participant must not already
be ptrace-owned by ROCgdb; sharing ownership of the same PID requires the planned
profiler/debugger handoff. Wait for its registration before triggering capture.
Finish the capture before detaching the profiler.

Output is grouped under `traces/att_gdb_<session>/process_<pid>_<generation>/`.
The session directory contains `capture_<id>.json`, recording trigger PID,
participant identities, acknowledgements, errors, deadline, and output locations.
`[att] ACTIVE` means all participant start calls returned successfully. It does
not promise that all visible GPUs have tracing resources. `[att] DONE` means the
participant stop calls completed. Raw ATT callbacks finish during stop; final
decoding/output completes at process exit or profiler detach.

An orderly participant exit flushes its active trace and acknowledges completion;
other participants continue until their stop boundary. An unexplained disconnect
marks the capture incomplete. If the trigger process exits while participants
still need its stop breakpoint, an already armed timeout is required to continue.

The maintained backend uses a dedicated SDK ATT context, independent of
application ROCTx Pause/Resume. Workers sleep on sockets and one-shot timers;
there is no added dispatch interception, periodic polling, heartbeat, or trace
scanning. Other application threads and GPU work can advance during trigger
handling. Coordinated CPU/GPU holds, CWSR, and GPU breakpoints remain future work.
Boundary latency still exists, and this implementation does not establish
cycle-identical execution to an unattached application.

Source breakpoints require suitable debug information. Optimization can inline
functions or split/reorder source lines. In particular, the source line after
`hipDeviceSynchronize()` does not reliably mean “after synchronization.” Inspect
the compiled continuation when selecting an instruction breakpoint. Automatic
kernel-dispatch selection and synchronization insertion are not implemented.

The normal SDK build installs the tool and frontend. The frontend and transport
tests can also be built separately against an SDK with the debugger backend:

```sh
cmake -S source/bin/rocprofv3-att-gdb -B build/att-gdb -DBUILD_TESTING=ON
cmake --build build/att-gdb
ctest --test-dir build/att-gdb --output-on-failure
```

Tests require pytest, a C++ compiler, and GDB/ROCgdb with Python. GPU tests also
need HIP and an ATT-capable GPU. Enable them with
`-DROCPROFV3_ATT_GDB_GPU_TESTS=ON`, set `ROCPROFV3_ATT_GDB_GPU_ARCH` as needed, and
point `ROCPROFV3_ATT_GDB_PROFILER` at the matching rocprofv3. The ROCTx adapter
built for CPU tests exercises the shared transport; it is not installed or used
by the launcher.

See the [user guide](../../docs/how-to/using-thread-trace.rst) and
[design plan](../../docs/conceptual/debugger-triggered-att-plan.md).
