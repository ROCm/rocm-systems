# rocprofv3-att-gdb prototype

Trigger ATT from CPU breakpoints without adding ROCTx calls to an application:

```sh
rocprofv3-att-gdb --start begin_iteration --stop end_iteration \
    --att-shader-engine-mask 0x3 --att-buffer-size 16777216 -d traces -- ./application
```

Use your usual rocprofv3 ATT settings. `--timeout 50ms` can replace or supplement
the end breakpoint. Add `--skip 100` to ignore the first 100 start-breakpoint hits
and capture on hit 101:

```sh
rocprofv3-att-gdb --start begin_iteration --skip 100 --stop end_iteration \
    -d traces -- ./application
```

This counts CPU breakpoint hits across threads and resolved locations. It selects
dispatch 101 only if the location executes once per desired dispatch. The end
breakpoint stays disabled during warmup, and the start breakpoint is removed
before capture starts. Skipped hits can still incur debugger overhead during
warmup, but do not call the profiler.

The start and stop may use the same location to capture a loop round trip:

```sh
rocprofv3-att-gdb --start application.cpp:120 --stop application.cpp:120 \
    --skip 100 -d traces -- ./application
```

Capture starts on hit 101 and stops on the next hit. GDB resumes past the
current breakpoint instruction before the stop can fire at that location.
Both triggers match across threads, so another thread can supply the next hit.

Source-line order does not guarantee execution order. In particular, the line
after `hipDeviceSynchronize()` is not a reliable stop-after-synchronization
location: optimization can split, move, or merge the associated instructions.
For a specific compiled call, inspect `disassemble /s FUNCTION` and prepare an
instruction breakpoint at its normal-return continuation before capture starts.
Verify that the selected execution path reaches it after the intended call.
A source-line breakpoint is suitable only if its resolved locations have been
verified in that particular binary; recompilation requires checking again.

GDB has no general source-line-exit breakpoint. Its Python `FinishBreakpoint`
targets the return of an existing stack frame; stopping at the synchronize
function's entry to create one would add a stop during capture. The prototype
does not automatically resolve a stop-after-call location.

With no `--start`, the launcher opens ROCgdb: use
`att arm --start LOCATION --skip 100 --stop LOCATION`, then `run`. `att status`
reports the capture state and remaining skip count; `att cancel` cancels it.
Add `--batch` for unattended scripts: the debugger exits when the application
finishes, returning nonzero if capture does not finish or the application fails.
Without `--batch`, ROCgdb stays open for inspection. This option does not select
dispatch batches or end the application when capture stops. Source locations
need debug information; place the start after GPU/kernel initialization.

You can also let the application run before choosing the capture locations:

```text
$ rocprofv3-att-gdb -d traces -- ./application
(gdb) run
... press Ctrl+C when ready to choose the triggers ...
(gdb) att arm --start begin_iteration --stop end_iteration
(gdb) continue
```

In the prototype's non-stop mode, Ctrl+C stops the selected thread and
`continue` resumes it; other threads, including the profiler worker, keep
running. Arming creates the breakpoints without starting capture immediately.
If supplied here, `--skip N` counts future hits after arming, excluding any
iterations that already ran. This workflow uses an application launched with
the helper; injecting the profiler into an independently started process needs
the planned attach support.

The timeout interval starts after both `roctxProfilerResume(0)` and the
debugger's continuation of the triggering thread have completed. The debugger
then tells the helper to arm a one-shot monotonic timer for the full requested
duration. Setup, continuation, and delivery of this notification can extend the
trace; they do not consume the interval. If a shared user breakpoint keeps the
triggering thread stopped, the timer waits for manual continuation. Later user
stops do not pause or restart the timer. A supplied end breakpoint or explicit
cancellation can still stop earlier.

A one-shot debugger watchdog reports an error if stopping has not completed
within 10 seconds after the capture deadline. The target worker blocks on its
socket and timer, with no periodic checks during capture. A very short interval
can still finish before a kernel launches; it does not measure GPU execution time.
An early stop in the loader (for example, `starti`) can delay helper setup;
connection attempts resume on continuation or at the selected start hit.

For a particular kernel, choose its CPU launch line or a host wrapper. Compiled
HIP code may also expose a CPU launch stub, usable as, for example,
`--start '__device_stub__my_kernel(float*)'`. This requires an actual call to
that stub: optimization can inline the call even when the symbol still exists.
The stub path is tested with inlining disabled; it is not a general kernel-name
selector for optimized, JIT, or graph launches. GPU-entry breakpoints require
future GPU support and can hit per wave rather than per dispatch.
The kernel-specific trigger chooses the start boundary; other GPU work in the
selected region can also be traced.

The normal SDK build installs the launcher, Python extension, and helper. To
build only this prototype against an existing ROCm installation:

```sh
cmake -S source/bin/rocprofv3-att-gdb -B build/att-gdb
cmake --build build/att-gdb
build/att-gdb/bin/rocprofv3-att-gdb --rocprofv3 /opt/rocm/bin/rocprofv3 \
    --start begin_iteration --timeout 50ms -d traces -- ./application
```

You can install it to a chosen prefix with `cmake --install build/att-gdb
--prefix /your/prefix`. The installed command discovers rocprofv3/ROCgdb from
its installation or PATH, with `/opt/rocm` as a fallback. `--rocprofv3` and
`--rocgdb` override those choices. No Python package installation is required
for the launcher or extension.

Developer tests require pytest and a C++ compiler. The GPU tests additionally
need HIP, ROCgdb with Python, an ATT-capable GPU, and the trace decoder used by
rocprofv3. Debugger state tests use ROCgdb or GDB with Python and a CPU fixture
with controlled ROCTx calls; they are skipped if neither debugger is available:

```sh
cmake -S source/bin/rocprofv3-att-gdb -B build/att-gdb \
    -DROCPROFV3_ATT_GDB_GPU_TESTS=ON
cmake --build build/att-gdb
ctest --test-dir build/att-gdb --output-on-failure
```

`ROCPROFV3_ATT_GDB_ROCM_ROOT` selects the test ROCm installation and
`ROCPROFV3_ATT_GDB_GPU_ARCH` selects the HIP target (default `native`). Test
coverage includes breakpoint/timeout capture, competing stop sources, literal
application arguments, concurrent CPU trigger hits, repeated captures,
cancellation, preservation of user breakpoints, missing triggers, duplicate/stale
commands, disconnect cleanup, identical start/stop source lines for a loop round
trip, warmup skipping (including interactive use and an
unreached selected hit), and a HIP host-stub trigger with inlining disabled.
Regression cases cover blocked timed stops, deadline completion grace, exclusion
of slow profiler setup and debugger continuation from the timeout, manual
continuation of shared user breakpoints, short captures followed by rearming,
preservation of new user breakpoint/signal stops while a control call is pending,
and connection retry after a loader stop.
GPU checks decode the traces and assert that only the inside kernel was traced.

This is the single-process launch prototype. It uses a separate, preloaded
helper to call existing ROCTx controls and does not add a maintained control API
to the rocprofv3 tool. The helper blocks on a socket and optional one-shot timer
during capture. The Python extension runs in ROCgdb and performs GDB operations
only on the debugger thread, deferring breakpoint mutations through stop events.

Application ROCTx Pause/Resume shares the prototype's state. Other application
threads and GPU work remain running during trigger handling. Start/stop
latencies and a host timer do not promise exact GPU boundaries; already running
waves are not forced through CWSR. Prototype launch support does not implement
production `rocprofv3 --attach`, GPU breakpoints, or multi-process coordination.
See the [user guide](../../docs/how-to/using-thread-trace.rst) for usage and the
[design plan](../../docs/conceptual/debugger-triggered-att-plan.md) for production
requirements.
