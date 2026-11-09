(range-replay-host)=
# Host range replay

Packet replay records the dispatch packets a range submits and re-submits them. That is cheap, but
it only reproduces a range whose whole effect is those packets on one queue. A real time-stepping
loop is not like that. Each step mixes kernels on several streams, device-to-device and host-device
copies, memsets, graph launches, scratch allocations, and host computation whose results feed the
next step's kernels. Packet replay declines such a range, and it has to: the recording does not
contain the copies, the host computation, or the second stream.

Host replay is the alternate path for those ranges. Nothing is recorded. The application's own code
between begin and end *is* the program, and each extra pass runs it again. The application either
brackets that code with a macro and `rocprofiler_range_replay_end`, which jump back with
`siglongjmp`, or passes it as a function to `rocprofiler_range_replay_host_run`, which calls it once
per pass (see "API-driven host replay" below). The jump-based protocol:

```text
 ROCPROFILER_RANGE_REPLAY_HOST_BEGIN  <-----------------------------+
   (sigsetjmp in the caller's frame; begin + CONFIG on first entry)  |
                                                                     |
   ... application code: kernels, copies, memsets, graphs,           |
       malloc/free pairs, host computation, several streams ...      |
                                                                     |
 rocprofiler_range_replay_end()                                      |
   fence compute queues, PASS exit                                  |
   more passes?  yes -> restore device memory + registered host      |
                        memory, PASS enter, siglongjmp  -------------+
                 no  -> CLOSE, return
```

## Opting in

Both sides have to opt in.

The tool sets `host_replay` on the CONFIG record, alongside `pass_count_cb`:

```cpp
case ROCPROFILER_RANGE_REPLAY_CONFIG:
    data->pass_count_cb = my_pass_count;
    data->host_replay   = 1;
    break;
```

The application opens the range with the macro instead of calling `rocprofiler_range_replay_begin`
directly. The macro takes the jump target in the caller's frame, which is the only place it stays
valid: a `sigsetjmp` inside `begin()` would be dead as soon as `begin()` returned.

```cpp
#include <rocprofiler-sdk/experimental/range_replay.h>
#include <setjmp.h>

rocprofiler_status_t status;
hipDeviceSynchronize();             // on every participating device
rocprofiler_range_replay_host_track_memory(&host_state, sizeof(host_state));
ROCPROFILER_RANGE_REPLAY_HOST_BEGIN(range_id, status);

for(int step = 0; step < n_steps; ++step)
    time_step(step);                  // kernels, copies, host computation, ...
hipDeviceSynchronize();

rocprofiler_range_replay_end();       // jumps back to the macro N-1 times, then returns
```

The macro expands to a statement. ISO C only allows `sigsetjmp` as the entire controlling
expression of an `if`, `switch`, or loop, or compared against a constant there. So the jump target is
taken in an `if`, never in a ternary or behind `||`. CONFIG is delivered once. On re-entry the macro
writes `ROCPROFILER_STATUS_SUCCESS` to `status` without calling begin again.

An application that resolves the SDK with `dlsym` cannot use the macro, because the macro names the
SDK functions directly. It spells the same thing out:

```cpp
if(sigsetjmp(*static_cast<sigjmp_buf*>(jmpbuf_fn()), 1) == 0)
    begin_status = begin_fn(range_id);
```

`rocprofiler_range_replay_host_jmpbuf()` arms the jump for the next begin on the thread. Begin
consumes the arming whether or not host replay is then used, so a buffer from an earlier range is
never a jump target.

| Tool sets `host_replay` | Application | Result |
|---|---|---|
| no | any | packet replay |
| yes | armed the jump (the macro) | host replay |
| yes | called `rocprofiler_range_replay_host_run` | host replay |
| yes | called begin without arming the jump | declined, `HOST_JMP_NOT_SET` (the range runs once) |

## API-driven host replay

`rocprofiler_range_replay_host_run(range_id, body, user_data)` runs the same protocol without a
jump. The application passes the range body as a function. The SDK opens the range as begin does
(CONFIG is delivered once), calls the body for every pass, and returns after CLOSE:

```text
 rocprofiler_range_replay_host_run(range_id, body, user_data)
   begin + CONFIG
   +--> body(pass, user_data)       kernels, copies, host computation, ...
   |    fence compute queues, PASS exit
   |    more passes?  yes -> restore device memory + registered host
   +--------------------- memory, PASS enter
                  no  -> CLOSE, return
```

From C:

```c
static void
run_time_steps(uint64_t pass, void* user_data)
{
    struct solver* s = user_data;
    for(int step = 0; step < s->n_steps; ++step)
        time_step(s, step, pass);  // kernels, copies, host computation, ...
    hipDeviceSynchronize();
}

hipDeviceSynchronize();  // on every participating device
rocprofiler_range_replay_host_track_memory(&solver.host_state, sizeof(solver.host_state));
rocprofiler_range_replay_host_run(range_id, run_time_steps, &solver);
```

From C++17, the header's `rocprofiler::range_replay::host_run` wrapper forwards any callable,
invoked as `f(pass)` or `f()`. It is called by reference, so captures are not copied:

```cpp
rocprofiler::range_replay::host_run(range_id, [&](uint64_t pass) {
    auto residuals = std::vector<double>(n_steps);  // destroyed at the end of every pass
    for(int step = 0; step < n_steps; ++step)
        residuals[step] = time_step(step, pass);
    hipDeviceSynchronize();
});
```

Both paths share the fence, the snapshot, the restore, the divergence check and the decline rules
on this page. They differ in how a pass is re-entered:

| | Macro and `rocprofiler_range_replay_end` | `rocprofiler_range_replay_host_run` |
|---|---|---|
| Re-entry | `siglongjmp` to the caller's `sigsetjmp` | the SDK calls the body again |
| C++ objects with destructors | must not be live across the jump | allowed; destroyed at the end of each pass |
| Locals the range modifies | `volatile`, or re-initialized in the range | no rule; the body's locals start fresh each pass |
| Callers | C or C++ with POSIX `sigsetjmp`, same function and thread | anything that can pass a C function pointer |
| Exceptions | must not leave the range | close the range with `BODY_EXCEPTION` and propagate |
| `HOST_JMP_NOT_SET` | when the jump was not armed | never |
| Cost per pass | one `siglongjmp` | one indirect call |

Rules 1 to 3 of the application's contract below do not apply to the body.
Rules 4 to 7 do: GPU work submitted before the call must be complete, and all GPU work the body
submits must be complete when it returns.

The tool still chooses the replay. Without `host_replay` the body is called once, with pass 0, and
its dispatches are recorded and packet-replayed as between begin and end.

When the range cannot be opened, the body still runs once, so the application's work happens
whether or not a tool replays it, and `host_run` returns the error begin would have returned:
`CONTEXT_NOT_FOUND`, `CONTEXT_NOT_STARTED`, or `INVALID_ARGUMENT` when a range is already open on
the thread. The body is then passed 0, or the current pass of the enclosing host range, and host
regions it registers are dropped when `host_run` returns rather than carried to the next range. A
NULL body returns `INVALID_ARGUMENT` and runs nothing.

Ranges do not nest. From inside the body, `rocprofiler_range_replay_begin`,
`rocprofiler_range_replay_end` and `rocprofiler_range_replay_host_run` return `INVALID_ARGUMENT`
and leave the running range unchanged; a nested `host_run` still runs its own body once.
`rocprofiler_range_replay_get_host_pass` returns the body's `pass`, and
`rocprofiler_range_replay_host_track_memory` behaves as between begin and end.

If the body throws a C++ exception, the range closes without its remaining passes. The SDK fences the
pass's GPU work, raises PASS exit for a re-executed pass so every PASS enter is paired, and CLOSE
reports `BODY_EXCEPTION`, or the reason the range had already been declined for. The exception then
propagates to the caller of `host_run`. Nothing is restored: the application continues from the
state the throwing pass left. The body must return or throw; leaving it with `longjmp` leaves the
range open on the thread.

Other languages call the C entry point directly. With Python `ctypes`, for example:

```python
import ctypes

sdk = ctypes.CDLL("librocprofiler-sdk.so")  # already loaded by the tool
BODY = ctypes.CFUNCTYPE(None, ctypes.c_uint64, ctypes.c_void_p)
sdk.rocprofiler_range_replay_host_run.argtypes = [ctypes.c_uint64, BODY, ctypes.c_void_p]
sdk.rocprofiler_range_replay_host_run.restype = ctypes.c_int

body = BODY(lambda pass_index, _: run_iteration(pass_index))
sdk.rocprofiler_range_replay_host_run(range_id, body, None)
```

A Python exception does not propagate through a `ctypes` callback: `ctypes` reports it and the
callback returns normally, so the SDK continues with the next pass. Python objects are host memory
like any other and are not rewound.

## What is restored before each extra pass

- **Device memory on every GPU agent.** This is the same snapshot kernel replay and packet range
  replay use: tracked coarse-grained device allocations plus module-scope `__device__` /
  `__constant__` variables. Packet replay snapshots one agent. Host replay snapshots all of them,
  because the range may touch any of them.
- **Host memory the application registered** with
  `rocprofiler_range_replay_host_track_memory(ptr, size)`. A region registered while no range is
  open is captured when the next range opens on the thread. A region registered during the
  application's own run of the range is captured at the call. Registrations made during re-executed
  passes are ignored. Registrations last until the range closes. Where registered regions overlap,
  each byte is restored from the earliest capture that covers it.

Nothing else is rewound. That includes unregistered host memory, files, sockets, HIP's own host-side
state, random-number generators, and the clock. The application decides what the range depends on,
and registers it.

No exit snapshot is taken or restored. The last pass re-ran the host code as well, so host and
device state at the end are consistent with each other, and the application continues from them.
For a deterministic range they are what pass 0 produced. With `ROCPROF_RANGE_REPLAY_VERIFY=1`, the
device and registered-host digests at the end of the last pass are compared with the end of pass 0.
Any difference is reported as `divergence_count` on CLOSE.

## What still declines

| Condition | Status |
|---|---|
| Tool asked for host replay, application did not arm the jump | `HOST_JMP_NOT_SET` |
| The body passed to `rocprofiler_range_replay_host_run` threw | `BODY_EXCEPTION` |
| Another thread dispatches to any GPU while the range is open | `CONCURRENT_DISPATCH` |
| Another thread copies into device memory while the range is open | `MEMORY_COPY_IN_RANGE` |
| Tracked device allocation addresses, sizes, or lifetimes differ at the end of a pass (a leak, or freeing/replacing pre-existing memory) | `ALLOCATION_CHANGED_IN_RANGE` |
| A code object is unloaded while the range is open | `CODE_OBJECT_CHANGED_IN_RANGE` |
| The entry snapshot could not be captured (host memory pressure) | `SNAPSHOT_FAILED` |
| No dispatch in the application's own run | `NO_DISPATCH` |
| Queue interposition (write-pointer virtualization) is in use | `UNSUPPORTED_QUEUE_PATH` |

None of these apply to host replay: `MULTI_QUEUE`, `MULTI_AGENT`, `GRAPH_LAUNCH`,
`PROGRAM_TOO_LARGE`, `UNKNOWN_KERNARG_SIZE`, `STAGING_FAILED`, or a copy made by the range's own
thread.

A decline detected at the end of a re-executed pass stops the range there. CLOSE reports the reason,
and the PASS callbacks already delivered stand.

A code object loaded during the application's own run of the range does not decline. HIP loads a
module on the first launch of one of its kernels, so declining would refuse ordinary applications.
The module variables of such a code object are not in the entry snapshot, so they are not rewound,
and the SDK logs a warning. Launch every kernel once before the range (a warm-up step) to avoid
both. Do the same for any library that allocates lazily, such as FFT plans or BLAS workspaces, or
the first-use allocation declines `ALLOCATION_CHANGED_IN_RANGE`.

## The application's contract

`siglongjmp` restores registers and the stack pointer. It does not run destructors or restore
locals. The code between the macro and `end` must follow these rules. A `host_run` body follows
rules 4 to 7 only:

1. **Same thread, same function.** The frame that expanded the macro must still be live at `end`.
2. **No C++ object with a non-trivial destructor may be live across the jump.** Put the range body
   in a function, or in a block that closes before `end`, so its `std::vector`s and similar objects
   are destroyed before the jump.
3. **Locals of the enclosing function that the range modifies** must be `volatile` or re-initialized
   inside the range. ISO C leaves them indeterminate after the jump. A loop counter declared inside
   the range is fine.
4. **Register host state the range reads and writes** with
   `rocprofiler_range_replay_host_track_memory`.
5. **Synchronize all GPU work before begin and before `end` on every pass.** This includes
   asynchronous copies and non-compute queues on every participating device. Use their supported
   completion mechanisms (for example, `hipDeviceSynchronize` on each participating device).
   The SDK additionally fences compute queues and drains dispatch handlers, but these do not wait
   for untracked asynchronous copies or non-compute work. No other thread may submit GPU work or
   copy into device memory while the range is open.
6. **Keep once-only side effects out of re-executed passes.** `rocprofiler_range_replay_get_host_pass`
   returns 0 on the application's own run and 1..N-1 after that.
7. **Balanced allocations only.** `hipMalloc` / `hipFree` pairs inside the range are fine. So is
   `hipStreamCreate` / `hipStreamDestroy`, as long as it does not allocate device memory that
   outlives the pass. Pool allocators that cache across frees (`hipMallocAsync`) are not.

## Interaction with services

Re-executed passes are the application's own submissions, so every service sees them as ordinary
dispatches. That includes counters, thread trace, SPM, PC sampling, and API and kernel tracing.
Unlike packet replay, a pass is not serialized dispatch by dispatch: streams overlap exactly as they
did in pass 0.

The PASS context toggles work as in packet replay and kernel replay. They are legal during PASS
`PHASE_ENTER` and sticky for the rest of the range. The override is installed for the whole host pass
rather than only around a submit, because the dispatches that must honor it are the application's.

## Cost

Per range, host replay pays one fence of every compute queue and one snapshot of all tracked device
memory on every GPU agent (held in host memory until CLOSE). Per extra pass it pays one fence and one
restore of that snapshot, plus a `memcpy` of each registered host region. The passes themselves run
at the application's own speed. Size ranges so the snapshot is small relative to the range's
runtime, just as for packet replay. See [Performance](range_replay_performance.md).

## Tests

- `range-replay-host-replay-test` is a GPU-free unit test. It covers the arming consumption, the
  begin macro's two returns, the registry rules that differ from packet replay, and host-region
  capture and restore, including a full jump-and-rewind loop. It also covers `host_run` with no
  context (the body runs once) and with a NULL body.
- `range-replay-host-run-test` is a GPU-free unit test of `host_run` against a tool registered with
  `rocprofiler_force_configure`. It covers the pass sequence, packet mode, nested entry points in
  the body, host-region restore, exceptions on the first and on a replayed pass, and the C++
  wrapper.
- `tests/range-replay-host-api` is the `host_run` integration test. Each step runs kernels on two
  streams joined by events and a host computation that feeds the next step, through the C++
  wrapper. The application checks every pass and the final state bit for bit against a host
  reference of one execution. Variants check that packet replay declines the range and that a body
  that throws on pass 1 closes it with `BODY_EXCEPTION`.
- `tests/range-replay-host` is the integration test. A time-stepping solver runs inside one range.
  Each step does an FFT, a spectral filter, an inverse FFT, a 5-point stencil, Jacobi sweeps, a
  device-to-device copy, a memset, a device-to-host residual reduced on the host, and a
  host-computed boundary row copied back for the next step. The work runs on two streams joined by
  events. Variants add a HIP graph launch per step, a scratch `hipMalloc` / `hipFree` per step, an
  open-ended pass count stopped by `replay_continue_cb`, and a local counter stop. The checks are
  that every pass ends with the same device checksum and host state and that registered host state
  was rewound. They also check the PASS order, a divergence count of 0, and the exact counter record
  count. Two negative variants check that packet replay declines the same range and that an unarmed
  application declines `HOST_JMP_NOT_SET`.
