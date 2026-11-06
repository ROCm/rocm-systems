# Range replay samples

Small **custom tools** plus a HIP application, following the same layout as
`samples/kernel_replay/`: each sample is a shared-library client (`.so`) preloaded onto a minimal
app (`main.cpp`). These are **not** `rocprofv3` integration tests.

Build with `-DROCPROFILER_BUILD_SAMPLES=ON`.

## How range replay differs from kernel replay

Kernel replay re-executes one dispatch **in place**, before the application observes its
completion, so the application never knows it happened and the tool alone drives it.

A range covers a *sequence* of dispatches the application has **already observed**. There is no
"in place" left, so the SDK records the range while it runs and re-executes the recording
afterwards:

- **pass 0** is the application's own execution, which the SDK observes rather than drives, and
  which raises no PASS callback
- **passes 1..N-1** are SDK-driven re-executions from the memory state captured at range entry

That is also why the application is a participant here and not just a subject: only it knows where
its range boundaries are. `main.cpp` calls `rocprofiler_range_replay_begin` / `_end` around the
dispatches it wants replayed. It resolves those two entry points with `dlsym` out of the preloaded
client rather than linking the SDK, because linking the SDK into the HIP executable makes HIP
report `hipErrorInvalidDeviceFunction` on the first kernel launch on gfx942.

## The application

`main.cpp` runs three dispatches of one kernel inside a range:

```
acc = 0  ->  acc*3+1  ->  acc*3+2  ->  acc*3+3   ==  18
```

Each dispatch reads what its predecessor wrote, so the chain is order-dependent and not
idempotent. Two separate checks use it, and they verify different things:

- **`acc == 18` in the application** says the replay window was **transparent**: the application
  saw exactly the value its own execution produced. The executor restores the range-exit state
  after the last pass, so this covers that restore and the window not corrupting device memory
  along the way. All three samples check it, including the two where no pass runs.
- **`divergence_count == 0` at `CLOSE`** says the range was **repeatable**: the state after the
  final replayed pass matched the state the application's own execution left behind. This is the
  check that a between-pass rewind actually happened and that the recording replayed in order —
  the range-exit restore runs last and would otherwise hide both. It needs
  `ROCPROF_RANGE_REPLAY_VERIFY=1`, which the `range-replay-basic` CTest target sets.

The three dispatches share a kernel object but differ in their kernargs, so the replay has to
stage a distinct kernarg slot for each of them.

`RR_APP_MODE` selects what else the application does inside the range (the CTest targets set it):

| Mode | Inside the range |
|---|---|
| `plain` (default) | Three dispatches on one stream |
| `multi-queue` | The same, plus a dispatch on a second stream |

`multi-queue` also needs `GPU_MAX_HW_QUEUES` to be at least 2 (the CTest target pins it to 4).
HIP pools hardware queues and round-robins streams onto them, so with a pool of one the two
streams would share a queue and the range would be replayed rather than declined.

## What each sample shows

| Sample | Asks for | CLOSE status | Shows |
|---|---|---|---|
| `range-replay-basic` | 4 passes | `REPLAYED` | The whole loop: one CONFIG, three PASS callbacks (passes 1-3), one CLOSE, and a zero divergence count. |
| `range-replay-opt-out` | nothing | `NO_PASS_COUNT` | Leaving `pass_count_cb` NULL is the per-range opt-out: the range is still opened, tracked and closed, but no pass runs. |
| `range-replay-decline` | 4 passes | `MULTI_QUEUE` | A range the SDK refuses. CLOSE names the reason, and the application's result is untouched. |

`range-replay-decline` is the shape every decline takes, not a special case. A range is replayed
only when every recorded dispatch targets one queue on one agent, no HIP graph launch occurs
inside it, no async copy writes device memory inside it, no device allocation is made or freed
inside it, and no other thread dispatches to the same agent while it is open. Each of those has
its own `rocprofiler_range_replay_status_t`, and all of them reach the tool the same way: the
application's live run completes normally, and CLOSE reports why the extra passes were skipped.
The sample triggers `MULTI_QUEUE` because two HIP streams are the one decline an application can
provoke deterministically on a single GPU.

## Divergence checking

Set `ROCPROF_RANGE_REPLAY_VERIFY=1` to have the SDK hash the snapshot regions after the final pass
and compare them against the state the application's own execution produced. CLOSE then reports
`divergence_count`: the number of regions that differed. A non-zero count means the range is not
self-contained under the snapshot's coverage, so per-pass measurements describe different inputs.
It is off by default because it costs an extra snapshot and a full hash of it.

`range-replay-basic` runs with it on, because it is the only way the sample can tell a repeatable
replay from a replay that merely finished. Everything else a tool observes — the pass callbacks,
the `REPLAYED` status, and the application's own result — looks identical whether or not the
between-pass rewind worked.

## Run

From the sample build directory:

```bash
ctest -R '^range-replay-' --output-on-failure
```

Or manually:

```bash
export LD_PRELOAD=./librange-replay-basic-client.so
./range-replay-basic
```

## See also

- `source/docs/conceptual/range_replay/index.md` — the model, and what makes a range a candidate
- `source/docs/conceptual/range_replay/range_replay_soundness.md` — what replay does and does not
  preserve
- `samples/kernel_replay/` — the single-dispatch service this one extends
