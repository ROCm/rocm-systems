# HIP Event Tracing Sample

Demonstrates the HIP event barrier tracing services. `hipEventRecord` and `hipStreamWaitEvent`
are implemented on the GPU as barrier packets; these services report those barriers as they are
enqueued and as they complete.

## Services

- HIP event callback tracing (`ROCPROFILER_CALLBACK_TRACING_HIP_EVENT`), both the `RECORD` and
  `WAIT` operations
- HIP event buffer tracing (`ROCPROFILER_BUFFER_TRACING_HIP_EVENT`)
- Kernel dispatch callback tracing (`ROCPROFILER_CALLBACK_TRACING_KERNEL_DISPATCH`, the `COMPLETE`
  operation only), so the barriers can be read against the kernels they order
- Kernel dispatch buffer tracing (`ROCPROFILER_BUFFER_TRACING_KERNEL_DISPATCH`), reporting those
  same dispatches a second time through the buffered path
- Code object callback tracing for mapping kernel IDs to kernel names

## Properties

- Every record is printed as it arrives, as an entry of at most two lines: a timestamped explainer
  line saying in prose what happened, then one line carrying all of the record's raw fields
- Timestamps are a `+N us` offset taken from `rocprofiler_get_timestamp()`; the same clock as
  the `start_timestamp`/`end_timestamp` fields inside the records. These timestamps show when the
  CPU processed a record, including buffer flushing.
- The application narrates what it is about to do through `client::narrate()`, which shares that
  stream, so its `APP ::` lines interleave in order with the records they produce
- The same entries are also accumulated and written to `hip_event_trace.log` at finalization
  (override with `ROCPROFILER_SAMPLE_OUTPUT_FILE`)
- Buffer size of 4096 bytes which is automatically flushed once >= 87.5% of buffer is filled (3584 bytes)
- Creation of dedicated thread for buffer callback delivery
- Receives notifications for internal thread creation

## Options

- `--size N`: elements per `scale_kernel` launch (default 1024)
- `--iterations N`: iterations of each case (default 2)
- `--spin-iters N`: loop count of the long kernel that keeps a record barrier in flight
  (default 5000000); raise it if case 1 reports that no `HIP_EVENT_WAIT` barrier completed

## Callback tracing

A HIP event barrier produces three callbacks:

| Phase | Meaning | Timestamps |
|---|---|---|
| `ROCPROFILER_CALLBACK_PHASE_ENTER` | the barrier is about to be enqueued | not yet available |
| `ROCPROFILER_CALLBACK_PHASE_EXIT` | the barrier has been enqueued | not yet available |
| `ROCPROFILER_CALLBACK_PHASE_NONE` | the barrier has completed on the GPU | populated |

Because the enqueue phases carry no timestamps at all, the sample omits the `start`/`end`/`elapsed`
columns from those entries rather than printing empty ones. The buffer record is emitted only at
completion and always carries timestamps.

## Buffer tracing

Buffered records are also available for HIP event tracing and are in this sample to show their
usage. The CPU timestamps provided are when the buffer record is received by the sample.
Internally, the records are appended to a ring buffer and handed to the buffer callback thread only
once the buffer passes its watermark or something flushes it.

Kernel dispatch is traced through both services, so each dispatch is reported twice: once inline
by the callback service as `DISPATCH`, and once through the buffer as `DISP BUF`. Both kinds of
buffered record go into the same buffer, so a single flush delivers them together.

The sample flushes once per case, after every iteration of that case has finished, rather than
after each iteration. Every buffered record from every iteration of the case then arrives as one
batch, barrier records and dispatch records intermixed:

```
[+ 745828.749 us] APP       :: all case 1 iterations are complete; flushing the buffer ...
[+ 745916.882 us] DISP BUF  :: _Z11spin_kernelPfm.kd dispatch on queue 1 delivered via the buffered service ...
[+ 745946.747 us] EVENT BUF :: HIP_EVENT_RECORD barrier on queue 1 delivered via the buffered service ...
[+ 745960.067 us] EVENT BUF :: HIP_EVENT_WAIT   barrier on queue 2 delivered via the buffered service ...
[+ 745972.216 us] DISP BUF  :: _Z15follower_kernelPfm.kd dispatch on queue 2 delivered via the buffered service ...
```

The sample does not sort them. Records come out in the order they were appended to the buffer,
which is not guaranteed to match their `start`/`end` timestamps; compare the fields rather than
reading the batch top to bottom. The inline `DISPATCH` and `EVENT CB` entries earlier in the log
are the ones that appear in true chronological order.

With a large enough `--iterations` the buffer will reach its watermark before the explicit flush
and deliver some records early on its own, which is the same mechanism seen from the other side.

## What the workload shows

Two streams and one dedicated event per case. A warm-up phase runs one of each kernel and flushes
first, so that one-time setup work (the `fillBufferAligned` blits behind `hipMemset`, code object
load, queue creation) does not land in the middle of the first traced iteration. All iterations of
case 1 then run before any iteration of case 2, and each phase is introduced by a banner. Stream B
always runs its own, shorter `follower_kernel` after its wait.

**Case 1, deferred wait.** The long `spin_kernel` is launched on stream A, the event is recorded
on stream A while that kernel is still running, and stream B then waits on the event. Because the
event has not completed, HIP must schedule a real barrier:

```
[+ 647129.214 us] EVENT CB  :: HIP_EVENT_WAIT barrier about to be enqueued on queue 2, waiting on an event recorded in queue 1
                    op=HIP_EVENT_WAIT   phase=ENTER cid=  10 queue=2 src_queue=1 event=0x00003fa55c90
```

followed at completion by the `phase=NONE` callback, carrying timestamps and sitting in true
chronological order between the two kernel dispatches it separates:

```
[+ 667896.765 us] DISPATCH  :: _Z11spin_kernelPfm.kd completed on queue 1
[+ 667908.973 us] EVENT CB  :: HIP_EVENT_RECORD barrier completed on the GPU on queue 1
[+ 667918.588 us] EVENT CB  :: HIP_EVENT_WAIT barrier completed on the GPU on queue 2, waiting on an event recorded in queue 1
                    op=HIP_EVENT_WAIT   phase=NONE  cid=  10 queue=2 src_queue=1 event=0x00003fa55c90 start=2423053642410275 end=2423053707180118 elapsed= 64769.843us
[+ 671703.152 us] DISPATCH  :: _Z15follower_kernelPfm.kd completed on queue 2
```

**Case 2, already-complete wait.** The event is recorded behind a short kernel and then
synchronized on, so it is complete before stream B waits. No barrier is scheduled, and nothing at
all is traced for the wait; the narration is followed directly by stream B's kernel:

```
[+ 712020.406 us] APP       :: stream B waits on the already-complete event; expect NO HIP_EVENT_WAIT record at all, ...
[+ 712025.844 us] APP       :: launching follower_kernel on stream B; nothing gates it
```

This shows the documented behavior of `rocprofiler_hip_event_operation_t`: not every
`hipStreamWaitEvent` call produces a barrier, and when none is produced no enqueue or completion
callback is generated.
