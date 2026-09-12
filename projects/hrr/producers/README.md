# HRR region producers

HRR interposes the HIP dispatch table, so it records exactly the memory that
crosses a HIP API and nothing else. Two things routinely fall outside that:

- **The block layout inside a segment.** A framework allocator calls `hipMalloc`
  once for a large segment and then carves per-object blocks out of it with host
  pointer arithmetic. PyTorch's HIP caching allocator is the usual example. HRR
  sees the segment; the blocks never cross a HIP entry point, so at replay a
  write that ran past the end of a tensor lands in a live neighbour instead of
  faulting, exactly as it did in the recording — and the recording is the run you
  were trying to explain.
- **Whole allocations that bypass HIP.** A library calling
  `hsa_amd_memory_pool_allocate` directly, a custom VMM pool, memory imported
  from another process. Kernel arguments pointing into those resolve in no map at
  all, so they reach the GPU as addresses belonging to a process that no longer
  exists.

A **producer** is any code that knows one of these facts and writes it down. It
lives outside the HIP runtime, calls no HRR function, and links against nothing:
it appends fixed-size records to a file. The format in
[`hrr_regions.h`](../hrr_regions.h) is the entire contract.

## Where to write

The capture writer puts each process's archive at
`$HIP_HRR_CAPTURE_OUTPUT/pid-<getpid()>/`. A producer writes to the `regions/`
subdirectory of that path:

```
$HIP_HRR_CAPTURE_OUTPUT/pid-<pid>/regions/<producer-name>.hrrr
```

Pick a name that identifies the producer, so several can run at once; the
replayer merges every `*.hrrr` it finds and orders the records by timestamp.

**Checking whether capture is active** reduces to checking that directory
exists. There is no symbol to resolve and nothing to `dlopen`:

```python
active = "HIP_HRR_CAPTURE_OUTPUT" in os.environ and os.path.isdir(
    os.path.join(os.environ["HIP_HRR_CAPTURE_OUTPUT"], f"pid-{os.getpid()}"))
```

The runtime creates the directory when capture starts, which may be after your
producer loads, so re-check rather than deciding once at import time.

## What to write

An 8-byte file header, once, then batches. Every integer is little-endian.

```
file header (8 bytes)            magic = 0x52525248 ("HRRR"), version = 1, u16 zero
batch, repeated:
  hrr_event_header (32 bytes)    event_type = 0xFFFD
                                 sequence_id = 0 (unused; records order by time)
                                 timestamp_ns = CLOCK_MONOTONIC ns
                                 thread_id = 0
                                 payload_length = 40 + 32*n
                                 reserved[2] = 0
  u32 n                          number of records in this batch
  u32 flags                      0
  hrr_region_rec[n] (32 each)    u8 op, u8 kind, u8 device, u8 flags,
                                 u32 tag, u64 base, u64 size, i64 mono_ns
```

- `op` — `0` ADD (the region becomes live), `1` DEL (it stops being live).
- `kind` — `0` BLOCK, a sub-range of memory the runtime already knows about;
  `1` SEGMENT, a whole allocator segment.
- `base`, `size` — the device VA range, as the recorded process sees it.
- `device` — the ordinal the memory lives on. Fill it even on a single-GPU run:
  when the replayer has to back a segment that bypassed HIP, this is what puts
  the buffer on the right GPU, and a wrong ordinal silently places it on
  another one. It is not part of the lookup key — a VA identifies its
  allocation on its own within a process — so it matters only for placement.
- `mono_ns` — when the change happened, on `CLOCK_MONOTONIC`. **Zero means "live
  since before this stream began"**, which is how you declare state that already
  existed when your producer started observing. Baseline is therefore just a run
  of ordinary ADD records, not a separate mechanism.
- `tag` — yours. A pool id, a stream id, anything that helps you read your own
  output back. The replayer carries it and does not interpret it.

Declare segments **uniformly**. You do not need to work out which of your
allocations crossed a HIP API. The replayer allocates nothing for a segment
until a pointer into it turns out to resolve in no map, which only happens when
HIP really was bypassed. Declaring a segment that HRR did capture therefore
costs nothing, and it gives the replayer the bounds it needs to spot a pointer
that has fallen out of every block.

### The clock has to be the right clock

`mono_ns` is compared against the archive's own event timestamps, which come from
`amd::Os::timeNanos` — `CLOCK_MONOTONIC` on POSIX. In Python that is
`time.monotonic_ns()`. If your source of truth stamps wall-clock time (PyTorch's
memory snapshots do), convert it with an offset sampled from both clocks at the
same instant. The replayer warns when region timestamps fall outside the span of
the archive's events, which is what a wrong clock looks like, but a merely
*skewed* clock produces plausible-looking nonsense — blocks classified live at
the wrong instant — so get this right.

## Durability

The recorded process is usually one that crashes; that is why it is being
recorded. Two rules make a crash harmless:

1. **One `write()` per batch.** Build the whole batch in memory and issue a
   single write. Records are self-delimiting and fixed-size, so a partial write
   can only ever damage the last batch, and the reader discards a torn tail and
   keeps everything before it.
2. **Do not buffer for long.** Anything still in your buffer when the process
   dies is gone. Flush after each batch; the file is append-only and small.

No locking protocol is needed between producers, because each writes its own
file.

## Reference producers

- [`pytorch/hrr_torch_regions.py`](pytorch/hrr_torch_regions.py) — the PyTorch
  caching allocator, via `torch.cuda.memory._snapshot()`. Auto-loads through
  `PYTHONPATH` and needs no change to the application.

## Notes

- Region annotations are consumed by single-threaded replay only, which is
  `hrr-playback`'s default. Under `--multi-thread` there is no single ordered
  stream for the cursor to follow, and the records are ignored with a message.
- The transport is deliberately not an exported symbol. If a producer ever moves
  inside `libamdhip64` — an HSA allocation hook is the obvious candidate — it can
  emit the identical payload through the capture writer with no change to this
  format. The format is the contract, not an ABI.
