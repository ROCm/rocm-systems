# hipCopyEngineBench

Microbenchmarks for kernel-submitted copies through the HIP copy-engine API
(`hip/hip_ext_copy_engine.h`: `hipExtCopyEngineGetDeviceHandles` on the host,
`hipExtCopyEngineSubmitLinearOneLane` in the kernel). They mirror two tools of rocm-xio's
`examples/sdma-ep-bench` (`sdma-ep-bw`, `sdma-ep-latency`) and the device-initiated allgather of its
`examples/sdma-ep-collectives`, with the same options and CSV layout where the public HIP API
allows it, and use nothing but that API: no rocm-xio, no HSA queue setup, no ABCE ring or packet-builder headers.

| Tool | Measures |
|---|---|
| `copy-engine-bw` | Bandwidth of copies pushed from `--srcGpu` to one or more peer GPUs by warps of one kernel. |
| `copy-engine-latency` | Submit-to-completion latency of one copy (or a few), split into submit and transfer time. |
| `copy-engine-allgather` | In-place allgather across `--numGpus` GPUs (default all), one process, one kernel per GPU. |
| `copy-engine-allgather-mpi` | The same allgather with one MPI rank per GPU, like the reference; built when CMake finds MPI. |

## Build

Needs a HIP built with the kernel-side copy engine (`hip_ext_copy_engine.h`); the sample is skipped
when that header is missing. Put that HIP first on `CMAKE_PREFIX_PATH`:

```bash
cmake -S projects/hip-tests/samples/1_Utils/hipCopyEngineBench -B build \
  -DCMAKE_PREFIX_PATH="<hip-install>;/opt/rocm" -DCMAKE_CXX_COMPILER=amdclang++
cmake --build build
```

Inside the full samples tree: `make build_utils`.

## Run

The runtime must publish kernel-side rings, otherwise each tool says so and exits with status 1:

```bash
export DEBUG_CLR_ENABLE_ABCE=1 GPU_MAX_SDMA_DEVICE_QUEUES=16
./copy-engine-bw -b 1024 -e 4194304 -c 4 -d 7 --wgsPerDestination 2
./copy-engine-latency -b 256 -e 1048576 -c 1
./copy-engine-allgather -g 8 -b 4096 -e 268435456
mpirun --bind-to core --map-by numa -x LD_LIBRARY_PATH -x DEBUG_CLR_ENABLE_ABCE \
  -x GPU_MAX_SDMA_DEVICE_QUEUES -n 8 ./copy-engine-allgather-mpi
```

`--help` lists every option. Sizes sweep in powers of two from `-b` to `-e`. Each size is verified
(every destination word must be `0xDEADBEEF`) unless `--skip-verification`. Results go to stdout and
to the CSV file (`-o`). All timings are in microseconds; the device clock rate comes from
`hipDeviceAttributeWallClockRate`, not a fixed 100 MHz.

The host picks the copy engines before launch: each tool calls `hipExtCopyEngineGetDeviceHandles`
for its source/destination pair and passes the handles to the kernel by value, so a submit does no
lookups. A completion is a `hipExtCopyEngineSignalSet` on a word the kernel then polls; the word
carries a rising epoch, so it is never reset.

`--signalMemory device|host` picks where the per-copy completion words live (default `device`,
uncached device memory). Host memory was measured slower: 8.5 us vs 6.9 us for a 256 B copy.

## Sweep scripts

`scripts/` has the sweep runners, each writing a timestamped `results_*` directory with one CSV per
run, a merged `summary.csv` and `benchmark.log` (the exact commands). They look for the executables
in `<sample>/build` by default; set `BUILD_DIR`, or `BENCHMARK` / `LATENCY_BENCHMARK`, to point
elsewhere. They default `DEBUG_CLR_ENABLE_ABCE=1` and `GPU_MAX_SDMA_DEVICE_QUEUES=16` unless
already set.

| Script | Sweep | Reference script |
|---|---|---|
| `bench-bandwidth-single.sh` | destinations 1..7, copy sizes 1 KiB..1 GiB | `bench-bandwidth-single.sh` |
| `bench-bandwidth-multiproducer.sh` | work-groups per destination 2..32 (doubling), 7 destinations, 100 copies | `bench-bandwidth-multiproducer.sh` (its queues x work-groups-per-queue grid becomes total work-groups per destination) |
| `bench-contention.sh` | work-groups sharing one destination: 16..304, 100 copies | `bench-contention-singlequeue.sh` |
| `bench-latency.sh` | copy size 256 B..1 MiB, one run per size | `bench-latency.sh` (no `FINE_GRAINED`) |
| `bench-allgather.sh` | GPU count 2..8 (doubling), total size 4 KiB..256 MiB; `USE_MPI=1` runs the MPI build under `MPIRUN` | `sdma-ep-collectives` allgather, one run per GPU count in place of `mpirun -n` |

Every setting is an environment override, named as in the reference where the setting exists
(`MIN_COPY_SIZE`, `MAX_COPY_SIZE`, `NUM_COPY_COMMANDS`, `NUM_DESTINATIONS`, `SRC_GPU`, `DST_GPU`,
`WARMUP`, `ITERATIONS`, `OUTPUT_ROOT`, ...; the top of each script lists them). The defaults are
large, so use small overrides for a smoke run, for example
`MAX_COPY_SIZE=65536 MAX_DESTINATIONS=2 scripts/bench-bandwidth-single.sh`.

`scripts/plot-allgather.py` draws latency and bus bandwidth against size as an SVG from the
`copy-engine-allgather` CSVs (and from the reference's output converted to the same columns), with
the Python standard library only. Several CSVs per label give the median per size, or the best with
`--statistic min`.

Not ported: `bench-bandwidth-single-device-triggered.sh` (no device-triggered mode), the
packet-rate tool and its scripts (the public API has no poll or atomic packets or batched submits, so
a rate run would only time the submit call that `copy-engine-latency` already reports in
`Submit [us]`), and the three plot scripts, which read columns this port does not write (the
reserve/build/atomic latency breakdown).

## Allgather

Every GPU's buffer holds one chunk per rank (`total / numGpus` bytes each, in place). A one-block
kernel runs on each GPU with a warp per peer. That warp's first lane copies this GPU's chunk into
the peer's buffer, queues a signal in the peer's memory behind the copy ("arrived") and one in its
own ("sent"), then waits for the peer's chunk to arrive here and for its own send to finish. The
signals carry a rising epoch and are never reset. Each size is verified once before timing (every
GPU must end up with every chunk).

Two ways to run it, with the same kernel:

- `copy-engine-allgather`: one process sets up every GPU, launches one kernel on each in its own
  stream and times each GPU with events.
- `copy-engine-allgather-mpi`: one MPI rank per GPU, as in the reference. Rank `i` drives GPU `i`
  and reaches its peers' buffers and signals through IPC handles. Run it with
  `mpirun --bind-to core --map-by numa -n <GPUs>`, without the binding the times vary a lot from
  run to run; `--numGpus` must be 0 or equal to the number of ranks.

Both report the mean and the maximum over GPUs of the per-iteration event time.

Every collective starts with a barrier over all ranks (`--skip-barrier` leaves it out), like the
reference's `crossGpuBarrier`: each warp stores the epoch into its peer's barrier word and waits for
the peer's, then the block syncs. Each copy then carries two signals, one in the peer's memory
("arrived") and one in the sender's ("sent"):

- `--submit fused` (default) queues copy and both signals in one submit with
  `hipExtCopyEngineSubmitLinearTwoCompletionsOneLane`, one reservation like the reference's
  `putSignalCounter`. `--submit separate` queues the "sent" signal as a second submit.
- `--signalOp add` (default) adds 1, one atomic packet, as the reference does; `set` stores the
  epoch, which is two fence packets on gfx9. Add needs SDMA atomics, otherwise the submit returns
  `hipErrorNotSupported`.

Output columns follow the reference's: total size, time, algorithm bandwidth (`total / time`) and
bus bandwidth (`(n - 1) / n` of that), in GiB/s as in the reference, whose header says GB/s.

## Engines and handles

`hipExtCopyEngineGetDeviceHandles` returns engines for a pair, best first, one ring per engine.
Both tools use one handle per destination: rank 0, the pair's affinity engine, which is also the
engine rocm-xio picks. Every warp and work-group of a destination submits to that one ring.
`copy-engine-latency --handleRank N` submits on the Nth instead, as a diagnostic.

Rings are created on first use and count against `GPU_MAX_SDMA_DEVICE_QUEUES` (default 0: off). The
HIP API has no query for that limit; a call that cannot get a ring comes back with fewer handles
than asked for, and the tools say so. `-d` destinations need that many rings.

On an 8x MI300X node only rank 0 is ranked for the pair; the rest are the other xGMI engines in
ascending id, and the odd ones are unusable from a kernel: a 16 KiB copy takes about 7 us on an
even engine and 200-340 ms on an odd one. Which ranks that is depends on the pair (GPU 0 to GPU 1:
ranks 1, 3, 5, 7; GPU 0 to GPU 2: rank 2). Check a rank with `copy-engine-latency --handleRank N
--srcGpu S --dstGpu D -b 16384 -e 16384 -n 20` before using it.

## Differences from sdma-ep-bench

The reference kernels drive SDMA queues directly. The HIP API submits whole copies on a ring the
host chose, and every copy retires its own completion word. So:

| Reference | Here |
|---|---|
| `sdma-ep-bw --no-queue-state`, `--device-triggered` | Not available: no queue-state object, no host-preprogrammed wait-then-put. |
| `sdma-ep-bw --numOfQueuesPerDestination` | Not available: the API makes one ring per engine, not several queues on one engine, so a destination always has one. The `#Queues` CSV column is gone. |
| `sdma-ep-bw --wgsPerQueue` | `--wgsPerDestination`: work-groups per destination, all sharing its ring. |
| `sdma-ep-bw` signals | One completion word per copy instead of one atomic per batch; a warp waits for all of its copies. |
| `sdma-ep-latency --fine-grained` and the Reserve/Build/Submit/Atomic* columns | Replaced by two always-on columns: `Submit [us]` (mean per submit call, including ring reserve, packet build and doorbell) and `Transfer [us]` (last submit returned until every copy retired). |
| Reference quirks | Not reproduced: `sdma-ep-bw` printed std in ms under a us label; `sdma-ep-latency` timed only its last iteration and always reported std 0. Both now report real per-iteration statistics in us. |
| Output file defaults | `copy_engine_bw.csv`, `copy_engine_latency.csv`. |

## Reading the results

- On an 8x MI300X node, `copy-engine-bw -d 7` reached ~290 GB/s against ~41 GB/s for `-d 1`.
  Warps sharing a destination share one ring, while different destinations use different routes.
- Allgather on 8 MI300X GPUs under MPI, validation on, in microseconds per collective (mean over
  GPUs; three interleaved rounds, the range is over rounds). **Bind the ranks**:
  `mpirun --bind-to core --map-by numa` puts ranks 0-3 on cores of package 0 and 4-7 on package 1;
  with Open MPI's default placement the 4 KiB time varied between 11 and 34 us from run to run for
  the reference and for this tool alike (the kernel itself takes about 10 us; what varied was a
  gap of 2.5 to 35 us between consecutive kernels). I did not find which part of the placement
  matters.

  | | 4 KiB | 64 KiB | 1 MiB |
  |---|---|---|---|
  | reference, no barrier | 11.8 - 12.0 | 11.7 - 11.8 | 14.6 - 14.7 |
  | `--skip-barrier --signalOp add` (fused or separate) | 11.7 - 13.5 | 12.2 - 12.8 | 15.3 - 16.3 |
  | `--skip-barrier --signalOp set` (fused or separate) | 12.9 - 13.6 | 13.5 - 14.4 | 16.4 - 17.7 |
  | reference with `crossGpuBarrier` | 13.0 - 13.3 | 12.8 - 12.9 | 15.7 - 15.9 |
  | defaults (barrier, fused, add) | 13.1 - 13.5 | 13.4 - 13.9 | 16.9 - 17.1 |
  | barrier, separate, set | 14.8 - 15.2 | 15.3 - 16.2 | 18.5 - 19.1 |

  - The signal operation matters most: add is about 1.5 us faster than set, because a set is two
    fence packets where an add is one atomic. The reference uses atomic increments.
  - Fusing the second signal into the copy's submit makes no measurable difference with add and
    saves up to 0.5 us with set: the second submit overlaps with the copy already running on the
    engine.
  - The barrier costs about 1.2 us, in this tool and in the reference.
  - With add, this tool matches the reference at 4 KiB and is 0.4 - 1.5 us above it at 64 KiB and
    1 MiB. At 32 MiB and 256 MiB all variants agree (about 100 us and 724 us, 345 GiB/s).
  - Single iterations occasionally stall for milliseconds in any variant (for example 2.8 ms mean
    at 256 MiB in one run); the cause is unknown.
- A submit call costs about 2.8 us: reserve ring space, write the packets, ring the doorbell. That
  is the `Submit [us]` column.
- Mean device latency of one copy, 5 interleaved runs against `sdma-ep-latency` (which reports one
  sample per size):

  | Copy size | sdma-ep-latency | `copy-engine-latency` |
  |---|---|---|
  | 1 KiB | 6.4 - 7.2 us | 6.1 - 6.3 us |
  | 64 KiB | 7.3 - 7.5 us | 7.0 - 7.7 us |
  | 1 MiB | 26.9 - 27.4 us | 27.0 - 27.4 us |

- Keep the completion argument of the submit a constant at each call site. Selecting between a
  signal pointer and `nullptr` at run time (`last ? &completion : nullptr`) makes the compiler keep
  the signal in memory and costs about 1.7 us per call; `SubmitBatchAndWait` branches instead.
