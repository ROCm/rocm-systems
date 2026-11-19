# Minimal reproducers for the hipFile data-visibility failures

Each program here is the smallest form of one class of failure that
`hipfilewrite-visibility`, `hipfileread-gpu-visibility`,
`hipfileread-gpu-watcher-visibility`, and `hipfileread-cpu-visibility`
report. Each is one short loop: store new data,
transfer it with hipFile, check what arrived. The fixes we have measured are
in the source, at the place where they go, and each can be turned on from the
command line.

## Build and run

```bash
./build.sh <hipFile build directory>
./run-all.sh <directory on an AIS-capable file system>
```

`build.sh` needs a built hipFile (`src/amd_detail/libhipfile.so` in its build
directory) and ROCm's `hipcc`. `run-all.sh` runs every program without a fix
and then with each fix, and prints one line for each run:

```text
04-write-small-after-large-hipmemcpy   FAIL (fix: none): 186 of 200 iterations had wrong data
04-write-small-after-large-hipmemcpy   PASS (fix: hdp-flush): 0 of 200 iterations had wrong data
```

To run one program yourself, with ROCm's libraries on `LD_LIBRARY_PATH`:

```bash
bin/04-write-small-after-large-hipmemcpy <directory> [fix]
```

Run without arguments, a program lists its fixes. A program exits with 0 if
no iteration had wrong data, 1 if some had, and 2 if it could not run.

Run one program at a time, with nothing else using the GPU. Another process's
work changes what the GPU's caches hold, and can turn a failure into a pass.

## The reproducers

"Wrong" is the number of iterations with wrong data without a fix, in two
runs, on a Radeon RX 6800 XT (gfx1030, ROCm 7.15) and on an MI300X (gfx942,
ROCm 7.14). Every program takes a few seconds.

| Program | What the application does | Wrong on RX 6800 XT | Wrong on MI300X | Cause |
|---|---|---|---|---|
| `01-write-after-hipmemset` | `hipMemset`, then fastpath `hipFileWrite` | 199, 200 of 200 | 199, 199 of 200 | A |
| `02-write-after-device-to-device-hipmemcpy` | `hipMemcpy` between GPU buffers, then fastpath `hipFileWrite` | 99, 99 of 100 | 97, 98 of 100 | A |
| `02-write-after-hipmemcpywithstream-fallback` | `hipMemcpyWithStream` between GPU buffers, then **fallback** `hipFileWrite` | 99, 99 of 100 | 98, 97 of 100 | A |
| `03-write-after-kernel-without-sync` | A kernel stores, the host waits for it without HIP, then fastpath `hipFileWrite` | 50, 50 of 50 | 0 | B |
| `04-write-small-after-large-hipmemcpy` | `hipMemcpy` of 1 MiB from the host, then fastpath `hipFileWrite` of 4 KiB | 186, 185 of 200 | 0 | C |
| `05-write-small-after-fallback-read` | Fallback `hipFileRead` of 64 KiB, then fastpath `hipFileWrite` of 4 KiB | 177, 175 of 200 | 0 | C |
| `06-read-then-kernel-sees-old-data` | A kernel reads the buffer, fastpath `hipFileRead`, a second kernel reads the buffer | 50, 50 of 50 | 0 | B |
| `07-read-then-cpu-sees-old-data` | Fallback `hipFileRead` while CPU threads read the buffer, then the CPU reads it | 226, 178 of 300 | 0 | C |

`build.sh` also builds the variants the sources describe: `hipMemsetD8`,
`hipMemsetD16`, and `hipMemsetD32` for 01, and `hipMemcpyDtoD` and
`hipMemcpyWithStream` with fastpath for 02. They fail as often as the programs
they are variants of.

`05-write-small-after-hipmemcpy` is 05 with the application's own `hipMemcpy`
of the 64 KiB in place of the fallback `hipFileRead`. It fails as often (177,
179, 177 of 200 against 175, 175, 176), which shows that the fallback read
fails because of its `hipMemcpy`. A fastpath `hipFileRead` in the same place
does not fail (0 of 200 in three runs), so it is not built.

### Failures found on other GPUs

These were measured with ROCm 10.1 on an MI100 (gfx908, Xeon Gold 6132 host,
amdgpu 7.1.9) and on an RX 9070 (gfx1201, Threadripper PRO 7955WX host,
amdgpu 6.16.13). Each source file lists its fixes and what each did. An
earlier version of this file said that all of them pass on the RX 6800 XT;
the two 09 programs do not (see "09 on the RX 6800 XT" below).

| Program | What the application does | Without a fix | Fixes | Not fixes | Cause |
|---|---|---|---|---|---|
| `03-write-after-kernel-without-sync-fallback` | 03 with **fallback** `hipFileWrite` | MI100: 49 of 50 | `stream-sync`, `uncached`, `blit-copy` | `sdma-copy`, `no-sdma-hdp-flush`, `hdp-flush` | B, through fallback's SDMA copy |
| `08-write-after-cpu-stores-over-kernel-stores` | A kernel stores, the host waits for it without HIP, the host stores over the buffer, then fallback `hipFileWrite` | MI100: 1 of 50 (the first) | `sync-before-cpu-stores`, `uncached` | `hdp-flush`; `sync-after-cpu-stores` makes all 50 fail | B: L2 writes back over the host's stores |
| `09-fastpath-read-during-kernel` | A running kernel waits on a host-memory flag the host sets after a fastpath `hipFileRead` of 4 KiB | RX 9070: 10 failures in 13 iterations; RX 6800 XT: below | `bar-read`, `vram-flag`, `wait-5us`, `wait-100us`, each 0 of 200000 | `uncached`, `hdp-flush` | D |
| `09-cpu-stores-during-kernel` | The same, with the CPU storing the data through the BAR | RX 9070: 10 failures in 10 iterations; MI100: 0 of 200000; RX 6800 XT: below | `bar-read`, `vram-flag`, `wait-100us`, each 0 of 200000 | `wait-5us`, `uncached`, `hdp-flush` | D |
| `09-fallback-read-during-kernel` | The same, with a fallback `hipFileRead` of 1 MiB | RX 9070: hung the amdgpu driver (killed unfinished after 900 s; TTM workers stuck in D state, `rocminfo` hangs; the node needs a reboot) | none known | ROCm 10.1 (it hung with 7.2 too) | unknown |

The `wait-5us` and `wait-100us` results for both 09 programs were repeated on
a second RX 9070 node of the same model, with the same outcome.

Programs stop at 10 wrong iterations, which is why some lines say "10
failures in 13 iterations". `09-fallback-read-during-kernel` refuses to run
unless `REPRO_MAY_HANG_DRIVER=1` is set, and `run-all.sh` skips it unless the
variable is set.

#### 09 on the RX 6800 XT

Measured 2026-11-25 on the RX 6800 XT (gfx1030, ROCm 10.0.0, amdgpu 7.1.3,
Threadripper 3960X host). Each entry is one run: wrong iterations, then the
iterations the run did. A run stops at 10 wrong iterations or at 200000
iterations.

| Fix | `09-fastpath-read-during-kernel` | `09-cpu-stores-during-kernel` |
|---|---|---|
| none | 10 in 15235, 103152, 93513, 41065, 20021 | 1, 9, 6, 5, 0 in 200000 |
| `bar-read` | 0, 0 in 200000 | 0, 0 in 200000 |
| `vram-flag` | 0, 0 in 200000 | 0, 0 in 200000 |
| `wait-5us` | 0, 0 in 200000 | 3, 1 in 200000 |
| `wait-100us` | 0, 0 in 200000 | 0, 0 in 200000 |
| `uncached` | 10 in 43955, 36239 | 1, 4 in 200000 |
| `hdp-flush` | 10 in 179517; 4 in 200000 | 2, 2 in 200000 |

The fixes and the things that are not fixes are the same as on the RX 9070,
including the 5 us wait that cures the drive's data and not the CPU's. The
failure is far rarer here: about 1 iteration in 5000 with fastpath, against
10 in 13 on the RX 9070. The HDP flush made the fastpath failure about 9
times rarer and did not cure it. `09-fallback-read-during-kernel` was not run.

### A kernel that stays alive across transfers

Programs 10 and 11 are `hipfileread-gpu-watcher-visibility` reduced to its
loop. One kernel is launched once and keeps running while the host does
transfer after transfer into its buffer. Each work-item is a watcher with one
word of the buffer to read. The host puts new data in the buffer and then
stores a flag in host memory; a watcher's read after it has seen the flag
must return the new data. The kernel and the host's side of it are in
`long-running-kernel.h`, which both programs share.

Measured 2026-11-25 on the RX 6800 XT (gfx1030, ROCm 10.0.0, HIP 7.15,
amdgpu 7.1.3, Threadripper 3960X host) and on an MI300X (gfx942, ROCm 7.14,
amdgpu 6.19.14, EPYC 9654 host).

#### 10: the acquire a running kernel needs (expected)

`10-running-kernel-acquire-scope.cpp` asks what the watcher has to do between
the flag and its read. The fix chooses the acquire; without one the watcher
does none. It is built for each producer (`10-fastpath-read-…`,
`10-cpu-stores-…`, `10-fallback-read-seen-by-running-kernel`), and each of
those for fine-grained memory (no suffix), uncached memory (`-uncached`),
coarse-grained `hipMalloc` memory (`-coarse`), and fine-grained memory with a
16 MiB transfer in place of 1 MiB (`-16M`).

Wrong iterations of 20. Every cell is the same in four runs.

| GPU | Programs | none | `acquire-workgroup` | `acquire-agent` | `acquire-system` |
|---|---|---|---|---|---|
| RX 6800 XT | all three producers: fine-grained, `-uncached`, `-16M` | 20 | 0 | 0 | 0 |
| RX 6800 XT | `10-fastpath-read-…-coarse`, `10-cpu-stores-…-coarse` | 20 | 20 | 20 | 20 |
| RX 6800 XT | `10-fallback-read-…-coarse` | 20 | 20 | 0 | 0 |
| MI300X | all three producers, all three kinds of memory; fastpath and CPU stores `-16M` | 20 | 20 | 0 | 0 |
| MI300X | `10-fallback-read-…-16M` | 0 | 0 | 0 | 0 |

In every failing iteration all 1024 watchers read old data, and none of them
reads the new data on a later read with the same acquire. After the
system-scope acquire that starts the next iteration no watcher still reads
old data, except in the two RX 6800 XT rows where every acquire fails: there
all 1024 still do, and they go on reading the data from before the kernel
started.

What this shows:

- **A running kernel needs an acquire, and the scope it needs depends on the
  GPU.** This is HIP's memory model, not a defect: data from outside the GPU
  calls for a system-scope acquire, and that is enough everywhere HIP
  promises anything. That a workgroup-scope acquire is enough on the RX 6800
  XT is an accident of what it compiles to there:

  | Acquire | RX 6800 XT (gfx1030) | MI300X (gfx942) |
  |---|---|---|
  | workgroup | `buffer_gl0_inv` | nothing |
  | agent | `buffer_gl1_inv`, `buffer_gl0_inv` | `buffer_inv sc1` |
  | system | `buffer_gl1_inv`, `buffer_gl0_inv` | `buffer_inv sc0 sc1` |

- **Uncached memory does not replace the acquire.** It fails exactly as
  fine-grained memory does. It bypasses L2; the kernel keeps the old line in
  a cache nearer to it.
- **Fastpath, CPU stores, and fallback fail alike** without the acquire. The
  failure is in the kernel that reads, not in what wrote the data.
- **Coarse-grained memory cannot be made to work on the RX 6800 XT** for a
  fastpath read or CPU stores: no acquire in the kernel shows the data. This
  is cause B with a kernel that never ends: L2 holds the old lines, and only
  a queue packet invalidates L2. HIP does not promise this works. A fallback
  read is seen with an agent-scope acquire, since the GPU does the copy. On
  the MI300X coarse-grained memory behaves as fine-grained memory does.
- **Nothing outside the kernel can fix this.** hipFile cannot acquire for a
  kernel that is already running.

**The pass at 16 MiB on the MI300X is luck.** `10-fallback-read-…-16M` passes
there with no acquire at all. HIP pins the host buffer of a `hipMemcpy` of
more than 1 MiB, which is what a fallback read of that size does, and
`AMD_LOG_LEVEL=4` shows the pinning ("Locking to pool") in exactly the runs
that pass. The fallback programs take two more arguments that turn the
pinning on and off, with no acquire (wrong iterations of 20, four runs):

| Program | GPU | none | `pinned-copy` | `unpinned-copy` |
|---|---|---|---|---|
| `10-fallback-read-…` (1 MiB) | MI300X | 20 | 0 | 20 |
| `10-fallback-read-…-16M` | MI300X | 0 | 0 | 20 |
| `10-fallback-read-…` (1 MiB) | RX 6800 XT | 20 | 20 | 20 |
| `10-fallback-read-…-16M` | RX 6800 XT | 20 | 20 | 20 |

`pinned-copy` sets `GPU_PINNED_MIN_XFER_SIZE=0` and `unpinned-copy` sets it
to 64. On the MI300X the `-uncached` and `-coarse` programs give the first
row's results. We infer that pinning changes the process's GPU page tables,
and that the change drops what the running kernel has cached; we have not
looked for the mechanism. On the RX 6800 XT, where HIP 7.15 pins at 1 MiB
too, pinning does not rescue the read; `unpinned-copy` at 16 MiB leaves 63 of
the 1024 watchers stale in place of all of them. `HSA_ENABLE_SDMA=0`,
`GPU_FORCE_BLIT_COPY_SIZE=0` and `HSA_ENABLE_SDMA_HDP_FLUSH=0` do not change
the pass on the MI300X; `GPU_FORCE_BLIT_COPY_SIZE=1048576` turns it into a
failure in all 20 iterations.

**Some watchers pass without an acquire when their words are a power of two
apart.** The watchers' words are an odd number of words apart for that
reason. `-DSPACING_WORDS=n` sets the distance. Watchers of 1024 that read old
data with no acquire, on the RX 6800 XT, in three runs:

| Producer | Transfer | 255 or 4095 words apart (the default) | 256 or 4096 words apart |
|---|---|---|---|
| fastpath read | 1 MiB | 1024 | 856 to 889 |
| fastpath read | 16 MiB | 1024 | 977 to 990 |
| CPU stores | 1 MiB | 1024 | 1 to 42, in 8 to 13 of the 20 iterations |
| CPU stores | 16 MiB | 1024 | 944 to 966 |
| fallback read | 1 MiB | 1024 | 214 to 434 |
| fallback read | 16 MiB | 1024 | 940 to 955 |

On the MI300X the spacing changes nothing at 32 watchers in a workgroup. It
does at 256 (`-DWATCHERS_PER_WORKGROUP=256`) and 16 MiB: a fastpath read
leaves 7829 to 7917 of 8192 watchers stale at 511 words apart and 1655 to
1841 at 512. We infer that words a power of two apart push each other out of
a cache, and that a watcher whose line has left the cache reads VRAM again.
That would explain why the stress test, which spaces its watchers evenly,
often reports that most watchers failed and not all of them. The 1 ms the host sleeps between
the data and the flag changes no count: 0, 1 and 10 ms gave the same results,
so the old lines do not leave the caches with time.

#### 11: the data is late for a kernel that does everything right (not expected)

`11-running-kernel-sees-data-late.cpp` is the program HIP's rules ask for:
fine-grained memory and a system-scope acquire between the flag and the read.
On the RX 6800 XT a watcher still reads old data, and reads the new data on a
later read. This is cause D, the failure 09 shows. It is built four ways:

- `11-fastpath-read-late-for-running-kernel`: a fastpath `hipFileRead` of
  1 MiB, with one workgroup of 256 watchers, one for each 4 KiB page.
- `11-cpu-stores-late-for-running-kernel`: the same, with the CPU storing the
  1 MiB through the BAR. No hipFile and no drive.
- `11-cpu-stores-64-bytes-late-for-running-kernel`: the CPU stores only the
  first 64 bytes, and one watcher reads the first word. This is the smallest
  form: it fails in every iteration.
- `11-cpu-stores-4096-bytes-seen-by-running-kernel`: the same with 4096
  bytes, which passes.

RX 6800 XT, five runs for each fix. Each entry is one run: wrong iterations,
then the iterations the run did. A run stops at 10 wrong iterations or at
20000 iterations.

| Fix | fastpath read, 1 MiB | CPU stores, 1 MiB | CPU stores, 64 bytes | CPU stores, 4096 bytes |
|---|---|---|---|---|
| none | 10 in 1468, 4250, 2638, 5044, 4474 | 7, 8, 6, 8 in 20000; 10 in 12698 | 10 in 10, five times | 0 in 20000, five times |
| `bar-read` | 0 in 20000, five times | 0 in 20000, five times | 0, 1, 1, 0, 0 in 20000 | 0 in 20000, five times |
| `vram-flag` | 0 in 20000, five times | 0 in 20000, five times | 0 in 20000, five times | 0 in 20000, five times |
| `wait-5us` | 0 in 20000, five times | 2, 0, 3, 0, 1 in 20000 | 3, 2, 2, 5 in 20000; 10 in 19718 | 0 in 20000, five times |
| `wait-100us` | 0 in 20000, five times | 0 in 20000, five times | 0 in 20000, five times | 0 in 20000, five times |
| `uncached` | 10 in 5176, 6262, 3022, 2994, 2287 | 9, 5 in 20000; 10 in 13344, 11349, 19011 | 10 in 10, five times | 0 in 20000, five times |
| `hdp-flush` | 10 in 6340, 6788, 4096, 3351, 2540 | 7, 6, 8, 5, 9 in 20000 | 10 in 11, five times | 0, 0, 1, 1, 0 in 20000 |

MI300X: 0 wrong iterations in 20000 for all four programs and every fix, in
three runs each.

What a small store does on the RX 6800 XT, measured with a variant of the
64-byte program that runs a fixed 300000 iterations (percent of iterations in
which the watcher read old data):

| The CPU stores | Late |
|---|---|
| 4, 64, 1024, 2048 or 3584 bytes | 98.5% to 98.9% |
| 4096 bytes | 0 (1 iteration in 500000 in a longer run) |
| the whole 1 MiB | 0.07% |
| 64 bytes, then waits 1, 2 or 3 us before the flag | 98.6%, 98.4%, 97.6% |
| 64 bytes, then waits 4 us | 0.35% |
| 64 bytes, then waits 5 us | 0.02% to 0.03% |
| 64 bytes, then waits 10 us or more | 0 |
| 64 bytes, and 64 bytes into another VRAM buffer, before or after | 98.7% to 98.9% |
| 64 bytes, and 4096 bytes into another VRAM buffer, before or after | 0 |
| 64 bytes, then the HDP memory flush register (writing 1 or 0) | 98.8% to 98.9% |
| 64 bytes, then the other HDP flush register (`hdpRegFlushCntl`) | 98.7% |
| 64 bytes, then the HDP memory flush register and a 2 us wait | 98.6% |
| 64 bytes, then a read of the buffer, or of another VRAM buffer, through the BAR | 0.003% to 0.005% |
| 64 bytes, with or without an `sfence` after the store | 98.6%, 98.7% |

On the MI300X the same variant never reads old data: 0 of 300000 iterations
for each of 4, 64, 1024, 2048, 3584, 4096, 8192, 65536 bytes and 1 MiB. As a
check that it can fail there, storing the flag 50 us before the data is wrong
in every iteration. When the host stores the flag before the data on purpose,
the MI300X's data still reaches the kernel first until the flag's lead is 1.2
us, for 64 bytes and for 4096 bytes alike. On the RX 6800 XT a 4096-byte
store still wins in 92% of iterations with a lead of 0.5 us and loses at 1
us, and a 64-byte store loses whatever the lead.

What this shows:

- **On the RX 6800 XT something holds a small amount of written data back,
  as a write combiner with a timer would.** Less than 4 KiB stored through
  the BAR becomes readable to the kernel between 3 and 4 us later. It comes
  through at once when 4 KiB are written, to the same buffer or to any other
  VRAM, and a read through the BAR drains it. The MI300X has no such delay.
- **Where it is held is not known, and the HDP's write combiner is not shown
  to be it.** The amdgpu register headers give `HDP_HOST_PATH_CNTL` the
  fields `WRITE_COMBINE_EN` and `WRITE_COMBINE_TIMER`, and this card's HDP
  (version 5.0.3) has them at `WRITE_COMBINE_EN=1` and
  `WRITE_COMBINE_TIMER=1` (the register reads `0x00680000`). Written through
  the driver's debugfs register file, `WRITE_COMBINE_EN=0` and each of the
  four timer values read back as written and changed nothing: the 64-byte
  store stayed late in 98.4% to 99.0% of 100000 iterations, with the same
  step between a 3 us and a 4 us wait, and the 1 MiB fastpath read failed as
  before. So either something else holds the data, or these fields do not
  act when they are changed on a running card. The switch on the card and
  the host's root port are also on the path. It is not the CPU: an `sfence`
  changes nothing, and a fastpath read, where the CPU stores nothing, is
  late on the same time scale and cured the same way.
- **The HDP memory flush register does not drain it,** written with 1 or
  with 0, with or without a wait. It behaves exactly as a write to the other
  flush register does.
- **A read completion overtaking the writes does not explain it.** That was
  09's suspicion. It would give neither a step at 4096 bytes nor a store that
  arrives on time because 4 KiB went to a different buffer.
- **Why 1 MiB transfers fail rarely:** almost all of the data goes through at
  once, and only a small remainder at the end is held. Without a fix the
  1 MiB programs fail about 1 iteration in 220 with a fastpath read and 1 in
  1900 with CPU stores; `09-fastpath-read-during-kernel` fails about 1 in
  5000 on the same machine.
- **Which watcher is stale follows what is written last.** With a fastpath
  read it is always one watcher, of the last 4 KiB page of a 16 KiB piece of
  the transfer, between watchers 127 and 255. With CPU stores of 1 MiB it is
  watcher 0, the first word of the buffer; we infer that `memcpy` stores the
  head of a large copy last.
- **The drive matters for the fastpath rate.** With the test file on the
  machine's other NVMe drive the 1 MiB fastpath read fails 3 to 4 times as
  often.
- **Fixes:** `vram-flag` (0 late in 500000 iterations with the 64-byte
  store), a wait of 10 us or more, and `bar-read` (which leaves a few
  iterations in 100000 late with the 64-byte store). `wait-5us` cures the
  fastpath read and leaves CPU stores late a few times in 20000 iterations.
  `uncached` and `hdp-flush` are not fixes.
- **Not explained:** about 10 in 100000 CPU stores of 1 MiB are still late
  after a 5 us wait, though a lone 64-byte store is nearly always on time by
  then; they are gone at 10 us.
- **The kernel does not have to be long-running.** A variant of 09 whose
  work-items each poll the flag and read right after it failed about 5 times
  as often as 09 with a fastpath read (30 wrong iterations in 43167 against
  30 in 211900), and as often as 09 with CPU stores. It is not in this
  directory.

## The causes, and what fixes each

**A. A synchronous HIP call returns before its kernel has finished.**
`hipMemset` and its variants, and every `hipMemcpy` between GPU buffers, are
done by a kernel that HIP does not wait for. A fastpath transfer is not
ordered after that kernel, and neither is a fallback transfer when the kernel
runs on a non-blocking stream. Both GPUs. On gfx1030, cause B adds to it for
`hipMalloc` memory: once the kernel has finished, its data is still in L2.

With `AMD_LOG_LEVEL=4`, HIP logs the kernel (`__amd_rocclr_fillBufferAligned`
for `hipMemset`, `__amd_rocclr_copyBuffer` for the copy) and then the call's
return, with no wait in between. On MI300X, `hipMemset` of 1 GiB returns in
2.4 µs, and a `hipDeviceSynchronize` right after it takes 270 to 308 µs;
`hipMemcpy` from host memory takes 19 ms and the synchronize after it 3 µs.

- Fixes: `hipStreamSynchronize` on the kernel's stream or
  `hipDeviceSynchronize` before the transfer, or the asynchronous call
  followed by `hipStreamSynchronize`.
- Not a fix: flushing the HDP.

**B. The GPU's L2 cache and VRAM disagree.** `hipMalloc` memory is
coarse-grained, which the GPU caches in L2. A kernel's stores stay in L2
until a queue packet with a system-scope release writes them back, and L2
keeps what a kernel read until a packet with a system-scope acquire
invalidates it. `hipStreamSynchronize` enqueues both. A fastpath transfer
reads and writes VRAM directly, so it misses what is only in L2 (03) and
leaves L2 stale (06). gfx1030 only.

- Fixes: `hipStreamSynchronize` on the kernel's stream, before or after the
  transfer, or `hipDeviceSynchronize`; or uncached memory
  (`hipExtMallocWithFlags` with `hipDeviceMallocUncached`).
- Not a fix: flushing the HDP; sleeping (03 sleeps 50 ms before the write).
- By HIP's rules the application in 03 is at fault, because it never
  synchronizes with the kernel whose data it writes. 06 is less clear: its
  second kernel starts after `hipFileRead` has returned, and it does
  synchronize with that one. hipFile's documentation asks for neither, and the
  fallback backend needs neither.

**C. The HDP's read cache holds old data.** The CPU and the NVMe drive read
VRAM through the GPU's host data path (HDP), which caches about 16 KiB of
what it reads. When the GPU's SDMA engine writes VRAM, nothing invalidates
that cache, so the next read through the HDP of the same lines returns the
old data. A copy done by a kernel protects a later read by the drive, but not
a CPU that keeps reading during the copy (07); we infer that the kernel's
queue packet invalidates the cache before the kernel stores. gfx1030 only.
The applications in 04, 05, and 07 do nothing wrong: every call is
synchronous, and nothing overlaps in 04 and 05.

- Fixes: flushing the HDP after the copy (writing 1 to
  `hipDeviceProp_t::hdpMemFlushCntl`, which is what
  `hipDeviceFlushGPUDirectRDMAWrites` does in the HIP versions that have it);
  or running any kernel and waiting for it.
- For 04 and 05, `HSA_ENABLE_SDMA=0` or `GPU_FORCE_BLIT_COPY_SIZE=1048576` in
  the environment, which make a kernel do the copy, also make them pass. They
  do not help 07.
- Not a fix: `hipDeviceSynchronize` with nothing queued; writing
  `hdpRegFlushCntl` instead of `hdpMemFlushCntl`; a 5 ms sleep, which only
  makes the failure rarer. ROCr's own HDP flush before each SDMA copy makes
  no difference either way (`HSA_ENABLE_SDMA_HDP_FLUSH=0` changes nothing).

**D. New data reaches VRAM after the flag that announces it.** Only a kernel
that is running during the transfer sees this (09 and 11). The kernel learns
that the data is in the buffer by reading a flag in host memory. Some of what
was written into VRAM before the host stored the flag, by the drive or by the
CPU through the BAR, becomes readable to the kernel a few microseconds after
the kernel has read the flag. The data is late, not cached: the kernel
acquires at system scope before its read, and a later read with nothing else
changed returns the new data. RX 9070 in a Threadripper PRO 7955WX host, and
RX 6800 XT in a Threadripper 3960X host. Not seen on MI300X (EPYC 9654 host),
on MI100 with CPU stores (Xeon Gold 6132 host), or on RX 9070 in a
Threadripper PRO 5975WX host.

- On the RX 6800 XT the cause is measured: less than 4 KiB written into VRAM
  from outside the GPU is held for 3 to 4 us, unless more writes or a read
  through the BAR push it out (see 11 above). It behaves as a write combiner
  with a timer would. What holds the data is not known: neither the HDP's
  flush register nor its write-combine enable and timer fields change it.
- On the RX 9070 this has not been measured. 09's source suspects the PCIe
  ordering between the kernel's read of host memory and the writes into
  VRAM. On the RX 6800 XT that suspicion does not fit what 11 shows, and
  Relaxed Ordering is enabled on that GPU and on every port above it, so
  PCIe would allow such reordering there. The 64-byte program has to run on
  the RX 9070 before either explanation is taken for that GPU.
- Fixes: the flag is in VRAM, stored through the BAR after the data; or the
  CPU reads one word of the buffer through the BAR before it stores the
  flag; or the host waits before the flag (10 us was enough on the RX 6800
  XT; 5 us was not for the CPU's data, on either GPU).
- Not a fix: uncached memory; writing the HDP memory flush register.

## Where hipFile could apply the fixes

The fixes are written as the application would apply them. hipFile could
apply the same ones itself:

- For A and B, `hipDeviceSynchronize` at the start of a fastpath write and at
  the end of a fastpath read, and at the start of a fallback write for the
  `hipMemcpyWithStream` case. That is a HIP call that synchronizes, on the I/O
  path.
- For C, an HDP flush at the start of a fastpath write and at the end of a
  fallback read. ROCr could instead flush after each copy into VRAM.
- For D, a read of one word of the buffer through the BAR at the end of a
  fastpath read, where the CPU can map the buffer. That is `bar-read`, done
  by hipFile: 0 wrong iterations in 200000 fastpath reads of 1 MiB on the RX
  6800 XT, reading the buffer or any other VRAM. hipFile has no part in the
  CPU-stores form of D. Turning the HDP's write combiner off, or changing
  its timer, on the running card does not help.
- For the acquire a running kernel needs (10), nothing: only the kernel can
  acquire. hipFile's documentation could say that a kernel that is running
  during a `hipFileRead` must do a system-scope acquire before it reads the
  buffer, and that the buffer must not be coarse-grained (`hipMalloc`).

None of these has been tried inside hipFile.

## What is not here

- A kernel that is still running during a `hipFileWrite`, with the buffer in
  coarse-grained memory (`hipfilewrite-visibility --kernel during --memory
  coarse`). HIP does not promise that works; the test uses fine-grained
  memory for it. For `hipFileRead`, the `10-*-coarse` programs show it.
- The failures that need other threads writing the same buffer to the file
  (`hipfilewrite-visibility --background-writers`). They are cause C, and the
  HDP flush cures them too.

## The HDP flush register

`repro-common.h` writes the register and never reads it. Reading it back has
caused PCIe errors on some systems (drm/amd issues 3908 and 4119).
