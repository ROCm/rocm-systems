# Minimal reproducers for the hipFile data-visibility failures

Each program here is the smallest form of one class of failure that
`hipfilewrite-visibility`, `hipfileread-gpu-visibility`, and
`hipfileread-cpu-visibility` report. Each is one short loop: store new data,
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

## The three causes, and what fixes each

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

## Where hipFile could apply the fixes

The fixes are written as the application would apply them. hipFile could
apply the same ones itself:

- For A and B, `hipDeviceSynchronize` at the start of a fastpath write and at
  the end of a fastpath read, and at the start of a fallback write for the
  `hipMemcpyWithStream` case. That is a HIP call that synchronizes, on the I/O
  path.
- For C, an HDP flush at the start of a fastpath write and at the end of a
  fallback read. ROCr could instead flush after each copy into VRAM.

None of these has been tried inside hipFile.

## What is not here

- A kernel that is still running during the transfer, with the buffer in
  coarse-grained memory (`--kernel during --memory coarse` in both tests).
  HIP does not promise that works; the tests use fine-grained memory for it.
- The failures that need other threads writing the same buffer to the file
  (`hipfilewrite-visibility --background-writers`). They are cause C, and the
  HDP flush cures them too.
- A CPU that stores to a buffer a kernel has just stored to, without a
  synchronization in between (`hipfilewrite-visibility --producer cpu --prime
  store --wait flag` with fallback). It is cause B, and corrupts the buffer
  itself: L2 is written back over the CPU's data.

## The HDP flush register

`repro-common.h` writes the register and never reads it. Reading it back has
caused PCIe errors on some systems (drm/amd issues 3908 and 4119).
