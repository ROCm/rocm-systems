/*
Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
*/

/*
 * One-way LL bandwidth/latency test, single node, 2 GPUs, HIP/AMD.
 *
 * Measures time(GPU0 -> GPU1) for `data_bytes` payloads delivered via four
 * NCCL-style LL flag/data signaling protocols. The pong is a single 8-byte
 * ACK per iteration, which serves only as backpressure: GPU0 can't start
 * iteration k+1 until every block on GPU1 has finished iteration k. Without
 * this, GPU0 races ahead and overwrites flag k with flag k+1 before GPU1
 * polls k.
 *
 *   LL       16-B line, 4 B data + 4 B flag twice. Receiver waits for both
 *            flag halves to match. 50% wire efficiency.
 *
 *   LL<N>-sc N-byte line, last 8B is the flag. Sender's flag lane does an
 *            s_waitcnt vmcnt(0) to drain the wavefront's data stores before
 *            writing the flag. Receiver spins via __any() ballot on the
 *            flag lane and re-reads data once it's seen.
 *
 *   LL<N>-sc+wb  Like sc but adds __threadfence_system() between the s_waitcnt
 *                and the flag store (extra cache writeback per line).
 *
 *   LL<N>-tf Same line layout, but uses one __threadfence_system() per
 *            iteration instead of per line. Send is two passes: data, fence,
 *            flags. Recv polls flags then re-reads data.
 *
 * Set the line size at compile time with -DLINE_BYTES=64 (default), 128, or 256.
 * Override the per-block thread count with -DNTHREADS=<N> (default 256;
 * must be a multiple of LINE_BYTES/8).
 *
 * Each run does warmup, then a 1-iter verify pass (with checks), then timed
 * iters. Per-iter timing = (kernel time / iters), where the per-iter ACK
 * adds ~1 µs of overhead independent of message size.
 *
 * Build:
 *   hipcc -O3 --offload-arch=<arch> -std=c++17 -x hip llx_latency_bench.c -o llx_latency_bench
 *
 * Run:  ./llx_latency_bench_<N> [warmup [iters [timeout_s [max_bytes]]]]
 *        max_bytes accepts K/M/G suffixes, e.g. 16M, 1G (default: 128M)
 */

#include <hip/hip_runtime.h>
#include <cstdio>
#include <cstdint>
#include <cstdlib>
#include <cerrno>
#include <climits>
#include <atomic>
#include <thread>
#include <chrono>

// ---------------------------------------------------------------------------
// Compile-time line geometry
// ---------------------------------------------------------------------------

#ifndef LINE_BYTES
#define LINE_BYTES 64
#endif

#define _STR(x) #x
#define STR(x)  _STR(x)

#define LINE_ELEMS (LINE_BYTES / 8)
#define DATA_ELEMS (LINE_ELEMS - 1)
#define DATA_BYTES (DATA_ELEMS * 8)

static_assert(LINE_BYTES % 8 == 0,         "LINE_BYTES must be a multiple of 8");
static_assert(LINE_BYTES >= 16,            "LINE_BYTES must be at least 16");
// SC variants use s_waitcnt vmcnt(0) and __any() to order/poll within a single
// wavefront. LINE_ELEMS must fit in one wave; cap at 32 for wave32 portability.
static_assert(LINE_ELEMS <= 32,            "LINE_ELEMS must fit in one wavefront (LINE_BYTES <= 256)");

#define PROTO_SC    "LL" STR(LINE_BYTES) "-sc"
#define PROTO_SC_WB "LL" STR(LINE_BYTES) "-sc+wb"
#define PROTO_TF    "LL" STR(LINE_BYTES) "-tf"

// ---------------------------------------------------------------------------
// Tuning constants
// ---------------------------------------------------------------------------

#ifndef NTHREADS
#define NTHREADS    256
#endif
#define NBLOCKS_MAX  64
#define ITERS_MAX  4096
#define MIN_SWEEP_BYTES 8

static_assert(NTHREADS > 0,                "NTHREADS must be positive");
static_assert(NTHREADS % LINE_ELEMS == 0,  "NTHREADS must be divisible by LINE_ELEMS");

// ---------------------------------------------------------------------------
// Watchdog
// ---------------------------------------------------------------------------

static std::atomic<bool> g_watchdog_active{false};
static double            g_watchdog_timeout_s = 30.0;
static std::thread       g_watchdog_thread;

static void watchdog_start()
{
  g_watchdog_active = true;
  g_watchdog_thread = std::thread([]() {
    auto deadline = std::chrono::steady_clock::now() +
        std::chrono::milliseconds((long long)(g_watchdog_timeout_s * 1000));
    while (g_watchdog_active.load()) {
      if (std::chrono::steady_clock::now() >= deadline) {
        fprintf(stderr, "\nWATCHDOG: kernel hung >%.0fs — resetting GPUs\n",
                g_watchdog_timeout_s);
        fflush(stderr);
        hipSetDevice(0); hipDeviceReset();
        hipSetDevice(1); hipDeviceReset();
        exit(1);
      }
      std::this_thread::sleep_for(std::chrono::milliseconds(200));
    }
  });
}

static void watchdog_stop()
{
  g_watchdog_active = false;
  if (g_watchdog_thread.joinable()) g_watchdog_thread.join();
}

// ---------------------------------------------------------------------------
// HIP error check
// ---------------------------------------------------------------------------

#define HIP_CHECK(x) \
  do { hipError_t _e = (x); \
       if (_e != hipSuccess) { \
         fprintf(stderr, "HIP error %s at %s:%d\n", \
                 hipGetErrorString(_e), __FILE__, __LINE__); \
         exit(1); } } while (0)

// System-scope 64-bit load/store. Bypasses caches; the peer GPU sees stores
// without explicit cache flush.

typedef uint64_t __attribute__((address_space(1)))* gptr64;
typedef const uint64_t __attribute__((address_space(1)))* cgptr64;

__device__ __forceinline__
uint64_t sys_load64(cgptr64 p)
{ return __hip_atomic_load(p, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_SYSTEM); }

__device__ __forceinline__
void sys_store64(gptr64 p, uint64_t v)
{ __hip_atomic_store(p, v, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_SYSTEM); }

// Payload hash: nonzero for any (grp, lane, flag) where flag >= 1 — keeps a
// freshly-zeroed buffer from passing verify by accident.

__device__ __forceinline__
uint64_t make_payload(int grp, int lane, uint64_t flag)
{
  return flag  * 0x9e3779b97f4a7c15ULL
       + (uint64_t)grp  * 0x517cc1b727220a95ULL
       + (uint64_t)lane * 0xbf58476d1ce4e5b9ULL;
}

// ---------------------------------------------------------------------------
// LL (baseline) — 16-byte lines
// ---------------------------------------------------------------------------

union LLLine { struct { uint32_t d1, f1, d2, f2; }; uint64_t v[2]; };

__device__ __forceinline__
void storeLL(LLLine* dst, uint64_t val, uint32_t flag)
{
  sys_store64((gptr64)(dst->v),   (uint64_t)(uint32_t)val          | ((uint64_t)flag << 32));
  sys_store64((gptr64)(dst->v+1), (uint64_t)(uint32_t)(val >> 32)  | ((uint64_t)flag << 32));
}

__device__ __forceinline__
uint64_t readLL(const LLLine* src, uint32_t flag)
{
  uint64_t lo, hi;
  do {
    lo = sys_load64((cgptr64)src->v);
    hi = sys_load64((cgptr64)(src->v+1));
  } while ((uint32_t)(lo>>32) != flag || (uint32_t)(hi>>32) != flag);
  return (uint64_t)(uint32_t)lo | ((uint64_t)(uint32_t)hi << 32);
}

// LL<N>-sc helpers.

__device__ __forceinline__
void storeLL_sc(uint64_t* base, int lane, uint64_t data, uint64_t flag)
{
  if (lane < DATA_ELEMS) {
    sys_store64((gptr64)(base + lane), data);
  } else {
    asm volatile("s_waitcnt vmcnt(0)" ::: "memory");
    sys_store64((gptr64)(base + DATA_ELEMS), flag);
  }
}

// Per-line fence: each call only has to drain a wavefront's-worth of stores,
// so the cost is small. Compare to tf below, where one big fence per iter sits
// in series with the entire data pass — that drain dominates at large sizes.
__device__ __forceinline__
void storeLL_sc_wb(uint64_t* base, int lane, uint64_t data, uint64_t flag)
{
  if (lane < DATA_ELEMS) {
    sys_store64((gptr64)(base + lane), data);
  } else {
    asm volatile("s_waitcnt vmcnt(0)" ::: "memory");
    __threadfence_system();
    sys_store64((gptr64)(base + DATA_ELEMS), flag);
  }
}

__device__ __forceinline__
uint64_t readLL_sc(const uint64_t* base, int lane, uint64_t flag)
{
  bool is_flag = (lane == DATA_ELEMS);
  uint64_t val = 0;
  do {
    val = sys_load64((cgptr64)(base + lane));
  } while (__any(is_flag && (val != flag)));
  if (!is_flag) val = sys_load64((cgptr64)(base + lane));
  return val;
}

// ---------------------------------------------------------------------------
// One-way kernels with per-iteration ACK.
//
// The ACK is the minimum sync needed to keep GPU0 from racing ahead and
// overwriting flags before GPU1 sees them. Per iter:
//   sender: write data + flag, then poll ack_buf for this iter's value
//   recv:   poll flag, then atomicAdd into block_counters[it]; the last block
//           to do so writes ack_buf with this iter's expected value
//
// GPU0 and GPU1 advance in lockstep, one iter at a time. Per-iter cost is
// T(A->B) + T(8-byte B->A) ≈ T(A->B) + ~1 µs.
// ---------------------------------------------------------------------------

__device__ __forceinline__
void wait_iter_ack(uint64_t* ack_buf, uint64_t target)
{
  if (threadIdx.x == 0) {
    uint64_t v;
    do { v = sys_load64((cgptr64)ack_buf);
    } while (v != target);
  }
  __syncthreads();
}

__device__ __forceinline__
void signal_iter_done(int* block_counters_it, uint64_t* ack_buf, uint64_t target)
{
  if (threadIdx.x == 0) {
    int last = atomicAdd(block_counters_it, 1);
    if (last == gridDim.x - 1) {
      sys_store64((gptr64)ack_buf, target);
    }
  }
  // No __syncthreads: each block independently proceeds to its next iter.
}

// ── LL baseline ─────────────────────────────────────────────────────────────

__global__
void send_ll_kernel(LLLine* ping, uint64_t* ack_buf,
                    int nlines, int iters, int flag_start)
{
  const int gtid = blockIdx.x * blockDim.x + threadIdx.x;
  const int tot  = gridDim.x  * blockDim.x;
  for (int it = 0; it < iters; ++it) {
    uint32_t flag = (uint32_t)(flag_start + it + 1);
    for (int i = gtid; i < nlines; i += tot)
      storeLL(ping + i, make_payload(i, 0, (uint64_t)flag), flag);
    __syncthreads();
    wait_iter_ack(ack_buf, (uint64_t)flag);
  }
}

__global__
void recv_ll_kernel(LLLine* ping, uint64_t* ack_buf, int* block_counters,
                    int nlines, int iters, int flag_start, int* nerrors)
{
  const int gtid = blockIdx.x * blockDim.x + threadIdx.x;
  const int tot  = gridDim.x  * blockDim.x;
  int lerr = 0;
  for (int it = 0; it < iters; ++it) {
    uint32_t flag = (uint32_t)(flag_start + it + 1);
    for (int i = gtid; i < nlines; i += tot) {
      uint64_t v = readLL(ping + i, flag);
      if (nerrors && v != make_payload(i, 0, (uint64_t)flag)) lerr++;
    }
    __syncthreads();
    signal_iter_done(&block_counters[it], ack_buf, (uint64_t)flag);
  }
  if (nerrors && lerr) atomicAdd(nerrors, lerr);
}

// ── LL<N>-sc ────────────────────────────────────────────────────────────────

__global__
void send_ll_sc_kernel(uint64_t* ping, uint64_t* ack_buf,
                       int nlines, int iters, int flag_start)
{
  const int gtid = blockIdx.x * blockDim.x + threadIdx.x;
  const int tot  = gridDim.x  * blockDim.x;
  const int lane = gtid % LINE_ELEMS;
  for (int it = 0; it < iters; ++it) {
    uint64_t flag = (uint64_t)(flag_start + it + 1);
    for (int g = gtid / LINE_ELEMS; g < nlines; g += tot / LINE_ELEMS)
      storeLL_sc(ping + g * LINE_ELEMS, lane, make_payload(g, lane, flag), flag);
    __syncthreads();
    wait_iter_ack(ack_buf, flag);
  }
}

__global__
void recv_ll_sc_kernel(uint64_t* ping, uint64_t* ack_buf, int* block_counters,
                       int nlines, int iters, int flag_start, int* nerrors)
{
  const int gtid = blockIdx.x * blockDim.x + threadIdx.x;
  const int tot  = gridDim.x  * blockDim.x;
  const int lane = gtid % LINE_ELEMS;
  int lerr = 0;
  for (int it = 0; it < iters; ++it) {
    uint64_t flag = (uint64_t)(flag_start + it + 1);
    for (int g = gtid / LINE_ELEMS; g < nlines; g += tot / LINE_ELEMS) {
      uint64_t v = readLL_sc(ping + g * LINE_ELEMS, lane, flag);
      if (nerrors && lane < DATA_ELEMS && v != make_payload(g, lane, flag)) lerr++;
    }
    __syncthreads();
    signal_iter_done(&block_counters[it], ack_buf, flag);
  }
  if (nerrors && lerr) atomicAdd(nerrors, lerr);
}

// ── LL<N>-sc+wb ─────────────────────────────────────────────────────────────

__global__
void send_ll_sc_wb_kernel(uint64_t* ping, uint64_t* ack_buf,
                          int nlines, int iters, int flag_start)
{
  const int gtid = blockIdx.x * blockDim.x + threadIdx.x;
  const int tot  = gridDim.x  * blockDim.x;
  const int lane = gtid % LINE_ELEMS;
  for (int it = 0; it < iters; ++it) {
    uint64_t flag = (uint64_t)(flag_start + it + 1);
    for (int g = gtid / LINE_ELEMS; g < nlines; g += tot / LINE_ELEMS)
      storeLL_sc_wb(ping + g * LINE_ELEMS, lane, make_payload(g, lane, flag), flag);
    __syncthreads();
    wait_iter_ack(ack_buf, flag);
  }
}

__global__
void recv_ll_sc_wb_kernel(uint64_t* ping, uint64_t* ack_buf, int* block_counters,
                          int nlines, int iters, int flag_start, int* nerrors)
{
  const int gtid = blockIdx.x * blockDim.x + threadIdx.x;
  const int tot  = gridDim.x  * blockDim.x;
  const int lane = gtid % LINE_ELEMS;
  int lerr = 0;
  for (int it = 0; it < iters; ++it) {
    uint64_t flag = (uint64_t)(flag_start + it + 1);
    for (int g = gtid / LINE_ELEMS; g < nlines; g += tot / LINE_ELEMS) {
      uint64_t v = readLL_sc(ping + g * LINE_ELEMS, lane, flag);
      if (nerrors && lane < DATA_ELEMS && v != make_payload(g, lane, flag)) lerr++;
    }
    __syncthreads();
    signal_iter_done(&block_counters[it], ack_buf, flag);
  }
  if (nerrors && lerr) atomicAdd(nerrors, lerr);
}

// ── LL<N>-tf ────────────────────────────────────────────────────────────────

__global__
void send_ll_tf_kernel(uint64_t* ping, uint64_t* ack_buf,
                       int nlines, int iters, int flag_start)
{
  const int gtid = blockIdx.x * blockDim.x + threadIdx.x;
  const int tot  = gridDim.x  * blockDim.x;
  const int lane = gtid % LINE_ELEMS;
  for (int it = 0; it < iters; ++it) {
    uint64_t flag = (uint64_t)(flag_start + it + 1);

    // Pass 1: data
    for (int g = gtid / LINE_ELEMS; g < nlines; g += tot / LINE_ELEMS)
      if (lane < DATA_ELEMS)
        sys_store64((gptr64)(ping + g * LINE_ELEMS + lane), make_payload(g, lane, flag));
    __threadfence_system(); __syncthreads();

    // Pass 2: flag
    if (lane == DATA_ELEMS) {
      for (int g = gtid / LINE_ELEMS; g < nlines; g += tot / LINE_ELEMS)
        sys_store64((gptr64)(ping + g * LINE_ELEMS + DATA_ELEMS), flag);
    }
    __syncthreads();
    wait_iter_ack(ack_buf, flag);
  }
}

__global__
void recv_ll_tf_kernel(uint64_t* ping, uint64_t* ack_buf, int* block_counters,
                       int nlines, int iters, int flag_start, int* nerrors)
{
  const int gtid = blockIdx.x * blockDim.x + threadIdx.x;
  const int tot  = gridDim.x  * blockDim.x;
  const int lane = gtid % LINE_ELEMS;
  int lerr = 0;
  for (int it = 0; it < iters; ++it) {
    uint64_t flag = (uint64_t)(flag_start + it + 1);

    if (lane == DATA_ELEMS) {
      for (int g = gtid / LINE_ELEMS; g < nlines; g += tot / LINE_ELEMS) {
        uint64_t v;
        do { v = sys_load64((cgptr64)(ping + g * LINE_ELEMS + DATA_ELEMS));
        } while (v != flag);
      }
    }
    __syncthreads();

    for (int g = gtid / LINE_ELEMS; g < nlines; g += tot / LINE_ELEMS) {
      if (lane < DATA_ELEMS) {
        uint64_t v = sys_load64((cgptr64)(ping + g * LINE_ELEMS + lane));
        if (nerrors && v != make_payload(g, lane, flag)) lerr++;
      }
    }
    __syncthreads();
    signal_iter_done(&block_counters[it], ack_buf, flag);
  }
  if (nerrors && lerr) atomicAdd(nerrors, lerr);
}

// ---------------------------------------------------------------------------
// Host helpers
// ---------------------------------------------------------------------------

static inline int ll_nblocks(int nlines)
{
  int nb = nlines / NTHREADS;
  return (nb < 1) ? 1 : (nb > NBLOCKS_MAX) ? NBLOCKS_MAX : nb;
}

static inline int lln_nblocks(int nlines)
{
  int nb = nlines / (NTHREADS / LINE_ELEMS);
  return (nb < 1) ? 1 : (nb > NBLOCKS_MAX) ? NBLOCKS_MAX : nb;
}

static void print_result(const char* proto, size_t data_bytes,
                         int nlines, int nblocks,
                         double t_us, int nerr)
{
  const char* tag = (nerr < 0) ? "[VERIFY HANG]"
                  : (nerr == 0) ? "[OK]" : nullptr;
  if (tag)
    printf("%-12s %10zu B  nlines=%8d  nb=%2d  T(A->B)=%9.2f us  BW=%7.3f GB/s  %s\n",
           proto, data_bytes, nlines, nblocks, t_us,
           (double)data_bytes / (t_us * 1e3), tag);
  else
    printf("%-12s %10zu B  nlines=%8d  nb=%2d  T(A->B)=%9.2f us  BW=%7.3f GB/s  [ERRORS: %d]\n",
           proto, data_bytes, nlines, nblocks, t_us,
           (double)data_bytes / (t_us * 1e3), nerr);
  fflush(stdout);
}

typedef void (*send_ll_fn) (LLLine*,   uint64_t*,        int,int,int);
typedef void (*recv_ll_fn) (LLLine*,   uint64_t*, int*,  int,int,int,int*);
typedef void (*send_lln_fn)(uint64_t*, uint64_t*,        int,int,int);
typedef void (*recv_lln_fn)(uint64_t*, uint64_t*, int*,  int,int,int,int*);

// Reset ack_buf and block_counters[0..iters-1]. Must happen before each launch.
static void reset_sync(uint64_t* ack_buf, int* block_counters, int iters)
{
  hipSetDevice(0); hipMemset(ack_buf, 0, sizeof(uint64_t));
  hipSetDevice(1); hipMemset(block_counters, 0, sizeof(int) * iters);
}

static int verify_ll(int nb, int nlines, int fs,
                     send_ll_fn ks, recv_ll_fn kr,
                     LLLine* ping, uint64_t* ack_buf, int* block_counters)
{
  int* e1;
  hipSetDevice(1);
  HIP_CHECK(hipMalloc(&e1, sizeof(int)));
  hipMemset(e1, 0, sizeof(int));
  reset_sync(ack_buf, block_counters, 1);

  watchdog_start();
  hipSetDevice(1); hipLaunchKernelGGL(kr, nb, NTHREADS, 0, 0, ping, ack_buf, block_counters, nlines, 1, fs, e1);
  hipSetDevice(0); hipLaunchKernelGGL(ks, nb, NTHREADS, 0, 0, ping, ack_buf, nlines, 1, fs);
  hipSetDevice(0); hipError_t r0 = hipDeviceSynchronize();
  hipSetDevice(1); hipError_t r1 = hipDeviceSynchronize();
  watchdog_stop();
  int h1 = 0, ret = -1;
  if (r0 == hipSuccess && r1 == hipSuccess) {
    hipSetDevice(1); hipMemcpy(&h1, e1, sizeof(int), hipMemcpyDeviceToHost);
    ret = h1;
  }
  hipSetDevice(1); hipFree(e1);
  return ret;
}

static int verify_lln(int nb, int nlines, int fs,
                      send_lln_fn ks, recv_lln_fn kr,
                      uint64_t* ping, uint64_t* ack_buf, int* block_counters)
{
  int* e1;
  hipSetDevice(1);
  HIP_CHECK(hipMalloc(&e1, sizeof(int)));
  hipMemset(e1, 0, sizeof(int));
  reset_sync(ack_buf, block_counters, 1);

  watchdog_start();
  hipSetDevice(1); hipLaunchKernelGGL(kr, nb, NTHREADS, 0, 0, ping, ack_buf, block_counters, nlines, 1, fs, e1);
  hipSetDevice(0); hipLaunchKernelGGL(ks, nb, NTHREADS, 0, 0, ping, ack_buf, nlines, 1, fs);
  hipSetDevice(0); hipError_t r0 = hipDeviceSynchronize();
  hipSetDevice(1); hipError_t r1 = hipDeviceSynchronize();
  watchdog_stop();
  int h1 = 0, ret = -1;
  if (r0 == hipSuccess && r1 == hipSuccess) {
    hipSetDevice(1); hipMemcpy(&h1, e1, sizeof(int), hipMemcpyDeviceToHost);
    ret = h1;
  }
  hipSetDevice(1); hipFree(e1);
  return ret;
}

static double time_one_way_ll(int nb, int nlines, int fs, int iters,
                              send_ll_fn ks, recv_ll_fn kr,
                              LLLine* ping, uint64_t* ack_buf, int* block_counters)
{
  reset_sync(ack_buf, block_counters, iters);
  hipEvent_t ev0, ev1;
  hipSetDevice(0); hipEventCreate(&ev0); hipEventCreate(&ev1);
  watchdog_start();
  hipSetDevice(1); hipLaunchKernelGGL(kr, nb, NTHREADS, 0, 0, ping, ack_buf, block_counters, nlines, iters, fs, nullptr);
  hipSetDevice(0);
  hipEventRecord(ev0);
  hipLaunchKernelGGL(ks, nb, NTHREADS, 0, 0, ping, ack_buf, nlines, iters, fs);
  hipEventRecord(ev1);
  hipError_t e0 = hipDeviceSynchronize();
  hipSetDevice(1); hipError_t e1 = hipDeviceSynchronize();
  watchdog_stop();
  if (e0 != hipSuccess || e1 != hipSuccess) {
    hipEventDestroy(ev0); hipEventDestroy(ev1); return -1;
  }
  float ms = 0; hipEventElapsedTime(&ms, ev0, ev1);
  hipEventDestroy(ev0); hipEventDestroy(ev1);
  return (double)ms * 1e3 / iters;
}

static double time_one_way_lln(int nb, int nlines, int fs, int iters,
                               send_lln_fn ks, recv_lln_fn kr,
                               uint64_t* ping, uint64_t* ack_buf, int* block_counters)
{
  reset_sync(ack_buf, block_counters, iters);
  hipEvent_t ev0, ev1;
  hipSetDevice(0); hipEventCreate(&ev0); hipEventCreate(&ev1);
  watchdog_start();
  hipSetDevice(1); hipLaunchKernelGGL(kr, nb, NTHREADS, 0, 0, ping, ack_buf, block_counters, nlines, iters, fs, nullptr);
  hipSetDevice(0);
  hipEventRecord(ev0);
  hipLaunchKernelGGL(ks, nb, NTHREADS, 0, 0, ping, ack_buf, nlines, iters, fs);
  hipEventRecord(ev1);
  hipError_t e0 = hipDeviceSynchronize();
  hipSetDevice(1); hipError_t e1 = hipDeviceSynchronize();
  watchdog_stop();
  if (e0 != hipSuccess || e1 != hipSuccess) {
    hipEventDestroy(ev0); hipEventDestroy(ev1); return -1;
  }
  float ms = 0; hipEventElapsedTime(&ms, ev0, ev1);
  hipEventDestroy(ev0); hipEventDestroy(ev1);
  return (double)ms * 1e3 / iters;
}

static void run_ll(int warmup, int iters, size_t data_bytes,
                   void* g0ping, uint64_t* ack_buf, int* block_counters)
{
  int nlines = (int)((data_bytes + 7) / 8); if (nlines < 1) nlines = 1;
  int nb     = ll_nblocks(nlines);
  auto* ping = (LLLine*)g0ping;

  if (time_one_way_ll(nb, nlines, 0, warmup, send_ll_kernel, recv_ll_kernel,
                      ping, ack_buf, block_counters) < 0) {
    printf("LL           %10zu B  WARMUP HANG\n", data_bytes); return;
  }
  int nerr = verify_ll(nb, nlines, warmup, send_ll_kernel, recv_ll_kernel,
                       ping, ack_buf, block_counters);
  double t = time_one_way_ll(nb, nlines, warmup + 1, iters,
                             send_ll_kernel, recv_ll_kernel,
                             ping, ack_buf, block_counters);
  if (t < 0) { printf("LL           %10zu B  TIMED HANG\n", data_bytes); return; }
  print_result("LL", data_bytes, nlines, nb, t, nerr);
}

static void run_lln_sc(int warmup, int iters, size_t data_bytes,
                       void* g0ping, uint64_t* ack_buf, int* block_counters)
{
  int nlines = (int)((data_bytes + DATA_BYTES - 1) / DATA_BYTES);
  if (nlines < 1) nlines = 1;
  int nb = lln_nblocks(nlines);
  auto* ping = (uint64_t*)g0ping;

  if (time_one_way_lln(nb, nlines, 0, warmup, send_ll_sc_kernel, recv_ll_sc_kernel,
                       ping, ack_buf, block_counters) < 0) {
    printf(PROTO_SC "    %10zu B  WARMUP HANG\n", data_bytes); return;
  }
  int nerr = verify_lln(nb, nlines, warmup, send_ll_sc_kernel, recv_ll_sc_kernel,
                        ping, ack_buf, block_counters);
  double t = time_one_way_lln(nb, nlines, warmup + 1, iters,
                              send_ll_sc_kernel, recv_ll_sc_kernel,
                              ping, ack_buf, block_counters);
  if (t < 0) { printf(PROTO_SC "    %10zu B  TIMED HANG\n", data_bytes); return; }
  print_result(PROTO_SC, data_bytes, nlines, nb, t, nerr);
}

static void run_lln_sc_wb(int warmup, int iters, size_t data_bytes,
                          void* g0ping, uint64_t* ack_buf, int* block_counters)
{
  int nlines = (int)((data_bytes + DATA_BYTES - 1) / DATA_BYTES);
  if (nlines < 1) nlines = 1;
  int nb = lln_nblocks(nlines);
  auto* ping = (uint64_t*)g0ping;

  if (time_one_way_lln(nb, nlines, 0, warmup, send_ll_sc_wb_kernel, recv_ll_sc_wb_kernel,
                       ping, ack_buf, block_counters) < 0) {
    printf(PROTO_SC_WB " %10zu B  WARMUP HANG\n", data_bytes); return;
  }
  int nerr = verify_lln(nb, nlines, warmup, send_ll_sc_wb_kernel, recv_ll_sc_wb_kernel,
                        ping, ack_buf, block_counters);
  double t = time_one_way_lln(nb, nlines, warmup + 1, iters,
                              send_ll_sc_wb_kernel, recv_ll_sc_wb_kernel,
                              ping, ack_buf, block_counters);
  if (t < 0) { printf(PROTO_SC_WB " %10zu B  TIMED HANG\n", data_bytes); return; }
  print_result(PROTO_SC_WB, data_bytes, nlines, nb, t, nerr);
}

static void run_lln_tf(int warmup, int iters, size_t data_bytes,
                       void* g0ping, uint64_t* ack_buf, int* block_counters)
{
  int nlines = (int)((data_bytes + DATA_BYTES - 1) / DATA_BYTES);
  if (nlines < 1) nlines = 1;
  int nb = lln_nblocks(nlines);
  auto* ping = (uint64_t*)g0ping;

  if (time_one_way_lln(nb, nlines, 0, warmup, send_ll_tf_kernel, recv_ll_tf_kernel,
                       ping, ack_buf, block_counters) < 0) {
    printf(PROTO_TF "    %10zu B  WARMUP HANG\n", data_bytes); return;
  }
  int nerr = verify_lln(nb, nlines, warmup, send_ll_tf_kernel, recv_ll_tf_kernel,
                        ping, ack_buf, block_counters);
  double t = time_one_way_lln(nb, nlines, warmup + 1, iters,
                              send_ll_tf_kernel, recv_ll_tf_kernel,
                              ping, ack_buf, block_counters);
  if (t < 0) { printf(PROTO_TF "    %10zu B  TIMED HANG\n", data_bytes); return; }
  print_result(PROTO_TF, data_bytes, nlines, nb, t, nerr);
}

// ---------------------------------------------------------------------------
// main
// ---------------------------------------------------------------------------

static bool parse_int(const char* s, int* out)
{
  char* end = nullptr;
  errno = 0;
  long v = strtol(s, &end, 10);
  if (errno != 0 || end == s || *end != '\0') return false;
  int iv = (int)v;
  if ((long)iv != v) return false;
  *out = iv;
  return true;
}

static bool parse_double(const char* s, double* out)
{
  char* end = nullptr;
  errno = 0;
  double v = strtod(s, &end);
  if (errno != 0 || end == s || *end != '\0') return false;
  *out = v;
  return true;
}

static bool parse_size(const char* s, size_t* out)
{
  char* end = nullptr;
  errno = 0;
  unsigned long long v = strtoull(s, &end, 0);
  if (errno != 0 || end == s || v > (unsigned long long)SIZE_MAX) return false;
  if (*end == 'K' || *end == 'k') {
    if (v > ((unsigned long long)SIZE_MAX >> 10)) return false;
    v <<= 10;
    ++end;
  } else if (*end == 'M' || *end == 'm') {
    if (v > ((unsigned long long)SIZE_MAX >> 20)) return false;
    v <<= 20;
    ++end;
  } else if (*end == 'G' || *end == 'g') {
    if (v > ((unsigned long long)SIZE_MAX >> 30)) return false;
    v <<= 30;
    ++end;
  }
  if (*end != '\0') return false;
  *out = (size_t)v;
  return true;
}

int main(int argc, char** argv)
{
  int warmup = 10;
  int iters = 100;
  g_watchdog_timeout_s = 30.0;
  size_t max_data = (128ULL << 20);

  if ((argc > 1 && !parse_int(argv[1], &warmup)) ||
      (argc > 2 && !parse_int(argv[2], &iters)) ||
      (argc > 3 && !parse_double(argv[3], &g_watchdog_timeout_s)) ||
      (argc > 4 && !parse_size(argv[4], &max_data))) {
    fprintf(stderr, "Invalid numeric argument(s)\n");
    return 1;
  }

  if (warmup < 0 || warmup > ITERS_MAX || iters <= 0 || iters > ITERS_MAX) {
    fprintf(stderr, "warmup must be in [0, %d], iters must be in [1, %d]\n",
            ITERS_MAX, ITERS_MAX);
    return 1;
  }
  if (g_watchdog_timeout_s <= 0.0) {
    fprintf(stderr, "timeout_s must be > 0\n");
    return 1;
  }
  if (max_data < MIN_SWEEP_BYTES) {
    fprintf(stderr, "max_bytes must be >= %d\n", MIN_SWEEP_BYTES);
    return 1;
  }
  if (max_data > SIZE_MAX / 2) {
    fprintf(stderr, "max_bytes too large (max supported: %zu)\n", SIZE_MAX / 2);
    return 1;
  }


  int can01 = 0, can10 = 0;
  HIP_CHECK(hipDeviceCanAccessPeer(&can01, 0, 1));
  HIP_CHECK(hipDeviceCanAccessPeer(&can10, 1, 0));
  if (!can01 || !can10) { fprintf(stderr, "No peer access\n"); return 1; }
  HIP_CHECK(hipSetDevice(0));
  { hipError_t e = hipDeviceEnablePeerAccess(1, 0);
    if (e != hipSuccess && e != hipErrorPeerAccessAlreadyEnabled) {
      fprintf(stderr, "HIP error %s at %s:%d\n", hipGetErrorString(e), __FILE__, __LINE__);
      return 1; } }
  HIP_CHECK(hipSetDevice(1));
  { hipError_t e = hipDeviceEnablePeerAccess(0, 0);
    if (e != hipSuccess && e != hipErrorPeerAccessAlreadyEnabled) {
      fprintf(stderr, "HIP error %s at %s:%d\n", hipGetErrorString(e), __FILE__, __LINE__);
      return 1; } }

  printf("LL one-way benchmark  LINE_BYTES=%d  DATA_BYTES=%d  "
         "warmup=%d  iters=%d  timeout=%.0fs  max=%zu B\n",
         LINE_BYTES, DATA_BYTES, warmup, iters, g_watchdog_timeout_s, max_data);
  printf("Protocols: LL (16-B lines, 50%% eff)  "
         PROTO_SC " (s_waitcnt)  "
         PROTO_SC_WB " (+ threadfence_system)  "
         PROTO_TF " (one fence per iter)\n");
  printf("Wire efficiency LL<N>: %d/%d = %.1f%%\n\n",
         DATA_ELEMS, LINE_ELEMS, 100.0 * DATA_ELEMS / LINE_ELEMS);

  // Buffer must hold the worst-case wire footprint across all protocols at the
  // largest swept size. LL baseline: data_bytes * 2 (16-B lines, 50% eff).
  // LL<N>: ceil(data_bytes / DATA_BYTES) * LINE_BYTES — this also guarantees
  // at least one full line, covering the case where max_data < LINE_BYTES.
  const size_t ll_wire  = max_data * 2;
  const size_t lln_wire = ((max_data + DATA_BYTES - 1) / DATA_BYTES) * LINE_BYTES;
  size_t BUF_WIRE = ll_wire > lln_wire ? ll_wire : lln_wire;
  if (BUF_WIRE < (size_t)LINE_BYTES) BUF_WIRE = LINE_BYTES;

  // ping buffer on GPU1 (GPU0 writes remotely, GPU1 reads locally)
  // ack_buf on GPU0 (GPU1 writes remotely, GPU0 polls locally)
  // block_counters on GPU1 (GPU1 atomics; one int per iter)
  void *gpu1_ping;
  uint64_t* ack_buf;
  int* block_counters;
  hipSetDevice(1);
  HIP_CHECK(hipMalloc(&gpu1_ping, BUF_WIRE));     hipMemset(gpu1_ping, 0, BUF_WIRE);
  HIP_CHECK(hipMalloc(&block_counters, sizeof(int) * ITERS_MAX));
  hipSetDevice(0);
  HIP_CHECK(hipMalloc(&ack_buf, sizeof(uint64_t)));

  for (size_t sz = MIN_SWEEP_BYTES; sz <= max_data; sz *= 2) {
    run_ll        (warmup, iters, sz, gpu1_ping, ack_buf, block_counters);
    run_lln_sc    (warmup, iters, sz, gpu1_ping, ack_buf, block_counters);
    run_lln_sc_wb (warmup, iters, sz, gpu1_ping, ack_buf, block_counters);
    run_lln_tf    (warmup, iters, sz, gpu1_ping, ack_buf, block_counters);
    printf("\n"); fflush(stdout);
  }

  hipSetDevice(0); hipFree(ack_buf);          hipDeviceDisablePeerAccess(1);
  hipSetDevice(1); hipFree(gpu1_ping);        hipFree(block_counters); hipDeviceDisablePeerAccess(0);
  return 0;
}
