// SPDX-License-Identifier: MIT
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

// Correctness coverage for cooperative_groups::memcpy_async across the paths the
// implementation can select. The element type decides whether the accelerated path is
// eligible, by alignof(T) % 4 == 0. Alignments of 1 (unsigned char) and 2 (short) stay
// on the ordinary copy. Alignments of 4 (int, float), 8 (double, unsigned long long),
// and 16 take LDS DMA where the target provides it, and that path always moves 4 bytes
// per lane. LDS -> global is covered as well because only the global -> LDS direction
// has an LDS DMA path. The element-count overload is covered too: it copies
// nelem * sizeof(T) bytes and uses the same alignment gate.
//
// Block sizes include partial waves and non-multiples of either wave size on purpose.
// Where the copy is performed by LDS DMA the destination base is wave uniform and the
// hardware supplies the per-lane offset, so the partitioning depends on how ranks map
// onto lanes and those sizes are what stress it.
//
// Each case also checks a short guard region past the requested count. A copy that moves
// whole dwords has to stop short of a ragged tail rather than round it up, and without a
// guard an implementation that rounds up would still compare equal over the requested
// range and pass silently.
//
// This is a correctness test, not a path test: it is expected to pass regardless of
// which path the target selects.

#include <hip_test_common.hh>

#include <hip/hip_cooperative_groups.h>
#include <hip/cooperative_groups/memcpy_async.h>

#include <algorithm>
#include <vector>

namespace cg = cooperative_groups;

namespace {

constexpr int kSharedCap = 8192;
// Bytes checked past the requested count, so a copy that rounds the ragged tail up to a whole
// dword is caught instead of silently passing.
constexpr int kGuard = 8;

template <typename T>
__global__ void globalToLds(const T* in, T* out, int bytes, int readBack) {
  __shared__ __align__(16) unsigned char raw[kSharedCap];
  cg::thread_block block = cg::this_thread_block();

  for (int i = block.thread_rank(); i < kSharedCap; i += block.num_threads()) raw[i] = 0;
  block.sync();

  cg::memcpy_async(block, reinterpret_cast<T*>(raw), in, static_cast<size_t>(bytes));
  // group.sync() drains LDS DMA (vmcnt, before the barrier) on gfx9/gfx10, including
  // MI210, MI300A, and gfx950, and drains the gfx12.5 async counter. __syncthreads()
  // alone does not drain the gfx12.5 counter.
  block.sync();

  // readBack extends past bytes. raw was zero filled, so anything memcpy_async wrote outside the
  // requested range surfaces in the guard region as a non-zero byte.
  unsigned char* o = reinterpret_cast<unsigned char*>(out);
  for (int i = block.thread_rank(); i < readBack; i += block.num_threads()) o[i] = raw[i];
}

// alignas(16) element. LDS DMA still issues 4-byte loads for this type.
struct alignas(16) Wide16 {
  unsigned int e[4];
};

// Element-count overload. The copied length is nelem * sizeof(T) bytes.
template <typename T>
__global__ void globalToLdsElements(const T* in, T* out, int nelem, int readBack) {
  __shared__ __align__(16) unsigned char raw[kSharedCap];
  cg::thread_block block = cg::this_thread_block();

  for (int i = block.thread_rank(); i < kSharedCap; i += block.num_threads()) raw[i] = 0;
  block.sync();

  cg::memcpy_async(block, reinterpret_cast<T*>(raw), nelem, in, nelem);
  block.sync();

  unsigned char* o = reinterpret_cast<unsigned char*>(out);
  for (int i = block.thread_rank(); i < readBack; i += block.num_threads()) o[i] = raw[i];
}

template <typename T>
__global__ void ldsToGlobal(const T* in, T* out, int bytes) {
  __shared__ __align__(16) unsigned char raw[kSharedCap];
  cg::thread_block block = cg::this_thread_block();

  // Zero the whole buffer first so the bytes past the requested count hold a known value rather
  // than whatever was left in LDS. A copy that reads past bytes then stores zeros into the
  // destination guard region, which differs from the 0xAB the host put there.
  // Each rank owns the same indices in both loops, so no barrier is needed between them.
  for (int i = block.thread_rank(); i < kSharedCap; i += block.num_threads()) raw[i] = 0;

  const unsigned char* i8 = reinterpret_cast<const unsigned char*>(in);
  for (int i = block.thread_rank(); i < bytes; i += block.num_threads()) raw[i] = i8[i];
  block.sync();

  cg::memcpy_async(block, out, reinterpret_cast<T*>(raw), static_cast<size_t>(bytes));
  block.sync();
}

// whole waves, partial waves, and non-multiples of either wave size
constexpr int kThreads[] = {32, 64, 65, 100, 128, 129, 192, 200, 256, 320, 512, 1024};
// dword multiples, ragged tails, counts below the group size, and large copies
constexpr int kBytes[] = {2,   4,   6,    8,    12,   64,   100,  255,  256,  257, 258,
                          259, 1024, 1025, 1027, 2048, 4095, 4096, 8188};

template <typename T>
void runCase(const char* tag, int threads, int bytes, const std::vector<unsigned char>& ref,
             unsigned char* d_in, unsigned char* d_out, bool ldsToGlobalDir) {
  HIP_CHECK(hipMemset(d_out, 0xAB, kSharedCap));

  const int readBack = std::min(bytes + kGuard, kSharedCap);
  // Past `bytes` the destination must be untouched: 0xAB from the memset for the lds -> global
  // direction, 0 from the LDS zero fill for global -> lds.
  const unsigned guardByte = ldsToGlobalDir ? 0xABu : 0x00u;

  if (ldsToGlobalDir) {
    ldsToGlobal<T><<<1, threads>>>(reinterpret_cast<const T*>(d_in), reinterpret_cast<T*>(d_out),
                                   bytes);
  } else {
    globalToLds<T><<<1, threads>>>(reinterpret_cast<const T*>(d_in), reinterpret_cast<T*>(d_out),
                                   bytes, readBack);
  }
  HIP_CHECK(hipGetLastError());
  HIP_CHECK(hipDeviceSynchronize());

  std::vector<unsigned char> got(readBack, 0);
  HIP_CHECK(hipMemcpy(got.data(), d_out, readBack, hipMemcpyDeviceToHost));

  // Compare the whole buffer and assert once per case, so each case contributes an assertion
  // without emitting one per byte.
  int mismatch = -1;
  unsigned expected = 0;
  for (int i = 0; i < readBack; i++) {
    expected = (i < bytes) ? ref[i] : guardByte;
    if (got[i] != expected) {
      mismatch = i;
      break;
    }
  }

  if (mismatch >= 0) {
    // INFO registers a message scoped to the enclosing block, so one created inside this `if` is
    // destroyed before the REQUIRE below runs and never prints. UNSCOPED_INFO survives until the
    // next assertion, which is the one that needs it.
    UNSCOPED_INFO(tag << " threads: " << threads << " bytes: " << bytes
                      << (mismatch >= bytes ? " wrote past the requested count at index: "
                                            : " first mismatch at index: ")
                      << mismatch << " got: " << static_cast<unsigned>(got[mismatch])
                      << " expected: " << expected);
  }
  REQUIRE(mismatch == -1);
}

template <typename T>
void runElements(const char* tag, int threads, int nelem, const std::vector<unsigned char>& ref,
                 unsigned char* d_in, unsigned char* d_out) {
  const int bytes = nelem * static_cast<int>(sizeof(T));
  const int readBack = std::min(bytes + kGuard, kSharedCap);
  HIP_CHECK(hipMemset(d_out, 0xAB, kSharedCap));
  globalToLdsElements<T><<<1, threads>>>(reinterpret_cast<const T*>(d_in),
                                         reinterpret_cast<T*>(d_out), nelem, readBack);
  HIP_CHECK(hipGetLastError());
  HIP_CHECK(hipDeviceSynchronize());

  std::vector<unsigned char> got(readBack, 0);
  HIP_CHECK(hipMemcpy(got.data(), d_out, readBack, hipMemcpyDeviceToHost));

  int mismatch = -1;
  unsigned expected = 0;
  for (int i = 0; i < readBack; i++) {
    expected = (i < bytes) ? ref[i] : 0x00u;
    if (got[i] != expected) {
      mismatch = i;
      break;
    }
  }
  if (mismatch >= 0) {
    UNSCOPED_INFO(tag << " threads: " << threads << " elements: " << nelem << " bytes: " << bytes
                      << (mismatch >= bytes ? " wrote past the requested count at index: "
                                            : " first mismatch at index: ")
                      << mismatch << " got: " << static_cast<unsigned>(got[mismatch])
                      << " expected: " << expected);
  }
  REQUIRE(mismatch == -1);
}

// Scratch is not a global address. The whole block participates so the completion path is the
// real workgroup barrier, including a partial last wave on both wave32 and wave64.
__global__ void privateSourceToLds(unsigned char* out, int* probe) {
  __shared__ __align__(16) unsigned char raw[16];
  cg::thread_block block = cg::this_thread_block();
  // volatile plus the asm use keep this in scratch. A plain array is legal to promote
  // into LDS, which would make this case pass without ever rejecting a private pointer.
  // Every rank writes the same first three bytes and tags the fourth with its own rank,
  // so the fourth byte identifies which rank's scratch the fallback actually read.
  volatile unsigned char mine[4];
  mine[0] = 0x11;
  mine[1] = 0x22;
  mine[2] = 0x33;
  mine[3] = static_cast<unsigned char>(0x40 + block.thread_rank());
  const void* p = const_cast<unsigned char*>(mine);
  asm volatile("" : "+v"(p)::"memory");

  if (block.thread_rank() == 0) {
#if __has_builtin(__builtin_amdgcn_is_private) && __has_builtin(__builtin_amdgcn_is_shared)
    const auto* q = (const __attribute__((address_space(0))) void*)p;
    probe[0] = __builtin_amdgcn_is_private(q);
    probe[1] = __builtin_amdgcn_is_shared(q);
#else
    probe[0] = 1;
    probe[1] = 0;
    (void)p;
#endif
  }

  for (int i = block.thread_rank(); i < 16; i += block.num_threads()) raw[i] = 0;
  block.sync();

  cg::memcpy_async(block, raw, const_cast<unsigned char*>(mine), static_cast<size_t>(4));
  block.sync();

  if (block.thread_rank() == 0) {
    for (int i = 0; i < 16; i++) out[i] = raw[i];
  }
}


// True when this target completes memcpy_async through the gfx12.5 async counter. That
// counter is not drained by any fence or barrier, so those targets require group.sync().
// Everywhere else the copy retires through vmcnt, which a workgroup barrier does wait for.
__global__ void probeAsyncCounter(int* out) {
#if __has_builtin(__builtin_amdgcn_s_wait_asynccnt)
  *out = __builtin_amdgcn_is_invocable(__builtin_amdgcn_s_wait_asynccnt) ? 1 : 0;
#else
  *out = 0;
#endif
}

// Same copy as globalToLds, synchronised with __syncthreads() rather than block.sync().
// A large amount of existing code is written this way, so on every target whose copy
// retires through vmcnt it has to keep working.
template <typename T>
__global__ void globalToLdsSyncthreads(const T* in, T* out, int bytes, int readBack) {
  __shared__ __align__(16) unsigned char raw[kSharedCap];
  cg::thread_block block = cg::this_thread_block();

  for (int i = block.thread_rank(); i < kSharedCap; i += block.num_threads()) raw[i] = 0;
  __syncthreads();

  cg::memcpy_async(block, reinterpret_cast<T*>(raw), in, static_cast<size_t>(bytes));
  __syncthreads();

  unsigned char* o = reinterpret_cast<unsigned char*>(out);
  for (int i = block.thread_rank(); i < readBack; i += block.num_threads()) o[i] = raw[i];
}

// The LDS destination does not have to start at the beginning of the shared allocation, and
// the source does not have to be 16-byte aligned. Where LDS DMA runs, the destination base
// is wave uniform and the hardware supplies the per-lane offset, so a base that is not the
// start of the block is what exercises that. Offsets that are not a multiple of 4 must be
// rejected by the alignment gate and fall back.
template <typename T>
__global__ void globalToLdsOffsetBase(const T* in, T* out, int bytes, int readBack, int off) {
  __shared__ __align__(16) unsigned char raw[kSharedCap + 64];
  cg::thread_block block = cg::this_thread_block();

  for (int i = block.thread_rank(); i < kSharedCap + 64; i += block.num_threads()) raw[i] = 0;
  block.sync();

  const unsigned char* s8 = reinterpret_cast<const unsigned char*>(in);
  cg::memcpy_async(block, reinterpret_cast<T*>(raw + off), reinterpret_cast<const T*>(s8 + off),
                   static_cast<size_t>(bytes));
  block.sync();

  unsigned char* o = reinterpret_cast<unsigned char*>(out);
  for (int i = block.thread_rank(); i < readBack; i += block.num_threads()) o[i] = raw[off + i];
}

// Two copies into disjoint LDS regions with one sync after both, which is the double
// buffering shape memcpy_async exists for. Both have to have landed when the barrier
// releases, not just the most recent one.
__global__ void twoCopiesOneSync(const int* in, int* out, int bytes) {
  __shared__ __align__(16) unsigned char a[kSharedCap / 2];
  __shared__ __align__(16) unsigned char b[kSharedCap / 2];
  cg::thread_block block = cg::this_thread_block();

  for (int i = block.thread_rank(); i < kSharedCap / 2; i += block.num_threads()) {
    a[i] = 0;
    b[i] = 0;
  }
  block.sync();

  const unsigned char* s8 = reinterpret_cast<const unsigned char*>(in);
  cg::memcpy_async(block, reinterpret_cast<int*>(a), in, static_cast<size_t>(bytes));
  cg::memcpy_async(block, reinterpret_cast<int*>(b), reinterpret_cast<const int*>(s8 + bytes),
                   static_cast<size_t>(bytes));
  block.sync();

  unsigned char* o = reinterpret_cast<unsigned char*>(out);
  for (int i = block.thread_rank(); i < bytes; i += block.num_threads()) o[i] = a[i];
  for (int i = block.thread_rank(); i < bytes; i += block.num_threads()) o[bytes + i] = b[i];
}

// The split barrier has to complete the copy too. barrier_arrive is where the wait has to
// happen: the last wave's arrive can release everyone else, so waiting in barrier_wait
// would be too late.
__global__ void arriveWaitPath(const int* in, int* out, int bytes, int readBack) {
  __shared__ __align__(16) unsigned char raw[kSharedCap];
  cg::thread_block block = cg::this_thread_block();

  for (int i = block.thread_rank(); i < kSharedCap; i += block.num_threads()) raw[i] = 0;
  block.sync();

  cg::memcpy_async(block, reinterpret_cast<int*>(raw), in, static_cast<size_t>(bytes));
  block.barrier_wait(block.barrier_arrive());

  unsigned char* o = reinterpret_cast<unsigned char*>(out);
  for (int i = block.thread_rank(); i < readBack; i += block.num_threads()) o[i] = raw[i];
}

// Many resident blocks at once. A single block cannot surface a completion bug that only
// appears when the memory system is loaded enough for the copy to still be outstanding at
// the barrier.
__global__ void globalToLdsManyBlocks(const int* in, unsigned char* out, int bytes) {
  __shared__ __align__(16) unsigned char raw[kSharedCap];
  cg::thread_block block = cg::this_thread_block();

  for (int i = block.thread_rank(); i < kSharedCap; i += block.num_threads()) raw[i] = 0;
  block.sync();

  cg::memcpy_async(block, reinterpret_cast<int*>(raw), in, static_cast<size_t>(bytes));
  block.sync();

  unsigned char* o = out + static_cast<size_t>(blockIdx.x) * kSharedCap;
  for (int i = block.thread_rank(); i < bytes; i += block.num_threads()) o[i] = raw[i];
}

}  // namespace

TEST_CASE("Unit_device_memcpy_async_paths") {
  unsigned char *d_in = nullptr, *d_out = nullptr;
  HIP_CHECK(hipMalloc(&d_in, kSharedCap));
  HIP_CHECK(hipMalloc(&d_out, kSharedCap));

  std::vector<unsigned char> ref(kSharedCap);
  for (size_t i = 0; i < ref.size(); i++) ref[i] = static_cast<unsigned char>(i * 7 + 3);
  HIP_CHECK(hipMemcpy(d_in, ref.data(), kSharedCap, hipMemcpyHostToDevice));

  SECTION("global to lds, dword aligned elements") {
    for (int t : kThreads)
      for (int b : kBytes) runCase<int>("int g2l", t, b, ref, d_in, d_out, false);
  }

  SECTION("global to lds, byte elements") {
    for (int t : kThreads)
      for (int b : kBytes) runCase<unsigned char>("char g2l", t, b, ref, d_in, d_out, false);
  }

  SECTION("global to lds, 2-byte elements") {
    for (int t : kThreads)
      for (int b : kBytes) runCase<short>("short g2l", t, b, ref, d_in, d_out, false);
  }

  SECTION("global to lds, 4-byte float") {
    for (int t : kThreads)
      for (int b : kBytes) runCase<float>("float g2l", t, b, ref, d_in, d_out, false);
  }

  SECTION("global to lds, 8-byte elements") {
    for (int t : kThreads) {
      for (int b : kBytes) {
        runCase<double>("double g2l", t, b, ref, d_in, d_out, false);
        runCase<unsigned long long>("u64 g2l", t, b, ref, d_in, d_out, false);
      }
    }
  }

  SECTION("global to lds, 16-byte elements") {
    for (int t : kThreads)
      for (int b : kBytes) runCase<Wide16>("wide16 g2l", t, b, ref, d_in, d_out, false);
  }

  SECTION("global to lds, element count") {
    const int threads[] = {32, 65, 128, 256, 1024};
    for (int t : threads) {
      runElements<short>("short elements", t, 100, ref, d_in, d_out);
      runElements<float>("float elements", t, 100, ref, d_in, d_out);
      runElements<double>("double elements", t, 100, ref, d_in, d_out);
      runElements<unsigned long long>("u64 elements", t, 100, ref, d_in, d_out);
      runElements<Wide16>("wide16 elements", t, 100, ref, d_in, d_out);
    }
  }

  SECTION("lds to global, dword aligned elements") {
    for (int t : kThreads)
      for (int b : kBytes) runCase<int>("int l2g", t, b, ref, d_in, d_out, true);
  }

  // A per-thread stack address is scratch. LDS DMA must not consume it: the global
  // aperture cast is the wrong memory on MI210, and on MI300A the load is not replayed
  // if it faults. The ordinary copy reads it instead, one rank per byte.
  SECTION("private source falls back") {
    unsigned char* d_flag = nullptr;
    int* d_probe = nullptr;
    HIP_CHECK(hipMalloc(&d_flag, 16));
    HIP_CHECK(hipMalloc(&d_probe, 2 * sizeof(int)));
    HIP_CHECK(hipMemset(d_flag, 0xAB, 16));
    HIP_CHECK(hipMemset(d_probe, 0xFF, 2 * sizeof(int)));
    privateSourceToLds<<<1, 128>>>(d_flag, d_probe);
    HIP_CHECK(hipGetLastError());
    HIP_CHECK(hipDeviceSynchronize());
    unsigned char got[16] = {};
    int probe[2] = {-1, -1};
    HIP_CHECK(hipMemcpy(got, d_flag, 16, hipMemcpyDeviceToHost));
    HIP_CHECK(hipMemcpy(probe, d_probe, sizeof(probe), hipMemcpyDeviceToHost));
    REQUIRE(probe[0] == 1);
    REQUIRE(probe[1] == 0);
    // The first three bytes are identical in every thread, so they pin down that the
    // fallback read scratch rather than some other aperture. The fourth says which rank's
    // scratch supplied it. Which rank that is depends on how the ordinary copy partitions
    // the bytes, which is not part of the contract -- memcpy_async is collective and a
    // per-thread source pointer is outside it -- so any rank in range is a pass. A value
    // outside the range means the bytes did not come from a scratch read at all.
    const unsigned char expect[3] = {0x11, 0x22, 0x33};
    for (int i = 0; i < 3; i++) {
      INFO("private source byte " << i);
      REQUIRE(got[i] == expect[i]);
    }
    INFO("private source rank byte: " << static_cast<unsigned>(got[3]));
    REQUIRE(got[3] >= 0x40);
    REQUIRE(got[3] < 0x40 + 128);  // 128 is the block size launched above
    for (int i = 4; i < 16; i++) {
      INFO("private source wrote past 4 bytes at " << i);
      REQUIRE(got[i] == 0);
    }
    HIP_CHECK(hipFree(d_flag));
    HIP_CHECK(hipFree(d_probe));
  }


  // The LDS base the accelerated copy is handed is wave uniform, so a destination that does
  // not start at the beginning of the shared allocation is a distinct case. Offsets of 1, 2
  // and 3 additionally have to be turned away by the alignment gate.
  SECTION("global to lds, offset and misaligned destination base") {
    for (int t : {64, 128, 129, 256, 1024}) {
      for (int b : {64, 255, 1024, 4095}) {
        for (int off : {1, 2, 3, 4, 8, 12, 36}) {
          HIP_CHECK(hipMemset(d_out, 0xAB, kSharedCap));
          const int readBack = std::min(b + kGuard, kSharedCap);
          globalToLdsOffsetBase<int><<<1, t>>>(reinterpret_cast<const int*>(d_in),
                                               reinterpret_cast<int*>(d_out), b, readBack, off);
          HIP_CHECK(hipGetLastError());
          HIP_CHECK(hipDeviceSynchronize());
          std::vector<unsigned char> got(readBack, 0);
          HIP_CHECK(hipMemcpy(got.data(), d_out, readBack, hipMemcpyDeviceToHost));
          int mismatch = -1;
          for (int i = 0; i < readBack; i++) {
            const unsigned expected = (i < b) ? ref[off + i] : 0x00u;
            if (got[i] != expected) {
              mismatch = i;
              break;
            }
          }
          if (mismatch >= 0) {
            UNSCOPED_INFO("offset base off: " << off << " threads: " << t << " bytes: " << b
                                              << " first mismatch at index: " << mismatch);
          }
          REQUIRE(mismatch == -1);
        }
      }
    }
  }

  SECTION("two copies, one sync") {
    for (int t : {64, 128, 200, 256, 1024}) {
      for (int b : {256, 1024, 4096}) {
        HIP_CHECK(hipMemset(d_out, 0xAB, kSharedCap));
        twoCopiesOneSync<<<1, t>>>(reinterpret_cast<const int*>(d_in),
                                   reinterpret_cast<int*>(d_out), b);
        HIP_CHECK(hipGetLastError());
        HIP_CHECK(hipDeviceSynchronize());
        std::vector<unsigned char> got(2 * b, 0);
        HIP_CHECK(hipMemcpy(got.data(), d_out, 2 * b, hipMemcpyDeviceToHost));
        int mismatch = -1;
        for (int i = 0; i < 2 * b; i++) {
          if (got[i] != ref[i]) {
            mismatch = i;
            break;
          }
        }
        if (mismatch >= 0) {
          UNSCOPED_INFO("two copies threads: " << t << " bytes: " << b
                                               << " first mismatch at index: " << mismatch);
        }
        REQUIRE(mismatch == -1);
      }
    }
  }

  SECTION("barrier_arrive and barrier_wait complete the copy") {
    for (int t : {64, 128, 200, 256, 1024}) {
      for (int b : {256, 1025, 4096}) {
        HIP_CHECK(hipMemset(d_out, 0xAB, kSharedCap));
        const int readBack = std::min(b + kGuard, kSharedCap);
        arriveWaitPath<<<1, t>>>(reinterpret_cast<const int*>(d_in),
                                 reinterpret_cast<int*>(d_out), b, readBack);
        HIP_CHECK(hipGetLastError());
        HIP_CHECK(hipDeviceSynchronize());
        std::vector<unsigned char> got(readBack, 0);
        HIP_CHECK(hipMemcpy(got.data(), d_out, readBack, hipMemcpyDeviceToHost));
        int mismatch = -1;
        for (int i = 0; i < readBack; i++) {
          const unsigned expected = (i < b) ? ref[i] : 0x00u;
          if (got[i] != expected) {
            mismatch = i;
            break;
          }
        }
        if (mismatch >= 0) {
          UNSCOPED_INFO("barrier_arrive threads: " << t << " bytes: " << b
                                                   << " first mismatch at index: " << mismatch);
        }
        REQUIRE(mismatch == -1);
      }
    }
  }

  // One block cannot surface a completion bug that needs a loaded memory system.
  SECTION("global to lds, many concurrent blocks") {
    constexpr int kBlocks = 1024;
    unsigned char* d_many = nullptr;
    HIP_CHECK(hipMalloc(&d_many, static_cast<size_t>(kBlocks) * kSharedCap));
    for (int t : {64, 256, 1024}) {
      for (int b : {1024, 4096, 8188}) {
        HIP_CHECK(hipMemset(d_many, 0xAB, static_cast<size_t>(kBlocks) * kSharedCap));
        globalToLdsManyBlocks<<<kBlocks, t>>>(reinterpret_cast<const int*>(d_in), d_many, b);
        HIP_CHECK(hipGetLastError());
        HIP_CHECK(hipDeviceSynchronize());
        std::vector<unsigned char> got(static_cast<size_t>(kBlocks) * kSharedCap);
        HIP_CHECK(hipMemcpy(got.data(), d_many, got.size(), hipMemcpyDeviceToHost));
        int badBlock = -1, mismatch = -1;
        for (int blk = 0; blk < kBlocks && badBlock < 0; blk++) {
          for (int i = 0; i < b; i++) {
            if (got[static_cast<size_t>(blk) * kSharedCap + i] != ref[i]) {
              badBlock = blk;
              mismatch = i;
              break;
            }
          }
        }
        if (badBlock >= 0) {
          UNSCOPED_INFO("many blocks threads: " << t << " bytes: " << b << " block: " << badBlock
                                                << " first mismatch at index: " << mismatch);
        }
        REQUIRE(badBlock == -1);
      }
    }
    HIP_CHECK(hipFree(d_many));
  }

  // Targets whose copy retires through vmcnt are drained by any workgroup barrier, because
  // the barrier performs a workgroup release fence. Only the gfx12.5 async counter needs
  // group.sync() specifically, so that target is skipped here rather than asserted against.
  SECTION("global to lds, synchronised with __syncthreads()") {
    int* d_async = nullptr;
    HIP_CHECK(hipMalloc(&d_async, sizeof(int)));
    HIP_CHECK(hipMemset(d_async, 0, sizeof(int)));
    probeAsyncCounter<<<1, 1>>>(d_async);
    HIP_CHECK(hipGetLastError());
    HIP_CHECK(hipDeviceSynchronize());
    int usesAsyncCounter = 0;
    HIP_CHECK(hipMemcpy(&usesAsyncCounter, d_async, sizeof(int), hipMemcpyDeviceToHost));
    HIP_CHECK(hipFree(d_async));

    if (usesAsyncCounter) {
      WARN("target completes memcpy_async through the async counter; __syncthreads() is not "
           "sufficient there by design, skipping");
    } else {
      for (int t : kThreads) {
        for (int b : kBytes) {
          HIP_CHECK(hipMemset(d_out, 0xAB, kSharedCap));
          const int readBack = std::min(b + kGuard, kSharedCap);
          globalToLdsSyncthreads<int><<<1, t>>>(reinterpret_cast<const int*>(d_in),
                                                reinterpret_cast<int*>(d_out), b, readBack);
          HIP_CHECK(hipGetLastError());
          HIP_CHECK(hipDeviceSynchronize());
          std::vector<unsigned char> got(readBack, 0);
          HIP_CHECK(hipMemcpy(got.data(), d_out, readBack, hipMemcpyDeviceToHost));
          int mismatch = -1;
          for (int i = 0; i < readBack; i++) {
            const unsigned expected = (i < b) ? ref[i] : 0x00u;
            if (got[i] != expected) {
              mismatch = i;
              break;
            }
          }
          if (mismatch >= 0) {
            UNSCOPED_INFO("__syncthreads threads: " << t << " bytes: " << b
                                                    << " first mismatch at index: " << mismatch);
          }
          REQUIRE(mismatch == -1);
        }
      }
    }
  }

  HIP_CHECK(hipFree(d_in));
  HIP_CHECK(hipFree(d_out));
}
