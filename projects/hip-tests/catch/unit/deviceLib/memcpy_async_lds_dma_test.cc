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
// eligible: only types aligned to at least a dword can use it, so int-typed copies
// exercise the accelerated path where the target provides one while char-typed copies
// exercise the traditional fallback. LDS -> global is covered as well because only the
// global -> LDS direction has an LDS DMA path.
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
  block.sync();  // memcpy_async is asynchronous; the barrier is what makes raw readable

  // readBack extends past bytes. raw was zero filled, so anything memcpy_async wrote outside the
  // requested range surfaces in the guard region as a non-zero byte.
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
constexpr int kBytes[] = {4,   8,    12,   64,   100,  255,  256,  257, 258,
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

  SECTION("lds to global, dword aligned elements") {
    for (int t : kThreads)
      for (int b : kBytes) runCase<int>("int l2g", t, b, ref, d_in, d_out, true);
  }

  HIP_CHECK(hipFree(d_in));
  HIP_CHECK(hipFree(d_out));
}
