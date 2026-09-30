/*
Copyright (c) 2025 Advanced Micro Devices, Inc. All rights reserved.

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

#include <hip_test_common.hh>

#include <hip/hip_cooperative_groups.h>
#include <hip/cooperative_groups/memcpy_async.h>

#include <algorithm>
#include <utility>
#include <vector>

// trivial vector add using shared memory
// a[i] * x + b[i]
__global__ void vector_add_mem(float* out, float* a, float* b, float x, size_t size) {
  extern __shared__ float b_smem[];
  auto tg = cooperative_groups::this_thread_block();

  // Async copy memory
  cooperative_groups::memcpy_async(tg, b_smem, b, size * sizeof(float));

  size_t i = threadIdx.x;

  // While copy is being done, we a[i] * x
  float tmp = a[i] * x;

  tg.sync();  // make sure copy is finished

  // do `+ b[i]` from shared mem
  tmp += b_smem[i];

  // Write back to shared mem, we can simplify this (do this in one go) but this is a test
  b_smem[i] = tmp;

  // Copy back data to global memory
  cooperative_groups::memcpy_async(tg, out, b_smem, size * sizeof(float));

  // Wait till its done
  tg.sync();
}

__global__ void vector_add_mem_layout(float* out, float* a, float* b, float x, size_t size) {
  extern __shared__ float b_smem[];
  auto tg = cooperative_groups::this_thread_block();

  // Async copy memory
  cooperative_groups::memcpy_async(tg, b_smem, size, b, size);

  size_t i = threadIdx.x;

  // While copy is being done, we a[i] * x
  float tmp = a[i] * x;

  tg.sync();  // make sure copy is finished

  // do `+ b[i]` from shared mem
  tmp += b_smem[i];

  // Write back to shared mem, we can simplify this (do this in one go) but this is a test
  b_smem[i] = tmp;

  // Copy back data to global memory
  cooperative_groups::memcpy_async(tg, out, size, b_smem, size);

  // Wait till its done
  tg.sync();
}

TEST_CASE("Unit_device_memcpy_async") {
  std::vector<std::string> supported_arch{"gfx1250"};
  hipDeviceProp_t prop;

  HIP_CHECK(hipGetDeviceProperties(&prop, 0));
  const std::string arch_name{prop.gcnArchName};
  std::cout << "cluster lauch: " << prop.clusterLaunch << std::endl;

  const bool should_run = std::any_of(
      supported_arch.begin(), supported_arch.end(),
      [&arch_name](const std::string_view& in) { return in.find(arch_name) != std::string::npos; });

  if (should_run) {
    SECTION("bytes copy") {
      for (const size_t size : {32, 64, 128, 129 /* weird non aligned size */}) {
        const size_t alloc_size = size * sizeof(float);
        const float x = 2.0f;
        float *d_out, *d_a, *d_b;

        HIP_CHECK(hipMalloc(&d_out, alloc_size));
        HIP_CHECK(hipMalloc(&d_a, alloc_size));
        HIP_CHECK(hipMalloc(&d_b, alloc_size));

        std::vector<float> a(size, 0.0f), b(size, 0.0f), cpu_out(size, 0.0f), gpu_out(size, 0.0f);
        for (size_t i = 0; i < size; i++) {
          a[i] = i + 1;
          b[i] = (i + 1) * 2;
          cpu_out[i] = (a[i] * x) + b[i];
        }

        HIP_CHECK(hipMemcpy(d_a, a.data(), alloc_size, hipMemcpyHostToDevice));
        HIP_CHECK(hipMemcpy(d_b, b.data(), alloc_size, hipMemcpyHostToDevice));
        HIP_CHECK(hipMemset(d_out, 0, alloc_size));

        vector_add_mem<<<1, size, alloc_size, nullptr>>>(d_out, d_a, d_b, x, size);

        HIP_CHECK(hipMemcpy(gpu_out.data(), d_out, alloc_size, hipMemcpyDeviceToHost));

        HIP_CHECK(hipFree(d_out));
        HIP_CHECK(hipFree(d_a));
        HIP_CHECK(hipFree(d_b));

        for (size_t i = 0; i < size; i++) {
          INFO("size: " << size << " index: " << i << " calc: " << a[i] << " * " << x << " + "
                        << b[i] << " = " << cpu_out[i] << ", " << gpu_out[i]);
          REQUIRE(cpu_out[i] == gpu_out[i]);
        }
      }
    }

    SECTION("layout copy") {
      for (const size_t size : {31, 65, 128, 256}) {
        const size_t alloc_size = size * sizeof(float);
        const float x = 2.0f;
        float *d_out, *d_a, *d_b;

        HIP_CHECK(hipMalloc(&d_out, alloc_size));
        HIP_CHECK(hipMalloc(&d_a, alloc_size));
        HIP_CHECK(hipMalloc(&d_b, alloc_size));

        std::vector<float> a(size, 0.0f), b(size, 0.0f), cpu_out(size, 0.0f), gpu_out(size, 0.0f);
        for (size_t i = 0; i < size; i++) {
          a[i] = i + 1;
          b[i] = (i + 1) * 2;
          cpu_out[i] = (a[i] * x) + b[i];
        }

        HIP_CHECK(hipMemcpy(d_a, a.data(), alloc_size, hipMemcpyHostToDevice));
        HIP_CHECK(hipMemcpy(d_b, b.data(), alloc_size, hipMemcpyHostToDevice));
        HIP_CHECK(hipMemset(d_out, 0, alloc_size));

        vector_add_mem_layout<<<1, size, alloc_size, nullptr>>>(d_out, d_a, d_b, x, size);

        HIP_CHECK(hipMemcpy(gpu_out.data(), d_out, alloc_size, hipMemcpyDeviceToHost));

        HIP_CHECK(hipFree(d_out));
        HIP_CHECK(hipFree(d_a));
        HIP_CHECK(hipFree(d_b));

        for (size_t i = 0; i < size; i++) {
          INFO("size: " << size << " index: " << i << " calc: " << a[i] << " * " << x << " + "
                        << b[i] << " = " << cpu_out[i] << ", " << gpu_out[i]);
          REQUIRE(cpu_out[i] == gpu_out[i]);
        }
      }
    }
  }
}

// Exercise the cooperative_groups::memcpy_async() API surface: supported group types, edge cases,
// and the layout overload. All of them assert the documented contract, that the copy is complete
// once the group has synced. Reading the destination without a group sync is undefined, so there
// is no device on which failing this is acceptable behaviour.

namespace {

// Pre-filled into shared memory so that a copy which has not landed yet is
// distinguishable from a copy that produced the right answer.
constexpr float kSentinel = -42.0f;

template <unsigned int TileSize>
__global__ void tile_memcpy_async_kernel(float* out, const float* in, size_t per_tile_bytes) {
  namespace cg = cooperative_groups;
  extern __shared__ float smem[];
  auto block = cg::this_thread_block();
  auto tile = cg::tiled_partition<TileSize>(block);

  const unsigned int tiles_per_block = blockDim.x / TileSize;
  const unsigned int tile_id = threadIdx.x / TileSize;
  const size_t per_tile_elems = per_tile_bytes / sizeof(float);

  float* smem_chunk = smem + tile_id * per_tile_elems;
  const float* in_chunk = in + (blockIdx.x * tiles_per_block + tile_id) * per_tile_elems;
  float* out_chunk = out + (blockIdx.x * tiles_per_block + tile_id) * per_tile_elems;

  cg::memcpy_async(tile, smem_chunk, in_chunk, per_tile_bytes);
  tile.sync();

  // Touch + double the data to prove it actually arrived.
  for (unsigned int i = tile.thread_rank(); i < per_tile_elems; i += TileSize) {
    smem_chunk[i] = smem_chunk[i] * 2.0f;
  }
  tile.sync();

  cg::memcpy_async(tile, out_chunk, smem_chunk, per_tile_bytes);
  tile.sync();
}

__global__ void coalesced_memcpy_async_kernel(float* out, const float* in, size_t chunk_elems) {
  namespace cg = cooperative_groups;
  extern __shared__ float smem[];
  auto active = cg::coalesced_threads();

  // coalesced_threads() groups the converged lanes of a single wave, so a multi-wave block yields
  // one group per wave and active.sync() orders only that wave. Each group therefore needs a
  // disjoint chunk.
  const size_t offset = (threadIdx.x / static_cast<unsigned int>(warpSize)) * chunk_elems;
  const size_t chunk_bytes = chunk_elems * sizeof(float);
  float* smem_chunk = smem + offset;

  cg::memcpy_async(active, smem_chunk, in + offset, chunk_bytes);
  active.sync();

  for (size_t i = active.thread_rank(); i < chunk_elems; i += active.size()) {
    smem_chunk[i] += 1.0f;
  }
  active.sync();

  cg::memcpy_async(active, out + offset, smem_chunk, chunk_bytes);
  active.sync();
}

__global__ void layout_min_kernel(float* out, const float* in, size_t dst_count,
                                  size_t src_count) {
  namespace cg = cooperative_groups;
  extern __shared__ float smem[];
  auto tb = cg::this_thread_block();
  // Layout-overload: copies min(dst_count, src_count) elements.
  cg::memcpy_async(tb, smem, dst_count, in, src_count);
  tb.sync();

  const size_t copied = dst_count < src_count ? dst_count : src_count;
  for (size_t i = threadIdx.x; i < copied; i += blockDim.x) {
    out[i] = smem[i];
  }
}

__global__ void zero_count_kernel(float* out, const float* in, size_t size) {
  namespace cg = cooperative_groups;
  extern __shared__ float smem[];
  auto tb = cg::this_thread_block();

  // A zero-byte memcpy_async must leave the destination alone, so the sentinel is what should
  // surface in out[].
  size_t i = threadIdx.x;
  smem[i] = kSentinel;
  tb.sync();

  cg::memcpy_async(tb, smem, in, /*count=*/static_cast<size_t>(0));
  tb.sync();

  out[i] = smem[i];
  // Independently, do a real copy of `size` floats to confirm later calls still work after a
  // zero-count call.
  cg::memcpy_async(tb, smem, in, size * sizeof(float));
  tb.sync();
  out[size + i] = smem[i];
}

__global__ void unaligned_size_kernel(float* out, const float* in, size_t bytes) {
  namespace cg = cooperative_groups;
  extern __shared__ unsigned char smem_b[];
  auto tb = cg::this_thread_block();
  cg::memcpy_async(tb, smem_b, reinterpret_cast<const unsigned char*>(in), bytes);
  tb.sync();
  // Single-thread copy-out so we don't depend on element alignment of `out`.
  if (threadIdx.x == 0) {
    auto* dst = reinterpret_cast<unsigned char*>(out);
    for (size_t i = 0; i < bytes; i++) dst[i] = smem_b[i];
  }
}

// Nothing at all happens between the copy and the sync, so the sync is the only thing that can
// make the data visible. A sync that does not wait for the async copy leaves the sentinel in place.
__global__ void sync_publishes_block_kernel(float* out, const float* in, size_t bytes) {
  namespace cg = cooperative_groups;
  extern __shared__ float smem[];
  auto tb = cg::this_thread_block();

  smem[threadIdx.x] = kSentinel;
  tb.sync();

  cg::memcpy_async(tb, smem, in, bytes);
  tb.sync();

  out[threadIdx.x] = smem[threadIdx.x];
}

template <unsigned int TileSize>
__global__ void sync_publishes_tile_kernel(float* out, const float* in, size_t bytes) {
  namespace cg = cooperative_groups;
  extern __shared__ float smem[];
  auto tb = cg::this_thread_block();
  auto tile = cg::tiled_partition<TileSize>(tb);

  smem[threadIdx.x] = kSentinel;
  tb.sync();

  cg::memcpy_async(tile, smem, in, bytes);
  tile.sync();

  out[threadIdx.x] = smem[threadIdx.x];
}

// Each thread reads the element some other wave was responsible for copying, so the group barrier
// - not the reading wave's own state - is what has to publish the data. A drain placed after the
// barrier instead of before it passes the kernel above but fails this one.
__global__ void cross_wave_visibility_kernel(float* out, const float* in, size_t bytes) {
  namespace cg = cooperative_groups;
  extern __shared__ float smem[];
  auto tb = cg::this_thread_block();

  smem[threadIdx.x] = kSentinel;
  tb.sync();

  cg::memcpy_async(tb, smem, in, bytes);
  tb.sync();

  out[threadIdx.x] = smem[blockDim.x - 1 - threadIdx.x];
}

// The hardware async path only implements global<->LDS. A global->global copy has to fall back to
// the traditional path rather than silently moving nothing.
__global__ void global_to_global_kernel(float* out, const float* in, size_t bytes) {
  namespace cg = cooperative_groups;
  auto tb = cg::this_thread_block();
  cg::memcpy_async(tb, out, in, bytes);
  tb.sync();
}

}  // namespace

HIP_TEST_CASE(Unit_coop_memcpy_async_thread_block_tile_Basic) {
  constexpr unsigned int kTile = 32;
  for (const unsigned int block_threads : {32u, 64u, 128u, 256u}) {
    if (block_threads % kTile != 0) continue;
    const unsigned int tiles_per_block = block_threads / kTile;
    const size_t per_tile_elems = 64;
    const size_t per_tile_bytes = per_tile_elems * sizeof(float);
    const size_t total_elems = tiles_per_block * per_tile_elems;
    const size_t total_bytes = total_elems * sizeof(float);

    float *d_in, *d_out;
    HIP_CHECK(hipMalloc(&d_in, total_bytes));
    HIP_CHECK(hipMalloc(&d_out, total_bytes));

    std::vector<float> in(total_elems), out(total_elems, 0.0f);
    for (size_t i = 0; i < total_elems; i++) in[i] = static_cast<float>(i + 1);
    HIP_CHECK(hipMemcpy(d_in, in.data(), total_bytes, hipMemcpyHostToDevice));
    HIP_CHECK(hipMemset(d_out, 0, total_bytes));

    INFO("block_threads " << block_threads);
    tile_memcpy_async_kernel<kTile>
        <<<1, block_threads, total_bytes>>>(d_out, d_in, per_tile_bytes);
    HIP_CHECK(hipGetLastError());
    HIP_CHECK(hipMemcpy(out.data(), d_out, total_bytes, hipMemcpyDeviceToHost));
    HIP_CHECK(hipFree(d_in));
    HIP_CHECK(hipFree(d_out));

    for (size_t i = 0; i < total_elems; i++) {
      INFO("idx " << i);
      REQUIRE(out[i] == Catch::Approx(in[i] * 2.0f));
    }
  }
}

HIP_TEST_CASE(Unit_coop_memcpy_async_coalesced_group_Basic) {
  hipDeviceProp_t prop;
  HIP_CHECK(hipGetDeviceProperties(&prop, 0));
  const unsigned int wave = static_cast<unsigned int>(prop.warpSize);
  constexpr size_t chunk_elems = 64;

  // Size the block in whole waves so the number of coalesced groups is known and each one owns a
  // chunk, on both wave32 and wave64.
  for (const unsigned int waves : {1u, 2u, 4u}) {
    const unsigned int threads = wave * waves;
    if (threads > static_cast<unsigned int>(prop.maxThreadsPerBlock)) continue;

    const size_t total_elems = chunk_elems * waves;
    const size_t bytes = total_elems * sizeof(float);

    float *d_in, *d_out;
    HIP_CHECK(hipMalloc(&d_in, bytes));
    HIP_CHECK(hipMalloc(&d_out, bytes));

    std::vector<float> in(total_elems), out(total_elems, 0.0f);
    for (size_t i = 0; i < total_elems; i++) {
      in[i] = static_cast<float>(i * 2 + 1);
    }
    HIP_CHECK(hipMemcpy(d_in, in.data(), bytes, hipMemcpyHostToDevice));
    HIP_CHECK(hipMemset(d_out, 0, bytes));

    INFO("waves " << waves << " threads " << threads);
    coalesced_memcpy_async_kernel<<<1, threads, bytes>>>(d_out, d_in, chunk_elems);
    HIP_CHECK(hipGetLastError());
    HIP_CHECK(hipMemcpy(out.data(), d_out, bytes, hipMemcpyDeviceToHost));
    HIP_CHECK(hipFree(d_in));
    HIP_CHECK(hipFree(d_out));

    for (size_t i = 0; i < total_elems; i++) {
      INFO("idx " << i);
      REQUIRE(out[i] == Catch::Approx(in[i] + 1.0f));
    }
  }
}

HIP_TEST_CASE(Unit_coop_memcpy_async_LayoutMin) {
  // Try all three orderings: dst<src, dst==src, dst>src.
  for (const auto& [dst_count, src_count] :
       std::vector<std::pair<size_t, size_t>>{{32, 64}, {64, 64}, {128, 64}}) {
    const size_t copied = dst_count < src_count ? dst_count : src_count;
    const size_t alloc_elems = std::max(dst_count, src_count);
    const size_t smem_bytes = alloc_elems * sizeof(float);

    float *d_in, *d_out;
    HIP_CHECK(hipMalloc(&d_in, alloc_elems * sizeof(float)));
    HIP_CHECK(hipMalloc(&d_out, alloc_elems * sizeof(float)));

    std::vector<float> in(alloc_elems), out(alloc_elems, -1.0f);
    for (size_t i = 0; i < alloc_elems; i++) in[i] = static_cast<float>(i + 1);
    HIP_CHECK(hipMemcpy(d_in, in.data(), alloc_elems * sizeof(float), hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(d_out, out.data(), alloc_elems * sizeof(float), hipMemcpyHostToDevice));

    const unsigned int threads = 64;
    INFO("dst_count " << dst_count << " src_count " << src_count << " expected_copied " << copied);
    layout_min_kernel<<<1, threads, smem_bytes>>>(d_out, d_in, dst_count, src_count);
    HIP_CHECK(hipGetLastError());
    HIP_CHECK(hipMemcpy(out.data(), d_out, alloc_elems * sizeof(float), hipMemcpyDeviceToHost));
    HIP_CHECK(hipFree(d_in));
    HIP_CHECK(hipFree(d_out));

    for (size_t i = 0; i < copied; i++) {
      INFO("copied region idx " << i);
      REQUIRE(out[i] == Catch::Approx(in[i]));
    }
    // Beyond the copied region, the tail of `out` was not overwritten by the kernel - it should
    // still hold the sentinel from the initial host copy.
    for (size_t i = copied; i < alloc_elems; i++) {
      INFO("untouched tail idx " << i);
      REQUIRE(out[i] == Catch::Approx(-1.0f));
    }
  }
}

HIP_TEST_CASE(Unit_coop_memcpy_async_ZeroCount) {
  constexpr size_t size = 32;
  const size_t smem_bytes = size * sizeof(float);

  float *d_in, *d_out;
  HIP_CHECK(hipMalloc(&d_in, smem_bytes));
  HIP_CHECK(hipMalloc(&d_out, 2 * smem_bytes));

  std::vector<float> in(size), out(2 * size, 0.0f);
  for (size_t i = 0; i < size; i++) in[i] = static_cast<float>(i + 100);
  HIP_CHECK(hipMemcpy(d_in, in.data(), smem_bytes, hipMemcpyHostToDevice));

  zero_count_kernel<<<1, size, smem_bytes>>>(d_out, d_in, size);
  HIP_CHECK(hipGetLastError());
  HIP_CHECK(hipMemcpy(out.data(), d_out, 2 * smem_bytes, hipMemcpyDeviceToHost));
  HIP_CHECK(hipFree(d_in));
  HIP_CHECK(hipFree(d_out));

  for (size_t i = 0; i < size; i++) {
    INFO("zero-count idx " << i);
    REQUIRE(out[i] == Catch::Approx(kSentinel));
  }
  for (size_t i = 0; i < size; i++) {
    INFO("post-zero real-copy idx " << i);
    REQUIRE(out[size + i] == Catch::Approx(in[i]));
  }
}

HIP_TEST_CASE(Unit_coop_memcpy_async_Unaligned) {
  // Byte-granularity sizes including non-multiples of 16/8/4 and sizes smaller than the group,
  // plus sizes with a tail beyond the per-thread share.
  for (const size_t bytes : {1u, 3u, 7u, 15u, 31u, 33u, 65u, 129u}) {
    unsigned char *d_in, *d_out;
    HIP_CHECK(hipMalloc(&d_in, bytes));
    HIP_CHECK(hipMalloc(&d_out, bytes));

    std::vector<unsigned char> in(bytes), out(bytes, 0);
    for (size_t i = 0; i < bytes; i++) in[i] = static_cast<unsigned char>(i + 1);
    HIP_CHECK(hipMemcpy(d_in, in.data(), bytes, hipMemcpyHostToDevice));

    INFO("bytes " << bytes);
    const unsigned int threads = 32;
    unaligned_size_kernel<<<1, threads, bytes + 16>>>(reinterpret_cast<float*>(d_out),
                                                      reinterpret_cast<float*>(d_in), bytes);
    HIP_CHECK(hipGetLastError());
    HIP_CHECK(hipMemcpy(out.data(), d_out, bytes, hipMemcpyDeviceToHost));
    HIP_CHECK(hipFree(d_in));
    HIP_CHECK(hipFree(d_out));

    for (size_t i = 0; i < bytes; i++) {
      INFO("byte idx " << i);
      REQUIRE(static_cast<int>(out[i]) == static_cast<int>(in[i]));
    }
  }
}

// Regression: group.sync() must wait for an in-flight async copy.
HIP_TEST_CASE(Unit_coop_memcpy_async_SyncPublishesCopy) {
  for (const unsigned int threads : {32u, 64u, 256u}) {
    const size_t bytes = threads * sizeof(float);

    float *d_in, *d_out;
    HIP_CHECK(hipMalloc(&d_in, bytes));
    HIP_CHECK(hipMalloc(&d_out, bytes));

    std::vector<float> in(threads), out(threads, 0.0f);
    for (size_t i = 0; i < threads; i++) in[i] = static_cast<float>(i + 100);
    HIP_CHECK(hipMemcpy(d_in, in.data(), bytes, hipMemcpyHostToDevice));
    HIP_CHECK(hipMemset(d_out, 0, bytes));

    INFO("threads " << threads);
    sync_publishes_block_kernel<<<1, threads, bytes>>>(d_out, d_in, bytes);
    HIP_CHECK(hipGetLastError());
    HIP_CHECK(hipMemcpy(out.data(), d_out, bytes, hipMemcpyDeviceToHost));
    HIP_CHECK(hipFree(d_in));
    HIP_CHECK(hipFree(d_out));

    for (size_t i = 0; i < threads; i++) {
      INFO("idx " << i << " (sentinel " << kSentinel << " means the copy had not landed)");
      REQUIRE(out[i] == Catch::Approx(in[i]));
    }
  }
}

// Same regression, on a tile whose sync() is a wavefront fence rather than a barrier.
HIP_TEST_CASE(Unit_coop_memcpy_async_SyncPublishesCopy_Tile) {
  constexpr unsigned int kTile = 32;
  const unsigned int threads = kTile;
  const size_t bytes = threads * sizeof(float);

  float *d_in, *d_out;
  HIP_CHECK(hipMalloc(&d_in, bytes));
  HIP_CHECK(hipMalloc(&d_out, bytes));

  std::vector<float> in(threads), out(threads, 0.0f);
  for (size_t i = 0; i < threads; i++) in[i] = static_cast<float>(i + 100);
  HIP_CHECK(hipMemcpy(d_in, in.data(), bytes, hipMemcpyHostToDevice));
  HIP_CHECK(hipMemset(d_out, 0, bytes));

  sync_publishes_tile_kernel<kTile><<<1, threads, bytes>>>(d_out, d_in, bytes);
  HIP_CHECK(hipGetLastError());
  HIP_CHECK(hipMemcpy(out.data(), d_out, bytes, hipMemcpyDeviceToHost));
  HIP_CHECK(hipFree(d_in));
  HIP_CHECK(hipFree(d_out));

  for (size_t i = 0; i < threads; i++) {
    INFO("idx " << i << " (sentinel " << kSentinel << " means the copy had not landed)");
    REQUIRE(out[i] == Catch::Approx(in[i]));
  }
}

// Regression: the wait has to happen before the barrier, so that data copied by one wave is
// visible to every other wave once the barrier releases.
HIP_TEST_CASE(Unit_coop_memcpy_async_CrossWaveVisibility) {
  hipDeviceProp_t prop;
  HIP_CHECK(hipGetDeviceProperties(&prop, 0));

  for (const unsigned int threads : {128u, 256u}) {
    if (threads > static_cast<unsigned int>(prop.maxThreadsPerBlock)) continue;
    const size_t bytes = threads * sizeof(float);

    float *d_in, *d_out;
    HIP_CHECK(hipMalloc(&d_in, bytes));
    HIP_CHECK(hipMalloc(&d_out, bytes));

    std::vector<float> in(threads), out(threads, 0.0f);
    for (size_t i = 0; i < threads; i++) in[i] = static_cast<float>(i + 100);
    HIP_CHECK(hipMemcpy(d_in, in.data(), bytes, hipMemcpyHostToDevice));
    HIP_CHECK(hipMemset(d_out, 0, bytes));

    INFO("threads " << threads);
    cross_wave_visibility_kernel<<<1, threads, bytes>>>(d_out, d_in, bytes);
    HIP_CHECK(hipGetLastError());
    HIP_CHECK(hipMemcpy(out.data(), d_out, bytes, hipMemcpyDeviceToHost));
    HIP_CHECK(hipFree(d_in));
    HIP_CHECK(hipFree(d_out));

    for (size_t i = 0; i < threads; i++) {
      INFO("idx " << i);
      REQUIRE(out[i] == Catch::Approx(in[threads - 1 - i]));
    }
  }
}

// A copy that is not global<->LDS must still move the data.
HIP_TEST_CASE(Unit_coop_memcpy_async_GlobalToGlobal) {
  for (const size_t elems : {32u, 129u}) {
    const size_t bytes = elems * sizeof(float);

    float *d_in, *d_out;
    HIP_CHECK(hipMalloc(&d_in, bytes));
    HIP_CHECK(hipMalloc(&d_out, bytes));

    std::vector<float> in(elems), out(elems, 0.0f);
    for (size_t i = 0; i < elems; i++) in[i] = static_cast<float>(i + 1);
    HIP_CHECK(hipMemcpy(d_in, in.data(), bytes, hipMemcpyHostToDevice));
    HIP_CHECK(hipMemset(d_out, 0, bytes));

    INFO("elems " << elems);
    global_to_global_kernel<<<1, 32>>>(d_out, d_in, bytes);
    HIP_CHECK(hipGetLastError());
    HIP_CHECK(hipMemcpy(out.data(), d_out, bytes, hipMemcpyDeviceToHost));
    HIP_CHECK(hipFree(d_in));
    HIP_CHECK(hipFree(d_out));

    for (size_t i = 0; i < elems; i++) {
      INFO("idx " << i);
      REQUIRE(out[i] == Catch::Approx(in[i]));
    }
  }
}
