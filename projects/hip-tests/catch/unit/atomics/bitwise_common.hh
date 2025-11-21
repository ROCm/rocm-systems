/*
Copyright (c) 2022 Advanced Micro Devices, Inc. All rights reserved.

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

#pragma once

#include <cmd_options.hh>
#include <hip_test_common.hh>
#include <hip/hip_cooperative_groups.h>
#include <resource_guards.hh>

namespace cg = cooperative_groups;

namespace Bitwise {
enum class AtomicOperation {
  kAnd = 0,
  kAndSystem,
  kOr,
  kOrSystem,
  kXor,
  kXorSystem,
  kBuiltinAnd,
  kBuiltinOr,
  kBuiltinXor
};

constexpr auto kMask = 0xAAAA;
constexpr auto kTestValue = 0x4545;
constexpr auto kAndTestValue = 0xFFFF;

template <typename TestType, AtomicOperation operation>
__host__ __device__ TestType GetTestValue() {
  if constexpr (operation == AtomicOperation::kAnd || operation == AtomicOperation::kAndSystem) {
    return kAndTestValue;
  }

  return kTestValue;
}

template <typename TestType, AtomicOperation operation, int memory_scope = __HIP_MEMORY_SCOPE_AGENT>
__device__ TestType PerformAtomicOperation(TestType* const mem) {
  const auto mask = kMask;

  if constexpr (operation == AtomicOperation::kAnd) {
    return atomicAnd(mem, mask);
  } else if constexpr (operation == AtomicOperation::kAndSystem) {
    return atomicAnd_system(mem, mask);
  } else if constexpr (operation == AtomicOperation::kOr) {
    return atomicOr(mem, mask);
  } else if constexpr (operation == AtomicOperation::kOrSystem) {
    return atomicOr_system(mem, mask);
  } else if constexpr (operation == AtomicOperation::kXor) {
    return atomicXor(mem, mask);
  } else if constexpr (operation == AtomicOperation::kXorSystem) {
    return atomicXor_system(mem, mask);
  } else if constexpr (operation == AtomicOperation::kBuiltinAnd) {
    return __hip_atomic_fetch_and(mem, mask, __ATOMIC_RELAXED, memory_scope);
  } else if constexpr (operation == AtomicOperation::kBuiltinOr) {
    return __hip_atomic_fetch_or(mem, mask, __ATOMIC_RELAXED, memory_scope);
  } else if constexpr (operation == AtomicOperation::kBuiltinXor) {
    return __hip_atomic_fetch_xor(mem, mask, __ATOMIC_RELAXED, memory_scope);
  }
}

template <typename TestType, AtomicOperation operation, bool use_shared_mem,
          int memory_scope = __HIP_MEMORY_SCOPE_AGENT>
__global__ void TestKernel(TestType* const global_mem, TestType* const old_vals) {
  __shared__ TestType shared_mem;

  const auto tid = cg::this_grid().thread_rank();

  TestType* const mem = use_shared_mem ? &shared_mem : global_mem;

  if constexpr (use_shared_mem) {
    if (tid == 0) mem[0] = global_mem[0];
    __syncthreads();
  }

  old_vals[tid] = PerformAtomicOperation<TestType, operation, memory_scope>(mem);

  if constexpr (use_shared_mem) {
    __syncthreads();
    if (tid == 0) global_mem[0] = mem[0];
  }
}

template <typename TestType>
__host__ __device__ TestType* PitchedOffset(TestType* const ptr, const unsigned int pitch,
                                            const unsigned int idx) {
  const auto byte_ptr = reinterpret_cast<uint8_t*>(ptr);
  return reinterpret_cast<TestType*>(byte_ptr + idx * pitch);
}

__device__ void GenerateMemoryTraffic(uint8_t* const begin_addr, uint8_t* const end_addr) {
  for (volatile uint8_t* addr = begin_addr; addr != end_addr; ++addr) {
    uint8_t val = *addr;
    val ^= 0xAB;
    *addr = val;
  }
}

template <typename TestType, AtomicOperation operation, bool use_shared_mem,
          int memory_scope = __HIP_MEMORY_SCOPE_AGENT>
__global__ void TestKernel(TestType* const global_mem, TestType* const old_vals,
                           const unsigned int width, const unsigned pitch) {
  extern __shared__ uint8_t shared_mem[];

  const auto tid = cg::this_grid().thread_rank();

  TestType* const mem = use_shared_mem ? reinterpret_cast<TestType*>(shared_mem) : global_mem;

  if constexpr (use_shared_mem) {
    if (tid < width) {
      const auto target = PitchedOffset(mem, pitch, tid);
      *target = *PitchedOffset(global_mem, pitch, tid);
    };
    __syncthreads();
  }

  const auto n = cooperative_groups::this_grid().size();

  TestType* atomic_addr = PitchedOffset(mem, pitch, tid % width);

  if (tid < n) {
    old_vals[tid] = PerformAtomicOperation<TestType, operation, memory_scope>(
        PitchedOffset(mem, pitch, tid % width));
  } else {
    uint8_t* const begin_addr = reinterpret_cast<uint8_t*>(atomic_addr + 1);
    uint8_t* const end_addr = reinterpret_cast<uint8_t*>(atomic_addr) + pitch;
    GenerateMemoryTraffic(begin_addr, end_addr);
  }

  if constexpr (use_shared_mem) {
    __syncthreads();
    if (tid < width) {
      const auto target = PitchedOffset(global_mem, pitch, tid);
      *target = *PitchedOffset(mem, pitch, tid);
    };
  }
}

struct TestParams {
  auto ThreadCount() const {
    return blocks.x * blocks.y * blocks.z * threads.x * threads.y * threads.z;
  }

  dim3 blocks;
  dim3 threads;
  unsigned int num_devices = 1u;
  unsigned int kernel_count = 1u;
  unsigned int width = 1u;
  unsigned int pitch = 0u;
  unsigned int host_thread_count = 0u;
  LinearAllocs alloc_type;
};

template <typename TestType, AtomicOperation operation>
std::tuple<std::vector<TestType>, std::vector<TestType>> TestKernelHostRef(const TestParams& p) {
  const auto thread_count_per_kernel = p.ThreadCount();
  const auto thread_count = p.num_devices * p.kernel_count * p.ThreadCount();

  TestType test_value = GetTestValue<TestType, operation>();
  const auto mask = kMask;
  std::vector<TestType> res_vals(p.num_devices * p.width, test_value);
  std::vector<TestType> old_vals;
  old_vals.reserve(thread_count);

  for (auto i = 0u; i < p.num_devices; ++i) {
    for (auto j = 0u; j < p.kernel_count; ++j) {
      for (auto tid = 0u; tid < thread_count_per_kernel; ++tid) {
        auto& res = res_vals[tid % p.width + (i * p.width)];
        old_vals.push_back(res);

        if constexpr (operation == AtomicOperation::kAnd || operation == AtomicOperation::kAndSystem ||
                      operation == AtomicOperation::kBuiltinAnd) {
          res = res & mask;
        } else if constexpr (operation == AtomicOperation::kOr ||
                             operation == AtomicOperation::kOrSystem ||
                             operation == AtomicOperation::kBuiltinOr) {
          res = res | mask;
        } else if constexpr (operation == AtomicOperation::kXor ||
                             operation == AtomicOperation::kXorSystem ||
                             operation == AtomicOperation::kBuiltinXor) {
          res = res ^ mask;
        }
      }
    }
  }

  return {res_vals, old_vals};
}

template <typename TestType, AtomicOperation operation>
void Verify(const TestParams& p, std::vector<TestType>& res_vals, std::vector<TestType>& old_vals) {
  auto [expected_res_vals, expected_old_vals] = TestKernelHostRef<TestType, operation>(p);

  for (auto i = 0u; i < res_vals.size(); ++i) {
    INFO("Results index: " << i);
    REQUIRE(expected_res_vals[i] == res_vals[i]);
  }

  std::sort(begin(old_vals), end(old_vals));
  std::sort(begin(expected_old_vals), end(expected_old_vals));
  for (auto i = 0u; i < old_vals.size(); ++i) {
    INFO("Old values index: " << i);
    REQUIRE(expected_old_vals[i] == old_vals[i]);
  }
}

template <typename TestType, AtomicOperation operation, bool use_shared_mem,
          int memory_scope = __HIP_MEMORY_SCOPE_AGENT>
void LaunchKernel(const TestParams& p, hipStream_t stream, TestType* const mem_ptr,
                  TestType* const old_vals) {
  const auto shared_mem_size = use_shared_mem ? p.width * p.pitch : 0u;
  if (p.width == 1 && p.pitch == sizeof(TestType))
    TestKernel<TestType, operation, use_shared_mem, memory_scope>
        <<<p.blocks, p.threads, shared_mem_size, stream>>>(mem_ptr, old_vals);
  else
    TestKernel<TestType, operation, use_shared_mem, memory_scope>
        <<<p.blocks, p.threads, shared_mem_size, stream>>>(mem_ptr, old_vals, p.width, p.pitch);
}

template <typename TestType, AtomicOperation operation, bool use_shared_mem,
          int memory_scope = __HIP_MEMORY_SCOPE_AGENT>
void TestCore(const TestParams& p) {
  const auto old_vals_alloc_size = p.kernel_count * p.ThreadCount() * sizeof(TestType);
  const auto mem_alloc_size = p.width * p.pitch;
  // Device Memory
  std::vector<LinearAllocGuard<TestType>> old_vals_devs;
  std::vector<LinearAllocGuard<TestType>> mem_devs;
  std::vector<StreamGuard> streams;
  for (auto i = 0; i < p.num_devices; ++i) {
    HIP_CHECK(hipSetDevice(i));
    old_vals_devs.emplace_back(LinearAllocs::hipMalloc, old_vals_alloc_size);
    for (auto j = 0; j < p.kernel_count; ++j) {
      streams.emplace_back(Streams::created);
    }
    mem_devs.emplace_back(p.alloc_type, mem_alloc_size);
  }

  // Host Memory
  std::vector<TestType> old_vals(p.num_devices * p.kernel_count * p.ThreadCount());
  std::vector<TestType> res_vals(p.num_devices * p.width );

  // Test Values on Device
  TestType test_value = GetTestValue<TestType, operation>();
  for (auto i = 0u; i < p.num_devices; ++i) {
    TestType* const mem_ptr =
        p.alloc_type == LinearAllocs::hipMalloc ? mem_devs[i].ptr() : mem_devs[i].host_ptr();
    HIP_CHECK(hipMemset(mem_ptr, 0, mem_alloc_size));
    for (int i = 0; i < p.width * p.pitch / sizeof(TestType); ++i) {
      HIP_CHECK(hipMemcpy(&mem_ptr[i], &test_value, sizeof(TestType), hipMemcpyHostToDevice));
    }
  }
  // Launch Kernel and get back old vals
  for (auto i = 0u; i < p.num_devices; ++i) {
    HIP_CHECK(hipSetDevice(i));
    for (auto j = 0u; j < p.kernel_count; ++j) {
      const auto& stream = streams[i * p.kernel_count + j].stream();
      const auto old_vals = old_vals_devs[i].ptr() + j * p.ThreadCount();
      LaunchKernel<TestType, operation, use_shared_mem, memory_scope>(p, stream,
          mem_devs[i].ptr(), old_vals);
    }
  }
  for (auto i = 0; i < p.num_devices; ++i) {
    HIP_CHECK(hipSetDevice(i));
    HIP_CHECK(hipDeviceSynchronize());
  }
  // Copy results back to Host
  for (auto i = 0u; i < p.num_devices; ++i) {
    const auto device_offset = i * p.kernel_count * p.ThreadCount();
    HIP_CHECK(hipMemcpy(old_vals.data() + device_offset, old_vals_devs[i].ptr(),
                        old_vals_alloc_size, hipMemcpyDeviceToHost));
    HIP_CHECK(hipMemcpy2D(res_vals.data() + i*p.width, sizeof(TestType), mem_devs[i].ptr(), p.pitch, sizeof(TestType),
                       p.width, hipMemcpyDeviceToHost));
  }

  Verify<TestType, operation>(p, res_vals, old_vals);
}

inline dim3 GenerateThreadDimensions() { return dim3(1024); }

inline dim3 GenerateBlockDimensions() {
  return dim3(8);
}

template <typename TestType, AtomicOperation operation, int memory_scope = __HIP_MEMORY_SCOPE_AGENT>
void SingleDeviceSingleKernelTest(const unsigned int width, const unsigned int pitch) {
  TestParams params;
  params.num_devices = 1;
  params.kernel_count = 1;
  if constexpr ((operation == AtomicOperation::kBuiltinAnd ||
                 operation == AtomicOperation::kBuiltinOr ||
                 operation == AtomicOperation::kBuiltinXor) &&
                memory_scope == __HIP_MEMORY_SCOPE_SINGLETHREAD) {
    params.threads = 1;
  } else if constexpr ((operation == AtomicOperation::kBuiltinAnd ||
                        operation == AtomicOperation::kBuiltinOr ||
                        operation == AtomicOperation::kBuiltinXor) &&
                       memory_scope == __HIP_MEMORY_SCOPE_WAVEFRONT) {
    int warp_size = 0;
    HIP_CHECK(hipDeviceGetAttribute(&warp_size, hipDeviceAttributeWarpSize, 0));
    params.threads = dim3(warp_size);
  } else {
    params.threads = GenerateThreadDimensions();
  }
  params.width = width;
  params.pitch = pitch;

  SECTION("Global memory") {
    if constexpr ((operation == AtomicOperation::kBuiltinAnd ||
                   operation == AtomicOperation::kBuiltinOr ||
                   operation == AtomicOperation::kBuiltinXor) &&
                  (memory_scope == __HIP_MEMORY_SCOPE_SINGLETHREAD ||
                   memory_scope == __HIP_MEMORY_SCOPE_WAVEFRONT ||
                   memory_scope == __HIP_MEMORY_SCOPE_WORKGROUP)) {
      params.blocks = dim3(1);
    } else {
      params.blocks = GenerateBlockDimensions();
    }
    using LA = LinearAllocs;
    for (const auto alloc_type :
         {LA::hipMalloc}) {
      params.alloc_type = alloc_type;
      DYNAMIC_SECTION("Allocation type: " << to_string(alloc_type)) {
        TestCore<TestType, operation, false>(params);
      }
    }
  }
#ifdef __linux__
  SECTION("Shared memory") {
    params.blocks = dim3(1);
    params.alloc_type = LinearAllocs::hipMalloc;
    TestCore<TestType, operation, true>(params);
  }
#endif
}

template <typename TestType, AtomicOperation operation>
void SingleDeviceMultipleKernelTest(const unsigned int kernel_count, const unsigned int width,
                                    const unsigned int pitch) {
  int concurrent_kernels = 0;
  HIP_CHECK(hipDeviceGetAttribute(&concurrent_kernels, hipDeviceAttributeConcurrentKernels, 0));
  if (!concurrent_kernels) {
    HipTest::HIP_SKIP_TEST("Test requires support for concurrent kernel execution");
    return;
  }

  TestParams params;
  params.num_devices = 1;
  params.kernel_count = kernel_count;
  params.blocks = GenerateBlockDimensions();
  params.threads = GenerateThreadDimensions();
  params.width = width;
  params.pitch = pitch;

  using LA = LinearAllocs;
  for (const auto alloc_type :
       {LA::hipMalloc}) {
    params.alloc_type = alloc_type;
    DYNAMIC_SECTION("Allocation type: " << to_string(alloc_type)) {
      TestCore<TestType, operation, false>(params);
    }
  }
}

template <typename TestType, AtomicOperation operation>
void MultipleDeviceMultipleKernelTest(const unsigned int num_devices,
                                      const unsigned int kernel_count, const unsigned int width,
                                      const unsigned int pitch) {
  if (num_devices > 1) {
    if (HipTest::getDeviceCount() < num_devices) {
      std::string msg = std::to_string(num_devices) + " devices are required";
      HipTest::HIP_SKIP_TEST(msg.c_str());
      return;
    }
  }

  if (!HipTest::checkConcurrentKernels(num_devices)) {
    HipTest::HIP_SKIP_TEST("Test requires support for concurrent kernel execution");
    return;
  }

  TestParams params;
  params.num_devices = num_devices;
  params.kernel_count = kernel_count;
  params.blocks = GenerateBlockDimensions();
  params.threads = GenerateThreadDimensions();
  params.width = width;
  params.pitch = pitch;

  using LA = LinearAllocs;
  for (const auto alloc_type : {LA::hipMalloc, LA::hipHostMalloc}) {
    params.alloc_type = alloc_type;
    DYNAMIC_SECTION("Allocation type: " << to_string(alloc_type)) {
      TestCore<TestType, operation, false, __HIP_MEMORY_SCOPE_SYSTEM>(params);
    }
  }
}

}  // namespace Bitwise
