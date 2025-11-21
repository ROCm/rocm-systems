/*
Copyright (c) 2023 Advanced Micro Devices, Inc. All rights reserved.

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

#include <numeric>

#include <cmd_options.hh>
#include <hip_test_common.hh>
#include <resource_guards.hh>
#include <hip/hip_cooperative_groups.h>

enum class AtomicScopes { device, system, builtin };

template <typename T, AtomicScopes scope, int memory_scope = __HIP_MEMORY_SCOPE_AGENT>
__device__ T perform_atomic_exch(T* address, T val) {
  if constexpr (scope == AtomicScopes::device) {
    return atomicExch(address, val);
  } else if (scope == AtomicScopes::system) {
    return atomicExch_system(address, val);
  } else if (scope == AtomicScopes::builtin) {
    return __hip_atomic_exchange(address, val, __ATOMIC_RELAXED, memory_scope);
  }
}

template <typename T, bool use_shared_mem, AtomicScopes scope,
          int memory_scope = __HIP_MEMORY_SCOPE_AGENT>
__global__ void atomic_exch_kernel_compile_time(T* const global_mem, T* const old_vals) {
  __shared__ T shared_mem;

  const auto tid = cooperative_groups::this_grid().thread_rank();

  T* const mem = use_shared_mem ? &shared_mem : global_mem;

  if constexpr (use_shared_mem) {
    if (tid == 0) mem[0] = global_mem[0];
    __syncthreads();
  }

  old_vals[tid] = perform_atomic_exch<T, scope, memory_scope>(mem, static_cast<T>(tid + 1));

  if constexpr (use_shared_mem) {
    __syncthreads();
    if (tid == 0) global_mem[0] = mem[0];
  }
}

template <typename T>
__host__ __device__ T* pitched_offset(T* const ptr, const unsigned int pitch,
                                      const unsigned int idx) {
  const auto byte_ptr = reinterpret_cast<uint8_t*>(ptr);
  return reinterpret_cast<T*>(byte_ptr + idx * pitch);
}

__device__ void generate_memory_traffic(uint8_t* const begin_addr, uint8_t* const end_addr) {
  for (volatile uint8_t* addr = begin_addr; addr != end_addr; ++addr) {
    uint8_t val = *addr;
    val ^= 0xAB;
    *addr = val;
  }
}

template <typename T, bool use_shared_mem, AtomicScopes scope,
          int memory_scope = __HIP_MEMORY_SCOPE_AGENT>
__global__ void atomic_exch_kernel(T* const global_mem, T* const old_vals, const unsigned int width,
                                   const unsigned pitch, const T base_val = 0) {
  extern __shared__ uint8_t shared_mem[];

  const auto tid = cooperative_groups::this_grid().thread_rank();

  T* const mem = use_shared_mem ? reinterpret_cast<T*>(shared_mem) : global_mem;

  if constexpr (use_shared_mem) {
    if (tid < width) {
      const auto target = pitched_offset(mem, pitch, tid);
      *target = *pitched_offset(global_mem, pitch, tid);
    };
    __syncthreads();
  }

  const auto n = cooperative_groups::this_grid().size();

  T* atomic_addr = pitched_offset(mem, pitch, tid % width);

  if (tid < n) {
    old_vals[tid] = perform_atomic_exch<T, scope, memory_scope>(
        pitched_offset(mem, pitch, tid % width), base_val + static_cast<T>(tid + width));
  } else {
    uint8_t* const begin_addr = reinterpret_cast<uint8_t*>(atomic_addr + 1);
    uint8_t* const end_addr = reinterpret_cast<uint8_t*>(atomic_addr) + pitch;
    generate_memory_traffic(begin_addr, end_addr);
  }

  if constexpr (use_shared_mem) {
    __syncthreads();
    if (tid < width) {
      const auto target = pitched_offset(global_mem, pitch, tid);
      *target = *pitched_offset(mem, pitch, tid);
    };
  }
}


template <typename TestType, bool use_shared_mem, AtomicScopes scope>
void AtomicExchSameAddress(const dim3 blocks, const dim3 threads, const LinearAllocs alloc_type) {
  LinearAllocGuard<TestType> mem_dev(alloc_type, sizeof(TestType));

  const auto thread_count = blocks.x * blocks.y * blocks.z * threads.x * threads.y * threads.z;
  const auto old_vals_alloc_size = thread_count * sizeof(TestType);
  LinearAllocGuard<TestType> old_vals_dev(LinearAllocs::hipMalloc, old_vals_alloc_size);
  std::vector<TestType> old_vals(thread_count + 1);


  HIP_CHECK(hipMemset(mem_dev.ptr(), 0, sizeof(TestType)));
  atomic_exch_kernel_compile_time<TestType, use_shared_mem, scope>
      <<<blocks, threads>>>(mem_dev.ptr(), old_vals_dev.ptr());
  HIP_CHECK(
      hipMemcpy(old_vals.data(), old_vals_dev.ptr(), old_vals_alloc_size, hipMemcpyDeviceToHost));
  HIP_CHECK(hipMemcpy(old_vals.data() + thread_count, mem_dev.ptr(), sizeof(TestType),
                      hipMemcpyDeviceToHost));
  HIP_CHECK(hipDeviceSynchronize());

  // Every thread will exchange its grid-wide linear id into a target location within mem_dev,
  // receiving back the value previously present therein. This previous value is written to
  // old_vals_dev.
  // old_vals_dev will not contain values that the final scheduled warp exchanged into mem_dev, but
  // mem_dev obviously will.
  // Given that mem_dev initially contains values in the range [0, width) and that the maximum value
  // the final thread shall write is thread_count + width - 1, presuming correct operation of
  // atomicExch, the union of mem_dev and old_vals_dev shall contain values in the range
  //[0, thread_count + width)
  std::sort(old_vals.begin(), old_vals.end());
  for (auto i = 0u; i < old_vals.size(); ++i) {
    REQUIRE(i == old_vals[i]);
  }
}

template <typename TestType, AtomicScopes scope> void AtomicExchSameAddressTest() {
  const auto threads = GENERATE(dim3(1024), dim3(1023), dim3(511), dim3(17), dim3(31));

  SECTION("Global memory") {
    const auto blocks = GENERATE(dim3(20));
    using LA = LinearAllocs;
    const auto allocation_type =
        GENERATE(LA::hipMalloc, LA::hipHostMalloc, LA::hipMallocManaged, LA::mallocAndRegister);
    AtomicExchSameAddress<TestType, false, AtomicScopes::device>(blocks, threads, allocation_type);
  }

  SECTION("Shared memory") {
    const auto blocks = dim3(1);
    AtomicExchSameAddress<TestType, true, AtomicScopes::device>(blocks, threads,
                                                                LinearAllocs::hipMalloc);
  }
}

struct AtomicExchParams {
  dim3 blocks;
  dim3 threads;
  unsigned int num_devices = 1u;
  unsigned int kernel_count = 1u;
  unsigned int width = 1u;
  unsigned int pitch = 0u;
  unsigned int host_thread_count = 0u;
  LinearAllocs alloc_type;
};


template <typename Derived, typename T, bool use_shared_mem, AtomicScopes scope>
class AtomicExchCRTP {
 public:
  void run(const AtomicExchParams& p) const {
    const auto thread_count =
        p.blocks.x * p.blocks.y * p.blocks.z * p.threads.x * p.threads.y * p.threads.z;

    const auto old_vals_alloc_size = p.kernel_count * thread_count * sizeof(T);
    std::vector<LinearAllocGuard<T>> old_vals_devs;
    std::vector<StreamGuard> streams;
    for (auto i = 0; i < p.num_devices; ++i) {
      HIP_CHECK(hipSetDevice(i));
      old_vals_devs.emplace_back(LinearAllocs::hipMalloc, old_vals_alloc_size);
      for (auto j = 0; j < p.kernel_count; ++j) {
        streams.emplace_back(Streams::created);
      }
    }

    const auto mem_alloc_size = p.width * p.pitch;
    LinearAllocGuard<T> mem_dev(p.alloc_type, mem_alloc_size);

    const auto host_iters_per_thread =
        std::max(p.num_devices * p.kernel_count * thread_count / 20, p.width);

    std::vector<T> old_vals(p.num_devices * p.kernel_count * thread_count + p.width +
                            p.host_thread_count * host_iters_per_thread);
    std::iota(old_vals.begin(), old_vals.begin() + p.width, 0);

    HIP_CHECK(hipMemcpy2D(mem_dev.ptr(), p.pitch, old_vals.data(), sizeof(T), sizeof(T), p.width,
                          hipMemcpyHostToDevice));

    const auto shared_mem_size = use_shared_mem ? mem_alloc_size : 0u;
    for (auto i = 0u; i < p.num_devices; ++i) {
      HIP_CHECK(hipSetDevice(i));
      const auto device_offset = i * p.kernel_count * thread_count;
      for (auto j = 0u; j < p.kernel_count; ++j) {
        const auto& stream = streams[i * p.kernel_count + j].stream();
        const auto kern_offset = j * thread_count;
        const auto old_vals = old_vals_devs[i].ptr() + kern_offset;
        CastToDerived().LaunchKernel(shared_mem_size, stream, mem_dev.ptr(), old_vals,
                                     device_offset + kern_offset, p);
      }
    }

    PerformHostAtomicExchange(p.host_thread_count, host_iters_per_thread, mem_dev.host_ptr(),
                              old_vals.data(), p);
    for (auto i = 0; i < p.num_devices; ++i) {
      HIP_CHECK(hipSetDevice(i));
      HIP_CHECK(hipDeviceSynchronize());
    }
    for (auto i = 0u; i < p.num_devices; ++i) {
      const auto device_offset = i * p.kernel_count * thread_count;
      HIP_CHECK(hipMemcpy(old_vals.data() + device_offset, old_vals_devs[i].ptr(),
                          old_vals_alloc_size, hipMemcpyDeviceToHost));
    }
    HIP_CHECK(hipMemcpy2D(old_vals.data() + p.num_devices * p.kernel_count * thread_count,
                          sizeof(T), mem_dev.ptr(), p.pitch, sizeof(T), p.width,
                          hipMemcpyDeviceToHost));

    CastToDerived().ValidateResults(old_vals);
  }

 private:
  const Derived& CastToDerived() const { return static_cast<const Derived&>(*this); }

  static void HostAtomicExchange(const unsigned int iterations, T* mem, T* const old_vals,
                                 const unsigned int width, const unsigned pitch, T base_val) {
    for (auto i = 0u; i < iterations; ++i) {
      T new_val = base_val + static_cast<T>(i);
      T old_val;
      __atomic_exchange(pitched_offset(mem, pitch, i % width), &new_val, &old_val,
                        __ATOMIC_RELAXED);
      old_vals[i] = old_val;
    }
  }

  void PerformHostAtomicExchange(const unsigned int thread_count, const unsigned int iterations,
                                 T* mem, T* const old_vals, const AtomicExchParams& p) const {
    if (thread_count == 0) {
      return;
    }
    const auto dev_threads =
        p.blocks.x * p.blocks.y * p.blocks.z * p.threads.x * p.threads.y * p.threads.z;
    const auto host_base_val = p.num_devices * p.kernel_count * dev_threads + p.width;

    std::vector<std::thread> threads;
    for (auto i = 0u; i < thread_count; ++i) {
      const auto thread_base_val = host_base_val + i * iterations;
      threads.push_back(std::thread(HostAtomicExchange, iterations, mem, old_vals + thread_base_val,
                                    p.width, p.pitch, thread_base_val));
    }

    for (auto& th : threads) {
      th.join();
    }
  }
};

template <typename T, bool use_shared_mem, AtomicScopes scope,
          int memory_scope = __HIP_MEMORY_SCOPE_AGENT>
class AtomicExch
    : public AtomicExchCRTP<AtomicExch<T, use_shared_mem, scope>, T, use_shared_mem, scope> {
 public:
  void LaunchKernel(const unsigned int shared_mem_size, const hipStream_t stream, T* const mem,
                    T* const old_vals, const T base_val, const AtomicExchParams& p) const {
    atomic_exch_kernel<T, use_shared_mem, scope, memory_scope>
        <<<p.blocks, p.threads, shared_mem_size, stream>>>(mem, old_vals, p.width, p.pitch,
                                                           base_val);
  }

  void ValidateResults(std::vector<T>& old_vals) const {
    std::sort(old_vals.begin(), old_vals.end());
    for (auto i = 0u; i < old_vals.size(); ++i) {
      REQUIRE(i == old_vals[i]);
    }
  }
};

inline dim3 GenerateAtomicExchThreadDimensions() { return dim3(1024); }

inline dim3 GenerateAtomicExchBlockDimensions() {
  return dim3(8);
}

template <typename TestType, AtomicScopes scope, int memory_scope = __HIP_MEMORY_SCOPE_AGENT>
void AtomicExchSingleDeviceSingleKernelTest(const unsigned int width, const unsigned int pitch) {
  AtomicExchParams params;
  params.num_devices = 1;
  params.kernel_count = 1;
  if constexpr (scope == AtomicScopes::builtin && memory_scope == __HIP_MEMORY_SCOPE_SINGLETHREAD) {
    params.threads = 1;
  } else if constexpr (scope == AtomicScopes::builtin &&
                       memory_scope == __HIP_MEMORY_SCOPE_WAVEFRONT) {
    int warp_size = 0;
    HIP_CHECK(hipDeviceGetAttribute(&warp_size, hipDeviceAttributeWarpSize, 0));
    params.threads = dim3(warp_size);
  } else {
    params.threads = GenerateAtomicExchThreadDimensions();
  }
  params.width = width;
  params.pitch = pitch;

  SECTION("Global memory") {
    if constexpr (scope == AtomicScopes::builtin &&
                  (memory_scope == __HIP_MEMORY_SCOPE_SINGLETHREAD ||
                   memory_scope == __HIP_MEMORY_SCOPE_WAVEFRONT ||
                   memory_scope == __HIP_MEMORY_SCOPE_WORKGROUP)) {
      params.blocks = dim3(1);
    } else {
      params.blocks = GenerateAtomicExchBlockDimensions();
    }
    using LA = LinearAllocs;
    for (const auto alloc_type :
         {LA::hipMalloc, LA::hipHostMalloc, LA::hipMallocManaged}) {
      params.alloc_type = alloc_type;
      DYNAMIC_SECTION("Allocation type: " << to_string(alloc_type)) {
        AtomicExch<TestType, false, scope, memory_scope>().run(params);
      }
    }
  }

  SECTION("Shared memory") {
    params.blocks = dim3(1);
    params.alloc_type = LinearAllocs::hipMalloc;
    AtomicExch<TestType, true, scope, memory_scope>().run(params);
  }
}

template <typename TestType, AtomicScopes scope>
void AtomicExchSingleDeviceMultipleKernelTest(const unsigned int kernel_count,
                                              const unsigned int width, const unsigned int pitch) {
  int concurrent_kernels = 0;
  HIP_CHECK(hipDeviceGetAttribute(&concurrent_kernels, hipDeviceAttributeConcurrentKernels, 0));
  if (!concurrent_kernels) {
    HipTest::HIP_SKIP_TEST("Test requires support for concurrent kernel execution");
    return;
  }

  AtomicExchParams params;
  params.num_devices = 1;
  params.kernel_count = kernel_count;
  params.blocks = GenerateAtomicExchBlockDimensions();
  params.threads = GenerateAtomicExchThreadDimensions();
  params.width = width;
  params.pitch = pitch;

  using LA = LinearAllocs;
  for (const auto alloc_type :
       {LA::hipMalloc, LA::hipHostMalloc, LA::hipMallocManaged}) {
    params.alloc_type = alloc_type;
    DYNAMIC_SECTION("Allocation type: " << to_string(alloc_type)) {
      AtomicExch<TestType, false, scope>().run(params);
    }
  }
}

template <typename TestType>
void AtomicExchMultipleDeviceMultipleKernelAndHostTest(const unsigned int num_devices,
                                                       const unsigned int kernel_count,
                                                       const unsigned int width,
                                                       const unsigned int pitch,
                                                       const unsigned int host_thread_count = 0u) {
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

  AtomicExchParams params;
  params.num_devices = num_devices;
  params.kernel_count = kernel_count;
  params.blocks = GenerateAtomicExchBlockDimensions();
  params.threads = GenerateAtomicExchThreadDimensions();
  params.width = width;
  params.pitch = pitch;
  params.host_thread_count = host_thread_count;

  using LA = LinearAllocs;
  for (const auto alloc_type : {LA::hipHostMalloc , LA::hipMallocManaged}) {
    params.alloc_type = alloc_type;
    DYNAMIC_SECTION("Allocation type: " << to_string(alloc_type)) {
      AtomicExch<TestType, false, AtomicScopes::system>().run(params);
    }
  }
}
