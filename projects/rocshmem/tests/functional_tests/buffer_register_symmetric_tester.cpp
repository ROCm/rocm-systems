/******************************************************************************
 * Copyright (c) Advanced Micro Devices, Inc. All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to
 * deal in the Software without restriction, including without limitation the
 * rights to use, copy, modify, merge, publish, distribute, sublicense, and/or
 * sell copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
 * IN THE SOFTWARE.
 *****************************************************************************/

#include "buffer_register_symmetric_tester.hpp"

#include <hip/hip_runtime.h>
#include <rocshmem/rocshmem.hpp>

#include <algorithm>
#include <cstdio>
#include <limits>
#include <vector>

using namespace rocshmem;

namespace {

unsigned char source_pattern(int pe, size_t worker, size_t byte) {
  return static_cast<unsigned char>(
      (static_cast<size_t>(pe) + worker + byte) & 0xff);
}

__global__ void BufferRegisterSymmetricTest(unsigned char *dest,
                                            const unsigned char *source,
                                            size_t message_size,
                                            ShmemContextType ctx_type) {
  __shared__ rocshmem_ctx_t ctx;
  rocshmem_wg_ctx_create(ctx_type, &ctx);

  const size_t worker =
      static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const size_t offset = worker * message_size;
  int my_pe = rocshmem_ctx_my_pe(ctx);
  int n_pes = rocshmem_ctx_n_pes(ctx);
  int next_pe = (my_pe + 1) % n_pes;

  rocshmem_ctx_putmem(ctx, dest + offset, source + offset, message_size,
                      next_pe);
  rocshmem_ctx_quiet(ctx);
  __syncthreads();

  rocshmem_wg_ctx_destroy(&ctx);
}

#if HIP_VERSION >= 70200000
bool device_supports_vmm(int device_id) {
  int supported = 0;
  CHECK_HIP(hipDeviceGetAttribute(
      &supported, hipDeviceAttributeVirtualMemoryManagementSupported,
      device_id));
  return supported != 0;
}

bool vmm_alloc(void **ptr, hipMemGenericAllocationHandle_t *handle,
               size_t requested_size, size_t *allocation_size, int device_id) {
  hipMemAllocationProp prop = {};
  prop.type = hipMemAllocationTypePinned;
  prop.location.type = hipMemLocationTypeDevice;
  prop.location.id = device_id;
  prop.requestedHandleTypes = hipMemHandleTypePosixFileDescriptor;
  prop.allocFlags.gpuDirectRDMACapable = 1;

  size_t granularity = 0;
  if (hipMemGetAllocationGranularity(&granularity, &prop,
                                     hipMemAllocationGranularityMinimum) !=
          hipSuccess ||
      granularity == 0) {
    return false;
  }

  size_t size =
      ((requested_size + granularity - 1) / granularity) * granularity;
  if (hipMemCreate(handle, size, &prop, 0) != hipSuccess) {
    return false;
  }

  void *address = nullptr;
  if (hipMemAddressReserve(&address, size, 0, 0, 0) != hipSuccess) {
    (void)hipMemRelease(*handle);
    return false;
  }

  if (hipMemMap(address, size, 0, *handle, 0) != hipSuccess) {
    (void)hipMemAddressFree(address, size);
    (void)hipMemRelease(*handle);
    return false;
  }

  hipMemAccessDesc access_desc[2] = {};
  access_desc[0].location.type = hipMemLocationTypeDevice;
  access_desc[0].location.id = device_id;
  access_desc[0].flags = hipMemAccessFlagsProtReadWrite;
  access_desc[1].location.type = hipMemLocationTypeHost;
  access_desc[1].location.id = 0;
  access_desc[1].flags = hipMemAccessFlagsProtReadWrite;

  if (hipMemSetAccess(address, size, access_desc, 2) != hipSuccess) {
    (void)hipMemUnmap(address, size);
    (void)hipMemAddressFree(address, size);
    (void)hipMemRelease(*handle);
    return false;
  }

  *ptr = address;
  *allocation_size = size;
  return true;
}

void vmm_free(void *ptr, hipMemGenericAllocationHandle_t handle, size_t size) {
  CHECK_HIP(hipMemUnmap(ptr, size));
  CHECK_HIP(hipMemAddressFree(ptr, size));
  CHECK_HIP(hipMemRelease(handle));
}
#endif

}  // namespace

BufferRegisterSymmetricTester::BufferRegisterSymmetricTester(
    TesterArguments args)
    : Tester(args) {
  _type = BufferRegisterSymmetricTestType;
  _print_results = false;
  this->args.min_msg_size = sizeof(int);
  max_msg_size =
      args.max_msg_size_set
          ? std::max(args.max_msg_size, this->args.min_msg_size)
          : this->args.min_msg_size;
  worker_count_ = static_cast<size_t>(args.num_wgs) * args.wg_size;

  if (worker_count_ == 0 ||
      max_msg_size > std::numeric_limits<size_t>::max() / worker_count_) {
    std::fprintf(stderr,
                 "[PE %d] buffer_register_symmetric: invalid buffer size\n",
                 this->args.myid);
    rocshmem_global_exit(1);
    return;
  }
  const size_t requested_size = worker_count_ * max_msg_size;

  if (rocshmem_query_backend_type() == BackendType::RO_BACKEND) {
    if (this->args.myid == 0) {
      std::printf("buffer_register_symmetric: SKIPPED "
                  "(reverse-offload backend is unsupported)\n");
    }
    skip_ = true;
    return;
  }

#if HIP_VERSION >= 70200000
  if (!device_supports_vmm(device_id)) {
    std::fprintf(stderr,
                 "[PE %d] buffer_register_symmetric: GPU does not support "
                 "HIP VMM; launcher preflight should have skipped this test\n",
                 this->args.myid);
    rocshmem_global_exit(1);
    return;
  }

  if (!vmm_alloc(&original_, &handle_, requested_size, &allocation_size_,
                 device_id)) {
    std::fprintf(stderr,
                 "[PE %d] buffer_register_symmetric: VMM allocation failed\n",
                 this->args.myid);
    skip_ = true;
    rocshmem_global_exit(1);
    return;
  }

  source_ = static_cast<unsigned char *>(alloc_test_buffer(allocation_size_));
  registerBuffer();
#else
  if (this->args.myid == 0) {
    std::printf("buffer_register_symmetric: SKIPPED "
                "(requires ROCm 7.2 or newer)\n");
  }
  skip_ = true;
  return;
#endif
}

BufferRegisterSymmetricTester::~BufferRegisterSymmetricTester() {
#if HIP_VERSION >= 70200000
  if (alias_ != nullptr) {
    unregisterBuffer();
  }

  if (original_ != nullptr) {
    rocshmem_barrier_all();
    vmm_free(original_, handle_, allocation_size_);
    original_ = nullptr;
  }
  if (source_ != nullptr) {
    free_test_buffer(source_);
    source_ = nullptr;
  }
#endif
}

void BufferRegisterSymmetricTester::resetBuffers(
    uint64_t size) {
#if HIP_VERSION >= 70200000
  if (skip_) {
    return;
  }

  const size_t buffer_size = worker_count_ * size;
  std::vector<unsigned char> source(buffer_size);
  for (size_t worker = 0; worker < worker_count_; ++worker) {
    for (size_t byte = 0; byte < size; ++byte) {
      source[worker * size + byte] =
          source_pattern(args.myid, worker, byte);
    }
  }

  CHECK_HIP(hipMemsetAsync(original_, 0, buffer_size, stream));
  CHECK_HIP(hipMemcpyAsync(source_, source.data(), buffer_size,
                           hipMemcpyHostToDevice, stream));
  CHECK_HIP(hipStreamSynchronize(stream));
#endif
}

void BufferRegisterSymmetricTester::registerBuffer() {
#if HIP_VERSION >= 70200000
  if (skip_) {
    return;
  }

  /*
   * A granularity-aligned hipMalloc allocation isolates the non-VMM rejection
   * from the API's independent length-alignment validation.
   */
  void *plain_buffer = nullptr;
  CHECK_HIP(hipMalloc(&plain_buffer, allocation_size_));
  void *plain_alias =
      rocshmem_buffer_register_symmetric(plain_buffer, allocation_size_);
  // The expected rejection of non-VMM memory may leave a HIP error pending.
  (void)hipGetLastError();
  if (plain_alias != nullptr) {
    int unregister_status =
        rocshmem_buffer_unregister_symmetric(plain_alias);
    if (unregister_status == ROCSHMEM_SUCCESS) {
      CHECK_HIP(hipFree(plain_buffer));
    }
    std::fprintf(stderr,
                 "[PE %d] buffer_register_symmetric: accepted non-VMM memory\n",
                 args.myid);
    skip_ = true;
    rocshmem_global_exit(1);
    return;
  }
  CHECK_HIP(hipFree(plain_buffer));

  alias_ = static_cast<unsigned char *>(
      rocshmem_buffer_register_symmetric(original_, allocation_size_));
  if (alias_ == nullptr) {
    std::fprintf(stderr,
                 "[PE %d] buffer_register_symmetric: VMM registration failed\n",
                 args.myid);
    skip_ = true;
    rocshmem_global_exit(1);
    return;
  }

  if (alias_ == original_) {
    std::fprintf(stderr,
                 "[PE %d] buffer_register_symmetric: returned the original "
                 "address instead of a managed alias\n",
                 args.myid);
    pass_ = false;
  }
#endif
}

void BufferRegisterSymmetricTester::launchKernel(
    [[maybe_unused]] dim3 gridSize, [[maybe_unused]] dim3 blockSize,
    int loop, uint64_t size) {
#if HIP_VERSION >= 70200000
  if (skip_) {
    return;
  }

  hipLaunchKernelGGL(BufferRegisterSymmetricTest, gridSize, blockSize, 0, stream,
                     alias_, source_, size, _shmem_context);
  num_msgs = (loop + args.skip) * worker_count_;
  num_timed_msgs = loop * worker_count_;
#endif
}

void BufferRegisterSymmetricTester::verifyResults(
    uint64_t size) {
#if HIP_VERSION >= 70200000
  if (skip_) {
    return;
  }

  const size_t buffer_size = worker_count_ * size;
  std::vector<unsigned char> received(buffer_size);
  CHECK_HIP(hipMemcpy(received.data(), original_, buffer_size,
                      hipMemcpyDeviceToHost));
  int previous_pe = (args.myid - 1 + args.numprocs) % args.numprocs;
  for (size_t worker = 0; worker < worker_count_; ++worker) {
    for (size_t byte = 0; byte < size; ++byte) {
      const unsigned char expected =
          source_pattern(previous_pe, worker, byte);
      const unsigned char actual = received[worker * size + byte];
      if (actual != expected) {
        std::fprintf(
            stderr,
            "[PE %d] buffer_register_symmetric: offset %zu received %u, "
            "expected %u from PE %d\n",
            args.myid, worker * size + byte,
            static_cast<unsigned int>(actual),
            static_cast<unsigned int>(expected), previous_pe);
        pass_ = false;
        return;
      }
    }
  }
#endif
}

void BufferRegisterSymmetricTester::unregisterBuffer() {
#if HIP_VERSION >= 70200000
  int unregister_status = rocshmem_buffer_unregister_symmetric(alias_);
  if (unregister_status != ROCSHMEM_SUCCESS) {
    std::fprintf(stderr,
                 "[PE %d] buffer_register_symmetric: unregister returned %d\n",
                 args.myid, unregister_status);
    rocshmem_global_exit(1);
    return;
  }
  alias_ = nullptr;

  // Verify unregister leaves the original VMM allocation usable.
  constexpr unsigned char kPostUnregisterValue = 0xa5;
  CHECK_HIP(hipMemset(original_, kPostUnregisterValue, allocation_size_));
  std::vector<unsigned char> post_unregister(allocation_size_);
  CHECK_HIP(hipMemcpy(post_unregister.data(), original_, allocation_size_,
                      hipMemcpyDeviceToHost));
  if (!std::all_of(post_unregister.begin(), post_unregister.end(),
                   [](unsigned char value) {
                     return value == kPostUnregisterValue;
                   })) {
    std::fprintf(stderr,
                 "[PE %d] buffer_register_symmetric: original VMM allocation "
                 "is unusable after unregister\n",
                 args.myid);
    pass_ = false;
  }

  if (!pass_) {
    rocshmem_global_exit(1);
    return;
  }

  // Do not print PASS while another PE may still report a local failure.
  rocshmem_barrier_all();

  if (args.myid == 0) {
    if (args.verif) {
      std::printf("PASS: symmetric VMM buffer registration, remote RMA, and "
                  "unregistration succeeded\n");
    } else {
      std::printf("PASS: symmetric VMM buffer registration, RMA launch, and "
                  "unregistration succeeded (data verification disabled)\n");
    }
  }
#endif
}
