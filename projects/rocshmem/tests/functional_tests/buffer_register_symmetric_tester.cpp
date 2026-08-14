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

#include <cstdio>

using namespace rocshmem;

namespace {

constexpr int kValueBase = 1000;

__global__ void BufferRegisterSymmetricTest(int *dest,
                                            ShmemContextType ctx_type) {
  __shared__ rocshmem_ctx_t ctx;
  rocshmem_wg_ctx_create(ctx_type, &ctx);

  if (hipBlockIdx_x == 0 && is_thread_zero_in_block()) {
    int my_pe = rocshmem_ctx_my_pe(ctx);
    int n_pes = rocshmem_ctx_n_pes(ctx);
    int next_pe = (my_pe + 1) % n_pes;
    rocshmem_ctx_int_p(ctx, dest, kValueBase + my_pe, next_pe);
    rocshmem_ctx_quiet(ctx);
  }

  rocshmem_wg_ctx_destroy(&ctx);
}

#if HIP_VERSION >= 70200000
bool device_supports_vmm(int device_id) {
  int supported = 0;
  hipError_t err = hipDeviceGetAttribute(
      &supported, hipDeviceAttributeVirtualMemoryManagementSupported,
      device_id);
  return err == hipSuccess && supported != 0;
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
  max_msg_size = sizeof(int);

  if (rocshmem_query_backend_type() == BackendType::RO_BACKEND) {
    if (this->args.myid == 0) {
      std::printf("buffer_register_symmetric: SKIPPED "
                  "(reverse-offload backend is unsupported)\n");
    }
    skip_ = true;
    return;
  }

#if HIP_VERSION < 70200000
  if (this->args.myid == 0) {
    std::printf("buffer_register_symmetric: SKIPPED "
                "(requires ROCm 7.2 or newer)\n");
  }
  skip_ = true;
  return;
#else
  if (!device_supports_vmm(device_id)) {
    if (this->args.myid == 0) {
      std::printf("buffer_register_symmetric: SKIPPED "
                  "(GPU does not support HIP VMM)\n");
    }
    skip_ = true;
    return;
  }

  if (!vmm_alloc(&original_, &handle_, sizeof(int), &allocation_size_,
                 device_id)) {
    std::fprintf(stderr,
                 "[PE %d] buffer_register_symmetric: VMM allocation failed\n",
                 this->args.myid);
    skip_ = true;
    rocshmem_global_exit(1);
    return;
  }

  registerBuffer();
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
#endif
}

void BufferRegisterSymmetricTester::resetBuffers(
    [[maybe_unused]] uint64_t size) {
#if HIP_VERSION >= 70200000
  if (skip_) {
    return;
  }

  CHECK_HIP(hipMemset(original_, 0xff, sizeof(int)));
  CHECK_HIP(hipDeviceSynchronize());
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

  alias_ = static_cast<int *>(
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
    [[maybe_unused]] int loop, [[maybe_unused]] uint64_t size) {
#if HIP_VERSION >= 70200000
  if (skip_) {
    return;
  }

  hipLaunchKernelGGL(BufferRegisterSymmetricTest, gridSize, blockSize, 0, stream,
                     alias_, _shmem_context);
  num_msgs = 1;
  num_timed_msgs = 1;
#endif
}

void BufferRegisterSymmetricTester::verifyResults(
    [[maybe_unused]] uint64_t size) {
#if HIP_VERSION >= 70200000
  if (skip_) {
    return;
  }

  int received = -1;
  CHECK_HIP(
      hipMemcpy(&received, original_, sizeof(int), hipMemcpyDeviceToHost));
  int previous_pe = (args.myid - 1 + args.numprocs) % args.numprocs;
  int expected = kValueBase + previous_pe;
  if (received != expected) {
    std::fprintf(stderr,
                 "[PE %d] buffer_register_symmetric: received %d, expected "
                 "%d from PE %d\n",
                 args.myid, received, expected, previous_pe);
    pass_ = false;
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

  constexpr int kPostUnregisterValue = 0x13579bdf;
  int post_unregister_value = 0;
  CHECK_HIP(hipMemcpy(original_, &kPostUnregisterValue, sizeof(int),
                      hipMemcpyHostToDevice));
  CHECK_HIP(hipMemcpy(&post_unregister_value, original_, sizeof(int),
                      hipMemcpyDeviceToHost));
  if (post_unregister_value != kPostUnregisterValue) {
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

  if (args.myid == 0) {
    std::printf("PASS: symmetric VMM buffer registration, remote RMA, and "
                "unregistration succeeded\n");
  }
#endif
}
