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

#include "signaling_operations_tester.hpp"

#include <rocshmem/rocshmem.hpp>

using namespace rocshmem;

/******************************************************************************
 * DEVICE TEST KERNEL
 *****************************************************************************/
__global__ void PutmemSignalTest(int loop, int skip, long long int *start_time,
                                 long long int *end_time, char *s_buf,
                                 char *r_buf, size_t size, uint64_t *sig_addr,
                                 TestType type, ShmemContextType ctx_type,
                                 int sig_op) {
  __shared__ rocshmem_ctx_t ctx;
  int wg_id = get_flat_grid_id();
  rocshmem_wg_ctx_create(ctx_type, &ctx);

  uint64_t signal = 1;

  for (int i = 0; i < loop + skip; i++) {
    if (i == skip) {
        __syncthreads();
        start_time[wg_id] = wall_clock64();
    }

    switch (type) {
      case PutSignalTestType:
        rocshmem_ctx_putmem_signal(ctx, r_buf, s_buf, size, sig_addr,
                                   signal, sig_op, 1);
        break;
      case WGPutSignalTestType:
        rocshmem_ctx_putmem_signal_wg(ctx, r_buf, s_buf, size, sig_addr,
                                      signal, sig_op, 1);
        break;
      case WAVEPutSignalTestType:
        rocshmem_ctx_putmem_signal_wave(ctx, r_buf, s_buf, size, sig_addr,
                                        signal, sig_op, 1);
        break;
      case PutSignalNBITestType:
        rocshmem_ctx_putmem_signal_nbi(ctx, r_buf, s_buf, size, sig_addr,
                                       signal, sig_op, 1);
        break;
      case WGPutSignalNBITestType:
        rocshmem_ctx_putmem_signal_nbi_wg(ctx, r_buf, s_buf, size, sig_addr,
                                          signal, sig_op, 1);
        break;
      case WAVEPutSignalNBITestType:
        rocshmem_ctx_putmem_signal_nbi_wave(ctx, r_buf, s_buf, size, sig_addr,
                                            signal, sig_op, 1);
        break;
      default:
        break;
    }
  }

  rocshmem_ctx_quiet(ctx);

  __syncthreads();

  if (hipThreadIdx_x == 0) {
    end_time[wg_id] = wall_clock64();
  }

  rocshmem_wg_ctx_destroy(&ctx);
}

__global__ void SignalFetchTest(int loop, int skip, long long int *start_time,
                                long long int *end_time, uint64_t *sig_addr,
                                uint64_t *fetched_value, TestType type) {

  int wg_id = get_flat_grid_id();

  for (int i = 0; i < loop + skip; i++) {
    if (i == skip) {
        __syncthreads();
        start_time[wg_id] = wall_clock64();
    }

    switch (type) {
      case SignalFetchTestType:
        *fetched_value = rocshmem_signal_fetch(sig_addr);
        break;
      case WGSignalFetchTestType:
        *fetched_value = rocshmem_signal_fetch_wg(sig_addr);
        break;
      case WAVESignalFetchTestType:
        *fetched_value = rocshmem_signal_fetch_wave(sig_addr);
        break;
      default:
        break;
    }
  }

  __syncthreads();

  if (hipThreadIdx_x == 0) {
    end_time[wg_id] = wall_clock64();
  }

}

/******************************************************************************
 * HOST TESTER CLASS METHODS
 *****************************************************************************/
SignalingOperationsTester::SignalingOperationsTester(TesterArguments args)
  : Tester(args) {
  s_buf = (char *)rocshmem_malloc(max_msg_size * args.wg_size);
  r_buf = (char *)rocshmem_malloc(max_msg_size * args.wg_size);
  sig_addr = (uint64_t *)rocshmem_malloc(sizeof(uint64_t));
  CHECK_HIP(hipMallocManaged(&fetched_value, sizeof(uint64_t), hipMemAttachHost));
}

SignalingOperationsTester::SignalingOperationsTester(TesterArguments args,
                                                     int signal_op)
  : SignalingOperationsTester(args) {
  sig_op = signal_op;
}

SignalingOperationsTester::~SignalingOperationsTester() {
  rocshmem_free(s_buf);
  rocshmem_free(r_buf);
  rocshmem_free(sig_addr);
  CHECK_HIP(hipFree(fetched_value));
}

void SignalingOperationsTester::resetBuffers(size_t size) {
  memset(s_buf, '0', max_msg_size * args.wg_size);
  memset(r_buf, '1', max_msg_size * args.wg_size);
  *fetched_value = -1;
  *sig_addr = args.myid + 123;
}

void SignalingOperationsTester::launchKernel(dim3 gridSize, dim3 blockSize, int loop,
                                             size_t size) {
  size_t shared_bytes = 0;


  if ((_type == SignalFetchTestType)     ||
      (_type == WAVESignalFetchTestType) ||
      (_type == WGSignalFetchTestType)) {
    hipLaunchKernelGGL(SignalFetchTest, gridSize, blockSize, shared_bytes, stream,
                       loop, args.skip, start_time, end_time, sig_addr, fetched_value, _type);
  } else {
    hipLaunchKernelGGL(PutmemSignalTest, gridSize, blockSize, shared_bytes, stream,
                       loop, args.skip, start_time, end_time, s_buf, r_buf, size, sig_addr,
                       _type, _shmem_context, sig_op);
  }

  num_msgs = (loop + args.skip) * gridSize.x;
  num_timed_msgs = loop;
}

void SignalingOperationsTester::verifyResults(size_t size) {
  if (_type == SignalFetchTestType     ||
      _type == WAVESignalFetchTestType ||
      _type == WGSignalFetchTestType) {
    if (0 == args.myid) {
      uint64_t value = *fetched_value;
      uint64_t expected_value = (args.myid + 123);
      if (value != expected_value) {
        fprintf(stderr, "Fetched Value %lu, Expected %lu\n", value, expected_value);
        exit(-1);
      }
      return;
    }
  } else {
    if (1 == args.myid) {
      // Validate Data
      for (uint64_t i = 0; i < size; i++) {
        if (r_buf[i] != '0') {
          fprintf(stderr, "Data validation error at idx %lu\n", i);
          fprintf(stderr, "Got %c, Expected %c\n", r_buf[i], '0');
          exit(-1);
        }
      }
      // Validate Signal
      if (ROCSHMEM_SIGNAL_SET == sig_op) {
        uint64_t expected_value = 1;
        uint64_t value = *sig_addr;

        if (value != expected_value) {
          fprintf(stderr, "ROCSHMEM_SIGNAL_SET Value %lu, Expected %lu\n", value, expected_value);
          exit(-1);
        }
      } else if (ROCSHMEM_SIGNAL_ADD == sig_op) {
        uint64_t value = *sig_addr;
        uint64_t expected_value = (args.myid + 123); // Initial Value

        switch (_type) {
          case PutSignalTestType:
          case PutSignalNBITestType:
            expected_value += ((args.skip + args.loop) * args.wg_size * args.num_wgs);
            break;
          case WGPutSignalTestType:
          case WGPutSignalNBITestType:
            expected_value += ((args.skip + args.loop) * args.num_wgs);
            break;
          case WAVEPutSignalTestType:
          case WAVEPutSignalNBITestType:
            expected_value += ((args.skip + args.loop) * args.num_wgs * num_warps);
            break;
          default:
            fprintf(stderr, "Invalid Test\n");
            exit(-1);
        }

        if (value != expected_value) {
          fprintf(stderr, "ROCSHMEM_SIGNAL_ADD Value %lu, Expected %lu\n", value, expected_value);
          exit(-1);
        }
      }
    }
  }
}
