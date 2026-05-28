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

#include "random_access_tester.hpp"

#include <rocshmem/rocshmem.hpp>
#include <cassert>

using namespace rocshmem;

/******************************************************************************
 * DEVICE TEST KERNEL
 *****************************************************************************/

__device__ bool thread_passing(int num_bins, uint32_t *bin_threads,
                               uint32_t *off_bins, uint32_t *PE_bins,
                               size_t *offset, int *PE, int coal_coef,
                               size_t size) {
  bool pass = false;
  int wave_id = ((hipThreadIdx_x + hipBlockIdx_x * hipBlockDim_x) /
                 64);  // get_global_wave_id();

  int off = wave_id * num_bins;
  for (int i = 0; i < num_bins; i++) {
    if (((hipThreadIdx_x % 64) >= bin_threads[i + off]) &&
        ((hipThreadIdx_x % 64) < bin_threads[i + off] + coal_coef)) {
      pass = true;
      *offset = off_bins[i + off] +
                (((hipThreadIdx_x % 64) - bin_threads[i + off]) * size);
      *PE = PE_bins[i + off];
    }
  }
  return pass;
}

__global__ void RandomAccessTest(int loop, int skip, long long int *start_time,
                                 long long int *end_time, int *s_buf,
                                 int *r_buf, size_t size, OpType type,
                                 int coal_coef, int num_bins, int num_waves,
                                 uint32_t *threads_bins, uint32_t *off_bins,
                                 uint32_t *PE_bins, ShmemContextType ctx_type) {
  __shared__ rocshmem_ctx_t ctx;
  int wg_id = get_flat_grid_id();
  rocshmem_wg_ctx_create(ctx_type, &ctx);

  int pe = rocshmem_ctx_my_pe(ctx);
  size_t offset;
  int PE;

  if (thread_passing(num_bins, threads_bins, off_bins, PE_bins, &offset, &PE,
                     coal_coef, (size / sizeof(int))) == true) {
    s_buf = s_buf + offset;
    r_buf = r_buf + offset;

    for (int i = 0; i < loop + skip; i++) {
      if (i == skip) {
        start_time[wg_id] = wall_clock64();
      }
      switch (type) {
        case GetType:
          rocshmem_ctx_getmem(ctx, r_buf, s_buf, size, PE);
          break;
        case PutType:
          rocshmem_ctx_putmem(ctx, (char *)r_buf, (char *)s_buf, size, PE);
          break;
        default:
          break;
      }
    }

    rocshmem_ctx_quiet(ctx);

    // atomicAdd((unsigned long long *)&timer[hipBlockIdx_x],
    //           rocshmem_timer() - start);
    end_time[wg_id] = wall_clock64();
  }
  rocshmem_wg_ctx_destroy(&ctx);
}

/******************************************************************************
 * HOST HELPER FUNCTIONS
 ****************************************************************************/
__host__ void init_bins(int num_bins, int num_waves, uint32_t *off_bins,
                        uint32_t *threads_bins, uint32_t *PE_bins, int size,
                        int coal_coef, int num_pes, int max_size) {
  srand(time(NULL));

  for (int j = 0; j < num_waves; j++) {
    for (int i = 0; i < num_bins; i++) {
      int current_bin = j * num_bins + i;
      assert((64 % num_bins) == 0);
      int quad_size = 64 / num_bins;
      int allowed_index_range = quad_size - coal_coef - 1;
      assert(allowed_index_range >= 0);
      int rand_val = 0;

      if (allowed_index_range) rand_val = rand() % allowed_index_range;

      threads_bins[current_bin] = rand_val + i * quad_size;

      quad_size = max_size / (num_bins + num_waves);
      rand_val = rand() % (quad_size - (size * coal_coef - 1));
      off_bins[current_bin] = rand_val + current_bin * quad_size;

      PE_bins[current_bin] = rand() % num_pes;
    }
  }
}

/******************************************************************************
 * HOST TESTER CLASS METHODS
 *****************************************************************************/
RandomAccessTester::RandomAccessTester(TesterArguments args) : Tester(args) {
  int max_size = args.max_msg_size;
  int wg_size = args.wg_size;
  _num_waves = (args.wg_size / 64) * args.num_wgs;
  _num_bins = args.thread_access / args.coal_coef;
  if ((args.wg_size / 64) > 1 || (64 % _num_bins) != 0) {
    printf("Argument are incorrect\n");
    assert((args.wg_size / 64) <= 1);
    assert((64 % _num_bins) == 0);
    abort();
  }

  s_buf = (int *)rocshmem_malloc(max_size * wg_size * space);
  r_buf = (int *)rocshmem_malloc(max_size * wg_size * space);
  h_buf = (int *)malloc(max_size * wg_size * space);
  h_dev_buf = (int *)malloc(max_size * wg_size * space);
  CHECK_HIP(hipMalloc((void **)&_threads_bins, sizeof(uint32_t) * _num_waves * _num_bins));
  CHECK_HIP(hipMalloc((void **)&_off_bins, sizeof(uint32_t) * _num_waves * _num_bins));
  CHECK_HIP(hipMalloc((void **)&_PE_bins, sizeof(uint32_t) * _num_waves * _num_bins));
  memset(_threads_bins, 0, sizeof(uint32_t) * _num_waves * _num_bins);
  memset(_off_bins, 0, sizeof(uint32_t) * _num_waves * _num_bins);
  memset(_PE_bins, 0, sizeof(uint32_t) * _num_waves * _num_bins);
}

RandomAccessTester::~RandomAccessTester() {
  rocshmem_free(s_buf);
  rocshmem_free(r_buf);
  free(h_buf);
  free(h_dev_buf);
  CHECK_HIP(hipFree(_threads_bins));
  CHECK_HIP(hipFree(_off_bins));
  CHECK_HIP(hipFree(_PE_bins));
}

void RandomAccessTester::resetBuffers(size_t size) {
  for (size_t i = 0; i < args.max_msg_size / sizeof(int) * args.wg_size * space;
       i++) {
    s_buf[i] = 1;
    r_buf[i] = 0;
    h_buf[i] = 0;
  }
}

void RandomAccessTester::launchKernel(dim3 gridSize, dim3 blockSize, int loop,
                                      size_t size) {
  size_t shared_bytes = 0;

  int _thread_access = args.thread_access;
  int _coal_coef = args.coal_coef;

  assert(_coal_coef >= 1);
  assert(gridSize.x == 1 && gridSize.y == 1 && gridSize.z == 1);

  init_bins(_num_bins, _num_waves, _off_bins, _threads_bins, _PE_bins,
            size / sizeof(int), _coal_coef, args.numprocs,
            (space * size * args.wg_size) / sizeof(int));

  if (args.myid == 0) {
    hipLaunchKernelGGL(RandomAccessTest, gridSize, blockSize, shared_bytes,
                       stream, loop, args.skip, start_time, end_time, s_buf,
                       r_buf, size, (OpType)args.op_type, _coal_coef,
                       _num_bins, _num_waves, _threads_bins, _off_bins,
                       _PE_bins, _shmem_context);
  }
  num_msgs = (loop + args.skip) * _num_waves * _thread_access;
  num_timed_msgs = loop * _num_waves * _thread_access;
}

void RandomAccessTester::verifyResults(size_t size) {
  uint64_t offset;
  for (int k = 0; k < _num_waves; k++) {
    for (int i = 0; i < _num_bins; i++) {
      int index = i + _num_bins * k;
      if (args.op_type == PutType) {
        if (_PE_bins[index] == static_cast<uint32_t>(args.myid)) {
          offset = _off_bins[index];
          for (uint64_t j = 0; j < ((size / sizeof(int)) * args.coal_coef); j++) {
            h_buf[offset + j] = 1;
          }
        }
      } else {
        if (args.myid == 0) {
          offset = _off_bins[index];
          for (uint64_t j = 0; j < ((size / sizeof(int)) * args.coal_coef); j++) {
            h_buf[offset + j] = 1;
          }
        }
      }
    }
  }

  CHECK_HIP(hipMemcpy(h_dev_buf, r_buf, space * args.wg_size * size,
                      hipMemcpyDeviceToHost));
  CHECK_HIP(hipDeviceSynchronize());
  for (uint64_t i = 0; i < (space * args.wg_size * size / sizeof(int)); i++) {
    if (h_dev_buf[i] != h_buf[i]) {
      printf("PE %d  Got Data Validation: expecting %d got %d at  %lu\n",
             args.myid, h_buf[i], h_dev_buf[i], i);
      exit(-1);
    }
  }
}
