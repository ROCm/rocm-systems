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

#include "multi_ctx_put_tester.hpp"

#include <rocshmem/rocshmem.hpp>

using namespace rocshmem;

/******************************************************************************
 * DEVICE TEST KERNEL
 *
 * Each workgroup creates a context from its own team (wg_id-th team) and
 * performs put_nbi through that context.  The number of contexts equals
 * the number of workgroups, controlled by the -w option.
 *****************************************************************************/
__global__ void MultiCtxPutTest(int loop, int skip,
                                long long int *start_time,
                                long long int *end_time,
                                char *source, char *dest, size_t size,
                                ShmemContextType ctx_type, int wf_size,
                                rocshmem_team_t *teams, int num_ctxs) {
  __shared__ rocshmem_ctx_t ctx;
  int wg_id = get_flat_grid_id();
  int t_id  = get_flat_block_id();
  int wf_id = t_id / wf_size;

  rocshmem_wg_team_create_ctx(teams[wg_id % num_ctxs], ctx_type, &ctx);

  __shared__ long long int wf_start_time[32];

  size_t offset = size * get_flat_id();
  source += offset;
  dest += offset;

  for (int i = 0; i < loop + skip; i++) {
    if (i == skip) {
      __syncthreads();
      if (is_thread_zero_in_block()) {
        rocshmem_ctx_quiet(ctx);
      }
      __syncthreads();
      wf_start_time[wf_id] = wall_clock64();
    }

    rocshmem_ctx_putmem_nbi(ctx, dest, source, size, 1);
  }

  __syncthreads();
  if (is_thread_zero_in_block()) {
    rocshmem_ctx_quiet(ctx);
  }

  end_time[wg_id] = wall_clock64();

  // Find the earliest start time
  int num_wfs = (get_flat_block_size() - 1) / wf_size + 1;
  for (int i = num_wfs / 2; i > 0; i >>= 1) {
    if (t_id < i) {
      wf_start_time[t_id] = min(wf_start_time[t_id], wf_start_time[t_id + i]);
    }
  }
  __syncthreads();

  if (t_id == 0) {
    start_time[wg_id] = wf_start_time[0];
  }

  rocshmem_wg_ctx_destroy(&ctx);
}

/******************************************************************************
 * HOST TESTER CLASS METHODS
 *****************************************************************************/
MultiCtxPutTester::MultiCtxPutTester(TesterArguments args) : Tester(args) {
  num_ctxs_ = args.num_wgs;

  size_t buff_size = max_msg_size * args.wg_size * args.num_wgs;
  source = (char *)rocshmem_malloc(buff_size);
  dest = (char *)rocshmem_malloc(buff_size);

  if (source == nullptr || dest == nullptr) {
    std::cerr << "Error allocating memory from symmetric heap" << std::endl;
    std::cerr << "source: " << source << ", dest: " << dest << std::endl;
    if (source) {
      rocshmem_free(source);
    }
    if (dest) {
      rocshmem_free(dest);
    }
    rocshmem_global_exit(1);
  }

  for (size_t i = 0; i < buff_size; i++) {
    source[i] = static_cast<char>('a' + i % 26);
  }
}

MultiCtxPutTester::~MultiCtxPutTester() {
  rocshmem_free(source);
  rocshmem_free(dest);
}

void MultiCtxPutTester::resetBuffers(size_t size) {
  size_t buff_size = size * args.wg_size * args.num_wgs;
  memset(dest, '1', buff_size);
}

void MultiCtxPutTester::preLaunchKernel() {
  int n_pes = rocshmem_team_n_pes(ROCSHMEM_TEAM_WORLD);

  teams_.resize(num_ctxs_);
  for (int i = 0; i < num_ctxs_; i++) {
    teams_[i] = ROCSHMEM_TEAM_INVALID;
    rocshmem_team_split_strided(ROCSHMEM_TEAM_WORLD, 0, 1, n_pes, nullptr, 0,
                                &teams_[i]);
    if (teams_[i] == ROCSHMEM_TEAM_INVALID) {
      std::cerr << "Failed to create team " << i << std::endl;
      rocshmem_global_exit(1);
    }
  }

  CHECK_HIP(hipMalloc(&teams_on_device_,
                      sizeof(rocshmem_team_t) * num_ctxs_));
  CHECK_HIP(hipMemcpy(teams_on_device_, teams_.data(),
                      sizeof(rocshmem_team_t) * num_ctxs_,
                      hipMemcpyHostToDevice));
}

void MultiCtxPutTester::launchKernel(dim3 gridSize, dim3 blockSize, int loop,
                                     size_t size) {
  size_t shared_bytes = 0;

  hipLaunchKernelGGL(MultiCtxPutTest, gridSize, blockSize, shared_bytes,
                     stream, loop, args.skip, start_time, end_time, source,
                     dest, size, _shmem_context, wf_size,
                     teams_on_device_, num_ctxs_);

  num_msgs = (loop + args.skip) * gridSize.x * blockSize.x;
  num_timed_msgs = loop * gridSize.x * blockSize.x;
}

void MultiCtxPutTester::postLaunchKernel() {
  CHECK_HIP(hipFree(teams_on_device_));
  teams_on_device_ = nullptr;

  for (int i = 0; i < num_ctxs_; i++) {
    rocshmem_team_destroy(teams_[i]);
  }
  teams_.clear();
}

void MultiCtxPutTester::verifyResults(size_t size) {
  if (args.myid == 1) {
    size_t buff_size = size * args.wg_size * args.num_wgs;
    for (uint64_t i = 0; i < buff_size; i++) {
      if (dest[i] != source[i]) {
        std::cerr << "Data validation error at idx " << i << std::endl;
        std::cerr << " Got " << dest[i] << ", Expected "
                  << source[i] << std::endl;
        rocshmem_global_exit(1);
      }
    }
  }
}
