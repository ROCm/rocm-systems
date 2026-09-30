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
#include <hip_test_checkers.hh>
#include <hip/hip_cooperative_groups.h>

#include <array>

struct cluster_output {
  dim3 block_index;
  unsigned int block_rank;
  dim3 thread_index;
  unsigned int thread_rank;
  dim3 dim_blocks;
  unsigned int num_blocks;
  dim3 dim_threads;
  unsigned int num_threads;
};

static __global__ void CLUSTER_DIMS(2, 1, 1) cluster_class_validation(cluster_output* out) {
  namespace cg = cooperative_groups;
  cg::cluster_group c = cg::this_cluster();
  size_t i = blockIdx.x * blockDim.x + threadIdx.x;
  auto tok = c.barrier_arrive();
  out[i].block_index = c.block_index();
  out[i].block_rank = c.block_rank();
  out[i].thread_index = c.thread_index();
  out[i].thread_rank = c.thread_rank();
  out[i].dim_blocks = c.dim_blocks();
  out[i].num_blocks = c.num_blocks();
  out[i].dim_threads = c.dim_threads();
  out[i].num_threads = c.num_threads();
  c.barrier_wait(std::move(tok));
}

HIP_TEST_CASE(Unit_cluster_coop_group_class) {
  constexpr unsigned int grds = 2, blks = 8, total_threads = grds * blks;
  dim3 grid(grds, 1, 1);
  dim3 block(blks, 1, 1);

  std::array<cluster_output, total_threads> out;

  cluster_output* d_out;
  HIP_CHECK(hipMalloc(&d_out, sizeof(cluster_output) * total_threads));
  HIP_CHECK(hipMemset(d_out, 0, sizeof(cluster_output) * total_threads));
  cluster_class_validation<<<grid, block>>>(d_out);
  HIP_CHECK(
      hipMemcpy(out.data(), d_out, sizeof(cluster_output) * total_threads, hipMemcpyDeviceToHost));

  // These checks are closly tied with cluster dims, a change in dims might result in failure
  for (size_t i = 0; i < total_threads; i++) {
    const auto& o = out[i];
    INFO("Index: " << i);

    REQUIRE(o.thread_index.x == i);
    REQUIRE(o.thread_index.y == 0);
    REQUIRE(o.thread_index.z == 0);
    REQUIRE(o.thread_rank == i);

    REQUIRE(o.dim_blocks.x == 2);
    REQUIRE(o.dim_blocks.y == 1);
    REQUIRE(o.dim_blocks.z == 1);

    REQUIRE(o.num_blocks == 2);
    REQUIRE(o.num_threads == total_threads);

    if (i < 8) {
      REQUIRE(o.block_index.x == 0);
      REQUIRE(o.block_index.y == 0);
      REQUIRE(o.block_index.z == 0);

      REQUIRE(o.block_rank == 0);
    } else {
      REQUIRE(o.block_index.x == 1);
      REQUIRE(o.block_index.y == 0);
      REQUIRE(o.block_index.z == 0);

      REQUIRE(o.block_rank == 1);
    }
  }
}
