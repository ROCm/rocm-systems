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

#include <hip/hip_cooperative_groups.h>
#include <hip_test_common.hh>

static __global__ void wg_split_barrier(float *out, float *in) {
  namespace cg = cooperative_groups;

  __shared__ float mid[32];
  size_t i = threadIdx.x;
  auto tb = cg::this_thread_block();

  out[i] = in[i] * 2.0f;

  auto tok = tb.barrier_arrive();

  // use tid 0 to populate shared mem
  if (i == 0) {
    for (size_t j = 0; j < 32; j++) {
      mid[j] = in[j];
    }
  }

  tb.barrier_wait(std::move(tok));

  out[i] += mid[i];
}

TEST_CASE("Unit_coop_thread_block_split_barrier") {
  constexpr size_t size = 32;
  float *d_out, *d_in;

  HIP_CHECK(hipMalloc(&d_out, sizeof(float) * size));
  HIP_CHECK(hipMalloc(&d_in, sizeof(float) * size));

  std::vector<float> in(size, 0.0f), out = in;
  for (size_t i = 0; i < size; i++) {
    in[i] = i + 1;
  }

  HIP_CHECK(hipMemset(d_out, 0, sizeof(float) * size));
  HIP_CHECK(
      hipMemcpy(d_in, in.data(), sizeof(float) * size, hipMemcpyHostToDevice));
  wg_split_barrier<<<1, size>>>(d_out, d_in);
  HIP_CHECK(hipMemcpy(out.data(), d_out, sizeof(float) * size,
                      hipMemcpyDeviceToHost));

  HIP_CHECK(hipFree(d_out));
  HIP_CHECK(hipFree(d_in));

  for (size_t i = 0; i < size; i++) {
    INFO("Index: " << i << " in: " << in[i] << " out: " << out[i]);
    REQUIRE((in[i] * 3.0f) == Catch::Approx(out[i]));
  }
}

static __global__ void grid_split_barrier(int *data, int *result, int N) {
  namespace cg = cooperative_groups;
  cg::grid_group grid = cg::this_grid();

  int gid = blockIdx.x * blockDim.x + threadIdx.x;
  auto tok = grid.barrier_arrive();
  if (gid < N) {
    data[gid] = gid + 1;
  }

  grid.barrier_wait(std::move(tok));

  if (grid.thread_rank() == 0) {
    int sum = 0;
    for (int i = 0; i < N; i++)
      sum += data[i];
    *result = sum;
  }
}

TEST_CASE("Unit_coop_grids_split_barrier") {
  hipDeviceProp_t prop;
  HIP_CHECK(hipGetDeviceProperties(&prop, 0));

  if (prop.cooperativeLaunch != 0) {
    int N = 1024;
    const int threads = 128;
    const int blocks = (N + threads - 1) / threads;

    int *d_in, *d_out;
    HIP_CHECK(hipMalloc(&d_in, N * sizeof(int)));
    HIP_CHECK(hipMalloc(&d_out, sizeof(int)));

    void *args[] = {&d_in, &d_out, &N};

    dim3 grid(blocks);
    dim3 block(threads);

    HIP_CHECK(hipLaunchCooperativeKernel((void *)grid_split_barrier, grid,
                                         block, args, 0, 0));
    HIP_CHECK(hipDeviceSynchronize());

    int out = 0;
    HIP_CHECK(hipMemcpy(&out, d_out, sizeof(int), hipMemcpyDeviceToHost));

    HIP_CHECK(hipFree(d_in));
    HIP_CHECK(hipFree(d_out));
    REQUIRE(out == ((N * (N + 1)) / 2));
  }
}
