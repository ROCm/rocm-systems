/*
Copyright (c) 2022-25 Advanced Micro Devices, Inc. All rights reserved.

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

#include <functional>

#include <hip_test_helper.hh>

#include <hip_test_common.hh>
#include <hip_test_checkers.hh>
#include <hip_test_kernels.hh>

#include "graph_memset_node_test_common.hh"
#include "graph_tests_common.hh"

/**
 * @addtogroup hipGraphLaunch
 * @{
 * @ingroup GraphTest
 * `hipGraphLaunch(hipGraphExec_t graphExec, hipStream_t stream)` -
 * Launches an executable graph on the multi device
 */

/**
 * Test Description
 * ------------------------
 *  - Launches the single branch graph on multi device and verify the result
 * ------------------------
 *  - catch/unit/graph//hipGraphMultiDevice.cc
 * Test requirements
 * ------------------------
 *  - Multi-device
 *  - HIP_VERSION >= 7.2
 */
static void check_output(int* inp, int* out, size_t size) {
  for (size_t i = 0; i < size; i++) {
    REQUIRE(out[i] == ((inp[i] * inp[i]) * (inp[i] * inp[i])));
  }
}

static void init_input(int* a, size_t size) {
  unsigned int seed = time(nullptr);
  for (size_t i = 0; i < size; i++) {
    a[i] = (HipTest::RAND_R(&seed) & 0xFF);
  }
}


TEST_CASE("Unit_hipGraphMultiDevice") {
  int nGpus = 0;
  HIP_CHECK(hipGetDeviceCount(&nGpus));
  if (nGpus < 2) {
    fprintf(stderr, "Need at least 2 GPUs, skipped!\n");
    return;
  }
  hipStream_t streamdev1, streamdev2;
  hipEvent_t eventdev1, eventdev2;
  hipGraph_t graph = nullptr;
  hipGraphExec_t graph_exec = nullptr;

  constexpr size_t buffer_size = (1024 * 1024);
  constexpr auto blocksPerCU = 6;
  constexpr int block_size = 512;

  int *ibuf_h, *buf_d1, *buf_d2, *outbuf_h;
  ibuf_h = new int[buffer_size];
  outbuf_h = new int[buffer_size];
  REQUIRE(ibuf_h != nullptr);

  HIP_CHECK(hipSetDevice(0));
  HIP_CHECK(hipStreamCreate(&streamdev1));
  HIP_CHECK(hipMalloc(&buf_d1, buffer_size * sizeof(int)));
  HIP_CHECK(hipEventCreate(&eventdev1));

  HIP_CHECK(hipSetDevice(1));
  HIP_CHECK(hipStreamCreate(&streamdev2));
  HIP_CHECK(hipMalloc(&buf_d2, buffer_size * sizeof(int)));
  HIP_CHECK(hipEventCreate(&eventdev2));

  HIP_CHECK(hipSetDevice(0));
  init_input(ibuf_h, buffer_size);
  unsigned grid_size = HipTest::setNumBlocks(blocksPerCU, block_size, buffer_size);

  HIP_CHECK(hipStreamBeginCapture(streamdev1, hipStreamCaptureModeGlobal));

  HIP_CHECK(
      hipMemcpyAsync(buf_d1, ibuf_h, sizeof(int) * buffer_size, hipMemcpyHostToDevice, streamdev1));
  HipTest::vector_square<int>
      <<<grid_size, block_size, 0, streamdev1>>>(buf_d1, buf_d1, buffer_size);
  HIP_CHECK(hipEventRecord(eventdev1, streamdev1));
  HIP_CHECK(hipStreamWaitEvent(streamdev2, eventdev1));

  HIP_CHECK(hipSetDevice(1));
  HIP_CHECK(hipMemcpyDtoDAsync(buf_d2, buf_d1, sizeof(int) * buffer_size, streamdev2));
  HipTest::vector_square<int>
      <<<grid_size, block_size, 0, streamdev2>>>(buf_d2, buf_d2, buffer_size);
  HIP_CHECK(hipEventRecord(eventdev2, streamdev2));
  HIP_CHECK(hipStreamWaitEvent(streamdev1, eventdev2));

  HIP_CHECK(hipStreamEndCapture(streamdev1, &graph));

  HIP_CHECK(hipSetDevice(0));
  HIP_CHECK(hipGraphInstantiate(&graph_exec, graph, nullptr, nullptr, 0));
  HIP_CHECK(hipGraphLaunch(graph_exec, streamdev1));
  HIP_CHECK(hipStreamSynchronize(streamdev1));

  HIP_CHECK(hipSetDevice(1));
  HIP_CHECK(hipMemcpy(outbuf_h, buf_d2, sizeof(int) * buffer_size, hipMemcpyHostToDevice));
  check_output(ibuf_h, outbuf_h, buffer_size);

  HIP_CHECK(hipGraphExecDestroy(graph_exec));
  HIP_CHECK(hipGraphDestroy(graph));

  delete[] ibuf_h;
  delete[] outbuf_h;
  HIP_CHECK(hipFree(buf_d1));
  HIP_CHECK(hipFree(buf_d2));
  HIP_CHECK(hipStreamDestroy(streamdev1));
  HIP_CHECK(hipStreamDestroy(streamdev2));
  HIP_CHECK(hipEventDestroy(eventdev1));
  HIP_CHECK(hipEventDestroy(eventdev2));
}

/**
 * End doxygen group GraphMultiDevice.
 * @}
 */
