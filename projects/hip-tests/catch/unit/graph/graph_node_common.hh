/*
Copyright (c) 2024 Advanced Micro Devices, Inc. All rights reserved.
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

#include <hip_test_checkers.hh>
#include <hip_test_common.hh>
#include <hip_test_defgroups.hh>
#include <memcpy3d_tests_common.hh>
#include <resource_guards.hh>
#include <utils.hh>

#include <functional>
#include <vector>

#include "graph_memset_node_test_common.hh"
#include "graph_tests_common.hh"

#pragma clang diagnostic ignored "-Wunused-parameter"
#pragma clang diagnostic ignored "-Wunused-function"

namespace {
static constexpr size_t N = 10;
size_t Nbytes = N * sizeof(int);
hipMemcpy3DParms params = {0};
hipGraphNodeParams node_params = {};

void callbackSum(void *hostdata) {
  int *A_h = reinterpret_cast<int *>(hostdata);
  for (int i = 0; i < N; i++) {
    A_h[i] = i + i;
  }
}

void callbackSquare(void *hostdata) {
  int *B_h = reinterpret_cast<int *>(hostdata);
  for (int i = 0; i < N; i++) {
    B_h[i] = i * i;
  }
}
}  // namespace

static hipGraphNodeParams getNodeTypeKernel(void *kernel_args[] = {},
                                            void *func = nullptr) {
  node_params.type = hipGraphNodeTypeKernel;
  node_params.kernel.func = reinterpret_cast<void *>(func);
  node_params.kernel.gridDim = dim3(1);
  node_params.kernel.blockDim = dim3(1);
  node_params.kernel.kernelParams = reinterpret_cast<void **>(kernel_args);
  node_params.kernel.extra = nullptr;
  return node_params;
}

static hipGraphNodeParams getNodeTypeMemset(int *A_d = nullptr, int value = 0) {
  node_params.type = hipGraphNodeTypeMemset;
  node_params.memset.dst = A_d;
  node_params.memset.elementSize = sizeof(int);
  node_params.memset.width = N;
  node_params.memset.height = 1;
  node_params.memset.pitch = N;
  node_params.memset.value = value;
  return node_params;
}

static hipGraphNodeParams
getNodeTypememcpy(int **A_d = nullptr, int **B_d = nullptr,
                  hipMemcpyKind memcopyKind = hipMemcpyDefault) {
  if (memcopyKind == hipMemcpyHostToDevice) {
    params = GetMemcpy3DParms(
        make_hipPitchedPtr(*B_d, N * sizeof(int), N * sizeof(int), 0),
        make_hipPos(0, 0, 0),
        make_hipPitchedPtr(*A_d, N * sizeof(int), N * sizeof(int), 0),
        make_hipPos(0, 0, 0), make_hipExtent(N * sizeof(int), 1, 1),
        memcopyKind);
  } else if (memcopyKind == hipMemcpyHostToHost) {
    params = GetMemcpy3DParms(
        make_hipPitchedPtr(*B_d, N * sizeof(int), N * sizeof(int), 0),
        make_hipPos(0, 0, 0),
        make_hipPitchedPtr(*A_d, N * sizeof(int), N * sizeof(int), 0),
        make_hipPos(0, 0, 0), make_hipExtent(N * sizeof(int), 1, 1),
        memcopyKind);
  } else if (memcopyKind == hipMemcpyDeviceToDevice) {
    params = GetMemcpy3DParms(
        make_hipPitchedPtr(*B_d, N * sizeof(int), N * sizeof(int), 0),
        make_hipPos(0, 0, 0),
        make_hipPitchedPtr(*A_d, N * sizeof(int), N * sizeof(int), 0),
        make_hipPos(0, 0, 0), make_hipExtent(N * sizeof(int), 1, 1),
        memcopyKind);
  } else if (memcopyKind == hipMemcpyDeviceToHost) {
    params = GetMemcpy3DParms(
        make_hipPitchedPtr(*B_d, N * sizeof(int), N * sizeof(int), 0),
        make_hipPos(0, 0, 0),
        make_hipPitchedPtr(*A_d, N * sizeof(int), N * sizeof(int), 0),
        make_hipPos(0, 0, 0), make_hipExtent(N * sizeof(int), 1, 1),
        memcopyKind);
  } else {
    params = GetMemcpy3DParms(
        make_hipPitchedPtr(*B_d, N * sizeof(int), N * sizeof(int), 0),
        make_hipPos(0, 0, 0),
        make_hipPitchedPtr(*A_d, N * sizeof(int), N * sizeof(int), 0),
        make_hipPos(0, 0, 0), make_hipExtent(N * sizeof(int), 1, 1),
        memcopyKind);
  }
  node_params.type = hipGraphNodeTypeMemcpy;
  memset(&node_params.memcpy, 0, sizeof(hipMemcpyNodeParams));
  node_params.memcpy.copyParams = params;
  return node_params;
}

static hipGraphNodeParams getNodeTypeHost(void *userData, void *func) {
  node_params.type = hipGraphNodeTypeHost;
  node_params.host.fn = reinterpret_cast<hipHostFn_t>(func);
  node_params.host.userData = userData;
  return node_params;
}

static hipGraphNodeParams getNodeTypeGraph(hipGraph_t childgraph) {
  node_params.type = hipGraphNodeTypeGraph;
  node_params.graph.graph = childgraph;
  return node_params;
}

static void verifyMemcpyResult(int *A, int *B) {
  int *src, *dst;
  HIP_CHECK(hipHostMalloc(&src, Nbytes));
  HIP_CHECK(hipHostMalloc(&dst, Nbytes));
  HIP_CHECK(hipMemcpy(src, A, Nbytes, hipMemcpyDefault));
  HIP_CHECK(hipMemcpy(dst, B, Nbytes, hipMemcpyDefault));
  for (int i = 0; i < N; i++)
    REQUIRE(src[i] == dst[i]);

  HIP_CHECK(hipHostFree(src));
  HIP_CHECK(hipHostFree(dst));
}
