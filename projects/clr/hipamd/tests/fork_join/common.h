#pragma once
#include <hip/hip_runtime.h>
#include <cstdio>
#include <cstdlib>

#define CHECK(cmd)                                                          \
  do {                                                                      \
    hipError_t e = (cmd);                                                   \
    if (e != hipSuccess) {                                                  \
      fprintf(stderr, "FAIL %s:%d: %s -> %s\n", __FILE__, __LINE__, #cmd,   \
              hipGetErrorString(e));                                        \
      exit(1);                                                              \
    }                                                                       \
  } while (0)

static inline hipGraphNode_t addKernel(hipGraph_t g, void* fn, dim3 grid,
                                       dim3 block, void** args,
                                       hipGraphNode_t* deps, size_t ndeps) {
  hipKernelNodeParams kp = {};
  kp.func = fn;
  kp.gridDim = grid;
  kp.blockDim = block;
  kp.kernelParams = args;
  hipGraphNode_t node = nullptr;
  CHECK(hipGraphAddKernelNode(&node, g, deps, ndeps, &kp));
  return node;
}
