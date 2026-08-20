// Wide fork: more lanes than MAX_HW_QUEUES (8). Lanes beyond the pool are
// mapped round-robin onto shared HW queues and serialize there, but every lane
// must still run exactly once and the join must wait for all of them.
//
//        A
//    /  / | \  \      NLANES parallel lanes
//   L0 L1 ... L(N-1)
//    \  \ | /  /
//        D            D sums every lane's output
#include "common.h"

#ifndef NLANES
#define NLANES 12
#endif

__global__ void kA(int* base)                 { *base = 3; }
__global__ void kLane(const int* base, int* out, int idx) {
  out[idx] = *base + idx;
}
__global__ void kD(const int* out, int n, int* sum) {
  int s = 0;
  for (int i = 0; i < n; ++i) s += out[i];
  *sum = s;
}

int main() {
  int *base, *out, *sum;
  CHECK(hipMalloc(&base, sizeof(int)));
  CHECK(hipMalloc(&out, NLANES * sizeof(int)));
  CHECK(hipMalloc(&sum, sizeof(int)));

  hipGraph_t graph;
  CHECK(hipGraphCreate(&graph, 0));

  void* aArgs[] = {&base};
  hipGraphNode_t A = addKernel(graph, (void*)kA, dim3(1), dim3(1), aArgs, nullptr, 0);

  int idxStore[NLANES];
  hipGraphNode_t lanes[NLANES];
  for (int i = 0; i < NLANES; ++i) {
    idxStore[i] = i;
    void* args[] = {&base, &out, &idxStore[i]};
    lanes[i] = addKernel(graph, (void*)kLane, dim3(1), dim3(1), args, &A, 1);
  }

  int n = NLANES;
  void* dArgs[] = {&out, &n, &sum};
  addKernel(graph, (void*)kD, dim3(1), dim3(1), dArgs, lanes, NLANES);

  hipGraphExec_t exec;
  CHECK(hipGraphInstantiate(&exec, graph, nullptr, nullptr, 0));

  hipStream_t stream;
  CHECK(hipStreamCreate(&stream));

  // Expected sum = sum_i (base + i) = NLANES*base + (0+..+N-1)
  int expected = NLANES * 3 + (NLANES * (NLANES - 1)) / 2;

  for (int run = 0; run < 5; ++run) {
    CHECK(hipMemset(base, 0, sizeof(int)));
    CHECK(hipMemset(out, 0, NLANES * sizeof(int)));
    CHECK(hipMemset(sum, 0, sizeof(int)));

    CHECK(hipGraphLaunch(exec, stream));
    CHECK(hipStreamSynchronize(stream));

    int hs = -1;
    CHECK(hipMemcpy(&hs, sum, sizeof(int), hipMemcpyDeviceToHost));
    printf("run %d: sum=%d (exp %d)\n", run, hs, expected);
    if (hs != expected) {
      fprintf(stderr, "MISMATCH on run %d\n", run);
      return 1;
    }
  }

  printf("PASS\n");
  CHECK(hipStreamDestroy(stream));
  CHECK(hipGraphExecDestroy(exec));
  CHECK(hipGraphDestroy(graph));
  return 0;
}
