// Basic structured fork/join diamond on the command-buffer fast path:
//
//        A            A writes base
//       / \           B, C are parallel lanes (FORK -> two HW queues)
//      B   C          D is the join (atomic-counter JOIN continues here)
//       \ /
//        D            D reads both lane outputs
//
// Verifies both lanes run before the join and the result is correct across
// repeated launches (the fork re-arms its join counter each launch).
#include "common.h"

__global__ void kA(int* base)            { *base = 7; }
__global__ void kB(const int* base, int* bOut) { *bOut = *base + 1; }
__global__ void kC(const int* base, int* cOut) { *cOut = *base + 2; }
__global__ void kD(const int* bOut, const int* cOut, int* dOut) {
  *dOut = *bOut + *cOut;
}

int main() {
  int *base, *bOut, *cOut, *dOut;
  CHECK(hipMalloc(&base, sizeof(int)));
  CHECK(hipMalloc(&bOut, sizeof(int)));
  CHECK(hipMalloc(&cOut, sizeof(int)));
  CHECK(hipMalloc(&dOut, sizeof(int)));

  hipGraph_t graph;
  CHECK(hipGraphCreate(&graph, 0));

  void* aArgs[] = {&base};
  hipGraphNode_t A = addKernel(graph, (void*)kA, dim3(1), dim3(1), aArgs, nullptr, 0);

  void* bArgs[] = {&base, &bOut};
  hipGraphNode_t B = addKernel(graph, (void*)kB, dim3(1), dim3(1), bArgs, &A, 1);

  void* cArgs[] = {&base, &cOut};
  hipGraphNode_t C = addKernel(graph, (void*)kC, dim3(1), dim3(1), cArgs, &A, 1);

  void* dArgs[] = {&bOut, &cOut, &dOut};
  hipGraphNode_t deps[2] = {B, C};
  addKernel(graph, (void*)kD, dim3(1), dim3(1), dArgs, deps, 2);

  hipGraphExec_t exec;
  CHECK(hipGraphInstantiate(&exec, graph, nullptr, nullptr, 0));

  hipStream_t stream;
  CHECK(hipStreamCreate(&stream));

  for (int run = 0; run < 5; ++run) {
    int zero = 0;
    CHECK(hipMemcpy(base, &zero, sizeof(int), hipMemcpyHostToDevice));
    CHECK(hipMemcpy(bOut, &zero, sizeof(int), hipMemcpyHostToDevice));
    CHECK(hipMemcpy(cOut, &zero, sizeof(int), hipMemcpyHostToDevice));
    CHECK(hipMemcpy(dOut, &zero, sizeof(int), hipMemcpyHostToDevice));

    CHECK(hipGraphLaunch(exec, stream));
    CHECK(hipStreamSynchronize(stream));

    int hb = -1, hc = -1, hd = -1;
    CHECK(hipMemcpy(&hb, bOut, sizeof(int), hipMemcpyDeviceToHost));
    CHECK(hipMemcpy(&hc, cOut, sizeof(int), hipMemcpyDeviceToHost));
    CHECK(hipMemcpy(&hd, dOut, sizeof(int), hipMemcpyDeviceToHost));

    printf("run %d: bOut=%d (exp 8), cOut=%d (exp 9), dOut=%d (exp 17)\n",
           run, hb, hc, hd);
    if (hb != 8 || hc != 9 || hd != 17) {
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
