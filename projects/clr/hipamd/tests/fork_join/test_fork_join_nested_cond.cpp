// Fork/join where one lane contains a conditional (IF) node. This exercises
// the sub-graph lane path: the conditional lane's exit is routed through a
// dedicated ATOMIC_JOIN block (both IF branches converge on it), while the
// other lane is a plain kernel fused directly with its ATOMIC_JOIN terminator.
//
//        A
//       / \
//      B   C(IF)        B: plain lane   C: conditional lane
//       \ /
//        D
#include "common.h"

__global__ void kA(int* base)                   { *base = 1; }
__global__ void kB(const int* base, int* bOut)  { *bOut = *base + 10; }
__global__ void kIfTrue(const int* base, int* cOut) { *cOut = *base + 100; }
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

  // IF handle defaults to 1 -> the true branch always runs in this test.
  hipGraphConditionalHandle ifHandle;
  CHECK(hipGraphConditionalHandleCreate(&ifHandle, graph, 1, 0));

  void* aArgs[] = {&base};
  hipGraphNode_t A = addKernel(graph, (void*)kA, dim3(1), dim3(1), aArgs, nullptr, 0);

  void* bArgs[] = {&base, &bOut};
  hipGraphNode_t B = addKernel(graph, (void*)kB, dim3(1), dim3(1), bArgs, &A, 1);

  // Conditional (IF) lane: true body writes cOut; empty false body.
  hipGraph_t ifTrue, ifFalse;
  CHECK(hipGraphCreate(&ifTrue, 0));
  CHECK(hipGraphCreate(&ifFalse, 0));
  void* ifArgs[] = {&base, &cOut};
  addKernel(ifTrue, (void*)kIfTrue, dim3(1), dim3(1), ifArgs, nullptr, 0);

  hipGraph_t ifBranches[2] = {ifTrue, ifFalse};
  hipGraphNode_t C;
  CHECK(hipGraphAddConditionalNode(&C, graph, &A, 1, ifHandle,
                                   hipGraphCondTypeIf, 2, ifBranches, 0));

  void* dArgs[] = {&bOut, &cOut, &dOut};
  hipGraphNode_t deps[2] = {B, C};
  addKernel(graph, (void*)kD, dim3(1), dim3(1), dArgs, deps, 2);

  hipGraphExec_t exec;
  CHECK(hipGraphInstantiate(&exec, graph, nullptr, nullptr, 0));

  hipStream_t stream;
  CHECK(hipStreamCreate(&stream));

  for (int run = 0; run < 5; ++run) {
    CHECK(hipMemset(base, 0, sizeof(int)));
    CHECK(hipMemset(bOut, 0, sizeof(int)));
    CHECK(hipMemset(cOut, 0, sizeof(int)));
    CHECK(hipMemset(dOut, 0, sizeof(int)));

    CHECK(hipGraphLaunch(exec, stream));
    CHECK(hipStreamSynchronize(stream));

    int hb = -1, hc = -1, hd = -1;
    CHECK(hipMemcpy(&hb, bOut, sizeof(int), hipMemcpyDeviceToHost));
    CHECK(hipMemcpy(&hc, cOut, sizeof(int), hipMemcpyDeviceToHost));
    CHECK(hipMemcpy(&hd, dOut, sizeof(int), hipMemcpyDeviceToHost));

    printf("run %d: bOut=%d (exp 11), cOut=%d (exp 101), dOut=%d (exp 112)\n",
           run, hb, hc, hd);
    if (hb != 11 || hc != 101 || hd != 112) {
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
