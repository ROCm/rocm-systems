// Two sequential fork/join diamonds in one graph (multi-branch chain). Each
// diamond must coalesce into its own multi-queue block, and the join of the
// first must correctly feed the fork of the second.
//
//   A
//  / \        diamond 1
// B   C
//  \ /
//   M
//  / \        diamond 2
// P   Q
//  \ /
//   Z
#include "common.h"

__global__ void kA(int* base)                        { *base = 2; }
__global__ void kB(const int* base, int* bo)         { *bo = *base + 1; }
__global__ void kC(const int* base, int* co)         { *co = *base + 2; }
__global__ void kM(const int* bo, const int* co, int* mo) { *mo = *bo + *co; }
__global__ void kP(const int* mo, int* po)           { *po = *mo + 100; }
__global__ void kQ(const int* mo, int* qo)           { *qo = *mo + 200; }
__global__ void kZ(const int* po, const int* qo, int* zo) { *zo = *po + *qo; }

int main() {
  int *base, *bo, *co, *mo, *po, *qo, *zo;
  CHECK(hipMalloc(&base, sizeof(int))); CHECK(hipMalloc(&bo, sizeof(int)));
  CHECK(hipMalloc(&co, sizeof(int)));   CHECK(hipMalloc(&mo, sizeof(int)));
  CHECK(hipMalloc(&po, sizeof(int)));   CHECK(hipMalloc(&qo, sizeof(int)));
  CHECK(hipMalloc(&zo, sizeof(int)));

  hipGraph_t graph;
  CHECK(hipGraphCreate(&graph, 0));

  void* aArgs[] = {&base};
  hipGraphNode_t A = addKernel(graph, (void*)kA, dim3(1), dim3(1), aArgs, nullptr, 0);

  void* bA[] = {&base, &bo};
  hipGraphNode_t B = addKernel(graph, (void*)kB, dim3(1), dim3(1), bA, &A, 1);
  void* cA[] = {&base, &co};
  hipGraphNode_t C = addKernel(graph, (void*)kC, dim3(1), dim3(1), cA, &A, 1);
  void* mA[] = {&bo, &co, &mo};
  hipGraphNode_t BC[2] = {B, C};
  hipGraphNode_t M = addKernel(graph, (void*)kM, dim3(1), dim3(1), mA, BC, 2);

  void* pA[] = {&mo, &po};
  hipGraphNode_t P = addKernel(graph, (void*)kP, dim3(1), dim3(1), pA, &M, 1);
  void* qA[] = {&mo, &qo};
  hipGraphNode_t Q = addKernel(graph, (void*)kQ, dim3(1), dim3(1), qA, &M, 1);
  void* zA[] = {&po, &qo, &zo};
  hipGraphNode_t PQ[2] = {P, Q};
  addKernel(graph, (void*)kZ, dim3(1), dim3(1), zA, PQ, 2);

  hipGraphExec_t exec;
  CHECK(hipGraphInstantiate(&exec, graph, nullptr, nullptr, 0));
  hipStream_t stream;
  CHECK(hipStreamCreate(&stream));

  // B=3,C=4,M=7,P=107,Q=207,Z=314
  for (int run = 0; run < 5; ++run) {
    CHECK(hipMemset(base, 0, sizeof(int)));  CHECK(hipMemset(mo, 0, sizeof(int)));
    CHECK(hipMemset(zo, 0, sizeof(int)));
    CHECK(hipGraphLaunch(exec, stream));
    CHECK(hipStreamSynchronize(stream));
    int hm = -1, hz = -1;
    CHECK(hipMemcpy(&hm, mo, sizeof(int), hipMemcpyDeviceToHost));
    CHECK(hipMemcpy(&hz, zo, sizeof(int), hipMemcpyDeviceToHost));
    printf("run %d: M=%d (exp 7), Z=%d (exp 314)\n", run, hm, hz);
    if (hm != 7 || hz != 314) { fprintf(stderr, "MISMATCH on run %d\n", run); return 1; }
  }

  printf("PASS\n");
  CHECK(hipStreamDestroy(stream));
  CHECK(hipGraphExecDestroy(exec));
  CHECK(hipGraphDestroy(graph));
  return 0;
}
