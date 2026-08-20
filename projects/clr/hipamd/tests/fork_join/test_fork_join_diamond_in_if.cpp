// Plain fork/join diamond INSIDE a conditional IF true-body. Exercises the
// recursive coalescing: the diamond must collapse into a single multi-queue
// block even though it lives inside a conditional body ("inside conditional it
// should become the static graph").
//
//   A
//   |
//   C(IF true)              IF body:   X
//   |                                 / \      plain diamond
//   E                                Y   Z
//                                     \ /
//                                      W
#include "common.h"

__global__ void kA(int* base)                        { *base = 5; }
__global__ void kX(const int* base, int* xo)         { *xo = *base + 1; }
__global__ void kY(const int* xo, int* yo)           { *yo = *xo + 10; }
__global__ void kZ(const int* xo, int* zo)           { *zo = *xo + 20; }
__global__ void kW(const int* yo, const int* zo, int* wo) { *wo = *yo + *zo; }
__global__ void kE(const int* wo, int* eo)           { *eo = *wo + 1000; }

int main() {
  int *base, *xo, *yo, *zo, *wo, *eo;
  CHECK(hipMalloc(&base, sizeof(int))); CHECK(hipMalloc(&xo, sizeof(int)));
  CHECK(hipMalloc(&yo, sizeof(int)));   CHECK(hipMalloc(&zo, sizeof(int)));
  CHECK(hipMalloc(&wo, sizeof(int)));   CHECK(hipMalloc(&eo, sizeof(int)));

  hipGraph_t graph;
  CHECK(hipGraphCreate(&graph, 0));

  hipGraphConditionalHandle ifHandle;
  CHECK(hipGraphConditionalHandleCreate(&ifHandle, graph, 1, 0));  // default true

  void* aArgs[] = {&base};
  hipGraphNode_t A = addKernel(graph, (void*)kA, dim3(1), dim3(1), aArgs, nullptr, 0);

  // IF true-body: plain diamond X -> {Y,Z} -> W.
  hipGraph_t ifTrue, ifFalse;
  CHECK(hipGraphCreate(&ifTrue, 0));
  CHECK(hipGraphCreate(&ifFalse, 0));
  void* xA[] = {&base, &xo};
  hipGraphNode_t X = addKernel(ifTrue, (void*)kX, dim3(1), dim3(1), xA, nullptr, 0);
  void* yA[] = {&xo, &yo};
  hipGraphNode_t Y = addKernel(ifTrue, (void*)kY, dim3(1), dim3(1), yA, &X, 1);
  void* zA[] = {&xo, &zo};
  hipGraphNode_t Z = addKernel(ifTrue, (void*)kZ, dim3(1), dim3(1), zA, &X, 1);
  void* wA[] = {&yo, &zo, &wo};
  hipGraphNode_t YZ[2] = {Y, Z};
  addKernel(ifTrue, (void*)kW, dim3(1), dim3(1), wA, YZ, 2);

  hipGraph_t ifBranches[2] = {ifTrue, ifFalse};
  hipGraphNode_t C;
  CHECK(hipGraphAddConditionalNode(&C, graph, &A, 1, ifHandle,
                                   hipGraphCondTypeIf, 2, ifBranches, 0));

  void* eArgs[] = {&wo, &eo};
  addKernel(graph, (void*)kE, dim3(1), dim3(1), eArgs, &C, 1);

  hipGraphExec_t exec;
  CHECK(hipGraphInstantiate(&exec, graph, nullptr, nullptr, 0));
  hipStream_t stream;
  CHECK(hipStreamCreate(&stream));

  // X=6, Y=16, Z=26, W=42, E=1042
  for (int run = 0; run < 5; ++run) {
    CHECK(hipMemset(base, 0, sizeof(int)));  CHECK(hipMemset(wo, 0, sizeof(int)));
    CHECK(hipMemset(eo, 0, sizeof(int)));
    CHECK(hipGraphLaunch(exec, stream));
    CHECK(hipStreamSynchronize(stream));
    int hw = -1, he = -1;
    CHECK(hipMemcpy(&hw, wo, sizeof(int), hipMemcpyDeviceToHost));
    CHECK(hipMemcpy(&he, eo, sizeof(int), hipMemcpyDeviceToHost));
    printf("run %d: W=%d (exp 42), E=%d (exp 1042)\n", run, hw, he);
    if (hw != 42 || he != 1042) { fprintf(stderr, "MISMATCH on run %d\n", run); return 1; }
  }

  printf("PASS\n");
  CHECK(hipStreamDestroy(stream));
  CHECK(hipGraphExecDestroy(exec));
  CHECK(hipGraphDestroy(graph));
  return 0;
}
