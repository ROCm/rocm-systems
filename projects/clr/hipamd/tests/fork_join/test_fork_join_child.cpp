// Fork/join diamond living INSIDE a child graph node. The parent graph is a
// simple chain A -> child -> E; the child graph contains a plain diamond
// X -> {Y,Z} -> W. Recursive flattening must coalesce the diamond inside the
// child into a single multi-queue block ("sync plan should do similar to child
// graph").
//
//   A                     child graph:   X
//   |                                    / \
//   child (diamond)                     Y   Z
//   |                                    \ /
//   E                                     W
#include "common.h"

__global__ void kA(int* base)                        { *base = 7; }
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

  void* aArgs[] = {&base};
  hipGraphNode_t A = addKernel(graph, (void*)kA, dim3(1), dim3(1), aArgs, nullptr, 0);

  // Child graph with a plain diamond X -> {Y,Z} -> W.
  hipGraph_t child;
  CHECK(hipGraphCreate(&child, 0));
  void* xA[] = {&base, &xo};
  hipGraphNode_t X = addKernel(child, (void*)kX, dim3(1), dim3(1), xA, nullptr, 0);
  void* yA[] = {&xo, &yo};
  hipGraphNode_t Y = addKernel(child, (void*)kY, dim3(1), dim3(1), yA, &X, 1);
  void* zA[] = {&xo, &zo};
  hipGraphNode_t Z = addKernel(child, (void*)kZ, dim3(1), dim3(1), zA, &X, 1);
  void* wA[] = {&yo, &zo, &wo};
  hipGraphNode_t YZ[2] = {Y, Z};
  addKernel(child, (void*)kW, dim3(1), dim3(1), wA, YZ, 2);

  hipGraphNode_t childNode;
  CHECK(hipGraphAddChildGraphNode(&childNode, graph, &A, 1, child));

  void* eArgs[] = {&wo, &eo};
  addKernel(graph, (void*)kE, dim3(1), dim3(1), eArgs, &childNode, 1);

  hipGraphExec_t exec;
  CHECK(hipGraphInstantiate(&exec, graph, nullptr, nullptr, 0));
  hipStream_t stream;
  CHECK(hipStreamCreate(&stream));

  // X=8, Y=18, Z=28, W=46, E=1046
  for (int run = 0; run < 5; ++run) {
    CHECK(hipMemset(base, 0, sizeof(int)));  CHECK(hipMemset(wo, 0, sizeof(int)));
    CHECK(hipMemset(eo, 0, sizeof(int)));
    CHECK(hipGraphLaunch(exec, stream));
    CHECK(hipStreamSynchronize(stream));
    int hw = -1, he = -1;
    CHECK(hipMemcpy(&hw, wo, sizeof(int), hipMemcpyDeviceToHost));
    CHECK(hipMemcpy(&he, eo, sizeof(int), hipMemcpyDeviceToHost));
    printf("run %d: W=%d (exp 46), E=%d (exp 1046)\n", run, hw, he);
    if (hw != 46 || he != 1046) { fprintf(stderr, "MISMATCH on run %d\n", run); return 1; }
  }

  printf("PASS\n");
  CHECK(hipStreamDestroy(stream));
  CHECK(hipGraphExecDestroy(exec));
  CHECK(hipGraphDestroy(graph));
  return 0;
}
