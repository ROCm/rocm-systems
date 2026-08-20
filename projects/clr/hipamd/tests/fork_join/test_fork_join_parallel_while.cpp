// Parallel WHILE: each loop iteration forks into two lanes, joins, then the
// join node updates the loop counter and re-tests the WHILE condition. The
// fork/join structure is re-issued (and its atomic join counter re-armed)
// every iteration by the WHILE self-loop.
//
//   WHILE (counter < limit):
//        A              (fork)
//       / \
//      B   C            lane 0: accA += 1   lane 1: accB += 2
//       \ /
//        D              (join) counter++, accSum = accA+accB, set condition
#include "common.h"

__global__ void kA(int* /*unused*/) {}
__global__ void kB(int* accA) { *accA += 1; }
__global__ void kC(int* accB) { *accB += 2; }
__global__ void kD(hipGraphConditionalHandle h, int* counter, const int* limit,
                   int* accSum, const int* accA, const int* accB) {
  *counter += 1;
  *accSum = *accA + *accB;
  hipGraphSetConditional(h, (*counter < *limit) ? 1 : 0);
}

int main() {
  const int limit = 4;  // 4 iterations

  int *dummy, *counter, *limitD, *accSum, *accA, *accB;
  CHECK(hipMalloc(&dummy, sizeof(int)));
  CHECK(hipMalloc(&counter, sizeof(int)));
  CHECK(hipMalloc(&limitD, sizeof(int)));
  CHECK(hipMalloc(&accSum, sizeof(int)));
  CHECK(hipMalloc(&accA, sizeof(int)));
  CHECK(hipMalloc(&accB, sizeof(int)));
  CHECK(hipMemcpy(limitD, &limit, sizeof(int), hipMemcpyHostToDevice));

  hipGraph_t graph;
  CHECK(hipGraphCreate(&graph, 0));

  hipGraphConditionalHandle whileHandle;
  CHECK(hipGraphConditionalHandleCreate(&whileHandle, graph, 1, 0));

  hipGraph_t body;
  CHECK(hipGraphCreate(&body, 0));

  void* aArgs[] = {&dummy};
  hipGraphNode_t A = addKernel(body, (void*)kA, dim3(1), dim3(1), aArgs, nullptr, 0);

  void* bArgs[] = {&accA};
  hipGraphNode_t B = addKernel(body, (void*)kB, dim3(1), dim3(1), bArgs, &A, 1);

  void* cArgs[] = {&accB};
  hipGraphNode_t C = addKernel(body, (void*)kC, dim3(1), dim3(1), cArgs, &A, 1);

  void* dArgs[] = {&whileHandle, &counter, &limitD, &accSum, &accA, &accB};
  hipGraphNode_t deps[2] = {B, C};
  addKernel(body, (void*)kD, dim3(1), dim3(1), dArgs, deps, 2);

  hipGraphNode_t whileNode;
  CHECK(hipGraphAddConditionalNode(&whileNode, graph, nullptr, 0, whileHandle,
                                   hipGraphCondTypeWhile, 1, &body, 0));

  hipGraphExec_t exec;
  CHECK(hipGraphInstantiate(&exec, graph, nullptr, nullptr, 0));

  hipStream_t stream;
  CHECK(hipStreamCreate(&stream));

  for (int run = 0; run < 3; ++run) {
    CHECK(hipMemset(counter, 0, sizeof(int)));
    CHECK(hipMemset(accSum, 0, sizeof(int)));
    CHECK(hipMemset(accA, 0, sizeof(int)));
    CHECK(hipMemset(accB, 0, sizeof(int)));

    CHECK(hipGraphLaunch(exec, stream));
    CHECK(hipStreamSynchronize(stream));

    int hc = -1, ha = -1, hb = -1, hs = -1;
    CHECK(hipMemcpy(&hc, counter, sizeof(int), hipMemcpyDeviceToHost));
    CHECK(hipMemcpy(&ha, accA, sizeof(int), hipMemcpyDeviceToHost));
    CHECK(hipMemcpy(&hb, accB, sizeof(int), hipMemcpyDeviceToHost));
    CHECK(hipMemcpy(&hs, accSum, sizeof(int), hipMemcpyDeviceToHost));

    printf("run %d: counter=%d (exp %d), accA=%d (exp %d), accB=%d (exp %d), "
           "accSum=%d (exp %d)\n",
           run, hc, limit, ha, limit, hb, 2 * limit, hs, 3 * limit);
    if (hc != limit || ha != limit || hb != 2 * limit || hs != 3 * limit) {
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
