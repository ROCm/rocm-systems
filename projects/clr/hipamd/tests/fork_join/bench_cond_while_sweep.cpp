// Decompose the GPU-resident WHILE per-iteration cost by sweeping the number of
// dispatches per iteration. The body block is a chain of K trivial kernels; the
// last one sets the WHILE condition. Each iteration therefore issues K body
// dispatches + 1 fused condBranchWhile terminator = (K+1) dispatches.
//
// Measuring per-iter time vs (K+1) and linear-fitting gives:
//   slope    = per-dispatch cost through the GPU scheduler
//   intercept= fixed per-iteration overhead (condition read, block re-issue)
#include "common.h"
#include <chrono>
#include <vector>

__global__ void kNop(int* p) { (void)p; }
__global__ void kCond(hipGraphConditionalHandle h, int* counter, const int* limit) {
  *counter += 1;
  hipGraphSetConditional(h, (*counter < *limit) ? 1 : 0);
}

static double now_us() {
  using namespace std::chrono;
  return duration<double, std::micro>(steady_clock::now().time_since_epoch()).count();
}

// Build a WHILE graph whose body is a chain of (K-1) nop kernels + 1 cond kernel.
static double bench_K(int K, int limit, int iters, int warmup, int* counter, int* limitD, int* scratch) {
  hipGraph_t graph;
  CHECK(hipGraphCreate(&graph, 0));
  hipGraphConditionalHandle h;
  CHECK(hipGraphConditionalHandleCreate(&h, graph, 1, 0));

  hipGraph_t body;
  CHECK(hipGraphCreate(&body, 0));
  hipGraphNode_t prev = nullptr;
  for (int i = 0; i < K - 1; ++i) {
    void* a[] = {&scratch};
    prev = addKernel(body, (void*)kNop, dim3(1), dim3(1), a, prev ? &prev : nullptr, prev ? 1 : 0);
  }
  void* cArgs[] = {&h, &counter, &limitD};
  addKernel(body, (void*)kCond, dim3(1), dim3(1), cArgs, prev ? &prev : nullptr, prev ? 1 : 0);

  hipGraphNode_t whileNode;
  CHECK(hipGraphAddConditionalNode(&whileNode, graph, nullptr, 0, h,
                                   hipGraphCondTypeWhile, 1, &body, 0));

  hipGraphExec_t exec;
  CHECK(hipGraphInstantiate(&exec, graph, nullptr, nullptr, 0));
  hipStream_t stream;
  CHECK(hipStreamCreate(&stream));

  for (int i = 0; i < warmup; ++i) {
    CHECK(hipMemset(counter, 0, sizeof(int)));
    CHECK(hipGraphLaunch(exec, stream));
    CHECK(hipStreamSynchronize(stream));
  }
  int hc = -1;
  CHECK(hipMemcpy(&hc, counter, sizeof(int), hipMemcpyDeviceToHost));
  if (hc != limit) { fprintf(stderr, "K=%d MISMATCH counter=%d exp=%d\n", K, hc, limit); exit(1); }

  double t0 = now_us();
  for (int i = 0; i < iters; ++i) {
    CHECK(hipMemset(counter, 0, sizeof(int)));
    CHECK(hipGraphLaunch(exec, stream));
    CHECK(hipStreamSynchronize(stream));
  }
  double per_launch = (now_us() - t0) / iters;

  CHECK(hipStreamDestroy(stream));
  CHECK(hipGraphExecDestroy(exec));
  CHECK(hipGraphDestroy(body));
  CHECK(hipGraphDestroy(graph));
  return per_launch / limit;  // us per iteration
}

int main(int argc, char** argv) {
  const int limit  = (argc > 1) ? atoi(argv[1]) : 500;
  const int iters  = (argc > 2) ? atoi(argv[2]) : 100;
  const int warmup = 30;

  int *counter, *limitD, *scratch;
  CHECK(hipMalloc(&counter, sizeof(int)));
  CHECK(hipMalloc(&limitD, sizeof(int)));
  CHECK(hipMalloc(&scratch, sizeof(int)));
  CHECK(hipMemcpy(limitD, &limit, sizeof(int), hipMemcpyHostToDevice));

  printf("WHILE per-iteration cost vs dispatches/iter (limit=%d, launches=%d)\n", limit, iters);
  printf("  K(body kernels)  dispatches/iter   us/iter\n");
  std::vector<std::pair<int,double>> pts;
  for (int K = 1; K <= 5; ++K) {
    double us_iter = bench_K(K, limit, iters, warmup, counter, limitD, scratch);
    int disp = K + 1;  // K body + 1 fused terminator
    pts.push_back({disp, us_iter});
    printf("       %d              %d            %8.3f\n", K, disp, us_iter);
  }

  // Least-squares linear fit: us_iter = slope*disp + intercept
  double n = pts.size(), sx = 0, sy = 0, sxx = 0, sxy = 0;
  for (auto& p : pts) { sx += p.first; sy += p.second; sxx += (double)p.first * p.first; sxy += (double)p.first * p.second; }
  double slope = (n * sxy - sx * sy) / (n * sxx - sx * sx);
  double intercept = (sy - slope * sx) / n;
  printf("\n  fit: us/iter = %.3f * dispatches + %.3f\n", slope, intercept);
  printf("  => per-dispatch cost ~ %.3f us ; fixed per-iter overhead ~ %.3f us\n", slope, intercept);
  return 0;
}
