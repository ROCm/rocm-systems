// Perf check for the fork/join diamond on the reliable 2-lane path.
//
//        A
//       / \        two independent lanes B, C
//      B   C
//       \ /
//        D         D consumes both lanes
//
// Compares two ways of running the SAME diamond, back to back, many times:
//   (1) GPU-resident HIP graph  -> our command-buffer fork/join fast path
//   (2) Host-driven streams      -> classic multi-stream + event join
// Reports average per-launch latency (wall clock) and, for the graph, the
// on-GPU time via HIP events.
#include "common.h"
#include <chrono>

__global__ void kA(int* base)                         { *base = 3; }
__global__ void kLane(const int* base, int* out)      { *out = *base + 1; }
__global__ void kD(const int* b, const int* c, int* s){ *s = *b + *c; }

static double now_us() {
  using namespace std::chrono;
  return duration<double, std::micro>(steady_clock::now().time_since_epoch()).count();
}

int main(int argc, char** argv) {
  int iters = (argc > 1) ? atoi(argv[1]) : 300;
  int warmup = 50;

  int *base, *bOut, *cOut, *sum;
  CHECK(hipMalloc(&base, sizeof(int)));
  CHECK(hipMalloc(&bOut, sizeof(int)));
  CHECK(hipMalloc(&cOut, sizeof(int)));
  CHECK(hipMalloc(&sum,  sizeof(int)));

  hipStream_t tstream;
  CHECK(hipStreamCreate(&tstream));
  auto time_exec = [&](hipGraphExec_t gx) -> double {
    for (int i = 0; i < warmup; ++i) { CHECK(hipGraphLaunch(gx, tstream)); CHECK(hipStreamSynchronize(tstream)); }
    double a = now_us();
    for (int i = 0; i < iters; ++i) { CHECK(hipGraphLaunch(gx, tstream)); CHECK(hipStreamSynchronize(tstream)); }
    return (now_us() - a) / iters;
  };

  // Baseline A: single-kernel graph (pure GPU-resident scheduler overhead).
  hipGraph_t g1; CHECK(hipGraphCreate(&g1, 0));
  { void* a1[] = {&base}; addKernel(g1, (void*)kA, dim3(1), dim3(1), a1, nullptr, 0); }
  hipGraphExec_t e1; CHECK(hipGraphInstantiate(&e1, g1, nullptr, nullptr, 0));

  // Baseline B: linear 4-kernel chain A->B->C->D (no fork; sequential blocks).
  hipGraph_t g4; CHECK(hipGraphCreate(&g4, 0));
  {
    void* a1[] = {&base};              hipGraphNode_t n1 = addKernel(g4, (void*)kA, dim3(1), dim3(1), a1, nullptr, 0);
    void* a2[] = {&base, &bOut};       hipGraphNode_t n2 = addKernel(g4, (void*)kLane, dim3(1), dim3(1), a2, &n1, 1);
    void* a3[] = {&base, &cOut};       hipGraphNode_t n3 = addKernel(g4, (void*)kLane, dim3(1), dim3(1), a3, &n2, 1);
    void* a4[] = {&bOut, &cOut, &sum}; addKernel(g4, (void*)kD, dim3(1), dim3(1), a4, &n3, 1);
  }
  hipGraphExec_t e4; CHECK(hipGraphInstantiate(&e4, g4, nullptr, nullptr, 0));

  // ---------- Build the graph (GPU-resident fork/join path) ----------
  hipGraph_t graph;
  CHECK(hipGraphCreate(&graph, 0));
  void* aArgs[] = {&base};
  hipGraphNode_t A = addKernel(graph, (void*)kA, dim3(1), dim3(1), aArgs, nullptr, 0);
  void* bArgs[] = {&base, &bOut};
  void* cArgs[] = {&base, &cOut};
  hipGraphNode_t B = addKernel(graph, (void*)kLane, dim3(1), dim3(1), bArgs, &A, 1);
  hipGraphNode_t C = addKernel(graph, (void*)kLane, dim3(1), dim3(1), cArgs, &A, 1);
  hipGraphNode_t BC[2] = {B, C};
  void* dArgs[] = {&bOut, &cOut, &sum};
  addKernel(graph, (void*)kD, dim3(1), dim3(1), dArgs, BC, 2);

  hipGraphExec_t exec;
  CHECK(hipGraphInstantiate(&exec, graph, nullptr, nullptr, 0));

  hipStream_t stream;
  CHECK(hipStreamCreate(&stream));
  hipEvent_t evStart, evStop;
  CHECK(hipEventCreate(&evStart));
  CHECK(hipEventCreate(&evStop));

  // NOTE: the command-buffer exec has a single shared device ExecutionState, so
  // the same exec must complete before it is relaunched. We sync every launch.

  // Warmup
  for (int i = 0; i < warmup; ++i) {
    CHECK(hipGraphLaunch(exec, stream));
    CHECK(hipStreamSynchronize(stream));
  }

  // Timed: per-launch wall latency, split into host-enqueue vs GPU-wait.
  double host_sum = 0.0, wait_sum = 0.0;
  double t0 = now_us();
  for (int i = 0; i < iters; ++i) {
    double a = now_us();
    CHECK(hipGraphLaunch(exec, stream));
    double b = now_us();
    CHECK(hipStreamSynchronize(stream));
    double c = now_us();
    host_sum += (b - a);
    wait_sum += (c - b);
  }
  double graph_wall = (now_us() - t0) / iters;
  double graph_host = host_sum / iters;
  double graph_wait = wait_sum / iters;

  double graph_gpu = 0.0;  // (event-based GPU loop removed to conserve signals)

  // ---------- Host-driven baseline: streams + events ----------
  hipStream_t s0, s1;
  CHECK(hipStreamCreate(&s0));
  CHECK(hipStreamCreate(&s1));
  hipEvent_t eA, eC;
  CHECK(hipEventCreate(&eA));
  CHECK(hipEventCreate(&eC));

  auto launch_manual = [&]() {
    kA<<<1, 1, 0, s0>>>(base);
    CHECK(hipEventRecord(eA, s0));
    // lane B on s0, lane C on s1 (waits for A)
    kLane<<<1, 1, 0, s0>>>(base, bOut);
    CHECK(hipStreamWaitEvent(s1, eA, 0));
    kLane<<<1, 1, 0, s1>>>(base, cOut);
    CHECK(hipEventRecord(eC, s1));
    // join on s0
    CHECK(hipStreamWaitEvent(s0, eC, 0));
    kD<<<1, 1, 0, s0>>>(bOut, cOut, sum);
  };

  for (int i = 0; i < warmup; ++i) launch_manual();
  CHECK(hipStreamSynchronize(s0));

  t0 = now_us();
  for (int i = 0; i < iters; ++i) {
    launch_manual();
    CHECK(hipStreamSynchronize(s0));
  }
  double manual_wall = (now_us() - t0) / iters;

  double t_single = time_exec(e1);
  double t_linear = time_exec(e4);

  printf("iters=%d\n", iters);
  printf("  [decompose, same GPU-resident command-buffer path, wall us/launch]\n");
  printf("    single kernel  (A)              : %8.2f\n", t_single);
  printf("    linear chain   (A->B->C->D)     : %8.2f  (+%.2f vs single)\n",
         t_linear, t_linear - t_single);
  printf("    diamond fork/j (A->{B,C}->D)    : %8.2f  (+%.2f vs linear)\n",
         graph_wall, graph_wall - t_linear);
  printf("      of diamond: host-enqueue=%.2f  GPU-wait=%.2f  GPU(event)=%.2f\n",
         graph_host, graph_wait, graph_gpu);
  printf("  [baseline] manual streams+events  : %8.2f  (diamond speedup %.2fx)\n",
         manual_wall, manual_wall / graph_wall);
  return 0;
}
