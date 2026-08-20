// Nested control-flow correctness for the GPU-resident scheduler.
//
// The fused-WHILE fast path only accepts a single WHILE around a single linear
// body block. These shapes are all outside that envelope and exercise the
// CFG-walking interpreter (HIP_GRAPH_WALK_LOOP=1) against the self-enqueueing
// terminator path, which is the reference:
//
//   1. WHILE-in-WHILE   outer loop whose body is itself a WHILE loop
//   2. IF-in-WHILE      loop body branches, both arms converge, then latches
//   3. SWITCH-in-WHILE  loop body dispatches on a rotating case index
//   4. loop-free IF     no loop at all: IF/ELSE at the top level
//   5. loop-free SWITCH  no loop at all: SWITCH at the top level
//
// Every case computes a closed-form expected value, so the same binary is the
// oracle on either path -- run it with and without HIP_GRAPH_WALK_LOOP=1.
#include "common.h"

// ---------------------------------------------------------------- kernels

__global__ void kOuterLatch(hipGraphConditionalHandle h, int* i, const int* n) {
  *i += 1;
  hipGraphSetConditional(h, (*i < *n) ? 1 : 0);
}

// Resets the inner loop's induction variable, then arms the inner condition.
__global__ void kInnerInit(hipGraphConditionalHandle h, int* j, const int* m) {
  *j = 0;
  hipGraphSetConditional(h, (*m > 0) ? 1 : 0);
}

__global__ void kInnerBody(hipGraphConditionalHandle h, int* j, int* acc,
                           const int* m) {
  *j += 1;
  *acc += 1;
  hipGraphSetConditional(h, (*j < *m) ? 1 : 0);
}

// Arms an IF on the parity of the outer induction variable.
__global__ void kSetIf(hipGraphConditionalHandle h, const int* i) {
  hipGraphSetConditional(h, (*i % 2 == 0) ? 1 : 0);
}

__global__ void kAddTo(int* dst, int v) { *dst += v; }

__global__ void kSetSwitch(hipGraphConditionalHandle h, const int* i,
                           int num_cases) {
  hipGraphSetConditional(h, *i % num_cases);
}

// ---------------------------------------------------------------- helpers

struct Dev {
  int* p = nullptr;
  explicit Dev(int init) {
    CHECK(hipMalloc(&p, sizeof(int)));
    CHECK(hipMemcpy(p, &init, sizeof(int), hipMemcpyHostToDevice));
  }
  int get() const {
    int v = -1;
    CHECK(hipMemcpy(&v, p, sizeof(int), hipMemcpyDeviceToHost));
    return v;
  }
  void set(int v) { CHECK(hipMemcpy(p, &v, sizeof(int), hipMemcpyHostToDevice)); }
};

static int g_failures = 0;

static void expect(const char* name, int got, int want) {
  printf("  %-28s got=%-8d exp=%-8d %s\n", name, got, want,
         (got == want) ? "ok" : "MISMATCH");
  if (got != want) g_failures++;
}

// Launch a graph `runs` times, resetting state before each launch via `reset`.
template <typename Reset>
static void runGraph(hipGraph_t g, int runs, Reset reset) {
  hipGraphExec_t exec;
  CHECK(hipGraphInstantiate(&exec, g, nullptr, nullptr, 0));
  hipStream_t s;
  CHECK(hipStreamCreate(&s));
  for (int r = 0; r < runs; ++r) {
    reset();
    CHECK(hipGraphLaunch(exec, s));
    CHECK(hipStreamSynchronize(s));
  }
  CHECK(hipStreamDestroy(s));
  CHECK(hipGraphExecDestroy(exec));
}

// ---------------------------------------------------------------- 1: WHILE in WHILE

// for i in 0..N-1: for j in 0..M-1: acc++     =>  acc == N*M
static void testWhileInWhile(int N, int M, int runs) {
  printf("WHILE-in-WHILE (N=%d, M=%d)\n", N, M);
  Dev i(0), j(0), acc(0), n(N), m(M);

  hipGraph_t g;
  CHECK(hipGraphCreate(&g, 0));
  hipGraphConditionalHandle outerH, innerH;
  CHECK(hipGraphConditionalHandleCreate(&outerH, g, 1, 0));
  CHECK(hipGraphConditionalHandleCreate(&innerH, g, 1, 0));

  // Outer body: reset+arm the inner loop, run it, then advance the outer index.
  hipGraph_t outerBody;
  CHECK(hipGraphCreate(&outerBody, 0));

  void* initArgs[] = {&innerH, &j.p, &m.p};
  hipGraphNode_t init =
      addKernel(outerBody, (void*)kInnerInit, dim3(1), dim3(1), initArgs, nullptr, 0);

  hipGraph_t innerBody;
  CHECK(hipGraphCreate(&innerBody, 0));
  void* innerArgs[] = {&innerH, &j.p, &acc.p, &m.p};
  addKernel(innerBody, (void*)kInnerBody, dim3(1), dim3(1), innerArgs, nullptr, 0);

  hipGraphNode_t innerWhile;
  CHECK(hipGraphAddConditionalNode(&innerWhile, outerBody, &init, 1, innerH,
                                   hipGraphCondTypeWhile, 1, &innerBody, 0));

  void* latchArgs[] = {&outerH, &i.p, &n.p};
  addKernel(outerBody, (void*)kOuterLatch, dim3(1), dim3(1), latchArgs, &innerWhile, 1);

  hipGraphNode_t outerWhile;
  CHECK(hipGraphAddConditionalNode(&outerWhile, g, nullptr, 0, outerH,
                                   hipGraphCondTypeWhile, 1, &outerBody, 0));

  runGraph(g, runs, [&]() { i.set(0); j.set(0); acc.set(0); });
  expect("acc == N*M", acc.get(), N * M);
  expect("i == N", i.get(), N);

  CHECK(hipGraphDestroy(innerBody));
  CHECK(hipGraphDestroy(outerBody));
  CHECK(hipGraphDestroy(g));
}

// ---------------------------------------------------------------- 2: IF in WHILE

// for i in 0..N-1: if (i even) even_acc += 1 else odd_acc += 1
static void testIfInWhile(int N, int runs) {
  printf("IF-in-WHILE (N=%d)\n", N);
  Dev i(0), evenAcc(0), oddAcc(0), n(N);

  hipGraph_t g;
  CHECK(hipGraphCreate(&g, 0));
  hipGraphConditionalHandle whileH, ifH;
  CHECK(hipGraphConditionalHandleCreate(&whileH, g, 1, 0));
  CHECK(hipGraphConditionalHandleCreate(&ifH, g, 0, 0));

  hipGraph_t body;
  CHECK(hipGraphCreate(&body, 0));

  void* setArgs[] = {&ifH, &i.p};
  hipGraphNode_t setIf =
      addKernel(body, (void*)kSetIf, dim3(1), dim3(1), setArgs, nullptr, 0);

  int one = 1;
  hipGraph_t thenG, elseG;
  CHECK(hipGraphCreate(&thenG, 0));
  CHECK(hipGraphCreate(&elseG, 0));
  void* thenArgs[] = {&evenAcc.p, &one};
  void* elseArgs[] = {&oddAcc.p, &one};
  addKernel(thenG, (void*)kAddTo, dim3(1), dim3(1), thenArgs, nullptr, 0);
  addKernel(elseG, (void*)kAddTo, dim3(1), dim3(1), elseArgs, nullptr, 0);

  hipGraph_t ifBodies[2] = {thenG, elseG};
  hipGraphNode_t ifNode;
  CHECK(hipGraphAddConditionalNode(&ifNode, body, &setIf, 1, ifH,
                                   hipGraphCondTypeIf, 2, ifBodies, 0));

  void* latchArgs[] = {&whileH, &i.p, &n.p};
  addKernel(body, (void*)kOuterLatch, dim3(1), dim3(1), latchArgs, &ifNode, 1);

  hipGraphNode_t whileNode;
  CHECK(hipGraphAddConditionalNode(&whileNode, g, nullptr, 0, whileH,
                                   hipGraphCondTypeWhile, 1, &body, 0));

  runGraph(g, runs, [&]() { i.set(0); evenAcc.set(0); oddAcc.set(0); });
  // i takes values 0..N-1 when the IF is evaluated (the latch increments after).
  expect("even iterations", evenAcc.get(), (N + 1) / 2);
  expect("odd iterations", oddAcc.get(), N / 2);

  CHECK(hipGraphDestroy(thenG));
  CHECK(hipGraphDestroy(elseG));
  CHECK(hipGraphDestroy(body));
  CHECK(hipGraphDestroy(g));
}

// ---------------------------------------------------------------- 3: SWITCH in WHILE

// for i in 0..N-1: case (i % 3) -> bucket[i % 3] += 1
static void testSwitchInWhile(int N, int runs) {
  printf("SWITCH-in-WHILE (N=%d, cases=3)\n", N);
  Dev i(0), b0(0), b1(0), b2(0), n(N);

  hipGraph_t g;
  CHECK(hipGraphCreate(&g, 0));
  hipGraphConditionalHandle whileH, swH;
  CHECK(hipGraphConditionalHandleCreate(&whileH, g, 1, 0));
  CHECK(hipGraphConditionalHandleCreate(&swH, g, 0, 0));

  hipGraph_t body;
  CHECK(hipGraphCreate(&body, 0));

  int numCases = 3;
  void* setArgs[] = {&swH, &i.p, &numCases};
  hipGraphNode_t setSw =
      addKernel(body, (void*)kSetSwitch, dim3(1), dim3(1), setArgs, nullptr, 0);

  int one = 1;
  hipGraph_t cases[3];
  int* targets[3] = {b0.p, b1.p, b2.p};
  void* caseArgs[3][2];
  for (int c = 0; c < 3; ++c) {
    CHECK(hipGraphCreate(&cases[c], 0));
    caseArgs[c][0] = &targets[c];
    caseArgs[c][1] = &one;
    addKernel(cases[c], (void*)kAddTo, dim3(1), dim3(1), caseArgs[c], nullptr, 0);
  }

  hipGraphNode_t swNode;
  CHECK(hipGraphAddConditionalNode(&swNode, body, &setSw, 1, swH,
                                   hipGraphCondTypeSwitch, 3, cases, 0));

  void* latchArgs[] = {&whileH, &i.p, &n.p};
  addKernel(body, (void*)kOuterLatch, dim3(1), dim3(1), latchArgs, &swNode, 1);

  hipGraphNode_t whileNode;
  CHECK(hipGraphAddConditionalNode(&whileNode, g, nullptr, 0, whileH,
                                   hipGraphCondTypeWhile, 1, &body, 0));

  runGraph(g, runs, [&]() { i.set(0); b0.set(0); b1.set(0); b2.set(0); });
  expect("case 0 hits", b0.get(), (N + 2) / 3);
  expect("case 1 hits", b1.get(), (N + 1) / 3);
  expect("case 2 hits", b2.get(), N / 3);

  for (int c = 0; c < 3; ++c) CHECK(hipGraphDestroy(cases[c]));
  CHECK(hipGraphDestroy(body));
  CHECK(hipGraphDestroy(g));
}

// ---------------------------------------------------------------- 4: loop-free IF

// No loop anywhere: the interpreter must walk an IF and return.
static void testLoopFreeIf(int runs) {
  printf("loop-free IF/ELSE\n");
  Dev sel(0), taken(0), notTaken(0);

  hipGraph_t g;
  CHECK(hipGraphCreate(&g, 0));
  hipGraphConditionalHandle ifH;
  CHECK(hipGraphConditionalHandleCreate(&ifH, g, 0, 0));

  // sel is even => take the "then" arm.
  void* setArgs[] = {&ifH, &sel.p};
  hipGraphNode_t setIf =
      addKernel(g, (void*)kSetIf, dim3(1), dim3(1), setArgs, nullptr, 0);

  int one = 1;
  hipGraph_t thenG, elseG;
  CHECK(hipGraphCreate(&thenG, 0));
  CHECK(hipGraphCreate(&elseG, 0));
  void* thenArgs[] = {&taken.p, &one};
  void* elseArgs[] = {&notTaken.p, &one};
  addKernel(thenG, (void*)kAddTo, dim3(1), dim3(1), thenArgs, nullptr, 0);
  addKernel(elseG, (void*)kAddTo, dim3(1), dim3(1), elseArgs, nullptr, 0);

  hipGraph_t bodies[2] = {thenG, elseG};
  hipGraphNode_t ifNode;
  CHECK(hipGraphAddConditionalNode(&ifNode, g, &setIf, 1, ifH,
                                   hipGraphCondTypeIf, 2, bodies, 0));

  const int runs_even = runs;
  runGraph(g, runs_even, [&]() { sel.set(0); taken.set(0); notTaken.set(0); });
  expect("then arm taken", taken.get(), 1);
  expect("else arm skipped", notTaken.get(), 0);

  runGraph(g, runs_even, [&]() { sel.set(1); taken.set(0); notTaken.set(0); });
  expect("else arm taken", notTaken.get(), 1);
  expect("then arm skipped", taken.get(), 0);

  CHECK(hipGraphDestroy(thenG));
  CHECK(hipGraphDestroy(elseG));
  CHECK(hipGraphDestroy(g));
}

// ---------------------------------------------------------------- 5: loop-free SWITCH

static void testLoopFreeSwitch(int runs) {
  printf("loop-free SWITCH\n");
  Dev sel(0), b0(0), b1(0), b2(0);

  hipGraph_t g;
  CHECK(hipGraphCreate(&g, 0));
  hipGraphConditionalHandle swH;
  CHECK(hipGraphConditionalHandleCreate(&swH, g, 0, 0));

  int numCases = 3;
  void* setArgs[] = {&swH, &sel.p, &numCases};
  hipGraphNode_t setSw =
      addKernel(g, (void*)kSetSwitch, dim3(1), dim3(1), setArgs, nullptr, 0);

  int one = 1;
  hipGraph_t cases[3];
  int* targets[3] = {b0.p, b1.p, b2.p};
  void* caseArgs[3][2];
  for (int c = 0; c < 3; ++c) {
    CHECK(hipGraphCreate(&cases[c], 0));
    caseArgs[c][0] = &targets[c];
    caseArgs[c][1] = &one;
    addKernel(cases[c], (void*)kAddTo, dim3(1), dim3(1), caseArgs[c], nullptr, 0);
  }

  hipGraphNode_t swNode;
  CHECK(hipGraphAddConditionalNode(&swNode, g, &setSw, 1, swH,
                                   hipGraphCondTypeSwitch, 3, cases, 0));

  for (int pick = 0; pick < 3; ++pick) {
    runGraph(g, runs, [&]() {
      sel.set(pick);
      b0.set(0); b1.set(0); b2.set(0);
    });
    char name[64];
    snprintf(name, sizeof(name), "sel=%d -> case %d", pick, pick);
    int got = (pick == 0) ? b0.get() : (pick == 1) ? b1.get() : b2.get();
    expect(name, got, 1);
    int others = b0.get() + b1.get() + b2.get() - got;
    expect("other cases skipped", others, 0);
  }

  for (int c = 0; c < 3; ++c) CHECK(hipGraphDestroy(cases[c]));
  CHECK(hipGraphDestroy(g));
}

// ----------------------------------------------------------------

int main(int argc, char** argv) {
  const int runs = (argc > 1) ? atoi(argv[1]) : 3;

  testWhileInWhile(5, 4, runs);
  testIfInWhile(7, runs);
  testSwitchInWhile(10, runs);
  testLoopFreeIf(runs);
  testLoopFreeSwitch(runs);

  printf("%s\n", g_failures == 0 ? "PASS" : "FAIL");
  return g_failures == 0 ? 0 : 1;
}
