// Salmon end-to-end HIP integration test.
//
// Loads a gfx950 vecadd kernel via hipModuleLoadData on gfx942 hardware.
// When run with the Salmon LD_PRELOAD shim (libsalmon_intercept.so) and
// Salmon-enabled HSA runtime, the ISA mismatch is transparently handled:
//
//   Layer 1 (shim):  PatchElfIsa patches e_flags so HIP accepts the ELF
//   Layer 2 (ROCR):  Salmon detects gfx950 in MSGPACK, raises → LLVM IR → gfx942
//
// Build:
//   hipcc -std=c++17 -O2 salmon_hip_test.cpp -o salmon_hip_test
//
// Run:
//   HSA_HOTSWAP_ISA_OVERRIDE=gfx942 HSA_HOTSWAP_RULES=/dev/null \
//   HSA_HOTSWAP_IR_RAISER=1 \
//   LD_PRELOAD=./libsalmon_intercept.so \
//   LD_LIBRARY_PATH=$ROCR_BUILD/rocr/lib \
//   ./salmon_hip_test [path/to/vecadd_gfx950.co]

#include <hip/hip_runtime.h>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <string>
#include <vector>

#define HIP_CHECK(call)                                                        \
  do {                                                                         \
    hipError_t err = (call);                                                   \
    if (err != hipSuccess) {                                                   \
      fprintf(stderr, "FAIL: %s at %s:%d — %s\n", #call, __FILE__, __LINE__,  \
              hipGetErrorString(err));                                          \
      return 1;                                                                \
    }                                                                          \
  } while (0)

static std::vector<uint8_t> readFile(const std::string &path) {
  std::ifstream f(path, std::ios::binary | std::ios::ate);
  if (!f) return {};
  auto sz = f.tellg();
  f.seekg(0);
  std::vector<uint8_t> buf(sz);
  f.read(reinterpret_cast<char *>(buf.data()), sz);
  return buf;
}

int main(int argc, char **argv) {
  const char *default_path = "vecadd_gfx950.co";
  const char *co_path = (argc > 1) ? argv[1] : default_path;

  printf("=== Salmon HIP Integration Test ===\n");
  printf("  Code object:             %s\n", co_path);
  printf("  HSA_HOTSWAP_ISA_OVERRIDE = %s\n",
         getenv("HSA_HOTSWAP_ISA_OVERRIDE")
             ? getenv("HSA_HOTSWAP_ISA_OVERRIDE")
             : "(unset)");
  printf("  HSA_HOTSWAP_IR_RAISER    = %s\n",
         getenv("HSA_HOTSWAP_IR_RAISER") ? getenv("HSA_HOTSWAP_IR_RAISER")
                                         : "(unset)");
  printf("  LD_PRELOAD               = %s\n",
         getenv("LD_PRELOAD") ? getenv("LD_PRELOAD") : "(unset)");

  auto coData = readFile(co_path);
  if (coData.empty()) {
    fprintf(stderr, "FAIL: cannot read %s\n", co_path);
    return 1;
  }
  printf("  Code object size:        %zu bytes\n", coData.size());

  hipDeviceProp_t props;
  HIP_CHECK(hipGetDeviceProperties(&props, 0));
  printf("  GPU:                     %s\n\n", props.name);

  // Load the gfx950 code object via standard HIP API.
  // The LD_PRELOAD shim patches e_flags so HIP accepts it.
  // The HSA runtime detects the real ISA from MSGPACK and triggers Salmon.
  printf("  Loading code object via hipModuleLoadData...\n");
  hipModule_t mod;
  HIP_CHECK(hipModuleLoadData(&mod, coData.data()));
  printf("  Module loaded.\n");

  hipFunction_t func;
  HIP_CHECK(hipModuleGetFunction(&func, mod, "vecadd"));
  printf("  Kernel 'vecadd' found.\n");

  const int N = 1024;
  const size_t bytes = N * sizeof(float);

  std::vector<float> h_A(N), h_B(N), h_C(N, 0.0f);
  for (int i = 0; i < N; i++) {
    h_A[i] = static_cast<float>(i);
    h_B[i] = static_cast<float>(i * 2);
  }

  float *d_A, *d_B, *d_C;
  HIP_CHECK(hipMalloc(&d_A, bytes));
  HIP_CHECK(hipMalloc(&d_B, bytes));
  HIP_CHECK(hipMalloc(&d_C, bytes));
  HIP_CHECK(hipMemcpy(d_A, h_A.data(), bytes, hipMemcpyHostToDevice));
  HIP_CHECK(hipMemcpy(d_B, h_B.data(), bytes, hipMemcpyHostToDevice));
  HIP_CHECK(hipMemset(d_C, 0, bytes));

  printf("\n  Dispatching vecadd(%d elements)...\n", N);
  unsigned block = 256;
  unsigned grid = (N + block - 1) / block;

  struct {
    const float *A;
    const float *B;
    float *C;
    int N;
  } args = {d_A, d_B, d_C, N};

  size_t argSize = sizeof(args);
  void *config[] = {HIP_LAUNCH_PARAM_BUFFER_POINTER, &args,
                    HIP_LAUNCH_PARAM_BUFFER_SIZE,     &argSize,
                    HIP_LAUNCH_PARAM_END};

  HIP_CHECK(hipModuleLaunchKernel(func, grid, 1, 1, block, 1, 1, 0, nullptr,
                                   nullptr, config));
  HIP_CHECK(hipDeviceSynchronize());

  HIP_CHECK(hipMemcpy(h_C.data(), d_C, bytes, hipMemcpyDeviceToHost));

  int errors = 0;
  for (int i = 0; i < N; i++) {
    float expected = static_cast<float>(i) + static_cast<float>(i * 2);
    if (std::fabs(h_C[i] - expected) > 1e-5f) {
      if (errors < 5)
        fprintf(stderr, "  MISMATCH at [%d]: got %f, expected %f\n", i, h_C[i],
                expected);
      errors++;
    }
  }

  printf("\n=== Result ===\n");
  if (errors == 0) {
    printf("  PASS: %d/%d elements correct\n", N, N);
    printf("  Salmon transpiled gfx950 -> gfx942 and executed correctly.\n");
  } else {
    printf("  FAIL: %d/%d errors\n", errors, N);
  }

  (void)hipFree(d_A);
  (void)hipFree(d_B);
  (void)hipFree(d_C);
  (void)hipModuleUnload(mod);
  return errors > 0 ? 1 : 0;
}
