#include "rj_fuzz_input.h"

#include <hip/hip_runtime_api.h>
#include <hip/hiprtc.h>
#include <unistd.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

__AFL_FUZZ_INIT();

namespace {

constexpr const char *kKernelSource = R"(
extern "C" __global__ void two_vector_add(const float *A,
                                          const float *B,
                                          const float *D,
                                          const float *E,
                                          float *C,
                                          float *F,
                                          unsigned N) {
  unsigned i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < N) {
    C[i] = A[i] + B[i];
    F[i] = D[i] + E[i];
  }
}
)";

bool ok(hiprtcResult status) { return status == HIPRTC_SUCCESS; }

std::string current_device_arch() {
  int device = 0;
  rj_fuzz::crash_on_hip_error("hipGetDevice", hipGetDevice(&device));

  hipDeviceProp_t props{};
  rj_fuzz::crash_on_hip_error("hipGetDeviceProperties", hipGetDeviceProperties(&props, device));

  std::string arch = props.gcnArchName;
  const size_t feature_delimiter = arch.find(':');
  if (feature_delimiter != std::string::npos)
    arch.resize(feature_delimiter);
  if (arch.empty()) {
    std::fprintf(stderr, "hipGetDeviceProperties returned an empty gcnArchName\n");
    std::abort();
  }
  return arch;
}

std::vector<char> compile_kernel() {
  hiprtcProgram program = nullptr;
  if (!ok(hiprtcCreateProgram(&program, kKernelSource, "two_vector_add.hip", 0, nullptr,
                              nullptr))) {
    std::fprintf(stderr, "hiprtcCreateProgram failed\n");
    std::abort();
  }

  const std::string arch_option = "--gpu-architecture=" + current_device_arch();
  const char *options[] = {arch_option.c_str(), "-O2"};
  const hiprtcResult compile_status = hiprtcCompileProgram(program, 2, options);

  size_t log_size = 0;
  (void)hiprtcGetProgramLogSize(program, &log_size);
  if (log_size > 1) {
    std::string log(log_size, '\0');
    (void)hiprtcGetProgramLog(program, log.data());
    std::fprintf(stderr, "%s\n", log.c_str());
  }

  if (!ok(compile_status)) {
    std::fprintf(stderr, "hiprtcCompileProgram failed: %s\n", hiprtcGetErrorString(compile_status));
    (void)hiprtcDestroyProgram(&program);
    std::abort();
  }

  size_t code_size = 0;
  if (!ok(hiprtcGetCodeSize(program, &code_size)) || code_size == 0) {
    std::fprintf(stderr, "hiprtcGetCodeSize failed\n");
    (void)hiprtcDestroyProgram(&program);
    std::abort();
  }

  std::vector<char> code(code_size);
  if (!ok(hiprtcGetCode(program, code.data()))) {
    std::fprintf(stderr, "hiprtcGetCode failed\n");
    (void)hiprtcDestroyProgram(&program);
    std::abort();
  }

  (void)hiprtcDestroyProgram(&program);
  return code;
}

struct HipModule {
  hipModule_t module = nullptr;
  hipFunction_t function = nullptr;

  explicit HipModule(const std::vector<char> &code) {
    rj_fuzz::crash_on_hip_error("hipModuleLoadData", hipModuleLoadData(&module, code.data()));
    rj_fuzz::crash_on_hip_error("hipModuleGetFunction",
                                hipModuleGetFunction(&function, module, "two_vector_add"));
  }

  ~HipModule() {
    if (module != nullptr)
      (void)hipModuleUnload(module);
  }
};

unsigned interesting_vector_length(rj_fuzz::ByteStream &stream) {
  static constexpr std::array<unsigned, 14> kLengths = {1,   7,   16,  31,  64,   127,  128,
                                                        255, 256, 511, 512, 1024, 2048, 4096};
  return stream.pick(kLengths);
}

int run_case(HipModule &module, const std::vector<uint8_t> &input) {
  using namespace rj_fuzz;

  ByteStream stream(input);
  unsigned n = interesting_vector_length(stream);

  std::vector<float> host_a(n);
  std::vector<float> host_b(n);
  std::vector<float> host_d(n);
  std::vector<float> host_e(n);
  fill_floats(host_a, stream);
  fill_floats(host_b, stream);
  fill_floats(host_d, stream);
  fill_floats(host_e, stream);

  DeviceBuffer<float> device_a;
  DeviceBuffer<float> device_b;
  DeviceBuffer<float> device_c;
  DeviceBuffer<float> device_d;
  DeviceBuffer<float> device_e;
  DeviceBuffer<float> device_f;
  if (!device_a.allocate(n) || !device_b.allocate(n) || !device_c.allocate(n) ||
      !device_d.allocate(n) || !device_e.allocate(n) || !device_f.allocate(n))
    return 0;

  if (!device_a.copy_from_host(host_a) || !device_b.copy_from_host(host_b) ||
      !device_d.copy_from_host(host_d) || !device_e.copy_from_host(host_e))
    return 0;

  constexpr unsigned block_size = 128;
  const unsigned grid_size = (n + block_size - 1) / block_size;
  float *a = device_a.get();
  float *b = device_b.get();
  float *c = device_c.get();
  float *d = device_d.get();
  float *e = device_e.get();
  float *f = device_f.get();
  void *args[] = {&a, &b, &d, &e, &c, &f, &n};
  crash_on_hip_error("hipModuleLaunchKernel",
                     hipModuleLaunchKernel(module.function, grid_size, 1, 1, block_size, 1, 1, 0,
                                           nullptr, args, nullptr));

  crash_on_hip_error("hipDeviceSynchronize", hipDeviceSynchronize());

  std::vector<float> host_c(n);
  std::vector<float> host_f(n);
  crash_on_hip_error("hipMemcpy C D2H",
                     device_c.copy_to_host(host_c) ? hipSuccess : hipErrorUnknown);
  crash_on_hip_error("hipMemcpy F D2H",
                     device_f.copy_to_host(host_f) ? hipSuccess : hipErrorUnknown);

  for (unsigned i = 0; i < n; ++i) {
    if (std::fabs(host_c[i] - (host_a[i] + host_b[i])) > 1.0e-5f ||
        std::fabs(host_f[i] - (host_d[i] + host_e[i])) > 1.0e-5f) {
      std::fprintf(stderr, "two_vector_add mismatch at %u\n", i);
      std::abort();
    }
  }

  volatile float sink = 0.0f;
  for (size_t i = 0; i < std::min<size_t>(host_c.size(), 16); ++i)
    sink += host_c[i] + host_f[i];
  (void)sink;

  return 0;
}

} // namespace

int main(int argc, char **argv) {
  if (!rj_fuzz::have_hip_device())
    return 0;

  const std::vector<char> code = compile_kernel();
  HipModule module(code);

#ifdef RJ_AFL_PERSISTENT_MODE
  __AFL_INIT();

#ifdef __AFL_FUZZ_TESTCASE_BUF
  unsigned char *afl_buf = __AFL_FUZZ_TESTCASE_BUF;
  while (__AFL_LOOP(1000)) {
    const size_t len = __AFL_FUZZ_TESTCASE_LEN;
    std::vector<uint8_t> input(afl_buf, afl_buf + len);
    const int rc = run_case(module, input);
    if (rc != 0)
      return rc;
  }
#else
  while (__AFL_LOOP(1)) {
    const int rc = run_case(module, rj_fuzz::read_input(argc, argv));
    if (rc != 0)
      return rc;
  }
#endif
#else
  const int rc = run_case(module, rj_fuzz::read_input(argc, argv));
  if (rc != 0)
    return rc;
#endif

  return 0;
}
