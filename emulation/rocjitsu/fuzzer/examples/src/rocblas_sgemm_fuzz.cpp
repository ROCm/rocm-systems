#include "rocfuzz_example_input.h"

#include <rocblas/rocblas.h>

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdio>
#include <vector>

__AFL_FUZZ_INIT();

namespace {

bool ok(rocblas_status status) { return status == rocblas_status_success; }

bool report_failure(const char *what, rocblas_status status) {
  if (ok(status))
    return false;
  std::fprintf(stderr, "%s failed: %d\n", what, static_cast<int>(status));
  return true;
}

struct RocblasHandle {
  rocblas_handle handle = nullptr;

  ~RocblasHandle() {
    if (handle != nullptr)
      (void)rocblas_destroy_handle(handle);
  }
};

int interesting_gemm_dim(rocfuzz::examples::ByteStream &stream) {
  static constexpr std::array<int, 18> kDims = {1,  2,  3,  4,  5,  7,
                                               8,  15, 16, 17, 31, 32,
                                               33, 48, 64, 96, 127, 128};
  return stream.pick(kDims);
}

rocblas_operation interesting_operation(rocfuzz::examples::ByteStream &stream) {
  static constexpr std::array<rocblas_operation, 4> kOps = {
      rocblas_operation_none, rocblas_operation_none, rocblas_operation_transpose,
      rocblas_operation_transpose};
  return stream.pick(kOps);
}

size_t matrix_count(int rows, int cols, int leading_dim) {
  return static_cast<size_t>(leading_dim) * static_cast<size_t>(cols);
}

int run_case(rocblas_handle handle, const std::vector<uint8_t> &input,
             rocfuzz::examples::PersistentHook begin,
             rocfuzz::examples::PersistentHook end) {
  using namespace rocfuzz::examples;

  ByteStream stream(input);
  const int m = interesting_gemm_dim(stream);
  const int n = interesting_gemm_dim(stream);
  const int k = interesting_gemm_dim(stream);
  const rocblas_operation trans_a = interesting_operation(stream);
  const rocblas_operation trans_b = interesting_operation(stream);

  const int a_rows = trans_a == rocblas_operation_none ? m : k;
  const int a_cols = trans_a == rocblas_operation_none ? k : m;
  const int b_rows = trans_b == rocblas_operation_none ? k : n;
  const int b_cols = trans_b == rocblas_operation_none ? n : k;
  const int lda = std::max(1, a_rows);
  const int ldb = std::max(1, b_rows);
  const int ldc = std::max(1, m);

  std::vector<float> host_a(matrix_count(a_rows, a_cols, lda));
  std::vector<float> host_b(matrix_count(b_rows, b_cols, ldb));
  std::vector<float> host_c(matrix_count(m, n, ldc));
  fill_floats(host_a, stream);
  fill_floats(host_b, stream);
  fill_floats(host_c, stream);

  DeviceBuffer<float> device_a;
  DeviceBuffer<float> device_b;
  DeviceBuffer<float> device_c;
  if (!device_a.allocate(host_a.size()) || !device_b.allocate(host_b.size()) ||
      !device_c.allocate(host_c.size()))
    return 0;

  if (!device_a.copy_from_host(host_a) || !device_b.copy_from_host(host_b) ||
      !device_c.copy_from_host(host_c))
    return 0;

  if (!ok(rocblas_set_pointer_mode(handle, rocblas_pointer_mode_host)))
    return 0;

  float alpha = stream.next_float();
  if (alpha == 0.0f)
    alpha = 1.0f;
  const float beta = stream.next_float();

  if (call_persistent_hook(begin, "rocjitsu_afl_persistent_begin") != 0)
    return 3;

  const rocblas_status status =
      rocblas_sgemm(handle, trans_a, trans_b, m, n, k, &alpha, device_a.get(), lda,
                    device_b.get(), ldb, &beta, device_c.get(), ldc);
  if (report_failure("rocblas_sgemm", status))
    return 0;

  if (end != nullptr) {
    if (call_persistent_hook(end, "rocjitsu_afl_persistent_end") != 0)
      return 3;
  } else {
    crash_on_hip_error("hipDeviceSynchronize", hipDeviceSynchronize());
  }

  std::vector<float> out(host_c.size());
  crash_on_hip_error("hipMemcpy D2H", device_c.copy_to_host(out) ? hipSuccess : hipErrorUnknown);

  volatile float sink = 0.0f;
  for (size_t i = 0; i < std::min<size_t>(out.size(), 16); ++i)
    sink += out[i];
  (void)sink;

  return 0;
}

} // namespace

int main(int argc, char **argv) {
  using namespace rocfuzz::examples;

  if (!have_hip_device())
    return 0;

  RocblasHandle handle;
  if (!ok(rocblas_create_handle(&handle.handle)))
    return 0;

#if defined(ROCFUZZ_EXAMPLE_PERSISTENT) && ROCFUZZ_EXAMPLE_PERSISTENT
  PersistentHook begin = load_persistent_hook("rocjitsu_afl_persistent_begin");
  PersistentHook end = load_persistent_hook("rocjitsu_afl_persistent_end");
  if (persistent_hooks_required() && (begin == nullptr || end == nullptr)) {
    std::fprintf(stderr, "rocjitsu AFL persistent hooks are not available\n");
    return 4;
  }

  __AFL_INIT();

#ifdef __AFL_FUZZ_TESTCASE_BUF
  unsigned char *afl_buf = __AFL_FUZZ_TESTCASE_BUF;
  while (__AFL_LOOP(1000)) {
    const size_t len = __AFL_FUZZ_TESTCASE_LEN;
    std::vector<uint8_t> input(afl_buf, afl_buf + len);
    const int rc = run_case(handle.handle, input, begin, end);
    if (rc != 0)
      return rc;
  }
#else
  while (__AFL_LOOP(1)) {
    const int rc = run_case(handle.handle, read_input(argc, argv), begin, end);
    if (rc != 0)
      return rc;
  }
#endif
#else
  const int rc = run_case(handle.handle, read_input(argc, argv), nullptr, nullptr);
  if (rc != 0)
    return rc;
#endif

  return 0;
}
