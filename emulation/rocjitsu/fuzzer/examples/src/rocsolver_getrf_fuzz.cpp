#include "rocfuzz_example_input.h"

#include <rocblas/rocblas.h>
#include <rocsolver/rocsolver.h>

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

int interesting_factor_dim(rocfuzz::examples::ByteStream &stream) {
  static constexpr std::array<int, 14> kDims = {1,  2,  3,  4,  5,  7,  8,
                                               15, 16, 17, 24, 31, 32, 48};
  return stream.pick(kDims);
}

int run_case(rocblas_handle handle, const std::vector<uint8_t> &input,
             rocfuzz::examples::PersistentHook begin,
             rocfuzz::examples::PersistentHook end) {
  using namespace rocfuzz::examples;

  ByteStream stream(input);
  const rocblas_int m = interesting_factor_dim(stream);
  const rocblas_int n = interesting_factor_dim(stream);
  const rocblas_int lda = m + static_cast<rocblas_int>(stream.next() % 3);
  const size_t matrix_count = static_cast<size_t>(lda) * static_cast<size_t>(n);
  const rocblas_int pivots = std::min(m, n);

  std::vector<float> host_a(matrix_count);
  fill_floats(host_a, stream);
  for (rocblas_int i = 0; i < pivots; ++i) {
    if ((stream.next() & 3) != 0)
      host_a[static_cast<size_t>(i) + static_cast<size_t>(i) * lda] += 1.0f;
  }

  DeviceBuffer<float> device_a;
  DeviceBuffer<rocblas_int> device_ipiv;
  DeviceBuffer<rocblas_int> device_info;
  if (!device_a.allocate(host_a.size()) || !device_ipiv.allocate(pivots) ||
      !device_info.allocate(1))
    return 0;

  std::vector<rocblas_int> host_ipiv(pivots, 0);
  std::vector<rocblas_int> host_info(1, 0);
  if (!device_a.copy_from_host(host_a) || !device_ipiv.copy_from_host(host_ipiv) ||
      !device_info.copy_from_host(host_info))
    return 0;

  if (call_persistent_hook(begin, "rocjitsu_afl_persistent_begin") != 0)
    return 3;

  const rocblas_status status =
      rocsolver_sgetrf(handle, m, n, device_a.get(), lda, device_ipiv.get(), device_info.get());
  if (report_failure("rocsolver_sgetrf", status))
    return 0;

  if (end != nullptr) {
    if (call_persistent_hook(end, "rocjitsu_afl_persistent_end") != 0)
      return 3;
  } else {
    crash_on_hip_error("hipDeviceSynchronize", hipDeviceSynchronize());
  }

  std::vector<float> out_a(host_a.size());
  std::vector<rocblas_int> out_info(1);
  crash_on_hip_error("hipMemcpy A D2H",
                     device_a.copy_to_host(out_a) ? hipSuccess : hipErrorUnknown);
  crash_on_hip_error("hipMemcpy info D2H",
                     device_info.copy_to_host(out_info) ? hipSuccess : hipErrorUnknown);

  volatile float sink = static_cast<float>(out_info[0]);
  for (size_t i = 0; i < std::min<size_t>(out_a.size(), 16); ++i)
    sink += out_a[i];
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
