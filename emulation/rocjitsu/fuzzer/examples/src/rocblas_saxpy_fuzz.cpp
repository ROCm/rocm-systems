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

struct RocblasHandle {
  rocblas_handle handle = nullptr;

  ~RocblasHandle() {
    if (handle != nullptr)
      (void)rocblas_destroy_handle(handle);
  }
};

int interesting_vector_length(rocfuzz::examples::ByteStream &stream) {
  static constexpr std::array<int, 18> kLengths = {
      1, 2, 7, 16, 31, 64, 127, 128, 255, 256, 511, 512, 1023, 1024, 2048, 4096,
      8192, 16384};
  return stream.pick(kLengths);
}

int run_case(rocblas_handle handle, const std::vector<uint8_t> &input,
             rocfuzz::examples::PersistentHook begin,
             rocfuzz::examples::PersistentHook end) {
  using namespace rocfuzz::examples;

  ByteStream stream(input);
  const int n = interesting_vector_length(stream);
  const int incx = stream.pick(std::array<int, 6>{1, 1, 2, 3, 4, 8});
  const int incy = stream.pick(std::array<int, 6>{1, 1, 2, 3, 4, 8});
  const size_t x_count = static_cast<size_t>(1 + (n - 1) * incx);
  const size_t y_count = static_cast<size_t>(1 + (n - 1) * incy);

  std::vector<float> host_x(x_count);
  std::vector<float> host_y(y_count);
  fill_floats(host_x, stream);
  fill_floats(host_y, stream);

  DeviceBuffer<float> device_x;
  DeviceBuffer<float> device_y;
  if (!device_x.allocate(host_x.size()) || !device_y.allocate(host_y.size()))
    return 0;

  if (!device_x.copy_from_host(host_x) || !device_y.copy_from_host(host_y))
    return 0;

  if (!ok(rocblas_set_pointer_mode(handle, rocblas_pointer_mode_host)))
    return 0;

  float alpha = stream.next_float();
  if (alpha == 0.0f)
    alpha = 1.0f;

  if (call_persistent_hook(begin, "rocjitsu_afl_persistent_begin") != 0)
    return 3;

  const rocblas_status status = rocblas_saxpy(handle, n, &alpha, device_x.get(), incx,
                                             device_y.get(), incy);
  if (!ok(status))
    return 0;

  if (end != nullptr) {
    if (call_persistent_hook(end, "rocjitsu_afl_persistent_end") != 0)
      return 3;
  } else {
    crash_on_hip_error("hipDeviceSynchronize", hipDeviceSynchronize());
  }

  std::vector<float> out(host_y.size());
  crash_on_hip_error("hipMemcpy D2H", device_y.copy_to_host(out) ? hipSuccess : hipErrorUnknown);

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
