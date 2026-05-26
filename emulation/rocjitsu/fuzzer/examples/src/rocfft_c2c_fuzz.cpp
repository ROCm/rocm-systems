#include "rocfuzz_example_input.h"

#include <rocfft/rocfft.h>

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdio>
#include <vector>

__AFL_FUZZ_INIT();

namespace {

bool ok(rocfft_status status) { return status == rocfft_status_success; }

struct RocfftSetup {
  bool ready = false;

  RocfftSetup() : ready(ok(rocfft_setup())) {}

  ~RocfftSetup() {
    if (ready)
      (void)rocfft_cleanup();
  }
};

struct RocfftPlan {
  rocfft_plan plan = nullptr;

  ~RocfftPlan() {
    if (plan != nullptr)
      (void)rocfft_plan_destroy(plan);
  }
};

struct RocfftExecutionInfo {
  rocfft_execution_info info = nullptr;

  ~RocfftExecutionInfo() {
    if (info != nullptr)
      (void)rocfft_execution_info_destroy(info);
  }
};

size_t interesting_fft_length(rocfuzz::examples::ByteStream &stream) {
  static constexpr std::array<size_t, 16> kLengths = {8,   16,  32,  64,  81,  128,
                                                      192, 256, 384, 512, 768, 1024,
                                                      1536, 2048, 4096, 8192};
  return stream.pick(kLengths);
}

int run_case(const std::vector<uint8_t> &input, rocfuzz::examples::PersistentHook begin,
             rocfuzz::examples::PersistentHook end) {
  using namespace rocfuzz::examples;

  ByteStream stream(input);
  const size_t length = interesting_fft_length(stream);
  const size_t batch = stream.pick(std::array<size_t, 4>{1, 1, 2, 4});
  const bool inverse = (stream.next() & 1) != 0;
  const size_t complex_values = length * batch;

  std::vector<float> host_data(complex_values * 2);
  fill_floats(host_data, stream);

  DeviceBuffer<float> device_data;
  if (!device_data.allocate(host_data.size()) || !device_data.copy_from_host(host_data))
    return 0;

  RocfftPlan plan;
  const rocfft_transform_type transform =
      inverse ? rocfft_transform_type_complex_inverse : rocfft_transform_type_complex_forward;
  if (!ok(rocfft_plan_create(&plan.plan, rocfft_placement_inplace, transform,
                             rocfft_precision_single, 1, &length, batch, nullptr)))
    return 0;

  size_t work_buffer_size = 0;
  if (!ok(rocfft_plan_get_work_buffer_size(plan.plan, &work_buffer_size)))
    return 0;

  DeviceBuffer<uint8_t> work_buffer;
  RocfftExecutionInfo info;
  if (!ok(rocfft_execution_info_create(&info.info)))
    return 0;
  if (work_buffer_size != 0) {
    if (!work_buffer.allocate(work_buffer_size))
      return 0;
    if (!ok(rocfft_execution_info_set_work_buffer(info.info, work_buffer.get(), work_buffer_size)))
      return 0;
  }

  if (call_persistent_hook(begin, "rocjitsu_afl_persistent_begin") != 0)
    return 3;

  void *in_buffers[] = {device_data.get()};
  const rocfft_status status = rocfft_execute(plan.plan, in_buffers, nullptr, info.info);
  if (!ok(status))
    return 0;

  if (end != nullptr) {
    if (call_persistent_hook(end, "rocjitsu_afl_persistent_end") != 0)
      return 3;
  } else {
    crash_on_hip_error("hipDeviceSynchronize", hipDeviceSynchronize());
  }

  std::vector<float> out(host_data.size());
  crash_on_hip_error("hipMemcpy D2H", device_data.copy_to_host(out) ? hipSuccess : hipErrorUnknown);

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

  RocfftSetup setup;
  if (!setup.ready)
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
    const int rc = run_case(input, begin, end);
    if (rc != 0)
      return rc;
  }
#else
  while (__AFL_LOOP(1)) {
    const int rc = run_case(read_input(argc, argv), begin, end);
    if (rc != 0)
      return rc;
  }
#endif
#else
  const int rc = run_case(read_input(argc, argv), nullptr, nullptr);
  if (rc != 0)
    return rc;
#endif

  return 0;
}
