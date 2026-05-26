#include "rocfuzz_example_input.h"

#include <rocrand/rocrand.h>

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdio>
#include <cstdint>
#include <vector>

__AFL_FUZZ_INIT();

namespace {

bool ok(rocrand_status status) { return status == ROCRAND_STATUS_SUCCESS; }

bool report_failure(const char *what, rocrand_status status) {
  if (ok(status))
    return false;
  std::fprintf(stderr, "%s failed: %d\n", what, static_cast<int>(status));
  return true;
}

struct RocrandGenerator {
  rocrand_generator generator = nullptr;

  ~RocrandGenerator() {
    if (generator != nullptr)
      (void)rocrand_destroy_generator(generator);
  }
};

rocrand_rng_type interesting_generator(rocfuzz::examples::ByteStream &stream) {
  static constexpr std::array<rocrand_rng_type, 4> kTypes = {
      ROCRAND_RNG_PSEUDO_PHILOX4_32_10, ROCRAND_RNG_PSEUDO_XORWOW,
      ROCRAND_RNG_PSEUDO_MRG32K3A, ROCRAND_RNG_PSEUDO_MRG31K3P};
  return stream.pick(kTypes);
}

size_t interesting_count(rocfuzz::examples::ByteStream &stream) {
  static constexpr std::array<size_t, 12> kCounts = {
      64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768, 65536, 131072};
  return stream.pick(kCounts);
}

uint64_t next_u64(rocfuzz::examples::ByteStream &stream) {
  uint64_t value = 0;
  for (unsigned i = 0; i < 8; ++i)
    value |= static_cast<uint64_t>(stream.next()) << (i * 8);
  return value;
}

int run_case(const std::vector<uint8_t> &input, rocfuzz::examples::PersistentHook begin,
             rocfuzz::examples::PersistentHook end) {
  using namespace rocfuzz::examples;

  ByteStream stream(input);
  RocrandGenerator generator;
  if (!ok(rocrand_create_generator(&generator.generator, interesting_generator(stream))))
    return 0;
  if (!ok(rocrand_set_seed(generator.generator, next_u64(stream))))
    return 0;

  const size_t count = interesting_count(stream);
  DeviceBuffer<float> output;
  if (!output.allocate(count))
    return 0;

  if (call_persistent_hook(begin, "rocjitsu_afl_persistent_begin") != 0)
    return 3;

  const rocrand_status status = rocrand_generate_uniform(generator.generator, output.get(), count);
  if (report_failure("rocrand_generate_uniform", status))
    return 0;

  if (end != nullptr) {
    if (call_persistent_hook(end, "rocjitsu_afl_persistent_end") != 0)
      return 3;
  } else {
    crash_on_hip_error("hipDeviceSynchronize", hipDeviceSynchronize());
  }

  std::vector<float> out(count);
  crash_on_hip_error("hipMemcpy D2H", output.copy_to_host(out) ? hipSuccess : hipErrorUnknown);

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
