#include "rocfuzz_example_input.h"

#include <miopen/miopen.h>

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdio>
#include <vector>

__AFL_FUZZ_INIT();

namespace {

bool ok(miopenStatus_t status) { return status == miopenStatusSuccess; }

bool report_failure(const char *what, miopenStatus_t status) {
  if (ok(status))
    return false;
  std::fprintf(stderr, "%s failed: %d\n", what, static_cast<int>(status));
  return true;
}

struct MiopenHandle {
  miopenHandle_t handle = nullptr;

  ~MiopenHandle() {
    if (handle != nullptr)
      (void)miopenDestroy(handle);
  }
};

struct TensorDescriptor {
  miopenTensorDescriptor_t desc = nullptr;

  ~TensorDescriptor() {
    if (desc != nullptr)
      (void)miopenDestroyTensorDescriptor(desc);
  }
};

struct ActivationDescriptor {
  miopenActivationDescriptor_t desc = nullptr;

  ~ActivationDescriptor() {
    if (desc != nullptr)
      (void)miopenDestroyActivationDescriptor(desc);
  }
};

miopenActivationMode_t interesting_activation(rocfuzz::examples::ByteStream &stream) {
  static constexpr std::array<miopenActivationMode_t, 10> kModes = {
      miopenActivationPASTHRU, miopenActivationLOGISTIC, miopenActivationTANH,
      miopenActivationRELU,    miopenActivationSOFTRELU, miopenActivationABS,
      miopenActivationCLIPPEDRELU, miopenActivationLEAKYRELU, miopenActivationELU,
      miopenActivationCLAMP};
  return stream.pick(kModes);
}

double positive_param(rocfuzz::examples::ByteStream &stream) {
  double value = static_cast<double>(stream.next_float());
  if (value < 0.0)
    value = -value;
  return value + 0.25;
}

void tune_activation_params(miopenActivationMode_t mode, double &alpha, double &beta,
                            double &gamma, rocfuzz::examples::ByteStream &stream) {
  switch (mode) {
  case miopenActivationCLIPPEDRELU:
    alpha = positive_param(stream);
    break;
  case miopenActivationLEAKYRELU:
  case miopenActivationELU:
    alpha = positive_param(stream);
    break;
  case miopenActivationCLAMP:
    alpha = -positive_param(stream);
    beta = positive_param(stream);
    break;
  case miopenActivationTANH:
    if (alpha == 0.0)
      alpha = 1.0;
    if (beta == 0.0)
      beta = 1.0;
    break;
  default:
    (void)gamma;
    break;
  }
}

int run_case(miopenHandle_t handle, const std::vector<uint8_t> &input,
             rocfuzz::examples::PersistentHook begin,
             rocfuzz::examples::PersistentHook end) {
  using namespace rocfuzz::examples;

  ByteStream stream(input);
  const int n = stream.pick(std::array<int, 4>{1, 1, 2, 4});
  const int c = stream.pick(std::array<int, 5>{1, 3, 4, 8, 16});
  const int h = stream.pick(std::array<int, 7>{1, 3, 4, 8, 16, 31, 32});
  const int w = stream.pick(std::array<int, 7>{1, 3, 4, 8, 16, 31, 32});
  const size_t element_count =
      static_cast<size_t>(n) * static_cast<size_t>(c) * static_cast<size_t>(h) *
      static_cast<size_t>(w);

  TensorDescriptor x_desc;
  TensorDescriptor y_desc;
  ActivationDescriptor activation;
  if (!ok(miopenCreateTensorDescriptor(&x_desc.desc)) ||
      !ok(miopenCreateTensorDescriptor(&y_desc.desc)) ||
      !ok(miopenCreateActivationDescriptor(&activation.desc)))
    return 0;

  if (!ok(miopenSet4dTensorDescriptor(x_desc.desc, miopenFloat, n, c, h, w)) ||
      !ok(miopenSet4dTensorDescriptor(y_desc.desc, miopenFloat, n, c, h, w)))
    return 0;

  const miopenActivationMode_t mode = interesting_activation(stream);
  double activation_alpha = static_cast<double>(stream.next_float());
  double activation_beta = static_cast<double>(stream.next_float());
  double activation_gamma = positive_param(stream);
  tune_activation_params(mode, activation_alpha, activation_beta, activation_gamma, stream);
  if (!ok(miopenSetActivationDescriptor(activation.desc, mode, activation_alpha, activation_beta,
                                        activation_gamma)))
    return 0;

  std::vector<float> x(element_count);
  std::vector<float> y(element_count);
  fill_floats(x, stream);
  fill_floats(y, stream);

  DeviceBuffer<float> device_x;
  DeviceBuffer<float> device_y;
  if (!device_x.allocate(x.size()) || !device_y.allocate(y.size()))
    return 0;
  if (!device_x.copy_from_host(x) || !device_y.copy_from_host(y))
    return 0;

  const float alpha = 1.0f;
  const float beta = 0.0f;

  if (call_persistent_hook(begin, "rocjitsu_afl_persistent_begin") != 0)
    return 3;

  const miopenStatus_t status =
      miopenActivationForward(handle, activation.desc, &alpha, x_desc.desc, device_x.get(), &beta,
                              y_desc.desc, device_y.get());
  if (report_failure("miopenActivationForward", status))
    return 0;

  if (end != nullptr) {
    if (call_persistent_hook(end, "rocjitsu_afl_persistent_end") != 0)
      return 3;
  } else {
    crash_on_hip_error("hipDeviceSynchronize", hipDeviceSynchronize());
  }

  std::vector<float> out(y.size());
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

  MiopenHandle handle;
  if (!ok(miopenCreate(&handle.handle)))
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
