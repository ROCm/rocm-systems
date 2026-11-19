#include <rccl/rccl.h>
#include <charconv>
#include <cstdio>
#include <cstdlib>
#include <limits>
#include <string_view>
#include <vector>

static void check(hipError_t status, const char* operation) {
  if (status != hipSuccess) {
    std::fprintf(stderr, "%s: %s\n", operation, hipGetErrorString(status));
    std::exit(1);
  }
}
static void check(ncclResult_t status, const char* operation) {
  if (status != ncclSuccess) {
    std::fprintf(stderr, "%s: %s\n", operation, ncclGetErrorString(status));
    std::exit(2);
  }
}
int main(int argc, char** argv) {
  std::setvbuf(stdout, nullptr, _IONBF, 0);
  int device = 0;
  size_t count = 1024;
  for (int arg = 1; arg < argc; ++arg) {
    const std::string_view option(argv[arg]);
    if (option == "--help") {
      std::puts("Usage: rccl_windows_smoke [--device ordinal] [--count elements]");
      return 0;
    }
    if ((option != "--device" && option != "--count") || arg + 1 == argc) {
      std::fprintf(stderr, "Expected --device ordinal or --count elements.\n");
      return 64;
    }
    const std::string_view value(argv[++arg]);
    size_t parsed = 0;
    const auto result = std::from_chars(value.data(), value.data() + value.size(), parsed);
    if (result.ec != std::errc() || result.ptr != value.data() + value.size() ||
        (option == "--device" && parsed > static_cast<size_t>(std::numeric_limits<int>::max())) ||
        (option == "--count" && (parsed == 0 || parsed > std::numeric_limits<size_t>::max() / sizeof(float)))) {
      std::fprintf(stderr, "Invalid value for %.*s.\n", static_cast<int>(option.size()), option.data());
      return 64;
    }
    if (option == "--device") device = static_cast<int>(parsed);
    else count = parsed;
  }
  int devices = 0;
  const hipError_t discovery = hipGetDeviceCount(&devices);
  if (discovery == hipErrorNoDevice || (discovery == hipSuccess && devices == 0)) {
    std::puts("SKIP: no HIP GPU is available.");
    return 77;
  }
  check(discovery, "GPU discovery");
  if (device >= devices) {
    std::fprintf(stderr, "Device %d is unavailable (%d HIP devices found).\n", device, devices);
    return 64;
  }
  check(hipSetDevice(device), "select GPU");
  int version = 0;
  check(ncclGetVersion(&version), "RCCL version");
  std::printf("RCCL version: %d\n", version);
  ncclComm_t comm = nullptr;
  check(ncclCommInitAll(&comm, 1, &device), "initialize communicator");
  std::vector<float> input(count), output(count, -1.0f);
  for (size_t i = 0; i < count; ++i) input[i] = static_cast<float>(i % 251);
  float *send = nullptr, *receive = nullptr;
  check(hipMalloc(&send, input.size() * sizeof(float)), "allocate input");
  check(hipMalloc(&receive, output.size() * sizeof(float)), "allocate output");
  check(hipMemcpy(send, input.data(), input.size() * sizeof(float), hipMemcpyHostToDevice), "upload");
  hipStream_t stream;
  check(hipStreamCreate(&stream), "create stream");
  check(ncclAllReduce(send, receive, input.size(), ncclFloat32, ncclSum, comm, stream), "AllReduce");
  check(hipStreamSynchronize(stream), "synchronize");
  check(hipMemcpy(output.data(), receive, output.size() * sizeof(float), hipMemcpyDeviceToHost), "download");
  for (size_t i = 0; i < count; ++i) {
    if (output[i] != input[i]) {
      std::fprintf(stderr, "Mismatch at %zu: %g != %g\n", i, output[i], input[i]);
      return 3;
    }
  }
  check(ncclCommDestroy(comm), "destroy communicator");
  check(hipStreamDestroy(stream), "destroy stream");
  check(hipFree(send), "free input");
  check(hipFree(receive), "free output");
  std::printf("PASS: RCCL single-GPU float32 AllReduce verified (%zu elements, device %d).\n", count, device);
}
