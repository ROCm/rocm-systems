#include <rccl/rccl.h>
#include <cstdio>
#include <cstdlib>
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
int main() {
  std::setvbuf(stdout, nullptr, _IONBF, 0);
  int version = 0;
  check(ncclGetVersion(&version), "RCCL version");
  std::printf("RCCL version: %d\n", version);
  ncclComm_t comm = nullptr;
  int device = 0;
  check(ncclCommInitAll(&comm, 1, &device), "initialize communicator");
  std::vector<float> input(1024), output(1024, -1.0f);
  for (int i = 0; i < 1024; ++i) input[i] = static_cast<float>(i);
  float *send = nullptr, *receive = nullptr;
  check(hipMalloc(&send, input.size() * sizeof(float)), "allocate input");
  check(hipMalloc(&receive, output.size() * sizeof(float)), "allocate output");
  check(hipMemcpy(send, input.data(), input.size() * sizeof(float), hipMemcpyHostToDevice), "upload");
  hipStream_t stream;
  check(hipStreamCreate(&stream), "create stream");
  check(ncclAllReduce(send, receive, input.size(), ncclFloat32, ncclSum, comm, stream), "AllReduce");
  check(hipStreamSynchronize(stream), "synchronize");
  check(hipMemcpy(output.data(), receive, output.size() * sizeof(float), hipMemcpyDeviceToHost), "download");
  for (int i = 0; i < 1024; ++i) {
    if (output[i] != input[i]) {
      std::fprintf(stderr, "Mismatch at %d: %g != %g\n", i, output[i], input[i]);
      return 3;
    }
  }
  check(ncclCommDestroy(comm), "destroy communicator");
  check(hipStreamDestroy(stream), "destroy stream");
  check(hipFree(send), "free input");
  check(hipFree(receive), "free output");
  std::puts("PASS: RCCL single-GPU float32 AllReduce verified (1024 elements).");
}
