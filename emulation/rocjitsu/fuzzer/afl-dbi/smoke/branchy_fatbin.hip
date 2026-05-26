#include <hip/hip_runtime.h>

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <atomic>
#include <fstream>
#include <thread>
#include <vector>

#define HIP_CHECK(expr)                                                                           \
  do {                                                                                             \
    hipError_t err__ = (expr);                                                                     \
    if (err__ != hipSuccess) {                                                                     \
      fprintf(stderr, "HIP error at %s:%d: %s\n", __FILE__, __LINE__, hipGetErrorString(err__));   \
      return 1;                                                                                    \
    }                                                                                              \
  } while (false)

__global__ void fatbin_kernel_a(const uint8_t *input, uint32_t *scratch) {
  uint32_t lane = static_cast<uint32_t>(threadIdx.x);
  uint32_t value = input[lane & 7] + lane;
  for (uint32_t i = 0; i < 8; ++i) {
    if (((value + i) & 3u) == 0)
      value ^= (lane << (i & 3));
    else if ((value & 1u) != 0)
      value += i * 17u;
    else
      value = (value << 1) ^ (i + 13u);
  }
  scratch[lane] = value;
}

__global__ void fatbin_kernel_b(const uint32_t *scratch, uint32_t *output) {
  uint32_t lane = static_cast<uint32_t>(threadIdx.x);
  uint32_t value = scratch[lane];
  if ((value & 0x10u) != 0) {
    value ^= scratch[(lane + 3) & 63u];
  } else {
    value += scratch[(lane + 17) & 63u];
  }
  if ((value % 7u) == 0)
    atomicAdd(output, value ^ 0x13579u);
  else
    atomicAdd(output, value + 0x2468u);
}

std::vector<uint8_t> read_input(const char *path) {
  std::ifstream file(path, std::ios::binary);
  if (!file)
    return {};
  std::vector<uint8_t> data(8, 0);
  file.read(reinterpret_cast<char *>(data.data()), static_cast<std::streamsize>(data.size()));
  return data;
}

int synchronize_after_launches() {
  const char *mode = getenv("ROCFUZZ_FATBIN_SYNC");
  if (mode != nullptr && strcmp(mode, "none") == 0)
    return 0;
  if (mode != nullptr && strcmp(mode, "stream") == 0) {
    HIP_CHECK(hipStreamSynchronize(nullptr));
    return 0;
  }
  if (mode != nullptr && strcmp(mode, "event") == 0) {
    hipEvent_t event = nullptr;
    HIP_CHECK(hipEventCreate(&event));
    HIP_CHECK(hipEventRecord(event, nullptr));
    HIP_CHECK(hipEventSynchronize(event));
    HIP_CHECK(hipEventDestroy(event));
    return 0;
  }
  HIP_CHECK(hipDeviceSynchronize());
  return 0;
}

int launch_kernel_b_once(const uint32_t *scratch, uint32_t *output) {
  hipLaunchKernelGGL(fatbin_kernel_b, dim3(1), dim3(64), 0, 0, scratch, output);
  HIP_CHECK(hipGetLastError());
  return 0;
}

int launch_kernel_b_concurrently(const uint32_t *scratch, uint32_t *output) {
  std::atomic<int> ready{0};
  std::atomic<bool> start{false};
  int results[2] = {0, 0};

  auto worker = [&](int index) {
    ready.fetch_add(1, std::memory_order_release);
    while (!start.load(std::memory_order_acquire))
      std::this_thread::yield();
    results[index] = launch_kernel_b_once(scratch, output);
  };

  std::thread first(worker, 0);
  std::thread second(worker, 1);
  while (ready.load(std::memory_order_acquire) != 2)
    std::this_thread::yield();
  start.store(true, std::memory_order_release);
  first.join();
  second.join();
  if (results[0] != 0)
    return results[0];
  return results[1];
}

int main(int argc, char **argv) {
  if (argc != 2) {
    fprintf(stderr, "usage: %s <input>\n", argv[0]);
    return 2;
  }
  std::vector<uint8_t> input = read_input(argv[1]);
  if (input.empty()) {
    fprintf(stderr, "failed to read input %s\n", argv[1]);
    return 1;
  }

  uint8_t *device_input = nullptr;
  uint32_t *scratch = nullptr;
  uint32_t *output = nullptr;
  HIP_CHECK(hipMalloc(reinterpret_cast<void **>(&device_input), input.size()));
  HIP_CHECK(hipMalloc(reinterpret_cast<void **>(&scratch), sizeof(uint32_t) * 64));
  HIP_CHECK(hipMalloc(reinterpret_cast<void **>(&output), sizeof(uint32_t)));
  HIP_CHECK(hipMemcpy(device_input, input.data(), input.size(), hipMemcpyHostToDevice));
  HIP_CHECK(hipMemset(output, 0, sizeof(uint32_t)));

  if (getenv("ROCFUZZ_SKIP_FATBIN_KERNEL_A") == nullptr) {
    hipLaunchKernelGGL(fatbin_kernel_a, dim3(1), dim3(64), 0, 0, device_input, scratch);
    HIP_CHECK(hipGetLastError());
  } else {
    HIP_CHECK(hipMemset(scratch, 0, sizeof(uint32_t) * 64));
  }
  if (getenv("ROCFUZZ_SKIP_FATBIN_KERNEL_B") == nullptr) {
    if (getenv("ROCFUZZ_CONCURRENT_FATBIN_KERNEL_B") != nullptr) {
      if (launch_kernel_b_concurrently(scratch, output) != 0)
        return 1;
    } else if (launch_kernel_b_once(scratch, output) != 0) {
      return 1;
    }
    if (getenv("ROCFUZZ_REPEAT_FATBIN_KERNEL_B") != nullptr) {
      if (launch_kernel_b_once(scratch, output) != 0)
        return 1;
    }
  }

  if (synchronize_after_launches() != 0)
    return 1;

  uint32_t host_output = 0;
  HIP_CHECK(hipMemcpy(&host_output, output, sizeof(host_output), hipMemcpyDeviceToHost));
  printf("fatbin result=%u\n", host_output);

  HIP_CHECK(hipFree(output));
  HIP_CHECK(hipFree(scratch));
  HIP_CHECK(hipFree(device_input));
  return 0;
}
