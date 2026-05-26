/*
 * Host driver for the manual HIP device-coverage PoC.
 *
 * This file intentionally contains no HIP kernel syntax, so it can be compiled
 * with AFL's C++ compiler wrapper. The HIP kernel is loaded from a separate
 * code object produced by hipcc.
 */

#include <hip/hip_runtime_api.h>

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/shm.h>

#include <vector>

namespace {

constexpr uint32_t kMapSize = 65536;
constexpr uint32_t kDeviceStart = kMapSize / 2;
constexpr uint32_t kThreadsPerBlock = 256;
constexpr uint32_t kBlocks = 4;
constexpr uint32_t kWorkItems = kThreadsPerBlock * kBlocks;

#define HIP_CHECK(expr)                                                        \
  do {                                                                         \
    hipError_t err__ = (expr);                                                 \
    if (err__ != hipSuccess) {                                                 \
      fprintf(stderr, "HIP error at %s:%d: %s\n", __FILE__, __LINE__,          \
              hipGetErrorString(err__));                                       \
      exit(2);                                                                 \
    }                                                                          \
  } while (0)

uint8_t counter_to_byte(uint32_t counter) {
  if (counter >= 65536) {
    return 7;
  }
  if (counter >= 16384) {
    return 6;
  }
  if (counter >= 4096) {
    return 5;
  }
  if (counter >= 512) {
    return 4;
  }
  if (counter >= 3) {
    return 3;
  }
  if (counter >= 2) {
    return 2;
  }
  if (counter >= 1) {
    return 1;
  }
  return 0;
}

uint8_t merge_counter(uint8_t host_counter, uint8_t device_counter) {
  const uint32_t merged = host_counter + device_counter;
  const uint8_t merged8 = static_cast<uint8_t>(merged & 0xffu);
  if (merged == 0) {
    return 0;
  }
  return merged8 == 0 ? 1 : merged8;
}

uint8_t *map_afl_trace_bits(std::vector<uint8_t> *fallback) {
  const char *shm_id_env = getenv("__AFL_SHM_ID");
  if (shm_id_env == nullptr || shm_id_env[0] == '\0') {
    fallback->assign(kMapSize, 0);
    return fallback->data();
  }

  char *end = nullptr;
  long shm_id = strtol(shm_id_env, &end, 10);
  if (end == shm_id_env || shm_id < 0) {
    fprintf(stderr, "Invalid __AFL_SHM_ID value: %s\n", shm_id_env);
    exit(2);
  }

  void *mapped = shmat(static_cast<int>(shm_id), nullptr, 0);
  if (mapped == reinterpret_cast<void *>(-1)) {
    perror("shmat");
    exit(2);
  }

  return static_cast<uint8_t *>(mapped);
}

int read_select(const char *path) {
  FILE *file = fopen(path, "rb");
  if (file == nullptr) {
    perror(path);
    exit(1);
  }

  char buffer[64] = {};
  const size_t bytes_read = fread(buffer, 1, sizeof(buffer) - 1, file);
  fclose(file);

  if (bytes_read == 0) {
    return 0;
  }

  char *end = nullptr;
  long parsed = strtol(buffer, &end, 10);
  if (end != buffer) {
    return static_cast<int>(parsed);
  }

  return static_cast<unsigned char>(buffer[0]) % 4;
}

void merge_device_coverage(uint8_t *trace_bits,
                           const std::vector<uint32_t> &device_counters) {
  for (uint32_t i = kDeviceStart; i < kMapSize; ++i) {
    const uint8_t device_byte = counter_to_byte(device_counters[i]);
    trace_bits[i] = merge_counter(trace_bits[i], device_byte);
  }
}

void maybe_print_device_bytes(const std::vector<uint32_t> &device_counters) {
  if (getenv("ROCM_POC_VERBOSE") == nullptr) {
    return;
  }

  uint32_t count = 0;
  for (uint32_t i = kDeviceStart; i < kMapSize; ++i) {
    if (device_counters[i] != 0) {
      fprintf(stderr, "device_cov[%u]=%u\n", i, device_counters[i]);
      ++count;
    }
  }
  fprintf(stderr, "device_cov_bytes=%u\n", count);
}

} // namespace

int main(int argc, char **argv) {
  if (argc != 3) {
    fprintf(stderr, "usage: %s <manual_device_coverage.hsaco> <input>\n",
            argv[0]);
    return 1;
  }

  std::vector<uint8_t> fallback_trace_bits;
  uint8_t *trace_bits = map_afl_trace_bits(&fallback_trace_bits);

  const int select = read_select(argv[2]);

  hipModule_t module = nullptr;
  hipFunction_t kernel = nullptr;
  uint32_t *device_result = nullptr;
  uint32_t *device_coverage = nullptr;
  uint32_t *device_previous_bb = nullptr;

  HIP_CHECK(hipModuleLoad(&module, argv[1]));
  HIP_CHECK(hipModuleGetFunction(&kernel, module, "manual_coverage_kernel"));

  HIP_CHECK(hipMalloc(reinterpret_cast<void **>(&device_result),
                      sizeof(uint32_t)));
  HIP_CHECK(
      hipMalloc(reinterpret_cast<void **>(&device_coverage),
                sizeof(uint32_t) * kMapSize));
  HIP_CHECK(
      hipMalloc(reinterpret_cast<void **>(&device_previous_bb),
                sizeof(uint32_t) * kWorkItems));

  HIP_CHECK(hipMemset(device_result, 0, sizeof(uint32_t)));
  HIP_CHECK(hipMemset(device_coverage, 0, sizeof(uint32_t) * kMapSize));
  HIP_CHECK(hipMemset(device_previous_bb, 0, sizeof(uint32_t) * kWorkItems));

  uint32_t previous_entries = kWorkItems;
  void *args[] = {const_cast<int *>(&select), &device_result, &device_coverage,
                  &device_previous_bb, &previous_entries};

  HIP_CHECK(hipModuleLaunchKernel(kernel, kBlocks, 1, 1, kThreadsPerBlock, 1, 1,
                                  0, nullptr, args, nullptr));
  HIP_CHECK(hipDeviceSynchronize());

  std::vector<uint32_t> host_coverage(kMapSize, 0);
  uint32_t host_result = 0;
  HIP_CHECK(hipMemcpy(host_coverage.data(), device_coverage,
                      sizeof(uint32_t) * kMapSize, hipMemcpyDeviceToHost));
  HIP_CHECK(hipMemcpy(&host_result, device_result, sizeof(uint32_t),
                      hipMemcpyDeviceToHost));

  merge_device_coverage(trace_bits, host_coverage);
  maybe_print_device_bytes(host_coverage);

  if (getenv("ROCM_POC_VERBOSE") != nullptr) {
    fprintf(stderr, "select=%d result=%u\n", select, host_result);
  }

  HIP_CHECK(hipFree(device_previous_bb));
  HIP_CHECK(hipFree(device_coverage));
  HIP_CHECK(hipFree(device_result));
  HIP_CHECK(hipModuleUnload(module));

  return 0;
}
