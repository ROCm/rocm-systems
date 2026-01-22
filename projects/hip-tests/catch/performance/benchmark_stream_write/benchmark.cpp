#include <cassert>
#include <chrono>
#include <cstring> // memset
#include <hip/hip_runtime.h>
#include <hip/hip_runtime_api.h>
#include <iostream>
#include <rccl/rccl.h>
#include <rocblas/rocblas.h>
#include <unistd.h> // for sleep
#include <vector>

constexpr bool skip_iteration_logs = true;

#define CHECK_HIP(cmd)                                                         \
  do {                                                                         \
    hipError_t e = cmd;                                                        \
    if (e != hipSuccess) {                                                     \
      std::cerr << "HIP error: " << hipGetErrorString(e) << __LINE__ << "\n";  \
      exit(EXIT_FAILURE);                                                      \
    }                                                                          \
  } while (0)

// simple kernel that updates pinned/shared memory with system-scope atomic
__global__ void sync_kernel(uint32_t *test_ptrs, int i) {
  atomicAdd_system(&test_ptrs[i], 1);
}

struct Timings {
  double chrono_total = 0;
  double streamOp_write_exec_total = 0;
  double streamOp_wait_exec_total = 0;
  double streamOp_exec_total = 0;
  double streamOp_exec_max_total = 0;
  double streamOp_exec_min_total = 0;
};

Timings baseline_timings{};
Timings optimized_user_kernel_timings{};
Timings optimized_new_flags_timings{};

constexpr int num_devices = 1;
uint32_t value = 0;

std::vector<int> devices(num_devices);

std::vector<hipStream_t> streams(num_devices);

uint32_t *test_ptrs;
std::vector<hipEvent_t> eventStartWrite(num_devices);
std::vector<hipEvent_t> eventStopWrite(num_devices);
std::vector<hipEvent_t> eventStartWait(num_devices);
std::vector<hipEvent_t> eventStopWait(num_devices);

std::vector<std::vector<hipStreamBatchMemOpParams>> paramArray;

int events;
int synchronize;
int finer;
int scale;
int warmup_iters;
int measured_iters;

static inline void setup_baseline() {
  for (int i = 0; i < num_devices; ++i)
    devices[i] = i;

  CHECK_HIP(hipHostMalloc((void **)&test_ptrs,
                          num_devices * sizeof(uint32_t) * scale,
                          hipMallocSignalMemory));
  for (int i = 0; i < num_devices; i++) {
    for (int j = 0; j < scale; j++) {
      int signal_id =
          (i * scale) + j; // if scale=1 - no scaling. then this is just i.
      __atomic_store(&test_ptrs[signal_id], &value, __ATOMIC_RELEASE);
    }
  }

  paramArray = std::vector<std::vector<hipStreamBatchMemOpParams>>(
      num_devices,
      std::vector<hipStreamBatchMemOpParams>((num_devices - 1) * scale));
  for (int i = 0; i < num_devices; ++i) {
    CHECK_HIP(hipSetDevice(i));
    CHECK_HIP(hipStreamCreate(&streams[i]));
    CHECK_HIP(hipEventCreate(&eventStartWrite[i]));
    CHECK_HIP(hipEventCreate(&eventStopWrite[i]));
    CHECK_HIP(hipEventCreate(&eventStartWait[i]));
    CHECK_HIP(hipEventCreate(&eventStopWait[i]));

    int param_index = 0;

    for (int j = 0; j < num_devices; j++) {
      if (i == j)
        continue;
      for (int k = 0; k < scale; k++) {
        int signal_id = (j * scale) + k;
        memset(&paramArray[i][param_index], 0,
               sizeof(hipStreamBatchMemOpParams));
        paramArray[i][param_index].operation = hipStreamMemOpWaitValue32;

        paramArray[i][param_index].waitValue.address = &test_ptrs[signal_id];
        paramArray[i][param_index].waitValue.value = 1;
        paramArray[i][param_index].waitValue.flags = hipStreamWaitValueEq;
        param_index++;
      }
    }
  }
}

static inline void setup_optimized() {
  for (int i = 0; i < num_devices; ++i)
    devices[i] = i;

  CHECK_HIP(hipHostMalloc((void **)&test_ptrs, sizeof(uint32_t),
                          hipMallocSignalMemory));

  for (int i = 0; i < num_devices; ++i) {
    CHECK_HIP(hipSetDevice(i));
    CHECK_HIP(hipStreamCreate(&streams[i]));
    CHECK_HIP(hipEventCreate(&eventStartWrite[i]));
    CHECK_HIP(hipEventCreate(&eventStopWrite[i]));
    CHECK_HIP(hipEventCreate(&eventStartWait[i]));
    CHECK_HIP(hipEventCreate(&eventStopWait[i]));
  }
}

static inline void store_initial_value_baseline() {
  for (int i = 0; i < num_devices; i++) {
    for (int j = 0; j < scale; j++) {
      int signal_id = (i * scale) + j;
      __atomic_store(&test_ptrs[signal_id], &value, __ATOMIC_RELEASE);
    }
  }
}

static inline void store_initial_value_optimized() {
  __atomic_store(&test_ptrs[0], &value, __ATOMIC_RELEASE);
}

static inline void execute_baseline() {
  // unique signals written by each of the streams.
  // t3-t2 will give us API launch time -we don't konow when will they execute
  // events will give us exec time
  for (int i = 0; i < num_devices; i++) {
    CHECK_HIP(hipSetDevice(i));
    if (events) {
      CHECK_HIP(hipEventRecord(eventStartWrite[i], streams[i]));
    }
    for (int j = 0; j < scale; j++) {
      int signal_id = (i * scale) + j;
      CHECK_HIP(hipStreamWriteValue32(streams[i], &test_ptrs[signal_id], 1, 0));
    }
    if (events && finer) {
      CHECK_HIP(hipEventRecord(eventStopWrite[i], streams[i]));
      CHECK_HIP(hipEventRecord(eventStartWait[i], streams[i]));
    }

    CHECK_HIP(hipStreamBatchMemOp(streams[i], (num_devices - 1) * scale,
                                  paramArray[i].data(), 0));

    if (events) {
      CHECK_HIP(hipEventRecord(eventStopWait[i], streams[i]));
    }
  }
  if (synchronize) {
    for (int i = 0; i < num_devices; i++) {
      CHECK_HIP(hipSetDevice(i));
      CHECK_HIP(hipEventSynchronize(eventStopWait[i]));
    }
  }
};

static inline void execute_optimized_user_kernel() {
  // wait for all gemms here - unique signals written by each of the streams.
  // t3-t2 will give us API launch time -we don't konow when will they execute
  // events will give us exec time
  for (int i = 0; i < num_devices; i++) {
    CHECK_HIP(hipSetDevice(i));
    if (events) {
      CHECK_HIP(hipEventRecord(eventStartWrite[i], streams[i]));
    }
    for (int j = 0; j < scale; j++) {
      CHECK_HIP(hipStreamWriteValue32(streams[i], &test_ptrs[0], 1, hipExtStreamWriteValueIncrement));
    }
    if (events && finer) {
      CHECK_HIP(hipEventRecord(eventStopWrite[i], streams[i]));
      CHECK_HIP(hipEventRecord(eventStartWait[i], streams[i]));
    }

    CHECK_HIP(hipStreamWaitValue32(streams[i], &test_ptrs[0], num_devices * scale,
                                   hipStreamWaitValueEq,
                                   0xFFFFFFFF));  /// all kernels wait for value to be 8=num_devices

    if (events) {
      CHECK_HIP(hipEventRecord(eventStopWait[i], streams[i]));
    }
  }
  if (synchronize) {
    for (int i = 0; i < num_devices; i++) {
      CHECK_HIP(hipSetDevice(i));
      CHECK_HIP(hipEventSynchronize(eventStopWait[i]));
    }
  }
};

static inline void log_current_run(int it, double chrono_time,
                                   Timings &timings) {

  float elapsed_write_ms_all_gpus = 0.0f;
  float elapsed_wait_ms_all_gpus = 0.0f;
  float elapsed_ms_all_gpus = 0.0f;
  float elapsed_ms_max_gpu = 0.0f;
  float elapsed_ms_min_gpu = 0.0f;
  if (events) {
    for (int i = 0; i < num_devices; i++) {
      CHECK_HIP(hipSetDevice(i));
      float elapsed_write_ms = 0.0f;
      float elapsed_wait_ms = 0.0f;
      float elapsed_ms = 0.0f;
      CHECK_HIP(hipStreamSynchronize(streams[i]));
      if (finer) {
        CHECK_HIP(hipEventElapsedTime(&elapsed_write_ms, eventStartWrite[i],
                                      eventStopWrite[i]));
        CHECK_HIP(hipEventElapsedTime(&elapsed_wait_ms, eventStartWait[i],
                                      eventStopWait[i]));
        elapsed_write_ms_all_gpus += elapsed_write_ms;
        timings.streamOp_write_exec_total += elapsed_write_ms;
        elapsed_wait_ms_all_gpus += elapsed_wait_ms;
        timings.streamOp_wait_exec_total += elapsed_wait_ms;
      } else {
        CHECK_HIP(hipEventElapsedTime(&elapsed_ms, eventStartWrite[i],
                                      eventStopWait[i]));
        elapsed_ms_all_gpus += elapsed_ms;
        timings.streamOp_exec_total += elapsed_ms;
        if (elapsed_ms > elapsed_ms_max_gpu) {
          elapsed_ms_max_gpu = elapsed_ms;
        }
        if (i == 0) {
          elapsed_ms_min_gpu = elapsed_ms;
        } else if (elapsed_ms < elapsed_ms_min_gpu) {
          elapsed_ms_min_gpu = elapsed_ms;
        }
      }
    }
    if (!finer) {
      timings.streamOp_exec_max_total += elapsed_ms_max_gpu;
      timings.streamOp_exec_min_total += elapsed_ms_min_gpu;
    }
  }

  timings.chrono_total += chrono_time;

  if (skip_iteration_logs) {
    return;
  }

  std::cout << "Iteration " << (it - warmup_iters + 1);
  if (synchronize)
    std::cout << ": total=" << chrono_time * 1000 << " us";
  else
    std::cout << ", StreamOpAPI=" << chrono_time * 1000 << " us";
  if (events) {
    if (finer) {
      std::cout << ", StreamOpWriteExec="
                << (elapsed_write_ms_all_gpus / num_devices) * 1000 << " us";
      std::cout << ", StreamOpWaitExec="
                << (elapsed_wait_ms_all_gpus / num_devices) * 1000 << " us";
    } else {
      std::cout << ", StreamOpExec="
                << (elapsed_ms_all_gpus / num_devices) * 1000 << " ms";
      std::cout << ", StreamOpExecMax =" << elapsed_ms_max_gpu * 1000 << " us";
      std::cout << ", StreamOpExecMin =" << elapsed_ms_min_gpu * 1000 << " us";
    }
  }
  std::cout << std::endl;
};

static inline void summarize(const Timings &timings) {
  std::cout << "\n=== Average Timings over " << measured_iters
            << " iterations ===\n";
  if (synchronize)
    std::cout << "Total time : "
              << (timings.chrono_total / measured_iters) * 1000 << " us\n";
  else
    std::cout << "Stream Op API time : "
              << (timings.chrono_total / measured_iters) * 1000 << " us\n";
  if (events) {
    if (finer) {
      std::cout << "Stream Op Write Exec time: "
                << (timings.streamOp_write_exec_total) /
                       (num_devices * measured_iters) * 1000
                << " us\n";
      std::cout << "Stream Op Wait Exec time: "
                << (timings.streamOp_wait_exec_total) /
                       (num_devices * measured_iters) * 1000
                << " us\n";
    } else {
      std::cout << "Stream Op Exec time: "
                << (timings.streamOp_exec_total) /
                       (num_devices * measured_iters) * 1000
                << " us\n";
      std::cout << "Stream Op Exec Max time: "
                << (timings.streamOp_exec_max_total) / (measured_iters) * 1000
                << " us\n";
      std::cout << "Stream Op Exec Min time: "
                << (timings.streamOp_exec_min_total) / (measured_iters) * 1000
                << " us\n";
    }
  }
};

static inline void validate() {
  for (int i = 0; i < num_devices; ++i) {
    if (test_ptrs[i] == 0) {
      std::cerr << "Signal " << i << " from device " << i
                << ": was not written to and is " << test_ptrs[i] << std::endl;
      exit(1);
    }
  }
  std::cout << "Validation PASSED.\n";
};

static inline void validate_optimized_user_kernel() {
  if (synchronize) {
    if (test_ptrs[0] != num_devices * scale) {
      std::cerr << "Signal " << 0 << " was not incremented properly "
                << test_ptrs[0] << std::endl;
      exit(1);
    } else {
      std::cout << "Validation PASSED.\n";
    }
  } else {
    std::cerr << "Signal " << 0 << " was at " << test_ptrs[0] << std::endl;
  }
};

static inline void synchronize_stream() {
  for (int i = 0; i < num_devices; i++) {
    CHECK_HIP(hipSetDevice(i));
    CHECK_HIP(hipStreamSynchronize(streams[i]));
  }
};

static inline void cleanup() {
  for (int i = 0; i < num_devices; ++i) {
    CHECK_HIP(hipSetDevice(i));
    CHECK_HIP(hipStreamDestroy(streams[i]));

    CHECK_HIP(hipEventDestroy(eventStartWrite[i]));
    CHECK_HIP(hipEventDestroy(eventStopWrite[i]));
    CHECK_HIP(hipEventDestroy(eventStartWait[i]));
    CHECK_HIP(hipEventDestroy(eventStopWait[i]));
  }

  CHECK_HIP(hipFree(test_ptrs));

  devices.clear();
  streams.clear();
  eventStartWrite.clear();
  eventStopWrite.clear();
  eventStartWait.clear();
  eventStopWait.clear();
  paramArray.clear();
};

static inline void run_baseline() {
  setup_baseline();

  for (int it = 0; it < warmup_iters + measured_iters; ++it) {
    store_initial_value_baseline();

    /*************** HIP RUNTIME TESTS ****************/
    auto t2 = std::chrono::high_resolution_clock::now();
    execute_baseline();
    auto t3 = std::chrono::high_resolution_clock::now();
    /*************** HIP RUNTIME TESTS ****************/
    synchronize_stream();

    if (it >= warmup_iters) {
      log_current_run(
          it, std::chrono::duration<double, std::milli>(t3 - t2).count(),
          baseline_timings);
    }
  }

  summarize(baseline_timings);
  synchronize_stream();
  validate();
  cleanup();
}

static inline void run_optimized_user_kernel() {
  setup_optimized();

  for (int it = 0; it < warmup_iters + measured_iters; ++it) {
    store_initial_value_optimized();

    /*************** HIP RUNTIME TESTS ****************/
    auto t2 = std::chrono::high_resolution_clock::now();
    execute_optimized_user_kernel();
    auto t3 = std::chrono::high_resolution_clock::now();
    /*************** HIP RUNTIME TESTS ****************/
    synchronize_stream();

    if (it >= warmup_iters) {
      log_current_run(
          it, std::chrono::duration<double, std::milli>(t3 - t2).count(),
          optimized_user_kernel_timings);
    }
  }

  summarize(optimized_user_kernel_timings);
  synchronize_stream();
  validate_optimized_user_kernel();
  cleanup();
}

static inline void run_optimized_new_flags() {
  setup_optimized();

  for (int it = 0; it < warmup_iters + measured_iters; ++it) {
    store_initial_value_optimized();

    /*************** HIP RUNTIME TESTS ****************/
    auto t2 = std::chrono::high_resolution_clock::now();
    execute_optimized_user_kernel();
    auto t3 = std::chrono::high_resolution_clock::now();
    /*************** HIP RUNTIME TESTS ****************/
    synchronize_stream();

    if (it >= warmup_iters) {
      log_current_run(
          it, std::chrono::duration<double, std::milli>(t3 - t2).count(),
          optimized_new_flags_timings);
    }
  }

  summarize(optimized_new_flags_timings);
  synchronize_stream();
  validate_optimized_user_kernel();
  cleanup();
}

static inline void print_speedup() {
  std::cout << "\n=== Speedup User Kernel Timings ===\n" << std::endl;
  std::cout << "Average : "
            << baseline_timings.streamOp_exec_total /
                   optimized_user_kernel_timings.streamOp_exec_total
            << "x\n";
  if (finer) {
    std::cout << "Write : "
              << baseline_timings.streamOp_write_exec_total /
                     optimized_user_kernel_timings.streamOp_write_exec_total
              << "x\n";
    std::cout << "Wait : "
              << baseline_timings.streamOp_wait_exec_total /
                     optimized_user_kernel_timings.streamOp_wait_exec_total
              << "x\n";
  }

  std::cout << "\n=== Speedup New Flags Timings ===\n" << std::endl;
  std::cout << "Average : "
            << baseline_timings.streamOp_exec_total /
                   optimized_new_flags_timings.streamOp_exec_total
            << "x\n";
  if (finer) {
    std::cout << "Write : "
              << baseline_timings.streamOp_write_exec_total /
                     optimized_new_flags_timings.streamOp_write_exec_total
              << "x\n";
    std::cout << "Wait : "
              << baseline_timings.streamOp_wait_exec_total /
                     optimized_new_flags_timings.streamOp_wait_exec_total
              << "x\n";
  }
}

int main(int argc, char *argv[]) {
  events = atoi(argv[1]);
  synchronize = atoi(argv[2]);
  finer = atoi(argv[3]);
  scale = atoi(argv[4]);
  warmup_iters = (argc > 5) ? atoi(argv[5]) : 5;
  measured_iters = (argc > 6) ? atoi(argv[6]) : 10;

//   std::cout << "Running baseline test" << std::endl;
//   run_baseline();

  std::cout << "\nRunning optimized test with user kernel" << std::endl;
  run_optimized_user_kernel();

  std::cout << "\nRunning optimized test with new flag" << std::endl;
  run_optimized_new_flags();

  print_speedup();

  return 0;
}
