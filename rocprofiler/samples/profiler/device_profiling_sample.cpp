#include "../common/common.h"
#include <chrono>
#include <thread>
#include <iostream>

#include <rocprofiler/v2/rocprofiler.h>

int main(int argc, char** argv) {
  int poll_duration = 5;
  if (argc > 1) poll_duration = atoi(argv[1]);

  hipDeviceProp_t devProp;
  HIP_CALL(hipGetDeviceProperties(&devProp, 0));

  CHECK_ROCPROFILER(rocprofiler_initialize());
  printf("initialize\n");

  rocprofiler_session_id_t dp_session_id;
  std::vector<const char*> counters;
  counters.emplace_back("GRBM_COUNT");

  printf("session create\n");

  int gpu_agent = 0;
  int cpu_agent = 0;
  CHECK_ROCPROFILER(rocprofiler_device_profiling_session_create(
      &counters[0], counters.size(), &dp_session_id, cpu_agent, gpu_agent));

  printf("session start \n");
  // start GPU device profiling
  CHECK_ROCPROFILER(rocprofiler_device_profiling_session_start(dp_session_id));

  using std::chrono::duration_cast;
  using std::chrono::high_resolution_clock;
  using std::chrono::milliseconds;

  auto t1 = high_resolution_clock::now();

  do {
    printf("polling\n");
    std::vector<rocprofiler_device_profile_metric_t> data(counters.size());
    // Poll metrics
    CHECK_ROCPROFILER(rocprofiler_device_profiling_session_poll(dp_session_id, &data[0]));

    for (size_t i = 0; i < data.size(); i++)
      std::cout << data[i].metric_name << ": " << data[i].value.value << std::endl;
    std::this_thread::sleep_for(std::chrono::milliseconds(1));
    // break;
  } while (--poll_duration > 0);

  auto t2 = high_resolution_clock::now();
  /* Getting number of milliseconds as an integer. */
  auto ms_int = duration_cast<milliseconds>(t2 - t1);

  std::cout << ms_int.count() << "ms\n";

  // Stop session
  CHECK_ROCPROFILER(rocprofiler_device_profiling_session_stop(dp_session_id));

  // Destroy session
  CHECK_ROCPROFILER(rocprofiler_device_profiling_session_destroy(dp_session_id));

  return 0;
}
