#include "../common/common.h"

int main(int argc, char** argv) {
  int* gpuMem;
  prepare();
  // Initialize the tools
  CHECK_ROCPROFILER(rocprofiler_initialize());

  // Creating the session with given replay mode
  rocprofiler_session_id_t session_id;
  CHECK_ROCPROFILER(rocprofiler_create_session(ROCPROFILER_NONE_REPLAY_MODE, &session_id));

  // Creating Output Buffer for the data
  rocprofiler_buffer_id_t buffer_id;
  CHECK_ROCPROFILER(rocprofiler_create_buffer(
      session_id,
      [](const rocprofiler_record_header_t* record, const rocprofiler_record_header_t* end_record,
         rocprofiler_session_id_t session_id, rocprofiler_buffer_id_t buffer_id) {
        WriteBufferRecords(record, end_record, session_id, buffer_id);
      },
      0x9999, &buffer_id));

  // Counter Collection Filter
  std::vector<const char*> counters;
  counters.emplace_back("GRBM_COUNT");
  rocprofiler_filter_id_t filter_id;
  [[maybe_unused]] rocprofiler_filter_property_t property = {};
  CHECK_ROCPROFILER(
      rocprofiler_create_filter(session_id, ROCPROFILER_COUNTERS_COLLECTION,
                                rocprofiler_filter_data_t{.counters_names = &counters[0]},
                                counters.size(), &filter_id, property));
  CHECK_ROCPROFILER(rocprofiler_set_filter_buffer(session_id, filter_id, buffer_id));

  // Normal HIP Calls
  hipDeviceProp_t devProp;
  HIP_CALL(hipGetDeviceProperties(&devProp, 0));
  HIP_CALL(hipMalloc((void**)&gpuMem, 1 * sizeof(int)));

  // KernelA and KernelB won't be profiled
  kernelCalls('A');
  kernelCalls('B');

  // Activating Profiling Session to profile whatever kernel launches occurs up
  // till the next terminate session
  CHECK_ROCPROFILER(rocprofiler_start_session(session_id));

  // KernelC, KernelD, KernelE and KernelF to be profiled as part of the session
  kernelCalls('C');
  kernelCalls('D');
  kernelCalls('E');
  kernelCalls('F');
  // Normal HIP Calls
  HIP_CALL(hipFree(gpuMem));

  // Deactivating session
  CHECK_ROCPROFILER(rocprofiler_terminate_session(session_id));

  // Manual Flush user buffer request
  CHECK_ROCPROFILER(rocprofiler_flush_data(session_id, buffer_id));

  // Destroy sessions
  CHECK_ROCPROFILER(rocprofiler_destroy_session(session_id));

  // Destroy all profiling related objects(User buffer, sessions, filters,
  // etc..)
  CHECK_ROCPROFILER(rocprofiler_finalize());

  return 0;
}