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

  // Tracing Filter
  std::vector<rocprofiler_tracer_activity_domain_t> apis_requested;
  apis_requested.emplace_back(ACTIVITY_DOMAIN_HIP_API);
  apis_requested.emplace_back(ACTIVITY_DOMAIN_HIP_OPS);
  apis_requested.emplace_back(ACTIVITY_DOMAIN_HSA_API);
  apis_requested.emplace_back(ACTIVITY_DOMAIN_HSA_OPS);
  apis_requested.emplace_back(ACTIVITY_DOMAIN_ROCTX);
  rocprofiler_filter_id_t api_tracing_filter_id;
  CHECK_ROCPROFILER(rocprofiler_create_filter(
      session_id, ROCPROFILER_API_TRACE, rocprofiler_filter_data_t{&apis_requested[0]},
      apis_requested.size(), &api_tracing_filter_id, rocprofiler_filter_property_t{}));
  CHECK_ROCPROFILER(rocprofiler_set_filter_buffer(session_id, api_tracing_filter_id, buffer_id));
  CHECK_ROCPROFILER(rocprofiler_set_api_trace_sync_callback(
      session_id, api_tracing_filter_id,
      [](rocprofiler_record_tracer_t record, rocprofiler_session_id_t session_id) {
        FlushTracerRecord(record, session_id);
      }));

  // Kernel Tracing
  rocprofiler_filter_id_t kernel_tracing_filter_id;
  CHECK_ROCPROFILER(rocprofiler_create_filter(
      session_id, ROCPROFILER_DISPATCH_TIMESTAMPS_COLLECTION, rocprofiler_filter_data_t{}, 0,
      &kernel_tracing_filter_id, rocprofiler_filter_property_t{}));
  CHECK_ROCPROFILER(rocprofiler_set_filter_buffer(session_id, kernel_tracing_filter_id, buffer_id));

  // Normal HIP Calls won't be traced
  hipDeviceProp_t devProp;
  HIP_CALL(hipGetDeviceProperties(&devProp, 0));
  HIP_CALL(hipMalloc((void**)&gpuMem, 1 * sizeof(int)));
  // KernelA and KernelB won't be traced
  kernelCalls('A');
  kernelCalls('B');

  // Activating Profiling Session to profile whatever kernel launches occurs up
  // till the next terminate session
  CHECK_ROCPROFILER(rocprofiler_start_session(session_id));

  // KernelC, KernelD, KernelE and KernelF to be traced as part of the session
  kernelCalls('C');
  kernelCalls('D');
  kernelCalls('E');
  kernelCalls('F');
  // Normal HIP Calls that will be traced
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