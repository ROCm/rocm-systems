#include "../common/common.h"

int main(int argc, char** argv) {
  int* gpuMem;
  int counter_option = 0;

  std::vector<std::string> pcie_counters = {
      "CI_PERF_slv_MemRd_Bandwidth0", "CI_PERF_slv_MemWr_Bandwidth0", "CI_PERF_slv_totalMemRdTx",
      "CI_PERF_slv_totalMemWrTx", "CI_PERF_slv_totalTx"};

  if (argc > 1) {
    counter_option = atoi(argv[1]);
  } else {
    std::cout << "Please provide one of the counter index options as argument:\n";
    for (int i = 0; i < pcie_counters.size(); i++) {
      std::cout << "[" << i << "]: " << pcie_counters[i] << std::endl;
    }
    std::cout << "Example:\n ./pcie_counters_sampler 1\n";
    exit(0);
  }

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
      0x999999, &buffer_id));

  // Counters Sampler Filter
  rocprofiler_filter_id_t filter_id;
  [[maybe_unused]] rocprofiler_filter_property_t property = {};


  rocprofiler_counters_sampler_counter_input_t counters_input[2] = {
      {.name = const_cast<char*>(pcie_counters[counter_option].c_str()),
       .type = ROCPROFILER_COUNTERS_SAMPLER_PCIE_COUNTERS}};

  uint32_t rate = 1000;
  uint32_t duration = 5000;

  rocprofiler_counters_sampler_parameters_t cs_parameters = {.counters = counters_input,
                                                             .counters_num = 1,
                                                             .sampling_rate = rate,
                                                             .sampling_duration = duration,
                                                             .gpu_agent_index = 0};
  CHECK_ROCPROFILER(rocprofiler_create_filter(
      session_id, ROCPROFILER_COUNTERS_SAMPLER,
      rocprofiler_filter_data_t{.counters_sampler_parameters = cs_parameters}, 0, &filter_id,
      property));
  CHECK_ROCPROFILER(rocprofiler_set_filter_buffer(session_id, filter_id, buffer_id));

  // Normal HIP Calls
  hipDeviceProp_t devProp;
  HIP_CALL(hipGetDeviceProperties(&devProp, 0));
  HIP_CALL(hipMalloc((void**)&gpuMem, 1 * sizeof(int)));

  // KernelA and KernelB won't be profiled
  kernelCalls('A');
  kernelCalls('B');

  std::cout << "Collecting samples for: " << pcie_counters[counter_option]
            << " ; sampling rate: " << rate << " ms; duration: " << duration << " ms" << std::endl;
  // Activating the session
  CHECK_ROCPROFILER(rocprofiler_start_session(session_id));

  // KernelC, KernelD, KernelE and KernelF to be profiled as part of the session
  kernelCalls('C');
  kernelCalls('D');
  kernelCalls('E');
  kernelCalls('F');
  // Normal HIP Calls
  HIP_CALL(hipFree(gpuMem));

  // allow sampler to run for 10 secs
  sleep(6);

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