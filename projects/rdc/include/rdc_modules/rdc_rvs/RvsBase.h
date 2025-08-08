/*
Copyright (c) 2023 - present Advanced Micro Devices, Inc. All rights reserved.

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
*/
#ifndef RDC_MODULES_RDC_RVS_RVSBASE_H_
#define RDC_MODULES_RDC_RVS_RVSBASE_H_

#include <amd_smi/amdsmi.h>

#include <cstddef>
#include <cstdio>
#include <map>
#include <string>
#include <vector>

#include "rdc/rdc.h"
#include "rdc_lib/RdcLogger.h"
#include "rvs/rvs.h"

static constexpr size_t MAX_CONFIG_LENGTH = 1024;
// this map only makes sense in context of test config locations as originally
// designed in RVS
static const std::map<uint64_t, std::string> gfx_to_rvs_conf = {
    {0x90a, "MI210"},   // ?
    {0x940, "MI300A"},  // ?
    {0x941, "MI300A"},  // ?
    {0x942, "MI300X"},  // ?
    {0x94a, "MI308X"},  // ?
    {0x1030, "nv21"},   //
    {0x1031, "nv21"},   // ?
    {0x1032, "nv21"},   // ?
    {0x1033, "nv21"},   // ?
    {0x1034, "nv21"},   // ?
    {0x1035, "nv21"},   // ?
    {0x1100, "nv31"},   // ?
    {0x1101, "nv31"},   // ?
    {0x1102, "nv31"},   // ?
    {0x1103, "nv31"},   // ?
};
static const std::map<rdc_diag_test_cases_t, std::string> test_to_name = {
    {RDC_DIAG_RVS_GST_TEST, "gst_single.conf"},
    {RDC_DIAG_RVS_MEMBW_TEST, "babel.conf"},
    {RDC_DIAG_RVS_H2DD2H_TEST, "pebb_single.conf"},
    {RDC_DIAG_RVS_IET_TEST, "iet_stress.conf"},
    {RDC_DIAG_RVS_GST_LONG_TEST, "gst_single_long.conf"},
    {RDC_DIAG_RVS_MEMBW_LONG_TEST, "babel_long.conf"},
    {RDC_DIAG_RVS_H2DD2H_LONG_TEST, "pebb_single_long.conf"},
    {RDC_DIAG_RVS_IET_LONG_TEST, "iet_stress_long.conf"},
    {RDC_DIAG_RVS_CUSTOM, "CUSTOM_CONFIG"},
};

namespace amd {
namespace rdc {

inline amdsmi_status_t get_processor_handle_from_id(uint32_t gpu_id,
                                                    amdsmi_processor_handle* processor_handle) {
  uint32_t socket_count = 0;
  uint32_t processor_count = 0;
  auto ret = amdsmi_get_socket_handles(&socket_count, nullptr);
  if (ret != AMDSMI_STATUS_SUCCESS) {
    return ret;
  }
  std::vector<amdsmi_socket_handle> sockets(socket_count);
  std::vector<amdsmi_processor_handle> all_processors{};
  ret = amdsmi_get_socket_handles(&socket_count, sockets.data());
  for (auto& socket : sockets) {
    ret = amdsmi_get_processor_handles(socket, &processor_count, nullptr);
    if (ret != AMDSMI_STATUS_SUCCESS) {
      return ret;
    }
    std::vector<amdsmi_processor_handle> processors(processor_count);
    ret = amdsmi_get_processor_handles(socket, &processor_count, processors.data());
    if (ret != AMDSMI_STATUS_SUCCESS) {
      return ret;
    }

    for (auto& processor : processors) {
      processor_type_t processor_type = {};
      ret = amdsmi_get_processor_type(processor, &processor_type);
      if (processor_type != AMDSMI_PROCESSOR_TYPE_AMD_GPU) {
        RDC_LOG(RDC_ERROR, "Expect AMD_GPU device type!");
        return AMDSMI_STATUS_NOT_SUPPORTED;
      }
      all_processors.push_back(processor);
    }
  }

  if (gpu_id >= all_processors.size()) {
    return AMDSMI_STATUS_INPUT_OUT_OF_BOUNDS;
  }

  // Get processor handle from GPU id
  *processor_handle = all_processors[gpu_id];

  return AMDSMI_STATUS_SUCCESS;
}

class RdcRVSBase {
 public:
  RdcRVSBase();

  ~RdcRVSBase();

  // only one instance allowed
  RdcRVSBase(const RdcRVSBase&) = delete;
  RdcRVSBase& operator=(const RdcRVSBase&) = delete;

  // no moving allowed
  RdcRVSBase(RdcRVSBase&&) = delete;
  RdcRVSBase& operator=(RdcRVSBase&&) = delete;

  rvs_status_t run_rvs_app(const char* config, size_t config_size, rdc_diag_callback_t* callback);
  std::vector<std::string> get_rvs_configs();
  std::map<rdc_diag_test_cases_t, std::string> get_test_to_conf();

 private:
  static RdcRVSBase* s_instance;
  volatile rvs_session_state_t _state = RVS_SESSION_STATE_IDLE;
  rdc_diag_callback_t* _callback = nullptr;
  rvs_session_callback _rvs_callback = nullptr;
  std::vector<std::string> _rvs_config_list = {};
  std::map<rdc_diag_test_cases_t, std::string> _test_to_conf = {};

  // Static callback function that the C API will call
  static void static_callback(rvs_session_id_t session_id, const rvs_results_t* results) {
    // Forward the call to the current instance if it exists
    if (s_instance) {
      s_instance->session_callback(session_id, results);
    }
  }
  void session_callback(rvs_session_id_t /*session_id*/, const rvs_results_t* results) {
    _state = results->state;
    // std::string output = "\n";
    // output += "session id -> " + std::to_string(session_id) + "\n";
    // output += "  state  -> " + std::to_string(results->state) + "\n";
    // output += "  status -> " + std::to_string(results->status) + "\n";
    // output += "  output -> " + std::string(results->output_log);
    std::string output = std::string(results->output_log);
    if (_callback != nullptr && _callback->callback != nullptr && _callback->cookie != nullptr) {
      _callback->callback(_callback->cookie, output.data());
    }
  }
};
}  // namespace rdc
}  // namespace amd

#endif  // RDC_MODULES_RDC_RVS_RVSBASE_H_
