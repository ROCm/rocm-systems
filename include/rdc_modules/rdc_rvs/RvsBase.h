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

#include <cstddef>
#include <cstdio>
#include <map>
#include <string>

#include "rdc/rdc.h"
#include "rvs/rvs.h"

static constexpr size_t MAX_CONFIG_LENGTH = 1024;
// NOTE: There MUST be a space after :
static const std::map<rdc_diag_test_cases_t, std::string> test_to_conf = {
    // derived from conf/gst_single.conf
    {RDC_DIAG_RVS_GST_TEST,
     "{actions: [{name: gpustress-9000-sgemm-false, device: all, "
     "device_index: '0', module: gst, parallel: false, count: 1, duration: "
     "10000, copy_matrix: false, target_stress: 9000, matrix_size_a: 8640, "
     "matrix_size_b: 8640, matrix_size_c: 8640, ops_type: sgemm, lda: 8640, "
     "ldb: 8640, ldc: 8640}]}"},
    // derived from conf/MI300X/babel.conf
    {RDC_DIAG_RVS_MEMBW_TEST,
     "{actions: [{name: babel-float-256MiB,"
     "device: all, module: babel, "
     "parallel: false, count: 1, num_iter: 5000, array_size: 268435456, "
     "test_type: 1, mibibytes: true, o/p_csv: false, subtest: 5}]}"},
    // derived from conf/MI300X/pebb_single.conf
    {RDC_DIAG_RVS_H2DD2H_TEST,
     "{actions: [{name: h2d-d2h-sequential-64MB,"
     "device: all, module: pebb, duration: 120000, device_to_host: true, "
     "host_to_device: true, parallel: false, block_size: 67108864, "
     "link_type: 2, warm_calls: 10, hot_calls: 100, b2b: true}]}"},
    // derived from conf/MI300X/iet_single.conf
    {RDC_DIAG_RVS_IET_TEST,
     "{actions: [{name: iet-400W-1K-rand-dgemm,"
     "device: all, module: iet, parallel: true, duration: 60000, "
     "sample_interval: 3000, target_power: 400, matrix_size: 1024, "
     "matrix_init: rand, ops_type: dgemm}]}"},
};
static const std::map<rdc_diag_test_cases_t, std::string> test_to_name = {
    {RDC_DIAG_RVS_GST_TEST, "RVS_GST_TEST"},
    {RDC_DIAG_RVS_MEMBW_TEST, "RVS_MEMBW_TEST"},
    {RDC_DIAG_RVS_H2DD2H_TEST, "RVS_H2DD2H_TEST"},
    {RDC_DIAG_RVS_IET_TEST, "RVS_IET_TEST"},
};

namespace amd {
namespace rdc {

class RdcRVSBase {
 public:
  RdcRVSBase() { s_instance = this; };
  ~RdcRVSBase() {
    if (s_instance == this) {
      s_instance = nullptr;
    }
  };

  // only one instance allowed
  RdcRVSBase(const RdcRVSBase&) = delete;
  RdcRVSBase& operator=(const RdcRVSBase&) = delete;

  // no moving allowed
  RdcRVSBase(RdcRVSBase&&) = delete;
  RdcRVSBase& operator=(RdcRVSBase&&) = delete;

  rvs_status_t run_rvs_app(const char* config, size_t config_size, rdc_diag_callback_t* callback);

 private:
  static RdcRVSBase* s_instance;
  volatile rvs_session_state_t _state = RVS_SESSION_STATE_IDLE;
  rdc_diag_callback_t* _callback = nullptr;
  rvs_session_callback _rvs_callback = nullptr;

  // Static callback function that the C API will call
  static void static_callback(rvs_session_id_t session_id, const rvs_results_t* results) {
    // Forward the call to the current instance if it exists
    if (s_instance) {
      s_instance->session_callback(session_id, results);
    }
  }
  void session_callback(rvs_session_id_t session_id, const rvs_results_t* results) {
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
