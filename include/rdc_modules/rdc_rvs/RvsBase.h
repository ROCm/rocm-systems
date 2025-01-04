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
#include <string>

#include "rdc/rdc.h"
#include "rvs/rvs.h"

static constexpr size_t MAX_CONFIG_LENGTH = 1024;

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
