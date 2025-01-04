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
#include "rdc_modules/rdc_rvs/RvsBase.h"

#include <string.h>

#include "rdc/rdc.h"
#include "rdc_lib/RdcLogger.h"
#include "rdc_lib/rdc_common.h"
#include "rdc_modules/rdc_rvs/RvsBase.h"
#include "rvs/rvs.h"

// TODO: Make generic test
// TODO: Allow for user to override defaults with a custom string

amd::rdc::RdcRVSBase* amd::rdc::RdcRVSBase::s_instance = nullptr;

rvs_status_t amd::rdc::RdcRVSBase::run_rvs_app(const char* config, const size_t config_size,
                                               rdc_diag_callback_t* callback) {
  char active_config[MAX_CONFIG_LENGTH];
  rvs_session_property_t session_property = {RVS_SESSION_TYPE_DEFAULT_CONF, {{RVS_MODULE_GST}}};
  rvs_session_id_t session_id;
  rvs_status_t status;
  // NOTE: device_index is NOT set by RDC unless a custom config is provided.
  // Meaning RDC index has no impact on RVS index.
  const char mem_config[MAX_CONFIG_LENGTH] =
      "{actions: [{name: action_1, device: all, module: mem, parallel: true, "
      "count: 1, wait: 100, mapped_memory: false, mem_blocks: 128, "
      "num_passes: 500, thrds_per_blk: 64, stress: true, num_iter: 50000, "
      "exclude: '5 6 7 8 9 10 11'}]}";
  const char gst_config[MAX_CONFIG_LENGTH] =
      "{actions: [{name: gpustress-9000-sgemm-false, device: all, "
      "device_index: '0', module: gst, parallel: false, count: 1, duration: "
      "10000, copy_matrix: false, target_stress: 9000, matrix_size_a: 8640, "
      "matrix_size_b: 8640, matrix_size_c: 8640, ops_type: sgemm, lda: 8640, "
      "ldb: 8640, ldc: 8640}]}";

  if ((config == nullptr) || (config_size == 0)) {
    RDC_LOG(RDC_INFO, "given config is NULL! Using predefined gst_config");
    strncpy_with_null(active_config, gst_config, MAX_CONFIG_LENGTH);
  } else if (config_size > MAX_CONFIG_LENGTH) {
    RDC_LOG(RDC_ERROR, "given config size is too large! Expected at most "
                           << MAX_CONFIG_LENGTH << ", got " << config_size << " instead.");
    return RVS_STATUS_INVALID_ARGUMENT;
  } else {
    RDC_LOG(RDC_DEBUG, "given config is correct");
    strncpy_with_null(active_config, config, config_size);
  }

  status = rvs_initialize();
  if (status == RVS_STATUS_FAILED) {
    RDC_LOG(RDC_ERROR, "rvs initialization failed");
    return status;
  }

  /*******************************/

  _state = RVS_SESSION_STATE_IDLE;

  /* Using custom gst configuration in string format */

  _callback = callback;
  status = rvs_session_create(&session_id, &RdcRVSBase::static_callback);

  session_property.type = RVS_SESSION_TYPE_CUSTOM_ACTION;
  session_property.custom_action.config = active_config;

  status = rvs_session_set_property(session_id, &session_property);

  status = rvs_session_execute(session_id);

  if (status != RVS_STATUS_SUCCESS) {
    RDC_LOG(RDC_ERROR, "RVS session execute failed with status: " << status);
    rvs_session_destroy(session_id);
    return status;
  }

  while (_state != RVS_SESSION_STATE_COMPLETED) {
  };

  _callback = nullptr;

  status = rvs_session_destroy(session_id);
  if (status != RVS_STATUS_SUCCESS) {
    RDC_LOG(RDC_ERROR, "RVS session destroy failed with status: " << status);
  }

  return status;
}
