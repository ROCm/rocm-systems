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

#include "rdc_lib/RdcLogger.h"
#include "rdc_lib/rdc_common.h"
#include "rvs/rvs.h"

// TODO: Make generic test
// TODO: Allow for user to override defaults with a custom string
static constexpr size_t MAX_CONFIG_LENGTH = 1024;

volatile rvs_session_state_t state = RVS_SESSION_STATE_IDLE;

rvs_status_t amd::rdc::run_rvs_app(const char* config, const size_t config_size) {
  char active_config[MAX_CONFIG_LENGTH];
  rvs_session_property_t session_property = {RVS_SESSION_TYPE_DEFAULT_CONF, {{RVS_MODULE_GST}}};
  rvs_session_id_t session_id;
  rvs_status_t status;
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

  state = RVS_SESSION_STATE_IDLE;

  /* Using custom gst configuration in string format */

  status = rvs_session_create(&session_id, amd::rdc::session_callback);

  session_property.type = RVS_SESSION_TYPE_CUSTOM_ACTION;
  session_property.custom_action.config = active_config;

  status = rvs_session_set_property(session_id, &session_property);

  status = rvs_session_execute(session_id);

  if (status != RVS_STATUS_SUCCESS) {
    RDC_LOG(RDC_ERROR, "RVS session execute failed with status: " << status);
    rvs_session_destroy(session_id);
    return status;
  }

  while (state != RVS_SESSION_STATE_COMPLETED) {
  };

  status = rvs_session_destroy(session_id);
  if (status != RVS_STATUS_SUCCESS) {
    RDC_LOG(RDC_ERROR, "RVS session destroy failed with status: " << status);
  }

  return status;
}

void amd::rdc::session_callback(rvs_session_id_t session_id, const rvs_results_t* results) {
  // NOTE: This is a placeholder!
  // TODO: Use GRPC to send message back to client periodically
  printf(
      "/*******************************************************************/"
      "\n");
  printf("session id -> %d state -> %d\n", session_id, results->state);
  printf("session id -> %d status -> %d\n", session_id, results->status);
  printf("session id -> %d output -> %s\n", session_id, results->output_log);
  printf(
      "/*******************************************************************/"
      "\n");

  state = results->state;
  printf("state -> %d\n", state);
}
