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

#include <filesystem>
#include <string>

#include "rdc/rdc.h"
#include "rdc_lib/RdcLogger.h"
#include "rdc_lib/rdc_common.h"
#include "rdc_modules/rdc_rvs/RvsBase.h"
#include "rvs/rvs.h"

#define CHECK_RVS(STATUS, SESSION)                                                               \
  do {                                                                                           \
    static_assert(std::is_same<decltype(STATUS), rvs_status_t>::value ||                         \
                      std::is_same<decltype(STATUS), rvs_status_t&>::value ||                    \
                      std::is_convertible<decltype(STATUS), rvs_status_t>::value,                \
                  "STATUS must be of type rvs_status_t");                                        \
    static_assert(std::is_same<decltype(SESSION), rvs_session_id_t>::value ||                    \
                      std::is_same<decltype(SESSION), rvs_session_id_t&>::value,                 \
                  "SESSION must be of type rvs_session_t");                                      \
    if ((STATUS) != RVS_STATUS_SUCCESS) {                                                        \
      RDC_LOG(RDC_ERROR,                                                                         \
              "RVS failed at[" << __FILE__ << ":" << __LINE__ << "] with status: " << (STATUS)); \
      rvs_session_destroy((SESSION));                                                            \
      return (STATUS);                                                                           \
    }                                                                                            \
  } while (0)

namespace amd::rdc {

RdcRVSBase* RdcRVSBase::s_instance = nullptr;
RdcRVSBase::RdcRVSBase() {
  std::string config_path(get_rocm_path("librdc.so"));
  s_instance = this;
  // these configs are installed with RDC and are mostly stripped down
  // versions of RVS configs
  config_path.append("/share/rdc/conf/rvs/");
  amdsmi_processor_handle processor_handle = nullptr;
  auto err = get_processor_handle_from_id(0, &processor_handle);
  if (err != AMDSMI_STATUS_SUCCESS) {
    RDC_LOG(RDC_ERROR, "get_processor_handle_from_id failed! " << err);
    return;
  }
  amdsmi_asic_info_t asic_info;
  err = amdsmi_get_gpu_asic_info(processor_handle, &asic_info);
  if (err != AMDSMI_STATUS_SUCCESS) {
    RDC_LOG(RDC_ERROR, "amdsmi_get_gpu_asic_info failed! " << err);
    return;
  }

  auto found_gpu = gfx_to_rvs_conf.find(asic_info.target_graphics_version);
  if (found_gpu == gfx_to_rvs_conf.end()) {
    // gpu name is not found
    RDC_LOG(RDC_INFO, "RVS couldn't match GFX version to name. Using \"default\"");
    config_path.append("default");
  } else {
    // gpu name is found - look up the name
    config_path.append(found_gpu->second);
  }

  RDC_LOG(RDC_DEBUG, "RVS CONFIG PATH: " << config_path);

  if (std::filesystem::exists(config_path)  && std::filesystem::is_directory(config_path)) {
    // populate configs
    for (auto& ent : std::filesystem::directory_iterator(config_path)) {
      if (ent.is_regular_file()) {
        _rvs_config_list.push_back(ent.path().string());
      }
    }
  }

  // map test enums to config paths
  for (rdc_diag_test_cases_t i = RDC_DIAG_TEST_FIRST; i < RDC_DIAG_TEST_LAST;
       i = static_cast<rdc_diag_test_cases_t>(i + 1)) {
    if (test_to_name.find(i) == test_to_name.end()) {
      continue;
    }
    for (int j = 0; j < _rvs_config_list.size(); j++) {
      std::filesystem::path config_path(_rvs_config_list.at(j));
      // error handling for path
      if (!config_path.has_filename()) {
        RDC_LOG(RDC_ERROR, "RVS config path has no filename: " << _rvs_config_list.at(j));
        continue;
      }
      // strip path, only keep filename
      std::string config = config_path.filename().string();
      if (test_to_name.at(i) == config) {
        _test_to_conf[i] = config_path.string();
        RDC_LOG(RDC_DEBUG, "TEST_ADDED " << test_to_name.at(i) << " = " << _test_to_conf[i]);
      }
    }
  }

  // manually add custom config
  _test_to_conf[RDC_DIAG_RVS_CUSTOM] = "";

  auto status = rvs_initialize();
  if (status != RVS_STATUS_SUCCESS) {
    RDC_LOG(RDC_ERROR, "rvs initialization failed");
  }
};

RdcRVSBase::~RdcRVSBase() {
  if (s_instance == this) {
    s_instance = nullptr;
  }
};

std::vector<std::string> RdcRVSBase::get_rvs_configs() { return _rvs_config_list; }

rvs_status_t RdcRVSBase::run_rvs_app(const char* config, const size_t config_size,
                                     rdc_diag_callback_t* callback) {
  char active_config[MAX_CONFIG_LENGTH];
  rvs_session_property_t session_property = {RVS_SESSION_TYPE_DEFAULT_CONF, {{RVS_MODULE_GST}}};
  rvs_session_id_t session_id = 0;
  rvs_status_t status = RVS_STATUS_FAILED;
  // NOTE: device_index is NOT set by RDC unless a custom config is provided.
  // Meaning RDC index has no impact on RVS index.

  if ((config == nullptr) || (config_size == 0)) {
    RDC_LOG(RDC_ERROR, "given config is NULL! Cannot run tests!");
    return RVS_STATUS_INVALID_ARGUMENT;
  } else if (config_size > MAX_CONFIG_LENGTH) {
    RDC_LOG(RDC_ERROR, "given config size is too large! Expected at most "
                           << MAX_CONFIG_LENGTH << ", got " << config_size << " instead.");
    return RVS_STATUS_INVALID_ARGUMENT;
  } else {
    RDC_LOG(RDC_DEBUG, "given config is correct");
    strncpy_with_null(active_config, config, config_size);
  }

  /*******************************/

  _state = RVS_SESSION_STATE_IDLE;

  /* Using custom gst configuration in string format */

  _callback = callback;
  status = rvs_session_create(&session_id, &RdcRVSBase::static_callback);

  CHECK_RVS(status, session_id);

  session_property.type = RVS_SESSION_TYPE_CUSTOM_CONF;
  session_property.custom_action.config = active_config;

  status = rvs_session_set_property(session_id, &session_property);

  CHECK_RVS(status, session_id);

  status = rvs_session_execute(session_id);

  CHECK_RVS(status, session_id);

  // TODO: remove?
  while (_state != RVS_SESSION_STATE_COMPLETED) {
  };

  _callback = nullptr;

  status = rvs_session_destroy(session_id);
  // this will try to destroy the session again, but it shouldn't matter
  // I don't want to define a second macro.
  CHECK_RVS(status, session_id);

  return status;
}

std::map<rdc_diag_test_cases_t, std::string> RdcRVSBase::get_test_to_conf() {
  return _test_to_conf;
}
}  // namespace amd::rdc
