/*
Copyright (c) 2022 - present Advanced Micro Devices, Inc. All rights reserved.

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

#include <dlfcn.h>
#include <math.h>
#include <sys/time.h>

#include <algorithm>
#include <cstring>
#include <filesystem>
#include <map>
#include <memory>
#include <stdexcept>
#include <vector>

#include "rdc/rdc.h"
#include "rdc_lib/RdcLogger.h"
#include "rdc_lib/RdcTelemetryLibInterface.h"
#include "rdc_lib/rdc_common.h"
#include "rdc_modules/rdc_rocp/RdcRocpBase.h"

std::unique_ptr<amd::rdc::RdcRocpBase> rocp_p;

bool is_rocp_disabled() {
  const char* value = std::getenv("RDC_DISABLE_ROCP");
  if (value == nullptr) return false;

  std::string value_str = value;
  std::transform(value_str.begin(), value_str.end(), value_str.begin(),
                 [](unsigned char c) { return std::tolower(c); });

  const std::vector<const char*> positive_list = {"yes", "true", "1", "on", "y", "t"};

  return std::any_of(positive_list.begin(), positive_list.end(),
                     [&value_str](const char* val) { return value_str == val; });
}

bool is_rocprofiler_metrics_path_set() {
  const char* rocprofiler_metrics_path = getenv("ROCPROFILER_METRICS_PATH");
  if (rocprofiler_metrics_path == nullptr) {
    return false;
  } else {
    return true;
  }
}

rdc_status_t rdc_module_init(uint64_t /*flags*/) {
  if (is_rocp_disabled()) {
    // rocprofiler does NOT work in gtest.
    // GTest starts up multiple instances of the progam under test,
    // however HSA does not allow for multiple instances. Since hsa_init() is called at the very
    // top of RdcRocpBase constructor - easiest to disable it alltogether when RDC_DISABLE_ROCP is
    // set.
    //
    // We cannot rely on GTEST_DECLARE_bool_ variable because RDC is compiled
    // before the tests are. Utilize an env var instead.
    return RDC_ST_DISABLED_MODULE;
  }

  if (is_rocprofiler_metrics_path_set() == false) {
    RDC_LOG(RDC_DEBUG, "Trying to find ROCPROFILER_METRICS_PATH");

    std::filesystem::path profiler_metrics_path;
    bool found_path = false;

    // Strategy 1: Find profiler_metrics_path relative to rocprofiler-sdk library
    Dl_info rocprofiler_dl_info = {};
    void* rocprofiler_symbol = dlsym(RTLD_DEFAULT, "rocprofiler_is_initialized");
    if (rocprofiler_symbol != nullptr && dladdr(rocprofiler_symbol, &rocprofiler_dl_info) != 0) {
      // NOTE: filesystem::canonical() returns absolute path, we need this for parent_path to work
      // properly.
      //
      // bad example without canonical path:
      //   rocprofiler is found in /opt/rocm/lib/rdc/../librocprofiler-sdk.so.
      //   location.parent_path().parent_path() will return /opt/rocm/lib/rdc.
      //   we're trying to find /opt/rocm!
      // with canonical path:
      //   rocprofiler is found in /opt/rocm/lib/rocprofiler-sdk.so
      //   location.parent_path().parent_path() will return /opt/rocm/.
      //   from here we can easily find /opt/rocm/share/rocprofiler-sdk
      try {
        const std::filesystem::path rocprofiler_lib_path(
            std::filesystem::canonical(rocprofiler_dl_info.dli_fname));
        RDC_LOG(RDC_DEBUG, "Found rocprofiler library at "
                               << rocprofiler_lib_path << " : "
                               << rocprofiler_lib_path.parent_path().parent_path());

        profiler_metrics_path =
            rocprofiler_lib_path.parent_path().parent_path() / "share/rocprofiler-sdk";
        if (std::filesystem::exists(profiler_metrics_path)) {
          found_path = true;
          RDC_LOG(RDC_DEBUG, "Found rocprofiler metrics via rocprofiler library location at "
                                 << profiler_metrics_path);
        }
      } catch (const std::filesystem::filesystem_error& e) {
        RDC_LOG(RDC_ERROR, "Failed to resolve rocprofiler library path: " << e.what());
      }
    }

    // Strategy 2: Find profiler_metrics_path relative to rdc
    if (!found_path) {
      RDC_LOG(RDC_DEBUG, "Rocprofiler library detection failed, trying fallback paths");
      Dl_info rdc_dl_info = {};
      if (dladdr(reinterpret_cast<const void*>(rdc_module_init), &rdc_dl_info) != 0) {
        try {
          const std::filesystem::path rdc_path(std::filesystem::canonical(rdc_dl_info.dli_fname));
          RDC_LOG(RDC_DEBUG, "Found rdc library at " << rdc_path);
          profiler_metrics_path =
              rdc_path.parent_path().parent_path().parent_path() / "share/rocprofiler-sdk";
          if (std::filesystem::exists(profiler_metrics_path)) {
            found_path = true;
            RDC_LOG(RDC_DEBUG,
                    "Found rocprofiler metrics via fallback path at " << profiler_metrics_path);
          }
        } catch (const std::filesystem::filesystem_error& e) {
          RDC_LOG(RDC_ERROR, "Failed to resolve rdc library path: " << e.what());
        }
      }
    }

    if (!found_path) {
      RDC_LOG(RDC_ERROR,
              "rocprofiler metrics are not available. Could not locate rocprofiler-sdk share "
              "directory.");
      RDC_LOG(RDC_ERROR,
              "Make sure rocprofiler-sdk is properly installed or set ROCPROFILER_METRICS_PATH "
              "manually.");
      return RDC_ST_FAIL_LOAD_MODULE;
    }

    RDC_LOG(RDC_DEBUG, "rocprofiler metrics are available at " << profiler_metrics_path);
    setenv("ROCPROFILER_METRICS_PATH", profiler_metrics_path.c_str(), 0);
  } else {
    RDC_LOG(RDC_DEBUG, "Using ROCPROFILER_METRICS_PATH from environment variable");
  }

  rocp_p = std::make_unique<amd::rdc::RdcRocpBase>();
  return RDC_ST_OK;
}
rdc_status_t rdc_module_destroy() {
  rocp_p.reset();
  return RDC_ST_OK;
}

// get supported field ids
rdc_status_t rdc_telemetry_fields_query(uint32_t field_ids[MAX_NUM_FIELDS], uint32_t* field_count) {
  // extract all keys from counter_map
  if (rocp_p == nullptr) {
    return RDC_ST_FAIL_LOAD_MODULE;
  }
  std::vector<rdc_field_t> fields = rocp_p->get_field_ids();
  std::vector<uint32_t> counter_keys(fields.begin(), fields.end());

  *field_count = counter_keys.size();
  // copy from vector into array
  std::copy(counter_keys.begin(), counter_keys.end(), field_ids);

  return RDC_ST_OK;
}

// Fetch
rdc_status_t rdc_telemetry_fields_value_get(rdc_gpu_field_t* fields, const uint32_t fields_count,
                                            rdc_field_value_f callback, void* user_data) {
  if (rocp_p == nullptr) {
    return RDC_ST_FAIL_LOAD_MODULE;
  }

  // Bulk fetch fields
  std::vector<rdc_gpu_field_value_t> bulk_results;

  struct timeval tv{};
  gettimeofday(&tv, nullptr);
  const uint64_t curTime = static_cast<uint64_t>(tv.tv_sec) * 1000 + tv.tv_usec / 1000;

  // Fetch it one by one for left fields
  const int BULK_FIELDS_MAX = 16;
  rdc_gpu_field_value_t values[BULK_FIELDS_MAX];
  uint32_t bulk_count = 0;
  rdc_status_t status = RDC_ST_UNKNOWN_ERROR;
  rdc_field_value_data data;
  rdc_field_type_t type = DOUBLE;

  for (uint32_t i = 0; i < fields_count; i++) {
    if (bulk_count >= BULK_FIELDS_MAX) {
      status = callback(values, bulk_count, user_data);
      // When the callback returns errors, stop processing and return.
      if (status != RDC_ST_OK) {
        return status;
      }
      bulk_count = 0;
    }

    status = rocp_p->rocp_lookup(fields[i], &data, &type);
    // get value
    values[bulk_count].gpu_index = fields[i].gpu_index;
    values[bulk_count].field_value.status = status;
    values[bulk_count].field_value.ts = curTime;
    values[bulk_count].field_value.type = type;
    values[bulk_count].field_value.field_id = fields[i].field_id;
    switch (type) {
      case DOUBLE:
        values[bulk_count].field_value.value.dbl = data.dbl;
        break;
      case INTEGER:
        values[bulk_count].field_value.value.l_int = data.l_int;
        break;
      case STRING:
      case BLOB:
        strncpy_with_null(values[bulk_count].field_value.value.str, data.str, RDC_MAX_STR_LENGTH);
        break;
      default:
        break;
    }
    bulk_count++;
  }
  if (bulk_count != 0) {
    status = callback(values, bulk_count, user_data);
    if (status != RDC_ST_OK) {
      return status;
    }
    bulk_count = 0;
  }

  return status;
}

rdc_status_t rdc_telemetry_fields_watch(rdc_gpu_field_t* fields, uint32_t fields_count) {
  rdc_status_t status = RDC_ST_OK;
  for (uint32_t i = 0; i < fields_count; i++) {
    RDC_LOG(RDC_DEBUG, "WATCH: " << fields[i].field_id);
    const rdc_status_t temp_status = RDC_ST_OK;
    if (temp_status != RDC_ST_OK) {
      status = temp_status;
    }
  }
  return status;
}

rdc_status_t rdc_telemetry_fields_unwatch(rdc_gpu_field_t* fields, uint32_t fields_count) {
  rdc_status_t status = RDC_ST_OK;
  for (uint32_t i = 0; i < fields_count; i++) {
    RDC_LOG(RDC_DEBUG, "UNWATCH: " << fields[i].field_id);
    const rdc_status_t temp_status = RDC_ST_OK;
    // return last non-ok status
    if (temp_status != RDC_ST_OK) {
      status = temp_status;
    }
  }
  return status;
}
