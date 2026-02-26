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
#include "rdc_modules/rdc_rocp/RdcRocpSubprocess.h"

std::unique_ptr<amd::rdc::RdcRocpSubprocess> rocp_p;

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

rdc_status_t rdc_module_init(uint64_t /*flags*/) {
  if (is_rocp_disabled()) {
    return RDC_ST_DISABLED_MODULE;
  }

  // In subprocess mode, rocpctl handles its own rocprofiler initialization.
  // We only need to set ROCPROFILER_METRICS_PATH if the user hasn't already,
  // so the subprocess can find metrics definitions.
  if (std::getenv("ROCPROFILER_METRICS_PATH") == nullptr) {
    RDC_LOG(RDC_DEBUG, "ROCPROFILER_METRICS_PATH not set, attempting auto-detection for rocpctl");

    std::filesystem::path profiler_metrics_path;
    bool found_path = false;

    // Find relative to this shared library
    Dl_info dl_info = {};
    if (dladdr(reinterpret_cast<const void*>(rdc_module_init), &dl_info) != 0) {
      try {
        const std::filesystem::path lib_path(std::filesystem::canonical(dl_info.dli_fname));
        profiler_metrics_path =
            lib_path.parent_path().parent_path().parent_path() / "share/rocprofiler-sdk";
        if (std::filesystem::exists(profiler_metrics_path)) {
          found_path = true;
        }
      } catch (const std::filesystem::filesystem_error& e) {
        RDC_LOG(RDC_ERROR, "Failed to resolve rocprofiler metrics path: " << e.what());
      }
    }

    // Try standard ROCm location
    if (!found_path) {
      profiler_metrics_path = "/opt/rocm/share/rocprofiler-sdk";
      if (std::filesystem::exists(profiler_metrics_path)) {
        found_path = true;
      }
    }

    if (found_path) {
      RDC_LOG(RDC_DEBUG, "Setting ROCPROFILER_METRICS_PATH=" << profiler_metrics_path);
      setenv("ROCPROFILER_METRICS_PATH", profiler_metrics_path.c_str(), 0);
    } else {
      RDC_LOG(RDC_DEBUG, "Could not auto-detect ROCPROFILER_METRICS_PATH; "
                         "rocpctl may fail if not installed in standard location");
    }
  }

  rocp_p = std::make_unique<amd::rdc::RdcRocpSubprocess>();
  return RDC_ST_OK;
}

rdc_status_t rdc_module_destroy() {
  rocp_p.reset();
  return RDC_ST_OK;
}

rdc_status_t rdc_telemetry_fields_query(uint32_t field_ids[MAX_NUM_FIELDS], uint32_t* field_count) {
  if (rocp_p == nullptr) {
    return RDC_ST_FAIL_LOAD_MODULE;
  }
  std::vector<rdc_field_t> fields = rocp_p->get_field_ids();
  std::vector<uint32_t> counter_keys(fields.begin(), fields.end());

  *field_count = counter_keys.size();
  std::copy(counter_keys.begin(), counter_keys.end(), field_ids);

  return RDC_ST_OK;
}

rdc_status_t rdc_telemetry_fields_value_get(rdc_gpu_field_t* fields, const uint32_t fields_count,
                                            rdc_field_value_f callback, void* user_data) {
  if (rocp_p == nullptr) {
    return RDC_ST_FAIL_LOAD_MODULE;
  }

  struct timeval tv{};
  gettimeofday(&tv, nullptr);
  const uint64_t curTime = static_cast<uint64_t>(tv.tv_sec) * 1000 + tv.tv_usec / 1000;

  // Group fields by GPU index for bulk sampling
  std::map<uint32_t, std::vector<uint32_t>> gpu_to_field_indices;
  for (uint32_t i = 0; i < fields_count; i++) {
    gpu_to_field_indices[fields[i].gpu_index].push_back(i);
  }

  const int BULK_FIELDS_MAX = 16;
  rdc_gpu_field_value_t values[BULK_FIELDS_MAX];
  uint32_t bulk_count = 0;
  rdc_status_t status = RDC_ST_OK;

  for (const auto& [gpu_index, field_indices] : gpu_to_field_indices) {
    std::vector<rdc_gpu_field_t> gpu_fields;
    gpu_fields.reserve(field_indices.size());
    for (uint32_t idx : field_indices) {
      gpu_fields.push_back(fields[idx]);
    }

    std::vector<rdc_field_value_data> bulk_data;
    std::vector<rdc_field_type_t> bulk_types;
    std::vector<rdc_status_t> bulk_statuses;

    status = rocp_p->rocp_lookup_bulk(gpu_fields, bulk_data, bulk_types, bulk_statuses);
    if (status != RDC_ST_OK) {
      RDC_LOG(RDC_ERROR, "Error in bulk lookup for GPU " << gpu_index);
      continue;
    }

    for (size_t j = 0; j < gpu_fields.size(); j++) {
      if (bulk_count >= BULK_FIELDS_MAX) {
        status = callback(values, bulk_count, user_data);
        if (status != RDC_ST_OK) {
          return status;
        }
        bulk_count = 0;
      }

      const uint32_t original_idx = field_indices[j];
      values[bulk_count].gpu_index = fields[original_idx].gpu_index;
      values[bulk_count].field_value.status = bulk_statuses[j];
      values[bulk_count].field_value.ts = curTime;
      values[bulk_count].field_value.type = bulk_types[j];
      values[bulk_count].field_value.field_id = fields[original_idx].field_id;

      switch (bulk_types[j]) {
        case DOUBLE:
          values[bulk_count].field_value.value.dbl = bulk_data[j].dbl;
          break;
        case INTEGER:
          values[bulk_count].field_value.value.l_int = bulk_data[j].l_int;
          break;
        case STRING:
        case BLOB:
          strncpy_with_null(values[bulk_count].field_value.value.str, bulk_data[j].str,
                            RDC_MAX_STR_LENGTH);
          break;
        default:
          break;
      }
      bulk_count++;
    }
  }

  if (bulk_count != 0) {
    status = callback(values, bulk_count, user_data);
    if (status != RDC_ST_OK) {
      return status;
    }
  }

  return status;
}

rdc_status_t rdc_telemetry_fields_watch(rdc_gpu_field_t* fields, uint32_t fields_count) {
  rdc_status_t status = RDC_ST_OK;
  for (uint32_t i = 0; i < fields_count; i++) {
    RDC_LOG(RDC_DEBUG, "WATCH: " << fields[i].field_id);
  }
  return status;
}

rdc_status_t rdc_telemetry_fields_unwatch(rdc_gpu_field_t* fields, uint32_t fields_count) {
  rdc_status_t status = RDC_ST_OK;
  for (uint32_t i = 0; i < fields_count; i++) {
    RDC_LOG(RDC_DEBUG, "UNWATCH: " << fields[i].field_id);
  }
  return status;
}
