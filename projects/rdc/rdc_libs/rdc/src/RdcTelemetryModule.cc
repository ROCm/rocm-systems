/*
Copyright (c) 2020 - present Advanced Micro Devices, Inc. All rights reserved.

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
#include "rdc_lib/impl/RdcTelemetryModule.h"

#include <algorithm>
#include <memory>

#include "rdc_lib/RdcException.h"
#include "rdc_lib/RdcLogger.h"
#include "rdc_lib/RdcMetricFetcher.h"
#include "rdc_lib/impl/RdcSmiLib.h"

namespace amd {
namespace rdc {

// Return all supported fields, at most MAX_NUM_FIELDS (the size of field_ids), and
// RDC_ST_MAX_LIMIT when the libraries support more. A library that fails adds none.
rdc_status_t RdcTelemetryModule::rdc_telemetry_fields_query(uint32_t field_ids[MAX_NUM_FIELDS],
                                                            uint32_t* field_count) {
  if (field_count == nullptr) {
    return RDC_ST_BAD_PARAMETER;
  }
  rdc_status_t result = RDC_ST_OK;
  *field_count = 0;
  // Each library may write MAX_NUM_FIELDS IDs, more than field_ids has room left for.
  std::vector<uint32_t> library_ids(MAX_NUM_FIELDS);
  for (auto& library : telemetry_modules_) {
    uint32_t count = 0;
    if (library->rdc_telemetry_fields_query(library_ids.data(), &count) != RDC_ST_OK) {
      continue;
    }
    count = std::min<uint32_t>(count, MAX_NUM_FIELDS);
    const uint32_t room = MAX_NUM_FIELDS - *field_count;
    if (count > room) {
      count = room;
      result = RDC_ST_MAX_LIMIT;
    }
    std::copy_n(library_ids.begin(), count, field_ids + *field_count);
    *field_count += count;
  }

  return result;
}

rdc_status_t RdcTelemetryModule::rdc_telemetry_fields_watch(rdc_gpu_field_t* fields,
                                                            uint32_t fields_count) {
  if (fields == nullptr) {
    return RDC_ST_BAD_PARAMETER;
  }

  std::map<RdcTelemetryPtr, std::vector<rdc_gpu_field_t>> fields_in_module;
  std::vector<rdc_gpu_field_value_t> unsupport_fields;
  get_fields_for_module(fields, fields_count, fields_in_module, unsupport_fields);

  // Every library gets its fields; the first failure is returned.
  rdc_status_t result = RDC_ST_OK;
  auto ite = fields_in_module.begin();
  for (; ite != fields_in_module.end(); ite++) {
    if (ite->second.size() > 0) {
      rdc_status_t status =
          ite->first->rdc_telemetry_fields_watch(&ite->second[0], ite->second.size());
      if (result == RDC_ST_OK) {
        result = status;
      }
    }
  }

  return result;
}

rdc_status_t RdcTelemetryModule::rdc_telemetry_fields_unwatch(rdc_gpu_field_t* fields,
                                                              uint32_t fields_count) {
  if (fields == nullptr) {
    return RDC_ST_BAD_PARAMETER;
  }

  std::map<RdcTelemetryPtr, std::vector<rdc_gpu_field_t>> fields_in_module;
  std::vector<rdc_gpu_field_value_t> unsupport_fields;
  get_fields_for_module(fields, fields_count, fields_in_module, unsupport_fields);

  // Every library gets its fields; the first failure is returned.
  rdc_status_t result = RDC_ST_OK;
  auto ite = fields_in_module.begin();
  for (; ite != fields_in_module.end(); ite++) {
    if (ite->second.size() > 0) {
      rdc_status_t status =
          ite->first->rdc_telemetry_fields_unwatch(&ite->second[0], ite->second.size());
      if (result == RDC_ST_OK) {
        result = status;
      }
    }
  }

  return result;
}

RdcTelemetryModule::RdcTelemetryModule(std::list<RdcTelemetryPtr> telemetry_modules)
    : telemetry_modules_(telemetry_modules) {
  auto ite = telemetry_modules_.begin();
  for (; ite != telemetry_modules_.end(); ite++) {
    uint32_t field_ids[MAX_NUM_FIELDS];
    uint32_t field_count = 0;
    rdc_status_t status = (*ite)->rdc_telemetry_fields_query(field_ids, &field_count);
    if (status == RDC_ST_OK) {
      for (uint32_t index = 0; index < field_count && index < MAX_NUM_FIELDS; index++) {
        fields_id_module_.insert({field_ids[index], (*ite)});
      }
    }
  }
}

void RdcTelemetryModule::get_fields_for_module(
    rdc_gpu_field_t* fields, uint32_t fields_count,
    std::map<RdcTelemetryPtr, std::vector<rdc_gpu_field_t>>& fields_in_module,
    std::vector<rdc_gpu_field_value_t>& unsupport_fields) const {
  for (uint32_t findex = 0; findex < fields_count; findex++) {
    // find(), not operator[], which would add an entry for every unknown field: only the
    // constructor fills the map.
    auto found = fields_id_module_.find(fields[findex].field_id);
    if (found != fields_id_module_.end()) {
      fields_in_module[found->second].push_back(fields[findex]);
    } else {
      RDC_LOG(RDC_DEBUG, "Unsupported field " << field_id_string(fields[findex].field_id));
      rdc_gpu_field_value_t value;
      value.gpu_index = fields[findex].gpu_index;
      value.field_value.field_id = fields[findex].field_id;
      value.field_value.status = RDC_ST_NOT_SUPPORTED;
      unsupport_fields.push_back(value);
    }
  }
}

rdc_status_t RdcTelemetryModule::rdc_telemetry_fields_value_get(rdc_gpu_field_t* fields,
                                                                uint32_t fields_count,
                                                                rdc_field_value_f callback,
                                                                void* user_data) {
  if (fields == nullptr) {
    return RDC_ST_BAD_PARAMETER;
  }

  // Dispatch the fields to the libraries
  std::map<RdcTelemetryPtr, std::vector<rdc_gpu_field_t>> fields_to_fetch;
  std::vector<rdc_gpu_field_value_t> unsupport_fields;
  get_fields_for_module(fields, fields_count, fields_to_fetch, unsupport_fields);

  // Every library gets its fields; the first failure is returned.
  rdc_status_t result = RDC_ST_OK;
  for (auto& [module, due] : fields_to_fetch) {
    // A library takes at most MAX_NUM_FIELDS fields per call, but the watches due in one
    // update can hold more.
    for (size_t first = 0; first < due.size(); first += MAX_NUM_FIELDS) {
      const size_t n = std::min<size_t>(MAX_NUM_FIELDS, due.size() - first);
      rdc_status_t status = module->rdc_telemetry_fields_value_get(
          &due[first], static_cast<uint32_t>(n), callback, user_data);
      if (result == RDC_ST_OK) {
        result = status;
      }
    }
  }

  // Notify the caller unsupported fields
  if (!unsupport_fields.empty()) {
    callback(unsupport_fields.data(), static_cast<uint32_t>(unsupport_fields.size()), user_data);
  }

  return result;
}

}  // namespace rdc
}  // namespace amd
