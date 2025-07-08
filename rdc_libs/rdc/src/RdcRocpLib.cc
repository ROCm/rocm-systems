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
#include "rdc_lib/impl/RdcRocpLib.h"

#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <string>

#include "rdc/rdc.h"
#include "rdc_lib/RdcException.h"
#include "rdc_lib/RdcTelemetryLibInterface.h"

namespace amd {
namespace rdc {

// TODO: Add init and destroy calls support
RdcRocpLib::RdcRocpLib()
    : telemetry_fields_query_(nullptr),
      telemetry_fields_value_get_(nullptr),
      telemetry_fields_watch_(nullptr),
      telemetry_fields_unwatch_(nullptr),
      rdc_module_init_(nullptr),
      rdc_module_destroy_(nullptr) {
  // must happen before library is loaded
  rdc_unset_hsa_tools_lib();

  rdc_status_t status = lib_loader_.load("librdc_rocp.so");
  if (status != RDC_ST_OK) {
    RDC_LOG(RDC_ERROR, "Rocp related function will not work.");
    return;
  }

  status = lib_loader_.load_symbol(&rdc_module_init_, "rdc_module_init");
  if (status != RDC_ST_OK) {
    rdc_module_init_ = nullptr;
    return;
  }

  status = rdc_module_init_(0);
  if (status != RDC_ST_OK) {
    if (status == RDC_ST_DISABLED_MODULE) {
      RDC_LOG(RDC_INFO, "Module: librdc_rocp.so is intentionally disabled!");
    } else {
      RDC_LOG(RDC_ERROR,
              "Fail to init librdc_rocp.so:" << rdc_status_string(status)
                                             << ". ROCP related function will not work.");
    }
    return;
  }

  status = lib_loader_.load_symbol(&rdc_module_destroy_, "rdc_module_destroy");
  if (status != RDC_ST_OK) {
    rdc_module_destroy_ = nullptr;
  }

  status = lib_loader_.load_symbol(&telemetry_fields_query_, "rdc_telemetry_fields_query");
  if (status != RDC_ST_OK) {
    telemetry_fields_query_ = nullptr;
  }

  status = lib_loader_.load_symbol(&telemetry_fields_value_get_, "rdc_telemetry_fields_value_get");
  if (status != RDC_ST_OK) {
    telemetry_fields_value_get_ = nullptr;
  }

  status = lib_loader_.load_symbol(&telemetry_fields_watch_, "rdc_telemetry_fields_watch");
  if (status != RDC_ST_OK) {
    telemetry_fields_watch_ = nullptr;
  }

  status = lib_loader_.load_symbol(&telemetry_fields_unwatch_, "rdc_telemetry_fields_unwatch");
  if (status != RDC_ST_OK) {
    telemetry_fields_unwatch_ = nullptr;
  }
}

RdcRocpLib::~RdcRocpLib() = default;

// get support field ids
rdc_status_t RdcRocpLib::rdc_telemetry_fields_query(uint32_t field_ids[MAX_NUM_FIELDS],
                                                    uint32_t* field_count) {
  if (field_count == nullptr) {
    return RDC_ST_BAD_PARAMETER;
  }
  if (telemetry_fields_query_ == nullptr) {
    return RDC_ST_FAIL_LOAD_MODULE;
  }

  return telemetry_fields_query_(field_ids, field_count);
}

// Fetch
rdc_status_t RdcRocpLib::rdc_telemetry_fields_value_get(rdc_gpu_field_t* fields,
                                                        uint32_t fields_count,
                                                        rdc_field_value_f callback,
                                                        void* user_data) {
  if (fields == nullptr) {
    return RDC_ST_BAD_PARAMETER;
  }
  if (telemetry_fields_value_get_ == nullptr) {
    return RDC_ST_FAIL_LOAD_MODULE;
  }

  return telemetry_fields_value_get_(fields, fields_count, callback, user_data);
}

rdc_status_t RdcRocpLib::rdc_telemetry_fields_watch(rdc_gpu_field_t* fields,
                                                    uint32_t fields_count) {
  if (fields == nullptr) {
    return RDC_ST_BAD_PARAMETER;
  }
  if (telemetry_fields_watch_ == nullptr) {
    return RDC_ST_FAIL_LOAD_MODULE;
  }

  return telemetry_fields_watch_(fields, fields_count);
}

rdc_status_t RdcRocpLib::rdc_telemetry_fields_unwatch(rdc_gpu_field_t* fields,
                                                      uint32_t fields_count) {
  if (fields == nullptr) {
    return RDC_ST_BAD_PARAMETER;
  }
  if (telemetry_fields_unwatch_ == nullptr) {
    return RDC_ST_FAIL_LOAD_MODULE;
  }

  return telemetry_fields_unwatch_(fields, fields_count);
}

void RdcRocpLib::rdc_unset_hsa_tools_lib() {
  int status = unsetenv("HSA_TOOLS_LIB");
  if (status != 0) {
    RDC_LOG(RDC_ERROR, "Failed to unset HSA_TOOLS_LIB environment variable.");
  }
}

}  // namespace rdc
}  // namespace amd
