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
#include "rdc_lib/impl/RdcModuleMgrImpl.h"

#include <cassert>
#include <memory>
#include <type_traits>

#include "rdc/rdc.h"
#include "rdc_lib/RdcException.h"
#include "rdc_lib/RdcTelemetry.h"
#include "rdc_lib/impl/RdcDiagnosticModule.h"
#include "rdc_lib/impl/RdcRVSLib.h"
#include "rdc_lib/impl/RdcRocpLib.h"
#include "rdc_lib/impl/RdcRocrLib.h"
#include "rdc_lib/impl/RdcSmiLib.h"
#include "rdc_lib/impl/RdcTelemetryModule.h"

namespace amd {
namespace rdc {

// pass shared_ptr instead of creating it
template <typename T>
rdc_status_t RdcModuleMgrImpl::insert_modules(std::shared_ptr<T> ptr) {
  static_assert(std::is_base_of_v<RdcDiagnostic, T> || std::is_base_of_v<RdcTelemetry, T>);
  RDC_LOG(RDC_DEBUG, "Inserting module: " << typeid(T).name());
  // same module can service multiple subsystems
  // e.g. Diagnostics and Telemetry
  if constexpr (std::is_base_of_v<RdcDiagnostic, T>) {
    diagnostic_modules_.push_back(ptr);
  }
  if constexpr (std::is_base_of_v<RdcTelemetry, T>) {
    telemetry_modules_.push_back(ptr);
  }
  return RDC_ST_OK;
}

// base case
template <typename T>
rdc_status_t RdcModuleMgrImpl::insert_modules() {
  static_assert(std::is_base_of_v<RdcDiagnostic, T> || std::is_base_of_v<RdcTelemetry, T>);
  try {
    auto ptr = std::make_shared<T>();
    return insert_modules(ptr);
  } catch (RdcException& e) {
    RDC_LOG(RDC_ERROR, "Failed to insert module: " << typeid(T).name() << "\n" << e.what());
    return e.error_code();
  }
}

// recursive case
template <typename T, typename R, typename... TArgs>
rdc_status_t RdcModuleMgrImpl::insert_modules() {
  rdc_status_t status = insert_modules<T>();
  rdc_status_t status_recursive = insert_modules<R, TArgs...>();
  if (status == RDC_ST_OK) {
    status = status_recursive;
  }
  return status;
}

RdcModuleMgrImpl::RdcModuleMgrImpl(const RdcMetricFetcherPtr& fetcher) : fetcher_(fetcher) {
  // this module has a unique constructor and must be initialized explicitly
  try {
    auto smi_module = std::make_shared<RdcSmiLib>(fetcher);
    insert_modules(smi_module);
  } catch (RdcException& e) {
    RDC_LOG(RDC_ERROR, "Failed to insert module: " << typeid(RdcSmiLib).name() << "\n" << e.what());
  }

  // all other modules get initialized by insert_modules
  insert_modules<RdcRVSLib, RdcRocrLib, RdcRocpLib>();
}

RdcTelemetryPtr RdcModuleMgrImpl::get_telemetry_module() {
  if (rdc_telemetry_module_ == nullptr) {
    rdc_telemetry_module_.reset(new RdcTelemetryModule(telemetry_modules_));
  }
  return rdc_telemetry_module_;
}

RdcDiagnosticPtr RdcModuleMgrImpl::get_diagnostic_module() {
  if (rdc_diagnostic_module_ == nullptr) {
    rdc_diagnostic_module_.reset(new RdcDiagnosticModule(diagnostic_modules_));
  }
  return rdc_diagnostic_module_;
}

}  // namespace rdc
}  // namespace amd
