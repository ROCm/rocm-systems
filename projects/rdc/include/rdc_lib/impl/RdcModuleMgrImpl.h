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
#ifndef INCLUDE_RDC_LIB_IMPL_RDCMODULEMGRIMPL_H_
#define INCLUDE_RDC_LIB_IMPL_RDCMODULEMGRIMPL_H_

#include <list>

#include "rdc_lib/RdcMetricFetcher.h"
#include "rdc_lib/RdcModuleMgr.h"
#include "rdc_lib/RdcTelemetry.h"

namespace amd {
namespace rdc {

class RdcModuleMgrImpl : public RdcModuleMgr {
 public:
  RdcTelemetryPtr get_telemetry_module() override;
  RdcDiagnosticPtr get_diagnostic_module() override;
  explicit RdcModuleMgrImpl(const RdcMetricFetcherPtr& fetcher);

 private:
  //  Modules
  std::list<RdcDiagnosticPtr> diagnostic_modules_;
  std::list<RdcTelemetryPtr> telemetry_modules_;

  // base case
  template <typename T>
  rdc_status_t insert_modules();

  // recursive case
  template <typename T, typename R, typename... TArgs>
  rdc_status_t insert_modules();

  // pass shared_ptr instead of creating it
  template <typename T>
  rdc_status_t insert_modules(std::shared_ptr<T> ptr);

  //  Function module
  RdcTelemetryPtr rdc_telemetry_module_;
  RdcDiagnosticPtr rdc_diagnostic_module_;

  //  Domain module
  RdcMetricFetcherPtr fetcher_;
};

}  // namespace rdc
}  // namespace amd

#endif  // INCLUDE_RDC_LIB_IMPL_RDCMODULEMGRIMPL_H_
