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

#include "rdc_lib/impl/RdcDiagnosticModule.h"
#include "rdc_lib/impl/RdcRasLib.h"
#include "rdc_lib/impl/RdcRocrLib.h"
#include "rdc_lib/impl/RdcTelemetryModule.h"

namespace amd {
namespace rdc {

RdcModuleMgrImpl::RdcModuleMgrImpl(const RdcMetricFetcherPtr& fetcher)
    : fetcher_(fetcher) {}

RdcTelemetryPtr RdcModuleMgrImpl::get_telemetry_module() {
    if (rdc_telemetry_module_) {
        return rdc_telemetry_module_;
    }

    if (!rdc_telemetry_module_) {
        rdc_telemetry_module_.reset(
            new RdcTelemetryModule(fetcher_));
    }

    return rdc_telemetry_module_;
}

RdcDiagnosticPtr RdcModuleMgrImpl::get_diagnostic_module() {
    if (rdc_diagnostic_module_) {
        return rdc_diagnostic_module_;
    }

    if (!rdc_diagnostic_module_) {
        rdc_diagnostic_module_.reset(
            new RdcDiagnosticModule(fetcher_));
    }

    return rdc_diagnostic_module_;
}

}  // namespace rdc
}  // namespace amd
