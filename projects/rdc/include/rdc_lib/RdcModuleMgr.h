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
#ifndef INCLUDE_RDC_LIB_RDCMODULEMGR_H_
#define INCLUDE_RDC_LIB_RDCMODULEMGR_H_

#include <memory>

#include "rdc_lib/RdcDiagnostic.h"
#include "rdc_lib/RdcTelemetry.h"

namespace amd {
namespace rdc {

class RdcModuleMgr {
 public:
  virtual ~RdcModuleMgr() = default;
  virtual RdcTelemetryPtr get_telemetry_module() = 0;
  virtual RdcDiagnosticPtr get_diagnostic_module() = 0;
};

typedef std::shared_ptr<RdcModuleMgr> RdcModuleMgrPtr;

}  // namespace rdc
}  // namespace amd

#endif  // INCLUDE_RDC_LIB_RDCMODULEMGR_H_
