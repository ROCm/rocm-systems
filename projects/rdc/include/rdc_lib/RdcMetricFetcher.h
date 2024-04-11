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
#ifndef INCLUDE_RDC_LIB_RDCMETRICFETCHER_H_
#define INCLUDE_RDC_LIB_RDCMETRICFETCHER_H_

#include <memory>
#include <vector>

#include "rdc/rdc.h"
#include "rdc_lib/RdcTelemetryLibInterface.h"
#include "rdc_lib/rdc_common.h"

namespace amd {
namespace rdc {

class RdcMetricFetcher {
 public:
  virtual rdc_status_t acquire_smi_handle(RdcFieldKey fk) = 0;
  virtual rdc_status_t delete_smi_handle(RdcFieldKey fk) = 0;

  virtual rdc_status_t fetch_smi_field(uint32_t gpu_index, rdc_field_t field_id,
                                       rdc_field_value* value) = 0;

  virtual rdc_status_t bulk_fetch_smi_fields(
      rdc_gpu_field_t* fields, uint32_t fields_count,
      std::vector<rdc_gpu_field_value_t>& results) = 0;  // NOLINT
  virtual ~RdcMetricFetcher() {}
};

typedef std::shared_ptr<RdcMetricFetcher> RdcMetricFetcherPtr;

}  // namespace rdc
}  // namespace amd

#endif  // INCLUDE_RDC_LIB_RDCMETRICFETCHER_H_
