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
#ifndef RDC_LIB_IMPL_RDCRASLIB_H_
#define RDC_LIB_IMPL_RDCRASLIB_H_

#include <map>
#include <list>
#include <vector>
#include <memory>
#include <algorithm>
#include <string>
#include "rdc_lib/RdcLibraryLoader.h"
#include "rdc_lib/RdcTelemetry.h"


namespace amd {
namespace rdc {
class RdcRasLib: public RdcTelemetry {
 public:
    // get support field ids
    rdc_status_t rdc_telemetry_fields_query(uint32_t field_ids[MAX_NUM_FIELDS],
         uint32_t* field_count) override;

    // Fetch
    rdc_status_t rdc_telemetry_fields_value_get(rdc_gpu_field_t* fields,
      uint32_t fields_count, rdc_field_value_f callback,
      void*  user_data) override;

    rdc_status_t rdc_telemetry_fields_watch(rdc_gpu_field_t* fields,
      uint32_t fields_count) override;

    rdc_status_t rdc_telemetry_fields_unwatch(rdc_gpu_field_t* fields,
        uint32_t fields_count) override;

    explicit RdcRasLib(const char* lib_name);

 private:
    RdcLibraryLoader lib_loader_;
    rdc_status_t (*fields_value_get_)(rdc_gpu_field_t*,
                uint32_t, rdc_field_value_f, void*);
    rdc_status_t (*fields_query_)(uint32_t[MAX_NUM_FIELDS], uint32_t*);
};
typedef std::shared_ptr<RdcRasLib> RdcRasLibPtr;


}  // namespace rdc
}  // namespace amd


#endif  // RDC_LIB_IMPL_RDCRASLIB_H_
