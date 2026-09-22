/*
Copyright (c) 2024 - present Advanced Micro Devices, Inc. All rights reserved.

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
#ifndef INCLUDE_RDC_LIB_RDCCONFIGSETTINGS_H_
#define INCLUDE_RDC_LIB_RDCCONFIGSETTINGS_H_

#include <memory.h>

#include "rdc/rdc.h"

namespace amd {
namespace rdc {

class RdcConfigSettings {
 public:
  // Set one configure
  virtual rdc_status_t rdc_config_set(rdc_gpu_group_t group_id, rdc_config_setting_t setting) = 0;

  // Get the setting
  virtual rdc_status_t rdc_config_get(rdc_gpu_group_t group_id,
                                      rdc_config_setting_list_t* settings) = 0;

  // Clear the setting
  virtual rdc_status_t rdc_config_clear(rdc_gpu_group_t group_id) = 0;

  virtual ~RdcConfigSettings() {}
};
typedef std::shared_ptr<RdcConfigSettings> RdcConfigSettingsPtr;
}  // namespace rdc
}  // namespace amd

#endif  // INCLUDE_RDC_LIB_RDCCONFIGSETTINGS_H_
