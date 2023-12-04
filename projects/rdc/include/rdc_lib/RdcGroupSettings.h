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
#ifndef INCLUDE_RDC_LIB_RDCGROUPSETTINGS_H_
#define INCLUDE_RDC_LIB_RDCGROUPSETTINGS_H_

#include <memory>

#include "rdc/rdc.h"
#include "rdc_lib/rdc_common.h"

namespace amd {
namespace rdc {

class RdcGroupSettings {
 public:
  virtual rdc_status_t rdc_group_gpu_create(const char* group_name,
                                            rdc_gpu_group_t* p_rdc_group_id) = 0;
  virtual rdc_status_t rdc_group_gpu_destroy(rdc_gpu_group_t p_rdc_group_id) = 0;
  virtual rdc_status_t rdc_group_gpu_add(rdc_gpu_group_t groupId, uint32_t gpu_index) = 0;
  virtual rdc_status_t rdc_group_gpu_get_info(rdc_gpu_group_t p_rdc_group_id,
                                              rdc_group_info_t* p_rdc_group_info) = 0;
  virtual rdc_status_t rdc_group_get_all_ids(rdc_gpu_group_t group_id_list[], uint32_t* count) = 0;

  virtual rdc_status_t rdc_group_field_create(uint32_t num_field_ids, rdc_field_t* field_ids,
                                              const char* field_group_name,
                                              rdc_field_grp_t* rdc_field_group_id) = 0;
  virtual rdc_status_t rdc_group_field_destroy(rdc_field_grp_t rdc_field_group_id) = 0;
  virtual rdc_status_t rdc_group_field_get_info(rdc_field_grp_t rdc_field_group_id,
                                                rdc_field_group_info_t* field_group_info) = 0;
  virtual rdc_status_t rdc_group_field_get_all_ids(rdc_field_grp_t field_group_id_list[],
                                                   uint32_t* count) = 0;

  virtual ~RdcGroupSettings() {}
};

typedef std::shared_ptr<RdcGroupSettings> RdcGroupSettingsPtr;
const uint32_t JOB_FIELD_ID = 0;

}  // namespace rdc
}  // namespace amd

#endif  // INCLUDE_RDC_LIB_RDCGROUPSETTINGS_H_
