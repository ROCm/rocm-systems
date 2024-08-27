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
#ifndef INCLUDE_RDC_LIB_RDCPOLICY_H_
#define INCLUDE_RDC_LIB_RDCPOLICY_H_

#include <memory>
#include <vector>

#include "rdc/rdc.h"
#include "rdc_lib/rdc_common.h"

namespace amd {
namespace rdc {

class RdcPolicy {
 public:
  virtual rdc_status_t rdc_policy_set(rdc_gpu_group_t group_id, rdc_policy_t policy) = 0;

  virtual rdc_status_t rdc_policy_get(rdc_gpu_group_t group_id, uint32_t* count,
                                      rdc_policy_t policies[RDC_MAX_POLICY_SETTINGS]) = 0;

  virtual rdc_status_t rdc_policy_delete(rdc_gpu_group_t group_id,
                                         rdc_policy_condition_type_t condition_type) = 0;

  virtual rdc_status_t rdc_policy_register(rdc_gpu_group_t group_id,
                                           rdc_policy_register_callback callback) = 0;

  virtual rdc_status_t rdc_policy_unregister(rdc_gpu_group_t group_id) = 0;

  virtual ~RdcPolicy() {}
};

typedef std::shared_ptr<RdcPolicy> RdcPolicyPtr;

}  // namespace rdc
}  // namespace amd

#endif  // INCLUDE_RDC_LIB_RDCPOLICY_H_
