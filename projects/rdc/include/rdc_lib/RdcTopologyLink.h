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
#ifndef INCLUDE_RDC_LIB_RDCTOPOLOGYLINK_H_
#define INCLUDE_RDC_LIB_RDCTOPOLOGYLINK_H_

#include <memory>
#include <vector>

#include "rdc/rdc.h"
#include "rdc_lib/rdc_common.h"

namespace amd {
namespace rdc {

class RdcTopologyLink {
 public:
  virtual rdc_status_t rdc_device_topology_get(uint32_t gpu_index,
                                               rdc_device_topology_t* results) = 0;
  virtual rdc_status_t rdc_link_status_get(rdc_link_status_t* results) = 0;

  virtual ~RdcTopologyLink() {}
};

typedef std::shared_ptr<RdcTopologyLink> RdcTopologyLinkPtr;

}  // namespace rdc
}  // namespace amd

#endif  // INCLUDE_RDC_LIB_RDCTOPOLOGYLINK_H_