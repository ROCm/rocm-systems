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
#ifndef INCLUDE_RDC_LIB_IMPL_RDCTOPOLINKYIMPL_H_
#define INCLUDE_RDC_LIB_IMPL_RDCTOPOLINKYIMPL_H_

#include <atomic>
#include <future>
#include <map>
#include <memory>
#include <mutex>  // NOLINT
#include <string>
#include <utility>
#include <vector>

#include "amd_smi/amdsmi.h"
#include "rdc_lib/RdcGroupSettings.h"
#include "rdc_lib/RdcMetricFetcher.h"
#include "rdc_lib/RdcTopologyLink.h"

namespace amd {
namespace rdc {

class RdcTopologyLinkImpl : public RdcTopologyLink {
 public:
  RdcTopologyLinkImpl(const RdcGroupSettingsPtr& group_settings,
                      RdcMetricFetcherPtr metric_fetcher);
  ~RdcTopologyLinkImpl();

  rdc_status_t rdc_device_topology_get(uint32_t gpu_index, rdc_device_topology_t* results) override;
  rdc_status_t rdc_link_status_get(rdc_link_status_t* results) override;

 private:
  RdcGroupSettingsPtr group_settings_;
  RdcMetricFetcherPtr metric_fetcher_;
  
};

}  // namespace rdc
}  // namespace amd

#endif  // INCLUDE_RDC_LIB_IMPL_RDCTOPOLINKYIMPL_H_