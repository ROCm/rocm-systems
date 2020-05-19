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
#ifndef RDC_LIB_IMPL_RDCMETRICFETCHERIMPL_H_
#define RDC_LIB_IMPL_RDCMETRICFETCHERIMPL_H_

#include <mutex>  // NOLINT(build/c++11)
#include <future>  // NOLINT(build/c++11)
#include <condition_variable>  // NOLINT(build/c++11)
#include <map>
#include <queue>
#include "rdc_lib/RdcMetricFetcher.h"
#include "rdc_lib/rdc_common.h"

namespace amd {
namespace rdc {

//!< Some metrics, like PCIe throughput may take a second to retreive. The
//!< MetricValue will cache those metrics for async retreive.
struct MetricValue {
    uint64_t cache_ttl;
    uint64_t last_time;
    rdc_field_value value;
};


//!< The data structure to store the async fetch task
class RdcMetricFetcherImpl;
struct MetricTask {
    RdcFieldKey field;
    std::function<void(RdcMetricFetcherImpl&, RdcFieldKey)> task;
};

class RdcMetricFetcherImpl: public RdcMetricFetcher {
 public:
    rdc_status_t fetch_smi_field(uint32_t gpu_index,
        uint32_t field_id, rdc_field_value* value) override;
    bool is_field_valid(uint32_t field_id) const override;
    RdcMetricFetcherImpl();
    ~RdcMetricFetcherImpl();
 private:
    uint64_t now();
    void get_ecc_error(uint32_t gpu_index,
        uint32_t field_id, rdc_field_value* value);

    //!< return true if starting async_get
    bool async_get_pcie_throughput(uint32_t gpu_index,
        uint32_t field_id, rdc_field_value* value);
    void get_pcie_throughput(const RdcFieldKey& key);

    //!< Async metric retreive
    std::map<RdcFieldKey, MetricValue> async_metrics_;
    std::queue<MetricTask> updated_tasks_;
    std::mutex task_mutex_;
    std::future<void> updater_;  // keep the future of updater
    std::condition_variable cv_;
    std::atomic<bool> task_started_;
};

}  // namespace rdc
}  // namespace amd

#endif  // RDC_LIB_IMPL_RDCMETRICFETCHERIMPL_H_
