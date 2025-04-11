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
#ifndef INCLUDE_RDC_LIB_IMPL_RDCMETRICFETCHERIMPL_H_
#define INCLUDE_RDC_LIB_IMPL_RDCMETRICFETCHERIMPL_H_

#include <condition_variable>  // NOLINT(build/c++11)
#include <future>              // NOLINT(build/c++11)
#include <map>
#include <memory>
#include <mutex>  // NOLINT(build/c++11)
#include <queue>

#include "amd_smi/amdsmi.h"
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

// This union represents any SMI handles require initialization and/or
// shut down. There should only be one instance of this for each raw event
// used. For example, if a field group includes a pseudo-event and the
// underlying raw event, then only one FieldSMIData should be created,
// and it should be used by both events.
struct FieldSMIData {
  union {
    amdsmi_event_handle_t evt_handle;
  };
  union {
    amdsmi_counter_value_t counter_val;
  };
  ~FieldSMIData() {}
  FieldSMIData() : evt_handle(0), counter_val{0, 0, 0} {}
};

//!< The data structure to store the async fetch task
class RdcMetricFetcherImpl;
struct MetricTask {
  RdcFieldKey field;
  std::function<void(RdcMetricFetcherImpl&, RdcFieldKey)> task;
};

class RdcMetricFetcherImpl final : public RdcMetricFetcher {
 public:
  rdc_status_t fetch_smi_field(uint32_t gpu_index, rdc_field_t field_id,
                               rdc_field_value* value) override;
  rdc_status_t bulk_fetch_smi_fields(
      rdc_gpu_field_t* fields, uint32_t fields_count,
      std::vector<rdc_gpu_field_value_t>& results) override;  // NOLINT
  RdcMetricFetcherImpl();
  ~RdcMetricFetcherImpl() final;

  rdc_status_t acquire_smi_handle(RdcFieldKey fk) override;
  rdc_status_t delete_smi_handle(RdcFieldKey fk) override;

 private:
  std::shared_ptr<FieldSMIData> get_smi_data(RdcFieldKey key);

  uint64_t now();
  void get_ecc(uint32_t gpu_index, rdc_field_t field_id, rdc_field_value* value);
  void get_ecc_total(uint32_t gpu_index, rdc_field_t field_id, rdc_field_value* value);

  //!< return true if starting async_get
  bool async_get_pcie_throughput(uint32_t gpu_index, rdc_field_t field_id, rdc_field_value* value);
  void get_pcie_throughput(const RdcFieldKey& key);

  //!< Async metric retreive
  std::map<RdcFieldKey, MetricValue> async_metrics_;
  std::map<RdcFieldKey, std::shared_ptr<FieldSMIData>> smi_data_;
  std::queue<MetricTask> updated_tasks_;
  std::mutex task_mutex_;
  std::future<void> updater_;  // keep the future of updater
  std::condition_variable cv_;
  std::atomic<bool> task_started_;
};

}  // namespace rdc
}  // namespace amd

#endif  // INCLUDE_RDC_LIB_IMPL_RDCMETRICFETCHERIMPL_H_
