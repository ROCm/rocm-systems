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
#include "rdc_lib/impl/RdcMetricFetcherImpl.h"
#include <sys/time.h>
#include <string.h>
#include <chrono>  //NOLINT
#include <algorithm>
#include <vector>
#include "rdc_lib/rdc_common.h"
#include "rdc_lib/RdcLogger.h"
#include "rocm_smi/rocm_smi.h"

namespace amd {
namespace rdc {

bool RdcMetricFetcherImpl::is_field_valid(uint32_t field_id) const {
     const std::vector<uint32_t> all_fields = {RDC_FI_GPU_MEMORY_USAGE,
     RDC_FI_GPU_MEMORY_TOTAL, RDC_FI_GPU_COUNT, RDC_FI_POWER_USAGE,
     RDC_FI_GPU_CLOCK, RDC_FI_GPU_UTIL, RDC_FI_DEV_NAME, RDC_FI_GPU_TEMP,
     RDC_FI_MEM_CLOCK, RDC_FI_PCIE_TX, RDC_FI_PCIE_RX,
     RDC_FI_ECC_CORRECT_TOTAL, RDC_FI_ECC_UNCORRECT_TOTAL, RDC_FI_MEMORY_TEMP};

     return std::find(all_fields.begin(), all_fields.end(), field_id)
               != all_fields.end();
}

RdcMetricFetcherImpl::RdcMetricFetcherImpl() {
    task_started_ = true;

    // kick off another thread for async fetch
    updater_ = std::async(std::launch::async, [this]() {
        while (task_started_) {
            std::unique_lock<std::mutex> lk(task_mutex_);
            // Wait for tasks or stop signal
            cv_.wait(lk,  [this]{
                return !updated_tasks_.empty() || !task_started_;
            });
            if (updated_tasks_.empty()) continue;

            // Get the tasks
            auto item = updated_tasks_.front();
            updated_tasks_.pop();
            // The task may take long time, release lock
            lk.unlock();

            // run task
            item.task(*this, item.field);
        }  // end while (task_started_)
    });
}

RdcMetricFetcherImpl::~RdcMetricFetcherImpl() {
    // Notify the async task to stop
    task_started_ = false;
    cv_.notify_all();
}

uint64_t RdcMetricFetcherImpl::now() {
    struct timeval  tv;
    gettimeofday(&tv, NULL);
    return static_cast<uint64_t>(tv.tv_sec) * 1000 + tv.tv_usec / 1000;
}

void RdcMetricFetcherImpl::get_ecc_error(uint32_t gpu_index,
                    uint32_t field_id, rdc_field_value* value) {
    rsmi_status_t err = RSMI_STATUS_SUCCESS;
    uint64_t correctable_err = 0;
    uint64_t uncorrectable_err = 0;
    rsmi_ras_err_state_t err_state;

    if (!value) {
      return;
    }
    for (uint32_t b = RSMI_GPU_BLOCK_FIRST;
                b <= RSMI_GPU_BLOCK_LAST; b = b*2) {
      err = rsmi_dev_ecc_status_get(gpu_index, static_cast<rsmi_gpu_block_t>(b),
                                                                  &err_state);
      if (err != RSMI_STATUS_SUCCESS) {
          RDC_LOG(RDC_INFO, "Get the ecc Status error " << b
                    << ":" << err);
          continue;
      }

      rsmi_error_count_t ec;
      err = rsmi_dev_ecc_count_get(gpu_index,
                    static_cast<rsmi_gpu_block_t>(b), &ec);

      if (err == RSMI_STATUS_SUCCESS) {
          correctable_err += ec.correctable_err;
          uncorrectable_err += ec.uncorrectable_err;
      }
    }

    value->status = RSMI_STATUS_SUCCESS;
    value->type = INTEGER;
    if (field_id == RDC_FI_ECC_CORRECT_TOTAL) {
        value->value.l_int =  correctable_err;
    }
    if (field_id == RDC_FI_ECC_UNCORRECT_TOTAL) {
        value->value.l_int = uncorrectable_err;
    }
}

void RdcMetricFetcherImpl::async_get_pcie_throughput(uint32_t gpu_index,
        uint32_t field_id, rdc_field_value* value) {
    if (!value) {
        return;
    }

    do {
        std::lock_guard<std::mutex> guard(task_mutex_);
        auto metric = async_metrics_.find({gpu_index, field_id});
        if ( metric != async_metrics_.end() ) {
            if (now() < metric->second.last_time + metric->second.cache_ttl) {
                RDC_LOG(RDC_DEBUG, "Fetch " << gpu_index << ":" <<
                        field_id_string(field_id) << " from cache");
                value->status = metric->second.value.status;
                value->type = metric->second.value.type;
                value->value = metric->second.value.value;
                return;
            }
        }

        // add to the async task queue
        MetricTask t;
        t.field = {gpu_index, field_id};
        t.task = &RdcMetricFetcherImpl::get_pcie_throughput;
        updated_tasks_.push(t);

        RDC_LOG(RDC_DEBUG, "Start async fetch " << gpu_index << ":" <<
                        field_id_string(field_id) << " to cache.");
    } while (0);
    cv_.notify_all();
}

void RdcMetricFetcherImpl::get_pcie_throughput(const RdcFieldKey& key) {
    uint32_t gpu_index = key.first;
    uint64_t sent, received, max_pkt_sz;
    rsmi_status_t ret;

    // Return if the cache does not expire yet
    do {
        std::lock_guard<std::mutex> guard(task_mutex_);
        auto metric = async_metrics_.find(key);
        if (metric != async_metrics_.end() &&
            now() < metric->second.last_time + metric->second.cache_ttl) {
            return;
        }
    } while (0);

    ret = rsmi_dev_pci_throughput_get(gpu_index, &sent, &received, &max_pkt_sz);

    uint64_t curTime = now();
    MetricValue value;
    value.cache_ttl = 30*1000;  // cache 30 seconds
    value.value.type = INTEGER;
    do {
        std::lock_guard<std::mutex> guard(task_mutex_);
        // Create new cache entry it does not exist
        auto tx_metric = async_metrics_.find({gpu_index, RDC_FI_PCIE_TX});
        if (tx_metric == async_metrics_.end()) {
            tx_metric = async_metrics_.insert(
                {{gpu_index, RDC_FI_PCIE_TX}, value}).first;
            tx_metric->second.value.field_id = RDC_FI_PCIE_TX;
        }
        auto rx_metric = async_metrics_.find({gpu_index, RDC_FI_PCIE_RX});
        if (rx_metric == async_metrics_.end()) {
            rx_metric = async_metrics_.insert(
                {{gpu_index, RDC_FI_PCIE_RX}, value}).first;
            rx_metric->second.value.field_id = RDC_FI_PCIE_RX;
        }

        // Always update the status and last_time
        tx_metric->second.last_time = curTime;
        tx_metric->second.value.status = ret;
        tx_metric->second.value.ts = curTime;

        rx_metric->second.last_time = curTime;
        rx_metric->second.value.status = ret;
        rx_metric->second.value.ts = curTime;

        if (ret == RSMI_STATUS_NOT_SUPPORTED) {
            RDC_LOG(RDC_ERROR,
                "PCIe throughput not supported on GPU " << gpu_index);
            return;
        }

        if (ret == RSMI_STATUS_SUCCESS) {
            rx_metric->second.value.value.l_int = received;
            tx_metric->second.value.value.l_int = sent;
            RDC_LOG(RDC_DEBUG, "Async updated " << gpu_index << ":" <<
                            "RDC_FI_PCIE_RX and RDC_FI_PCIE_TX to cache.");
        }
    } while (0);
}

rdc_status_t RdcMetricFetcherImpl::fetch_smi_field(uint32_t gpu_index,
         uint32_t field_id, rdc_field_value* value) {
    if (!value) {
         return RDC_ST_BAD_PARAMETER;
    }
    uint64_t i64 = 0;
    rsmi_temperature_type_t sensor_type;
    rsmi_clk_type_t clk_type;

    if (!is_field_valid(field_id)) {
         RDC_LOG(RDC_ERROR, "Fail to fetch field " << field_id
              << " which is not supported");
         return RDC_ST_NOT_SUPPORTED;
    }

    value->ts = now();
    value->field_id = field_id;
    value->status = RSMI_STATUS_NOT_SUPPORTED;

    switch (field_id) {
        case RDC_FI_GPU_MEMORY_USAGE:
             value->status = rsmi_dev_memory_usage_get(gpu_index,
               RSMI_MEM_TYPE_VRAM, &i64);
             value->type = INTEGER;
             if (value->status == RSMI_STATUS_SUCCESS) {
                value->value.l_int = static_cast<int64_t>(i64);
             }
             break;
        case RDC_FI_GPU_MEMORY_TOTAL:
             value->status = rsmi_dev_memory_total_get(gpu_index,
               RSMI_MEM_TYPE_VRAM, &i64);
             value->type = INTEGER;
             if (value->status == RSMI_STATUS_SUCCESS) {
               value->value.l_int = static_cast<int64_t>(i64);
             }
             break;
        case RDC_FI_GPU_COUNT:
             uint32_t num_gpu;
             value->status = rsmi_num_monitor_devices(&num_gpu);
             value->type = INTEGER;
             if (value->status == RSMI_STATUS_SUCCESS) {
                value->value.l_int = static_cast<int64_t>(num_gpu);
             }
             break;
        case RDC_FI_POWER_USAGE:
             value->status = rsmi_dev_power_ave_get(gpu_index,
               RSMI_TEMP_CURRENT, &i64);
             value->type = INTEGER;
             if (value->status == RSMI_STATUS_SUCCESS) {
                 value->value.l_int = static_cast<int64_t>(i64);
             }
             break;
        case RDC_FI_GPU_CLOCK:
        case RDC_FI_MEM_CLOCK:
             rsmi_frequencies_t f;
             clk_type = RSMI_CLK_TYPE_SYS;
             if (field_id == RDC_FI_MEM_CLOCK) {
                 clk_type = RSMI_CLK_TYPE_MEM;
             }
             value->status = rsmi_dev_gpu_clk_freq_get(gpu_index,
               clk_type, &f);
             value->type = INTEGER;
             if (value->status == RSMI_STATUS_SUCCESS) {
                value->value.l_int = f.frequency[f.current];
             }
             break;
        case RDC_FI_GPU_UTIL:
            uint32_t busy_percent;
            value->status = rsmi_dev_busy_percent_get(gpu_index, &busy_percent);
            value->type = INTEGER;
            if (value->status == RSMI_STATUS_SUCCESS) {
               value->value.l_int = static_cast<int64_t>(busy_percent);
            }
            break;
        case RDC_FI_DEV_NAME:
            value->status = rsmi_dev_name_get(gpu_index,
               value->value.str, RDC_MAX_STR_LENGTH);
            value->type = STRING;
            break;
        case RDC_FI_GPU_TEMP:
        case RDC_FI_MEMORY_TEMP:
            int64_t val_i64;
            sensor_type = RSMI_TEMP_TYPE_EDGE;
            if (field_id == RDC_FI_MEMORY_TEMP) {
                sensor_type = RSMI_TEMP_TYPE_MEMORY;
            }
            value->status = rsmi_dev_temp_metric_get(gpu_index,
                    sensor_type , RSMI_TEMP_CURRENT, &val_i64);

            value->type = INTEGER;
            if (value->status == RSMI_STATUS_SUCCESS) {
                  value->value.l_int = val_i64;
            }
            break;
         case RDC_FI_ECC_CORRECT_TOTAL:
         case RDC_FI_ECC_UNCORRECT_TOTAL:
            get_ecc_error(gpu_index, field_id, value);
            break;
         case RDC_FI_PCIE_TX:
         case RDC_FI_PCIE_RX:
            async_get_pcie_throughput(gpu_index, field_id, value);
            break;
        default:
            break;
    }

    int64_t latency = now()-value->ts;
    if (value->status != RSMI_STATUS_SUCCESS) {
         RDC_LOG(RDC_ERROR, "Fail to fetch " << gpu_index << ":" <<
              field_id_string(field_id) << " with rsmi error code "
              << value->status <<", latency " << latency);
    } else if (value->type == INTEGER) {
         RDC_LOG(RDC_DEBUG, "Fetch " << gpu_index << ":" <<
               field_id_string(field_id) << ":" << value->value.l_int
               << ", latency " << latency);
    } else if (value->type == DOUBLE) {
         RDC_LOG(RDC_DEBUG, "Fetch " << gpu_index << ":" <<
         field_id_string(field_id) << ":" << value->value.dbl
         << ", latency " << latency);
    } else if (value->type == STRING) {
         RDC_LOG(RDC_DEBUG, "Fetch " << gpu_index << ":" <<
         field_id_string(field_id) << ":" << value->value.str
         << ", latency " << latency);
    }

    return value->status == RSMI_STATUS_SUCCESS ?  RDC_ST_OK : RDC_ST_MSI_ERROR;
}


}  // namespace rdc
}  // namespace amd
