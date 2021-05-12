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
#include <assert.h>

#include <chrono>  //NOLINT
#include <algorithm>
#include <vector>
#include <set>
#include "rdc_lib/rdc_common.h"
#include "common/rdc_fields_supported.h"
#include "rdc_lib/RdcLogger.h"
#include "rocm_smi/rocm_smi.h"
#include "rdc_lib/impl/RsmiUtils.h"
#include "common/rdc_capabilities.h"

namespace amd {
namespace rdc {

static const std::unordered_map<rdc_field_t, rsmi_event_type_t>
                                                     rdc_evnt_2_rsmi_field = {
    {RDC_EVNT_XGMI_0_NOP_TX,      RSMI_EVNT_XGMI_0_NOP_TX},
    {RDC_EVNT_XGMI_0_REQ_TX,  RSMI_EVNT_XGMI_0_REQUEST_TX},
    {RDC_EVNT_XGMI_0_RESP_TX, RSMI_EVNT_XGMI_0_RESPONSE_TX},
    {RDC_EVNT_XGMI_0_BEATS_TX,    RSMI_EVNT_XGMI_0_BEATS_TX},
    {RDC_EVNT_XGMI_1_NOP_TX,      RSMI_EVNT_XGMI_1_NOP_TX},
    {RDC_EVNT_XGMI_1_REQ_TX,  RSMI_EVNT_XGMI_1_REQUEST_TX},
    {RDC_EVNT_XGMI_1_RESP_TX, RSMI_EVNT_XGMI_1_RESPONSE_TX},
    {RDC_EVNT_XGMI_1_BEATS_TX,    RSMI_EVNT_XGMI_1_BEATS_TX},

    {RDC_EVNT_XGMI_0_THRPUT, RSMI_EVNT_XGMI_DATA_OUT_0},
    {RDC_EVNT_XGMI_1_THRPUT, RSMI_EVNT_XGMI_DATA_OUT_1},
    {RDC_EVNT_XGMI_2_THRPUT, RSMI_EVNT_XGMI_DATA_OUT_2},
    {RDC_EVNT_XGMI_3_THRPUT, RSMI_EVNT_XGMI_DATA_OUT_3},
    {RDC_EVNT_XGMI_4_THRPUT, RSMI_EVNT_XGMI_DATA_OUT_4},
    {RDC_EVNT_XGMI_5_THRPUT, RSMI_EVNT_XGMI_DATA_OUT_5},
};

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
                               rdc_field_t field_id, rdc_field_value* value) {
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

bool RdcMetricFetcherImpl::async_get_pcie_throughput(uint32_t gpu_index,
    rdc_field_t field_id, rdc_field_value* value) {
    if (!value) {
        return false;
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
                return false;
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

    return true;
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

rdc_status_t RdcMetricFetcherImpl::bulk_fetch_smi_fields(
        rdc_gpu_field_t* fields, uint32_t fields_count,
        std::vector<rdc_gpu_field_value_t>& results) { // NOLINT
    const std::set<rdc_field_t> rdc_bulk_fields = {
      RDC_FI_GPU_CLOCK,  // current_gfxclk * 1000000
      RDC_FI_MEMORY_TEMP,  // temperature_mem * 1000
      RDC_FI_GPU_TEMP,  // temperature_edge * 1000
      RDC_FI_POWER_USAGE,  //  average_socket_power * 1000000
      RDC_FI_GPU_UTIL  // average_gfx_activity
    };

    // To prevent always call the bulk API even if it is not supported,
    // the static is used to cache last try.
    static rsmi_status_t rs = RSMI_STATUS_SUCCESS;
    if (rs !=  RSMI_STATUS_SUCCESS) {
      results.clear();
      return RDC_ST_NOT_SUPPORTED;
    }

    // Organize the fields per GPU
    std::map<uint32_t, std::vector<rdc_field_t>> bulk_fields;
    for (uint32_t i = 0; i < fields_count; i++) {
      if (rdc_bulk_fields.find(fields[i].field_id) != rdc_bulk_fields.end()) {
        bulk_fields[fields[i].gpu_index].push_back(fields[i].field_id);
      }
    }

    // Call the rocm_smi_lib API to bulk fetch the data
    auto cur_time = now();
    auto ite = bulk_fields.begin();
    for (; ite != bulk_fields.end(); ite++) {
      rsmi_gpu_metrics_t gpu_metrics;
      rs = rsmi_dev_gpu_metrics_info_get(ite->first, &gpu_metrics);
      if (rs != RSMI_STATUS_SUCCESS) {
        results.clear();
        return RDC_ST_NOT_SUPPORTED;
      }
      for (uint32_t j=0; j < ite->second.size(); j++) {
        auto field_id = ite->second[j];
        rdc_gpu_field_value_t value;
        value.gpu_index = ite->first;
        value.field_value.field_id = field_id;
        value.field_value.type = INTEGER;
        value.field_value.status = RSMI_STATUS_SUCCESS;
        value.field_value.ts = cur_time;

        switch (field_id) {
          case RDC_FI_GPU_CLOCK:   // current_gfxclk * 1000000
            value.field_value.value.l_int =
              static_cast<int64_t>(gpu_metrics.current_gfxclk * 1000000);
            break;
          case RDC_FI_MEMORY_TEMP:   // temperature_mem * 1000
            value.field_value.value.l_int =
              static_cast<int64_t>(gpu_metrics.temperature_mem * 1000);
            break;
          case RDC_FI_GPU_TEMP:   // temperature_edge * 1000
            value.field_value.value.l_int =
              static_cast<int64_t>(gpu_metrics.temperature_edge * 1000);
            break;
          case RDC_FI_POWER_USAGE:   // average_socket_power * 1000000
            value.field_value.value.l_int =
              static_cast<int64_t>(gpu_metrics.average_socket_power * 1000000);
            // Ignore if the power is 0, which will fallback to non-bulk fetch.
            if (value.field_value.value.l_int == 0) {
              RDC_LOG(RDC_DEBUG, "Bulk fetch " << value.gpu_index << ":" <<
                            "RDC_FI_POWER_USAGE fallback to regular way.");
              continue;
            }
            break;
          case RDC_FI_GPU_UTIL:   // average_gfx_activity
            value.field_value.value.l_int =
              static_cast<int64_t>(gpu_metrics.average_gfx_activity);
            break;
          default:
             value.field_value.status = RSMI_STATUS_NOT_SUPPORTED;
            break;
        }
        if (value.field_value.status == RSMI_STATUS_SUCCESS) {
          results.push_back(value);
        }
      }
    }

    return RDC_ST_OK;
}

static const uint64_t kGig = 1000000000;

rdc_status_t RdcMetricFetcherImpl::fetch_smi_field(uint32_t gpu_index,
    rdc_field_t field_id, rdc_field_value* value) {
    if (!value) {
         return RDC_ST_BAD_PARAMETER;
    }
    uint64_t i64 = 0;
    rsmi_temperature_type_t sensor_type;
    rsmi_clk_type_t clk_type;
    bool async_fetching = false;
    RdcFieldKey f_key(gpu_index, field_id);
    std::shared_ptr<FieldRSMIData> rsmi_data;
    double coll_time_sec;

    if (!is_field_valid(field_id)) {
         RDC_LOG(RDC_ERROR, "Fail to fetch field " << field_id
              << " which is not supported");
         return RDC_ST_NOT_SUPPORTED;
    }

    value->ts = now();
    value->field_id = field_id;
    value->status = RSMI_STATUS_NOT_SUPPORTED;

    auto read_rsmi_counter = [&](void) {
      rsmi_data = get_rsmi_data(f_key);
      if (rsmi_data == nullptr) {
        value->status = RSMI_STATUS_NOT_SUPPORTED;
        return;
      }

      value->status = rsmi_counter_read(rsmi_data->evt_handle,
                                                     &rsmi_data->counter_val);
      value->value.l_int = rsmi_data->counter_val.value;
      value->type = INTEGER;
    };

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
            async_fetching = async_get_pcie_throughput(
                            gpu_index, field_id, value);
            break;
         case RDC_EVNT_XGMI_0_NOP_TX:
         case RDC_EVNT_XGMI_0_REQ_TX:
         case RDC_EVNT_XGMI_0_RESP_TX:
         case RDC_EVNT_XGMI_0_BEATS_TX:
         case RDC_EVNT_XGMI_1_NOP_TX:
         case RDC_EVNT_XGMI_1_REQ_TX:
         case RDC_EVNT_XGMI_1_RESP_TX:
         case RDC_EVNT_XGMI_1_BEATS_TX:
           read_rsmi_counter();
           break;
         case RDC_EVNT_XGMI_0_THRPUT:
         case RDC_EVNT_XGMI_1_THRPUT:
         case RDC_EVNT_XGMI_2_THRPUT:
         case RDC_EVNT_XGMI_3_THRPUT:
         case RDC_EVNT_XGMI_4_THRPUT:
         case RDC_EVNT_XGMI_5_THRPUT:
           read_rsmi_counter();
           if (value->status == RDC_ST_OK) {
             if (rsmi_data->counter_val.time_running > 0) {
               coll_time_sec =
                 static_cast<float>(rsmi_data->counter_val.time_running)/kGig;
               value->value.l_int = (value->value.l_int * 32)/coll_time_sec;
             } else {
               value->value.l_int = 0;
             }
           }
           break;

        default:
            break;
    }

    int64_t latency = now()-value->ts;
    if (value->status != RSMI_STATUS_SUCCESS) {
        if (async_fetching) {  //!< Async fetching is not an error
            RDC_LOG(RDC_DEBUG, "Async fetch " << field_id_string(field_id));
        } else {
            RDC_LOG(RDC_ERROR, "Fail to fetch " << gpu_index << ":" <<
              field_id_string(field_id) << " with rsmi error code "
              << value->status <<", latency " << latency);
        }
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

    return value->status == RSMI_STATUS_SUCCESS ? RDC_ST_OK : RDC_ST_MSI_ERROR;
}

std::shared_ptr<FieldRSMIData>
RdcMetricFetcherImpl::get_rsmi_data(RdcFieldKey key) {
  std::map<RdcFieldKey, std::shared_ptr<FieldRSMIData>>::iterator r_info =
                                                        rsmi_data_.find(key);

  if (r_info != rsmi_data_.end()) {
    return r_info->second;
  }
  return nullptr;
}

static rdc_status_t init_rsmi_counter(RdcFieldKey fk,
                        rsmi_event_group_t grp, rsmi_event_handle_t *handle) {
  rsmi_status_t ret;
  uint32_t counters_available;
  uint32_t dv_ind = fk.first;
  rdc_field_t f = fk.second;

  assert(handle != nullptr);

  ret = rsmi_dev_counter_group_supported(dv_ind, grp);

  if (ret != RSMI_STATUS_SUCCESS) {
    return Rsmi2RdcError(ret);
  }

  ret = rsmi_counter_available_counters_get(dv_ind, grp, &counters_available);
  if (ret != RSMI_STATUS_SUCCESS) {
    return Rsmi2RdcError(ret);
  }
  if (counters_available == 0) {
    return RDC_ST_INSUFF_RESOURCES;
  }

  rsmi_event_type_t evt = rdc_evnt_2_rsmi_field.at(f);

  // Temporarily get DAC capability
  ScopedCapability sc(CAP_DAC_OVERRIDE, CAP_EFFECTIVE);

  if (sc.error()) {
    RDC_LOG(RDC_ERROR,
              "Failed to acquire required capabilities. Errno " << sc.error());
    return RDC_ST_PERM_ERROR;
  }

  ret = rsmi_dev_counter_create(dv_ind, evt, handle);
  if (ret != RSMI_STATUS_SUCCESS) {
    return Rsmi2RdcError(ret);
  }

  ret = rsmi_counter_control(*handle, RSMI_CNTR_CMD_START, nullptr);

  // Release DAC capability
  sc.Relinquish();

  if (sc.error()) {
    RDC_LOG(RDC_ERROR,
                    "Failed to relinquish capabilities. Errno " << sc.error());
    return RDC_ST_PERM_ERROR;
  }

  return Rsmi2RdcError(ret);
}

rdc_status_t RdcMetricFetcherImpl::delete_rsmi_handle(RdcFieldKey fk) {
  rsmi_status_t ret;

  switch (fk.second) {
    case RDC_EVNT_XGMI_0_NOP_TX:
    case RDC_EVNT_XGMI_0_REQ_TX:
    case RDC_EVNT_XGMI_0_RESP_TX:
    case RDC_EVNT_XGMI_0_BEATS_TX:
    case RDC_EVNT_XGMI_1_NOP_TX:
    case RDC_EVNT_XGMI_1_REQ_TX:
    case RDC_EVNT_XGMI_1_RESP_TX:
    case RDC_EVNT_XGMI_1_BEATS_TX:
    case RDC_EVNT_XGMI_0_THRPUT:
    case RDC_EVNT_XGMI_1_THRPUT:
    case RDC_EVNT_XGMI_2_THRPUT:
    case RDC_EVNT_XGMI_3_THRPUT:
    case RDC_EVNT_XGMI_4_THRPUT:
    case RDC_EVNT_XGMI_5_THRPUT: {
      rsmi_event_handle_t h;
      if (rsmi_data_.find(fk) == rsmi_data_.end()) {
        return RDC_ST_NOT_SUPPORTED;
      }

      h = rsmi_data_[fk]->evt_handle;

      // Stop counting.
      ret = rsmi_counter_control(h, RSMI_CNTR_CMD_STOP, nullptr);
      if (ret != RSMI_STATUS_SUCCESS) {
        rsmi_data_.erase(fk);

        RDC_LOG(RDC_ERROR, "Error in stopping event counter: " <<
                                                          Rsmi2RdcError(ret));
        return Rsmi2RdcError(ret);
      }

      // Release all resources (e.g., counter and memory resources) associated
      // with evnt_handle.
      ret = rsmi_dev_counter_destroy(h);

      rsmi_data_.erase(fk);
      return Rsmi2RdcError(ret);
    }
    default:
      return RDC_ST_NOT_SUPPORTED;
  }
}

rdc_status_t RdcMetricFetcherImpl::acquire_rsmi_handle(RdcFieldKey fk) {
  rdc_status_t ret;

  auto get_evnt_handle = [&](rsmi_event_group_t grp) {
    rsmi_event_handle_t handle;
    rdc_status_t result;

    if (get_rsmi_data(fk) != nullptr) {
      // This event has already been initialized.
      return RDC_ST_ALREADY_EXIST;
    }

    result = init_rsmi_counter(fk, grp, &handle);
    if (result != RDC_ST_OK) {
      RDC_LOG(RDC_ERROR, "Failed to init RSMI counter. Return:" << result);
      return result;
    }
    auto fsh = std::shared_ptr<FieldRSMIData>(new FieldRSMIData);

    if (fsh == nullptr) {
      return RDC_ST_INSUFF_RESOURCES;
    }

    fsh->evt_handle = handle;

    rsmi_data_[fk] = fsh;

    return RDC_ST_OK;
  };

  switch (fk.second) {
    case RDC_EVNT_XGMI_0_NOP_TX:
    case RDC_EVNT_XGMI_0_REQ_TX:
    case RDC_EVNT_XGMI_0_RESP_TX:
    case RDC_EVNT_XGMI_0_BEATS_TX:
    case RDC_EVNT_XGMI_1_NOP_TX:
    case RDC_EVNT_XGMI_1_REQ_TX:
    case RDC_EVNT_XGMI_1_RESP_TX:
    case RDC_EVNT_XGMI_1_BEATS_TX:
      ret = get_evnt_handle(RSMI_EVNT_GRP_XGMI);
      break;

    case RDC_EVNT_XGMI_0_THRPUT:
    case RDC_EVNT_XGMI_1_THRPUT:
    case RDC_EVNT_XGMI_2_THRPUT:
    case RDC_EVNT_XGMI_3_THRPUT:
    case RDC_EVNT_XGMI_4_THRPUT:
    case RDC_EVNT_XGMI_5_THRPUT:
      ret = get_evnt_handle(RSMI_EVNT_GRP_XGMI_DATA_OUT);
      break;

    default:
      break;
  }

  if (ret == RDC_ST_INSUFF_RESOURCES) {
    amd::rdc::fld_id2name_map_t &field_id_to_descript =
                                 amd::rdc::get_field_id_description_from_id();

    RDC_LOG(RDC_ERROR, "No event counters are available for " <<
                     field_id_to_descript.at(fk.second).enum_name << " event.");
  } else if (ret != RDC_ST_OK) {
    RDC_LOG(RDC_ERROR, "Error in getting event counter handle: " << ret);
  }
  return ret;
}

}  // namespace rdc
}  // namespace amd
