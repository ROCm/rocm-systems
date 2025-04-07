////////////////////////////////////////////////////////////////////////////////
//
// The University of Illinois/NCSA
// Open Source License (NCSA)
//
// Copyright (c) 2023, Advanced Micro Devices, Inc. All rights reserved.
//
// Developed by:
//
//                 AMD Research and AMD HSA Software Development
//
//                 Advanced Micro Devices, Inc.
//
//                 www.amd.com
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to
// deal with the Software without restriction, including without limitation
// the rights to use, copy, modify, merge, publish, distribute, sublicense,
// and/or sell copies of the Software, and to permit persons to whom the
// Software is furnished to do so, subject to the following conditions:
//
//  - Redistributions of source code must retain the above copyright notice,
//    this list of conditions and the following disclaimers.
//  - Redistributions in binary form must reproduce the above copyright
//    notice, this list of conditions and the following disclaimers in
//    the documentation and/or other materials provided with the distribution.
//  - Neither the names of Advanced Micro Devices, Inc,
//    nor the names of its contributors may be used to endorse or promote
//    products derived from this Software without specific prior written
//    permission.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
// THE CONTRIBUTORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR
// OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE,
// ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
// DEALINGS WITH THE SOFTWARE.
//
////////////////////////////////////////////////////////////////////////////////

#include "pcs_runtime.h"

#include <assert.h>
#include <mutex>

#include "core/inc/runtime.h"

#include "core/inc/amd_gpu_agent.h"

namespace rocr {
namespace pcs {

#define IS_BAD_PTR(ptr)                                          \
do {                                                           \
  if ((ptr) == NULL) return HSA_STATUS_ERROR_INVALID_ARGUMENT; \
} while (false)


PcsRuntime* PcsRuntime::instance() {
  PcsRuntime* instance = get_instance().load(std::memory_order_acquire);
  if (instance == NULL) {
    // Protect the initialization from multi threaded access.
    std::lock_guard<std::mutex> lock(instance_mutex());

    // Make sure we are not initializing it twice.
    instance = get_instance().load(std::memory_order_relaxed);
    if (instance != NULL) {
      return instance;
    }

    instance = CreateSingleton();
    if (instance == NULL) {
      return NULL;
    }
  }

  return instance;
}

PcsRuntime* PcsRuntime::CreateSingleton() {
  PcsRuntime* instance = new PcsRuntime();

  get_instance().store(instance, std::memory_order_release);
  return instance;
}

void PcsRuntime::DestroySingleton() {
  PcsRuntime* instance = get_instance().load(std::memory_order_acquire);
  if (instance == NULL) {
    return;
  }

  get_instance().store(NULL, std::memory_order_release);
  delete instance;
}

void ReleasePcSamplingRsrcs() { PcsRuntime::DestroySingleton(); }

bool PcsRuntime::SessionsActive() const {
  return pc_sampling_.size() > 0;
}

PcsRuntime::PcSamplingSession::PcSamplingSession(
    core::Agent* _agent, hsa_ven_amd_pcs_method_kind_t method, hsa_ven_amd_pcs_units_t units,
    size_t interval, size_t latency, size_t buffer_size,
    hsa_ven_amd_pcs_data_ready_callback_t data_ready_callback, void* client_callback_data)
    : agent(_agent), thunkId_(0), active_(false), valid_(true), sample_size_(0) {
  switch (method) {
    case HSA_VEN_AMD_PCS_METHOD_HOSTTRAP_V1:
      sample_size_ = sizeof(perf_sample_hosttrap_v1_t);
      break;
    case HSA_VEN_AMD_PCS_METHOD_STOCHASTIC_V1:
      sample_size_ = sizeof(perf_sample_snapshot_v1_t);
      break;
    default:
      valid_ = false;
      return;
  }

  if (!interval || !buffer_size || (buffer_size % (2 * sample_size_))) {
    valid_ = false;
    return;
  }

  csd.method = method;
  csd.units = units;
  csd.interval = interval;
  csd.latency = latency;
  csd.buffer_size = buffer_size;
  csd.data_ready_callback = data_ready_callback;
  csd.client_callback_data = client_callback_data;

  data_rdy.buf1 = nullptr;
  data_rdy.buf1_sz = 0;
  data_rdy.buf2 = nullptr;
  data_rdy.buf2_sz = 0;
}

void PcsRuntime::PcSamplingSession::GetHsaKmtSamplingInfo(HsaPcSamplingInfo* sampleInfo) {
  sampleInfo->value_min = 0;
  sampleInfo->value_max = 0;
  sampleInfo->flags = 0;
  sampleInfo->value = csd.interval;

  switch (csd.method) {
    case HSA_VEN_AMD_PCS_METHOD_HOSTTRAP_V1:
      sampleInfo->method = HSA_PC_SAMPLING_METHOD_KIND_HOSTTRAP_V1;
      break;
    case HSA_VEN_AMD_PCS_METHOD_STOCHASTIC_V1:
      sampleInfo->method = HSA_PC_SAMPLING_METHOD_KIND_STOCHASTIC_V1;
      break;
  }

  switch (csd.units) {
    case HSA_VEN_AMD_PCS_INTERVAL_UNITS_MICRO_SECONDS:
      sampleInfo->units = HSA_PC_SAMPLING_UNIT_INTERVAL_MICROSECONDS;
      break;
    case HSA_VEN_AMD_PCS_INTERVAL_UNITS_CLOCK_CYCLES:
      sampleInfo->units = HSA_PC_SAMPLING_UNIT_INTERVAL_CYCLES;
      break;
    case HSA_VEN_AMD_PCS_INTERVAL_UNITS_INSTRUCTIONS:
      sampleInfo->units = HSA_PC_SAMPLING_UNIT_INTERVAL_INSTRUCTIONS;
      break;
  }
}

hsa_status_t PcSamplingDataCopyCallback(void* _session, size_t bytes_to_copy, void* destination) {
  assert(_session);
  assert(destination);

  PcsRuntime::PcSamplingSession* session =
      reinterpret_cast<PcsRuntime::PcSamplingSession*>(_session);

  return session->DataCopyCallback(reinterpret_cast<uint8_t*>(destination), bytes_to_copy);
}

hsa_status_t PcsRuntime::PcSamplingSession::DataCopyCallback(uint8_t* buffer,
                                                             size_t bytes_to_copy) {
  if (bytes_to_copy != (data_rdy.buf1_sz + data_rdy.buf2_sz)) return HSA_STATUS_ERROR_EXCEPTION;

  if (data_rdy.buf1_sz) memcpy(buffer, data_rdy.buf1, data_rdy.buf1_sz);
  if (data_rdy.buf2_sz) memcpy(buffer + data_rdy.buf1_sz, data_rdy.buf2, data_rdy.buf2_sz);

  return HSA_STATUS_SUCCESS;
}

hsa_status_t PcsRuntime::PcSamplingSession::HandleSampleData(uint8_t* buf1, size_t buf1_sz,
                                                             uint8_t* buf2, size_t buf2_sz,
                                                             size_t lost_sample_count) {
  data_rdy.buf1 = buf1;
  data_rdy.buf1_sz = buf1_sz;
  data_rdy.buf2 = buf2;
  data_rdy.buf2_sz = buf2_sz;

  AMD::GpuAgent* gpuAgent = static_cast<AMD::GpuAgent*>(agent);

  switch (csd.method) {
    case HSA_VEN_AMD_PCS_METHOD_HOSTTRAP_V1: {
      size_t buf_samples = buf1_sz / sizeof(perf_sample_hosttrap_v1_t);
      perf_sample_hosttrap_v1_t* samples = reinterpret_cast<perf_sample_hosttrap_v1_t*>(buf1);
      while (buf_samples--) {
        samples->timestamp = gpuAgent->TranslateTime(samples->timestamp);
        samples++;
      }

      buf_samples = buf2_sz / sizeof(perf_sample_hosttrap_v1_t);
      samples = reinterpret_cast<perf_sample_hosttrap_v1_t*>(buf2);
      while (buf_samples--) {
        samples->timestamp = gpuAgent->TranslateTime(samples->timestamp);
        samples++;
      }
    }
    break;
    case HSA_VEN_AMD_PCS_METHOD_STOCHASTIC_V1: {
      size_t buf_samples = buf1_sz / sizeof(perf_sample_snapshot_v1_t);
      perf_sample_snapshot_v1_t* samples = reinterpret_cast<perf_sample_snapshot_v1_t*>(buf1);
      while (buf_samples--) {
        samples->timestamp = gpuAgent->TranslateTime(samples->timestamp);
        samples++;
      }

      buf_samples = buf2_sz / sizeof(perf_sample_snapshot_v1_t);
      samples = reinterpret_cast<perf_sample_snapshot_v1_t*>(buf2);
      while (buf_samples--) {
        samples->timestamp = gpuAgent->TranslateTime(samples->timestamp);
        samples++;
      }
    }
    break;
  }

  csd.data_ready_callback(csd.client_callback_data, buf1_sz + buf2_sz, lost_sample_count,
                          &PcSamplingDataCopyCallback,
                          /* hsa_callback_data*/ this);
  return HSA_STATUS_SUCCESS;
}

hsa_status_t PcsRuntime::PcSamplingIterateConfig(
    core::Agent* agent, hsa_ven_amd_pcs_iterate_configuration_callback_t configuration_callback,
    void* callback_data) {
  AMD::GpuAgentInt* gpu_agent = static_cast<AMD::GpuAgentInt*>(agent);
  return gpu_agent->PcSamplingIterateConfig(configuration_callback, callback_data);
}

hsa_status_t PcsRuntime::PcSamplingCreate(core::Agent* agent, hsa_ven_amd_pcs_method_kind_t method,
                                          hsa_ven_amd_pcs_units_t units, size_t interval,
                                          size_t latency, size_t buffer_size,
                                          hsa_ven_amd_pcs_data_ready_callback_t data_ready_cb,
                                          void* client_cb_data, hsa_ven_amd_pcs_t* handle) {

  IS_BAD_PTR(handle);
  IS_BAD_PTR(data_ready_cb);

  return PcSamplingCreateInternal(
      agent, method, units, interval, latency, buffer_size, data_ready_cb, client_cb_data, handle,
      [](core::Agent* agent_, PcSamplingSession& session_) {
        return static_cast<AMD::GpuAgentInt*>(agent_)->PcSamplingCreate(session_);
      });
}

hsa_status_t PcsRuntime::PcSamplingCreateFromId(uint32_t ioctl_pcs_id, core::Agent* agent,
                                                hsa_ven_amd_pcs_method_kind_t method,
                                                hsa_ven_amd_pcs_units_t units, size_t interval,
                                                size_t latency, size_t buffer_size,
                                                hsa_ven_amd_pcs_data_ready_callback_t data_ready_cb,
                                                void* client_cb_data, hsa_ven_amd_pcs_t* handle) {
  IS_BAD_PTR(handle);
  IS_BAD_PTR(data_ready_cb);

  return PcSamplingCreateInternal(
      agent, method, units, interval, latency, buffer_size, data_ready_cb, client_cb_data, handle,
      [&](core::Agent* agent_, PcSamplingSession& session_) {
        return static_cast<AMD::GpuAgentInt*>(agent_)->PcSamplingCreateFromId(ioctl_pcs_id,
                                                                              session_);
      });
}

hsa_status_t PcsRuntime::PcSamplingCreateInternal(
    core::Agent* agent, hsa_ven_amd_pcs_method_kind_t method, hsa_ven_amd_pcs_units_t units,
    size_t interval, size_t latency, size_t buffer_size,
    hsa_ven_amd_pcs_data_ready_callback_t data_ready_cb, void* client_cb_data,
    hsa_ven_amd_pcs_t* handle, agent_pcs_create_fn_t agent_pcs_create_fn) {
  ScopedAcquire<KernelMutex> lock(&pc_sampling_lock_);

  handle->handle = ++pc_sampling_id_;
  // create a new PcSamplingSession(agent, method, units, interval, latency, buffer_size,
  // data_ready_cb, client_cb_data) reference and insert into pc_sampling_
  pc_sampling_.emplace(std::piecewise_construct, std::forward_as_tuple(handle->handle),
                       std::forward_as_tuple(agent, method, units, interval, latency, buffer_size,
                                             data_ready_cb, client_cb_data));

  if (!pc_sampling_[handle->handle].isValid()) {
      pc_sampling_.erase(handle->handle);
      return HSA_STATUS_ERROR_INVALID_ARGUMENT;
  }

  hsa_status_t ret = agent_pcs_create_fn(agent, pc_sampling_[handle->handle]);
  if (ret != HSA_STATUS_SUCCESS) {
    pc_sampling_.erase(handle->handle);
    return ret;
  }
  return HSA_STATUS_SUCCESS;
}

hsa_status_t PcsRuntime::PcSamplingDestroy(hsa_ven_amd_pcs_t handle) {
  ScopedAcquire<KernelMutex> lock(&pc_sampling_lock_);
  auto pcSamplingSessionIt = pc_sampling_.find(reinterpret_cast<uint64_t>(handle.handle));
  if (pcSamplingSessionIt == pc_sampling_.end()) {
    debug_warning(false && "Cannot find PcSampling session");
    return HSA_STATUS_ERROR_INVALID_ARGUMENT;
  }
  AMD::GpuAgentInt* gpu_agent = static_cast<AMD::GpuAgentInt*>(pcSamplingSessionIt->second.agent);

  hsa_status_t ret = gpu_agent->PcSamplingDestroy(pcSamplingSessionIt->second);
  pc_sampling_.erase(pcSamplingSessionIt);
  return ret;
}

hsa_status_t PcsRuntime::PcSamplingStart(hsa_ven_amd_pcs_t handle) {
  ScopedAcquire<KernelMutex> lock(&pc_sampling_lock_);
  auto pcSamplingSessionIt = pc_sampling_.find(reinterpret_cast<uint64_t>(handle.handle));
  if (pcSamplingSessionIt == pc_sampling_.end()) {
    debug_warning(false && "Cannot find PcSampling session");
    return HSA_STATUS_ERROR_INVALID_ARGUMENT;
  }
  AMD::GpuAgentInt* gpu_agent = static_cast<AMD::GpuAgentInt*>(pcSamplingSessionIt->second.agent);

  return gpu_agent->PcSamplingStart(pcSamplingSessionIt->second);
}

hsa_status_t PcsRuntime::PcSamplingStop(hsa_ven_amd_pcs_t handle) {
  ScopedAcquire<KernelMutex> lock(&pc_sampling_lock_);
  auto pcSamplingSessionIt = pc_sampling_.find(reinterpret_cast<uint64_t>(handle.handle));
  if (pcSamplingSessionIt == pc_sampling_.end()) {
    debug_warning(false && "Cannot find PcSampling session");
    return HSA_STATUS_ERROR_INVALID_ARGUMENT;
  }
  AMD::GpuAgentInt* gpu_agent = static_cast<AMD::GpuAgentInt*>(pcSamplingSessionIt->second.agent);

  return gpu_agent->PcSamplingStop(pcSamplingSessionIt->second);
}

hsa_status_t PcsRuntime::PcSamplingFlush(hsa_ven_amd_pcs_t handle) {
  ScopedAcquire<KernelMutex> lock(&pc_sampling_lock_);
  auto pcSamplingSessionIt = pc_sampling_.find(reinterpret_cast<uint64_t>(handle.handle));
  if (pcSamplingSessionIt == pc_sampling_.end()) {
    debug_warning(false && "Cannot find PcSampling session");
    return HSA_STATUS_ERROR_INVALID_ARGUMENT;
  }
  AMD::GpuAgentInt* gpu_agent = static_cast<AMD::GpuAgentInt*>(pcSamplingSessionIt->second.agent);

  return gpu_agent->PcSamplingFlush(pcSamplingSessionIt->second);
}

}  // namespace pcs
}  // namespace rocr
