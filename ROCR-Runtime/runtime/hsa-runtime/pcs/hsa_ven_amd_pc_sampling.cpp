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
#include "core/inc/agent.h"
#include "core/inc/amd_gpu_agent.h"
#include "core/inc/exceptions.h"

namespace rocr {
namespace AMD {
hsa_status_t handleException();

template <class T> static __forceinline T handleExceptionT() {
  handleException();
  abort();
  return T();
}
}  // namespace AMD

#define IS_OPEN()                                                                                  \
  do {                                                                                             \
    if (!core::Runtime::runtime_singleton_->IsOpen()) return HSA_STATUS_ERROR_NOT_INITIALIZED;     \
  } while (false)

template <class T> static __forceinline bool IsValid(T* ptr) {
  return (ptr == NULL) ? NULL : ptr->IsValid();
}

#define TRY try {
#define CATCH                                                                                      \
  }                                                                                                \
  catch (...) {                                                                                    \
    return AMD::handleException();                                                                 \
  }
#define CATCHRET(RETURN_TYPE)                                                                      \
  }                                                                                                \
  catch (...) {                                                                                    \
    return AMD::handleExceptionT<RETURN_TYPE>();                                                   \
  }

namespace pcs {

hsa_status_t hsa_ven_amd_pcs_iterate_configuration(
    hsa_agent_t hsa_agent, hsa_ven_amd_pcs_iterate_configuration_callback_t configuration_callback,
    void* callback_data) {
  TRY;
  IS_OPEN();

  core::Agent* agent = core::Agent::Convert(hsa_agent);
  if (agent == NULL || !agent->IsValid() || agent->device_type() != core::Agent::kAmdGpuDevice)
    return HSA_STATUS_ERROR_INVALID_AGENT;

  return PcsRuntime::instance()->PcSamplingIterateConfig(agent, configuration_callback,
                                                         callback_data);
  CATCH;
}

hsa_status_t hsa_ven_amd_pcs_create(hsa_agent_t hsa_agent, hsa_ven_amd_pcs_method_kind_t method,
                                    hsa_ven_amd_pcs_units_t units, size_t interval, size_t latency,
                                    size_t buffer_size,
                                    hsa_ven_amd_pcs_data_ready_callback_t data_ready_cb,
                                    void* client_cb_data, hsa_ven_amd_pcs_t* handle) {
  TRY;
  IS_OPEN();
  core::Agent* agent = core::Agent::Convert(hsa_agent);
  if (agent == NULL || !agent->IsValid() || agent->device_type() != core::Agent::kAmdGpuDevice)
    return HSA_STATUS_ERROR_INVALID_AGENT;

  return PcsRuntime::instance()->PcSamplingCreate(
      agent, method, units, interval, latency, buffer_size, data_ready_cb, client_cb_data, handle);
  CATCH;
}

hsa_status_t hsa_ven_amd_pcs_create_from_id(uint32_t pcs_id, hsa_agent_t hsa_agent,
                                            hsa_ven_amd_pcs_method_kind_t method,
                                            hsa_ven_amd_pcs_units_t units, size_t interval,
                                            size_t latency, size_t buffer_size,
                                            hsa_ven_amd_pcs_data_ready_callback_t data_ready_cb,
                                            void* client_cb_data, hsa_ven_amd_pcs_t* handle) {
  TRY;
  IS_OPEN();
  core::Agent* agent = core::Agent::Convert(hsa_agent);
  if (agent == NULL || !agent->IsValid() || agent->device_type() != core::Agent::kAmdGpuDevice)
    return HSA_STATUS_ERROR_INVALID_AGENT;

  return PcsRuntime::instance()->PcSamplingCreateFromId(pcs_id, agent, method, units, interval,
                                                        latency, buffer_size, data_ready_cb,
                                                        client_cb_data, handle);
  CATCH;
}

hsa_status_t hsa_ven_amd_pcs_destroy(hsa_ven_amd_pcs_t handle) {
  TRY;
  return PcsRuntime::instance()->PcSamplingDestroy(handle);
  CATCH;
}

hsa_status_t hsa_ven_amd_pcs_start(hsa_ven_amd_pcs_t handle) {
  TRY;
  return PcsRuntime::instance()->PcSamplingStart(handle);
  CATCH;
}

hsa_status_t hsa_ven_amd_pcs_stop(hsa_ven_amd_pcs_t handle) {
  TRY;
  return PcsRuntime::instance()->PcSamplingStop(handle);
  CATCH;
}

hsa_status_t hsa_ven_amd_pcs_flush(hsa_ven_amd_pcs_t handle) {
  TRY;
  return PcsRuntime::instance()->PcSamplingFlush(handle);
  CATCH;
}

void LoadPcSampling(core::PcSamplingExtTableInternal* pcs_api) {
  pcs_api->hsa_ven_amd_pcs_iterate_configuration_fn = hsa_ven_amd_pcs_iterate_configuration;
  pcs_api->hsa_ven_amd_pcs_create_fn = hsa_ven_amd_pcs_create;
  pcs_api->hsa_ven_amd_pcs_create_from_id_fn = hsa_ven_amd_pcs_create_from_id;
  pcs_api->hsa_ven_amd_pcs_destroy_fn = hsa_ven_amd_pcs_destroy;
  pcs_api->hsa_ven_amd_pcs_start_fn = hsa_ven_amd_pcs_start;
  pcs_api->hsa_ven_amd_pcs_stop_fn = hsa_ven_amd_pcs_stop;
  pcs_api->hsa_ven_amd_pcs_flush_fn = hsa_ven_amd_pcs_flush;
}

}  //  namespace pcs
}  //  namespace rocr
