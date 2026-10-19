// Copyright © Advanced Micro Devices, Inc., or its affiliates.
//
// SPDX-License-Identifier: MIT

#include "core/inc/launch_descriptor.h"

#include "core/inc/agent.h"
#include "core/inc/exceptions.h"
#include "core/inc/runtime.h"

hsa_status_t HSA_API hsa_ven_amd_launch_descriptor_create(
    hsa_agent_t agent_handle,
    amd_launch_descriptor_t* launch_descriptor) {
  try {
    if (!rocr::core::Runtime::runtime_singleton_->IsOpen())
      return HSA_STATUS_ERROR_NOT_INITIALIZED;

    if (launch_descriptor == nullptr)
      return HSA_STATUS_ERROR_INVALID_ARGUMENT;

    const rocr::core::Agent* agent = rocr::core::Agent::Convert(agent_handle);
    if (agent == nullptr || !agent->IsValid())
      return HSA_STATUS_ERROR_INVALID_AGENT;

    if (agent->device_type() != rocr::core::Agent::kAmdGpuDevice)
      return HSA_STATUS_ERROR_INVALID_AGENT;

    uint8_t version = rocr::core::GetDescriptorVersion(agent);
    if (version == AMD_LAUNCH_DESCRIPTOR_VERSION_NONE)
      return HSA_STATUS_ERROR_INVALID_AGENT;

    void* data = rocr::core::CreateDescriptor(version);
    if (data == nullptr)
      return HSA_STATUS_ERROR_OUT_OF_RESOURCES;

    *launch_descriptor = rocr::core::ToDescriptorHandle(data);
    return HSA_STATUS_SUCCESS;
  } catch (const rocr::AMD::hsa_exception& ex) {
    return ex.error_code();
  } catch (...) {
    return HSA_STATUS_ERROR;
  }
}

hsa_status_t HSA_API hsa_ven_amd_launch_descriptor_destroy(
    amd_launch_descriptor_t launch_descriptor) {
  try {
    if (!rocr::core::Runtime::runtime_singleton_->IsOpen())
      return HSA_STATUS_ERROR_NOT_INITIALIZED;

    void* data = rocr::core::ConvertDescriptorHandle(launch_descriptor);
    rocr::core::DestroyDescriptor(data);
    return HSA_STATUS_SUCCESS;
  } catch (const rocr::AMD::hsa_exception& ex) {
    return ex.error_code();
  } catch (...) {
    return HSA_STATUS_ERROR;
  }
}

hsa_status_t HSA_API hsa_ven_amd_launch_descriptor_set(
    amd_launch_descriptor_t launch_descriptor,
    uint32_t field,
    uint64_t value) {
  try {
    if (!rocr::core::Runtime::runtime_singleton_->IsOpen())
      return HSA_STATUS_ERROR_NOT_INITIALIZED;

    void* data = rocr::core::ConvertDescriptorHandle(launch_descriptor);
    rocr::core::SetDescriptorField(data, field, value);
    return HSA_STATUS_SUCCESS;
  } catch (const rocr::AMD::hsa_exception& ex) {
    return ex.error_code();
  } catch (...) {
    return HSA_STATUS_ERROR;
  }
}

hsa_status_t HSA_API hsa_ven_amd_launch_descriptor_set_prefetch(
    amd_launch_descriptor_t launch_descriptor,
    uint32_t index,
    const amd_data_prefetch_t* prefetch) {
  try {
    if (!rocr::core::Runtime::runtime_singleton_->IsOpen())
      return HSA_STATUS_ERROR_NOT_INITIALIZED;

    void* data = rocr::core::ConvertDescriptorHandle(launch_descriptor);
    rocr::core::SetDescriptorPrefetch(data, index, prefetch);
    return HSA_STATUS_SUCCESS;
  } catch (const rocr::AMD::hsa_exception& ex) {
    return ex.error_code();
  } catch (...) {
    return HSA_STATUS_ERROR;
  }
}
