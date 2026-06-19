// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

#include "library/rocprofiler-sdk/fwd.hpp"

#include <cstdint>

#if ROCPROFILER_VERSION >= 10000

namespace rocprofsys
{
namespace rocprofiler_sdk
{

void
kfd_event_metadata_initialize(const client_data_t* tool_data);

void
tool_kfd_page_fault_callback(const client_data_t*                  tool_data,
                             const backend::kfd_page_fault_record* record);

void
tool_kfd_page_migrate_callback(const client_data_t*                    tool_data,
                               const backend::kfd_page_migrate_record* record);

void
tool_kfd_queue_callback(const client_data_t*             tool_data,
                        const backend::kfd_queue_record* record);

void
tool_kfd_event_queue_callback(const client_data_t*                   tool_data,
                              const backend::kfd_event_queue_record* record);

void
tool_kfd_event_unmap_from_gpu_callback(const client_data_t*                   tool_data,
                                       const backend::kfd_event_unmap_record* record);

void
tool_kfd_event_dropped_events_callback(const client_data_t*                     tool_data,
                                       const backend::kfd_event_dropped_record* record);

}  // namespace rocprofiler_sdk

}  // namespace rocprofsys
#endif
