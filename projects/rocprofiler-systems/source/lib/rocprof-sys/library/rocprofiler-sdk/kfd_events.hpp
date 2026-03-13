// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier:  MIT

#pragma once

#include <rocprofiler-sdk/buffer_tracing.h>
#include <rocprofiler-sdk/fwd.h>

#include <cstdint>

namespace rocprofsys
{
namespace rocprofiler_sdk
{
void
kfd_event_metadata_initialize();

void
tool_kfd_page_fault_callback(
    const rocprofiler_buffer_tracing_kfd_page_fault_record_t* record);

void
tool_kfd_page_migrate_callback(
    const rocprofiler_buffer_tracing_kfd_page_migrate_record_t* record);

void
tool_kfd_queue_callback(const rocprofiler_buffer_tracing_kfd_queue_record_t* record);

void
tool_kfd_event_queue_callback(
    const rocprofiler_buffer_tracing_kfd_event_queue_record_t* record);

void
tool_kfd_event_unmap_from_gpu_callback(
    const rocprofiler_buffer_tracing_kfd_event_unmap_from_gpu_record_t* record);

void
tool_kfd_event_dropped_events_callback(
    const rocprofiler_buffer_tracing_kfd_event_dropped_events_record_t* record);

}  // namespace rocprofiler_sdk

}  // namespace rocprofsys
