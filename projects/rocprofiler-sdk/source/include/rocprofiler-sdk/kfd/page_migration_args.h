// MIT License
//
// Copyright (c) 2023 Advanced Micro Devices, Inc. All rights reserved.
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
// THE SOFTWARE.

#pragma once

#include <rocprofiler-sdk/defines.h>
#include <rocprofiler-sdk/kfd/page_migration_id.h>

#include <stdint.h>

ROCPROFILER_EXTERN_C_INIT

typedef struct rocprofiler_page_migration_none_t
{
    char empty;
} rocprofiler_page_migration_none_t;

typedef struct rocprofiler_page_migration_page_migrate_start_t
{
    uint64_t                             start_addr;
    uint64_t                             end_addr;
    rocprofiler_agent_id_t               from_agent;
    rocprofiler_agent_id_t               to_agent;
    rocprofiler_agent_id_t               prefetch_agent;
    rocprofiler_agent_id_t               preferred_agent;
    rocprofiler_page_migration_trigger_t trigger;
} rocprofiler_page_migration_page_migrate_start_t;

typedef struct rocprofiler_page_migration_page_migrate_end_t
{
    uint64_t                             start_addr;
    uint64_t                             end_addr;
    rocprofiler_agent_id_t               from_agent;
    rocprofiler_agent_id_t               to_agent;
    rocprofiler_page_migration_trigger_t trigger;
    int32_t                              error_code;
} rocprofiler_page_migration_page_migrate_end_t;

typedef struct rocprofiler_page_migration_page_fault_start_t
{
    uint32_t               read_fault : 1;
    rocprofiler_agent_id_t agent_id;
    uint64_t               address;
} rocprofiler_page_migration_page_fault_start_t;

typedef struct rocprofiler_page_migration_page_fault_end_t
{
    uint32_t               migrated : 1;
    rocprofiler_agent_id_t agent_id;
    uint64_t               address;
} rocprofiler_page_migration_page_fault_end_t;

typedef struct rocprofiler_page_migration_queue_eviction_t
{
    rocprofiler_agent_id_t                             agent_id;
    rocprofiler_page_migration_queue_suspend_trigger_t trigger;
} rocprofiler_page_migration_queue_eviction_t;

typedef struct rocprofiler_page_migration_queue_restore_t
{
    uint32_t               rescheduled : 1;
    rocprofiler_agent_id_t agent_id;
} rocprofiler_page_migration_queue_restore_t;

typedef struct rocprofiler_page_migration_unmap_from_gpu_t
{
    uint64_t                                            start_addr;
    uint64_t                                            end_addr;
    rocprofiler_agent_id_t                              agent_id;
    rocprofiler_page_migration_unmap_from_gpu_trigger_t trigger;
} rocprofiler_page_migration_unmap_from_gpu_t;

typedef struct rocprofiler_page_migration_dropped_event_t
{
    uint32_t dropped_events_count;
} rocprofiler_page_migration_dropped_event_t;

typedef union
{
    rocprofiler_page_migration_none_t               none;
    rocprofiler_page_migration_page_migrate_start_t page_migrate_start;
    rocprofiler_page_migration_page_migrate_end_t   page_migrate_end;
    rocprofiler_page_migration_page_fault_start_t   page_fault_start;
    rocprofiler_page_migration_page_fault_end_t     page_fault_end;
    rocprofiler_page_migration_queue_eviction_t     queue_eviction;
    rocprofiler_page_migration_queue_restore_t      queue_restore;
    rocprofiler_page_migration_unmap_from_gpu_t     unmap_from_gpu;
    rocprofiler_page_migration_dropped_event_t      dropped_event;
    uint64_t                                        reserved[16];
} rocprofiler_page_migration_args_t;

ROCPROFILER_EXTERN_C_FINI
