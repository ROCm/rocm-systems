// MIT License
//
// Copyright (c) 2023-2025 Advanced Micro Devices, Inc. All rights reserved.
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
#include <rocprofiler-sdk/fwd.h>
#include <rocprofiler-sdk/hsa.h>
#include <rocprofiler-sdk/hsa/api_trace_version.h>

#include <stdint.h>

ROCPROFILER_EXTERN_C_INIT

/**
 * @brief Page migration triggers
 *
 */
typedef enum rocprofiler_page_migration_trigger_t
{
    ROCPROFILER_PAGE_MIGRATION_TRIGGER_NONE = -1,
    ROCPROFILER_PAGE_MIGRATION_TRIGGER_PREFETCH,       ///< Migration triggered by a prefetch
    ROCPROFILER_PAGE_MIGRATION_TRIGGER_PAGEFAULT_GPU,  ///< Triggered by a page fault on the GPU
    ROCPROFILER_PAGE_MIGRATION_TRIGGER_PAGEFAULT_CPU,  ///< Triggered by a page fault on the CPU
    ROCPROFILER_PAGE_MIGRATION_TRIGGER_TTM_EVICTION,   ///< Page evicted by linux TTM (Translation
                                                       ///< Table Manager)
    ROCPROFILER_PAGE_MIGRATION_TRIGGER_LAST,
} rocprofiler_page_migration_trigger_t;

/**
 * @brief Page migration triggers causing the queue to suspend
 *
 */
typedef enum rocprofiler_page_migration_queue_suspend_trigger_t
{
    ROCPROFILER_PAGE_MIGRATION_QUEUE_SUSPEND_TRIGGER_NONE = -1,
    ROCPROFILER_PAGE_MIGRATION_QUEUE_SUSPEND_TRIGGER_SVM,
    ROCPROFILER_PAGE_MIGRATION_QUEUE_SUSPEND_TRIGGER_USERPTR,
    ROCPROFILER_PAGE_MIGRATION_QUEUE_SUSPEND_TRIGGER_TTM,  ///< Queue suspended by TTM (Translation
                                                           ///< Table Manager) operation
    ROCPROFILER_PAGE_MIGRATION_QUEUE_SUSPEND_TRIGGER_SUSPEND,
    ROCPROFILER_PAGE_MIGRATION_QUEUE_SUSPEND_TRIGGER_CRIU_CHECKPOINT,  ///< Queues evicted due to
                                                                       ///< process save
                                                                       ///< (checkpoint) by CRIU
    ROCPROFILER_PAGE_MIGRATION_QUEUE_SUSPEND_TRIGGER_CRIU_RESTORE,     ///< Queues restored during
                                                                       ///< process restore by CRIU
    ROCPROFILER_PAGE_MIGRATION_QUEUE_SUSPEND_TRIGGER_LAST,
} rocprofiler_page_migration_queue_suspend_trigger_t;

/**
 * @brief Page migration triggers causing an unmap from the GPU
 *
 */
typedef enum rocprofiler_page_migration_unmap_from_gpu_trigger_t
{
    ROCPROFILER_PAGE_MIGRATION_UNMAP_FROM_GPU_TRIGGER_NONE = -1,
    ROCPROFILER_PAGE_MIGRATION_UNMAP_FROM_GPU_TRIGGER_MMU_NOTIFY,
    ROCPROFILER_PAGE_MIGRATION_UNMAP_FROM_GPU_TRIGGER_MMU_NOTIFY_MIGRATE,
    ROCPROFILER_PAGE_MIGRATION_UNMAP_FROM_GPU_TRIGGER_UNMAP_FROM_CPU,
    ROCPROFILER_PAGE_MIGRATION_UNMAP_FROM_GPU_TRIGGER_LAST,
} rocprofiler_page_migration_unmap_from_gpu_trigger_t;

ROCPROFILER_EXTERN_C_FINI
