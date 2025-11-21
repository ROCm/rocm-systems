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
#include <rocprofiler-sdk/hsa.h>
#include <rocprofiler-sdk/hsa/api_trace_version.h>
#include <rocprofiler-sdk/version.h>

#include <stdint.h>

ROCPROFILER_EXTERN_C_INIT

/**
 * @brief Page migration event operations. @see
 * rocprofiler_buffer_tracing_kfd_event_page_migrate_record_t
 */
typedef enum rocprofiler_kfd_event_page_migrate_operation_t
{
    ROCPROFILER_KFD_EVENT_PAGE_MIGRATE_NONE = -1,
    ROCPROFILER_KFD_EVENT_PAGE_MIGRATE_PREFETCH,       ///< Migration triggered by a prefetch
    ROCPROFILER_KFD_EVENT_PAGE_MIGRATE_PAGEFAULT_GPU,  ///< Migration triggered by a page fault on
                                                       ///< the GPU
    ROCPROFILER_KFD_EVENT_PAGE_MIGRATE_PAGEFAULT_CPU,  ///< Migration triggered by a page fault on
                                                       ///< the CPU
    ROCPROFILER_KFD_EVENT_PAGE_MIGRATE_TTM_EVICTION,   ///< Page evicted by linux TTM (Translation
                                                       ///< Table Manager)
    ROCPROFILER_KFD_EVENT_PAGE_MIGRATE_END,
    ROCPROFILER_KFD_EVENT_PAGE_MIGRATE_LAST,
} rocprofiler_kfd_event_page_migrate_operation_t;

/**
 * @brief Page fault event operations. @see rocprofiler_buffer_tracing_kfd_event_page_fault_record_t
 */
typedef enum rocprofiler_kfd_event_page_fault_operation_t
{
    ROCPROFILER_KFD_EVENT_PAGE_FAULT_NONE = -1,
    ROCPROFILER_KFD_EVENT_PAGE_FAULT_START,
    ROCPROFILER_KFD_EVENT_PAGE_FAULT_START_READ_FAULT,   ///< Page fault was due to a read operation
    ROCPROFILER_KFD_EVENT_PAGE_FAULT_START_WRITE_FAULT,  ///< Page fault was due to a write
                                                         ///< operation
    ROCPROFILER_KFD_EVENT_PAGE_FAULT_END_PAGE_MIGRATED,  ///< Fault resolved through a migration
    ROCPROFILER_KFD_EVENT_PAGE_FAULT_END_PAGE_UPDATED,   ///< Fault resolved through an update
    ROCPROFILER_KFD_EVENT_PAGE_FAULT_LAST,
} rocprofiler_kfd_event_page_fault_operation_t;

/**
 * @brief Queue evict/restore event operations. @see
 * rocprofiler_buffer_tracing_kfd_event_queue_record_t
 */
typedef enum rocprofiler_kfd_event_queue_operation_t
{
    ROCPROFILER_KFD_EVENT_QUEUE_NONE = -1,
    ROCPROFILER_KFD_EVENT_QUEUE_EVICT_SVM,              ///< SVM Buffer migration
    ROCPROFILER_KFD_EVENT_QUEUE_EVICT_USERPTR,          ///< userptr movement
    ROCPROFILER_KFD_EVENT_QUEUE_EVICT_TTM,              ///< TTM move buffer
    ROCPROFILER_KFD_EVENT_QUEUE_EVICT_SUSPEND,          ///< GPU suspend
    ROCPROFILER_KFD_EVENT_QUEUE_EVICT_CRIU_CHECKPOINT,  ///< Queues evicted due to process
                                                        ///< checkpoint by CRIU
    ROCPROFILER_KFD_EVENT_QUEUE_EVICT_CRIU_RESTORE,  ///< Queues restored during process restore by
                                                     ///< CRIU

    ROCPROFILER_KFD_EVENT_QUEUE_RESTORE_RESCHEDULED,  ///< Queue was not restored; will be restored
                                                      ///< later
    ROCPROFILER_KFD_EVENT_QUEUE_RESTORE,              ///< Queue was restored
    ROCPROFILER_KFD_EVENT_QUEUE_LAST,
} rocprofiler_kfd_event_queue_operation_t;

/**
 * @brief Memory unmap from GPU event operations. @see
 * rocprofiler_buffer_tracing_kfd_event_unmap_from_gpu_record_t
 */
typedef enum rocprofiler_kfd_event_unmap_from_gpu_operation_t
{
    ROCPROFILER_KFD_EVENT_UNMAP_FROM_GPU_NONE = -1,
    ROCPROFILER_KFD_EVENT_UNMAP_FROM_GPU_MMU_NOTIFY,          ///< MMU notifier CPU buffer movement
    ROCPROFILER_KFD_EVENT_UNMAP_FROM_GPU_MMU_NOTIFY_MIGRATE,  ///< MMU notifier page migration
    ROCPROFILER_KFD_EVENT_UNMAP_FROM_GPU_UNMAP_FROM_CPU,      ///< Unmap to free the buffer
    ROCPROFILER_KFD_EVENT_UNMAP_FROM_GPU_LAST,
} rocprofiler_kfd_event_unmap_from_gpu_operation_t;

/**
 * @brief KFD dropped event operations. @see
 * rocprofiler_buffer_tracing_kfd_event_dropped_events_record_t
 */
typedef enum rocprofiler_kfd_event_dropped_events_operation_t
{
    ROCPROFILER_KFD_EVENT_DROPPED_EVENTS_NONE = -1,
    ROCPROFILER_KFD_EVENT_DROPPED_EVENTS,
    ROCPROFILER_KFD_EVENT_DROPPED_EVENTS_LAST,
} rocprofiler_kfd_event_dropped_events_operation_t;

/**
 * @brief Operation kinds for @see rocprofiler_buffer_tracing_kfd_page_migrate_record_t
 */
typedef enum rocprofiler_kfd_page_migrate_operation_t
{
    ROCPROFILER_KFD_PAGE_MIGRATE_NONE = -1,
    ROCPROFILER_KFD_PAGE_MIGRATE_PREFETCH,       ///< Migration triggered by a prefetch
    ROCPROFILER_KFD_PAGE_MIGRATE_PAGEFAULT_GPU,  ///< Migration triggered by a page fault on the
                                                 ///< GPU
    ROCPROFILER_KFD_PAGE_MIGRATE_PAGEFAULT_CPU,  ///< Migration triggered by a page fault on the
                                                 ///< CPU
    ROCPROFILER_KFD_PAGE_MIGRATE_TTM_EVICTION,   ///< Page evicted by linux TTM (Translation Table
                                                 ///< Manager)
    ROCPROFILER_KFD_PAGE_MIGRATE_LAST,
} rocprofiler_kfd_page_migrate_operation_t;

/**
 * @brief Operation kinds for @see rocprofiler_buffer_tracing_kfd_page_fault_record_t
 */
typedef enum rocprofiler_kfd_page_fault_operation_t
{
    ROCPROFILER_KFD_PAGE_FAULT_NONE = -1,
    ROCPROFILER_KFD_PAGE_FAULT_READ_FAULT_MIGRATED,   ///< read fault resolved with a migrate
    ROCPROFILER_KFD_PAGE_FAULT_READ_FAULT_UPDATED,    ///< read fault resolved with an update
    ROCPROFILER_KFD_PAGE_FAULT_WRITE_FAULT_MIGRATED,  ///< write fault resolved with an migrate
    ROCPROFILER_KFD_PAGE_FAULT_WRITE_FAULT_UPDATED,   ///< write fault resolved with an update
    ROCPROFILER_KFD_PAGE_FAULT_LAST,
} rocprofiler_kfd_page_fault_operation_t;

/**
 * @brief Operation kinds for @see rocprofiler_buffer_tracing_kfd_queue_record_t
 */
typedef enum rocprofiler_kfd_queue_operation_t
{
    ROCPROFILER_KFD_QUEUE_NONE = -1,
    ROCPROFILER_KFD_QUEUE_EVICT_SVM,              ///< SVM Buffer migration
    ROCPROFILER_KFD_QUEUE_EVICT_USERPTR,          ///< userptr movement
    ROCPROFILER_KFD_QUEUE_EVICT_TTM,              ///< TTM move buffer
    ROCPROFILER_KFD_QUEUE_EVICT_SUSPEND,          ///< GPU suspend
    ROCPROFILER_KFD_QUEUE_EVICT_CRIU_CHECKPOINT,  ///< Queues evicted due to process checkpoint by
                                                  ///< CRIU
    ROCPROFILER_KFD_QUEUE_EVICT_CRIU_RESTORE,  ///< Queues restored during process restore by CRIU

    ROCPROFILER_KFD_QUEUE_LAST,
} rocprofiler_kfd_queue_operation_t;

ROCPROFILER_EXTERN_C_FINI
