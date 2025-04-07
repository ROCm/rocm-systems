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

#if defined(ROCPROFILER_LIB_ROCPROFILER_OMPT_OMPT_CPP_IMPL) &&                                     \
    ROCPROFILER_LIB_ROCPROFILER_OMPT_OMPT_CPP_IMPL == 1

#    include "lib/common/mpl.hpp"
#    include "lib/rocprofiler-sdk/ompt/defines.hpp"
#    include "lib/rocprofiler-sdk/ompt/ompt.hpp"
#    include "lib/rocprofiler-sdk/ompt/utils.hpp"

#    include <rocprofiler-sdk/external_correlation.h>
#    include <rocprofiler-sdk/ompt.h>

#    include <type_traits>

// clang-format off
OMPT_INFO_DEFINITION_V("omp_thread_begin", ompt_callback_thread_begin, ROCPROFILER_OMPT_ID_thread_begin, ompt_callback_thread_begin_t, thread_begin, ompt_thread_begin_fn, thread_type, thread_data)
OMPT_INFO_DEFINITION_V("omp_thread_end", ompt_callback_thread_end, ROCPROFILER_OMPT_ID_thread_end, ompt_callback_thread_end_t, thread_end, ompt_thread_end_fn, thread_data)
OMPT_INFO_DEFINITION_V("omp_parallel_begin", ompt_callback_parallel_begin, ROCPROFILER_OMPT_ID_parallel_begin, ompt_callback_parallel_begin_t, parallel_begin, ompt_parallel_begin_fn, encountering_task_data, encountering_task_frame, parallel_data, requested_parallelism, flags, codeptr_ra)
OMPT_INFO_DEFINITION_V("omp_parallel_end", ompt_callback_parallel_end, ROCPROFILER_OMPT_ID_parallel_end, ompt_callback_parallel_end_t, parallel_end, ompt_parallel_end_fn, parallel_data, encountering_task_data, flags, codeptr_ra)
OMPT_INFO_DEFINITION_V("omp_task_create", ompt_callback_task_create, ROCPROFILER_OMPT_ID_task_create, ompt_callback_task_create_t, task_create, ompt_task_create_fn, encountering_task_data, encountering_task_frame, new_task_data, flags, has_dependences, codeptr_ra)
OMPT_INFO_DEFINITION_V("omp_task_schedule", ompt_callback_task_schedule, ROCPROFILER_OMPT_ID_task_schedule, ompt_callback_task_schedule_t, task_schedule, ompt_task_schedule_fn, prior_task_data, prior_task_status, next_task_data)
OMPT_INFO_DEFINITION_V("omp_implicit_task", ompt_callback_implicit_task, ROCPROFILER_OMPT_ID_implicit_task, ompt_callback_implicit_task_t, implicit_task, ompt_implicit_task_fn, endpoint, parallel_data, task_data, actual_parallelism, index, flags)
OMPT_INFO_DEFINITION_V("omp_device_initialize", ompt_callback_device_initialize, ROCPROFILER_OMPT_ID_device_initialize, ompt_callback_device_initialize_t, device_initialize, ompt_device_initialize_fn, device_num, type, device, lookup, documentation)
OMPT_INFO_DEFINITION_V("omp_device_finalize", ompt_callback_device_finalize, ROCPROFILER_OMPT_ID_device_finalize, ompt_callback_device_finalize_t, device_finalize, ompt_device_finalize_fn, device_num)
OMPT_INFO_DEFINITION_V("omp_device_load", ompt_callback_device_load, ROCPROFILER_OMPT_ID_device_load, ompt_callback_device_load_t, device_load, ompt_device_load_fn, device_num, filename, offset_in_file, vma_in_file, bytes, host_addr, device_addr, module_id)
// OMPT_INFO_DEFINITION_V("omp_device_unload", ompt_callback_device_unload, ROCPROFILER_OMPT_ID_device_unload, ompt_callback_device_unload_t, device_unload, ompt_device_unload_fn, device_num, module_id)
OMPT_INFO_DEFINITION_V("omp_sync_region_wait", ompt_callback_sync_region_wait, ROCPROFILER_OMPT_ID_sync_region_wait, ompt_callback_sync_region_wait_t, sync_region_wait, ompt_sync_region_wait_fn, kind, endpoint, parallel_data, task_data, codeptr_ra)
OMPT_INFO_DEFINITION_V("omp_mutex_released", ompt_callback_mutex_released, ROCPROFILER_OMPT_ID_mutex_released, ompt_callback_mutex_released_t, mutex_released, ompt_mutex_released_fn, kind, wait_id, codeptr_ra)
OMPT_INFO_DEFINITION_V("omp_dependences", ompt_callback_dependences, ROCPROFILER_OMPT_ID_dependences, ompt_callback_dependences_t, dependences, ompt_dependences_fn, task_data, deps, ndeps)
OMPT_INFO_DEFINITION_V("omp_task_dependence", ompt_callback_task_dependence, ROCPROFILER_OMPT_ID_task_dependence, ompt_callback_task_dependence_t, task_dependence, ompt_task_dependence_fn, src_task_data, sink_task_data)
OMPT_INFO_DEFINITION_V("omp_work", ompt_callback_work, ROCPROFILER_OMPT_ID_work, ompt_callback_work_t, work, ompt_work_fn, work_type, endpoint, parallel_data, task_data, count, codeptr_ra)
OMPT_INFO_DEFINITION_V("omp_masked", ompt_callback_masked, ROCPROFILER_OMPT_ID_masked, ompt_callback_masked_t, masked, ompt_masked_fn, endpoint, parallel_data, task_data, codeptr_ra)
OMPT_INFO_DEFINITION_V("omp_sync_region", ompt_callback_sync_region, ROCPROFILER_OMPT_ID_sync_region, ompt_callback_sync_region_t, sync_region, ompt_sync_region_fn, kind, endpoint, parallel_data, task_data, codeptr_ra)
OMPT_INFO_DEFINITION_V("omp_lock_init", ompt_callback_lock_init, ROCPROFILER_OMPT_ID_lock_init, ompt_callback_lock_init_t, lock_init, ompt_lock_init_fn, kind, hint, impl, wait_id, codeptr_ra)
OMPT_INFO_DEFINITION_V("omp_lock_destroy", ompt_callback_lock_destroy, ROCPROFILER_OMPT_ID_lock_destroy, ompt_callback_lock_destroy_t, lock_destroy, ompt_lock_destroy_fn, kind, wait_id, codeptr_ra)
OMPT_INFO_DEFINITION_V("omp_mutex_acquire", ompt_callback_mutex_acquire, ROCPROFILER_OMPT_ID_mutex_acquire, ompt_callback_mutex_acquire_t, mutex_acquire, ompt_mutex_acquire_fn, kind, hint, impl, wait_id, codeptr_ra)
OMPT_INFO_DEFINITION_V("omp_mutex_acquired", ompt_callback_mutex_acquired, ROCPROFILER_OMPT_ID_mutex_acquired, ompt_callback_mutex_acquired_t, mutex_acquired, ompt_mutex_acquired_fn, kind, wait_id, codeptr_ra)
OMPT_INFO_DEFINITION_V("omp_nest_lock", ompt_callback_nest_lock, ROCPROFILER_OMPT_ID_nest_lock, ompt_callback_nest_lock_t, nest_lock, ompt_nest_lock_fn, endpoint, wait_id, codeptr_ra)
OMPT_INFO_DEFINITION_V("omp_flush", ompt_callback_flush, ROCPROFILER_OMPT_ID_flush, ompt_callback_flush_t, flush, ompt_flush_fn, thread_data, codeptr_ra)
OMPT_INFO_DEFINITION_V("omp_cancel", ompt_callback_cancel, ROCPROFILER_OMPT_ID_cancel, ompt_callback_cancel_t, cancel, ompt_cancel_fn, task_data, flags, codeptr_ra)
OMPT_INFO_DEFINITION_V("omp_reduction", ompt_callback_reduction, ROCPROFILER_OMPT_ID_reduction, ompt_callback_reduction_t, reduction, ompt_reduction_fn, kind, endpoint, parallel_data, task_data, codeptr_ra)
OMPT_INFO_DEFINITION_V("omp_dispatch", ompt_callback_dispatch, ROCPROFILER_OMPT_ID_dispatch, ompt_callback_dispatch_t, dispatch, ompt_dispatch_fn, parallel_data, task_data, kind, instance)
OMPT_INFO_DEFINITION_V("omp_target_emi", ompt_callback_target_emi, ROCPROFILER_OMPT_ID_target_emi, ompt_callback_target_emi_t, target_emi, ompt_target_emi_fn, kind, endpoint, device_num, task_data, target_task_data, target_data, codeptr_ra)
OMPT_INFO_DEFINITION_V("omp_target_data_op_emi", ompt_callback_target_data_op_emi, ROCPROFILER_OMPT_ID_target_data_op_emi, ompt_callback_target_data_op_emi_t, target_data_op_emi, ompt_target_data_op_emi_fn, endpoint, target_task_data, target_data, host_op_id, optype, src_address, src_device_num, dst_address, dst_device_num, bytes, codeptr_ra)
OMPT_INFO_DEFINITION_V("omp_target_submit_emi", ompt_callback_target_submit_emi, ROCPROFILER_OMPT_ID_target_submit_emi, ompt_callback_target_submit_emi_t, target_submit_emi, ompt_target_submit_emi_fn, endpoint, target_data, host_op_id, requested_num_teams)
// OMPT_INFO_DEFINITION_V("omp_target_map_emi", ompt_callback_target_map_emi, ROCPROFILER_OMPT_ID_target_map_emi, ompt_callback_target_map_emi_t, target_map_emi, ompt_target_map_emi_fn, nitems, host_addr, device_addr, bytes, mapping_flags, codeptr_ra)
OMPT_INFO_DEFINITION_V("omp_error", ompt_callback_error, ROCPROFILER_OMPT_ID_error, ompt_callback_error_t, error, ompt_error_fn, severity, message, length, codeptr_ra)
// clang-format on

#else
#    error "Do not compile this file directly. It is included by lib/rocprofiler-sdk/ompt/ompt.cpp"
#endif
