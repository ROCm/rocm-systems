////////////////////////////////////////////////////////////////////////////////
//
// The University of Illinois/NCSA
// Open Source License (NCSA)
//
// Copyright (c) 2014-2025, Advanced Micro Devices, Inc. All rights reserved.
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

#include "inc/hsa_api_trace.h"
#include "core/inc/hsa_api_trace_int.h"

#include "lttng/rocm_trace_emit.h"

/* Typed exit macros (mirror of HIP's hip_table_interface.cpp). Each
 * wrapper body emits the enter tracepoint as its first statement; these
 * macros expand at the return site and emit the matching exit tracepoint
 * based on the wrapper's return type. */
#define ROCR_TRACE_API_RET_STATUS(EXPR)                                                            \
  do {                                                                                             \
    const auto __rocm_rv = (EXPR);                                                                 \
    rocm_trace_emit_hsa_api_exit_status(__func__, static_cast<int32_t>(__rocm_rv));                \
    return __rocm_rv;                                                                              \
  } while (0)

#define ROCR_TRACE_API_RET_PTR(EXPR)                                                               \
  do {                                                                                             \
    auto* __rocm_rv = (EXPR);                                                                      \
    rocm_trace_emit_hsa_api_exit_ptr(__func__, __rocm_rv);                                         \
    return __rocm_rv;                                                                              \
  } while (0)

#define ROCR_TRACE_API_RET_VOID(EXPR)                                                              \
  do {                                                                                             \
    (EXPR);                                                                                        \
    rocm_trace_emit_hsa_api_exit_void(__func__);                                                   \
    return;                                                                                        \
  } while (0)

/* Wider integer returns from queue/signal hot-path ops. _U64 covers
 * uint64_t (queue index ops); _I64 covers hsa_signal_value_t / int64_t
 * (signal value ops). Full 64-bit width is preserved on the wire (vs.
 * truncation to int32 by the older _STATUS path). */
#define ROCR_TRACE_API_RET_U64(EXPR)                                                               \
  do {                                                                                             \
    const uint64_t __rocm_rv = static_cast<uint64_t>(EXPR);                                        \
    rocm_trace_emit_hsa_api_exit_u64(__func__, __rocm_rv);                                         \
    return __rocm_rv;                                                                              \
  } while (0)

#define ROCR_TRACE_API_RET_I64(EXPR)                                                               \
  do {                                                                                             \
    const int64_t __rocm_rv = static_cast<int64_t>(EXPR);                                          \
    rocm_trace_emit_hsa_api_exit_i64(__func__, __rocm_rv);                                         \
    return __rocm_rv;                                                                              \
  } while (0)

/* ---------- HSA curated parameter-capture variants ----------
 * Six macros: STATUS / PTR / VOID returns, each with a captured-args form
 * and a _NOARGS form (used for curated APIs with zero captured args).
 * Every typed helper takes (<captured-args...>, hsa_status_t status) --
 * status comes from the call's return expression (STATUS), is synthesized
 * from null-vs-non-null retval (PTR; rare on HSA), or is literal
 * HSA_STATUS_SUCCESS (VOID).
 *
 * The typed _args event AUGMENTS the generic exit event
 * (hsa_api_exit_status / _ptr / _void); it never replaces it.
 */

/* Captured-args variants. __VA_ARGS__ is non-empty by construction (the
 * _NOARGS variants are used for zero-arg APIs). */
#define ROCR_TRACE_API_RET_STATUS_CURATED_HSA(api, expr, ...)                                      \
  do {                                                                                             \
    const hsa_status_t __rocm_status = (expr);                                                     \
    rocm_trace_emit_##api##_args(__VA_ARGS__, __rocm_status);                                      \
    rocm_trace_emit_hsa_api_exit_status(__func__, static_cast<int32_t>(__rocm_status));            \
    return __rocm_status;                                                                          \
  } while (0)

#define ROCR_TRACE_API_RET_PTR_CURATED_HSA(api, ptr_type, expr, ...)                               \
  do {                                                                                             \
    ptr_type const __rocm_ptr = (expr);                                                            \
    const hsa_status_t __rocm_status =                                                             \
        (__rocm_ptr != nullptr) ? HSA_STATUS_SUCCESS : HSA_STATUS_ERROR_OUT_OF_RESOURCES;          \
    rocm_trace_emit_##api##_args(__VA_ARGS__, __rocm_status);                                      \
    rocm_trace_emit_hsa_api_exit_ptr(__func__, __rocm_ptr);                                        \
    return __rocm_ptr;                                                                             \
  } while (0)

#define ROCR_TRACE_API_RET_VOID_CURATED_HSA(api, expr, ...)                                        \
  do {                                                                                             \
    (expr);                                                                                        \
    rocm_trace_emit_##api##_args(__VA_ARGS__, HSA_STATUS_SUCCESS);                                 \
    rocm_trace_emit_hsa_api_exit_void(__func__);                                                   \
    return;                                                                                        \
  } while (0)

/* Zero-captured-args variants. Separate macros to avoid empty
 * __VA_ARGS__ expansion in the captured-args macros above. */
#define ROCR_TRACE_API_RET_STATUS_CURATED_HSA_NOARGS(api, expr)                                    \
  do {                                                                                             \
    const hsa_status_t __rocm_status = (expr);                                                     \
    rocm_trace_emit_##api##_args(__rocm_status);                                                   \
    rocm_trace_emit_hsa_api_exit_status(__func__, static_cast<int32_t>(__rocm_status));            \
    return __rocm_status;                                                                          \
  } while (0)

#define ROCR_TRACE_API_RET_PTR_CURATED_HSA_NOARGS(api, ptr_type, expr)                             \
  do {                                                                                             \
    ptr_type const __rocm_ptr = (expr);                                                            \
    const hsa_status_t __rocm_status =                                                             \
        (__rocm_ptr != nullptr) ? HSA_STATUS_SUCCESS : HSA_STATUS_ERROR_OUT_OF_RESOURCES;          \
    rocm_trace_emit_##api##_args(__rocm_status);                                                   \
    rocm_trace_emit_hsa_api_exit_ptr(__func__, __rocm_ptr);                                        \
    return __rocm_ptr;                                                                             \
  } while (0)

#define ROCR_TRACE_API_RET_VOID_CURATED_HSA_NOARGS(api, expr)                                      \
  do {                                                                                             \
    (expr);                                                                                        \
    rocm_trace_emit_##api##_args(HSA_STATUS_SUCCESS);                                              \
    rocm_trace_emit_hsa_api_exit_void(__func__);                                                   \
    return;                                                                                        \
  } while (0)

static const HsaApiTable* hsaApiTable;
static const CoreApiTable* coreApiTable;
static const AmdExtTable* amdExtTable;
static const ToolsApiTable* toolsApiTable;

void hsa_table_interface_init(const HsaApiTable* apiTable) {
  rocm_trace_emit_hsa_api_enter(__func__);
  hsaApiTable = apiTable;
  coreApiTable = apiTable->core_;
  amdExtTable = apiTable->amd_ext_;
  toolsApiTable = apiTable->tools_;
  rocm_trace_emit_hsa_api_exit_void(__func__);
}

const HsaApiTable* hsa_table_interface_get_table() {
  rocm_trace_emit_hsa_api_enter(__func__);
  ROCR_TRACE_API_RET_PTR(hsaApiTable);
}

// Pass through stub functions
hsa_status_t HSA_API hsa_init() {
  rocm_trace_emit_hsa_api_enter(__func__);

  // We initialize the api tables here once more since the code above is prone to a
  // link-time ordering condition: This compilation unit here may get its global
  // variables initialized earlier than the global objects in other compilation units.
  // In particular Init::Init may get called earlier than that the underlying hsa_api_table_
  // object in hsa_api_trace.cpp has been initialized.
  rocr::core::LoadInitialHsaApiTable();
  ROCR_TRACE_API_RET_STATUS(coreApiTable->hsa_init_fn());
}

hsa_status_t HSA_API hsa_shut_down() {
  rocm_trace_emit_hsa_api_enter(__func__);
  ROCR_TRACE_API_RET_STATUS(coreApiTable->hsa_shut_down_fn());
}

hsa_status_t HSA_API hsa_system_get_info(hsa_system_info_t attribute, void* value) {
  rocm_trace_emit_hsa_api_enter(__func__);
  ROCR_TRACE_API_RET_STATUS(coreApiTable->hsa_system_get_info_fn(attribute, value));
}

hsa_status_t HSA_API hsa_extension_get_name(uint16_t extension, const char** name) {
  rocm_trace_emit_hsa_api_enter(__func__);
  ROCR_TRACE_API_RET_STATUS(coreApiTable->hsa_extension_get_name_fn(extension, name));
}

hsa_status_t HSA_API hsa_system_extension_supported(uint16_t extension, uint16_t version_major,
                                                    uint16_t version_minor, bool* result) {
  rocm_trace_emit_hsa_api_enter(__func__);
  ROCR_TRACE_API_RET_STATUS(coreApiTable->hsa_system_extension_supported_fn(
      extension, version_major, version_minor, result));
}

hsa_status_t HSA_API hsa_system_major_extension_supported(uint16_t extension,
                                                          uint16_t version_major,
                                                          uint16_t* version_minor, bool* result) {
  rocm_trace_emit_hsa_api_enter(__func__);
  ROCR_TRACE_API_RET_STATUS(coreApiTable->hsa_system_major_extension_supported_fn(
      extension, version_major, version_minor, result));
}

hsa_status_t HSA_API hsa_system_get_extension_table(uint16_t extension, uint16_t version_major,
                                                    uint16_t version_minor, void* table) {
  rocm_trace_emit_hsa_api_enter(__func__);
  ROCR_TRACE_API_RET_STATUS(coreApiTable->hsa_system_get_extension_table_fn(
      extension, version_major, version_minor, table));
}

hsa_status_t HSA_API hsa_system_get_major_extension_table(uint16_t extension,
                                                          uint16_t version_major,
                                                          size_t table_length, void* table) {
  rocm_trace_emit_hsa_api_enter(__func__);
  ROCR_TRACE_API_RET_STATUS(coreApiTable->hsa_system_get_major_extension_table_fn(
      extension, version_major, table_length, table));
}

hsa_status_t HSA_API hsa_iterate_agents(hsa_status_t (*callback)(hsa_agent_t agent, void* data),
                                        void* data) {
  rocm_trace_emit_hsa_api_enter(__func__);
  ROCR_TRACE_API_RET_STATUS(coreApiTable->hsa_iterate_agents_fn(callback, data));
}

hsa_status_t HSA_API hsa_agent_get_info(hsa_agent_t agent, hsa_agent_info_t attribute,
                                        void* value) {
  rocm_trace_emit_hsa_api_enter(__func__);
  ROCR_TRACE_API_RET_STATUS(coreApiTable->hsa_agent_get_info_fn(agent, attribute, value));
}

hsa_status_t HSA_API hsa_agent_get_exception_policies(hsa_agent_t agent, hsa_profile_t profile,
                                                      uint16_t* mask) {
  rocm_trace_emit_hsa_api_enter(__func__);
  ROCR_TRACE_API_RET_STATUS(
      coreApiTable->hsa_agent_get_exception_policies_fn(agent, profile, mask));
}

hsa_status_t HSA_API hsa_cache_get_info(hsa_cache_t cache, hsa_cache_info_t attribute,
                                        void* value) {
  rocm_trace_emit_hsa_api_enter(__func__);
  ROCR_TRACE_API_RET_STATUS(coreApiTable->hsa_cache_get_info_fn(cache, attribute, value));
}

hsa_status_t HSA_API hsa_agent_iterate_caches(
    hsa_agent_t agent, hsa_status_t (*callback)(hsa_cache_t cache, void* data), void* value) {
  rocm_trace_emit_hsa_api_enter(__func__);
  ROCR_TRACE_API_RET_STATUS(coreApiTable->hsa_agent_iterate_caches_fn(agent, callback, value));
}

hsa_status_t HSA_API hsa_agent_extension_supported(uint16_t extension, hsa_agent_t agent,
                                                   uint16_t version_major, uint16_t version_minor,
                                                   bool* result) {
  rocm_trace_emit_hsa_api_enter(__func__);
  ROCR_TRACE_API_RET_STATUS(coreApiTable->hsa_agent_extension_supported_fn(
      extension, agent, version_major, version_minor, result));
}

hsa_status_t HSA_API hsa_agent_major_extension_supported(uint16_t extension, hsa_agent_t agent,
                                                         uint16_t version_major,
                                                         uint16_t* version_minor, bool* result) {
  rocm_trace_emit_hsa_api_enter(__func__);
  ROCR_TRACE_API_RET_STATUS(coreApiTable->hsa_agent_major_extension_supported_fn(
      extension, agent, version_major, version_minor, result));
}

hsa_status_t HSA_API hsa_queue_create(hsa_agent_t agent, uint32_t size, hsa_queue_type32_t type,
                                      void (*callback)(hsa_status_t status, hsa_queue_t* source,
                                                       void* data),
                                      void* data, uint32_t private_segment_size,
                                      uint32_t group_segment_size, hsa_queue_t** queue) {
  rocm_trace_emit_hsa_api_enter(__func__); /* __ROCM_CURATED__: hsa_queue_create */
  auto const __rocm_in_agent = agent;
  auto const __rocm_in_size = size;
  auto const __rocm_in_type = type;
  auto const __rocm_in_callback = callback;
  auto const __rocm_in_data = data;
  auto const __rocm_in_private_segment_size = private_segment_size;
  auto const __rocm_in_group_segment_size = group_segment_size;
  ROCR_TRACE_API_RET_STATUS_CURATED_HSA(
      hsa_queue_create,
      coreApiTable->hsa_queue_create_fn(agent, size, type, callback, data, private_segment_size,
                                        group_segment_size, queue),
      (uint64_t)((__rocm_in_agent).handle), (__rocm_in_size), (__rocm_in_type),
      (const void*)(uintptr_t)(__rocm_in_callback), (const void*)(uintptr_t)(__rocm_in_data),
      (__rocm_in_private_segment_size), (__rocm_in_group_segment_size), queue);
}

hsa_status_t HSA_API hsa_soft_queue_create(hsa_region_t region, uint32_t size,
                                           hsa_queue_type32_t type, uint32_t features,
                                           hsa_signal_t completion_signal, hsa_queue_t** queue) {
  rocm_trace_emit_hsa_api_enter(__func__);
  ROCR_TRACE_API_RET_STATUS(coreApiTable->hsa_soft_queue_create_fn(region, size, type, features,
                                                                   completion_signal, queue));
}

hsa_status_t HSA_API hsa_queue_destroy(hsa_queue_t* queue) {
  rocm_trace_emit_hsa_api_enter(__func__); /* __ROCM_CURATED__: hsa_queue_destroy */
  auto const __rocm_in_queue = queue;
  ROCR_TRACE_API_RET_STATUS_CURATED_HSA(hsa_queue_destroy,
                                        coreApiTable->hsa_queue_destroy_fn(queue),
                                        (uint64_t)(uintptr_t)(__rocm_in_queue));
}

hsa_status_t HSA_API hsa_queue_inactivate(hsa_queue_t* queue) {
  rocm_trace_emit_hsa_api_enter(__func__);
  ROCR_TRACE_API_RET_STATUS(coreApiTable->hsa_queue_inactivate_fn(queue));
}

uint64_t HSA_API hsa_queue_load_read_index_scacquire(const hsa_queue_t* queue) {
  rocm_trace_emit_hsa_api_enter(__func__);
  ROCR_TRACE_API_RET_U64(coreApiTable->hsa_queue_load_read_index_scacquire_fn(queue));
}

uint64_t HSA_API hsa_queue_load_read_index_relaxed(const hsa_queue_t* queue) {
  rocm_trace_emit_hsa_api_enter(__func__);
  ROCR_TRACE_API_RET_U64(coreApiTable->hsa_queue_load_read_index_relaxed_fn(queue));
}

uint64_t HSA_API hsa_queue_load_write_index_scacquire(const hsa_queue_t* queue) {
  rocm_trace_emit_hsa_api_enter(__func__);
  ROCR_TRACE_API_RET_U64(coreApiTable->hsa_queue_load_write_index_scacquire_fn(queue));
}

uint64_t HSA_API hsa_queue_load_write_index_relaxed(const hsa_queue_t* queue) {
  rocm_trace_emit_hsa_api_enter(__func__);
  ROCR_TRACE_API_RET_U64(coreApiTable->hsa_queue_load_write_index_relaxed_fn(queue));
}

void HSA_API hsa_queue_store_write_index_relaxed(const hsa_queue_t* queue, uint64_t value) {
  rocm_trace_emit_hsa_api_enter(__func__);
  ROCR_TRACE_API_RET_VOID(coreApiTable->hsa_queue_store_write_index_relaxed_fn(queue, value));
}

void HSA_API hsa_queue_store_write_index_screlease(const hsa_queue_t* queue, uint64_t value) {
  rocm_trace_emit_hsa_api_enter(__func__);
  ROCR_TRACE_API_RET_VOID(coreApiTable->hsa_queue_store_write_index_screlease_fn(queue, value));
}

uint64_t HSA_API hsa_queue_cas_write_index_scacq_screl(const hsa_queue_t* queue, uint64_t expected,
                                                       uint64_t value) {
  rocm_trace_emit_hsa_api_enter(__func__);
  ROCR_TRACE_API_RET_U64(
      coreApiTable->hsa_queue_cas_write_index_scacq_screl_fn(queue, expected, value));
}

uint64_t HSA_API hsa_queue_cas_write_index_scacquire(const hsa_queue_t* queue, uint64_t expected,
                                                     uint64_t value) {
  rocm_trace_emit_hsa_api_enter(__func__);
  ROCR_TRACE_API_RET_U64(
      coreApiTable->hsa_queue_cas_write_index_scacquire_fn(queue, expected, value));
}

uint64_t HSA_API hsa_queue_cas_write_index_relaxed(const hsa_queue_t* queue, uint64_t expected,
                                                   uint64_t value) {
  rocm_trace_emit_hsa_api_enter(__func__);
  ROCR_TRACE_API_RET_U64(
      coreApiTable->hsa_queue_cas_write_index_relaxed_fn(queue, expected, value));
}

uint64_t HSA_API hsa_queue_cas_write_index_screlease(const hsa_queue_t* queue, uint64_t expected,
                                                     uint64_t value) {
  rocm_trace_emit_hsa_api_enter(__func__);
  ROCR_TRACE_API_RET_U64(
      coreApiTable->hsa_queue_cas_write_index_screlease_fn(queue, expected, value));
}

uint64_t HSA_API hsa_queue_add_write_index_scacq_screl(const hsa_queue_t* queue, uint64_t value) {
  rocm_trace_emit_hsa_api_enter(__func__);
  ROCR_TRACE_API_RET_U64(coreApiTable->hsa_queue_add_write_index_scacq_screl_fn(queue, value));
}

uint64_t HSA_API hsa_queue_add_write_index_scacquire(const hsa_queue_t* queue, uint64_t value) {
  rocm_trace_emit_hsa_api_enter(__func__);
  ROCR_TRACE_API_RET_U64(coreApiTable->hsa_queue_add_write_index_scacquire_fn(queue, value));
}

uint64_t HSA_API hsa_queue_add_write_index_relaxed(const hsa_queue_t* queue, uint64_t value) {
  rocm_trace_emit_hsa_api_enter(__func__);
  ROCR_TRACE_API_RET_U64(coreApiTable->hsa_queue_add_write_index_relaxed_fn(queue, value));
}

uint64_t HSA_API hsa_queue_add_write_index_screlease(const hsa_queue_t* queue, uint64_t value) {
  rocm_trace_emit_hsa_api_enter(__func__);
  ROCR_TRACE_API_RET_U64(coreApiTable->hsa_queue_add_write_index_screlease_fn(queue, value));
}

void HSA_API hsa_queue_store_read_index_relaxed(const hsa_queue_t* queue, uint64_t value) {
  rocm_trace_emit_hsa_api_enter(__func__);
  ROCR_TRACE_API_RET_VOID(coreApiTable->hsa_queue_store_read_index_relaxed_fn(queue, value));
}

void HSA_API hsa_queue_store_read_index_screlease(const hsa_queue_t* queue, uint64_t value) {
  rocm_trace_emit_hsa_api_enter(__func__);
  ROCR_TRACE_API_RET_VOID(coreApiTable->hsa_queue_store_read_index_screlease_fn(queue, value));
}

hsa_status_t HSA_API hsa_agent_iterate_regions(
    hsa_agent_t agent, hsa_status_t (*callback)(hsa_region_t region, void* data), void* data) {
  rocm_trace_emit_hsa_api_enter(__func__);
  ROCR_TRACE_API_RET_STATUS(coreApiTable->hsa_agent_iterate_regions_fn(agent, callback, data));
}

hsa_status_t HSA_API hsa_region_get_info(hsa_region_t region, hsa_region_info_t attribute,
                                         void* value) {
  rocm_trace_emit_hsa_api_enter(__func__);
  ROCR_TRACE_API_RET_STATUS(coreApiTable->hsa_region_get_info_fn(region, attribute, value));
}

hsa_status_t HSA_API hsa_memory_register(void* address, size_t size) {
  rocm_trace_emit_hsa_api_enter(__func__);
  ROCR_TRACE_API_RET_STATUS(coreApiTable->hsa_memory_register_fn(address, size));
}

hsa_status_t HSA_API hsa_memory_deregister(void* address, size_t size) {
  rocm_trace_emit_hsa_api_enter(__func__);
  ROCR_TRACE_API_RET_STATUS(coreApiTable->hsa_memory_deregister_fn(address, size));
}

hsa_status_t HSA_API hsa_memory_allocate(hsa_region_t region, size_t size, void** ptr) {
  rocm_trace_emit_hsa_api_enter(__func__);
  ROCR_TRACE_API_RET_STATUS(coreApiTable->hsa_memory_allocate_fn(region, size, ptr));
}

hsa_status_t HSA_API hsa_memory_free(void* ptr) {
  rocm_trace_emit_hsa_api_enter(__func__);
  ROCR_TRACE_API_RET_STATUS(coreApiTable->hsa_memory_free_fn(ptr));
}

hsa_status_t HSA_API hsa_memory_copy(void* dst, const void* src, size_t size) {
  rocm_trace_emit_hsa_api_enter(__func__);
  ROCR_TRACE_API_RET_STATUS(coreApiTable->hsa_memory_copy_fn(dst, src, size));
}

hsa_status_t HSA_API hsa_memory_assign_agent(void* ptr, hsa_agent_t agent,
                                             hsa_access_permission_t access) {
  rocm_trace_emit_hsa_api_enter(__func__);
  ROCR_TRACE_API_RET_STATUS(coreApiTable->hsa_memory_assign_agent_fn(ptr, agent, access));
}

hsa_status_t HSA_API hsa_signal_create(hsa_signal_value_t initial_value, uint32_t num_consumers,
                                       const hsa_agent_t* consumers, hsa_signal_t* signal) {
  rocm_trace_emit_hsa_api_enter(__func__); /* __ROCM_CURATED__: hsa_signal_create */
  auto const __rocm_in_initial_value = initial_value;
  auto const __rocm_in_num_consumers = num_consumers;
  auto const __rocm_in_consumers = consumers;
  ROCR_TRACE_API_RET_STATUS_CURATED_HSA(
      hsa_signal_create,
      coreApiTable->hsa_signal_create_fn(initial_value, num_consumers, consumers, signal),
      (__rocm_in_initial_value), (__rocm_in_num_consumers),
      (const void*)(uintptr_t)(__rocm_in_consumers), signal);
}

hsa_status_t HSA_API hsa_signal_destroy(hsa_signal_t signal) {
  rocm_trace_emit_hsa_api_enter(__func__); /* __ROCM_CURATED__: hsa_signal_destroy */
  auto const __rocm_in_signal = signal;
  ROCR_TRACE_API_RET_STATUS_CURATED_HSA(hsa_signal_destroy,
                                        coreApiTable->hsa_signal_destroy_fn(signal),
                                        (uint64_t)((__rocm_in_signal).handle));
}

hsa_signal_value_t HSA_API hsa_signal_load_relaxed(hsa_signal_t signal) {
  rocm_trace_emit_hsa_api_enter(__func__);
  ROCR_TRACE_API_RET_I64(coreApiTable->hsa_signal_load_relaxed_fn(signal));
}

hsa_signal_value_t HSA_API hsa_signal_load_scacquire(hsa_signal_t signal) {
  rocm_trace_emit_hsa_api_enter(__func__);
  ROCR_TRACE_API_RET_I64(coreApiTable->hsa_signal_load_scacquire_fn(signal));
}

void HSA_API hsa_signal_store_relaxed(hsa_signal_t signal, hsa_signal_value_t value) {
  rocm_trace_emit_hsa_api_enter(__func__);
  ROCR_TRACE_API_RET_VOID(coreApiTable->hsa_signal_store_relaxed_fn(signal, value));
}

void HSA_API hsa_signal_store_screlease(hsa_signal_t signal, hsa_signal_value_t value) {
  rocm_trace_emit_hsa_api_enter(__func__);
  ROCR_TRACE_API_RET_VOID(coreApiTable->hsa_signal_store_screlease_fn(signal, value));
}

void HSA_API hsa_signal_silent_store_relaxed(hsa_signal_t signal, hsa_signal_value_t value) {
  rocm_trace_emit_hsa_api_enter(__func__);
  ROCR_TRACE_API_RET_VOID(coreApiTable->hsa_signal_silent_store_relaxed_fn(signal, value));
}

void HSA_API hsa_signal_silent_store_screlease(hsa_signal_t signal, hsa_signal_value_t value) {
  rocm_trace_emit_hsa_api_enter(__func__);
  ROCR_TRACE_API_RET_VOID(coreApiTable->hsa_signal_silent_store_screlease_fn(signal, value));
}

hsa_signal_value_t HSA_API hsa_signal_wait_relaxed(hsa_signal_t signal,
                                                   hsa_signal_condition_t condition,
                                                   hsa_signal_value_t compare_value,
                                                   uint64_t timeout_hint,
                                                   hsa_wait_state_t wait_expectancy_hint) {
  rocm_trace_emit_hsa_api_enter(__func__);
  ROCR_TRACE_API_RET_I64(coreApiTable->hsa_signal_wait_relaxed_fn(
      signal, condition, compare_value, timeout_hint, wait_expectancy_hint));
}

hsa_signal_value_t HSA_API hsa_signal_wait_scacquire(hsa_signal_t signal,
                                                     hsa_signal_condition_t condition,
                                                     hsa_signal_value_t compare_value,
                                                     uint64_t timeout_hint,
                                                     hsa_wait_state_t wait_expectancy_hint) {
  rocm_trace_emit_hsa_api_enter(__func__);
  ROCR_TRACE_API_RET_I64(coreApiTable->hsa_signal_wait_scacquire_fn(
      signal, condition, compare_value, timeout_hint, wait_expectancy_hint));
}

hsa_status_t HSA_API hsa_signal_group_create(uint32_t num_signals, const hsa_signal_t* signals,
                                             uint32_t num_consumers, const hsa_agent_t* consumers,
                                             hsa_signal_group_t* signal_group) {
  rocm_trace_emit_hsa_api_enter(__func__);
  ROCR_TRACE_API_RET_STATUS(coreApiTable->hsa_signal_group_create_fn(
      num_signals, signals, num_consumers, consumers, signal_group));
}

hsa_status_t HSA_API hsa_signal_group_destroy(hsa_signal_group_t signal_group) {
  rocm_trace_emit_hsa_api_enter(__func__);
  ROCR_TRACE_API_RET_STATUS(coreApiTable->hsa_signal_group_destroy_fn(signal_group));
}

hsa_status_t HSA_API hsa_signal_group_wait_any_relaxed(hsa_signal_group_t signal_group,
                                                       const hsa_signal_condition_t* conditions,
                                                       const hsa_signal_value_t* compare_values,
                                                       hsa_wait_state_t wait_state_hint,
                                                       hsa_signal_t* signal,
                                                       hsa_signal_value_t* value) {
  rocm_trace_emit_hsa_api_enter(__func__);
  ROCR_TRACE_API_RET_STATUS(coreApiTable->hsa_signal_group_wait_any_relaxed_fn(
      signal_group, conditions, compare_values, wait_state_hint, signal, value));
}

hsa_status_t HSA_API hsa_signal_group_wait_any_scacquire(hsa_signal_group_t signal_group,
                                                         const hsa_signal_condition_t* conditions,
                                                         const hsa_signal_value_t* compare_values,
                                                         hsa_wait_state_t wait_state_hint,
                                                         hsa_signal_t* signal,
                                                         hsa_signal_value_t* value) {
  rocm_trace_emit_hsa_api_enter(__func__);
  ROCR_TRACE_API_RET_STATUS(coreApiTable->hsa_signal_group_wait_any_scacquire_fn(
      signal_group, conditions, compare_values, wait_state_hint, signal, value));
}

void HSA_API hsa_signal_and_relaxed(hsa_signal_t signal, hsa_signal_value_t value) {
  rocm_trace_emit_hsa_api_enter(__func__);
  ROCR_TRACE_API_RET_VOID(coreApiTable->hsa_signal_and_relaxed_fn(signal, value));
}

void HSA_API hsa_signal_and_scacquire(hsa_signal_t signal, hsa_signal_value_t value) {
  rocm_trace_emit_hsa_api_enter(__func__);
  ROCR_TRACE_API_RET_VOID(coreApiTable->hsa_signal_and_scacquire_fn(signal, value));
}

void HSA_API hsa_signal_and_screlease(hsa_signal_t signal, hsa_signal_value_t value) {
  rocm_trace_emit_hsa_api_enter(__func__);
  ROCR_TRACE_API_RET_VOID(coreApiTable->hsa_signal_and_screlease_fn(signal, value));
}

void HSA_API hsa_signal_and_scacq_screl(hsa_signal_t signal, hsa_signal_value_t value) {
  rocm_trace_emit_hsa_api_enter(__func__);
  ROCR_TRACE_API_RET_VOID(coreApiTable->hsa_signal_and_scacq_screl_fn(signal, value));
}

void HSA_API hsa_signal_or_relaxed(hsa_signal_t signal, hsa_signal_value_t value) {
  rocm_trace_emit_hsa_api_enter(__func__);
  ROCR_TRACE_API_RET_VOID(coreApiTable->hsa_signal_or_relaxed_fn(signal, value));
}

void HSA_API hsa_signal_or_scacquire(hsa_signal_t signal, hsa_signal_value_t value) {
  rocm_trace_emit_hsa_api_enter(__func__);
  ROCR_TRACE_API_RET_VOID(coreApiTable->hsa_signal_or_scacquire_fn(signal, value));
}

void HSA_API hsa_signal_or_screlease(hsa_signal_t signal, hsa_signal_value_t value) {
  rocm_trace_emit_hsa_api_enter(__func__);
  ROCR_TRACE_API_RET_VOID(coreApiTable->hsa_signal_or_screlease_fn(signal, value));
}

void HSA_API hsa_signal_or_scacq_screl(hsa_signal_t signal, hsa_signal_value_t value) {
  rocm_trace_emit_hsa_api_enter(__func__);
  ROCR_TRACE_API_RET_VOID(coreApiTable->hsa_signal_or_scacq_screl_fn(signal, value));
}

void HSA_API hsa_signal_xor_relaxed(hsa_signal_t signal, hsa_signal_value_t value) {
  rocm_trace_emit_hsa_api_enter(__func__);
  ROCR_TRACE_API_RET_VOID(coreApiTable->hsa_signal_xor_relaxed_fn(signal, value));
}

void HSA_API hsa_signal_xor_scacquire(hsa_signal_t signal, hsa_signal_value_t value) {
  rocm_trace_emit_hsa_api_enter(__func__);
  ROCR_TRACE_API_RET_VOID(coreApiTable->hsa_signal_xor_scacquire_fn(signal, value));
}

void HSA_API hsa_signal_xor_screlease(hsa_signal_t signal, hsa_signal_value_t value) {
  rocm_trace_emit_hsa_api_enter(__func__);
  ROCR_TRACE_API_RET_VOID(coreApiTable->hsa_signal_xor_screlease_fn(signal, value));
}

void HSA_API hsa_signal_xor_scacq_screl(hsa_signal_t signal, hsa_signal_value_t value) {
  rocm_trace_emit_hsa_api_enter(__func__);
  ROCR_TRACE_API_RET_VOID(coreApiTable->hsa_signal_xor_scacq_screl_fn(signal, value));
}

void HSA_API hsa_signal_add_relaxed(hsa_signal_t signal, hsa_signal_value_t value) {
  rocm_trace_emit_hsa_api_enter(__func__);
  ROCR_TRACE_API_RET_VOID(coreApiTable->hsa_signal_add_relaxed_fn(signal, value));
}

void HSA_API hsa_signal_add_scacquire(hsa_signal_t signal, hsa_signal_value_t value) {
  rocm_trace_emit_hsa_api_enter(__func__);
  ROCR_TRACE_API_RET_VOID(coreApiTable->hsa_signal_add_scacquire_fn(signal, value));
}

void HSA_API hsa_signal_add_screlease(hsa_signal_t signal, hsa_signal_value_t value) {
  rocm_trace_emit_hsa_api_enter(__func__);
  ROCR_TRACE_API_RET_VOID(coreApiTable->hsa_signal_add_screlease_fn(signal, value));
}

void HSA_API hsa_signal_add_scacq_screl(hsa_signal_t signal, hsa_signal_value_t value) {
  rocm_trace_emit_hsa_api_enter(__func__);
  ROCR_TRACE_API_RET_VOID(coreApiTable->hsa_signal_add_scacq_screl_fn(signal, value));
}

void HSA_API hsa_signal_subtract_relaxed(hsa_signal_t signal, hsa_signal_value_t value) {
  rocm_trace_emit_hsa_api_enter(__func__);
  ROCR_TRACE_API_RET_VOID(coreApiTable->hsa_signal_subtract_relaxed_fn(signal, value));
}

void HSA_API hsa_signal_subtract_scacquire(hsa_signal_t signal, hsa_signal_value_t value) {
  rocm_trace_emit_hsa_api_enter(__func__);
  ROCR_TRACE_API_RET_VOID(coreApiTable->hsa_signal_subtract_scacquire_fn(signal, value));
}

void HSA_API hsa_signal_subtract_screlease(hsa_signal_t signal, hsa_signal_value_t value) {
  rocm_trace_emit_hsa_api_enter(__func__);
  ROCR_TRACE_API_RET_VOID(coreApiTable->hsa_signal_subtract_screlease_fn(signal, value));
}

void HSA_API hsa_signal_subtract_scacq_screl(hsa_signal_t signal, hsa_signal_value_t value) {
  rocm_trace_emit_hsa_api_enter(__func__);
  ROCR_TRACE_API_RET_VOID(coreApiTable->hsa_signal_subtract_scacq_screl_fn(signal, value));
}

hsa_signal_value_t HSA_API hsa_signal_exchange_relaxed(hsa_signal_t signal,
                                                       hsa_signal_value_t value) {
  rocm_trace_emit_hsa_api_enter(__func__);
  ROCR_TRACE_API_RET_I64(coreApiTable->hsa_signal_exchange_relaxed_fn(signal, value));
}

hsa_signal_value_t HSA_API hsa_signal_exchange_scacquire(hsa_signal_t signal,
                                                         hsa_signal_value_t value) {
  rocm_trace_emit_hsa_api_enter(__func__);
  ROCR_TRACE_API_RET_I64(coreApiTable->hsa_signal_exchange_scacquire_fn(signal, value));
}

hsa_signal_value_t HSA_API hsa_signal_exchange_screlease(hsa_signal_t signal,
                                                         hsa_signal_value_t value) {
  rocm_trace_emit_hsa_api_enter(__func__);
  ROCR_TRACE_API_RET_I64(coreApiTable->hsa_signal_exchange_screlease_fn(signal, value));
}

hsa_signal_value_t HSA_API hsa_signal_exchange_scacq_screl(hsa_signal_t signal,
                                                           hsa_signal_value_t value) {
  rocm_trace_emit_hsa_api_enter(__func__);
  ROCR_TRACE_API_RET_I64(coreApiTable->hsa_signal_exchange_scacq_screl_fn(signal, value));
}

hsa_signal_value_t HSA_API hsa_signal_cas_relaxed(hsa_signal_t signal, hsa_signal_value_t expected,
                                                  hsa_signal_value_t value) {
  rocm_trace_emit_hsa_api_enter(__func__);
  ROCR_TRACE_API_RET_I64(coreApiTable->hsa_signal_cas_relaxed_fn(signal, expected, value));
}

hsa_signal_value_t HSA_API hsa_signal_cas_scacquire(hsa_signal_t signal,
                                                    hsa_signal_value_t expected,
                                                    hsa_signal_value_t value) {
  rocm_trace_emit_hsa_api_enter(__func__);
  ROCR_TRACE_API_RET_I64(coreApiTable->hsa_signal_cas_scacquire_fn(signal, expected, value));
}

hsa_signal_value_t HSA_API hsa_signal_cas_screlease(hsa_signal_t signal,
                                                    hsa_signal_value_t expected,
                                                    hsa_signal_value_t value) {
  rocm_trace_emit_hsa_api_enter(__func__);
  ROCR_TRACE_API_RET_I64(coreApiTable->hsa_signal_cas_screlease_fn(signal, expected, value));
}

hsa_signal_value_t HSA_API hsa_signal_cas_scacq_screl(hsa_signal_t signal,
                                                      hsa_signal_value_t expected,
                                                      hsa_signal_value_t value) {
  rocm_trace_emit_hsa_api_enter(__func__);
  ROCR_TRACE_API_RET_I64(coreApiTable->hsa_signal_cas_scacq_screl_fn(signal, expected, value));
}

//===--- Instruction Set Architecture -------------------------------------===//

hsa_status_t HSA_API hsa_isa_from_name(const char* name, hsa_isa_t* isa) {
  rocm_trace_emit_hsa_api_enter(__func__);
  ROCR_TRACE_API_RET_STATUS(coreApiTable->hsa_isa_from_name_fn(name, isa));
}

hsa_status_t HSA_API hsa_agent_iterate_isas(hsa_agent_t agent,
                                            hsa_status_t (*callback)(hsa_isa_t isa, void* data),
                                            void* data) {
  rocm_trace_emit_hsa_api_enter(__func__);
  ROCR_TRACE_API_RET_STATUS(coreApiTable->hsa_agent_iterate_isas_fn(agent, callback, data));
}

/* deprecated */ hsa_status_t HSA_API hsa_isa_get_info(hsa_isa_t isa, hsa_isa_info_t attribute,
                                                       uint32_t index, void* value) {
  rocm_trace_emit_hsa_api_enter(__func__);
  ROCR_TRACE_API_RET_STATUS(coreApiTable->hsa_isa_get_info_fn(isa, attribute, index, value));
}

hsa_status_t HSA_API hsa_isa_get_info_alt(hsa_isa_t isa, hsa_isa_info_t attribute, void* value) {
  rocm_trace_emit_hsa_api_enter(__func__);
  ROCR_TRACE_API_RET_STATUS(coreApiTable->hsa_isa_get_info_alt_fn(isa, attribute, value));
}

hsa_status_t HSA_API hsa_isa_get_exception_policies(hsa_isa_t isa, hsa_profile_t profile,
                                                    uint16_t* mask) {
  rocm_trace_emit_hsa_api_enter(__func__);
  ROCR_TRACE_API_RET_STATUS(coreApiTable->hsa_isa_get_exception_policies_fn(isa, profile, mask));
}

hsa_status_t HSA_API hsa_isa_get_round_method(hsa_isa_t isa, hsa_fp_type_t fp_type,
                                              hsa_flush_mode_t flush_mode,
                                              hsa_round_method_t* round_method) {
  rocm_trace_emit_hsa_api_enter(__func__);
  ROCR_TRACE_API_RET_STATUS(
      coreApiTable->hsa_isa_get_round_method_fn(isa, fp_type, flush_mode, round_method));
}

hsa_status_t HSA_API hsa_wavefront_get_info(hsa_wavefront_t wavefront,
                                            hsa_wavefront_info_t attribute, void* value) {
  rocm_trace_emit_hsa_api_enter(__func__);
  ROCR_TRACE_API_RET_STATUS(coreApiTable->hsa_wavefront_get_info_fn(wavefront, attribute, value));
}

hsa_status_t HSA_API hsa_isa_iterate_wavefronts(
    hsa_isa_t isa, hsa_status_t (*callback)(hsa_wavefront_t wavefront, void* data), void* data) {
  rocm_trace_emit_hsa_api_enter(__func__);
  ROCR_TRACE_API_RET_STATUS(coreApiTable->hsa_isa_iterate_wavefronts_fn(isa, callback, data));
}

/* deprecated */ hsa_status_t HSA_API hsa_isa_compatible(hsa_isa_t code_object_isa,
                                                         hsa_isa_t agent_isa, bool* result) {
  rocm_trace_emit_hsa_api_enter(__func__);
  ROCR_TRACE_API_RET_STATUS(
      coreApiTable->hsa_isa_compatible_fn(code_object_isa, agent_isa, result));
}

//===--- Code Objects (deprecated) ----------------------------------------===//

/* deprecated */ hsa_status_t HSA_API hsa_code_object_serialize(
    hsa_code_object_t code_object,
    hsa_status_t (*alloc_callback)(size_t size, hsa_callback_data_t data, void** address),
    hsa_callback_data_t callback_data, const char* options, void** serialized_code_object,
    size_t* serialized_code_object_size) {
  rocm_trace_emit_hsa_api_enter(__func__);
  ROCR_TRACE_API_RET_STATUS(coreApiTable->hsa_code_object_serialize_fn(
      code_object, alloc_callback, callback_data, options, serialized_code_object,
      serialized_code_object_size));
}

/* deprecated */ hsa_status_t HSA_API
hsa_code_object_deserialize(void* serialized_code_object, size_t serialized_code_object_size,
                            const char* options, hsa_code_object_t* code_object) {
  rocm_trace_emit_hsa_api_enter(__func__);
  ROCR_TRACE_API_RET_STATUS(coreApiTable->hsa_code_object_deserialize_fn(
      serialized_code_object, serialized_code_object_size, options, code_object));
}

/* deprecated */ hsa_status_t HSA_API hsa_code_object_destroy(hsa_code_object_t code_object) {
  rocm_trace_emit_hsa_api_enter(__func__);
  ROCR_TRACE_API_RET_STATUS(coreApiTable->hsa_code_object_destroy_fn(code_object));
}

/* deprecated */ hsa_status_t HSA_API hsa_code_object_get_info(hsa_code_object_t code_object,
                                                               hsa_code_object_info_t attribute,
                                                               void* value) {
  rocm_trace_emit_hsa_api_enter(__func__);
  ROCR_TRACE_API_RET_STATUS(
      coreApiTable->hsa_code_object_get_info_fn(code_object, attribute, value));
}

/* deprecated */ hsa_status_t HSA_API hsa_code_object_get_symbol(hsa_code_object_t code_object,
                                                                 const char* symbol_name,
                                                                 hsa_code_symbol_t* symbol) {
  rocm_trace_emit_hsa_api_enter(__func__);
  ROCR_TRACE_API_RET_STATUS(
      coreApiTable->hsa_code_object_get_symbol_fn(code_object, symbol_name, symbol));
}

/* deprecated */ hsa_status_t HSA_API
hsa_code_object_get_symbol_from_name(hsa_code_object_t code_object, const char* module_name,
                                     const char* symbol_name, hsa_code_symbol_t* symbol) {
  rocm_trace_emit_hsa_api_enter(__func__);
  ROCR_TRACE_API_RET_STATUS(coreApiTable->hsa_code_object_get_symbol_from_name_fn(
      code_object, module_name, symbol_name, symbol));
}

/* deprecated */ hsa_status_t HSA_API hsa_code_symbol_get_info(hsa_code_symbol_t code_symbol,
                                                               hsa_code_symbol_info_t attribute,
                                                               void* value) {
  rocm_trace_emit_hsa_api_enter(__func__);
  ROCR_TRACE_API_RET_STATUS(
      coreApiTable->hsa_code_symbol_get_info_fn(code_symbol, attribute, value));
}

/* deprecated */ hsa_status_t HSA_API hsa_code_object_iterate_symbols(
    hsa_code_object_t code_object,
    hsa_status_t (*callback)(hsa_code_object_t code_object, hsa_code_symbol_t symbol, void* data),
    void* data) {
  rocm_trace_emit_hsa_api_enter(__func__);
  ROCR_TRACE_API_RET_STATUS(
      coreApiTable->hsa_code_object_iterate_symbols_fn(code_object, callback, data));
}

//===--- Executable -------------------------------------------------------===//

hsa_status_t HSA_API hsa_code_object_reader_create_from_file(
    hsa_file_t file, hsa_code_object_reader_t* code_object_reader) {
  rocm_trace_emit_hsa_api_enter(__func__);
  ROCR_TRACE_API_RET_STATUS(
      coreApiTable->hsa_code_object_reader_create_from_file_fn(file, code_object_reader));
}

hsa_status_t HSA_API hsa_code_object_reader_create_from_memory(
    const void* code_object, size_t size, hsa_code_object_reader_t* code_object_reader) {
  rocm_trace_emit_hsa_api_enter(__func__);
  ROCR_TRACE_API_RET_STATUS(coreApiTable->hsa_code_object_reader_create_from_memory_fn(
      code_object, size, code_object_reader));
}

hsa_status_t HSA_API hsa_code_object_reader_destroy(hsa_code_object_reader_t code_object_reader) {
  rocm_trace_emit_hsa_api_enter(__func__);
  ROCR_TRACE_API_RET_STATUS(coreApiTable->hsa_code_object_reader_destroy_fn(code_object_reader));
}

/* deprecated */ hsa_status_t HSA_API hsa_executable_create(hsa_profile_t profile,
                                                            hsa_executable_state_t executable_state,
                                                            const char* options,
                                                            hsa_executable_t* executable) {
  rocm_trace_emit_hsa_api_enter(__func__);
  ROCR_TRACE_API_RET_STATUS(
      coreApiTable->hsa_executable_create_fn(profile, executable_state, options, executable));
}

hsa_status_t HSA_API hsa_executable_create_alt(
    hsa_profile_t profile, hsa_default_float_rounding_mode_t default_float_rounding_mode,
    const char* options, hsa_executable_t* executable) {
  rocm_trace_emit_hsa_api_enter(__func__);
  ROCR_TRACE_API_RET_STATUS(coreApiTable->hsa_executable_create_alt_fn(
      profile, default_float_rounding_mode, options, executable));
}

hsa_status_t HSA_API hsa_executable_destroy(hsa_executable_t executable) {
  rocm_trace_emit_hsa_api_enter(__func__);
  ROCR_TRACE_API_RET_STATUS(coreApiTable->hsa_executable_destroy_fn(executable));
}

/* deprecated */ hsa_status_t HSA_API hsa_executable_load_code_object(hsa_executable_t executable,
                                                                      hsa_agent_t agent,
                                                                      hsa_code_object_t code_object,
                                                                      const char* options) {
  rocm_trace_emit_hsa_api_enter(__func__);
  ROCR_TRACE_API_RET_STATUS(
      coreApiTable->hsa_executable_load_code_object_fn(executable, agent, code_object, options));
}

hsa_status_t HSA_API hsa_executable_load_program_code_object(
    hsa_executable_t executable, hsa_code_object_reader_t code_object_reader, const char* options,
    hsa_loaded_code_object_t* loaded_code_object) {
  rocm_trace_emit_hsa_api_enter(__func__);
  ROCR_TRACE_API_RET_STATUS(coreApiTable->hsa_executable_load_program_code_object_fn(
      executable, code_object_reader, options, loaded_code_object));
}

hsa_status_t HSA_API hsa_executable_load_agent_code_object(
    hsa_executable_t executable, hsa_agent_t agent, hsa_code_object_reader_t code_object_reader,
    const char* options, hsa_loaded_code_object_t* loaded_code_object) {
  rocm_trace_emit_hsa_api_enter(__func__);
  ROCR_TRACE_API_RET_STATUS(coreApiTable->hsa_executable_load_agent_code_object_fn(
      executable, agent, code_object_reader, options, loaded_code_object));
}

hsa_status_t HSA_API hsa_executable_freeze(hsa_executable_t executable, const char* options) {
  rocm_trace_emit_hsa_api_enter(__func__);
  ROCR_TRACE_API_RET_STATUS(coreApiTable->hsa_executable_freeze_fn(executable, options));
}

hsa_status_t HSA_API hsa_executable_get_info(hsa_executable_t executable,
                                             hsa_executable_info_t attribute, void* value) {
  rocm_trace_emit_hsa_api_enter(__func__);
  ROCR_TRACE_API_RET_STATUS(coreApiTable->hsa_executable_get_info_fn(executable, attribute, value));
}

hsa_status_t HSA_API hsa_executable_global_variable_define(hsa_executable_t executable,
                                                           const char* variable_name,
                                                           void* address) {
  rocm_trace_emit_hsa_api_enter(__func__);
  ROCR_TRACE_API_RET_STATUS(
      coreApiTable->hsa_executable_global_variable_define_fn(executable, variable_name, address));
}

hsa_status_t HSA_API hsa_executable_agent_global_variable_define(hsa_executable_t executable,
                                                                 hsa_agent_t agent,
                                                                 const char* variable_name,
                                                                 void* address) {
  rocm_trace_emit_hsa_api_enter(__func__);
  ROCR_TRACE_API_RET_STATUS(coreApiTable->hsa_executable_agent_global_variable_define_fn(
      executable, agent, variable_name, address));
}

hsa_status_t HSA_API hsa_executable_readonly_variable_define(hsa_executable_t executable,
                                                             hsa_agent_t agent,
                                                             const char* variable_name,
                                                             void* address) {
  rocm_trace_emit_hsa_api_enter(__func__);
  ROCR_TRACE_API_RET_STATUS(coreApiTable->hsa_executable_readonly_variable_define_fn(
      executable, agent, variable_name, address));
}

hsa_status_t HSA_API hsa_executable_validate(hsa_executable_t executable, uint32_t* result) {
  rocm_trace_emit_hsa_api_enter(__func__);
  ROCR_TRACE_API_RET_STATUS(coreApiTable->hsa_executable_validate_fn(executable, result));
}

hsa_status_t HSA_API hsa_executable_validate_alt(hsa_executable_t executable, const char* options,
                                                 uint32_t* result) {
  rocm_trace_emit_hsa_api_enter(__func__);
  ROCR_TRACE_API_RET_STATUS(
      coreApiTable->hsa_executable_validate_alt_fn(executable, options, result));
}

/* deprecated */ hsa_status_t HSA_API hsa_executable_get_symbol(
    hsa_executable_t executable, const char* module_name, const char* symbol_name,
    hsa_agent_t agent, int32_t call_convention, hsa_executable_symbol_t* symbol) {
  rocm_trace_emit_hsa_api_enter(__func__);
  ROCR_TRACE_API_RET_STATUS(coreApiTable->hsa_executable_get_symbol_fn(
      executable, module_name, symbol_name, agent, call_convention, symbol));
}

hsa_status_t HSA_API hsa_executable_get_symbol_by_name(hsa_executable_t executable,
                                                       const char* symbol_name,
                                                       const hsa_agent_t* agent,
                                                       hsa_executable_symbol_t* symbol) {
  rocm_trace_emit_hsa_api_enter(__func__);
  ROCR_TRACE_API_RET_STATUS(
      coreApiTable->hsa_executable_get_symbol_by_name_fn(executable, symbol_name, agent, symbol));
}

hsa_status_t HSA_API hsa_executable_symbol_get_info(hsa_executable_symbol_t executable_symbol,
                                                    hsa_executable_symbol_info_t attribute,
                                                    void* value) {
  rocm_trace_emit_hsa_api_enter(__func__);
  ROCR_TRACE_API_RET_STATUS(
      coreApiTable->hsa_executable_symbol_get_info_fn(executable_symbol, attribute, value));
}

/* deprecated */ hsa_status_t HSA_API
hsa_executable_iterate_symbols(hsa_executable_t executable,
                               hsa_status_t (*callback)(hsa_executable_t executable,
                                                        hsa_executable_symbol_t symbol, void* data),
                               void* data) {
  rocm_trace_emit_hsa_api_enter(__func__);
  ROCR_TRACE_API_RET_STATUS(
      coreApiTable->hsa_executable_iterate_symbols_fn(executable, callback, data));
}

hsa_status_t HSA_API hsa_executable_iterate_agent_symbols(
    hsa_executable_t executable, hsa_agent_t agent,
    hsa_status_t (*callback)(hsa_executable_t exec, hsa_agent_t agent,
                             hsa_executable_symbol_t symbol, void* data),
    void* data) {
  rocm_trace_emit_hsa_api_enter(__func__);
  ROCR_TRACE_API_RET_STATUS(
      coreApiTable->hsa_executable_iterate_agent_symbols_fn(executable, agent, callback, data));
}

hsa_status_t HSA_API hsa_executable_iterate_program_symbols(
    hsa_executable_t executable,
    hsa_status_t (*callback)(hsa_executable_t exec, hsa_executable_symbol_t symbol, void* data),
    void* data) {
  rocm_trace_emit_hsa_api_enter(__func__);
  ROCR_TRACE_API_RET_STATUS(
      coreApiTable->hsa_executable_iterate_program_symbols_fn(executable, callback, data));
}

//===--- Runtime Notifications --------------------------------------------===//

hsa_status_t HSA_API hsa_status_string(hsa_status_t status, const char** status_string) {
  rocm_trace_emit_hsa_api_enter(__func__);
  ROCR_TRACE_API_RET_STATUS(coreApiTable->hsa_status_string_fn(status, status_string));
}

/*
 * Following set of functions are bundled as AMD Extension Apis
 */

// Pass through stub functions
hsa_status_t HSA_API hsa_amd_coherency_get_type(hsa_agent_t agent, hsa_amd_coherency_type_t* type) {
  rocm_trace_emit_hsa_api_enter(__func__);
  ROCR_TRACE_API_RET_STATUS(amdExtTable->hsa_amd_coherency_get_type_fn(agent, type));
}

// Pass through stub functions
hsa_status_t HSA_API hsa_amd_coherency_set_type(hsa_agent_t agent, hsa_amd_coherency_type_t type) {
  rocm_trace_emit_hsa_api_enter(__func__);
  ROCR_TRACE_API_RET_STATUS(amdExtTable->hsa_amd_coherency_set_type_fn(agent, type));
}

// Mirrors Amd Extension Apis
hsa_status_t HSA_API hsa_amd_profiling_set_profiler_enabled(hsa_queue_t* queue, int enable) {
  rocm_trace_emit_hsa_api_enter(__func__);
  ROCR_TRACE_API_RET_STATUS(amdExtTable->hsa_amd_profiling_set_profiler_enabled_fn(queue, enable));
}

hsa_status_t HSA_API hsa_amd_profiling_async_copy_enable(bool enable) {
  rocm_trace_emit_hsa_api_enter(__func__);
  ROCR_TRACE_API_RET_STATUS(amdExtTable->hsa_amd_profiling_async_copy_enable_fn(enable));
}

hsa_status_t HSA_API hsa_amd_agent_preload(hsa_agent_t agent, uint64_t flags) {
  rocm_trace_emit_hsa_api_enter(__func__);
  ROCR_TRACE_API_RET_STATUS(amdExtTable->hsa_amd_agent_preload_fn(agent, flags));
}

// Mirrors Amd Extension Apis
hsa_status_t HSA_API hsa_amd_profiling_get_dispatch_time(hsa_agent_t agent, hsa_signal_t signal,
                                                         hsa_amd_profiling_dispatch_time_t* time) {
  rocm_trace_emit_hsa_api_enter(__func__);
  ROCR_TRACE_API_RET_STATUS(
      amdExtTable->hsa_amd_profiling_get_dispatch_time_fn(agent, signal, time));
}

hsa_status_t HSA_API hsa_amd_profiling_get_async_copy_time(
    hsa_signal_t hsa_signal, hsa_amd_profiling_async_copy_time_t* time) {
  rocm_trace_emit_hsa_api_enter(__func__);
  ROCR_TRACE_API_RET_STATUS(
      amdExtTable->hsa_amd_profiling_get_async_copy_time_fn(hsa_signal, time));
}

// Mirrors Amd Extension Apis
hsa_status_t HSA_API hsa_amd_profiling_convert_tick_to_system_domain(hsa_agent_t agent,
                                                                     uint64_t agent_tick,
                                                                     uint64_t* system_tick) {
  rocm_trace_emit_hsa_api_enter(__func__);
  ROCR_TRACE_API_RET_STATUS(amdExtTable->hsa_amd_profiling_convert_tick_to_system_domain_fn(
      agent, agent_tick, system_tick));
}

// Mirrors Amd Extension Apis
hsa_status_t HSA_API hsa_amd_signal_async_handler(hsa_signal_t signal, hsa_signal_condition_t cond,
                                                  hsa_signal_value_t value,
                                                  hsa_amd_signal_handler handler, void* arg) {
  rocm_trace_emit_hsa_api_enter(__func__);
  ROCR_TRACE_API_RET_STATUS(
      amdExtTable->hsa_amd_signal_async_handler_fn(signal, cond, value, handler, arg));
}

// Mirrors Amd Extension Apis
hsa_status_t HSA_API hsa_amd_async_function(void (*callback)(void* arg), void* arg) {
  rocm_trace_emit_hsa_api_enter(__func__);
  ROCR_TRACE_API_RET_STATUS(amdExtTable->hsa_amd_async_function_fn(callback, arg));
}

// Mirrors Amd Extension Apis
uint32_t HSA_API hsa_amd_signal_wait_all(uint32_t signal_count, hsa_signal_t* signals,
                                         hsa_signal_condition_t* conds, hsa_signal_value_t* values,
                                         uint64_t timeout_hint, hsa_wait_state_t wait_hint,
                                         hsa_signal_value_t* satisfying_values) {
  rocm_trace_emit_hsa_api_enter(__func__);
  ROCR_TRACE_API_RET_STATUS(amdExtTable->hsa_amd_signal_wait_all_fn(
      signal_count, signals, conds, values, timeout_hint, wait_hint, satisfying_values));
}

// Mirrors Amd Extension Apis
uint32_t HSA_API hsa_amd_signal_wait_any(uint32_t signal_count, hsa_signal_t* signals,
                                         hsa_signal_condition_t* conds, hsa_signal_value_t* values,
                                         uint64_t timeout_hint, hsa_wait_state_t wait_hint,
                                         hsa_signal_value_t* satisfying_value) {
  rocm_trace_emit_hsa_api_enter(__func__);
  ROCR_TRACE_API_RET_STATUS(amdExtTable->hsa_amd_signal_wait_any_fn(
      signal_count, signals, conds, values, timeout_hint, wait_hint, satisfying_value));
}

// Mirrors Amd Extension Apis
hsa_status_t HSA_API hsa_amd_queue_cu_set_mask(const hsa_queue_t* queue, uint32_t num_cu_mask_count,
                                               const uint32_t* cu_mask) {
  rocm_trace_emit_hsa_api_enter(__func__);
  ROCR_TRACE_API_RET_STATUS(
      amdExtTable->hsa_amd_queue_cu_set_mask_fn(queue, num_cu_mask_count, cu_mask));
}

// Mirrors Amd Extension Apis
hsa_status_t HSA_API hsa_amd_queue_cu_get_mask(const hsa_queue_t* queue, uint32_t num_cu_mask_count,
                                               uint32_t* cu_mask) {
  rocm_trace_emit_hsa_api_enter(__func__);
  ROCR_TRACE_API_RET_STATUS(
      amdExtTable->hsa_amd_queue_cu_get_mask_fn(queue, num_cu_mask_count, cu_mask));
}

// Mirrors Amd Extension Apis
hsa_status_t HSA_API hsa_amd_memory_pool_get_info(hsa_amd_memory_pool_t memory_pool,
                                                  hsa_amd_memory_pool_info_t attribute,
                                                  void* value) {
  rocm_trace_emit_hsa_api_enter(__func__);
  ROCR_TRACE_API_RET_STATUS(
      amdExtTable->hsa_amd_memory_pool_get_info_fn(memory_pool, attribute, value));
}

// Mirrors Amd Extension Apis
hsa_status_t HSA_API hsa_amd_agent_iterate_memory_pools(
    hsa_agent_t agent, hsa_status_t (*callback)(hsa_amd_memory_pool_t memory_pool, void* data),
    void* data) {
  rocm_trace_emit_hsa_api_enter(__func__);
  ROCR_TRACE_API_RET_STATUS(
      amdExtTable->hsa_amd_agent_iterate_memory_pools_fn(agent, callback, data));
}

// Mirrors Amd Extension Apis
hsa_status_t HSA_API hsa_amd_memory_pool_allocate(hsa_amd_memory_pool_t memory_pool, size_t size,
                                                  uint32_t flags, void** ptr) {
  rocm_trace_emit_hsa_api_enter(__func__); /* __ROCM_CURATED__: hsa_amd_memory_pool_allocate */
  auto const __rocm_in_memory_pool = memory_pool;
  auto const __rocm_in_size = size;
  auto const __rocm_in_flags = flags;
  ROCR_TRACE_API_RET_STATUS_CURATED_HSA(
      hsa_amd_memory_pool_allocate,
      amdExtTable->hsa_amd_memory_pool_allocate_fn(memory_pool, size, flags, ptr),
      (uint64_t)((__rocm_in_memory_pool).handle), (__rocm_in_size), (__rocm_in_flags), ptr);
}

// Mirrors Amd Extension Apis
hsa_status_t HSA_API hsa_amd_memory_pool_free(void* ptr) {
  rocm_trace_emit_hsa_api_enter(__func__); /* __ROCM_CURATED__: hsa_amd_memory_pool_free */
  auto const __rocm_in_ptr = ptr;
  ROCR_TRACE_API_RET_STATUS_CURATED_HSA(hsa_amd_memory_pool_free,
                                        amdExtTable->hsa_amd_memory_pool_free_fn(ptr),
                                        (const void*)(uintptr_t)(__rocm_in_ptr));
}

// Mirrors Amd Extension Apis
hsa_status_t HSA_API hsa_amd_memory_async_copy(void* dst, hsa_agent_t dst_agent, const void* src,
                                               hsa_agent_t src_agent, size_t size,
                                               uint32_t num_dep_signals,
                                               const hsa_signal_t* dep_signals,
                                               hsa_signal_t completion_signal) {
  rocm_trace_emit_hsa_api_enter(__func__); /* __ROCM_CURATED__: hsa_amd_memory_async_copy */
  auto const __rocm_in_dst = dst;
  auto const __rocm_in_dst_agent = dst_agent;
  auto const __rocm_in_src = src;
  auto const __rocm_in_src_agent = src_agent;
  auto const __rocm_in_size = size;
  auto const __rocm_in_num_dep_signals = num_dep_signals;
  auto const __rocm_in_dep_signals = dep_signals;
  auto const __rocm_in_completion_signal = completion_signal;
  ROCR_TRACE_API_RET_STATUS_CURATED_HSA(
      hsa_amd_memory_async_copy,
      amdExtTable->hsa_amd_memory_async_copy_fn(dst, dst_agent, src, src_agent, size,
                                                num_dep_signals, dep_signals, completion_signal),
      (const void*)(uintptr_t)(__rocm_in_dst), (uint64_t)((__rocm_in_dst_agent).handle),
      (const void*)(uintptr_t)(__rocm_in_src), (uint64_t)((__rocm_in_src_agent).handle),
      (__rocm_in_size), (__rocm_in_num_dep_signals),
      (const void*)(uintptr_t)(__rocm_in_dep_signals),
      (uint64_t)((__rocm_in_completion_signal).handle));
}

// Mirrors Amd Extension Apis
hsa_status_t HSA_API hsa_amd_memory_async_copy_on_engine(
    void* dst, hsa_agent_t dst_agent, const void* src, hsa_agent_t src_agent, size_t size,
    uint32_t num_dep_signals, const hsa_signal_t* dep_signals, hsa_signal_t completion_signal,
    hsa_amd_sdma_engine_id_t engine_id, bool force_copy_on_sdma) {
  rocm_trace_emit_hsa_api_enter(
      __func__); /* __ROCM_CURATED__: hsa_amd_memory_async_copy_on_engine */
  auto const __rocm_in_dst = dst;
  auto const __rocm_in_dst_agent = dst_agent;
  auto const __rocm_in_src = src;
  auto const __rocm_in_src_agent = src_agent;
  auto const __rocm_in_size = size;
  auto const __rocm_in_num_dep_signals = num_dep_signals;
  auto const __rocm_in_completion_signal = completion_signal;
  auto const __rocm_in_engine_id = engine_id;
  auto const __rocm_in_force_copy_on_sdma = force_copy_on_sdma;
  ROCR_TRACE_API_RET_STATUS_CURATED_HSA(
      hsa_amd_memory_async_copy_on_engine,
      amdExtTable->hsa_amd_memory_async_copy_on_engine_fn(
          dst, dst_agent, src, src_agent, size, num_dep_signals, dep_signals, completion_signal,
          engine_id, force_copy_on_sdma),
      (const void*)(uintptr_t)(__rocm_in_dst), (uint64_t)((__rocm_in_dst_agent).handle),
      (const void*)(uintptr_t)(__rocm_in_src), (uint64_t)((__rocm_in_src_agent).handle),
      (__rocm_in_size), (__rocm_in_num_dep_signals),
      (uint64_t)((__rocm_in_completion_signal).handle), (int32_t)(__rocm_in_engine_id),
      (__rocm_in_force_copy_on_sdma));
}

// Mirrors Amd Extension Apis
hsa_status_t HSA_API hsa_amd_memory_async_batch_copy(const hsa_amd_memory_copy_op_t* copy_ops,
                                                     uint32_t num_copy_ops,
                                                     uint32_t num_dep_signals,
                                                     const hsa_signal_t* dep_signals) {
  rocm_trace_emit_hsa_api_enter(__func__);
  ROCR_TRACE_API_RET_STATUS(amdExtTable->hsa_amd_memory_async_batch_copy_fn(
      copy_ops, num_copy_ops, num_dep_signals, dep_signals));
}

// Mirrors Amd Extension Apis
hsa_status_t HSA_API hsa_amd_memory_copy_engine_status(hsa_agent_t dst_agent, hsa_agent_t src_agent,
                                                       uint32_t* engine_ids_mask) {
  rocm_trace_emit_hsa_api_enter(__func__);
  ROCR_TRACE_API_RET_STATUS(
      amdExtTable->hsa_amd_memory_copy_engine_status_fn(dst_agent, src_agent, engine_ids_mask));
}

// Mirrors Amd Extension Apis
hsa_status_t HSA_API hsa_amd_memory_get_preferred_copy_engine(hsa_agent_t dst_agent,
                                                              hsa_agent_t src_agent,
                                                              uint32_t* recommended_ids_mask) {
  rocm_trace_emit_hsa_api_enter(__func__);
  ROCR_TRACE_API_RET_STATUS(amdExtTable->hsa_amd_memory_get_preferred_copy_engine_fn(
      dst_agent, src_agent, recommended_ids_mask));
}

// Mirrors Amd Extension Apis
hsa_status_t HSA_API hsa_amd_memory_async_copy_rect(
    const hsa_pitched_ptr_t* dst, const hsa_dim3_t* dst_offset, const hsa_pitched_ptr_t* src,
    const hsa_dim3_t* src_offset, const hsa_dim3_t* range, hsa_agent_t copy_agent,
    hsa_amd_copy_direction_t dir, uint32_t num_dep_signals, const hsa_signal_t* dep_signals,
    hsa_signal_t completion_signal) {
  rocm_trace_emit_hsa_api_enter(__func__);
  ROCR_TRACE_API_RET_STATUS(amdExtTable->hsa_amd_memory_async_copy_rect_fn(
      dst, dst_offset, src, src_offset, range, copy_agent, dir, num_dep_signals, dep_signals,
      completion_signal));
}

// Mirrors Amd Extension Apis
hsa_status_t HSA_API hsa_amd_agent_memory_pool_get_info(hsa_agent_t agent,
                                                        hsa_amd_memory_pool_t memory_pool,
                                                        hsa_amd_agent_memory_pool_info_t attribute,
                                                        void* value) {
  rocm_trace_emit_hsa_api_enter(__func__);
  ROCR_TRACE_API_RET_STATUS(
      amdExtTable->hsa_amd_agent_memory_pool_get_info_fn(agent, memory_pool, attribute, value));
}

// Mirrors Amd Extension Apis
hsa_status_t HSA_API hsa_amd_agents_allow_access(uint32_t num_agents, const hsa_agent_t* agents,
                                                 const uint32_t* flags, const void* ptr) {
  rocm_trace_emit_hsa_api_enter(__func__);
  ROCR_TRACE_API_RET_STATUS(
      amdExtTable->hsa_amd_agents_allow_access_fn(num_agents, agents, flags, ptr));
}

// Mirrors Amd Extension Apis
hsa_status_t HSA_API hsa_amd_memory_pool_can_migrate(hsa_amd_memory_pool_t src_memory_pool,
                                                     hsa_amd_memory_pool_t dst_memory_pool,
                                                     bool* result) {
  rocm_trace_emit_hsa_api_enter(__func__);
  ROCR_TRACE_API_RET_STATUS(
      amdExtTable->hsa_amd_memory_pool_can_migrate_fn(src_memory_pool, dst_memory_pool, result));
}

// Mirrors Amd Extension Apis
hsa_status_t HSA_API hsa_amd_memory_migrate(const void* ptr, hsa_amd_memory_pool_t memory_pool,
                                            uint32_t flags) {
  rocm_trace_emit_hsa_api_enter(__func__);
  ROCR_TRACE_API_RET_STATUS(amdExtTable->hsa_amd_memory_migrate_fn(ptr, memory_pool, flags));
}

// Mirrors Amd Extension Apis
hsa_status_t HSA_API hsa_amd_memory_lock(void* host_ptr, size_t size, hsa_agent_t* agents,
                                         int num_agent, void** agent_ptr) {
  rocm_trace_emit_hsa_api_enter(__func__);
  ROCR_TRACE_API_RET_STATUS(
      amdExtTable->hsa_amd_memory_lock_fn(host_ptr, size, agents, num_agent, agent_ptr));
}

// Mirrors Amd Extension Apis
hsa_status_t HSA_API hsa_amd_memory_lock_to_pool(void* host_ptr, size_t size, hsa_agent_t* agents,
                                                 int num_agent, hsa_amd_memory_pool_t pool,
                                                 uint32_t flags, void** agent_ptr) {
  rocm_trace_emit_hsa_api_enter(__func__);
  ROCR_TRACE_API_RET_STATUS(amdExtTable->hsa_amd_memory_lock_to_pool_fn(
      host_ptr, size, agents, num_agent, pool, flags, agent_ptr));
}

// Mirrors Amd Extension Apis
hsa_status_t HSA_API hsa_amd_memory_unlock(void* host_ptr) {
  rocm_trace_emit_hsa_api_enter(__func__);
  ROCR_TRACE_API_RET_STATUS(amdExtTable->hsa_amd_memory_unlock_fn(host_ptr));
}

// Mirrors Amd Extension Apis
hsa_status_t HSA_API hsa_amd_memory_fill(void* ptr, uint32_t value, size_t count) {
  rocm_trace_emit_hsa_api_enter(__func__);
  ROCR_TRACE_API_RET_STATUS(amdExtTable->hsa_amd_memory_fill_fn(ptr, value, count));
}

// Mirrors Amd Extension Apis
hsa_status_t HSA_API hsa_amd_interop_map_buffer(uint32_t num_agents, hsa_agent_t* agents,
                                                hsa_handle_t interop_handle, uint32_t flags,
                                                size_t* size, void** ptr, size_t* metadata_size,
                                                const void** metadata) {
  rocm_trace_emit_hsa_api_enter(__func__);
  ROCR_TRACE_API_RET_STATUS(amdExtTable->hsa_amd_interop_map_buffer_fn(
      num_agents, agents, interop_handle, flags, size, ptr, metadata_size, metadata));
}

hsa_status_t HSA_API hsa_amd_interop_map_buffer_with_size(
    uint32_t num_agents, hsa_agent_t* agents, hsa_handle_t interop_handle, uint32_t flags,
    size_t size_hint, size_t* size, void** ptr, size_t* metadata_size, const void** metadata) {
  rocm_trace_emit_hsa_api_enter(__func__);
  ROCR_TRACE_API_RET_STATUS(amdExtTable->hsa_amd_interop_map_buffer_with_size_fn(
      num_agents, agents, interop_handle, flags, size_hint, size, ptr, metadata_size, metadata));
}

// Mirrors Amd Extension Apis
hsa_status_t HSA_API hsa_amd_interop_unmap_buffer(void* ptr) {
  rocm_trace_emit_hsa_api_enter(__func__);
  ROCR_TRACE_API_RET_STATUS(amdExtTable->hsa_amd_interop_unmap_buffer_fn(ptr));
}

// Mirrors Amd Extension Apis
hsa_status_t HSA_API hsa_amd_image_create(hsa_agent_t agent,
                                          const hsa_ext_image_descriptor_t* image_descriptor,
                                          const hsa_amd_image_descriptor_t* image_layout,
                                          const void* image_data,
                                          hsa_access_permission_t access_permission,
                                          hsa_ext_image_t* image) {
  rocm_trace_emit_hsa_api_enter(__func__);
  ROCR_TRACE_API_RET_STATUS(amdExtTable->hsa_amd_image_create_fn(
      agent, image_descriptor, image_layout, image_data, access_permission, image));
}

// Mirrors Amd Extension Apis
hsa_status_t HSA_API hsa_amd_image_create_v2(hsa_agent_t agent,
                                             const hsa_ext_image_descriptor_v2_t* image_descriptor,
                                             const hsa_amd_image_descriptor_t* image_layout,
                                             const void* image_data,
                                             hsa_access_permission_t access_permission,
                                             hsa_ext_image_t* image) {
  rocm_trace_emit_hsa_api_enter(__func__);
  ROCR_TRACE_API_RET_STATUS(amdExtTable->hsa_amd_image_create_v2_fn(
      agent, image_descriptor, image_layout, image_data, access_permission, image));
}

// Mirrors Amd Extension Apis
hsa_status_t hsa_amd_pointer_info(const void* ptr, hsa_amd_pointer_info_t* info,
                                  void* (*alloc)(size_t), uint32_t* num_agents_accessible,
                                  hsa_agent_t** accessible) {
  rocm_trace_emit_hsa_api_enter(__func__);
  ROCR_TRACE_API_RET_STATUS(
      amdExtTable->hsa_amd_pointer_info_fn(ptr, info, alloc, num_agents_accessible, accessible));
}

// Mirrors Amd Extension Apis
hsa_status_t hsa_amd_pointer_info_set_userdata(const void* ptr, void* userptr) {
  rocm_trace_emit_hsa_api_enter(__func__);
  ROCR_TRACE_API_RET_STATUS(amdExtTable->hsa_amd_pointer_info_set_userdata_fn(ptr, userptr));
}

// Mirrors Amd Extension Apis
hsa_status_t hsa_amd_ipc_memory_create(void* ptr, size_t len, hsa_amd_ipc_memory_t* handle) {
  rocm_trace_emit_hsa_api_enter(__func__);
  ROCR_TRACE_API_RET_STATUS(amdExtTable->hsa_amd_ipc_memory_create_fn(ptr, len, handle));
}

// Mirrors Amd Extension Apis
hsa_status_t hsa_amd_ipc_memory_attach(const hsa_amd_ipc_memory_t* ipc, size_t len,
                                       uint32_t num_agents, const hsa_agent_t* mapping_agents,
                                       void** mapped_ptr) {
  rocm_trace_emit_hsa_api_enter(__func__);
  ROCR_TRACE_API_RET_STATUS(
      amdExtTable->hsa_amd_ipc_memory_attach_fn(ipc, len, num_agents, mapping_agents, mapped_ptr));
}

// Mirrors Amd Extension Apis
hsa_status_t hsa_amd_ipc_memory_detach(void* mapped_ptr) {
  rocm_trace_emit_hsa_api_enter(__func__);
  ROCR_TRACE_API_RET_STATUS(amdExtTable->hsa_amd_ipc_memory_detach_fn(mapped_ptr));
}

// Mirrors Amd Extension Apis
hsa_status_t hsa_amd_signal_create(hsa_signal_value_t initial_value, uint32_t num_consumers,
                                   const hsa_agent_t* consumers, uint64_t attributes,
                                   hsa_signal_t* signal) {
  rocm_trace_emit_hsa_api_enter(__func__); /* __ROCM_CURATED__: hsa_amd_signal_create */
  auto const __rocm_in_initial_value = initial_value;
  auto const __rocm_in_num_consumers = num_consumers;
  auto const __rocm_in_consumers = consumers;
  auto const __rocm_in_attributes = attributes;
  ROCR_TRACE_API_RET_STATUS_CURATED_HSA(
      hsa_amd_signal_create,
      amdExtTable->hsa_amd_signal_create_fn(initial_value, num_consumers, consumers, attributes,
                                            signal),
      (__rocm_in_initial_value), (__rocm_in_num_consumers),
      (const void*)(uintptr_t)(__rocm_in_consumers), (__rocm_in_attributes), signal);
}

// Mirrors Amd Extension Apis
hsa_status_t HSA_API hsa_amd_ipc_signal_create(hsa_signal_t signal, hsa_amd_ipc_signal_t* handle) {
  rocm_trace_emit_hsa_api_enter(__func__);
  ROCR_TRACE_API_RET_STATUS(amdExtTable->hsa_amd_ipc_signal_create_fn(signal, handle));
}

// Mirrors Amd Extension Apis
hsa_status_t HSA_API hsa_amd_ipc_signal_attach(const hsa_amd_ipc_signal_t* handle,
                                               hsa_signal_t* signal) {
  rocm_trace_emit_hsa_api_enter(__func__);
  ROCR_TRACE_API_RET_STATUS(amdExtTable->hsa_amd_ipc_signal_attach_fn(handle, signal));
}

// Mirrors Amd Extension Apis
hsa_status_t HSA_API hsa_amd_register_system_event_handler(hsa_amd_system_event_callback_t callback,
                                                           void* data) {
  rocm_trace_emit_hsa_api_enter(__func__);
  ROCR_TRACE_API_RET_STATUS(amdExtTable->hsa_amd_register_system_event_handler_fn(callback, data));
}

// Mirrors Amd Extension Apis
hsa_status_t HSA_API hsa_amd_queue_set_priority(hsa_queue_t* queue,
                                                hsa_amd_queue_priority_t priority) {
  rocm_trace_emit_hsa_api_enter(__func__);
  ROCR_TRACE_API_RET_STATUS(amdExtTable->hsa_amd_queue_set_priority_fn(queue, priority));
}

// Mirrors Amd Extension Apis
hsa_status_t HSA_API hsa_amd_queue_create(hsa_agent_t agent, hsa_amd_queue_create_desc_t* descs,
                                          uint32_t num_descs) {
  rocm_trace_emit_hsa_api_enter(__func__);
  ROCR_TRACE_API_RET_STATUS(amdExtTable->hsa_amd_queue_create_fn(agent, descs, num_descs));
}

// Mirrors Amd Extension Apis
hsa_status_t HSA_API hsa_amd_register_deallocation_callback(
    void* ptr, hsa_amd_deallocation_callback_t callback, void* user_data) {
  rocm_trace_emit_hsa_api_enter(__func__);
  ROCR_TRACE_API_RET_STATUS(
      amdExtTable->hsa_amd_register_deallocation_callback_fn(ptr, callback, user_data));
}

// Mirrors Amd Extension Apis
hsa_status_t HSA_API
hsa_amd_deregister_deallocation_callback(void* ptr, hsa_amd_deallocation_callback_t callback) {
  rocm_trace_emit_hsa_api_enter(__func__);
  ROCR_TRACE_API_RET_STATUS(
      amdExtTable->hsa_amd_deregister_deallocation_callback_fn(ptr, callback));
}

// Mirrors Amd Extension Apis
hsa_status_t HSA_API hsa_amd_signal_value_pointer(hsa_signal_t signal,
                                                  volatile hsa_signal_value_t** value_ptr) {
  rocm_trace_emit_hsa_api_enter(__func__);
  ROCR_TRACE_API_RET_STATUS(amdExtTable->hsa_amd_signal_value_pointer_fn(signal, value_ptr));
}

// Mirrors Amd Extension Apis
hsa_status_t HSA_API hsa_amd_svm_attributes_set(void* ptr, size_t size,
                                                hsa_amd_svm_attribute_pair_t* attribute_list,
                                                size_t attribute_count) {
  rocm_trace_emit_hsa_api_enter(__func__);
  ROCR_TRACE_API_RET_STATUS(
      amdExtTable->hsa_amd_svm_attributes_set_fn(ptr, size, attribute_list, attribute_count));
}

// Mirrors Amd Extension Apis
hsa_status_t HSA_API hsa_amd_svm_attributes_get(void* ptr, size_t size,
                                                hsa_amd_svm_attribute_pair_t* attribute_list,
                                                size_t attribute_count) {
  rocm_trace_emit_hsa_api_enter(__func__);
  ROCR_TRACE_API_RET_STATUS(
      amdExtTable->hsa_amd_svm_attributes_get_fn(ptr, size, attribute_list, attribute_count));
}

// Mirrors Amd Extension Apis
hsa_status_t HSA_API hsa_amd_svm_prefetch_async(void* ptr, size_t size, hsa_agent_t agent,
                                                uint32_t num_dep_signals,
                                                const hsa_signal_t* dep_signals,
                                                hsa_signal_t completion_signal) {
  rocm_trace_emit_hsa_api_enter(__func__);
  ROCR_TRACE_API_RET_STATUS(amdExtTable->hsa_amd_svm_prefetch_async_fn(
      ptr, size, agent, num_dep_signals, dep_signals, completion_signal));
}

// Mirrors Amd Extension Apis
hsa_status_t HSA_API hsa_amd_spm_acquire(hsa_agent_t agent) {
  rocm_trace_emit_hsa_api_enter(__func__);
  ROCR_TRACE_API_RET_STATUS(amdExtTable->hsa_amd_spm_acquire_fn(agent));
}

// Mirrors Amd Extension Apis
hsa_status_t HSA_API hsa_amd_spm_release(hsa_agent_t agent) {
  rocm_trace_emit_hsa_api_enter(__func__);
  ROCR_TRACE_API_RET_STATUS(amdExtTable->hsa_amd_spm_release_fn(agent));
}

// Mirrors Amd Extension Apis
hsa_status_t HSA_API hsa_amd_spm_set_dest_buffer(hsa_agent_t agent, size_t size, uint32_t* timeout,
                                                 uint32_t* size_copied, void* dest,
                                                 bool* is_data_loss) {
  rocm_trace_emit_hsa_api_enter(__func__);
  ROCR_TRACE_API_RET_STATUS(amdExtTable->hsa_amd_spm_set_dest_buffer_fn(
      agent, size, timeout, size_copied, dest, is_data_loss));
}

// Mirrors Amd Extension Apis
hsa_status_t HSA_API hsa_amd_portable_export_dmabuf(const void* ptr, size_t size, int* dmabuf,
                                                    uint64_t* offset) {
  rocm_trace_emit_hsa_api_enter(__func__);
  ROCR_TRACE_API_RET_STATUS(
      amdExtTable->hsa_amd_portable_export_dmabuf_fn(ptr, size, dmabuf, offset));
}

// Mirrors Amd Extension Apis
hsa_status_t HSA_API hsa_amd_portable_export_dmabuf_v2(const void* ptr, size_t size, int* dmabuf,
                                                       uint64_t* offset, uint64_t flags) {
  rocm_trace_emit_hsa_api_enter(__func__);
  ROCR_TRACE_API_RET_STATUS(
      amdExtTable->hsa_amd_portable_export_dmabuf_v2_fn(ptr, size, dmabuf, offset, flags));
}

// Mirrors Amd Extension Apis
hsa_status_t HSA_API hsa_amd_portable_close_dmabuf(int dmabuf) {
  rocm_trace_emit_hsa_api_enter(__func__);
  ROCR_TRACE_API_RET_STATUS(amdExtTable->hsa_amd_portable_close_dmabuf_fn(dmabuf));
}

hsa_status_t HSA_API hsa_amd_vmem_address_reserve(void** ptr, size_t size, uint64_t address,
                                                  uint64_t flags) {
  rocm_trace_emit_hsa_api_enter(__func__);
  ROCR_TRACE_API_RET_STATUS(
      amdExtTable->hsa_amd_vmem_address_reserve_fn(ptr, size, address, flags));
}

hsa_status_t HSA_API hsa_amd_vmem_address_reserve_align(void** ptr, size_t size, uint64_t address,
                                                        uint64_t alignment, uint64_t flags) {
  rocm_trace_emit_hsa_api_enter(__func__);
  ROCR_TRACE_API_RET_STATUS(
      amdExtTable->hsa_amd_vmem_address_reserve_align_fn(ptr, size, address, alignment, flags));
}

hsa_status_t HSA_API hsa_amd_vmem_address_free(void* ptr, size_t size) {
  rocm_trace_emit_hsa_api_enter(__func__);
  ROCR_TRACE_API_RET_STATUS(amdExtTable->hsa_amd_vmem_address_free_fn(ptr, size));
}

hsa_status_t HSA_API hsa_amd_vmem_handle_create(hsa_amd_memory_pool_t pool, size_t size,
                                                hsa_amd_memory_type_t type, uint64_t flags,
                                                hsa_amd_vmem_alloc_handle_t* memory_handle) {
  rocm_trace_emit_hsa_api_enter(__func__);
  ROCR_TRACE_API_RET_STATUS(
      amdExtTable->hsa_amd_vmem_handle_create_fn(pool, size, type, flags, memory_handle));
}

hsa_status_t HSA_API hsa_amd_vmem_handle_release(hsa_amd_vmem_alloc_handle_t memory_handle) {
  rocm_trace_emit_hsa_api_enter(__func__);
  ROCR_TRACE_API_RET_STATUS(amdExtTable->hsa_amd_vmem_handle_release_fn(memory_handle));
}

hsa_status_t HSA_API hsa_amd_vmem_map(void* va, size_t size, size_t in_offset,
                                      hsa_amd_vmem_alloc_handle_t memory_handle, uint64_t flags) {
  rocm_trace_emit_hsa_api_enter(__func__);
  ROCR_TRACE_API_RET_STATUS(
      amdExtTable->hsa_amd_vmem_map_fn(va, size, in_offset, memory_handle, flags));
}

hsa_status_t HSA_API hsa_amd_vmem_unmap(void* va, size_t size) {
  rocm_trace_emit_hsa_api_enter(__func__);
  ROCR_TRACE_API_RET_STATUS(amdExtTable->hsa_amd_vmem_unmap_fn(va, size));
}

hsa_status_t HSA_API hsa_amd_vmem_set_access(void* va, size_t size,
                                             const hsa_amd_memory_access_desc_t* desc,
                                             const size_t desc_cnt) {
  rocm_trace_emit_hsa_api_enter(__func__);
  ROCR_TRACE_API_RET_STATUS(amdExtTable->hsa_amd_vmem_set_access_fn(va, size, desc, desc_cnt));
}

hsa_status_t HSA_API hsa_amd_vmem_get_access(void* va, hsa_access_permission_t* perms,
                                             const hsa_agent_t agent_handle) {
  rocm_trace_emit_hsa_api_enter(__func__);
  ROCR_TRACE_API_RET_STATUS(amdExtTable->hsa_amd_vmem_get_access_fn(va, perms, agent_handle));
}

hsa_status_t HSA_API hsa_amd_vmem_export_shareable_handle(int* dmabuf_fd,
                                                          hsa_amd_vmem_alloc_handle_t handle,
                                                          uint64_t flags) {
  rocm_trace_emit_hsa_api_enter(__func__);
  ROCR_TRACE_API_RET_STATUS(
      amdExtTable->hsa_amd_vmem_export_shareable_handle_fn(dmabuf_fd, handle, flags));
}

hsa_status_t HSA_API hsa_amd_vmem_import_shareable_handle(int dmabuf_fd,
                                                          hsa_amd_vmem_alloc_handle_t* handle) {
  rocm_trace_emit_hsa_api_enter(__func__);
  ROCR_TRACE_API_RET_STATUS(
      amdExtTable->hsa_amd_vmem_import_shareable_handle_fn(dmabuf_fd, handle));
}

hsa_status_t HSA_API hsa_amd_vmem_retain_alloc_handle(hsa_amd_vmem_alloc_handle_t* handle,
                                                      void* addr) {
  rocm_trace_emit_hsa_api_enter(__func__);
  ROCR_TRACE_API_RET_STATUS(amdExtTable->hsa_amd_vmem_retain_alloc_handle_fn(handle, addr));
}

hsa_status_t HSA_API hsa_amd_vmem_export_fabric_handle(hsa_fabric_handle_t* fabric_handle,
                                                       hsa_amd_vmem_alloc_handle_t handle,
                                                       uint64_t flags) {
  rocm_trace_emit_hsa_api_enter(__func__);
  ROCR_TRACE_API_RET_STATUS(
      amdExtTable->hsa_amd_vmem_export_fabric_handle_fn(fabric_handle, handle, flags));
}

hsa_status_t HSA_API hsa_amd_vmem_import_fabric_handle(hsa_fabric_handle_t fabric_handle,
                                                       hsa_amd_vmem_alloc_handle_t* handle) {
  rocm_trace_emit_hsa_api_enter(__func__);
  ROCR_TRACE_API_RET_STATUS(
      amdExtTable->hsa_amd_vmem_import_fabric_handle_fn(fabric_handle, handle));
}

hsa_status_t HSA_API hsa_amd_vmem_get_alloc_properties_from_handle(
    hsa_amd_vmem_alloc_handle_t alloc_handle, hsa_amd_memory_pool_t* pool,
    hsa_amd_memory_type_t* type) {
  rocm_trace_emit_hsa_api_enter(__func__);
  ROCR_TRACE_API_RET_STATUS(
      amdExtTable->hsa_amd_vmem_get_alloc_properties_from_handle_fn(alloc_handle, pool, type));
}

hsa_status_t HSA_API hsa_amd_agent_set_async_scratch_limit(hsa_agent_t agent, size_t threshold) {
  rocm_trace_emit_hsa_api_enter(__func__);
  ROCR_TRACE_API_RET_STATUS(
      amdExtTable->hsa_amd_agent_set_async_scratch_limit_fn(agent, threshold));
}

hsa_status_t HSA_API hsa_amd_queue_get_info(hsa_queue_t* queue,
                                            hsa_queue_info_attribute_t attribute, void* value) {
  rocm_trace_emit_hsa_api_enter(__func__);
  ROCR_TRACE_API_RET_STATUS(amdExtTable->hsa_amd_queue_get_info_fn(queue, attribute, value));
}

hsa_status_t HSA_API hsa_amd_enable_logging(uint8_t* flags, void* file) {
  rocm_trace_emit_hsa_api_enter(__func__);
  ROCR_TRACE_API_RET_STATUS(amdExtTable->hsa_amd_enable_logging_fn(flags, file));
}

hsa_status_t HSA_API hsa_amd_ais_file_write(hsa_amd_ais_file_handle_t handle, void* devicePtr,
                                            uint64_t size, int64_t file_offset,
                                            uint64_t* size_copied, int32_t* status) {
  rocm_trace_emit_hsa_api_enter(__func__);
  ROCR_TRACE_API_RET_STATUS(amdExtTable->hsa_amd_ais_file_write_fn(
      handle, devicePtr, size, file_offset, size_copied, status));
}

hsa_status_t HSA_API hsa_amd_ais_file_read(hsa_amd_ais_file_handle_t handle, void* devicePtr,
                                           uint64_t size, int64_t file_offset,
                                           uint64_t* size_copied, int32_t* status) {
  rocm_trace_emit_hsa_api_enter(__func__);
  ROCR_TRACE_API_RET_STATUS(amdExtTable->hsa_amd_ais_file_read_fn(
      handle, devicePtr, size, file_offset, size_copied, status));
}

hsa_status_t HSA_API hsa_amd_counted_queue_acquire(
    hsa_agent_t agent, hsa_queue_type_t type, hsa_amd_queue_priority_t priority,
    void (*callback)(hsa_status_t status, hsa_queue_t* source, void* data), void* data,
    uint64_t flags, hsa_queue_t** queue) {
  rocm_trace_emit_hsa_api_enter(__func__);
  ROCR_TRACE_API_RET_STATUS(amdExtTable->hsa_amd_counted_queue_acquire_fn(
      agent, type, priority, callback, data, flags, queue));
}

hsa_status_t HSA_API hsa_amd_counted_queue_release(hsa_queue_t* queue) {
  rocm_trace_emit_hsa_api_enter(__func__);
  ROCR_TRACE_API_RET_STATUS(amdExtTable->hsa_amd_counted_queue_release_fn(queue));
}

hsa_status_t HSA_API hsa_amd_svm_discard_batch_async(void** ptrs, size_t* sizes, uint32_t count,
                                                     uint32_t num_dep_signals,
                                                     const hsa_signal_t* dep_signals,
                                                     hsa_signal_t completion_signal) {
  rocm_trace_emit_hsa_api_enter(__func__);
  ROCR_TRACE_API_RET_STATUS(amdExtTable->hsa_amd_svm_discard_batch_async_fn(
      ptrs, sizes, count, num_dep_signals, dep_signals, completion_signal));
}

hsa_status_t HSA_API hsa_amd_signal_get_event_id(hsa_signal_t signal, uint32_t* event_id) {
  rocm_trace_emit_hsa_api_enter(__func__);
  ROCR_TRACE_API_RET_STATUS(amdExtTable->hsa_amd_signal_get_event_id_fn(signal, event_id));
}

// Mirrors Amd Extension Apis
hsa_status_t HSA_API hsa_amd_external_semaphore_handle_open(
    hsa_agent_t agent, const hsa_amd_external_semaphore_handle_descriptor_t* desc,
    hsa_amd_external_semaphore_t* out_sem) {
  rocm_trace_emit_hsa_api_enter(__func__);
  ROCR_TRACE_API_RET_STATUS(
      amdExtTable->hsa_amd_external_semaphore_handle_open_fn(agent, desc, out_sem));
}

// Mirrors Amd Extension Apis
hsa_status_t HSA_API hsa_amd_external_semaphore_handle_close(hsa_amd_external_semaphore_t sem) {
  rocm_trace_emit_hsa_api_enter(__func__);
  ROCR_TRACE_API_RET_STATUS(amdExtTable->hsa_amd_external_semaphore_handle_close_fn(sem));
}

// Mirrors Amd Extension Apis
hsa_status_t HSA_API hsa_amd_queue_signal_external_semaphore(hsa_queue_t* queue,
                                                             hsa_amd_external_semaphore_t sem,
                                                             uint64_t value) {
  rocm_trace_emit_hsa_api_enter(__func__);
  ROCR_TRACE_API_RET_STATUS(
      amdExtTable->hsa_amd_queue_signal_external_semaphore_fn(queue, sem, value));
}

// Mirrors Amd Extension Apis
hsa_status_t HSA_API hsa_amd_queue_wait_external_semaphore(hsa_queue_t* queue,
                                                           hsa_amd_external_semaphore_t sem,
                                                           uint64_t value) {
  rocm_trace_emit_hsa_api_enter(__func__);
  ROCR_TRACE_API_RET_STATUS(
      amdExtTable->hsa_amd_queue_wait_external_semaphore_fn(queue, sem, value));
}
// Tools only table interfaces.
namespace rocr {

// Mirrors Amd Extension Apis
hsa_status_t hsa_amd_queue_intercept_create(
    hsa_agent_t agent_handle, uint32_t size, hsa_queue_type32_t type,
    void (*callback)(hsa_status_t status, hsa_queue_t* source, void* data), void* data,
    uint32_t private_segment_size, uint32_t group_segment_size, hsa_queue_t** queue) {
  rocm_trace_emit_hsa_api_enter(__func__); /* __ROCM_CURATED__: hsa_amd_queue_intercept_create */
  auto const __rocm_in_agent_handle = agent_handle;
  auto const __rocm_in_size = size;
  auto const __rocm_in_type = type;
  auto const __rocm_in_callback = callback;
  auto const __rocm_in_data = data;
  auto const __rocm_in_private_segment_size = private_segment_size;
  auto const __rocm_in_group_segment_size = group_segment_size;
  ROCR_TRACE_API_RET_STATUS_CURATED_HSA(
      hsa_amd_queue_intercept_create,
      amdExtTable->hsa_amd_queue_intercept_create_fn(agent_handle, size, type, callback, data,
                                                     private_segment_size, group_segment_size,
                                                     queue),
      (uint64_t)((__rocm_in_agent_handle).handle), (__rocm_in_size), (__rocm_in_type),
      (const void*)(uintptr_t)(__rocm_in_callback), (const void*)(uintptr_t)(__rocm_in_data),
      (__rocm_in_private_segment_size), (__rocm_in_group_segment_size), queue);
}

// Mirrors Amd Extension Apis
hsa_status_t hsa_amd_queue_intercept_register(hsa_queue_t* queue,
                                              hsa_amd_queue_intercept_handler callback,
                                              void* user_data) {
  rocm_trace_emit_hsa_api_enter(__func__);
  ROCR_TRACE_API_RET_STATUS(
      amdExtTable->hsa_amd_queue_intercept_register_fn(queue, callback, user_data));
}

}  // namespace rocr
