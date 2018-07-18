/******************************************************************************
MIT License

Copyright (c) 2018 ROCm Core Technology

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
*******************************************************************************/

#ifndef _SRC_CORE_HSA_INTERCEPTOR_H
#define _SRC_CORE_HSA_INTERCEPTOR_H

#include <amd_hsa_kernel_code.h>
#include <atomic>

#include "inc/rocprofiler.h"

namespace rocprofiler {
extern decltype(hsa_memory_allocate)* hsa_memory_allocate_fn;
extern decltype(hsa_memory_assign_agent)* hsa_memory_assign_agent_fn;
extern decltype(hsa_amd_memory_pool_allocate)* hsa_amd_memory_pool_allocate_fn;
extern decltype(hsa_memory_copy)* hsa_memory_copy_fn;
extern decltype(hsa_amd_memory_async_copy)* hsa_amd_memory_async_copy_fn;

class HsaInterceptor {
 public:
  static void Enable(const bool& enable) { enable_ = enable; }

  static void HsaIntercept(HsaApiTable* table) {
    if (enable_) {
      table->amd_ext_->hsa_memory_allocate_fn = MemoryAllocate;
      table->amd_ext_->hsa_memory_assign_agent_fn = MemoryAssignAgent;
      table->amd_ext_->hsa_amd_memory_pool_allocate_fn = MemoryPoolAllocate;;
      table->core_->hsa_memory_copy_fn = MemoryCopy;
      table->amd_ext_->hsa_amd_memory_async_copy_fn = MemoryAsyncCopy;
    }
  }

  static void SetHsaAllocCallback(rocprofiler_hsa_callback_fun_t fun, void* arg) {
    alloc_callback_arg_ = arg;
    alloc_callback_fun_.store(fun);
  }

  static void SetHsaMemcopyCallback(rocprofiler_hsa_callback_fun_t fun, void* arg) {
    memcopy_callback_arg_ = arg;
    memcopy_callback_fun_.store(fun);
  }

 private:
  static hsa_status_t HSA_API MemoryAllocate(hsa_region_t region,
    size_t size,
    void** ptr)
  {
    const hsa_status_t status = hsa_memory_allocate_fn(region, size, ptr);
    if ((status == HSA_STATUS_SUCCESS) && (alloc_callback_fun_ != NULL)) {
      rocprofiler_hsa_callback_data_t data{};
      data.hsa_alloc.addr = *ptr;
      data.hsa_alloc.size = size;
      data.hsa_alloc.device_type = HSA_DEVICE_TYPE_CPU;

      hsa_status_t err = hsa_region_get_info(region, HSA_REGION_INFO_SEGMENT, &data.hsa_alloc.segment);
      if (err != HSA_STATUS_SUCCESS) data.hsa_alloc.addr = NULL;
      err = hsa_region_get_info(region, HSA_REGION_INFO_GLOBAL_FLAGS, &data.hsa_alloc.global_flag);
      if (err != HSA_STATUS_SUCCESS) data.hsa_alloc.addr = NULL;

      const hsa_status_t ret = alloc_callback_fun_(ROCPROFILER_HSA_CB_ID_ALLOC, &data, alloc_callback_arg_);
      if (ret != HSA_STATUS_SUCCESS) memcopy_callback_fun_.store(NULL);
    }
    return status;
  }

  static hsa_status_t MemoryAssignAgent(
    void *ptr,
    hsa_agent_t agent,
    hsa_access_permission_t access)
  {
    rocprofiler_hsa_callback_data_t data{};
    data.hsa_alloc.addr = ptr;

    hsa_status_t err = hsa_agent_get_info(agent, HSA_AGENT_INFO_DEVICE, &data.hsa_alloc.device_type);
    if (err != HSA_STATUS_SUCCESS) data.hsa_alloc.addr = NULL;

    alloc_callback_fun_(ROCPROFILER_HSA_CB_ID_ASSIGN, &data, alloc_callback_arg_);
    return hsa_memory_assign_agent(ptr, agent, access);
  }

  static hsa_status_t MemoryPoolAllocate(
    hsa_amd_memory_pool_t memory_pool, size_t size,
    uint32_t flags, void** ptr)
  {
    const hsa_status_t status = hsa_amd_memory_pool_allocate_fn(memory_pool, size, flags, ptr);
    if ((status == HSA_STATUS_SUCCESS) && (alloc_callback_fun_ != NULL)) {
      rocprofiler_hsa_callback_data_t data{};
      data.alloc.addr = *ptr;
      data.alloc.size = size;
      data.hsa_alloc.device_type = HSA_DEVICE_TYPE_CPU;

      hsa_status_t err = hsa_amd_memory_pool_get_info(memory_pool, HSA_AMD_MEMORY_POOL_INFO_SEGMENT, &data.hsa_alloc.segment);
      if (err != HSA_STATUS_SUCCESS) data.pool_alloc.addr = NULL;
      hsa_status_t err = hsa_amd_memory_pool_get_info(memory_pool, HSA_AMD_MEMORY_POOL_INFO_GLOBAL_FLAGS, &data.hsa_alloc.global_flag);
      if (err != HSA_STATUS_SUCCESS) data.pool_alloc.addr = NULL;
      hsa_status_t err = hsa_amd_memory_pool_get_info(memory_pool, HSA_AMD_MEMORY_POOL_INFO_ACCESSIBLE_BY_ALL, &data.hsa_alloc.global_mem);
      if (err != HSA_STATUS_SUCCESS) data.pool_alloc.addr = NULL;

      const hsa_status_t ret = alloc_callback_fun_(ROCPROFILER_HSA_CB_ID_POOL_ALLOC, &data, memcopy_callback_arg_);
      if (ret != HSA_STATUS_SUCCESS) memcopy_callback_fun_.store(NULL);
    }
    return status;
  }

  static hsa_status_t MemoryCopy(
    void *dst,
    const void *src,
    size_t size)
  {
    if (memcopy_callback_fun_) {
      rocprofiler_hsa_callback_data_t data{};
      data.memcopy.dst = dst;
      data.memcopy.src = src;
      data.memcopy.size = size;
      const hsa_status_t ret = memcopy_callback_fun_(ROCPROFILER_HSA_CB_ID_MEMCOPY, &data, memcopy_callback_arg_);
      if (ret != HSA_STATUS_SUCCESS) memcopy_callback_fun_.store(NULL);
    }
    return hsa_memory_copy(dst, src, size);
  }

  static hsa_status_t MemoryAsyncCopy(
    void* dst, hsa_agent_t dst_agent, const void* src,
    hsa_agent_t src_agent, size_t size,
    uint32_t num_dep_signals,
    const hsa_signal_t* dep_signals,
    hsa_signal_t completion_signal)
  {
    if (memcopy_callback_fun_) {
      rocprofiler_hsa_callback_data_t data{};
      data.memcopy.dst = dst;
      data.memcopy.src = src;
      data.memcopy.size = size;
      const hsa_status_t ret = memcopy_callback_fun_(ROCPROFILER_HSA_CB_ID_MEMCOPY, &data, memcopy_callback_arg_);
      if (ret != HSA_STATUS_SUCCESS) memcopy_callback_fun_.store(NULL);
    }
    return hsa_amd_memory_async_copy_fn(
      dst, dst_agent, src, src_agent, size,
      num_dep_signals, dep_signals, completion_signal);
  }

  static bool enable_;
  static std::atomic<rocprofiler_hsa_callback_fun_t> alloc_callback_fun_;
  static void* alloc_callback_arg_;
  static std::atomic<rocprofiler_hsa_callback_fun_t> memcopy_callback_fun_;
  static void* memcopy_callback_arg_;
};

}  // namespace rocprofiler

#endif  // _SRC_CORE_HSA_INTERCEPTOR_H
