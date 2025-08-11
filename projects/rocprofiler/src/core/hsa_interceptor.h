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

#include <cxxabi.h>
#include <hsa/hsa.h>
#include <hsa/hsa_ext_amd.h>
#include <hsa/hsa_ven_amd_loader.h>

#include <atomic>
#include <mutex>

#include "rocprofiler.h"
#include "util/exception.h"
#include "util/hsa_rsrc_factory.h"

#define HSA_RT(call)                                                                               \
  do {                                                                                             \
    const hsa_status_t status = call;                                                              \
    if (status != HSA_STATUS_SUCCESS) EXC_ABORT(status, #call);                                    \
  } while (0)

#define IS_HSA_CALLBACK(ID)                                                                        \
  const auto __id = ID;                                                                            \
  (void)__id;                                                                                      \
  void* __arg = arg_.load();                                                                       \
  (void)__arg;                                                                                     \
  rocprofiler_hsa_callback_fun_t __callback = (ID == ROCPROFILER_HSA_CB_ID_ALLOCATE)               \
      ? callbacks_.allocate                                                                        \
      : (ID == ROCPROFILER_HSA_CB_ID_DEVICE)  ? callbacks_.device                                  \
      : (ID == ROCPROFILER_HSA_CB_ID_MEMCOPY) ? callbacks_.memcopy                                 \
      : (ID == ROCPROFILER_HSA_CB_ID_SUBMIT)  ? callbacks_.submit                                  \
      : (ID == ROCPROFILER_HSA_CB_ID_KSYMBOL) ? callbacks_.ksymbol                                 \
                                              : callbacks_.codeobj;                                \
  if ((__callback != NULL) && (recursion_ == false))

#define DO_HSA_CALLBACK                                                                            \
  do {                                                                                             \
    recursion_ = true;                                                                             \
    __callback(__id, &data, __arg);                                                                \
    recursion_ = false;                                                                            \
  } while (0)

#define ISSUE_HSA_CALLBACK(ID)                                                                     \
  do {                                                                                             \
    IS_HSA_CALLBACK(ID) { DO_HSA_CALLBACK; }                                                       \
  } while (0)

// Demangle C++ symbol name
static const char* cpp_demangle(const char* symname) {
  size_t size = 0;
  int status;
  const char* ret = abi::__cxa_demangle(symname, NULL, &size, &status);
  return (ret != 0) ? ret : strdup(symname);
}

namespace rocprofiler {
extern decltype(::hsa_memory_allocate)* hsa_memory_allocate_fn;
extern decltype(::hsa_memory_assign_agent)* hsa_memory_assign_agent_fn;
extern decltype(::hsa_memory_copy)* hsa_memory_copy_fn;
extern decltype(::hsa_amd_memory_pool_allocate)* hsa_amd_memory_pool_allocate_fn;
extern decltype(::hsa_amd_memory_pool_free)* hsa_amd_memory_pool_free_fn;
extern decltype(::hsa_amd_agents_allow_access)* hsa_amd_agents_allow_access_fn;
extern decltype(::hsa_amd_memory_async_copy)* hsa_amd_memory_async_copy_fn;
extern decltype(::hsa_executable_freeze)* hsa_executable_freeze_fn;
extern decltype(::hsa_executable_destroy)* hsa_executable_destroy_fn;

class HsaInterceptor {
 public:
  typedef std::atomic<void*> arg_t;
  typedef std::mutex mutex_t;

  static void Enable(const bool& enable) { enable_ = enable; }

  static void HsaIntercept(HsaApiTable* table) {
    if (enable_) {
      // Fetching AMD Loader HSA extension API
      HSA_RT(hsa_system_get_major_extension_table(
          HSA_EXTENSION_AMD_LOADER, 1, sizeof(hsa_ven_amd_loader_1_01_pfn_t), &LoaderApiTable));

      // Saving original API functions
      hsa_memory_allocate_fn = table->core_->hsa_memory_allocate_fn;
      hsa_memory_assign_agent_fn = table->core_->hsa_memory_assign_agent_fn;
      hsa_memory_copy_fn = table->core_->hsa_memory_copy_fn;
      hsa_amd_memory_pool_allocate_fn = table->amd_ext_->hsa_amd_memory_pool_allocate_fn;
      hsa_amd_memory_pool_free_fn = table->amd_ext_->hsa_amd_memory_pool_free_fn;
      hsa_amd_agents_allow_access_fn = table->amd_ext_->hsa_amd_agents_allow_access_fn;
      hsa_amd_memory_async_copy_fn = table->amd_ext_->hsa_amd_memory_async_copy_fn;
      hsa_executable_freeze_fn = table->core_->hsa_executable_freeze_fn;
      hsa_executable_destroy_fn = table->core_->hsa_executable_destroy_fn;

      // Intercepting HSA API
      table->core_->hsa_memory_allocate_fn = MemoryAllocate;
      table->core_->hsa_memory_assign_agent_fn = MemoryAssignAgent;
      table->core_->hsa_memory_copy_fn = MemoryCopy;
      table->amd_ext_->hsa_amd_memory_pool_allocate_fn = MemoryPoolAllocate;
      table->amd_ext_->hsa_amd_memory_pool_free_fn = MemoryPoolFree;
      table->amd_ext_->hsa_amd_agents_allow_access_fn = AgentsAllowAccess;
      table->amd_ext_->hsa_amd_memory_async_copy_fn = MemoryAsyncCopy;
      table->core_->hsa_executable_freeze_fn = ExecutableFreeze;
      table->core_->hsa_executable_destroy_fn = ExecutableDestroy;
    }
  }

  static void SetCallbacks(rocprofiler_hsa_callbacks_t callbacks, void* arg) {
    std::lock_guard<mutex_t> lck(mutex_);
    callbacks_ = callbacks;
    arg_.store(arg);
  }

 private:
  static hsa_status_t HSA_API MemoryAllocate(hsa_region_t region, size_t size, void** ptr) {
    hsa_status_t status = HSA_STATUS_SUCCESS;
    HSA_RT(hsa_memory_allocate_fn(region, size, ptr));
    IS_HSA_CALLBACK(ROCPROFILER_HSA_CB_ID_ALLOCATE) {
      rocprofiler_hsa_callback_data_t data{};
      data.allocate.ptr = *ptr;
      data.allocate.size = size;

      HSA_RT(hsa_region_get_info(region, HSA_REGION_INFO_SEGMENT, &data.allocate.segment));
      HSA_RT(hsa_region_get_info(region, HSA_REGION_INFO_GLOBAL_FLAGS, &data.allocate.global_flag));

      DO_HSA_CALLBACK;
    }
    return status;
  }

  static hsa_status_t MemoryAssignAgent(void* ptr, hsa_agent_t agent,
                                        hsa_access_permission_t access) {
    hsa_status_t status = HSA_STATUS_SUCCESS;
    HSA_RT(hsa_memory_assign_agent_fn(ptr, agent, access));
    IS_HSA_CALLBACK(ROCPROFILER_HSA_CB_ID_DEVICE) {
      rocprofiler_hsa_callback_data_t data{};
      data.device.ptr = ptr;

      HSA_RT(hsa_agent_get_info(agent, HSA_AGENT_INFO_DEVICE, &data.device.type));

      DO_HSA_CALLBACK;
    }
    return status;
  }

  // Spawn device allow access callback
  static void DeviceCallback(uint32_t num_agents, const hsa_agent_t* agents, const void* ptr) {
    for (const hsa_agent_t* agent_p = agents; agent_p < (agents + num_agents); ++agent_p) {
      hsa_agent_t agent = *agent_p;
      rocprofiler_hsa_callback_data_t data{};
      data.device.id = util::HsaRsrcFactory::Instance().GetAgentInfo(agent)->dev_index;
      data.device.agent = agent;
      data.device.ptr = ptr;

      HSA_RT(hsa_agent_get_info(agent, HSA_AGENT_INFO_DEVICE, &data.device.type));

      ISSUE_HSA_CALLBACK(ROCPROFILER_HSA_CB_ID_DEVICE);
    }
  }

  // Agent allow access callback 'hsa_amd_agents_allow_access'
  static hsa_status_t AgentsAllowAccess(uint32_t num_agents, const hsa_agent_t* agents,
                                        const uint32_t* flags, const void* ptr) {
    hsa_status_t status = HSA_STATUS_SUCCESS;
    HSA_RT(hsa_amd_agents_allow_access_fn(num_agents, agents, flags, ptr));
    IS_HSA_CALLBACK(ROCPROFILER_HSA_CB_ID_DEVICE) { DeviceCallback(num_agents, agents, ptr); }
    return status;
  }

  // Callback function to get available in the system agents
  struct agent_callback_data_t {
    hsa_amd_memory_pool_t pool;
    void* ptr;
  };
  static hsa_status_t AgentCallback(hsa_agent_t agent, void* data) {
    agent_callback_data_t* callback_data = reinterpret_cast<agent_callback_data_t*>(data);
    hsa_amd_agent_memory_pool_info_t attribute = HSA_AMD_AGENT_MEMORY_POOL_INFO_ACCESS;
    hsa_amd_memory_pool_access_t value;
    HSA_RT(hsa_amd_agent_memory_pool_get_info(agent, callback_data->pool, attribute, &value));
    if (value == HSA_AMD_MEMORY_POOL_ACCESS_ALLOWED_BY_DEFAULT) {
      DeviceCallback(1, &agent, callback_data->ptr);
    }
    return HSA_STATUS_SUCCESS;
  }

  static hsa_status_t MemoryPoolAllocate(hsa_amd_memory_pool_t pool, size_t size, uint32_t flags,
                                         void** ptr) {
    hsa_status_t status = HSA_STATUS_SUCCESS;
    HSA_RT(hsa_amd_memory_pool_allocate_fn(pool, size, flags, ptr));
    if (size != 0) {
      IS_HSA_CALLBACK(ROCPROFILER_HSA_CB_ID_ALLOCATE) {
        rocprofiler_hsa_callback_data_t data{};
        data.allocate.ptr = *ptr;
        data.allocate.size = size;

        HSA_RT(hsa_amd_memory_pool_get_info(pool, HSA_AMD_MEMORY_POOL_INFO_SEGMENT,
                                            &data.allocate.segment));
        HSA_RT(hsa_amd_memory_pool_get_info(pool, HSA_AMD_MEMORY_POOL_INFO_GLOBAL_FLAGS,
                                            &data.allocate.global_flag));

        DO_HSA_CALLBACK;

        IS_HSA_CALLBACK(ROCPROFILER_HSA_CB_ID_DEVICE) {
          // Scan the pool assigned devices
          agent_callback_data_t callback_data{pool, *ptr};
          hsa_iterate_agents(AgentCallback, &callback_data);
        }
      }
    }
    return status;
  }
  static hsa_status_t MemoryPoolFree(void* ptr) {
    hsa_status_t status = HSA_STATUS_SUCCESS;
    IS_HSA_CALLBACK(ROCPROFILER_HSA_CB_ID_ALLOCATE) {
      rocprofiler_hsa_callback_data_t data{};
      data.allocate.ptr = ptr;
      data.allocate.size = 0;
      DO_HSA_CALLBACK;
    }
    HSA_RT(hsa_amd_memory_pool_free_fn(ptr));
    return status;
  }

  static hsa_status_t MemoryCopy(void* dst, const void* src, size_t size) {
    hsa_status_t status = HSA_STATUS_SUCCESS;
    HSA_RT(hsa_memory_copy_fn(dst, src, size));
    IS_HSA_CALLBACK(ROCPROFILER_HSA_CB_ID_MEMCOPY) {
      rocprofiler_hsa_callback_data_t data{};
      data.memcopy.dst = dst;
      data.memcopy.src = src;
      data.memcopy.size = size;
      DO_HSA_CALLBACK;
    }
    return status;
  }

  static hsa_status_t MemoryAsyncCopy(void* dst, hsa_agent_t dst_agent, const void* src,
                                      hsa_agent_t src_agent, size_t size, uint32_t num_dep_signals,
                                      const hsa_signal_t* dep_signals,
                                      hsa_signal_t completion_signal) {
    hsa_status_t status = HSA_STATUS_SUCCESS;
    HSA_RT(hsa_amd_memory_async_copy_fn(dst, dst_agent, src, src_agent, size, num_dep_signals,
                                        dep_signals, completion_signal));
    IS_HSA_CALLBACK(ROCPROFILER_HSA_CB_ID_MEMCOPY) {
      rocprofiler_hsa_callback_data_t data{};
      data.memcopy.dst = dst;
      data.memcopy.src = src;
      data.memcopy.size = size;
      DO_HSA_CALLBACK;
    }
    return status;
  }

  static hsa_status_t CodeObjectCallback(hsa_executable_t executable,
                                         hsa_loaded_code_object_t loaded_code_object, void* arg) {
    const int free_flag = reinterpret_cast<long>(arg);
    hsa_ven_amd_loader_code_object_storage_type_t storage_type =
        HSA_VEN_AMD_LOADER_CODE_OBJECT_STORAGE_TYPE_NONE;
    int storage_fd = -1;
    uint64_t memory_base = 0;
    uint64_t memory_size = 0;
    uint64_t load_base = 0;
    uint64_t load_size = 0;
    uint64_t load_delta = 0;
    uint32_t uri_len = 0;
    char* uri_str = NULL;

    HSA_RT(LoaderApiTable.hsa_ven_amd_loader_loaded_code_object_get_info(
        loaded_code_object, HSA_VEN_AMD_LOADER_LOADED_CODE_OBJECT_INFO_CODE_OBJECT_STORAGE_TYPE,
        &storage_type));

    if (storage_type == HSA_VEN_AMD_LOADER_CODE_OBJECT_STORAGE_TYPE_FILE) {
      HSA_RT(LoaderApiTable.hsa_ven_amd_loader_loaded_code_object_get_info(
          loaded_code_object, HSA_VEN_AMD_LOADER_LOADED_CODE_OBJECT_INFO_CODE_OBJECT_STORAGE_FILE,
          &storage_fd));
      if (storage_fd == -1) {
        printf("CodeObjectCallback: fd == -1\n");
        fflush(stdout);
        abort();
      }
    } else if (storage_type == HSA_VEN_AMD_LOADER_CODE_OBJECT_STORAGE_TYPE_MEMORY) {
      HSA_RT(LoaderApiTable.hsa_ven_amd_loader_loaded_code_object_get_info(
          loaded_code_object,
          HSA_VEN_AMD_LOADER_LOADED_CODE_OBJECT_INFO_CODE_OBJECT_STORAGE_MEMORY_BASE,
          &memory_base));
      HSA_RT(LoaderApiTable.hsa_ven_amd_loader_loaded_code_object_get_info(
          loaded_code_object,
          HSA_VEN_AMD_LOADER_LOADED_CODE_OBJECT_INFO_CODE_OBJECT_STORAGE_MEMORY_SIZE,
          &memory_size));
    }

    HSA_RT(LoaderApiTable.hsa_ven_amd_loader_loaded_code_object_get_info(
        loaded_code_object, HSA_VEN_AMD_LOADER_LOADED_CODE_OBJECT_INFO_LOAD_BASE, &load_base));
    HSA_RT(LoaderApiTable.hsa_ven_amd_loader_loaded_code_object_get_info(
        loaded_code_object, HSA_VEN_AMD_LOADER_LOADED_CODE_OBJECT_INFO_LOAD_SIZE, &load_size));
    HSA_RT(LoaderApiTable.hsa_ven_amd_loader_loaded_code_object_get_info(
        loaded_code_object, HSA_VEN_AMD_LOADER_LOADED_CODE_OBJECT_INFO_LOAD_DELTA, &load_delta));

    // Getting URI
    HSA_RT(LoaderApiTable.hsa_ven_amd_loader_loaded_code_object_get_info(
        loaded_code_object, HSA_VEN_AMD_LOADER_LOADED_CODE_OBJECT_INFO_URI_LENGTH, &uri_len));

    uri_str = (char*)calloc(uri_len + 1, sizeof(char));
    if (!uri_str) EXC_ABORT(HSA_STATUS_ERROR, "URI allocation");

    HSA_RT(LoaderApiTable.hsa_ven_amd_loader_loaded_code_object_get_info(
        loaded_code_object, HSA_VEN_AMD_LOADER_LOADED_CODE_OBJECT_INFO_URI, uri_str));

    if (storage_type != HSA_VEN_AMD_LOADER_CODE_OBJECT_STORAGE_TYPE_NONE) {
      IS_HSA_CALLBACK(ROCPROFILER_HSA_CB_ID_CODEOBJ) {
        rocprofiler_hsa_callback_data_t data{};
        data.codeobj.storage_type = storage_type;
        data.codeobj.storage_file = storage_fd;
        data.codeobj.memory_base = memory_base;
        data.codeobj.memory_size = memory_size;
        data.codeobj.load_base = load_base;
        data.codeobj.load_size = load_size;
        data.codeobj.load_delta = load_delta;
        data.codeobj.uri_length = uri_len;
        data.codeobj.uri = uri_str;
        data.codeobj.unload = free_flag;

        DO_HSA_CALLBACK;
      }
    }

    {
      IS_HSA_CALLBACK(ROCPROFILER_HSA_CB_ID_ALLOCATE) {
        // Local GPU memory
        // GLOBAL; FLAGS: COARSE GRAINED
        rocprofiler_hsa_callback_data_t data{};
        data.allocate.ptr = reinterpret_cast<void*>(load_base);
        data.allocate.size = (free_flag == 0) ? load_size : 0;
        data.allocate.segment = HSA_AMD_SEGMENT_GLOBAL;
        data.allocate.global_flag = HSA_AMD_MEMORY_POOL_GLOBAL_FLAG_COARSE_GRAINED;
        data.allocate.is_code = 1;

        DO_HSA_CALLBACK;
      }
    }

    if (free_flag != 0) {
      IS_HSA_CALLBACK(ROCPROFILER_HSA_CB_ID_DEVICE) {
        hsa_amd_pointer_info_t pointer_info{};
        uint32_t num_agents = 0;
        hsa_agent_t* agents = NULL;
        pointer_info.size = sizeof(hsa_amd_pointer_info_t);
        HSA_RT(hsa_amd_pointer_info(reinterpret_cast<void*>(load_base), &pointer_info, malloc,
                                    &num_agents, &agents));

        DeviceCallback(num_agents, agents, reinterpret_cast<void*>(load_base));
      }
    }

    return HSA_STATUS_SUCCESS;
  }

  static hsa_status_t KernelSymbolCallback(hsa_executable_t executable,
                                           hsa_executable_symbol_t symbol, void* arg) {
    const int free_flag = reinterpret_cast<long>(arg);
    hsa_symbol_kind_t kind = (hsa_symbol_kind_t)0;
    HSA_RT(hsa_executable_symbol_get_info(symbol, HSA_EXECUTABLE_SYMBOL_INFO_TYPE, &kind));

    if (kind == HSA_SYMBOL_KIND_KERNEL) {
      const char* name = NULL;
      uint32_t len = 0;
      uint64_t obj = 0;
      HSA_RT(
          hsa_executable_symbol_get_info(symbol, HSA_EXECUTABLE_SYMBOL_INFO_KERNEL_OBJECT, &obj));
      if (free_flag == 0) {
        HSA_RT(
            hsa_executable_symbol_get_info(symbol, HSA_EXECUTABLE_SYMBOL_INFO_NAME_LENGTH, &len));
        char sym_name[len + 1];
        HSA_RT(hsa_executable_symbol_get_info(symbol, HSA_EXECUTABLE_SYMBOL_INFO_NAME, sym_name));
        name = cpp_demangle(sym_name);
      }

      rocprofiler_hsa_callback_data_t data{};
      data.ksymbol.object = obj;
      data.ksymbol.name = name;
      data.ksymbol.name_length = len;
      data.ksymbol.unload = free_flag;

      ISSUE_HSA_CALLBACK(ROCPROFILER_HSA_CB_ID_KSYMBOL);
    }

    return HSA_STATUS_SUCCESS;
  }

  static hsa_status_t ExecutableFreeze(hsa_executable_t executable, const char* options) {
    hsa_status_t status = HSA_STATUS_SUCCESS;

    HSA_RT(hsa_executable_freeze_fn(executable, options));

    unsigned is_codeobj_cb = 0;
    { IS_HSA_CALLBACK(ROCPROFILER_HSA_CB_ID_CODEOBJ) is_codeobj_cb |= 1; }
    { IS_HSA_CALLBACK(ROCPROFILER_HSA_CB_ID_ALLOCATE) is_codeobj_cb |= 1; }
    if (is_codeobj_cb) {
      LoaderApiTable.hsa_ven_amd_loader_executable_iterate_loaded_code_objects(
          executable, CodeObjectCallback, reinterpret_cast<void*>(0));
    }

    IS_HSA_CALLBACK(ROCPROFILER_HSA_CB_ID_KSYMBOL) {
      HSA_RT(hsa_executable_iterate_symbols(executable, KernelSymbolCallback,
                                            reinterpret_cast<void*>(0)));
    }

    return status;
  }

  static hsa_status_t ExecutableDestroy(hsa_executable_t executable) {
    hsa_status_t status = HSA_STATUS_SUCCESS;

    IS_HSA_CALLBACK(ROCPROFILER_HSA_CB_ID_ALLOCATE) {
      LoaderApiTable.hsa_ven_amd_loader_executable_iterate_loaded_code_objects(
          executable, CodeObjectCallback, reinterpret_cast<void*>(1));
    }

    {
      IS_HSA_CALLBACK(ROCPROFILER_HSA_CB_ID_KSYMBOL) {
        HSA_RT(hsa_executable_iterate_symbols(executable, KernelSymbolCallback,
                                              reinterpret_cast<void*>(1)));
      }
    }

    HSA_RT(hsa_executable_destroy_fn(executable));

    return status;
  }

  static bool enable_;
  static thread_local bool recursion_;
  static hsa_ven_amd_loader_1_01_pfn_t LoaderApiTable;
  static rocprofiler_hsa_callbacks_t callbacks_;
  static arg_t arg_;
  static mutex_t mutex_;
};

}  // namespace rocprofiler

#endif  // _SRC_CORE_HSA_INTERCEPTOR_H
