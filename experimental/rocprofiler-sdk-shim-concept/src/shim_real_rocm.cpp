#include <cstdint>
#include <cstring>

#include "shim_real_probe.h"
#include "shim_runtime_bridge.h"

#if defined(__has_include)
#    if __has_include(<hip/amd_detail/hip_api_trace.hpp>) && (__has_include(<hsa/hsa.h>) || __has_include(<hsa.h>))
#        define hipDeviceProp_tR0000 hipDeviceProp_tR0600
typedef unsigned int GLuint;
typedef unsigned int GLenum;
typedef int          hipGLDeviceList;
#        include <hip/amd_detail/hip_api_trace.hpp>
#        if __has_include(<hsa/hsa.h>)
#            include <hsa/hsa.h>
#        else
#            include <hsa.h>
#        endif
#        define SHIM_REAL_ROCM_ENABLED 1
#    else
#        define SHIM_REAL_ROCM_ENABLED 0
#    endif
#else
#    define SHIM_REAL_ROCM_ENABLED 0
#endif

#if SHIM_REAL_ROCM_ENABLED

namespace
{
struct shim_hsa_api_table_version_t
{
    uint32_t major_id;
    uint32_t minor_id;
    uint32_t step_id;
    uint32_t reserved;
};

struct shim_hsa_core_api_table_t
{
    shim_hsa_api_table_version_t                  version;
    decltype(hsa_init)*                           hsa_init_fn;
    decltype(hsa_shut_down)*                      hsa_shut_down_fn;
    decltype(hsa_system_get_info)*                hsa_system_get_info_fn;
    decltype(hsa_system_extension_supported)*     hsa_system_extension_supported_fn;
    decltype(hsa_system_get_extension_table)*     hsa_system_get_extension_table_fn;
    decltype(hsa_iterate_agents)*                 hsa_iterate_agents_fn;
};

struct shim_hsa_api_table_t
{
    shim_hsa_api_table_version_t version;
    shim_hsa_core_api_table_t*   core_;
    void*                        amd_ext_;
    void*                        finalizer_ext_;
    void*                        image_ext_;
    void*                        tools_;
    void*                        pc_sampling_ext_;
};

using hip_get_device_count_t = t_hipGetDeviceCount;
using hip_malloc_t           = t_hipMalloc;
using hip_free_t             = t_hipFree;

using hsa_init_t           = decltype(hsa_init)*;
using hsa_iterate_agents_t = decltype(hsa_iterate_agents)*;
using hsa_shut_down_t      = decltype(hsa_shut_down)*;

HipDispatchTable*       g_real_hip_table = nullptr;
shim_hsa_api_table_t*   g_real_hsa_table = nullptr;

struct packed_hip_get_device_count_t
{
    uintptr_t count_ptr;
    int32_t   count_value;
    int32_t   result;
};

struct packed_hip_malloc_t
{
    uintptr_t out_ptr;
    uintptr_t allocated_ptr;
    uint64_t  size;
    int32_t   result;
};

struct packed_hip_free_t
{
    uintptr_t ptr;
    int32_t   result;
};

struct packed_hsa_init_t
{
    uint32_t result;
};

struct packed_hsa_iterate_agents_t
{
    uintptr_t callback;
    uintptr_t data;
    uint32_t  result;
};

struct packed_hsa_shutdown_t
{
    uint32_t result;
};

uint32_t g_real_hip_base = 0;
uint32_t g_real_hsa_base = 0;

constexpr const char* k_hip_names[] = {
    "hipGetDeviceCount",
    "hipMalloc",
    "hipFree",
};

constexpr const char* k_hsa_names[] = {
    "hsa_init",
    "hsa_iterate_agents",
    "hsa_shut_down",
};

template <typename FnT, typename PackedT, typename InvokeT>
auto traced_call(uint32_t slot, PackedT& packed, InvokeT&& invoke) -> decltype(invoke((FnT) nullptr))
{
    using ret_type = decltype(invoke((FnT) nullptr));

    auto mode = shim_load_mode_for_slot(slot);
    auto orig = reinterpret_cast<FnT>(shim_get_runtime_original((int) slot));

    if(mode == ROCP_SHIM_MODE_OFF || orig == nullptr) return invoke(orig);
    if(!shim_should_trace_slot(slot, &packed)) return invoke(orig);

    auto corr = shim_push_correlation_public();
    shim_emit_trace_record(slot, SHIM_PHASE_ENTER, &corr, nullptr, 0);

    auto next = reinterpret_cast<FnT>(shim_get_next_in_chain((int) slot));
    if(next == nullptr) next = orig;

    ret_type ret = invoke(next);

    shim_emit_trace_record(slot, SHIM_PHASE_EXIT, &corr, &packed, sizeof(packed));
    shim_pop_correlation_public();
    return ret;
}

extern "C" hipError_t shim_wrap_hip_get_device_count(int* count)
{
    packed_hip_get_device_count_t packed = {reinterpret_cast<uintptr_t>(count), 0, (int32_t) hipErrorUnknown};
    return traced_call<hip_get_device_count_t>(g_real_hip_base + 0, packed, [&](hip_get_device_count_t fn) {
        if(fn == nullptr) return hipErrorUnknown;
        auto ret       = fn(count);
        packed.result  = static_cast<int32_t>(ret);
        packed.count_value = (count != nullptr) ? *count : -1;
        return ret;
    });
}

extern "C" hipError_t shim_wrap_hip_malloc(void** ptr, size_t size)
{
    packed_hip_malloc_t packed = {reinterpret_cast<uintptr_t>(ptr), 0, (uint64_t) size, (int32_t) hipErrorUnknown};
    return traced_call<hip_malloc_t>(g_real_hip_base + 1, packed, [&](hip_malloc_t fn) {
        if(fn == nullptr) return hipErrorUnknown;
        auto ret          = fn(ptr, size);
        packed.result     = static_cast<int32_t>(ret);
        packed.allocated_ptr = (ptr != nullptr && ret == hipSuccess) ? reinterpret_cast<uintptr_t>(*ptr) : 0;
        return ret;
    });
}

extern "C" hipError_t shim_wrap_hip_free(void* ptr)
{
    packed_hip_free_t packed = {reinterpret_cast<uintptr_t>(ptr), (int32_t) hipErrorUnknown};
    return traced_call<hip_free_t>(g_real_hip_base + 2, packed, [&](hip_free_t fn) {
        if(fn == nullptr) return hipErrorUnknown;
        auto ret      = fn(ptr);
        packed.result = static_cast<int32_t>(ret);
        return ret;
    });
}

extern "C" hsa_status_t shim_wrap_hsa_init()
{
    packed_hsa_init_t packed = {static_cast<uint32_t>(HSA_STATUS_ERROR)};
    return traced_call<hsa_init_t>(g_real_hsa_base + 0, packed, [&](hsa_init_t fn) {
        if(fn == nullptr) return HSA_STATUS_ERROR;
        auto ret      = fn();
        packed.result = static_cast<uint32_t>(ret);
        return ret;
    });
}

extern "C" hsa_status_t shim_wrap_hsa_iterate_agents(hsa_status_t (*callback)(hsa_agent_t, void*), void* data)
{
    packed_hsa_iterate_agents_t packed = {
        reinterpret_cast<uintptr_t>(callback),
        reinterpret_cast<uintptr_t>(data),
        static_cast<uint32_t>(HSA_STATUS_ERROR)};
    return traced_call<hsa_iterate_agents_t>(g_real_hsa_base + 1, packed,
                                             [&](hsa_iterate_agents_t fn) {
                                                 if(fn == nullptr) return HSA_STATUS_ERROR;
                                                 auto ret      = fn(callback, data);
                                                 packed.result = static_cast<uint32_t>(ret);
                                                 return ret;
                                             });
}

extern "C" hsa_status_t shim_wrap_hsa_shut_down()
{
    packed_hsa_shutdown_t packed = {static_cast<uint32_t>(HSA_STATUS_ERROR)};
    return traced_call<hsa_shut_down_t>(g_real_hsa_base + 2, packed, [&](hsa_shut_down_t fn) {
        if(fn == nullptr) return HSA_STATUS_ERROR;
        auto ret      = fn();
        packed.result = static_cast<uint32_t>(ret);
        return ret;
    });
}

int install_real_hip_table(uint64_t lib_version, uint64_t lib_instance, void** tables, uint64_t num_tables)
{
    if(tables == nullptr || num_tables == 0) return -1;

    auto* table = static_cast<HipDispatchTable*>(tables[0]);
    if(table == nullptr) return -1;

    uint32_t base = 0;
    if(shim_register_table_metadata("hip", lib_version, lib_instance, 3, k_hip_names, &base) != 0)
        return -1;

    g_real_hip_base = base;
    g_real_hip_table = table;
    shim_install_slot_wrapper(base + 0, reinterpret_cast<void**>(&table->hipGetDeviceCount_fn),
                              reinterpret_cast<void*>(&shim_wrap_hip_get_device_count));
    shim_install_slot_wrapper(base + 1, reinterpret_cast<void**>(&table->hipMalloc_fn),
                              reinterpret_cast<void*>(&shim_wrap_hip_malloc));
    shim_install_slot_wrapper(base + 2, reinterpret_cast<void**>(&table->hipFree_fn),
                              reinterpret_cast<void*>(&shim_wrap_hip_free));
    return 0;
}

int install_real_hsa_table(uint64_t lib_version, uint64_t lib_instance, void** tables, uint64_t num_tables)
{
    if(tables == nullptr || num_tables == 0) return -1;

    auto* table = static_cast<shim_hsa_api_table_t*>(tables[0]);
    if(table == nullptr || table->core_ == nullptr) return -1;

    uint32_t base = 0;
    if(shim_register_table_metadata("hsa", lib_version, lib_instance, 3, k_hsa_names, &base) != 0)
        return -1;

    g_real_hsa_base = base;
    g_real_hsa_table = table;
    shim_install_slot_wrapper(base + 0, reinterpret_cast<void**>(&table->core_->hsa_init_fn),
                              reinterpret_cast<void*>(&shim_wrap_hsa_init));
    shim_install_slot_wrapper(base + 1, reinterpret_cast<void**>(&table->core_->hsa_iterate_agents_fn),
                              reinterpret_cast<void*>(&shim_wrap_hsa_iterate_agents));
    shim_install_slot_wrapper(base + 2, reinterpret_cast<void**>(&table->core_->hsa_shut_down_fn),
                              reinterpret_cast<void*>(&shim_wrap_hsa_shut_down));
    return 0;
}
}  // namespace

extern "C" int shim_real_probe_invoke_hip_get_device_count(int* count)
{
    if(g_real_hip_table == nullptr || g_real_hip_table->hipGetDeviceCount_fn == nullptr) return -1;
    return static_cast<int>(g_real_hip_table->hipGetDeviceCount_fn(count));
}

namespace
{
hsa_status_t shim_hsa_probe_count_cb(hsa_agent_t, void* data)
{
    if(data) (*static_cast<int*>(data))++;
    return HSA_STATUS_SUCCESS;
}
}  // namespace

extern "C" int shim_real_probe_invoke_hsa_iterate_agents(int* agent_count)
{
    if(g_real_hsa_table == nullptr || g_real_hsa_table->core_ == nullptr ||
       g_real_hsa_table->core_->hsa_iterate_agents_fn == nullptr) {
        return -1;
    }

    int local_count = 0;
    auto ret = g_real_hsa_table->core_->hsa_iterate_agents_fn(&shim_hsa_probe_count_cb, &local_count);
    if(agent_count) *agent_count = local_count;
    return static_cast<int>(ret);
}

extern "C" int rocprofiler_set_api_table(const char* name,
                                          uint64_t    lib_version,
                                          uint64_t    lib_instance,
                                          void**      tables,
                                          uint64_t    num_tables)
{
    if(name == nullptr) return -1;
    if(strcmp(name, "hip") == 0) return install_real_hip_table(lib_version, lib_instance, tables, num_tables);
    if(strcmp(name, "hsa") == 0) return install_real_hsa_table(lib_version, lib_instance, tables, num_tables);
    return 0;
}

extern "C" int rocprofiler_attach_set_api_table(const char* name,
                                                 uint64_t    lib_version,
                                                 uint64_t    lib_instance,
                                                 void**      tables,
                                                 uint64_t    num_tables,
                                                 void*       register_callback)
{
    (void) register_callback;
    return rocprofiler_set_api_table(name, lib_version, lib_instance, tables, num_tables);
}

extern "C" int rocprofiler_attach(void) { return 0; }
extern "C" int rocprofiler_detach(void) { return 0; }

#endif
