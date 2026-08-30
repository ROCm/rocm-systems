// MIT License
//
// Copyright (c) 2025-2026 Advanced Micro Devices, Inc. All rights reserved.
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

#include "code_object_registration.h"
#include "code_object_registration.hpp"
#include "table.hpp"

#include <hsa/hsa.h>
#include <hsa/hsa_ven_amd_loader.h>

#include "lib/common/static_object.hpp"

#include <cstring>
#include <mutex>
#include <unordered_map>
#include <vector>

namespace
{
using hsa_executable_freeze_t  = decltype(CoreApiTable::hsa_executable_freeze_fn);
using hsa_executable_destroy_t = decltype(CoreApiTable::hsa_executable_destroy_fn);
using hsa_loader_table_t       = hsa_ven_amd_loader_1_01_pfn_t;
using code_object_collection_t = std::vector<hsa_executable_t>;
// Keyed by loaded_code_object.handle; owned copy of the ELF bytes for
// memory-backed code objects, taken at executable_freeze
using memory_codeobj_cache_t = std::unordered_map<uint64_t, std::vector<std::byte>>;

struct code_object_cb_entry_t
{
    rocprofiler_attach_code_object_cb_t cb   = nullptr;
    void*                               data = nullptr;
};

struct code_object_registration_t
{
    // gates access to code_objects, cb_list, and memory_codeobj_cache
    std::mutex                          mutex;
    code_object_collection_t            code_objects;
    std::vector<code_object_cb_entry_t> cb_list;
    memory_codeobj_cache_t              memory_codeobj_cache;
    hsa_executable_freeze_t             hsa_executable_freeze_fn  = nullptr;
    hsa_executable_destroy_t            hsa_executable_destroy_fn = nullptr;
    hsa_loader_table_t                  loader_table              = {};
};

code_object_registration_t*
get_code_object_registration()
{
    static auto*& registration =
        rocprofiler::common::static_object<code_object_registration_t>::construct();
    return registration;
}

// Callback for hsa_ven_amd_loader_executable_iterate_loaded_code_objects.
// Copies ELF bytes for memory-backed code objects into the registration cache.
hsa_status_t
cache_memory_codeobj_cb(hsa_executable_t /*executable*/,
                        hsa_loaded_code_object_t loaded_code_object,
                        void*                    cb_data)
{
    auto* registration = static_cast<code_object_registration_t*>(cb_data);

#define ROCP_ATTACH_LOADER_GET_CODE_OBJECT_INFO(QUERY, DST)                                        \
    {                                                                                              \
        auto _status = registration->loader_table.hsa_ven_amd_loader_loaded_code_object_get_info(  \
            loaded_code_object, (QUERY), (DST));                                                   \
        ROCP_ERROR_IF(_status != HSA_STATUS_SUCCESS)                                               \
            << "hsa_ven_amd_loader_loaded_code_object_get_info(" #QUERY ") failed";                \
        if(_status != HSA_STATUS_SUCCESS) return _status;                                          \
    }

    auto storage_type = uint32_t{0};
    ROCP_ATTACH_LOADER_GET_CODE_OBJECT_INFO(
        HSA_VEN_AMD_LOADER_LOADED_CODE_OBJECT_INFO_CODE_OBJECT_STORAGE_TYPE, &storage_type);

    if(storage_type != HSA_VEN_AMD_LOADER_CODE_OBJECT_STORAGE_TYPE_MEMORY)
        return HSA_STATUS_SUCCESS;

    auto memory_base = uint64_t{0};
    auto memory_size = uint64_t{0};
    ROCP_ATTACH_LOADER_GET_CODE_OBJECT_INFO(
        HSA_VEN_AMD_LOADER_LOADED_CODE_OBJECT_INFO_CODE_OBJECT_STORAGE_MEMORY_BASE, &memory_base);
    ROCP_ATTACH_LOADER_GET_CODE_OBJECT_INFO(
        HSA_VEN_AMD_LOADER_LOADED_CODE_OBJECT_INFO_CODE_OBJECT_STORAGE_MEMORY_SIZE, &memory_size);

    if(memory_base == 0 || memory_size == 0) return HSA_STATUS_SUCCESS;

    // NOLINTNEXTLINE(performance-no-int-to-ptr)
    const auto* src   = reinterpret_cast<const std::byte*>(memory_base);
    auto        bytes = std::vector<std::byte>(src, src + static_cast<std::ptrdiff_t>(memory_size));

    ROCP_TRACE << "caching " << memory_size << " bytes for loaded_code_object "
               << loaded_code_object.handle;
    std::lock_guard lg(registration->mutex);
    registration->memory_codeobj_cache.emplace(loaded_code_object.handle, std::move(bytes));
    return HSA_STATUS_SUCCESS;
}

// Copy ELF bytes for all memory-backed loaded code objects belonging to
// |executable|. Must be called while the caller's original buffer is still
// live, i.e. from within the intercepted hsa_executable_freeze before it
// returns to the application.
// precond: registration mutex is NOT held by the caller
void
cache_memory_codeobjs(hsa_executable_t executable)
{
    auto* registration = CHECK_NOTNULL(get_code_object_registration());
    if(!registration->loader_table.hsa_ven_amd_loader_executable_iterate_loaded_code_objects ||
       !registration->loader_table.hsa_ven_amd_loader_loaded_code_object_get_info)
        return;

    registration->loader_table.hsa_ven_amd_loader_executable_iterate_loaded_code_objects(
        executable, cache_memory_codeobj_cb, registration);
}

hsa_status_t
executable_freeze(hsa_executable_t executable, const char* options)
{
    auto* registration = CHECK_NOTNULL(get_code_object_registration());
    auto  status       = registration->hsa_executable_freeze_fn(executable, options);

    if(status != HSA_STATUS_SUCCESS)
    {
        return status;
    }

    // Copy ELF bytes for memory-backed code objects now, while still inside the
    // application's call to hsa_executable_freeze. This is the only point at
    // which the original buffer is guaranteed live.
    cache_memory_codeobjs(executable);

    ROCP_TRACE << "adding code_object " << executable.handle;
    auto snapshot = std::vector<code_object_cb_entry_t>{};
    {
        std::lock_guard lg(registration->mutex);
        registration->code_objects.emplace_back(executable);
        snapshot = std::vector<code_object_cb_entry_t>{registration->cb_list};
    }
    for(auto& entry : snapshot)
    {
        if(entry.cb)
        {
            entry.cb(executable, ROCPROFILER_ATTACH_CODE_OBJECT_CREATED, entry.data);
        }
    }
    return HSA_STATUS_SUCCESS;
}

hsa_status_t
executable_destroy(hsa_executable_t executable)
{
    auto* registration = CHECK_NOTNULL(get_code_object_registration());
    ROCP_TRACE << "removing code_object " << executable.handle;

    // Collect the loaded_code_object handles whose cached bytes belong to this
    // executable so we can erase them after callbacks have fired.
    auto                  snapshot          = std::vector<code_object_cb_entry_t>{};
    std::vector<uint64_t> codeobj_to_remove = {};
    {
        std::lock_guard lg(registration->mutex);
        auto pred = [&](const hsa_executable_t& a) { return a.handle == executable.handle; };
        auto itr  = std::find_if(
            registration->code_objects.begin(), registration->code_objects.end(), pred);
        if(itr == registration->code_objects.end())
        {
            ROCP_WARNING << "remove code_object could not find " << executable.handle;
        }
        else
        {
            snapshot = std::vector<code_object_cb_entry_t>{registration->cb_list};
            registration->code_objects.erase(itr);
            for(const auto& [handle, _] : registration->memory_codeobj_cache)
                codeobj_to_remove.push_back(handle);
        }
    }

    // Fire callbacks after erasing from the collection but before calling the real destroy.
    // Erasing first prevents double-destroy races. Calling the real destroy last ensures the
    // handle remains valid during callbacks, and that any handle still in code_objects is live.
    // Iterate in reverse (LIFO) to mirror the SDK chaining convention: the most recently registered
    // handler runs first and calls the chained destroy callback after its own work, effectively
    // reversing the order of callbacks from freeze/create
    for(auto it = snapshot.rbegin(); it != snapshot.rend(); ++it)
    {
        if(it->cb)
        {
            it->cb(executable, ROCPROFILER_ATTACH_CODE_OBJECT_DESTROYED, it->data);
        }
    }

    // Now that all callbacks have returned, drop the owned ELF byte copies
    if(!codeobj_to_remove.empty())
    {
        std::lock_guard lg(registration->mutex);
        for(auto handle : codeobj_to_remove)
            registration->memory_codeobj_cache.erase(handle);
    }

    return registration->hsa_executable_destroy_fn(executable);
}

int
iterate_all_code_objects(rocprof_attach_code_object_iterator_t func, void* data)
{
    auto* registration = CHECK_NOTNULL(get_code_object_registration());

    auto snapshot = code_object_collection_t{};
    {
        std::lock_guard lg(registration->mutex);
        snapshot = code_object_collection_t{registration->code_objects};
    }
    for(const auto& code_object : snapshot)
    {
        func(code_object, data);
    }

    return ROCPROFILER_STATUS_SUCCESS;
}

int
add_code_object_cb(rocprofiler_attach_code_object_cb_t cb, void* data)
{
    if(!cb)
    {
        return ROCPROFILER_STATUS_ERROR_INVALID_ARGUMENT;
    }
    auto* registration = CHECK_NOTNULL(get_code_object_registration());
    auto  snapshot     = code_object_collection_t{};
    {
        auto lg = std::lock_guard{registration->mutex};
        registration->cb_list.push_back({cb, data});
        snapshot = code_object_collection_t{registration->code_objects};
    }
    for(const auto& code_object : snapshot)
    {
        cb(code_object, ROCPROFILER_ATTACH_CODE_OBJECT_CREATED, data);
    }
    return ROCPROFILER_STATUS_SUCCESS;
}

int
remove_code_object_cb(rocprofiler_attach_code_object_cb_t cb)
{
    if(!cb)
    {
        return ROCPROFILER_STATUS_ERROR_INVALID_ARGUMENT;
    }
    auto* registration = CHECK_NOTNULL(get_code_object_registration());
    auto  lg           = std::lock_guard{registration->mutex};
    auto  pred         = [cb](const code_object_cb_entry_t& e) { return e.cb == cb; };
    registration->cb_list.erase(
        std::remove_if(registration->cb_list.begin(), registration->cb_list.end(), pred),
        registration->cb_list.end());
    // Returns SUCCESS whether or not cb was found, to simplify cleanup paths.
    return ROCPROFILER_STATUS_SUCCESS;
}

int
lookup_memory_codeobj_data(hsa_loaded_code_object_t loaded_code_object,
                           const void**             data_out,
                           uint64_t*                size_out)
{
    if(!data_out || !size_out) return ROCPROFILER_STATUS_ERROR_INVALID_ARGUMENT;

    auto* registration = CHECK_NOTNULL(get_code_object_registration());
    auto  lg           = std::lock_guard{registration->mutex};
    auto  itr          = registration->memory_codeobj_cache.find(loaded_code_object.handle);
    if(itr == registration->memory_codeobj_cache.end())
        return ROCPROFILER_STATUS_ERROR_INVALID_ARGUMENT;

    *data_out = itr->second.data();
    *size_out = static_cast<uint64_t>(itr->second.size());
    return ROCPROFILER_STATUS_SUCCESS;
}

}  // namespace

namespace rocprofiler
{
namespace attach
{
void
code_object_registration_init(
    HsaApiTable* table)  // CoreApiTable& core_table, AmdExtTable& ext_table)
{
    ROCP_TRACE << "Initializing Code Object Registration";
    auto*         registration = CHECK_NOTNULL(get_code_object_registration());
    CoreApiTable& core_table   = *table->core_;

    // Acquire the AMD loader extension table so executable_freeze can copy ELF
    // bytes for memory-backed code objects.
    auto loader_status = core_table.hsa_system_get_major_extension_table_fn(
        HSA_EXTENSION_AMD_LOADER, 1, sizeof(hsa_loader_table_t), &registration->loader_table);
    ROCP_WARNING_IF(loader_status != HSA_STATUS_SUCCESS)
        << "hsa_system_get_major_extension_table(HSA_EXTENSION_AMD_LOADER) failed in attach "
           "init; memory-backed code object ELF bytes will not be cached";

    // route executable freeze and destroy to us, but also save the original entrypoint so we can
    // call it
    registration->hsa_executable_freeze_fn  = core_table.hsa_executable_freeze_fn;
    core_table.hsa_executable_freeze_fn     = executable_freeze;
    registration->hsa_executable_destroy_fn = core_table.hsa_executable_destroy_fn;
    core_table.hsa_executable_destroy_fn    = executable_destroy;
}

}  // namespace attach
}  // namespace rocprofiler

ROCPROFILER_EXTERN_C_INIT

int
rocprofiler_attach_iterate_all_code_objects(rocprof_attach_code_object_iterator_t func, void* data)
{
    return iterate_all_code_objects(func, data);
}

int
rocprofiler_attach_add_code_object_cb(rocprofiler_attach_code_object_cb_t cb, void* data)
{
    return add_code_object_cb(cb, data);
}

int
rocprofiler_attach_remove_code_object_cb(rocprofiler_attach_code_object_cb_t cb)
{
    return remove_code_object_cb(cb);
}

int
rocprofiler_attach_lookup_memory_codeobj_data(hsa_loaded_code_object_t loaded_code_object,
                                              const void**             data_out,
                                              uint64_t*                size_out)
{
    return lookup_memory_codeobj_data(loaded_code_object, data_out, size_out);
}

ROCPROFILER_EXTERN_C_FINI
