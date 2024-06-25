// MIT License
//
// Copyright (c) 2024 ROCm Developer Tools
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.

#include "lib/rocprofiler-sdk/thread_trace/code_object.hpp"
#include "lib/rocprofiler-sdk/code_object/code_object.hpp"

namespace rocprofiler
{
namespace thread_trace
{
namespace code_object
{
std::mutex                         CodeobjCallbackRegistry::mut;
std::set<CodeobjCallbackRegistry*> CodeobjCallbackRegistry::all_registries{};

CodeobjCallbackRegistry::CodeobjCallbackRegistry(LoadCallback _ld, UnloadCallback _unld)
: ld_fn(std::move(_ld))
, unld_fn(std::move(_unld))
{
    std::unique_lock<std::mutex> lg(mut);
    all_registries.insert(this);
}

CodeobjCallbackRegistry::~CodeobjCallbackRegistry()
{
    std::unique_lock<std::mutex> lg(mut);
    all_registries.erase(this);
}

void
CodeobjCallbackRegistry::Load(rocprofiler_agent_id_t agent,
                              uint64_t               id,
                              uint64_t               addr,
                              uint64_t               size)
{
    std::unique_lock<std::mutex> lg(mut);
    for(auto* reg : all_registries)
        reg->ld_fn(agent, id, addr, size);
}

void
CodeobjCallbackRegistry::Unload(uint64_t id)
{
    std::unique_lock<std::mutex> lg(mut);
    for(auto* reg : all_registries)
        reg->unld_fn(id);
}

void
CodeobjCallbackRegistry::IterateLoaded() const
{
    std::unique_lock<std::mutex> lg(mut);

    rocprofiler::code_object::iterate_loaded_code_objects(
        [&](const rocprofiler::code_object::hsa::code_object& code_object) {
            const auto& data = code_object.rocp_data;
            ld_fn(data.rocp_agent, data.code_object_id, data.load_delta, data.load_size);
        });
}

namespace
{
auto&
get_freeze_function()
{
    static decltype(::hsa_executable_freeze)* _v = nullptr;
    return _v;
}

auto&
get_destroy_function()
{
    static decltype(::hsa_executable_destroy)* _v = nullptr;
    return _v;
}

hsa_status_t
executable_freeze(hsa_executable_t executable, const char* options)
{
    // Call underlying function
    hsa_status_t status = CHECK_NOTNULL(get_freeze_function())(executable, options);
    if(status != HSA_STATUS_SUCCESS) return status;

    rocprofiler::code_object::iterate_loaded_code_objects(
        [&](const rocprofiler::code_object::hsa::code_object& code_object) {
            if(code_object.hsa_executable != executable) return;

            const auto& data = code_object.rocp_data;
            CodeobjCallbackRegistry::Load(
                data.rocp_agent, data.code_object_id, data.load_delta, data.load_size);
        });

    return HSA_STATUS_SUCCESS;
}

hsa_status_t
executable_destroy(hsa_executable_t executable)
{
    rocprofiler::code_object::iterate_loaded_code_objects(
        [&](const rocprofiler::code_object::hsa::code_object& code_object) {
            if(code_object.hsa_executable == executable)
                CodeobjCallbackRegistry::Unload(code_object.rocp_data.code_object_id);
        });

    // Call underlying function
    return CHECK_NOTNULL(get_destroy_function())(executable);
}
}  // namespace

void
initialize(HsaApiTable* table)
{
    (void) table;
    auto& core_table = *table->core_;

    get_freeze_function()                = CHECK_NOTNULL(core_table.hsa_executable_freeze_fn);
    get_destroy_function()               = CHECK_NOTNULL(core_table.hsa_executable_destroy_fn);
    core_table.hsa_executable_freeze_fn  = executable_freeze;
    core_table.hsa_executable_destroy_fn = executable_destroy;
    LOG_IF(FATAL, get_freeze_function() == core_table.hsa_executable_freeze_fn)
        << "infinite recursion";
    LOG_IF(FATAL, get_destroy_function() == core_table.hsa_executable_destroy_fn)
        << "infinite recursion";
}

}  // namespace code_object
}  // namespace thread_trace
}  // namespace rocprofiler
