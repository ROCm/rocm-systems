// MIT License
//
// Copyright (c) 2024-2025 ROCm Developer Tools
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
#include "lib/common/static_object.hpp"
#include "lib/common/synchronized.hpp"
#include "lib/rocprofiler-sdk/code_object/code_object.hpp"

#include <mutex>
#include <set>

namespace rocprofiler
{
namespace thread_trace
{
namespace code_object
{
using set_type_t = std::set<CodeobjCallbackRegistry*>;

auto*
_static_registries()
{
    static auto*& reg = common::static_object<common::Synchronized<set_type_t>>::construct();
    return reg;
}

auto&
get_registries()
{
    return *CHECK_NOTNULL(_static_registries());
}

CodeobjCallbackRegistry::CodeobjCallbackRegistry(LoadCallback _ld, UnloadCallback _unld)
: ld_fn(std::move(_ld))
, unld_fn(std::move(_unld))
{
    get_registries().wlock([this](set_type_t& t) { t.insert(this); });
}

CodeobjCallbackRegistry::~CodeobjCallbackRegistry()
{
    if(auto* reg = _static_registries()) reg->wlock([this](set_type_t& t) { t.erase(this); });
}

void
CodeobjCallbackRegistry::IterateLoaded() const
{
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

            const auto& co = code_object.rocp_data;

            get_registries().wlock([&](set_type_t& t) {
                for(auto* reg : t)
                    reg->ld_fn(co.rocp_agent, co.code_object_id, co.load_delta, co.load_size);
            });
        });

    return HSA_STATUS_SUCCESS;
}

hsa_status_t
executable_destroy(hsa_executable_t executable)
{
    rocprofiler::code_object::iterate_loaded_code_objects(
        [&](const rocprofiler::code_object::hsa::code_object& code_object) {
            if(code_object.hsa_executable != executable) return;

            get_registries().wlock([&](set_type_t& t) {
                for(auto* reg : t)
                    reg->unld_fn(code_object.rocp_data.code_object_id);
            });
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

void
finalize()
{
    get_registries().wlock([](set_type_t& t) { t.clear(); });
}

}  // namespace code_object
}  // namespace thread_trace
}  // namespace rocprofiler
