// MIT License
//
// Copyright (c) 2023 Advanced Micro Devices, Inc. All rights reserved.
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

#include <rocprofiler-sdk/fwd.h>
#include <rocprofiler-sdk/intercept_table.h>
#include <rocprofiler-sdk/rocprofiler.h>

#include "lib/common/container/stable_vector.hpp"
#include "lib/common/utility.hpp"
#include "lib/rocprofiler-sdk/buffer.hpp"
#include "lib/rocprofiler-sdk/context/context.hpp"
#include "lib/rocprofiler-sdk/internal_threading.hpp"
#include "lib/rocprofiler-sdk/registration.hpp"

#include <hsa/hsa_api_trace.h>

#include <cstdint>
#include <mutex>
#include <stdexcept>
#include <string>
#include <vector>

namespace rocprofiler
{
namespace intercept_table
{
namespace
{
template <rocprofiler_runtime_library_t... Idx>
using library_sequence_t = std::integer_sequence<rocprofiler_runtime_library_t, Idx...>;

// this is used to loop over the different libraries
constexpr auto intercept_library_seq = library_sequence_t<ROCPROFILER_HSA_LIBRARY,
                                                          ROCPROFILER_HIP_LIBRARY,
                                                          ROCPROFILER_MARKER_LIBRARY>{};

// check that intercept_library_seq is up to date
static_assert((1 << (intercept_library_seq.size())) == ROCPROFILER_LIBRARY_LAST,
              "Update intercept_library_seq to include new libraries");

// data structure holding list of callbacks
template <rocprofiler_runtime_library_t LibT>
struct intercept
{
    static constexpr auto value = LibT;

    std::vector<rocprofiler_intercept_library_cb_t> callbacks = {};
    std::vector<void*>                              user_data = {};
    std::mutex                                      mutex     = {};
};

// static accessor for intercept instance
template <rocprofiler_runtime_library_t LibT>
auto&
get_intercept()
{
    static auto _v = intercept<LibT>{};
    return _v;
}

// adds callbacks to intercept instance(s)
template <rocprofiler_runtime_library_t... Idx>
void
update_intercepts(rocprofiler_intercept_library_cb_t cb,
                  int                                libs,
                  void*                              data,
                  library_sequence_t<Idx...>)
{
    auto update = [cb, libs, data](auto& notifier) {
        if(libs == 0 || ((libs & notifier.value) == notifier.value))
        {
            notifier.mutex.lock();
            notifier.callbacks.emplace_back(cb);
            notifier.user_data.emplace_back(data);
            notifier.mutex.unlock();
        }
    };

    (update(get_intercept<Idx>()), ...);
}

template <typename... Tp, size_t... Idx>
auto
get_void_array(std::tuple<Tp*...> data, std::index_sequence<Idx...>)
{
    constexpr auto size = sizeof...(Idx);
    return std::array<void*, size>{static_cast<void*>(std::get<Idx>(data))...};
};

// invokes creation notifiers
template <typename... ApiTableT, rocprofiler_runtime_library_t... Idx>
void
execute_intercepts(rocprofiler_runtime_library_t lib,
                   uint64_t                      lib_version,
                   uint64_t                      lib_instance,
                   std::tuple<ApiTableT*...>     tables,
                   std::integer_sequence<rocprofiler_runtime_library_t, Idx...>)
{
    auto execute = [lib, lib_version, lib_instance, tables](auto& notifier) {
        if(((lib & notifier.value) == notifier.value))
        {
            constexpr uint64_t num_tables = sizeof...(ApiTableT);
            auto tables_v = get_void_array(tables, std::make_index_sequence<num_tables>{});

            notifier.mutex.lock();
            for(size_t i = 0; i < notifier.callbacks.size(); ++i)
            {
                auto itr = notifier.callbacks.at(i);
                if(itr)
                    itr(notifier.value,
                        lib_version,
                        lib_instance,
                        tables_v.data(),
                        num_tables,
                        notifier.user_data.at(i));
            }
            notifier.mutex.unlock();
        }
    };

    (execute(get_intercept<Idx>()), ...);
}
}  // namespace

template <typename... ApiTableT>
void
notify_runtime_api_registration(rocprofiler_runtime_library_t lib,
                                uint64_t                      lib_version,
                                uint64_t                      lib_instance,
                                std::tuple<ApiTableT*...>     tables)
{
    execute_intercepts(lib, lib_version, lib_instance, tables, intercept_library_seq);
}

// template instantiation for HsaApiTable
template void notify_runtime_api_registration(rocprofiler_runtime_library_t,
                                              uint64_t,
                                              uint64_t,
                                              std::tuple<HsaApiTable*>);
}  // namespace intercept_table
}  // namespace rocprofiler

extern "C" {
rocprofiler_status_t
rocprofiler_at_runtime_api_registration(rocprofiler_intercept_library_cb_t callback,
                                        int                                libs,
                                        void*                              data)
{
    // if this function is invoked after initialization, we cannot guarantee that the runtime
    // intercept API has not already be registered and returned to the runtime.
    if(rocprofiler::registration::get_init_status() > 0)
        return ROCPROFILER_STATUS_ERROR_CONFIGURATION_LOCKED;

    if((libs & ROCPROFILER_LIBRARY) == ROCPROFILER_LIBRARY)
        return ROCPROFILER_STATUS_ERROR_INVALID_ARGUMENT;
    else if((libs & ROCPROFILER_HIP_LIBRARY) == ROCPROFILER_HIP_LIBRARY)
        return ROCPROFILER_STATUS_ERROR_NOT_IMPLEMENTED;
    else if((libs & ROCPROFILER_MARKER_LIBRARY) == ROCPROFILER_MARKER_LIBRARY)
        return ROCPROFILER_STATUS_ERROR_NOT_IMPLEMENTED;

    rocprofiler::intercept_table::update_intercepts(
        callback, libs, data, rocprofiler::intercept_table::intercept_library_seq);
    return ROCPROFILER_STATUS_SUCCESS;
}
}
