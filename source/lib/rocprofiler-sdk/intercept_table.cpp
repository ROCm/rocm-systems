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

#include "lib/common/container/stable_vector.hpp"
#include "lib/common/utility.hpp"
#include "lib/rocprofiler-sdk/buffer.hpp"
#include "lib/rocprofiler-sdk/context/context.hpp"
#include "lib/rocprofiler-sdk/internal_threading.hpp"
#include "lib/rocprofiler-sdk/rccl/rccl.hpp"
#include "lib/rocprofiler-sdk/registration.hpp"
#include "lib/rocprofiler-sdk/rocdecode/rocdecode.hpp"
#include "lib/rocprofiler-sdk/rocjpeg/rocjpeg.hpp"

#include <rocprofiler-sdk/fwd.h>
#include <rocprofiler-sdk/hip.h>
#include <rocprofiler-sdk/hsa.h>
#include <rocprofiler-sdk/intercept_table.h>

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
template <rocprofiler_intercept_table_t... Idx>
using library_sequence_t = std::integer_sequence<rocprofiler_intercept_table_t, Idx...>;

#define ROCPROFILER_INTERCEPT_TABLE_KIND_STRING(TABLE, NAME)                                       \
    template <>                                                                                    \
    struct intercept_table_info<ROCPROFILER_##TABLE##_TABLE>                                       \
    {                                                                                              \
        static constexpr auto value  = ROCPROFILER_##TABLE##_TABLE;                                \
        static constexpr auto name   = NAME;                                                       \
        static constexpr auto length = std::string_view{NAME}.length();                            \
        static constexpr auto data   = std::pair<const char*, size_t>{name, length};               \
    };

template <rocprofiler_intercept_table_t Idx>
struct intercept_table_info;

ROCPROFILER_INTERCEPT_TABLE_KIND_STRING(HSA, "HSA")
ROCPROFILER_INTERCEPT_TABLE_KIND_STRING(HIP_RUNTIME, "HIP (runtime)")
ROCPROFILER_INTERCEPT_TABLE_KIND_STRING(HIP_COMPILER, "HIP (compiler)")
ROCPROFILER_INTERCEPT_TABLE_KIND_STRING(MARKER_CORE, "MARKER (ROCTx)")
ROCPROFILER_INTERCEPT_TABLE_KIND_STRING(MARKER_CONTROL, "MARKER (ROCTx Control)")
ROCPROFILER_INTERCEPT_TABLE_KIND_STRING(MARKER_NAME, "MARKER (ROCTx Name)")
ROCPROFILER_INTERCEPT_TABLE_KIND_STRING(RCCL, "RCCL")
ROCPROFILER_INTERCEPT_TABLE_KIND_STRING(ROCDECODE, "rocDecode")
ROCPROFILER_INTERCEPT_TABLE_KIND_STRING(ROCJPEG, "rocJPEG")

// this is used to loop over the different libraries
constexpr auto intercept_library_seq = library_sequence_t<ROCPROFILER_HSA_TABLE,
                                                          ROCPROFILER_HIP_RUNTIME_TABLE,
                                                          ROCPROFILER_HIP_COMPILER_TABLE,
                                                          ROCPROFILER_MARKER_CORE_TABLE,
                                                          ROCPROFILER_MARKER_CONTROL_TABLE,
                                                          ROCPROFILER_MARKER_NAME_TABLE,
                                                          ROCPROFILER_RCCL_TABLE,
                                                          ROCPROFILER_ROCDECODE_TABLE,
                                                          ROCPROFILER_ROCJPEG_TABLE>{};

// check that intercept_library_seq is up to date
static_assert((1 << (intercept_library_seq.size() - 1)) == ROCPROFILER_TABLE_LAST,
              "Update intercept_library_seq to include new libraries");

// data structure holding list of callbacks
template <rocprofiler_intercept_table_t LibT>
struct intercept
{
    static constexpr auto value = LibT;

    std::vector<rocprofiler_intercept_library_cb_t> callbacks = {};
    std::vector<void*>                              user_data = {};
    std::mutex                                      mutex     = {};
};

// static accessor for intercept instance
template <rocprofiler_intercept_table_t LibT>
auto&
get_intercept()
{
    static auto _v = intercept<LibT>{};
    return _v;
}

// adds callbacks to intercept instance(s)
template <rocprofiler_intercept_table_t Idx, rocprofiler_intercept_table_t... Tail>
auto
get_intercept_table_info(rocprofiler_intercept_table_t kind, library_sequence_t<Idx, Tail...>)
{
    using info_type   = intercept_table_info<Idx>;
    using return_type = decltype(info_type::data);

    if(info_type::value == kind)
    {
        return info_type::data;
    }

    if constexpr(sizeof...(Tail) > 0)
        return get_intercept_table_info(kind, library_sequence_t<Tail...>{});

    return return_type{nullptr, 0};
}

// adds callbacks to intercept instance(s)
template <rocprofiler_intercept_table_t... Idx>
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
template <typename... ApiTableT, rocprofiler_intercept_table_t... Idx>
void
execute_intercepts(rocprofiler_intercept_table_t lib,
                   uint64_t                      lib_version,
                   uint64_t                      lib_instance,
                   std::tuple<ApiTableT*...>     tables,
                   std::integer_sequence<rocprofiler_intercept_table_t, Idx...>)
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
notify_intercept_table_registration(rocprofiler_intercept_table_t lib,
                                    uint64_t                      lib_version,
                                    uint64_t                      lib_instance,
                                    std::tuple<ApiTableT*...>     tables)
{
    execute_intercepts(lib, lib_version, lib_instance, tables, intercept_library_seq);
}

// template instantiation for HsaApiTable
template void notify_intercept_table_registration(rocprofiler_intercept_table_t,
                                                  uint64_t,
                                                  uint64_t,
                                                  std::tuple<HsaApiTable*>);

template void notify_intercept_table_registration(rocprofiler_intercept_table_t,
                                                  uint64_t,
                                                  uint64_t,
                                                  std::tuple<roctxCoreApiTable_t*>);

template void notify_intercept_table_registration(rocprofiler_intercept_table_t,
                                                  uint64_t,
                                                  uint64_t,
                                                  std::tuple<roctxControlApiTable_t*>);

template void notify_intercept_table_registration(rocprofiler_intercept_table_t,
                                                  uint64_t,
                                                  uint64_t,
                                                  std::tuple<roctxNameApiTable_t*>);

template void notify_intercept_table_registration(rocprofiler_intercept_table_t,
                                                  uint64_t,
                                                  uint64_t,
                                                  std::tuple<HipDispatchTable*>);

template void notify_intercept_table_registration(rocprofiler_intercept_table_t,
                                                  uint64_t,
                                                  uint64_t,
                                                  std::tuple<HipCompilerDispatchTable*>);

template void notify_intercept_table_registration(rocprofiler_intercept_table_t,
                                                  uint64_t,
                                                  uint64_t,
                                                  std::tuple<rcclApiFuncTable*>);

template void notify_intercept_table_registration(rocprofiler_intercept_table_t,
                                                  uint64_t,
                                                  uint64_t,
                                                  std::tuple<RocDecodeDispatchTable*>);

template void notify_intercept_table_registration(rocprofiler_intercept_table_t,
                                                  uint64_t,
                                                  uint64_t,
                                                  std::tuple<RocJpegDispatchTable*>);
}  // namespace intercept_table
}  // namespace rocprofiler

extern "C" {
rocprofiler_status_t
rocprofiler_query_intercept_table_name(rocprofiler_intercept_table_t kind,
                                       const char**                  name,
                                       uint64_t*                     name_len)
{
    auto&& val = rocprofiler::intercept_table::get_intercept_table_info(
        kind, rocprofiler::intercept_table::intercept_library_seq);

    if(name) *name = val.first;
    if(name_len) *name_len = val.second;

    return (val.first) ? ROCPROFILER_STATUS_SUCCESS : ROCPROFILER_STATUS_ERROR_KIND_NOT_FOUND;
}

rocprofiler_status_t
rocprofiler_at_intercept_table_registration(rocprofiler_intercept_library_cb_t callback,
                                            int                                libs,
                                            void*                              data)
{
    // if this function is invoked after initialization, we cannot guarantee that the runtime
    // intercept API has not already be registered and returned to the runtime.
    if(rocprofiler::registration::get_init_status() > 0)
        return ROCPROFILER_STATUS_ERROR_CONFIGURATION_LOCKED;

    ROCP_WARNING_IF(libs == 0) << "invoking " << __FUNCTION__ << " with a value of zero is a no-op";

    rocprofiler::intercept_table::update_intercepts(
        callback, libs, data, rocprofiler::intercept_table::intercept_library_seq);
    return ROCPROFILER_STATUS_SUCCESS;
}
}
