// Copyright (c) 2018-2023 Advanced Micro Devices, Inc.
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

#include "lib/rocprofiler/hsa/hsa.hpp"

#include "lib/common/defines.hpp"
#include "lib/rocprofiler/hsa/ostream.hpp"
#include "lib/rocprofiler/hsa/types.hpp"
#include "lib/rocprofiler/hsa/utils.hpp"

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <type_traits>
#include <utility>

namespace rocprofiler
{
namespace hsa
{
namespace
{
std::atomic<activity_functor_t> report_activity = {};

struct null_type
{};

template <typename DataT, typename Tp>
void
set_data_retval(DataT& _data, Tp _val)
{
    if constexpr(std::is_same<Tp, hsa_signal_value_t>::value)
    {
        _data.hsa_signal_value_t_retval = _val;
    }
    else if constexpr(std::is_same<Tp, uint64_t>::value)
    {
        _data.uint64_t_retval = _val;
    }
    else if constexpr(std::is_same<Tp, uint32_t>::value)
    {
        _data.uint32_t_retval = _val;
    }
    else if constexpr(std::is_same<Tp, hsa_status_t>::value)
    {
        _data.hsa_status_t_retval = _val;
    }
    else
    {
        static_assert(std::is_void<Tp>::value, "Error! unsupported return type");
    }
}
}  // namespace

hsa_api_table_t&
get_table()
{
    static auto _core     = CoreApiTable{};
    static auto _amd_ext  = AmdExtTable{};
    static auto _img_ext  = ImageExtTable{};
    static auto _fini_ext = FinalizerExtTable{};
    static auto _v        = []() {
        _core.version = {
            HSA_CORE_API_TABLE_MAJOR_VERSION, sizeof(_core), HSA_CORE_API_TABLE_STEP_VERSION, 0};
        _amd_ext.version  = {HSA_AMD_EXT_API_TABLE_MAJOR_VERSION,
                            sizeof(_amd_ext),
                            HSA_AMD_EXT_API_TABLE_STEP_VERSION,
                            0};
        _img_ext.version  = {HSA_IMAGE_API_TABLE_MAJOR_VERSION,
                            sizeof(_img_ext),
                            HSA_IMAGE_API_TABLE_STEP_VERSION,
                            0};
        _fini_ext.version = {HSA_FINALIZER_API_TABLE_MAJOR_VERSION,
                             sizeof(_fini_ext),
                             HSA_FINALIZER_API_TABLE_STEP_VERSION,
                             0};
        auto _version     = ApiTableVersion{
            HSA_API_TABLE_MAJOR_VERSION, sizeof(HsaApiTable), HSA_API_TABLE_STEP_VERSION, 0};
        auto _val = hsa_api_table_t{_version, &_core, &_amd_ext, &_fini_ext, &_img_ext};
        return _val;
    }();
    return _v;
}

template <size_t Idx>
template <typename DataT, typename DataArgsT, typename... Args>
auto
hsa_api_impl<Idx>::phase_enter(DataT& _data, DataArgsT& _data_args, Args... args)
{
    using info_type = hsa_api_info<Idx>;

    activity_functor_t _func = report_activity.load(std::memory_order_relaxed);
    if(_func)
    {
        if constexpr(Idx == ROCPROFILER_HSA_API_ID_hsa_amd_memory_async_copy_rect)
        {
            auto _tuple                                            = std::make_tuple(args...);
            _data.api_data.args.hsa_amd_memory_async_copy_rect.dst = std::get<0>(_tuple);
            _data.api_data.args.hsa_amd_memory_async_copy_rect.dst_offset = std::get<1>(_tuple);
            _data.api_data.args.hsa_amd_memory_async_copy_rect.src        = std::get<2>(_tuple);
            _data.api_data.args.hsa_amd_memory_async_copy_rect.src_offset = std::get<3>(_tuple);
            _data.api_data.args.hsa_amd_memory_async_copy_rect.range      = std::get<4>(_tuple);
            _data.api_data.args.hsa_amd_memory_async_copy_rect.range__val = *(std::get<4>(_tuple));
            _data.api_data.args.hsa_amd_memory_async_copy_rect.copy_agent = std::get<5>(_tuple);
            _data.api_data.args.hsa_amd_memory_async_copy_rect.dir        = std::get<6>(_tuple);
            _data.api_data.args.hsa_amd_memory_async_copy_rect.num_dep_signals =
                std::get<7>(_tuple);
            _data.api_data.args.hsa_amd_memory_async_copy_rect.dep_signals = std::get<8>(_tuple);
            _data.api_data.args.hsa_amd_memory_async_copy_rect.completion_signal =
                std::get<9>(_tuple);
        }
        else
        {
            _data_args = DataArgsT{args...};
        }
        if(_func(info_type::domain_idx, info_type::operation_idx, &_data) == 0)
        {
            if(_data.phase_enter != nullptr) _data.phase_enter(info_type::operation_idx, &_data);
            return true;
        }
        return false;
    }
    return false;
}

template <size_t Idx>
template <typename DataT, typename... Args>
auto
hsa_api_impl<Idx>::phase_exit(DataT& _data)
{
    using info_type = hsa_api_info<Idx>;

    if(_data.phase_exit != nullptr)
    {
        _data.phase_exit(info_type::operation_idx, &_data);
        return true;
    }
    return false;
}

template <size_t Idx>
template <typename DataT, typename FuncT, typename... Args>
auto
hsa_api_impl<Idx>::exec(DataT& _data, FuncT&& _func, Args&&... args)
{
    using return_type = std::decay_t<std::invoke_result_t<FuncT, Args...>>;

    if(_func)
    {
        static_assert(std::is_void<return_type>::value || std::is_enum<return_type>::value ||
                          std::is_integral<return_type>::value,
                      "Error! unsupported return type");

        if constexpr(std::is_void<return_type>::value)
        {
            _func(std::forward<Args>(args)...);
            return null_type{};
        }
        else
        {
            auto _ret = _func(std::forward<Args>(args)...);
            set_data_retval(_data.api_data, _ret);
            return _ret;
        }
    }

    if constexpr(std::is_void<return_type>::value)
        return null_type{};
    else
        return return_type{HSA_STATUS_ERROR};
}

template <size_t Idx>
template <typename... Args>
auto
hsa_api_impl<Idx>::functor(Args&&... args)
{
    using info_type = hsa_api_info<Idx>;

    auto trace_data = rocprofiler_hsa_trace_data_t{};

    auto _enabled = phase_enter(
        trace_data, info_type::get_api_data_args(trace_data), std::forward<Args>(args)...);

    auto _ret = exec(trace_data, info_type::get_table_func(), std::forward<Args>(args)...);

    if(_enabled) phase_exit(trace_data);

    if constexpr(!std::is_same<decltype(_ret), null_type>::value)
        return _ret;
    else
        return HSA_STATUS_SUCCESS;
}
}  // namespace hsa
}  // namespace rocprofiler

// template specializations
#include "hsa.def.cpp"

namespace rocprofiler
{
namespace hsa
{
namespace
{
template <size_t Idx, size_t... IdxTail>
const char*
hsa_api_name(const uint32_t id, std::index_sequence<Idx, IdxTail...>)
{
    if(Idx == id) return hsa_api_info<Idx>::name;
    if constexpr(sizeof...(IdxTail) > 0)
        return hsa_api_name(id, std::index_sequence<IdxTail...>{});
    else
        return nullptr;
}

template <size_t Idx, size_t... IdxTail>
uint32_t
hsa_api_id_by_name(const char* name, std::index_sequence<Idx, IdxTail...>)
{
    if(std::string_view{hsa_api_info<Idx>::name} == std::string_view{name})
        return hsa_api_info<Idx>::operation_idx;
    if constexpr(sizeof...(IdxTail) > 0)
        return hsa_api_id_by_name(name, std::index_sequence<IdxTail...>{});
    else
        return ROCPROFILER_HSA_API_ID_NONE;
}

template <size_t Idx, size_t... IdxTail>
std::string
hsa_api_data_string(const uint32_t                      id,
                    const rocprofiler_hsa_trace_data_t& _data,
                    std::index_sequence<Idx, IdxTail...>)
{
    if(Idx == id) return hsa_api_info<Idx>::as_string(_data);
    if constexpr(sizeof...(IdxTail) > 0)
        return hsa_api_data_string(id, _data, std::index_sequence<IdxTail...>{});
    else
        return std::string{};
}

template <size_t Idx, size_t... IdxTail>
std::string
hsa_api_named_data_string(const uint32_t                      id,
                          const rocprofiler_hsa_trace_data_t& _data,
                          std::index_sequence<Idx, IdxTail...>)
{
    if(Idx == id) return hsa_api_info<Idx>::as_named_string(_data);
    if constexpr(sizeof...(IdxTail) > 0)
        return hsa_api_named_data_string(id, _data, std::index_sequence<IdxTail...>{});
    else
        return std::string{};
}

template <size_t Idx, size_t... IdxTail>
void
hsa_api_iterate_args(const uint32_t                      id,
                     const rocprofiler_hsa_trace_data_t& _data,
                     int (*_func)(const char*, const char*),
                     std::index_sequence<Idx, IdxTail...>)
{
    if(Idx == id)
    {
        for(auto&& itr : hsa_api_info<Idx>::as_arg_list(_data))
        {
            _func(itr.first.c_str(), itr.second.c_str());
        }
    }
    if constexpr(sizeof...(IdxTail) > 0)
        hsa_api_iterate_args(id, _data, _func, std::index_sequence<IdxTail...>{});
}

template <size_t... Idx>
void
hsa_api_get_ids(std::vector<uint32_t>& _id_list, std::index_sequence<Idx...>)
{
    auto _emplace = [](auto& _vec, uint32_t _v) {
        if(_v < ROCPROFILER_HSA_API_ID_LAST) _vec.emplace_back(_v);
    };

    (_emplace(_id_list, hsa_api_info<Idx>::operation_idx), ...);
}

template <size_t... Idx>
void
hsa_api_get_names(std::vector<const char*>& _name_list, std::index_sequence<Idx...>)
{
    auto _emplace = [](auto& _vec, const char* _v) {
        if(_v != nullptr && strnlen(_v, 1) > 0) _vec.emplace_back(_v);
    };

    (_emplace(_name_list, hsa_api_info<Idx>::name), ...);
}

template <size_t... Idx>
void
hsa_api_update_table(hsa_api_table_t* _orig, std::index_sequence<Idx...>)
{
    auto _update = [](hsa_api_table_t* _orig_v, auto _info) {
        // 1. get the sub-table containing the function pointer
        // 2. get reference to function pointer in sub-table
        // 3. update function pointer with functor
        auto& _table = _info.get_table(_orig_v);
        auto& _func  = _info.get_table_func(_table);
        _func        = _info.get_functor(_func);
    };

    (_update(_orig, hsa_api_info<Idx>{}), ...);
}
}  // namespace

// check out the assembly here... this compiles to a switch statement
const char*
hsa_api_name(uint32_t id)
{
    return hsa_api_name(id, std::make_index_sequence<ROCPROFILER_HSA_API_ID_LAST>{});
}

uint32_t
hsa_api_id_by_name(const char* name)
{
    return hsa_api_id_by_name(name, std::make_index_sequence<ROCPROFILER_HSA_API_ID_LAST>{});
}

std::string
hsa_api_data_string(uint32_t id, const rocprofiler_hsa_trace_data_t& _data)
{
    return hsa_api_data_string(id, _data, std::make_index_sequence<ROCPROFILER_HSA_API_ID_LAST>{});
}

std::string
hsa_api_named_data_string(uint32_t id, const rocprofiler_hsa_trace_data_t& _data)
{
    return hsa_api_named_data_string(
        id, _data, std::make_index_sequence<ROCPROFILER_HSA_API_ID_LAST>{});
}

void
hsa_api_iterate_args(uint32_t                            id,
                     const rocprofiler_hsa_trace_data_t& _data,
                     int (*_func)(const char*, const char*))
{
    if(_func)
        hsa_api_iterate_args(
            id, _data, _func, std::make_index_sequence<ROCPROFILER_HSA_API_ID_LAST>{});
}

std::vector<uint32_t>
hsa_api_get_ids()
{
    auto _data = std::vector<uint32_t>{};
    _data.reserve(ROCPROFILER_HSA_API_ID_LAST);
    hsa_api_get_ids(_data, std::make_index_sequence<ROCPROFILER_HSA_API_ID_LAST>{});
    return _data;
}

std::vector<const char*>
hsa_api_get_names()
{
    auto _data = std::vector<const char*>{};
    _data.reserve(ROCPROFILER_HSA_API_ID_LAST);
    hsa_api_get_names(_data, std::make_index_sequence<ROCPROFILER_HSA_API_ID_LAST>{});
    return _data;
}

void
hsa_api_set_callback(activity_functor_t _func)
{
    auto&& _v = report_activity.load();
    report_activity.compare_exchange_strong(_v, _func);
}

void
hsa_api_update_table(hsa_api_table_t* _orig)
{
    if(_orig) hsa_api_update_table(_orig, std::make_index_sequence<ROCPROFILER_HSA_API_ID_LAST>{});
}
}  // namespace hsa
}  // namespace rocprofiler

extern "C" {
bool
OnLoad(HsaApiTable*       table,
       uint64_t           runtime_version,
       uint64_t           failed_tool_count,
       const char* const* failed_tool_names)
{
    (void) runtime_version;
    (void) failed_tool_count;
    (void) failed_tool_names;

    fprintf(stderr, "[%s:%i] %s\n", __FILE__, __LINE__, __FUNCTION__);

    auto& _saved = rocprofiler::hsa::get_table();
    ::copyTables(table, &_saved);

    rocprofiler::hsa::hsa_api_update_table(table);

    return true;
}
}

/*
#include <iomanip>

int
main()
{
    rocprofiler::hsa::activity_functor_t _cb =
        [](rocprofiler_tracer_activity_domain_t domain, uint32_t operation_id, void* data) {
            const auto* _name    = rocprofiler::hsa::hsa_api_name(operation_id);
            auto        _name_id = rocprofiler::hsa::hsa_api_id_by_name(_name);
            auto&       _data    = *static_cast<rocprofiler::hsa::hsa_trace_data_t*>(data);
            std::cout << "[cb] domain=" << domain << ", op_id=" << operation_id << ", data=" << data
                      << ", name=" << _name << ", name_id=" << _name_id << ", named_string='"
                      << rocprofiler::hsa::hsa_api_named_data_string(operation_id, _data) << "'"
                      << "\n";
            auto _func = [](const char* name, const char* value) {
                std::cout << "    " << std::setw(20) << name << " = " << value << "\n";
                return 0;
            };
            rocprofiler::hsa::hsa_api_iterate_args(operation_id, _data, _func);
            return 0;
        };

    rocprofiler::hsa::report_activity.store(_cb);

    {
        double                 val         = 40;
        hsa_code_object_t      code_object = {};
        hsa_code_object_info_t attribute   = HSA_CODE_OBJECT_INFO_TYPE;
        void*                  value       = &val;

        auto _func =
            rocprofiler::hsa::hsa_api_info<HSA_API_ID_hsa_code_object_get_info>::get_functor();
        _func(code_object, attribute, value);
    }

    {
        bool     result = false;
        uint16_t ext    = 1;
        uint16_t major  = 4;
        uint16_t minor  = 2;

        auto _func = rocprofiler::hsa::hsa_api_info<
            HSA_API_ID_hsa_system_extension_supported>::get_functor();
        _func(ext, major, minor, &result);
    }
}
*/
