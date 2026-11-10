// MIT License
//
// Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
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

#include "lib/rocprofiler-sdk-collector/hip_records.hpp"

#include "lib/common/defines.hpp"

#include <rocprofiler-sdk/hip/compiler_api_id.h>
#include <rocprofiler-sdk/hip/runtime_api_id.h>
#include <rocprofiler-sdk/hip/table_id.h>

#include <hip/amd_detail/hip_api_trace.hpp>

#include <algorithm>
#include <array>
#include <cstring>
#include <string_view>
#include <tuple>
#include <type_traits>
#include <unordered_map>
#include <utility>
#include <vector>

namespace rocprofiler
{
namespace collector
{
namespace
{
// ------------------------------------------------------------------------------------ //
//  The probe fields of one call, by parameter
// ------------------------------------------------------------------------------------ //

struct field_ref
{
    const schema_field*     schema = nullptr;
    const ctf::field_value* value  = nullptr;
};

class arg_source
{
public:
    arg_source(const api_call& call, const schema_event& event, string_store* strings)
    : m_event{event}
    , m_strings{strings}
    {
        for(const auto& value : call.arguments)
        {
            const auto* field = find_schema_field(event, value.name);
            if(field != nullptr && field->role == "argument") m_fields.push_back({field, &value});
        }
    }

    // The fields of the parameter for the member at position of count members: the
    // parameter of the member's name, else the one at the same position when the counts
    // agree (the probe and rocprofiler-sdk may name a parameter differently).
    std::vector<field_ref> fields_of(const char* member, size_t position, size_t count) const
    {
        const auto& params = m_event.params;
        auto        param  = std::string_view{member};
        if(std::find(params.begin(), params.end(), param) == params.end())
        {
            if(params.size() != count || position >= params.size()) return {};
            param = params[position];
        }
        auto result = std::vector<field_ref>{};
        for(const auto& itr : m_fields)
            if(itr.schema->param == param) result.push_back(itr);
        return result;
    }

    string_store* strings() const { return m_strings; }

private:
    const schema_event&    m_event;
    string_store*          m_strings = nullptr;
    std::vector<field_ref> m_fields  = {};
};

// ------------------------------------------------------------------------------------ //
//  Assigning probe values to the typed union members
// ------------------------------------------------------------------------------------ //

const ctf::field_value*
single(const std::vector<field_ref>& fields)
{
    if(fields.size() != 1 || !fields.front().schema->path.empty()) return nullptr;
    return fields.front().value;
}

const ctf::field_value*
member(const std::vector<field_ref>& fields, std::string_view name)
{
    for(const auto& itr : fields)
        if(itr.schema->path.size() == 1 && itr.schema->path.front() == name) return itr.value;
    return nullptr;
}

template <typename Tp>
bool
assign_scalar(Tp* dst, const ctf::field_value* value)
{
    if(value == nullptr) return false;
    if constexpr(std::is_floating_point<Tp>::value)
    {
        *dst = static_cast<Tp>(value->f64);
    }
    else if constexpr(std::is_pointer<Tp>::value)
    {
        *dst = reinterpret_cast<Tp>(static_cast<uintptr_t>(value->u64));  // NOLINT
    }
    else if constexpr(std::is_same<Tp, bool>::value)
    {
        *dst = value->u64 != 0;
    }
    else
    {
        *dst = static_cast<Tp>(value->kind == ctf::value_kind::signed_integer ? value->i64
                                                                              : value->u64);
    }
    return true;
}

// Members of a flattened struct, by name; all must be present.
template <typename... Members>
bool
assign_members(const std::vector<field_ref>& fields, Members... members)
{
    auto all = true;
    ROCPROFILER_FOLD_EXPRESSION(
        all = assign_scalar(members.second, member(fields, members.first)) && all);
    return all;
}

template <typename Tp>
using unqualified_t = std::remove_cv_t<std::remove_reference_t<Tp>>;

template <typename Tp>
struct is_char_pointer : std::false_type
{};

template <>
struct is_char_pointer<char*> : std::true_type
{};

template <>
struct is_char_pointer<const char*> : std::true_type
{};

template <typename Tp>
constexpr bool is_char_pointer_v = is_char_pointer<Tp>::value;

template <typename Tp>
bool
assign(Tp* dst, const std::vector<field_ref>& fields, string_store* strings)
{
    using value_type = unqualified_t<Tp>;
    auto* out = const_cast<value_type*>(dst);  // NOLINT(cppcoreguidelines-pro-type-const-cast)

    if constexpr(std::is_same<value_type, rocprofiler_dim3_t>::value)
    {
        return assign_members(fields,
                              std::make_pair("x", &out->x),
                              std::make_pair("y", &out->y),
                              std::make_pair("z", &out->z));
    }
    else if constexpr(std::is_same<value_type, hipExtent>::value)
    {
        return assign_members(fields,
                              std::make_pair("width", &out->width),
                              std::make_pair("height", &out->height),
                              std::make_pair("depth", &out->depth));
    }
    else if constexpr(std::is_same<value_type, hipPitchedPtr>::value)
    {
        return assign_members(fields,
                              std::make_pair("ptr", &out->ptr),
                              std::make_pair("pitch", &out->pitch),
                              std::make_pair("xsize", &out->xsize),
                              std::make_pair("ysize", &out->ysize));
    }
    else if constexpr(std::is_same<value_type, hipPos>::value)
    {
        return assign_members(fields,
                              std::make_pair("x", &out->x),
                              std::make_pair("y", &out->y),
                              std::make_pair("z", &out->z));
    }
    else if constexpr(std::is_same<value_type, hipChannelFormatDesc>::value)
    {
        return assign_members(fields,
                              std::make_pair("x", &out->x),
                              std::make_pair("y", &out->y),
                              std::make_pair("z", &out->z),
                              std::make_pair("w", &out->w),
                              std::make_pair("f", &out->f));
    }
    else if constexpr(std::is_same<value_type, hipMemLocation>::value)
    {
        return assign_members(
            fields, std::make_pair("type", &out->type), std::make_pair("id", &out->id));
    }
    else if constexpr(is_char_pointer_v<value_type>)
    {
        // The probe records the text (at capture level full); a null pointer is recorded
        // as an empty string.
        const auto* value = single(fields);
        if(value == nullptr || value->kind != ctf::value_kind::string) return false;
        *out = const_cast<value_type>(intern_string(strings, value->str));  // NOLINT
        return true;
    }
    else if constexpr(std::is_arithmetic<value_type>::value || std::is_enum<value_type>::value ||
                      std::is_pointer<value_type>::value)
    {
        return assign_scalar(out, single(fields));
    }
    else
    {
        // A by-value struct the probes do not flatten the same way: left zero.
        (void) fields;
        (void) strings;
        return false;
    }
}

template <typename TupleT, size_t... Idx>
uint32_t
fill_members(const arg_source& source, TupleT&& members, std::index_sequence<Idx...>)
{
    constexpr auto count   = sizeof...(Idx);
    auto           missing = uint32_t{0};
    ROCPROFILER_FOLD_EXPRESSION(missing +=
                                assign(std::get<Idx>(members).second,
                                       source.fields_of(std::get<Idx>(members).first, Idx, count),
                                       source.strings())
                                    ? 0
                                    : 1);
    return missing;
}

template <typename... Members>
uint32_t
fill(const arg_source& source, Members... members)
{
    return fill_members(source, std::make_tuple(members...), std::index_sequence_for<Members...>{});
}

// ------------------------------------------------------------------------------------ //
//  Return values
// ------------------------------------------------------------------------------------ //

template <typename RetT, typename... Args>
RetT return_type_of(RetT (*)(Args...));

template <size_t TableIdx>
struct dispatch_table;

template <>
struct dispatch_table<ROCPROFILER_HIP_TABLE_ID_Runtime>
{
    using type = HipDispatchTable;
};

template <>
struct dispatch_table<ROCPROFILER_HIP_TABLE_ID_Compiler>
{
    using type = HipCompilerDispatchTable;
};

template <typename RetT>
void
set_retval(rocprofiler_hip_api_retval_t* retval, const api_call& call, string_store* strings)
{
    const ctf::field_value* result = nullptr;
    for(const auto& itr : call.results)
        if(itr.name == "result") result = &itr;
    if(result == nullptr) return;

    if constexpr(std::is_same<RetT, hipError_t>::value)
        assign_scalar(&retval->hipError_t_retval, result);
    else if constexpr(std::is_same<RetT, int>::value)
        assign_scalar(&retval->int_retval, result);
    else if constexpr(std::is_same<RetT, void**>::value)
        assign_scalar(&retval->voidpp_retval, result);
    else if constexpr(std::is_same<RetT, const char*>::value)
    {
        if(result->kind == ctf::value_kind::string)
            retval->const_charp_retval = intern_string(strings, result->str);
    }
    else
    {
        (void) retval;
        (void) strings;
    }
}

// ------------------------------------------------------------------------------------ //
//  The HIP APIs rocprofiler-sdk traces, from its own list
// ------------------------------------------------------------------------------------ //

using fill_function   = uint32_t (*)(rocprofiler_hip_api_args_t*, const arg_source&);
using retval_function = void (*)(rocprofiler_hip_api_retval_t*, const api_call&, string_store*);

struct hip_api_entry
{
    const char*                       name      = nullptr;
    rocprofiler_buffer_tracing_kind_t kind      = ROCPROFILER_BUFFER_TRACING_NONE;
    rocprofiler_tracing_operation_t   operation = 0;
    fill_function                     fill_args = nullptr;
    retval_function                   retval    = nullptr;
};

constexpr rocprofiler_buffer_tracing_kind_t
buffer_kind(size_t table)
{
    return (table == ROCPROFILER_HIP_TABLE_ID_Compiler)
               ? ROCPROFILER_BUFFER_TRACING_HIP_COMPILER_API_EXT
               : ROCPROFILER_BUFFER_TRACING_HIP_RUNTIME_API_EXT;
}

#define COLLECTOR_HIP_RETVAL(TABLE, FUNC_PTR)                                                      \
    [](rocprofiler_hip_api_retval_t* retval, const api_call& call, string_store* strings) {        \
        using return_type =                                                                        \
            decltype(return_type_of(std::declval<dispatch_table<TABLE>::type>().FUNC_PTR));        \
        set_retval<return_type>(retval, call, strings);                                            \
    }

#define HIP_API_INFO_DEFINITION_0(TABLE, API_ID, FUNC, FUNC_PTR)                                   \
    hip_api_entry{#FUNC,                                                                           \
                  buffer_kind(TABLE),                                                              \
                  API_ID,                                                                          \
                  [](rocprofiler_hip_api_args_t*, const arg_source&) { return uint32_t{0}; },      \
                  COLLECTOR_HIP_RETVAL(TABLE, FUNC_PTR)},

#define HIP_API_INFO_DEFINITION_V(TABLE, API_ID, FUNC, FUNC_PTR, ...)                              \
    hip_api_entry{#FUNC,                                                                           \
                  buffer_kind(TABLE),                                                              \
                  API_ID,                                                                          \
                  [](rocprofiler_hip_api_args_t* args, const arg_source& source) {                 \
                      auto& data = args->FUNC;                                                     \
                      return fill(source, GET_ADDR_NAMED_MEMBER_FIELDS(data, __VA_ARGS__));        \
                  },                                                                               \
                  COLLECTOR_HIP_RETVAL(TABLE, FUNC_PTR)},

// (name, address) of each member, so that assign() gets the member's exact type.
#define ADDR_NAMED_MEMBER_0(...)
#define ADDR_NAMED_MEMBER_1(PREFIX, FIELD) std::make_pair(#FIELD, &PREFIX.FIELD)
#define ADDR_NAMED_MEMBER_2(PREFIX, A, B)                                                          \
    ADDR_NAMED_MEMBER_1(PREFIX, A), ADDR_NAMED_MEMBER_1(PREFIX, B)
#define ADDR_NAMED_MEMBER_3(PREFIX, A, B, C)                                                       \
    ADDR_NAMED_MEMBER_2(PREFIX, A, B), ADDR_NAMED_MEMBER_1(PREFIX, C)
#define ADDR_NAMED_MEMBER_4(PREFIX, A, B, C, D)                                                    \
    ADDR_NAMED_MEMBER_3(PREFIX, A, B, C), ADDR_NAMED_MEMBER_1(PREFIX, D)
#define ADDR_NAMED_MEMBER_5(PREFIX, A, B, C, D, E)                                                 \
    ADDR_NAMED_MEMBER_4(PREFIX, A, B, C, D), ADDR_NAMED_MEMBER_1(PREFIX, E)
#define ADDR_NAMED_MEMBER_6(PREFIX, A, B, C, D, E, F)                                              \
    ADDR_NAMED_MEMBER_5(PREFIX, A, B, C, D, E), ADDR_NAMED_MEMBER_1(PREFIX, F)
#define ADDR_NAMED_MEMBER_7(PREFIX, A, B, C, D, E, F, G)                                           \
    ADDR_NAMED_MEMBER_6(PREFIX, A, B, C, D, E, F), ADDR_NAMED_MEMBER_1(PREFIX, G)
#define ADDR_NAMED_MEMBER_8(PREFIX, A, B, C, D, E, F, G, H)                                        \
    ADDR_NAMED_MEMBER_7(PREFIX, A, B, C, D, E, F, G), ADDR_NAMED_MEMBER_1(PREFIX, H)
#define ADDR_NAMED_MEMBER_9(PREFIX, A, B, C, D, E, F, G, H, I)                                     \
    ADDR_NAMED_MEMBER_8(PREFIX, A, B, C, D, E, F, G, H), ADDR_NAMED_MEMBER_1(PREFIX, I)
#define ADDR_NAMED_MEMBER_10(PREFIX, A, B, C, D, E, F, G, H, I, J)                                 \
    ADDR_NAMED_MEMBER_9(PREFIX, A, B, C, D, E, F, G, H, I), ADDR_NAMED_MEMBER_1(PREFIX, J)
#define ADDR_NAMED_MEMBER_11(PREFIX, A, B, C, D, E, F, G, H, I, J, K)                              \
    ADDR_NAMED_MEMBER_10(PREFIX, A, B, C, D, E, F, G, H, I, J), ADDR_NAMED_MEMBER_1(PREFIX, K)
#define ADDR_NAMED_MEMBER_12(PREFIX, A, B, C, D, E, F, G, H, I, J, K, L)                           \
    ADDR_NAMED_MEMBER_11(PREFIX, A, B, C, D, E, F, G, H, I, J, K), ADDR_NAMED_MEMBER_1(PREFIX, L)
#define ADDR_NAMED_MEMBER_13(PREFIX, A, B, C, D, E, F, G, H, I, J, K, L, M)                        \
    ADDR_NAMED_MEMBER_12(PREFIX, A, B, C, D, E, F, G, H, I, J, K, L), ADDR_NAMED_MEMBER_1(PREFIX, M)
#define ADDR_NAMED_MEMBER_14(PREFIX, A, B, C, D, E, F, G, H, I, J, K, L, M, N)                     \
    ADDR_NAMED_MEMBER_13(PREFIX, A, B, C, D, E, F, G, H, I, J, K, L, M),                           \
        ADDR_NAMED_MEMBER_1(PREFIX, N)
#define ADDR_NAMED_MEMBER_15(PREFIX, A, B, C, D, E, F, G, H, I, J, K, L, M, N, O)                  \
    ADDR_NAMED_MEMBER_14(PREFIX, A, B, C, D, E, F, G, H, I, J, K, L, M, N),                        \
        ADDR_NAMED_MEMBER_1(PREFIX, O)
#define GET_ADDR_NAMED_MEMBER_FIELDS(VAR, ...)                                                     \
    IMPL_DETAIL_FOR_EACH(ADDR_NAMED_MEMBER_, VAR, __VA_ARGS__)

// clang-format off
const hip_api_entry hip_api_entries[] = {
#include "rocprofiler-sdk-collector/hip_api_list.def"
};
// clang-format on

#undef HIP_API_INFO_DEFINITION_0
#undef HIP_API_INFO_DEFINITION_V
#undef COLLECTOR_HIP_RETVAL

const std::unordered_map<std::string_view, const hip_api_entry*>&
get_hip_api_index()
{
    static const auto* index = [] {
        auto* result = new std::unordered_map<std::string_view, const hip_api_entry*>{};
        for(const auto& itr : hip_api_entries)
            result->emplace(itr.name, &itr);
        return result;
    }();
    return *index;
}

const hip_api_entry*
find_hip_api(const std::string& api)
{
    const auto& index = get_hip_api_index();
    auto        itr   = index.find(api);
    return (itr == index.end()) ? nullptr : itr->second;
}
}  // namespace

const char*
intern_string(string_store* store, const std::string& value)
{
    return store->values.emplace(value).first->c_str();
}

bool
find_hip_operation(const std::string&                 api,
                   rocprofiler_buffer_tracing_kind_t* kind,
                   rocprofiler_tracing_operation_t*   operation)
{
    const auto* entry = find_hip_api(api);
    if(entry == nullptr) return false;
    *kind      = entry->kind;
    *operation = entry->operation;
    return true;
}

hip_record_status
make_hip_api_record(const api_call&                                  call,
                    const schema_event&                              enter,
                    const schema_event*                              exit,
                    const clock_conversion&                          clocks,
                    string_store*                                    strings,
                    rocprofiler_buffer_tracing_hip_api_ext_record_t* record)
{
    (void) exit;
    auto status = hip_record_status{};
    std::memset(static_cast<void*>(record), 0, sizeof(*record));
    record->size = sizeof(*record);

    const auto* entry = find_hip_api(call.name);
    if(entry == nullptr) return status;
    status.known_api = true;

    record->kind                    = entry->kind;
    record->operation               = entry->operation;
    record->correlation_id.internal = call.correlation_id;
    record->correlation_id.ancestor = call.parent_id;
    record->thread_id               = call.tid;
    record->start_timestamp         = to_boottime(clocks, call.start);
    record->end_timestamp =
        (call.end == no_end) ? record->start_timestamp : to_boottime(clocks, call.end);

    auto source         = arg_source{call, enter, strings};
    status.missing_args = entry->fill_args(&record->args, source);
    entry->retval(&record->retval, call, strings);
    return status;
}
}  // namespace collector
}  // namespace rocprofiler
