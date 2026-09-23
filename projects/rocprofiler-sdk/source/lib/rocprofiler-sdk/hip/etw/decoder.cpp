// MIT License
//
// Copyright (c) 2023-2026 Advanced Micro Devices, Inc. All rights reserved.
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

#include "lib/rocprofiler-sdk/hip/etw/decoder.hpp"

#include "lib/common/logging.hpp"
#include "lib/common/utility.hpp"
#include "lib/rocprofiler-sdk/context/correlation_id.hpp"
#include "lib/rocprofiler-sdk/hip/names.hpp"
#include "lib/rocprofiler-sdk/tracing/fwd.hpp"
#include "lib/rocprofiler-sdk/tracing/tracing.hpp"

#include <rocprofiler-sdk/buffer_tracing.h>
#include <rocprofiler-sdk/fwd.h>
#include <rocprofiler-sdk/hip/table_id.h>

#include <evntrace.h>
#include <tdh.h>

#include <array>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <unordered_map>
#include <vector>

namespace rocprofiler
{
namespace hip
{
namespace etw
{
namespace
{
// hip::trace::domain_t in clr/hipamd/src/trace/hip_trace_etw.hpp.
constexpr uint32_t producer_domain_runtime  = 0;
constexpr uint32_t producer_domain_compiler = 1;

// ROCPROFILER_HIP_RUNTIME_API_ID_NONE and ROCPROFILER_HIP_COMPILER_API_ID_NONE are both -1.
constexpr uint32_t invalid_operation_id = static_cast<uint32_t>(-1);

// The producer also carries op_id, but that is clr's HIP_API_ID_* numbering, which is not the
// SDK's ROCPROFILER_HIP_*_API_ID_* numbering. op_name is the bridge between the two, so only
// its offset is recorded here; op_id is still validated below so that a producer schema change
// is caught rather than silently misparsed.
struct event_schema
{
    bool     valid          = false;
    uint32_t domain_offset  = 0;
    uint32_t op_name_offset = 0;
};

struct record_data
{
    uint64_t start_timestamp = 0;
    uint64_t thread_id       = 0;
    uint32_t domain          = 0;
    uint32_t operation       = 0;
};

struct domain_mapping
{
    rocprofiler_buffer_tracing_kind_t buffered     = ROCPROFILER_BUFFER_TRACING_NONE;
    rocprofiler_buffer_tracing_kind_t buffered_ext = ROCPROFILER_BUFFER_TRACING_NONE;
    uint32_t                          table        = 0;
};

// Everything below is touched only from the single thread running ProcessTrace(), so none of
// it is synchronized. set_process_filter() and reset_decoder() are called from
// start_session()/stop_session() while that thread is not running.
uint32_t&
get_process_filter()
{
    static uint32_t _v = 0;
    return _v;
}

decoder_stats&
get_stats()
{
    static auto _v = decoder_stats{};
    return _v;
}

auto&
get_schemas()
{
    static auto _v = std::unordered_map<uint64_t, event_schema>{};
    return _v;
}

// TraceLogging leaves EVENT_DESCRIPTOR zeroed, so an event type is not identifiable by
// id/version/opcode. What does identify it is the self-describing metadata blob ETW attaches to
// every event, which is a verbatim copy of a static structure in the provider and therefore
// identical across all events of one type. Hashing it yields exactly one cache entry per event
// type. Returns 0 when the event carries no TraceLogging metadata, which read_event_schema then
// rejects.
uint64_t
get_schema_key(const EVENT_RECORD* event_record)
{
    for(auto i = 0U; i < event_record->ExtendedDataCount; ++i)
    {
        const auto& item = event_record->ExtendedData[i];  // NOLINT

        if(item.ExtType != EVENT_HEADER_EXT_TYPE_EVENT_SCHEMA_TL) continue;

        const auto* data = reinterpret_cast<const uint8_t*>(item.DataPtr);
        auto        hash = uint64_t{0xcbf29ce484222325};

        for(auto j = 0U; j < item.DataSize; ++j)
            hash = (hash ^ data[j]) * uint64_t{0x100000001b3};  // NOLINT

        return hash;
    }

    return 0;
}

event_schema
read_event_schema(const EVENT_RECORD* event_record)
{
    struct expected_property
    {
        const wchar_t* name    = nullptr;
        USHORT         in_type = 0;
    };

    static const auto expected =
        std::array<expected_property, 5>{{{L"domain", TDH_INTYPE_UINT32},
                                          {L"op_id", TDH_INTYPE_UINT32},
                                          {L"op_name", TDH_INTYPE_ANSISTRING},
                                          {L"start_qpc", TDH_INTYPE_UINT64},
                                          {L"retval", TDH_INTYPE_INT32}}};

    auto  schema         = event_schema{};
    auto* mutable_record = const_cast<EVENT_RECORD*>(event_record);

    auto size   = ULONG{0};
    auto status = ::TdhGetEventInformation(mutable_record, 0, nullptr, nullptr, &size);
    if(status != ERROR_INSUFFICIENT_BUFFER) return schema;

    auto  buffer = std::vector<std::byte>(size);
    auto* info   = reinterpret_cast<TRACE_EVENT_INFO*>(buffer.data());

    status = ::TdhGetEventInformation(mutable_record, 0, nullptr, info, &size);
    if(status != ERROR_SUCCESS) return schema;

    if(info->DecodingSource != DecodingSourceTlg || info->EventNameOffset == 0) return schema;

    const auto* event_name =
        reinterpret_cast<const wchar_t*>(buffer.data() + info->EventNameOffset);

    if(::wcscmp(event_name, L"hip_api") != 0) return schema;

    constexpr auto property_count = 5U;
    if(info->TopLevelPropertyCount != property_count)
    {
        ROCP_WARNING << "rocm_hip_tlg event has " << info->TopLevelPropertyCount
                     << " properties, expected " << property_count;
        return schema;
    }

    for(auto i = 0U; i < property_count; ++i)
    {
        const auto& property = info->EventPropertyInfoArray[i];  // NOLINT

        if(property.NameOffset == 0) return schema;

        const auto* name = reinterpret_cast<const wchar_t*>(buffer.data() + property.NameOffset);

        if(::wcscmp(name, expected.at(i).name) != 0 ||
           property.nonStructType.InType != expected.at(i).in_type)
        {
            ROCP_WARNING << "rocm_hip_tlg property " << i << " does not match the expected schema";
            return schema;
        }
    }

    // domain and op_id are fixed-width and lead the payload. TraceLogging packs the fields
    // after the variable-length op_name inline, so their offsets are computed per event.
    schema.domain_offset  = 0;
    schema.op_name_offset = 2 * sizeof(uint32_t);
    schema.valid          = true;

    return schema;
}

// Every read below is bounds checked: the payload comes from another process and a truncated
// or unterminated buffer must drop the event rather than read past it.
template <typename Tp>
bool
read_scalar(const EVENT_RECORD* event_record, uint32_t offset, Tp& out)
{
    if(static_cast<size_t>(offset) + sizeof(Tp) > event_record->UserDataLength) return false;

    ::memcpy(&out, static_cast<const std::byte*>(event_record->UserData) + offset, sizeof(Tp));
    return true;
}

const char*
read_ansi_string(const EVENT_RECORD* event_record, uint32_t offset, uint32_t& next_offset)
{
    if(offset >= event_record->UserDataLength) return nullptr;

    const auto* base      = static_cast<const char*>(event_record->UserData);
    const auto  available = static_cast<size_t>(event_record->UserDataLength) - offset;
    const auto  length    = ::strnlen(base + offset, available);

    if(length == available) return nullptr;

    next_offset = offset + static_cast<uint32_t>(length) + 1U;
    return base + offset;
}

bool
map_domain(uint32_t producer_domain, domain_mapping& out)
{
    if(producer_domain == producer_domain_runtime)
        out = {ROCPROFILER_BUFFER_TRACING_HIP_RUNTIME_API,
               ROCPROFILER_BUFFER_TRACING_HIP_RUNTIME_API_EXT,
               ROCPROFILER_HIP_TABLE_ID_Runtime};
    else if(producer_domain == producer_domain_compiler)
        out = {ROCPROFILER_BUFFER_TRACING_HIP_COMPILER_API,
               ROCPROFILER_BUFFER_TRACING_HIP_COMPILER_API_EXT,
               ROCPROFILER_HIP_TABLE_ID_Compiler};
    else
        return false;

    return true;
}

void
emit_record(const record_data& value, uint64_t end_timestamp, int32_t retval)
{
    auto mapping = domain_mapping{};
    if(!map_domain(value.domain, mapping)) return;

    auto contexts        = tracing::buffered_context_data_vec_t{};
    auto ext_contexts    = tracing::buffered_context_data_vec_t{};
    auto extern_corr_ids = tracing::external_correlation_id_map_t{};

    // External correlation ids are structurally unavailable out-of-process: the tool's
    // rocprofiler_push_external_correlation_id calls happen on the producer's threads, which
    // this process cannot see. populate_contexts leaves them as empty_user_data.
    tracing::populate_contexts(mapping.buffered, value.operation, contexts, extern_corr_ids);
    tracing::populate_contexts(
        mapping.buffered_ext, value.operation, ext_contexts, extern_corr_ids);

    if(contexts.empty() && ext_contexts.empty()) return;

    auto* corr_id = tracing::correlation_service::construct(1);
    if(!corr_id) return;  // finalization has begun

    // The correlation id is allocated here rather than carried in the event: the producer has no
    // use for one now that a call is a single self-contained event. ancestor reflects this
    // consumer thread's correlation stack rather than the producer's, so it is always 0 today.
    if(!contexts.empty())
    {
        auto record = common::init_public_api_struct(rocprofiler_buffer_tracing_hip_api_record_t{});

        record.start_timestamp = value.start_timestamp;
        record.end_timestamp   = end_timestamp;

        tracing::execute_buffer_record_emplace(contexts,
                                               value.thread_id,
                                               corr_id->internal,
                                               extern_corr_ids,
                                               corr_id->ancestor,
                                               mapping.buffered,
                                               value.operation,
                                               record);
    }

    if(!ext_contexts.empty())
    {
        auto record =
            common::init_public_api_struct(rocprofiler_buffer_tracing_hip_api_ext_record_t{});

        record.start_timestamp = value.start_timestamp;
        record.end_timestamp   = end_timestamp;

        // The exit event carries the return value but not the call arguments, so args stays
        // zero-filled. A tool cannot tell that apart from a call whose arguments were all zero,
        // which is why argument capture is documented as unavailable on Windows rather than
        // approximated here.
        record.retval.int_retval = retval;

        tracing::execute_buffer_record_emplace(ext_contexts,
                                               value.thread_id,
                                               corr_id->internal,
                                               extern_corr_ids,
                                               corr_id->ancestor,
                                               mapping.buffered_ext,
                                               value.operation,
                                               record);
    }

    corr_id->sub_ref_count();
    context::pop_latest_correlation_id(corr_id);

    ++get_stats().records_emitted;
}
}  // namespace

void
set_process_filter(uint32_t process_id)
{
    get_process_filter() = process_id;
}

void
handle_event(const EVENT_RECORD* event_record)
{
    if(!event_record) return;

    if(event_record->EventHeader.ProcessId != get_process_filter()) return;

    auto& schemas = get_schemas();
    auto  key     = get_schema_key(event_record);
    auto  itr     = schemas.find(key);

    if(itr == schemas.end()) itr = schemas.emplace(key, read_event_schema(event_record)).first;

    const auto& schema = itr->second;
    if(!schema.valid) return;

    auto domain    = uint32_t{0};
    auto name_end  = uint32_t{0};
    auto start_qpc = uint64_t{0};

    const char* op_name = nullptr;

    if(!read_scalar(event_record, schema.domain_offset, domain) ||
       (op_name = read_ansi_string(event_record, schema.op_name_offset, name_end)) == nullptr ||
       !read_scalar(event_record, name_end, start_qpc))
    {
        ++get_stats().events_dropped;
        return;
    }

    ++get_stats().events_decoded;

    auto mapping = domain_mapping{};
    if(!map_domain(domain, mapping))
    {
        ++get_stats().events_dropped;
        return;
    }

    // Feeding an unresolved id into context_filter would index the domain bitset out of range,
    // so drop and count instead.
    auto operation = id_by_name_impl(mapping.table, op_name);
    if(operation == invalid_operation_id)
    {
        ++get_stats().events_dropped;
        return;
    }

    // retval trails start_qpc, which trails the inline op_name.
    auto retval = int32_t{0};
    if(!read_scalar(event_record, name_end + static_cast<uint32_t>(sizeof(uint64_t)), retval))
    {
        ++get_stats().events_dropped;
        return;
    }

    // The producer samples QueryPerformanceCounter on entry and ETW stamps the header on
    // return, and the session sets Wnode.ClientContext = 1, so both are QPC ticks.
    auto value = record_data{common::qpc_ticks_to_ns(start_qpc),
                             static_cast<uint64_t>(event_record->EventHeader.ThreadId),
                             domain,
                             operation};

    emit_record(value,
                common::qpc_ticks_to_ns(
                    static_cast<uint64_t>(event_record->EventHeader.TimeStamp.QuadPart)),
                retval);
}

decoder_stats
reset_decoder()
{
    auto ret = get_stats();

    get_stats() = decoder_stats{};
    get_schemas().clear();

    return ret;
}
}  // namespace etw
}  // namespace hip
}  // namespace rocprofiler
