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
//

#pragma once

#include <rocprofiler-sdk/agent.h>
#include <rocprofiler-sdk/buffer.h>
#include <rocprofiler-sdk/buffer_tracing.h>
#include <rocprofiler-sdk/callback_tracing.h>
#include <rocprofiler-sdk/context.h>
#include <rocprofiler-sdk/counter_config.h>
#include <rocprofiler-sdk/counters.h>
#include <rocprofiler-sdk/device_counting_service.h>
#include <rocprofiler-sdk/dispatch_counting_service.h>
#include <rocprofiler-sdk/external_correlation.h>
#include <rocprofiler-sdk/fwd.h>
#include <rocprofiler-sdk/hip.h>
#include <rocprofiler-sdk/hsa.h>
#include <rocprofiler-sdk/intercept_table.h>
#include <rocprofiler-sdk/internal_threading.h>
#include <rocprofiler-sdk/marker.h>
#include <rocprofiler-sdk/pc_sampling.h>
#include <rocprofiler-sdk/cxx/name_info.hpp>
#include <rocprofiler-sdk/cxx/utility.hpp>

#include <cereal/archives/binary.hpp>
#include <cereal/archives/json.hpp>
#include <cereal/archives/portable_binary.hpp>
#include <cereal/cereal.hpp>
#include <cereal/types/array.hpp>
#include <cereal/types/atomic.hpp>
#include <cereal/types/bitset.hpp>
#include <cereal/types/chrono.hpp>
#include <cereal/types/common.hpp>
#include <cereal/types/complex.hpp>
#include <cereal/types/deque.hpp>
#include <cereal/types/functional.hpp>
#include <cereal/types/list.hpp>
#include <cereal/types/map.hpp>
#include <cereal/types/optional.hpp>
#include <cereal/types/queue.hpp>
#include <cereal/types/set.hpp>
#include <cereal/types/stack.hpp>
#include <cereal/types/string.hpp>
#include <cereal/types/unordered_map.hpp>
#include <cereal/types/unordered_set.hpp>
#include <cereal/types/utility.hpp>
#include <cereal/types/variant.hpp>
#include <cereal/types/vector.hpp>

#include <string>
#include <string_view>
#include <utility>
#include <vector>

#define ROCP_SDK_SAVE_DATA_FIELD(FIELD)       ar(make_nvp(#FIELD, data.FIELD))
#define ROCP_SDK_SAVE_DATA_VALUE(NAME, VALUE) ar(make_nvp(NAME, data.VALUE))
#define ROCP_SDK_SAVE_VALUE(NAME, VALUE)      ar(make_nvp(NAME, VALUE))
#define ROCP_SDK_SAVE_DATA_CSTR(FIELD)                                                             \
    ar(make_nvp(#FIELD, std::string{data.FIELD ? data.FIELD : ""}))
#define ROCP_SDK_SAVE_DATA_BITFIELD(NAME, VALUE)                                                   \
    {                                                                                              \
        auto _val = data.VALUE;                                                                    \
        ar(make_nvp(NAME, _val));                                                                  \
    }

#if !defined(ROCPROFILER_SDK_CEREAL_NAMESPACE_BEGIN)
#    define ROCPROFILER_SDK_CEREAL_NAMESPACE_BEGIN                                                 \
        namespace cereal                                                                           \
        {
#endif

#if !defined(ROCPROFILER_SDK_CEREAL_NAMESPACE_END)
#    define ROCPROFILER_SDK_CEREAL_NAMESPACE_END }  // namespace cereal
#endif

namespace rocprofiler
{
namespace sdk
{
namespace serialization
{
struct buffer_tracing_args
{
    std::string type  = {};
    std::string name  = {};
    std::string value = {};
};

template <typename Tp>
auto
get_buffer_tracing_args(Tp& data)
{
    auto populate_args_array = [](rocprofiler_buffer_tracing_kind_t /*kind*/,
                                  rocprofiler_tracing_operation_t /*operation*/,
                                  uint32_t arg_number,
                                  const void* const /*arg_value_addr*/,
                                  int32_t /*arg_indirection_count*/,
                                  const char* arg_type,
                                  const char* arg_name,
                                  const char* arg_value_str,
                                  void*       cb_data) -> int {
        if(!cb_data) return 1;

        auto* vec = static_cast<std::vector<buffer_tracing_args>*>(cb_data);
        auto  sz  = std::max<size_t>(arg_number + 1, vec->size());
        vec->resize(sz, buffer_tracing_args{});
        vec->at(arg_number) = buffer_tracing_args{arg_type, arg_name, arg_value_str};

        return 0;
    };

    auto ret    = std::vector<buffer_tracing_args>{};
    auto record = rocprofiler_record_header_t{};
    record.hash =
        rocprofiler_record_header_compute_hash(ROCPROFILER_BUFFER_CATEGORY_TRACING, data.kind);
    record.payload = &data;

    rocprofiler_iterate_buffer_tracing_record_args(record, populate_args_array, &ret);

    return ret;
}
}  // namespace serialization
}  // namespace sdk
}  // namespace rocprofiler

ROCPROFILER_SDK_CEREAL_NAMESPACE_BEGIN

namespace sdk = ::rocprofiler::sdk;

template <typename ArchiveT>
void
save(ArchiveT& ar, const sdk::serialization::buffer_tracing_args& data)
{
    ROCP_SDK_SAVE_DATA_FIELD(type);
    ROCP_SDK_SAVE_DATA_FIELD(name);
    ROCP_SDK_SAVE_DATA_FIELD(value);
}

template <typename ArchiveT>
void
save(ArchiveT& ar, rocprofiler_context_id_t data)
{
    ROCP_SDK_SAVE_DATA_FIELD(handle);
}

template <typename ArchiveT>
void
save(ArchiveT& ar, rocprofiler_agent_id_t data)
{
    ROCP_SDK_SAVE_DATA_FIELD(handle);
}

template <typename ArchiveT>
void
save(ArchiveT& ar, hsa_agent_t data)
{
    ROCP_SDK_SAVE_DATA_FIELD(handle);
}

template <typename ArchiveT>
void
save(ArchiveT& ar, rocprofiler_queue_id_t data)
{
    ROCP_SDK_SAVE_DATA_FIELD(handle);
}

template <typename ArchiveT>
void
save(ArchiveT& ar, rocprofiler_stream_id_t data)
{
    ROCP_SDK_SAVE_DATA_FIELD(handle);
}

template <typename ArchiveT>
void
save(ArchiveT& ar, rocprofiler_counter_id_t data)
{
    ROCP_SDK_SAVE_DATA_FIELD(handle);
}

template <typename ArchiveT>
void
save(ArchiveT& ar, rocprofiler_correlation_id_t data)
{
    ROCP_SDK_SAVE_DATA_FIELD(internal);
    ROCP_SDK_SAVE_DATA_VALUE("external", external.value);
    ROCP_SDK_SAVE_DATA_VALUE("ancestor", ancestor);
}

template <typename ArchiveT>
void
save(ArchiveT& ar, rocprofiler_async_correlation_id_t data)
{
    ROCP_SDK_SAVE_DATA_FIELD(internal);
    ROCP_SDK_SAVE_DATA_VALUE("external", external.value);
}

template <typename ArchiveT>
void
save(ArchiveT& ar, rocprofiler_dim3_t data)
{
    ROCP_SDK_SAVE_DATA_FIELD(x);
    ROCP_SDK_SAVE_DATA_FIELD(y);
    ROCP_SDK_SAVE_DATA_FIELD(z);
}

template <typename ArchiveT>
void
save(ArchiveT& ar, rocprofiler_address_t data)
{
    ROCP_SDK_SAVE_DATA_FIELD(handle);
}

template <typename ArchiveT>
void
save(ArchiveT& ar, rocprofiler_callback_tracing_code_object_load_data_t data)
{
    ROCP_SDK_SAVE_DATA_FIELD(size);
    ROCP_SDK_SAVE_DATA_FIELD(code_object_id);
    ROCP_SDK_SAVE_DATA_FIELD(agent_id);
    ROCP_SDK_SAVE_DATA_CSTR(uri);
    ROCP_SDK_SAVE_DATA_FIELD(load_base);
    ROCP_SDK_SAVE_DATA_FIELD(load_size);
    ROCP_SDK_SAVE_DATA_FIELD(load_delta);
    ROCP_SDK_SAVE_DATA_FIELD(storage_type);
    if(data.storage_type == ROCPROFILER_CODE_OBJECT_STORAGE_TYPE_FILE)
    {
        ROCP_SDK_SAVE_DATA_FIELD(storage_file);
    }
    else if(data.storage_type == ROCPROFILER_CODE_OBJECT_STORAGE_TYPE_MEMORY)
    {
        ROCP_SDK_SAVE_DATA_FIELD(memory_base);
        ROCP_SDK_SAVE_DATA_FIELD(memory_size);
    }
}

template <typename ArchiveT>
void
save(ArchiveT& ar, rocprofiler_callback_tracing_code_object_kernel_symbol_register_data_t data)
{
    ROCP_SDK_SAVE_DATA_FIELD(size);
    ROCP_SDK_SAVE_DATA_FIELD(kernel_id);
    ROCP_SDK_SAVE_DATA_FIELD(code_object_id);
    ROCP_SDK_SAVE_DATA_CSTR(kernel_name);
    ROCP_SDK_SAVE_DATA_FIELD(kernel_object);
    ROCP_SDK_SAVE_DATA_FIELD(kernarg_segment_size);
    ROCP_SDK_SAVE_DATA_FIELD(kernarg_segment_alignment);
    ROCP_SDK_SAVE_DATA_FIELD(group_segment_size);
    ROCP_SDK_SAVE_DATA_FIELD(private_segment_size);
    ROCP_SDK_SAVE_DATA_FIELD(sgpr_count);
    ROCP_SDK_SAVE_DATA_FIELD(arch_vgpr_count);
    ROCP_SDK_SAVE_DATA_FIELD(accum_vgpr_count);
    ROCP_SDK_SAVE_DATA_FIELD(kernel_code_entry_byte_offset);
    ROCP_SDK_SAVE_DATA_FIELD(kernel_address);
}

template <typename ArchiveT>
void
save(ArchiveT& ar, rocprofiler_callback_tracing_code_object_host_kernel_symbol_register_data_t data)
{
    ROCP_SDK_SAVE_DATA_FIELD(size);
    ROCP_SDK_SAVE_DATA_FIELD(host_function_id);
    ROCP_SDK_SAVE_DATA_FIELD(kernel_id);
    ROCP_SDK_SAVE_DATA_FIELD(code_object_id);
    ROCP_SDK_SAVE_DATA_FIELD(host_function);
    ROCP_SDK_SAVE_DATA_FIELD(modules);
    ROCP_SDK_SAVE_DATA_CSTR(device_function);
    ROCP_SDK_SAVE_DATA_FIELD(thread_limit);
    ROCP_SDK_SAVE_DATA_FIELD(thread_ids);
    ROCP_SDK_SAVE_DATA_FIELD(block_ids);
    ROCP_SDK_SAVE_DATA_FIELD(block_dims);
    ROCP_SDK_SAVE_DATA_FIELD(grid_dims);
    ROCP_SDK_SAVE_DATA_FIELD(workgroup_size);
}

template <typename ArchiveT>
void
save(ArchiveT& ar, rocprofiler_hsa_api_retval_t data)
{
    ROCP_SDK_SAVE_DATA_FIELD(uint64_t_retval);
}

template <typename ArchiveT>
void
save(ArchiveT& ar, const hsa_queue_t& data)
{
    ar(make_nvp("queue_id", data.id));
}

template <typename ArchiveT>
void
save(ArchiveT& ar, hsa_amd_event_scratch_alloc_start_t data)
{
    ar(make_nvp("queue_id", *data.queue));
    ROCP_SDK_SAVE_DATA_FIELD(dispatch_id);
}

template <typename ArchiveT>
void
save(ArchiveT& ar, hsa_amd_event_scratch_alloc_end_t data)
{
    ar(make_nvp("queue_id", *data.queue));
    ROCP_SDK_SAVE_DATA_FIELD(dispatch_id);
    ROCP_SDK_SAVE_DATA_FIELD(size);
    ROCP_SDK_SAVE_DATA_FIELD(num_slots);
    ROCP_SDK_SAVE_DATA_FIELD(flags);
}

template <typename ArchiveT>
void
save(ArchiveT& ar, hsa_amd_event_scratch_free_start_t data)
{
    ar(make_nvp("queue_id", *data.queue));
}

template <typename ArchiveT>
void
save(ArchiveT& ar, hsa_amd_event_scratch_free_end_t data)
{
    ar(make_nvp("queue_id", *data.queue));
    ROCP_SDK_SAVE_DATA_FIELD(flags);
}

template <typename ArchiveT>
void
save(ArchiveT& ar, hsa_amd_event_scratch_async_reclaim_start_t data)
{
    ar(make_nvp("queue_id", *data.queue));
}

template <typename ArchiveT>
void
save(ArchiveT& ar, hsa_amd_event_scratch_async_reclaim_end_t data)
{
    ar(make_nvp("queue_id", *data.queue));
    ROCP_SDK_SAVE_DATA_FIELD(flags);
}

template <typename ArchiveT>
void
save(ArchiveT& ar, rocprofiler_marker_api_retval_t data)
{
    ROCP_SDK_SAVE_DATA_FIELD(int64_t_retval);
}

template <typename ArchiveT>
void
save(ArchiveT& ar, rocprofiler_callback_tracing_hsa_api_data_t data)
{
    ROCP_SDK_SAVE_DATA_FIELD(size);
    // ROCP_SDK_SAVE_DATA_FIELD(args);
    ROCP_SDK_SAVE_DATA_FIELD(retval);
}

template <typename ArchiveT>
void
save(ArchiveT& ar, rocprofiler_callback_tracing_marker_api_data_t data)
{
    ROCP_SDK_SAVE_DATA_FIELD(size);
    // ROCP_SDK_SAVE_DATA_FIELD(args);
    ROCP_SDK_SAVE_DATA_FIELD(retval);
}

template <typename ArchiveT>
void
save(ArchiveT& ar, rocprofiler_hip_api_retval_t data)
{
    ROCP_SDK_SAVE_DATA_FIELD(uint64_t_retval);
}

template <typename ArchiveT>
void
save(ArchiveT& ar, rocprofiler_callback_tracing_hip_api_data_t data)
{
    ROCP_SDK_SAVE_DATA_FIELD(size);
    // ROCP_SDK_SAVE_DATA_FIELD(args);
    ROCP_SDK_SAVE_DATA_FIELD(retval);
}

template <typename ArchiveT>
void
save(ArchiveT& ar, rocprofiler_callback_tracing_scratch_memory_data_t data)
{
    ROCP_SDK_SAVE_DATA_FIELD(size);
    ROCP_SDK_SAVE_DATA_FIELD(agent_id);
    ROCP_SDK_SAVE_DATA_FIELD(queue_id);
    ROCP_SDK_SAVE_DATA_FIELD(flags);
    ROCP_SDK_SAVE_DATA_FIELD(args_kind);
}

template <typename ArchiveT>
void
save(ArchiveT& ar, rocprofiler_kernel_dispatch_info_t data)
{
    ROCP_SDK_SAVE_DATA_FIELD(size);
    ROCP_SDK_SAVE_DATA_FIELD(agent_id);
    ROCP_SDK_SAVE_DATA_FIELD(queue_id);
    ROCP_SDK_SAVE_DATA_FIELD(kernel_id);
    ROCP_SDK_SAVE_DATA_FIELD(dispatch_id);
    ROCP_SDK_SAVE_DATA_FIELD(private_segment_size);
    ROCP_SDK_SAVE_DATA_FIELD(group_segment_size);
    ROCP_SDK_SAVE_DATA_FIELD(workgroup_size);
    ROCP_SDK_SAVE_DATA_FIELD(grid_size);
}

template <typename ArchiveT>
void
save(ArchiveT& ar, rocprofiler_callback_tracing_kernel_dispatch_data_t data)
{
    ROCP_SDK_SAVE_DATA_FIELD(size);
    ROCP_SDK_SAVE_DATA_FIELD(start_timestamp);
    ROCP_SDK_SAVE_DATA_FIELD(end_timestamp);
    ROCP_SDK_SAVE_DATA_FIELD(dispatch_info);
}

template <typename ArchiveT>
void
save(ArchiveT& ar, rocprofiler_callback_tracing_memory_copy_data_t data)
{
    ROCP_SDK_SAVE_DATA_FIELD(size);
    ROCP_SDK_SAVE_DATA_FIELD(start_timestamp);
    ROCP_SDK_SAVE_DATA_FIELD(end_timestamp);
    ROCP_SDK_SAVE_DATA_FIELD(dst_agent_id);
    ROCP_SDK_SAVE_DATA_FIELD(src_agent_id);
    ROCP_SDK_SAVE_DATA_FIELD(bytes);
}

template <typename ArchiveT>
void
save(ArchiveT& ar, rocprofiler_callback_tracing_memory_allocation_data_t data)
{
    ROCP_SDK_SAVE_DATA_FIELD(size);
    ROCP_SDK_SAVE_DATA_FIELD(start_timestamp);
    ROCP_SDK_SAVE_DATA_FIELD(end_timestamp);
    ROCP_SDK_SAVE_DATA_FIELD(agent_id);
    ROCP_SDK_SAVE_VALUE("address", rocprofiler::sdk::utility::as_hex(data.address.handle, 16));
    ROCP_SDK_SAVE_DATA_FIELD(allocation_size);
}

template <typename ArchiveT>
void
save(ArchiveT& ar, rocprofiler_rccl_api_retval_t data)
{
    ROCP_SDK_SAVE_DATA_FIELD(ncclResult_t_retval);
}

template <typename ArchiveT>
void
save(ArchiveT& ar, rocprofiler_callback_tracing_rccl_api_data_t data)
{
    ROCP_SDK_SAVE_DATA_FIELD(size);
    // ROCP_SDK_SAVE_DATA_FIELD(args);
    ROCP_SDK_SAVE_DATA_FIELD(retval);
}

template <typename ArchiveT>
void
save(ArchiveT& ar, rocprofiler_rocdecode_api_retval_t data)
{
    ROCP_SDK_SAVE_DATA_FIELD(uint64_t_retval);
}

template <typename ArchiveT>
void
save(ArchiveT& ar, rocprofiler_callback_tracing_rocdecode_api_data_t data)
{
    ROCP_SDK_SAVE_DATA_FIELD(size);
    ROCP_SDK_SAVE_DATA_FIELD(retval);
}

template <typename ArchiveT>
void
save(ArchiveT& ar, rocprofiler_rocjpeg_api_retval_t data)
{
    ROCP_SDK_SAVE_DATA_FIELD(rocJpegStatus_retval);
}

template <typename ArchiveT>
void
save(ArchiveT& ar, rocprofiler_callback_tracing_rocjpeg_api_data_t data)
{
    ROCP_SDK_SAVE_DATA_FIELD(size);
    ROCP_SDK_SAVE_DATA_FIELD(retval);
}

template <typename ArchiveT>
void
save(ArchiveT& ar, rocprofiler_callback_tracing_ompt_data_t data)
{
    ROCP_SDK_SAVE_DATA_FIELD(size);
    // ROCP_SDK_SAVE_DATA_FIELD(args);
}

template <typename ArchiveT>
void
save(ArchiveT& ar, rocprofiler_dispatch_counting_service_data_t data)
{
    ROCP_SDK_SAVE_DATA_FIELD(size);
    ROCP_SDK_SAVE_DATA_FIELD(correlation_id);
    ROCP_SDK_SAVE_DATA_FIELD(start_timestamp);
    ROCP_SDK_SAVE_DATA_FIELD(end_timestamp);
    ROCP_SDK_SAVE_DATA_FIELD(dispatch_info);
}

template <typename ArchiveT>
void
save(ArchiveT& ar, rocprofiler_dispatch_counting_service_record_t data)
{
    ROCP_SDK_SAVE_DATA_FIELD(size);
    ROCP_SDK_SAVE_DATA_FIELD(num_records);
    ROCP_SDK_SAVE_DATA_FIELD(correlation_id);
    ROCP_SDK_SAVE_DATA_FIELD(start_timestamp);
    ROCP_SDK_SAVE_DATA_FIELD(end_timestamp);
    ROCP_SDK_SAVE_DATA_FIELD(dispatch_info);
}

template <typename ArchiveT>
void
save(ArchiveT& ar, rocprofiler_callback_tracing_record_t data)
{
    ROCP_SDK_SAVE_DATA_FIELD(context_id);
    ROCP_SDK_SAVE_DATA_FIELD(thread_id);
    ROCP_SDK_SAVE_DATA_FIELD(kind);
    ROCP_SDK_SAVE_DATA_FIELD(operation);
    ROCP_SDK_SAVE_DATA_FIELD(correlation_id);
    ROCP_SDK_SAVE_DATA_FIELD(phase);
}

template <typename ArchiveT, typename Tp>
void
save_buffer_tracing_api_record(ArchiveT& ar, Tp data)
{
    ROCP_SDK_SAVE_DATA_FIELD(size);
    ROCP_SDK_SAVE_DATA_FIELD(kind);
    ROCP_SDK_SAVE_DATA_FIELD(operation);
    ROCP_SDK_SAVE_DATA_FIELD(correlation_id);
    ROCP_SDK_SAVE_DATA_FIELD(start_timestamp);
    ROCP_SDK_SAVE_DATA_FIELD(end_timestamp);
    ROCP_SDK_SAVE_DATA_FIELD(thread_id);
}

template <typename ArchiveT>
void
save(ArchiveT& ar, rocprofiler_buffer_tracing_hsa_api_record_t data)
{
    save_buffer_tracing_api_record(ar, data);
}

template <typename ArchiveT>
void
save(ArchiveT& ar, rocprofiler_counter_record_t data)
{
    ROCP_SDK_SAVE_DATA_FIELD(id);
    ROCP_SDK_SAVE_DATA_FIELD(counter_value);
    ROCP_SDK_SAVE_DATA_FIELD(dispatch_id);
}

template <typename ArchiveT>
void
save(ArchiveT& ar, rocprofiler_buffer_tracing_hip_api_record_t data)
{
    save_buffer_tracing_api_record(ar, data);
}

template <typename ArchiveT>
void
save(ArchiveT& ar, rocprofiler_buffer_tracing_hip_api_ext_record_t data)
{
    save_buffer_tracing_api_record(ar, data);
    auto args = sdk::serialization::get_buffer_tracing_args(data);
    ROCP_SDK_SAVE_VALUE("args", args);
    ROCP_SDK_SAVE_DATA_FIELD(retval);
}

template <typename ArchiveT>
void
save(ArchiveT& ar, rocprofiler_buffer_tracing_marker_api_record_t data)
{
    save_buffer_tracing_api_record(ar, data);
}

template <typename ArchiveT>
void
save(ArchiveT& ar, rocprofiler_buffer_tracing_rccl_api_record_t data)
{
    save_buffer_tracing_api_record(ar, data);
}

template <typename ArchiveT>
void
save(ArchiveT& ar, rocprofiler_buffer_tracing_rocdecode_api_record_t data)
{
    save_buffer_tracing_api_record(ar, data);
}

template <typename ArchiveT>
void
save(ArchiveT& ar, rocprofiler_buffer_tracing_rocdecode_api_ext_record_t data)
{
    save_buffer_tracing_api_record(ar, data);
    auto args = sdk::serialization::get_buffer_tracing_args(data);
    ROCP_SDK_SAVE_VALUE("args", args);
    ROCP_SDK_SAVE_DATA_FIELD(retval);
}

template <typename ArchiveT>
void
save(ArchiveT& ar, rocprofiler_buffer_tracing_rocjpeg_api_record_t data)
{
    save_buffer_tracing_api_record(ar, data);
}

template <typename ArchiveT>
void
save(ArchiveT& ar, rocprofiler_buffer_tracing_ompt_target_t data)
{
    ROCP_SDK_SAVE_DATA_VALUE("kind", kind);
    ROCP_SDK_SAVE_DATA_VALUE("device", device_num);
    ROCP_SDK_SAVE_DATA_VALUE("task_id", task_id);
    ROCP_SDK_SAVE_DATA_VALUE("target_id", target_id);
}

template <typename ArchiveT>
void
save(ArchiveT& ar, rocprofiler_buffer_tracing_ompt_target_data_op_t data)
{
    ROCP_SDK_SAVE_DATA_VALUE("host_op_id", host_op_id);
    ROCP_SDK_SAVE_DATA_VALUE("optype", optype);
    ROCP_SDK_SAVE_DATA_VALUE("src_device_num", src_device_num);
    ROCP_SDK_SAVE_DATA_VALUE("dst_device_num", dst_device_num);
    ROCP_SDK_SAVE_DATA_VALUE("bytes", bytes);
}

template <typename ArchiveT>
void
save(ArchiveT& ar, rocprofiler_buffer_tracing_ompt_target_kernel_t data)
{
    ROCP_SDK_SAVE_DATA_VALUE("host_op_id", host_op_id);
    ROCP_SDK_SAVE_DATA_VALUE("device_num", device_num);
    ROCP_SDK_SAVE_DATA_VALUE("requested_num_teams", requested_num_teams);
}

template <typename ArchiveT>
void
save(ArchiveT& ar, rocprofiler_buffer_tracing_ompt_record_t data)
{
    save_buffer_tracing_api_record(ar, data);

    if(data.operation == ROCPROFILER_OMPT_ID_target_emi)
    {
        ROCP_SDK_SAVE_DATA_FIELD(target);
    }
    else if(data.operation == ROCPROFILER_OMPT_ID_target_data_op_emi)
    {
        ROCP_SDK_SAVE_DATA_FIELD(target_data_op);
    }
    else if(data.operation == ROCPROFILER_OMPT_ID_target_submit_emi)
    {
        ROCP_SDK_SAVE_DATA_FIELD(target_kernel);
    }
}

template <typename ArchiveT>
void
save(ArchiveT& ar, rocprofiler_buffer_tracing_kernel_dispatch_record_t data)
{
    ROCP_SDK_SAVE_DATA_FIELD(size);
    ROCP_SDK_SAVE_DATA_FIELD(kind);
    ROCP_SDK_SAVE_DATA_FIELD(operation);
    ROCP_SDK_SAVE_DATA_FIELD(thread_id);
    ROCP_SDK_SAVE_DATA_FIELD(correlation_id);
    ROCP_SDK_SAVE_DATA_FIELD(start_timestamp);
    ROCP_SDK_SAVE_DATA_FIELD(end_timestamp);
    ROCP_SDK_SAVE_DATA_FIELD(dispatch_info);
}

template <typename ArchiveT>
void
save(ArchiveT& ar, rocprofiler_buffer_tracing_memory_copy_record_t data)
{
    ROCP_SDK_SAVE_DATA_FIELD(size);
    ROCP_SDK_SAVE_DATA_FIELD(kind);
    ROCP_SDK_SAVE_DATA_FIELD(operation);
    ROCP_SDK_SAVE_DATA_FIELD(thread_id);
    ROCP_SDK_SAVE_DATA_FIELD(correlation_id);
    ROCP_SDK_SAVE_DATA_FIELD(start_timestamp);
    ROCP_SDK_SAVE_DATA_FIELD(end_timestamp);
    ROCP_SDK_SAVE_DATA_FIELD(dst_agent_id);
    ROCP_SDK_SAVE_DATA_FIELD(src_agent_id);
    ROCP_SDK_SAVE_DATA_FIELD(bytes);
}

template <typename ArchiveT>
void
save(ArchiveT& ar, rocprofiler_buffer_tracing_memory_allocation_record_t data)
{
    ROCP_SDK_SAVE_DATA_FIELD(size);
    ROCP_SDK_SAVE_DATA_FIELD(kind);
    ROCP_SDK_SAVE_DATA_FIELD(operation);
    ROCP_SDK_SAVE_DATA_FIELD(thread_id);
    ROCP_SDK_SAVE_DATA_FIELD(correlation_id);
    ROCP_SDK_SAVE_DATA_FIELD(start_timestamp);
    ROCP_SDK_SAVE_DATA_FIELD(end_timestamp);
    ROCP_SDK_SAVE_DATA_FIELD(agent_id);
    ROCP_SDK_SAVE_VALUE("address", rocprofiler::sdk::utility::as_hex(data.address.handle, 16));
    ROCP_SDK_SAVE_DATA_FIELD(allocation_size);
}

template <typename ArchiveT>
void
save(ArchiveT& ar, rocprofiler_buffer_tracing_kfd_event_page_migrate_record_t data)
{
    ROCP_SDK_SAVE_DATA_FIELD(size);
    ROCP_SDK_SAVE_DATA_FIELD(operation);
    ROCP_SDK_SAVE_DATA_FIELD(timestamp);
    ROCP_SDK_SAVE_DATA_FIELD(pid);
    ROCP_SDK_SAVE_DATA_FIELD(start_address);
    ROCP_SDK_SAVE_DATA_FIELD(end_address);
    ROCP_SDK_SAVE_DATA_FIELD(src_agent);
    ROCP_SDK_SAVE_DATA_FIELD(dst_agent);
    ROCP_SDK_SAVE_DATA_FIELD(prefetch_agent);
    ROCP_SDK_SAVE_DATA_FIELD(preferred_agent);
    ROCP_SDK_SAVE_DATA_FIELD(error_code);
}

template <typename ArchiveT>
void
save(ArchiveT& ar, rocprofiler_buffer_tracing_kfd_event_page_fault_record_t data)
{
    ROCP_SDK_SAVE_DATA_FIELD(size);
    ROCP_SDK_SAVE_DATA_FIELD(operation);
    ROCP_SDK_SAVE_DATA_FIELD(timestamp);
    ROCP_SDK_SAVE_DATA_FIELD(pid);
    ROCP_SDK_SAVE_DATA_FIELD(agent_id);
    ROCP_SDK_SAVE_DATA_FIELD(address);
}

template <typename ArchiveT>
void
save(ArchiveT& ar, rocprofiler_buffer_tracing_kfd_event_queue_record_t data)
{
    ROCP_SDK_SAVE_DATA_FIELD(size);
    ROCP_SDK_SAVE_DATA_FIELD(operation);
    ROCP_SDK_SAVE_DATA_FIELD(timestamp);
    ROCP_SDK_SAVE_DATA_FIELD(pid);
    ROCP_SDK_SAVE_DATA_FIELD(agent_id);
}

template <typename ArchiveT>
void
save(ArchiveT& ar, rocprofiler_buffer_tracing_kfd_event_unmap_from_gpu_record_t data)
{
    ROCP_SDK_SAVE_DATA_FIELD(size);
    ROCP_SDK_SAVE_DATA_FIELD(operation);
    ROCP_SDK_SAVE_DATA_FIELD(timestamp);
    ROCP_SDK_SAVE_DATA_FIELD(pid);
    ROCP_SDK_SAVE_DATA_FIELD(agent_id);
    ROCP_SDK_SAVE_DATA_FIELD(start_address);
    ROCP_SDK_SAVE_DATA_FIELD(end_address);
}

template <typename ArchiveT>
void
save(ArchiveT& ar, rocprofiler_buffer_tracing_kfd_event_dropped_events_record_t data)
{
    ROCP_SDK_SAVE_DATA_FIELD(size);
    ROCP_SDK_SAVE_DATA_FIELD(operation);
    ROCP_SDK_SAVE_DATA_FIELD(timestamp);
    ROCP_SDK_SAVE_DATA_FIELD(pid);
    ROCP_SDK_SAVE_DATA_FIELD(count);
}

template <typename ArchiveT>
void
save(ArchiveT& ar, rocprofiler_buffer_tracing_kfd_page_migrate_record_t data)
{
    ROCP_SDK_SAVE_DATA_FIELD(size);
    ROCP_SDK_SAVE_DATA_FIELD(operation);
    ROCP_SDK_SAVE_DATA_FIELD(start_timestamp);
    ROCP_SDK_SAVE_DATA_FIELD(end_timestamp);
    ROCP_SDK_SAVE_DATA_FIELD(pid);
    ROCP_SDK_SAVE_DATA_FIELD(start_address);
    ROCP_SDK_SAVE_DATA_FIELD(end_address);
    ROCP_SDK_SAVE_DATA_FIELD(src_agent);
    ROCP_SDK_SAVE_DATA_FIELD(dst_agent);
    ROCP_SDK_SAVE_DATA_FIELD(prefetch_agent);
    ROCP_SDK_SAVE_DATA_FIELD(preferred_agent);
    ROCP_SDK_SAVE_DATA_FIELD(error_code);
}

template <typename ArchiveT>
void
save(ArchiveT& ar, rocprofiler_buffer_tracing_kfd_page_fault_record_t data)
{
    ROCP_SDK_SAVE_DATA_FIELD(size);
    ROCP_SDK_SAVE_DATA_FIELD(operation);
    ROCP_SDK_SAVE_DATA_FIELD(start_timestamp);
    ROCP_SDK_SAVE_DATA_FIELD(end_timestamp);
    ROCP_SDK_SAVE_DATA_FIELD(pid);
    ROCP_SDK_SAVE_DATA_FIELD(agent_id);
    ROCP_SDK_SAVE_DATA_FIELD(address);
}

template <typename ArchiveT>
void
save(ArchiveT& ar, rocprofiler_buffer_tracing_kfd_queue_record_t data)
{
    ROCP_SDK_SAVE_DATA_FIELD(size);
    ROCP_SDK_SAVE_DATA_FIELD(operation);
    ROCP_SDK_SAVE_DATA_FIELD(start_timestamp);
    ROCP_SDK_SAVE_DATA_FIELD(end_timestamp);
    ROCP_SDK_SAVE_DATA_FIELD(pid);
    ROCP_SDK_SAVE_DATA_FIELD(agent_id);
}

template <typename ArchiveT>
void
save(ArchiveT& ar, rocprofiler_buffer_tracing_scratch_memory_record_t data)
{
    ROCP_SDK_SAVE_DATA_FIELD(size);
    ROCP_SDK_SAVE_DATA_FIELD(kind);
    ROCP_SDK_SAVE_DATA_FIELD(operation);
    ROCP_SDK_SAVE_DATA_FIELD(agent_id);
    ROCP_SDK_SAVE_DATA_FIELD(queue_id);
    ROCP_SDK_SAVE_DATA_FIELD(thread_id);
    ROCP_SDK_SAVE_DATA_FIELD(start_timestamp);
    ROCP_SDK_SAVE_DATA_FIELD(end_timestamp);
    ROCP_SDK_SAVE_DATA_FIELD(correlation_id);
    ROCP_SDK_SAVE_DATA_FIELD(flags);
}

template <typename ArchiveT>
void
save(ArchiveT& ar, rocprofiler_buffer_tracing_correlation_id_retirement_record_t data)
{
    ROCP_SDK_SAVE_DATA_FIELD(size);
    ROCP_SDK_SAVE_DATA_FIELD(kind);
    ROCP_SDK_SAVE_DATA_FIELD(timestamp);
    ROCP_SDK_SAVE_DATA_FIELD(internal_correlation_id);
}

template <typename ArchiveT>
void
save(ArchiveT& ar, HsaCacheType data)
{
    ROCP_SDK_SAVE_DATA_BITFIELD("Data", ui32.Data);
    ROCP_SDK_SAVE_DATA_BITFIELD("Instruction", ui32.Instruction);
    ROCP_SDK_SAVE_DATA_BITFIELD("CPU", ui32.CPU);
    ROCP_SDK_SAVE_DATA_BITFIELD("HSACU", ui32.HSACU);
}

template <typename ArchiveT>
void
save(ArchiveT& ar, HSA_LINKPROPERTY data)
{
    ROCP_SDK_SAVE_DATA_BITFIELD("Override", ui32.Override);
    ROCP_SDK_SAVE_DATA_BITFIELD("NonCoherent", ui32.NonCoherent);
    ROCP_SDK_SAVE_DATA_BITFIELD("NoAtomics32bit", ui32.NoAtomics32bit);
    ROCP_SDK_SAVE_DATA_BITFIELD("NoAtomics64bit", ui32.NoAtomics64bit);
    ROCP_SDK_SAVE_DATA_BITFIELD("NoPeerToPeerDMA", ui32.NoPeerToPeerDMA);
}

template <typename ArchiveT>
void
save(ArchiveT& ar, HSA_CAPABILITY data)
{
    ROCP_SDK_SAVE_DATA_BITFIELD("HotPluggable", ui32.HotPluggable);
    ROCP_SDK_SAVE_DATA_BITFIELD("HSAMMUPresent", ui32.HSAMMUPresent);
    ROCP_SDK_SAVE_DATA_BITFIELD("SharedWithGraphics", ui32.SharedWithGraphics);
    ROCP_SDK_SAVE_DATA_BITFIELD("QueueSizePowerOfTwo", ui32.QueueSizePowerOfTwo);
    ROCP_SDK_SAVE_DATA_BITFIELD("QueueSize32bit", ui32.QueueSize32bit);
    ROCP_SDK_SAVE_DATA_BITFIELD("QueueIdleEvent", ui32.QueueIdleEvent);
    ROCP_SDK_SAVE_DATA_BITFIELD("VALimit", ui32.VALimit);
    ROCP_SDK_SAVE_DATA_BITFIELD("WatchPointsSupported", ui32.WatchPointsSupported);
    ROCP_SDK_SAVE_DATA_BITFIELD("WatchPointsTotalBits", ui32.WatchPointsTotalBits);
    ROCP_SDK_SAVE_DATA_BITFIELD("DoorbellType", ui32.DoorbellType);
    ROCP_SDK_SAVE_DATA_BITFIELD("AQLQueueDoubleMap", ui32.AQLQueueDoubleMap);
    ROCP_SDK_SAVE_DATA_BITFIELD("DebugTrapSupported", ui32.DebugTrapSupported);
    ROCP_SDK_SAVE_DATA_BITFIELD("WaveLaunchTrapOverrideSupported",
                                ui32.WaveLaunchTrapOverrideSupported);
    ROCP_SDK_SAVE_DATA_BITFIELD("WaveLaunchModeSupported", ui32.WaveLaunchModeSupported);
    ROCP_SDK_SAVE_DATA_BITFIELD("PreciseMemoryOperationsSupported",
                                ui32.PreciseMemoryOperationsSupported);
    ROCP_SDK_SAVE_DATA_BITFIELD("DEPRECATED_SRAM_EDCSupport", ui32.DEPRECATED_SRAM_EDCSupport);
    ROCP_SDK_SAVE_DATA_BITFIELD("Mem_EDCSupport", ui32.Mem_EDCSupport);
    ROCP_SDK_SAVE_DATA_BITFIELD("RASEventNotify", ui32.RASEventNotify);
    ROCP_SDK_SAVE_DATA_BITFIELD("ASICRevision", ui32.ASICRevision);
    ROCP_SDK_SAVE_DATA_BITFIELD("SRAM_EDCSupport", ui32.SRAM_EDCSupport);
    ROCP_SDK_SAVE_DATA_BITFIELD("SVMAPISupported", ui32.SVMAPISupported);
    ROCP_SDK_SAVE_DATA_BITFIELD("CoherentHostAccess", ui32.CoherentHostAccess);
    ROCP_SDK_SAVE_DATA_BITFIELD("DebugSupportedFirmware", ui32.DebugSupportedFirmware);
}

template <typename ArchiveT>
void
save(ArchiveT& ar, HSA_MEMORYPROPERTY data)
{
    ROCP_SDK_SAVE_DATA_BITFIELD("HotPluggable", ui32.HotPluggable);
    ROCP_SDK_SAVE_DATA_BITFIELD("NonVolatile", ui32.NonVolatile);
}

template <typename ArchiveT>
void
save(ArchiveT& ar, HSA_ENGINE_VERSION data)
{
    ROCP_SDK_SAVE_DATA_BITFIELD("uCodeSDMA", uCodeSDMA);
    ROCP_SDK_SAVE_DATA_BITFIELD("uCodeRes", uCodeRes);
}

template <typename ArchiveT>
void
save(ArchiveT& ar, HSA_ENGINE_ID data)
{
    ROCP_SDK_SAVE_DATA_BITFIELD("uCode", ui32.uCode);
    ROCP_SDK_SAVE_DATA_BITFIELD("Major", ui32.Major);
    ROCP_SDK_SAVE_DATA_BITFIELD("Minor", ui32.Minor);
    ROCP_SDK_SAVE_DATA_BITFIELD("Stepping", ui32.Stepping);
}

template <typename ArchiveT>
void
save(ArchiveT& ar, rocprofiler_agent_cache_t data)
{
    ROCP_SDK_SAVE_DATA_FIELD(processor_id_low);
    ROCP_SDK_SAVE_DATA_FIELD(size);
    ROCP_SDK_SAVE_DATA_FIELD(level);
    ROCP_SDK_SAVE_DATA_FIELD(cache_line_size);
    ROCP_SDK_SAVE_DATA_FIELD(cache_lines_per_tag);
    ROCP_SDK_SAVE_DATA_FIELD(association);
    ROCP_SDK_SAVE_DATA_FIELD(latency);
    ROCP_SDK_SAVE_DATA_FIELD(type);
}
template <typename ArchiveT>
void
save(ArchiveT& ar, rocprofiler_pc_t data)
{
    ROCP_SDK_SAVE_DATA_FIELD(code_object_id);
    ROCP_SDK_SAVE_DATA_FIELD(code_object_offset);
}

template <typename ArchiveT>
void
save(ArchiveT& ar, rocprofiler_pc_sampling_hw_id_v0_t data)
{
    ROCP_SDK_SAVE_DATA_BITFIELD("chiplet", chiplet);
    ROCP_SDK_SAVE_DATA_BITFIELD("wave_id", wave_id);
    ROCP_SDK_SAVE_DATA_BITFIELD("simd_id", simd_id);
    ROCP_SDK_SAVE_DATA_BITFIELD("pipe_id", pipe_id);
    ROCP_SDK_SAVE_DATA_BITFIELD("cu_or_wgp_id", cu_or_wgp_id);
    ROCP_SDK_SAVE_DATA_BITFIELD("shader_array_id", shader_array_id);
    ROCP_SDK_SAVE_DATA_BITFIELD("shader_engine_id", shader_engine_id);
    ROCP_SDK_SAVE_DATA_BITFIELD("workgroup_id ", workgroup_id);
    ROCP_SDK_SAVE_DATA_BITFIELD("vm_id", vm_id);
    ROCP_SDK_SAVE_DATA_BITFIELD("queue_id", queue_id);
    ROCP_SDK_SAVE_DATA_BITFIELD("microengine_id", microengine_id);
}

template <typename ArchiveT>
void
save(ArchiveT& ar, rocprofiler_pc_sampling_record_host_trap_v0_t data)
{
    ROCP_SDK_SAVE_DATA_FIELD(hw_id);
    ROCP_SDK_SAVE_DATA_FIELD(pc);
    ROCP_SDK_SAVE_DATA_FIELD(exec_mask);
    ROCP_SDK_SAVE_DATA_FIELD(timestamp);
    ROCP_SDK_SAVE_DATA_FIELD(dispatch_id);
    ROCP_SDK_SAVE_DATA_VALUE("corr_id", correlation_id);
    ROCP_SDK_SAVE_DATA_VALUE("wrkgrp_id", workgroup_id);
    ROCP_SDK_SAVE_DATA_BITFIELD("wave_in_grp", wave_in_group);
}

template <typename ArchiveT>
void
save(ArchiveT& ar, rocprofiler_pc_sampling_record_stochastic_header_t data)
{
    ROCP_SDK_SAVE_DATA_BITFIELD("has_mem_cnt", has_memory_counter);
}

template <typename ArchiveT>
void
save_pc_sampling_inst_type(ArchiveT& ar, rocprofiler_pc_sampling_instruction_type_t inst_type)
{
    ar(make_nvp("inst_type",
                std::string(rocprofiler_get_pc_sampling_instruction_type_name(inst_type))));
}

template <typename ArchiveT>
void
save_pc_sampling_stall_reason(ArchiveT&                                               ar,
                              rocprofiler_pc_sampling_instruction_not_issued_reason_t stall_reason)
{
    ar(make_nvp(
        "stall_reason",
        std::string(rocprofiler_get_pc_sampling_instruction_not_issued_reason_name(stall_reason))));
}

template <typename ArchiveT>
void
save(ArchiveT& ar, rocprofiler_pc_sampling_snapshot_v0_t data)
{
    save_pc_sampling_stall_reason(
        ar,
        static_cast<rocprofiler_pc_sampling_instruction_not_issued_reason_t>(
            data.reason_not_issued));

    ROCP_SDK_SAVE_DATA_BITFIELD("dual_issue_valu", dual_issue_valu);

    // Arb state (pipe issued)
    ROCP_SDK_SAVE_DATA_BITFIELD("arb_state_issue_valu", arb_state_issue_valu);
    ROCP_SDK_SAVE_DATA_BITFIELD("arb_state_issue_matrix", arb_state_issue_matrix);
    ROCP_SDK_SAVE_DATA_BITFIELD("arb_state_issue_lds", arb_state_issue_lds);
    ROCP_SDK_SAVE_DATA_BITFIELD("arb_state_issue_lds_direct", arb_state_issue_lds_direct);
    ROCP_SDK_SAVE_DATA_BITFIELD("arb_state_issue_scalar", arb_state_issue_scalar);
    ROCP_SDK_SAVE_DATA_BITFIELD("arb_state_issue_vmem_tex", arb_state_issue_vmem_tex);
    ROCP_SDK_SAVE_DATA_BITFIELD("arb_state_issue_flat", arb_state_issue_flat);
    ROCP_SDK_SAVE_DATA_BITFIELD("arb_state_issue_exp", arb_state_issue_exp);
    ROCP_SDK_SAVE_DATA_BITFIELD("arb_state_issue_misc", arb_state_issue_misc);
    ROCP_SDK_SAVE_DATA_BITFIELD("arb_state_issue_brmsg", arb_state_issue_brmsg);
    // Arb state (pipe stalled)
    ROCP_SDK_SAVE_DATA_BITFIELD("arb_state_stall_valu", arb_state_stall_valu);
    ROCP_SDK_SAVE_DATA_BITFIELD("arb_state_stall_matrix", arb_state_stall_matrix);
    ROCP_SDK_SAVE_DATA_BITFIELD("arb_state_stall_lds", arb_state_stall_lds);
    ROCP_SDK_SAVE_DATA_BITFIELD("arb_state_stall_lds_direct", arb_state_stall_lds_direct);
    ROCP_SDK_SAVE_DATA_BITFIELD("arb_state_stall_scalar", arb_state_stall_scalar);
    ROCP_SDK_SAVE_DATA_BITFIELD("arb_state_stall_vmem_tex", arb_state_stall_vmem_tex);
    ROCP_SDK_SAVE_DATA_BITFIELD("arb_state_stall_flat", arb_state_stall_flat);
    ROCP_SDK_SAVE_DATA_BITFIELD("arb_state_stall_exp", arb_state_stall_exp);
    ROCP_SDK_SAVE_DATA_BITFIELD("arb_state_stall_misc", arb_state_stall_misc);
    ROCP_SDK_SAVE_DATA_BITFIELD("arb_state_stall_brmsg", arb_state_stall_brmsg);
}

template <typename ArchiveT>
void
save(ArchiveT& ar, rocprofiler_pc_sampling_record_stochastic_v0_t data)
{
    // flags specific for stochastic sampling
    ROCP_SDK_SAVE_DATA_FIELD(flags);

    // Common for host-trap and stochastic
    ROCP_SDK_SAVE_DATA_FIELD(hw_id);
    ROCP_SDK_SAVE_DATA_FIELD(pc);
    ROCP_SDK_SAVE_DATA_FIELD(exec_mask);
    ROCP_SDK_SAVE_DATA_FIELD(timestamp);
    ROCP_SDK_SAVE_DATA_FIELD(dispatch_id);
    ROCP_SDK_SAVE_DATA_VALUE("corr_id", correlation_id);
    ROCP_SDK_SAVE_DATA_VALUE("wrkgrp_id", workgroup_id);
    ROCP_SDK_SAVE_DATA_BITFIELD("wave_in_grp", wave_in_group);

    // fields specific for stochastic
    ROCP_SDK_SAVE_DATA_BITFIELD("wave_issued", wave_issued);
    save_pc_sampling_inst_type(
        ar, static_cast<rocprofiler_pc_sampling_instruction_type_t>(data.inst_type));
    ROCP_SDK_SAVE_DATA_BITFIELD("wave_cnt", wave_count);
    ROCP_SDK_SAVE_DATA_FIELD(snapshot);

    // TODO: add memory counters
}

template <typename ArchiveT>
void
save(ArchiveT& ar, rocprofiler_agent_io_link_t data)
{
    ROCP_SDK_SAVE_DATA_FIELD(type);
    ROCP_SDK_SAVE_DATA_FIELD(version_major);
    ROCP_SDK_SAVE_DATA_FIELD(version_minor);
    ROCP_SDK_SAVE_DATA_FIELD(node_from);
    ROCP_SDK_SAVE_DATA_FIELD(node_to);
    ROCP_SDK_SAVE_DATA_FIELD(weight);
    ROCP_SDK_SAVE_DATA_FIELD(min_latency);
    ROCP_SDK_SAVE_DATA_FIELD(max_latency);
    ROCP_SDK_SAVE_DATA_FIELD(min_bandwidth);
    ROCP_SDK_SAVE_DATA_FIELD(max_bandwidth);
    ROCP_SDK_SAVE_DATA_FIELD(recommended_transfer_size);
    ROCP_SDK_SAVE_DATA_FIELD(flags);
}

template <typename ArchiveT>
void
save(ArchiveT& ar, rocprofiler_agent_mem_bank_t data)
{
    ROCP_SDK_SAVE_DATA_FIELD(heap_type);
    ROCP_SDK_SAVE_DATA_FIELD(flags);
    ROCP_SDK_SAVE_DATA_FIELD(width);
    ROCP_SDK_SAVE_DATA_FIELD(mem_clk_max);
    ROCP_SDK_SAVE_DATA_FIELD(size_in_bytes);
}

template <typename ArchiveT>
void
save(ArchiveT& ar, rocprofiler_pc_sampling_configuration_t data)
{
    ROCP_SDK_SAVE_DATA_FIELD(method);
    ROCP_SDK_SAVE_DATA_FIELD(unit);
    ROCP_SDK_SAVE_DATA_FIELD(min_interval);
    ROCP_SDK_SAVE_DATA_FIELD(max_interval);
    ROCP_SDK_SAVE_DATA_FIELD(flags);
}

template <typename ArchiveT>
void
save(ArchiveT& ar, rocprofiler_agent_runtime_visiblity_t data)
{
    ROCP_SDK_SAVE_DATA_BITFIELD("hsa", hsa);
    ROCP_SDK_SAVE_DATA_BITFIELD("hip", hip);
    ROCP_SDK_SAVE_DATA_BITFIELD("rccl", rccl);
    ROCP_SDK_SAVE_DATA_BITFIELD("rocdecode", rocdecode);
}

template <typename ArchiveT>
void
save(ArchiveT& ar, rocprofiler_uuid_t data)
{
    ROCP_SDK_SAVE_DATA_FIELD(value);
}

template <typename ArchiveT>
void
save(ArchiveT& ar, const rocprofiler_agent_v0_t& data)
{
    ROCP_SDK_SAVE_DATA_FIELD(size);
    ROCP_SDK_SAVE_DATA_FIELD(id);
    ROCP_SDK_SAVE_DATA_FIELD(type);
    ROCP_SDK_SAVE_DATA_FIELD(cpu_cores_count);
    ROCP_SDK_SAVE_DATA_FIELD(simd_count);
    ROCP_SDK_SAVE_DATA_FIELD(mem_banks_count);
    ROCP_SDK_SAVE_DATA_FIELD(caches_count);
    ROCP_SDK_SAVE_DATA_FIELD(io_links_count);
    ROCP_SDK_SAVE_DATA_FIELD(cpu_core_id_base);
    ROCP_SDK_SAVE_DATA_FIELD(simd_id_base);
    ROCP_SDK_SAVE_DATA_FIELD(max_waves_per_simd);
    ROCP_SDK_SAVE_DATA_FIELD(lds_size_in_kb);
    ROCP_SDK_SAVE_DATA_FIELD(gds_size_in_kb);
    ROCP_SDK_SAVE_DATA_FIELD(num_gws);
    ROCP_SDK_SAVE_DATA_FIELD(wave_front_size);
    ROCP_SDK_SAVE_DATA_FIELD(num_xcc);
    ROCP_SDK_SAVE_DATA_FIELD(cu_count);
    ROCP_SDK_SAVE_DATA_FIELD(array_count);
    ROCP_SDK_SAVE_DATA_FIELD(num_shader_banks);
    ROCP_SDK_SAVE_DATA_FIELD(simd_arrays_per_engine);
    ROCP_SDK_SAVE_DATA_FIELD(cu_per_simd_array);
    ROCP_SDK_SAVE_DATA_FIELD(simd_per_cu);
    ROCP_SDK_SAVE_DATA_FIELD(max_slots_scratch_cu);
    ROCP_SDK_SAVE_DATA_FIELD(gfx_target_version);
    ROCP_SDK_SAVE_DATA_FIELD(vendor_id);
    ROCP_SDK_SAVE_DATA_FIELD(device_id);
    ROCP_SDK_SAVE_DATA_FIELD(location_id);
    ROCP_SDK_SAVE_DATA_FIELD(domain);
    ROCP_SDK_SAVE_DATA_FIELD(drm_render_minor);
    ROCP_SDK_SAVE_DATA_FIELD(num_sdma_engines);
    ROCP_SDK_SAVE_DATA_FIELD(num_sdma_xgmi_engines);
    ROCP_SDK_SAVE_DATA_FIELD(num_sdma_queues_per_engine);
    ROCP_SDK_SAVE_DATA_FIELD(num_cp_queues);
    ROCP_SDK_SAVE_DATA_FIELD(max_engine_clk_ccompute);
    ROCP_SDK_SAVE_DATA_FIELD(max_engine_clk_fcompute);
    ROCP_SDK_SAVE_DATA_FIELD(sdma_fw_version);
    ROCP_SDK_SAVE_DATA_FIELD(fw_version);
    ROCP_SDK_SAVE_DATA_FIELD(capability);
    ROCP_SDK_SAVE_DATA_FIELD(cu_per_engine);
    ROCP_SDK_SAVE_DATA_FIELD(max_waves_per_cu);
    ROCP_SDK_SAVE_DATA_FIELD(family_id);
    ROCP_SDK_SAVE_DATA_FIELD(workgroup_max_size);
    ROCP_SDK_SAVE_DATA_FIELD(grid_max_size);
    ROCP_SDK_SAVE_DATA_FIELD(local_mem_size);
    ROCP_SDK_SAVE_DATA_FIELD(hive_id);
    ROCP_SDK_SAVE_DATA_FIELD(gpu_id);
    ROCP_SDK_SAVE_DATA_FIELD(workgroup_max_dim);
    ROCP_SDK_SAVE_DATA_FIELD(grid_max_dim);
    ROCP_SDK_SAVE_DATA_CSTR(name);
    ROCP_SDK_SAVE_DATA_CSTR(vendor_name);
    ROCP_SDK_SAVE_DATA_CSTR(product_name);
    ROCP_SDK_SAVE_DATA_CSTR(model_name);
    ROCP_SDK_SAVE_DATA_FIELD(node_id);
    ROCP_SDK_SAVE_DATA_FIELD(logical_node_id);
    ROCP_SDK_SAVE_DATA_FIELD(logical_node_type_id);
    ROCP_SDK_SAVE_DATA_FIELD(runtime_visibility);
    ROCP_SDK_SAVE_DATA_FIELD(uuid);

    auto generate = [&](auto name, const auto* value, uint64_t size) {
        using value_type = std::remove_const_t<std::remove_pointer_t<decltype(value)>>;
        auto vec         = std::vector<value_type>{};
        vec.reserve(size);
        for(uint64_t i = 0; i < size; ++i)
            vec.emplace_back(value[i]);
        ar(make_nvp(name, vec));
    };

    generate("mem_banks", data.mem_banks, data.mem_banks_count);
    generate("caches", data.caches, data.caches_count);
    generate("io_links", data.io_links, data.io_links_count);
}

template <typename ArchiveT>
void
save(ArchiveT& ar, rocprofiler_counter_info_v0_t data)
{
    ROCP_SDK_SAVE_DATA_FIELD(id);
    ROCP_SDK_SAVE_DATA_BITFIELD("is_constant", is_constant);
    ROCP_SDK_SAVE_DATA_BITFIELD("is_derived", is_derived);
    ROCP_SDK_SAVE_DATA_CSTR(name);
    ROCP_SDK_SAVE_DATA_CSTR(description);
    ROCP_SDK_SAVE_DATA_CSTR(block);
    ROCP_SDK_SAVE_DATA_CSTR(expression);
}

template <typename ArchiveT>
void
save(ArchiveT& ar, rocprofiler_counter_info_v1_t data)
{
    ROCP_SDK_SAVE_DATA_FIELD(id);
    ROCP_SDK_SAVE_DATA_BITFIELD("is_constant", is_constant);
    ROCP_SDK_SAVE_DATA_BITFIELD("is_derived", is_derived);
    ROCP_SDK_SAVE_DATA_CSTR(name);
    ROCP_SDK_SAVE_DATA_CSTR(description);
    ROCP_SDK_SAVE_DATA_CSTR(block);
    ROCP_SDK_SAVE_DATA_CSTR(expression);

    auto convert = [](const auto* val, uint64_t sz) {
        using data_type = std::remove_cv_t<std::remove_pointer_t<decltype(val)>>;
        auto retval     = std::vector<data_type>{};
        for(uint64_t i = 0; i < sz; ++i)
            retval.emplace_back(val[i]);
        return retval;
    };

    ROCP_SDK_SAVE_VALUE("dims", convert(data.dimensions, data.dimensions_count));
    ROCP_SDK_SAVE_VALUE("instances", convert(data.instance_ids, data.instance_ids_count));
}

template <typename ArchiveT>
void
save(ArchiveT& ar, rocprofiler_counter_record_dimension_info_t data)
{
    ROCP_SDK_SAVE_DATA_FIELD(id);
    ROCP_SDK_SAVE_DATA_FIELD(instance_size);
    ROCP_SDK_SAVE_DATA_CSTR(name);
}

template <typename ArchiveT>
void
save(ArchiveT& ar, rocprofiler_callback_tracing_runtime_initialization_data_t data)
{
    ROCP_SDK_SAVE_DATA_FIELD(size);
    ROCP_SDK_SAVE_DATA_FIELD(version);
    ROCP_SDK_SAVE_DATA_FIELD(instance);
}

template <typename ArchiveT>
void
save(ArchiveT& ar, rocprofiler_buffer_tracing_runtime_initialization_record_t data)
{
    ROCP_SDK_SAVE_DATA_FIELD(size);
    ROCP_SDK_SAVE_DATA_FIELD(kind);
    ROCP_SDK_SAVE_DATA_FIELD(operation);
    ROCP_SDK_SAVE_DATA_FIELD(correlation_id);
    ROCP_SDK_SAVE_DATA_FIELD(timestamp);
    ROCP_SDK_SAVE_DATA_FIELD(thread_id);
    ROCP_SDK_SAVE_DATA_FIELD(version);
    ROCP_SDK_SAVE_DATA_FIELD(instance);
}

template <typename ArchiveT, typename EnumT, typename ValueT>
void
save(ArchiveT& ar, const rocprofiler::sdk::utility::name_info<EnumT, ValueT>& data)
{
    ar.makeArray();
    for(const auto& itr : data)
        ar(cereal::make_nvp("entry", itr));
}

template <typename ArchiveT, typename EnumT, typename ValueT>
void
save(ArchiveT& ar, const rocprofiler::sdk::utility::name_info_impl<EnumT, ValueT>& data)
{
    auto _name = std::string{data.name};
    auto _ops  = std::vector<std::string>{};
    _ops.reserve(data.operations.size());

    ar(cereal::make_nvp("kind", _name));
    for(auto itr : data.operations)
        _ops.emplace_back(itr);
    ar(cereal::make_nvp("operations", _ops));
}

ROCPROFILER_SDK_CEREAL_NAMESPACE_END

#undef ROCP_SDK_SAVE_DATA_FIELD
#undef ROCP_SDK_SAVE_DATA_VALUE
#undef ROCP_SDK_SAVE_DATA_CSTR
#undef ROCP_SDK_SAVE_DATA_BITFIELD
