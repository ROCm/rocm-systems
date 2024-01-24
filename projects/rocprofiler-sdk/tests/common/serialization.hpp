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
//

#pragma once

#include <rocprofiler-sdk/buffer.h>
#include <rocprofiler-sdk/callback_tracing.h>
#include <rocprofiler-sdk/external_correlation.h>
#include <rocprofiler-sdk/fwd.h>
#include <rocprofiler-sdk/internal_threading.h>
#include <rocprofiler-sdk/rocprofiler.h>

#include <cereal/archives/json.hpp>
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
#include <cereal/types/memory.hpp>
#include <cereal/types/optional.hpp>
#include <cereal/types/polymorphic.hpp>
#include <cereal/types/queue.hpp>
#include <cereal/types/set.hpp>
#include <cereal/types/stack.hpp>
#include <cereal/types/string.hpp>
#include <cereal/types/tuple.hpp>
#include <cereal/types/unordered_map.hpp>
#include <cereal/types/unordered_set.hpp>
#include <cereal/types/utility.hpp>
#include <cereal/types/variant.hpp>
#include <cereal/types/vector.hpp>

#define SAVE_DATA_FIELD(FIELD)       ar(make_nvp(#FIELD, data.FIELD))
#define SAVE_DATA_VALUE(NAME, VALUE) ar(make_nvp(NAME, data.VALUE))
#define SAVE_DATA_CSTR(FIELD)        ar(make_nvp(#FIELD, std::string{data.FIELD}))
#define SAVE_DATA_BITFIELD(NAME, VALUE)                                                            \
    {                                                                                              \
        auto _val = data.VALUE;                                                                    \
        ar(make_nvp(NAME, _val));                                                                  \
    }

namespace cereal
{
template <typename ArchiveT>
void
save(ArchiveT& ar, rocprofiler_context_id_t data)
{
    SAVE_DATA_FIELD(handle);
}

template <typename ArchiveT>
void
save(ArchiveT& ar, rocprofiler_agent_id_t data)
{
    SAVE_DATA_FIELD(handle);
}

template <typename ArchiveT>
void
save(ArchiveT& ar, hsa_agent_t data)
{
    SAVE_DATA_FIELD(handle);
}

template <typename ArchiveT>
void
save(ArchiveT& ar, rocprofiler_queue_id_t data)
{
    SAVE_DATA_FIELD(handle);
}

template <typename ArchiveT>
void
save(ArchiveT& ar, rocprofiler_correlation_id_t data)
{
    SAVE_DATA_FIELD(internal);
    SAVE_DATA_VALUE("external", external.value);
}

template <typename ArchiveT>
void
save(ArchiveT& ar, rocprofiler_dim3_t data)
{
    SAVE_DATA_FIELD(x);
    SAVE_DATA_FIELD(y);
    SAVE_DATA_FIELD(z);
}

template <typename ArchiveT>
void
save(ArchiveT& ar, rocprofiler_callback_tracing_code_object_load_data_t data)
{
    SAVE_DATA_FIELD(size);
    SAVE_DATA_FIELD(code_object_id);
    SAVE_DATA_FIELD(rocp_agent);
    SAVE_DATA_FIELD(hsa_agent);
    SAVE_DATA_CSTR(uri);
    SAVE_DATA_FIELD(load_base);
    SAVE_DATA_FIELD(load_size);
    SAVE_DATA_FIELD(load_delta);
    SAVE_DATA_FIELD(storage_type);
    if(data.storage_type == ROCPROFILER_CODE_OBJECT_STORAGE_TYPE_FILE)
    {
        SAVE_DATA_FIELD(storage_file);
    }
    else if(data.storage_type == ROCPROFILER_CODE_OBJECT_STORAGE_TYPE_MEMORY)
    {
        SAVE_DATA_FIELD(memory_base);
        SAVE_DATA_FIELD(memory_size);
    }
}

template <typename ArchiveT>
void
save(ArchiveT& ar, rocprofiler_callback_tracing_code_object_kernel_symbol_register_data_t data)
{
    SAVE_DATA_FIELD(size);
    SAVE_DATA_FIELD(kernel_id);
    SAVE_DATA_FIELD(code_object_id);
    SAVE_DATA_CSTR(kernel_name);
    SAVE_DATA_FIELD(kernel_object);
    SAVE_DATA_FIELD(kernarg_segment_size);
    SAVE_DATA_FIELD(kernarg_segment_alignment);
    SAVE_DATA_FIELD(group_segment_size);
    SAVE_DATA_FIELD(private_segment_size);
}

template <typename ArchiveT>
void
save(ArchiveT& ar, rocprofiler_hsa_api_retval_t data)
{
    SAVE_DATA_FIELD(uint64_t_retval);
}

template <typename ArchiveT>
void
save(ArchiveT& ar, rocprofiler_marker_api_retval_t data)
{
    SAVE_DATA_FIELD(int64_t_retval);
}

template <typename ArchiveT>
void
save(ArchiveT& ar, rocprofiler_callback_tracing_hsa_api_data_t data)
{
    SAVE_DATA_FIELD(size);
    // SAVE_DATA_FIELD(args);
    SAVE_DATA_FIELD(retval);
}

template <typename ArchiveT>
void
save(ArchiveT& ar, rocprofiler_callback_tracing_marker_api_data_t data)
{
    SAVE_DATA_FIELD(size);
    // SAVE_DATA_FIELD(args);
    SAVE_DATA_FIELD(retval);
}

template <typename ArchiveT>
void
save(ArchiveT& ar, rocprofiler_hip_api_retval_t data)
{
    SAVE_DATA_FIELD(hipError_t_retval);
}

template <typename ArchiveT>
void
save(ArchiveT& ar, rocprofiler_callback_tracing_hip_api_data_t data)
{
    SAVE_DATA_FIELD(size);
    // SAVE_DATA_FIELD(args);
    SAVE_DATA_FIELD(retval);
}

template <typename ArchiveT>
void
save(ArchiveT& ar, rocprofiler_callback_tracing_record_t data)
{
    SAVE_DATA_FIELD(context_id);
    SAVE_DATA_FIELD(thread_id);
    SAVE_DATA_FIELD(correlation_id);
    SAVE_DATA_FIELD(kind);
    SAVE_DATA_FIELD(operation);
    SAVE_DATA_FIELD(phase);
}

template <typename ArchiveT, typename Tp>
void
save_buffer_tracing_api_record(ArchiveT& ar, Tp data)
{
    SAVE_DATA_FIELD(size);
    SAVE_DATA_FIELD(kind);
    SAVE_DATA_FIELD(correlation_id);
    SAVE_DATA_FIELD(operation);
    SAVE_DATA_FIELD(start_timestamp);
    SAVE_DATA_FIELD(end_timestamp);
    SAVE_DATA_FIELD(thread_id);
}

template <typename ArchiveT>
void
save(ArchiveT& ar, rocprofiler_buffer_tracing_hsa_api_record_t data)
{
    save_buffer_tracing_api_record(ar, data);
}

template <typename ArchiveT>
void
save(ArchiveT& ar, rocprofiler_buffer_tracing_hip_api_record_t data)
{
    save_buffer_tracing_api_record(ar, data);
}

template <typename ArchiveT>
void
save(ArchiveT& ar, rocprofiler_buffer_tracing_marker_api_record_t data)
{
    save_buffer_tracing_api_record(ar, data);
}

template <typename ArchiveT>
void
save(ArchiveT& ar, rocprofiler_buffer_tracing_kernel_dispatch_record_t data)
{
    SAVE_DATA_FIELD(size);
    SAVE_DATA_FIELD(kind);
    SAVE_DATA_FIELD(correlation_id);
    SAVE_DATA_FIELD(start_timestamp);
    SAVE_DATA_FIELD(end_timestamp);
    SAVE_DATA_FIELD(agent_id);
    SAVE_DATA_FIELD(queue_id);
    SAVE_DATA_FIELD(kernel_id);
    SAVE_DATA_FIELD(private_segment_size);
    SAVE_DATA_FIELD(group_segment_size);
    SAVE_DATA_FIELD(workgroup_size);
    SAVE_DATA_FIELD(grid_size);
}

template <typename ArchiveT>
void
save(ArchiveT& ar, rocprofiler_buffer_tracing_memory_copy_record_t data)
{
    SAVE_DATA_FIELD(size);
    SAVE_DATA_FIELD(kind);
    SAVE_DATA_FIELD(correlation_id);
    SAVE_DATA_FIELD(operation);
    SAVE_DATA_FIELD(start_timestamp);
    SAVE_DATA_FIELD(end_timestamp);
    SAVE_DATA_FIELD(dst_agent_id);
    SAVE_DATA_FIELD(src_agent_id);
}

template <typename ArchiveT>
void
save(ArchiveT& ar, HsaCacheType data)
{
    SAVE_DATA_BITFIELD("Data", ui32.Data);
    SAVE_DATA_BITFIELD("Instruction", ui32.Instruction);
    SAVE_DATA_BITFIELD("CPU", ui32.CPU);
    SAVE_DATA_BITFIELD("HSACU", ui32.HSACU);
}

template <typename ArchiveT>
void
save(ArchiveT& ar, HSA_LINKPROPERTY data)
{
    SAVE_DATA_BITFIELD("Override", ui32.Override);
    SAVE_DATA_BITFIELD("NonCoherent", ui32.NonCoherent);
    SAVE_DATA_BITFIELD("NoAtomics32bit", ui32.NoAtomics32bit);
    SAVE_DATA_BITFIELD("NoAtomics64bit", ui32.NoAtomics64bit);
    SAVE_DATA_BITFIELD("NoPeerToPeerDMA", ui32.NoPeerToPeerDMA);
}

template <typename ArchiveT>
void
save(ArchiveT& ar, HSA_CAPABILITY data)
{
    SAVE_DATA_BITFIELD("HotPluggable", ui32.HotPluggable);
    SAVE_DATA_BITFIELD("HSAMMUPresent", ui32.HSAMMUPresent);
    SAVE_DATA_BITFIELD("SharedWithGraphics", ui32.SharedWithGraphics);
    SAVE_DATA_BITFIELD("QueueSizePowerOfTwo", ui32.QueueSizePowerOfTwo);
    SAVE_DATA_BITFIELD("QueueSize32bit", ui32.QueueSize32bit);
    SAVE_DATA_BITFIELD("QueueIdleEvent", ui32.QueueIdleEvent);
    SAVE_DATA_BITFIELD("VALimit", ui32.VALimit);
    SAVE_DATA_BITFIELD("WatchPointsSupported", ui32.WatchPointsSupported);
    SAVE_DATA_BITFIELD("WatchPointsTotalBits", ui32.WatchPointsTotalBits);
    SAVE_DATA_BITFIELD("DoorbellType", ui32.DoorbellType);
    SAVE_DATA_BITFIELD("AQLQueueDoubleMap", ui32.AQLQueueDoubleMap);
    SAVE_DATA_BITFIELD("DebugTrapSupported", ui32.DebugTrapSupported);
    SAVE_DATA_BITFIELD("WaveLaunchTrapOverrideSupported", ui32.WaveLaunchTrapOverrideSupported);
    SAVE_DATA_BITFIELD("WaveLaunchModeSupported", ui32.WaveLaunchModeSupported);
    SAVE_DATA_BITFIELD("PreciseMemoryOperationsSupported", ui32.PreciseMemoryOperationsSupported);
    SAVE_DATA_BITFIELD("DEPRECATED_SRAM_EDCSupport", ui32.DEPRECATED_SRAM_EDCSupport);
    SAVE_DATA_BITFIELD("Mem_EDCSupport", ui32.Mem_EDCSupport);
    SAVE_DATA_BITFIELD("RASEventNotify", ui32.RASEventNotify);
    SAVE_DATA_BITFIELD("ASICRevision", ui32.ASICRevision);
    SAVE_DATA_BITFIELD("SRAM_EDCSupport", ui32.SRAM_EDCSupport);
    SAVE_DATA_BITFIELD("SVMAPISupported", ui32.SVMAPISupported);
    SAVE_DATA_BITFIELD("CoherentHostAccess", ui32.CoherentHostAccess);
    SAVE_DATA_BITFIELD("DebugSupportedFirmware", ui32.DebugSupportedFirmware);
    SAVE_DATA_BITFIELD("Reserved", ui32.Reserved);
}

template <typename ArchiveT>
void
save(ArchiveT& ar, HSA_MEMORYPROPERTY data)
{
    SAVE_DATA_BITFIELD("HotPluggable", ui32.HotPluggable);
    SAVE_DATA_BITFIELD("NonVolatile", ui32.NonVolatile);
}

template <typename ArchiveT>
void
save(ArchiveT& ar, HSA_ENGINE_VERSION data)
{
    SAVE_DATA_BITFIELD("uCodeSDMA", uCodeSDMA);
    SAVE_DATA_BITFIELD("uCodeRes", uCodeRes);
}

template <typename ArchiveT>
void
save(ArchiveT& ar, HSA_ENGINE_ID data)
{
    SAVE_DATA_BITFIELD("uCode", ui32.uCode);
    SAVE_DATA_BITFIELD("Major", ui32.Major);
    SAVE_DATA_BITFIELD("Minor", ui32.Minor);
    SAVE_DATA_BITFIELD("Stepping", ui32.Stepping);
}

template <typename ArchiveT>
void
save(ArchiveT& ar, rocprofiler_agent_cache_t data)
{
    SAVE_DATA_FIELD(processor_id_low);
    SAVE_DATA_FIELD(size);
    SAVE_DATA_FIELD(level);
    SAVE_DATA_FIELD(cache_line_size);
    SAVE_DATA_FIELD(cache_lines_per_tag);
    SAVE_DATA_FIELD(association);
    SAVE_DATA_FIELD(latency);
    SAVE_DATA_FIELD(type);
}

template <typename ArchiveT>
void
save(ArchiveT& ar, rocprofiler_agent_io_link_t data)
{
    SAVE_DATA_FIELD(type);
    SAVE_DATA_FIELD(version_major);
    SAVE_DATA_FIELD(version_minor);
    SAVE_DATA_FIELD(node_from);
    SAVE_DATA_FIELD(node_to);
    SAVE_DATA_FIELD(weight);
    SAVE_DATA_FIELD(min_latency);
    SAVE_DATA_FIELD(max_latency);
    SAVE_DATA_FIELD(min_bandwidth);
    SAVE_DATA_FIELD(max_bandwidth);
    SAVE_DATA_FIELD(recommended_transfer_size);
    SAVE_DATA_FIELD(flags);
}

template <typename ArchiveT>
void
save(ArchiveT& ar, rocprofiler_agent_mem_bank_t data)
{
    SAVE_DATA_FIELD(heap_type);
    SAVE_DATA_FIELD(flags);
    SAVE_DATA_FIELD(width);
    SAVE_DATA_FIELD(mem_clk_max);
    SAVE_DATA_FIELD(size_in_bytes);
}

template <typename ArchiveT>
void
save(ArchiveT& ar, rocprofiler_pc_sampling_configuration_t data)
{
    SAVE_DATA_FIELD(method);
    SAVE_DATA_FIELD(unit);
    SAVE_DATA_FIELD(min_interval);
    SAVE_DATA_FIELD(max_interval);
    SAVE_DATA_FIELD(flags);
}

template <typename ArchiveT>
void
save(ArchiveT& ar, const rocprofiler_agent_t& data)
{
    SAVE_DATA_FIELD(size);
    SAVE_DATA_FIELD(id);
    SAVE_DATA_FIELD(type);
    SAVE_DATA_FIELD(cpu_cores_count);
    SAVE_DATA_FIELD(simd_count);
    SAVE_DATA_FIELD(mem_banks_count);
    SAVE_DATA_FIELD(caches_count);
    SAVE_DATA_FIELD(io_links_count);
    SAVE_DATA_FIELD(cpu_core_id_base);
    SAVE_DATA_FIELD(simd_id_base);
    SAVE_DATA_FIELD(max_waves_per_simd);
    SAVE_DATA_FIELD(lds_size_in_kb);
    SAVE_DATA_FIELD(gds_size_in_kb);
    SAVE_DATA_FIELD(num_gws);
    SAVE_DATA_FIELD(wave_front_size);
    SAVE_DATA_FIELD(num_xcc);
    SAVE_DATA_FIELD(cu_count);
    SAVE_DATA_FIELD(array_count);
    SAVE_DATA_FIELD(num_shader_banks);
    SAVE_DATA_FIELD(simd_arrays_per_engine);
    SAVE_DATA_FIELD(cu_per_simd_array);
    SAVE_DATA_FIELD(simd_per_cu);
    SAVE_DATA_FIELD(max_slots_scratch_cu);
    SAVE_DATA_FIELD(gfx_target_version);
    SAVE_DATA_FIELD(vendor_id);
    SAVE_DATA_FIELD(device_id);
    SAVE_DATA_FIELD(location_id);
    SAVE_DATA_FIELD(domain);
    SAVE_DATA_FIELD(drm_render_minor);
    SAVE_DATA_FIELD(num_sdma_engines);
    SAVE_DATA_FIELD(num_sdma_xgmi_engines);
    SAVE_DATA_FIELD(num_sdma_queues_per_engine);
    SAVE_DATA_FIELD(num_cp_queues);
    SAVE_DATA_FIELD(max_engine_clk_ccompute);
    SAVE_DATA_FIELD(max_engine_clk_fcompute);
    SAVE_DATA_FIELD(sdma_fw_version);
    SAVE_DATA_FIELD(fw_version);
    SAVE_DATA_FIELD(capability);
    SAVE_DATA_FIELD(cu_per_engine);
    SAVE_DATA_FIELD(max_waves_per_cu);
    SAVE_DATA_FIELD(family_id);
    SAVE_DATA_FIELD(workgroup_max_size);
    SAVE_DATA_FIELD(grid_max_size);
    SAVE_DATA_FIELD(local_mem_size);
    SAVE_DATA_FIELD(hive_id);
    SAVE_DATA_FIELD(gpu_id);
    SAVE_DATA_FIELD(workgroup_max_dim);
    SAVE_DATA_FIELD(grid_max_dim);
    SAVE_DATA_CSTR(name);
    SAVE_DATA_CSTR(vendor_name);
    SAVE_DATA_CSTR(product_name);
    SAVE_DATA_CSTR(model_name);
    SAVE_DATA_FIELD(num_pc_sampling_configs);
    SAVE_DATA_FIELD(node_id);

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
}  // namespace cereal

#undef SAVE_DATA_FIELD
