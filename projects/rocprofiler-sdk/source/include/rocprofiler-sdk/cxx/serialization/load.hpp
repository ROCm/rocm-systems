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

#include <rocprofiler-sdk/buffer.h>
#include <rocprofiler-sdk/callback_tracing.h>
#include <rocprofiler-sdk/external_correlation.h>
#include <rocprofiler-sdk/fwd.h>
#include <rocprofiler-sdk/internal_threading.h>
#include <rocprofiler-sdk/rocprofiler.h>
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
#include <type_traits>
#include <utility>
#include <vector>

#if defined(ROCPROFILER_SDK_CXX_SERIALIZATION_LOAD_DEBUG) &&                                       \
    ROCPROFILER_SDK_CXX_SERIALIZATION_LOAD_DEBUG > 0
#    define ROCP_SDK_LOAD_MESSAGE(NAME)                                                            \
        std::clog << "[" << __PRETTY_FUNCTION__ << "] loading JSON field " << NAME << "...\n"      \
                  << std::flush
#else
#    define ROCP_SDK_LOAD_MESSAGE(NAME)
#endif

#define ROCP_SDK_LOAD_DATA_FIELD(FIELD)                                                            \
    ROCP_SDK_LOAD_MESSAGE(#FIELD);                                                                 \
    ar(make_nvp(#FIELD, data.FIELD))
#define ROCP_SDK_LOAD_DATA_VALUE(NAME, VALUE)                                                      \
    ROCP_SDK_LOAD_MESSAGE(NAME);                                                                   \
    ar(make_nvp(NAME, data.VALUE))
#define ROCP_SDK_LOAD_VALUE(NAME, VALUE)                                                           \
    ROCP_SDK_LOAD_MESSAGE(NAME);                                                                   \
    ar(make_nvp(NAME, VALUE))
#define ROCP_SDK_LOAD_DATA_CSTR(FIELD)                                                             \
    {                                                                                              \
        ROCP_SDK_LOAD_MESSAGE(#FIELD);                                                             \
        auto _val = new std::string{};                                                             \
        ar(make_nvp(#FIELD, *_val));                                                               \
        data.FIELD = _val->c_str();                                                                \
    }
#define ROCP_SDK_LOAD_DATA_BITFIELD(NAME, VALUE)                                                   \
    {                                                                                              \
        ROCP_SDK_LOAD_MESSAGE(NAME);                                                               \
        auto _val = data.VALUE;                                                                    \
        ar(make_nvp(NAME, _val));                                                                  \
        data.VALUE = _val;                                                                         \
    }

#if !defined(ROCPROFILER_SDK_CEREAL_NAMESPACE_BEGIN)
#    define ROCPROFILER_SDK_CEREAL_NAMESPACE_BEGIN                                                 \
        namespace cereal                                                                           \
        {
#endif

#if !defined(ROCPROFILER_SDK_CEREAL_NAMESPACE_END)
#    define ROCPROFILER_SDK_CEREAL_NAMESPACE_END }  // namespace cereal
#endif

ROCPROFILER_SDK_CEREAL_NAMESPACE_BEGIN

template <typename ArchiveT>
void
load(ArchiveT& ar, rocprofiler_agent_id_t& data)
{
    ROCP_SDK_LOAD_DATA_FIELD(handle);
}

template <typename ArchiveT>
void
load(ArchiveT& ar, HsaCacheType& data)
{
    ROCP_SDK_LOAD_DATA_BITFIELD("Data", ui32.Data);
    ROCP_SDK_LOAD_DATA_BITFIELD("Instruction", ui32.Instruction);
    ROCP_SDK_LOAD_DATA_BITFIELD("CPU", ui32.CPU);
    ROCP_SDK_LOAD_DATA_BITFIELD("HSACU", ui32.HSACU);
}

template <typename ArchiveT>
void
load(ArchiveT& ar, HSA_LINKPROPERTY& data)
{
    ROCP_SDK_LOAD_DATA_BITFIELD("Override", ui32.Override);
    ROCP_SDK_LOAD_DATA_BITFIELD("NonCoherent", ui32.NonCoherent);
    ROCP_SDK_LOAD_DATA_BITFIELD("NoAtomics32bit", ui32.NoAtomics32bit);
    ROCP_SDK_LOAD_DATA_BITFIELD("NoAtomics64bit", ui32.NoAtomics64bit);
    ROCP_SDK_LOAD_DATA_BITFIELD("NoPeerToPeerDMA", ui32.NoPeerToPeerDMA);
}

template <typename ArchiveT>
void
load(ArchiveT& ar, HSA_CAPABILITY& data)
{
    ROCP_SDK_LOAD_DATA_BITFIELD("HotPluggable", ui32.HotPluggable);
    ROCP_SDK_LOAD_DATA_BITFIELD("HSAMMUPresent", ui32.HSAMMUPresent);
    ROCP_SDK_LOAD_DATA_BITFIELD("SharedWithGraphics", ui32.SharedWithGraphics);
    ROCP_SDK_LOAD_DATA_BITFIELD("QueueSizePowerOfTwo", ui32.QueueSizePowerOfTwo);
    ROCP_SDK_LOAD_DATA_BITFIELD("QueueSize32bit", ui32.QueueSize32bit);
    ROCP_SDK_LOAD_DATA_BITFIELD("QueueIdleEvent", ui32.QueueIdleEvent);
    ROCP_SDK_LOAD_DATA_BITFIELD("VALimit", ui32.VALimit);
    ROCP_SDK_LOAD_DATA_BITFIELD("WatchPointsSupported", ui32.WatchPointsSupported);
    ROCP_SDK_LOAD_DATA_BITFIELD("WatchPointsTotalBits", ui32.WatchPointsTotalBits);
    ROCP_SDK_LOAD_DATA_BITFIELD("DoorbellType", ui32.DoorbellType);
    ROCP_SDK_LOAD_DATA_BITFIELD("AQLQueueDoubleMap", ui32.AQLQueueDoubleMap);
    ROCP_SDK_LOAD_DATA_BITFIELD("DebugTrapSupported", ui32.DebugTrapSupported);
    ROCP_SDK_LOAD_DATA_BITFIELD("WaveLaunchTrapOverrideSupported",
                                ui32.WaveLaunchTrapOverrideSupported);
    ROCP_SDK_LOAD_DATA_BITFIELD("WaveLaunchModeSupported", ui32.WaveLaunchModeSupported);
    ROCP_SDK_LOAD_DATA_BITFIELD("PreciseMemoryOperationsSupported",
                                ui32.PreciseMemoryOperationsSupported);
    ROCP_SDK_LOAD_DATA_BITFIELD("DEPRECATED_SRAM_EDCSupport", ui32.DEPRECATED_SRAM_EDCSupport);
    ROCP_SDK_LOAD_DATA_BITFIELD("Mem_EDCSupport", ui32.Mem_EDCSupport);
    ROCP_SDK_LOAD_DATA_BITFIELD("RASEventNotify", ui32.RASEventNotify);
    ROCP_SDK_LOAD_DATA_BITFIELD("ASICRevision", ui32.ASICRevision);
    ROCP_SDK_LOAD_DATA_BITFIELD("SRAM_EDCSupport", ui32.SRAM_EDCSupport);
    ROCP_SDK_LOAD_DATA_BITFIELD("SVMAPISupported", ui32.SVMAPISupported);
    ROCP_SDK_LOAD_DATA_BITFIELD("CoherentHostAccess", ui32.CoherentHostAccess);
    ROCP_SDK_LOAD_DATA_BITFIELD("DebugSupportedFirmware", ui32.DebugSupportedFirmware);
}

template <typename ArchiveT>
void
load(ArchiveT& ar, rocprofiler_dim3_t& data)
{
    ROCP_SDK_LOAD_DATA_FIELD(x);
    ROCP_SDK_LOAD_DATA_FIELD(y);
    ROCP_SDK_LOAD_DATA_FIELD(z);
}

template <typename ArchiveT>
void
load(ArchiveT& ar, HSA_MEMORYPROPERTY& data)
{
    ROCP_SDK_LOAD_DATA_BITFIELD("HotPluggable", ui32.HotPluggable);
    ROCP_SDK_LOAD_DATA_BITFIELD("NonVolatile", ui32.NonVolatile);
}

template <typename ArchiveT>
void
load(ArchiveT& ar, HSA_ENGINE_VERSION& data)
{
    ROCP_SDK_LOAD_DATA_BITFIELD("uCodeSDMA", uCodeSDMA);
    ROCP_SDK_LOAD_DATA_BITFIELD("uCodeRes", uCodeRes);
}

template <typename ArchiveT>
void
load(ArchiveT& ar, HSA_ENGINE_ID& data)
{
    ROCP_SDK_LOAD_DATA_BITFIELD("uCode", ui32.uCode);
    ROCP_SDK_LOAD_DATA_BITFIELD("Major", ui32.Major);
    ROCP_SDK_LOAD_DATA_BITFIELD("Minor", ui32.Minor);
    ROCP_SDK_LOAD_DATA_BITFIELD("Stepping", ui32.Stepping);
}

template <typename ArchiveT>
void
load(ArchiveT& ar, rocprofiler_agent_cache_t& data)
{
    ROCP_SDK_LOAD_DATA_FIELD(processor_id_low);
    ROCP_SDK_LOAD_DATA_FIELD(size);
    ROCP_SDK_LOAD_DATA_FIELD(level);
    ROCP_SDK_LOAD_DATA_FIELD(cache_line_size);
    ROCP_SDK_LOAD_DATA_FIELD(cache_lines_per_tag);
    ROCP_SDK_LOAD_DATA_FIELD(association);
    ROCP_SDK_LOAD_DATA_FIELD(latency);
    ROCP_SDK_LOAD_DATA_FIELD(type);
}

template <typename ArchiveT>
void
load(ArchiveT& ar, rocprofiler_agent_io_link_t& data)
{
    ROCP_SDK_LOAD_DATA_FIELD(type);
    ROCP_SDK_LOAD_DATA_FIELD(version_major);
    ROCP_SDK_LOAD_DATA_FIELD(version_minor);
    ROCP_SDK_LOAD_DATA_FIELD(node_from);
    ROCP_SDK_LOAD_DATA_FIELD(node_to);
    ROCP_SDK_LOAD_DATA_FIELD(weight);
    ROCP_SDK_LOAD_DATA_FIELD(min_latency);
    ROCP_SDK_LOAD_DATA_FIELD(max_latency);
    ROCP_SDK_LOAD_DATA_FIELD(min_bandwidth);
    ROCP_SDK_LOAD_DATA_FIELD(max_bandwidth);
    ROCP_SDK_LOAD_DATA_FIELD(recommended_transfer_size);
    ROCP_SDK_LOAD_DATA_FIELD(flags);
}

template <typename ArchiveT>
void
load(ArchiveT& ar, rocprofiler_agent_mem_bank_t& data)
{
    ROCP_SDK_LOAD_DATA_FIELD(heap_type);
    ROCP_SDK_LOAD_DATA_FIELD(flags);
    ROCP_SDK_LOAD_DATA_FIELD(width);
    ROCP_SDK_LOAD_DATA_FIELD(mem_clk_max);
    ROCP_SDK_LOAD_DATA_FIELD(size_in_bytes);
}

template <typename ArchiveT>
void
load(ArchiveT& ar, rocprofiler_agent_v0_t& data)
{
    ROCP_SDK_LOAD_DATA_FIELD(size);
    ROCP_SDK_LOAD_DATA_FIELD(id);
    ROCP_SDK_LOAD_DATA_FIELD(type);
    ROCP_SDK_LOAD_DATA_FIELD(cpu_cores_count);
    ROCP_SDK_LOAD_DATA_FIELD(simd_count);
    ROCP_SDK_LOAD_DATA_FIELD(mem_banks_count);
    ROCP_SDK_LOAD_DATA_FIELD(caches_count);
    ROCP_SDK_LOAD_DATA_FIELD(io_links_count);
    ROCP_SDK_LOAD_DATA_FIELD(cpu_core_id_base);
    ROCP_SDK_LOAD_DATA_FIELD(simd_id_base);
    ROCP_SDK_LOAD_DATA_FIELD(max_waves_per_simd);
    ROCP_SDK_LOAD_DATA_FIELD(lds_size_in_kb);
    ROCP_SDK_LOAD_DATA_FIELD(gds_size_in_kb);
    ROCP_SDK_LOAD_DATA_FIELD(num_gws);
    ROCP_SDK_LOAD_DATA_FIELD(wave_front_size);
    ROCP_SDK_LOAD_DATA_FIELD(num_xcc);
    ROCP_SDK_LOAD_DATA_FIELD(cu_count);
    ROCP_SDK_LOAD_DATA_FIELD(array_count);
    ROCP_SDK_LOAD_DATA_FIELD(num_shader_banks);
    ROCP_SDK_LOAD_DATA_FIELD(simd_arrays_per_engine);
    ROCP_SDK_LOAD_DATA_FIELD(cu_per_simd_array);
    ROCP_SDK_LOAD_DATA_FIELD(simd_per_cu);
    ROCP_SDK_LOAD_DATA_FIELD(max_slots_scratch_cu);
    ROCP_SDK_LOAD_DATA_FIELD(gfx_target_version);
    ROCP_SDK_LOAD_DATA_FIELD(vendor_id);
    ROCP_SDK_LOAD_DATA_FIELD(device_id);
    ROCP_SDK_LOAD_DATA_FIELD(location_id);
    ROCP_SDK_LOAD_DATA_FIELD(domain);
    ROCP_SDK_LOAD_DATA_FIELD(drm_render_minor);
    ROCP_SDK_LOAD_DATA_FIELD(num_sdma_engines);
    ROCP_SDK_LOAD_DATA_FIELD(num_sdma_xgmi_engines);
    ROCP_SDK_LOAD_DATA_FIELD(num_sdma_queues_per_engine);
    ROCP_SDK_LOAD_DATA_FIELD(num_cp_queues);
    ROCP_SDK_LOAD_DATA_FIELD(max_engine_clk_ccompute);
    ROCP_SDK_LOAD_DATA_FIELD(max_engine_clk_fcompute);
    ROCP_SDK_LOAD_DATA_FIELD(sdma_fw_version);
    ROCP_SDK_LOAD_DATA_FIELD(fw_version);
    ROCP_SDK_LOAD_DATA_FIELD(capability);
    ROCP_SDK_LOAD_DATA_FIELD(cu_per_engine);
    ROCP_SDK_LOAD_DATA_FIELD(max_waves_per_cu);
    ROCP_SDK_LOAD_DATA_FIELD(family_id);
    ROCP_SDK_LOAD_DATA_FIELD(workgroup_max_size);
    ROCP_SDK_LOAD_DATA_FIELD(grid_max_size);
    ROCP_SDK_LOAD_DATA_FIELD(local_mem_size);
    ROCP_SDK_LOAD_DATA_FIELD(hive_id);
    ROCP_SDK_LOAD_DATA_FIELD(gpu_id);
    ROCP_SDK_LOAD_DATA_FIELD(workgroup_max_dim);
    ROCP_SDK_LOAD_DATA_FIELD(grid_max_dim);
    ROCP_SDK_LOAD_DATA_CSTR(name);
    ROCP_SDK_LOAD_DATA_CSTR(vendor_name);
    ROCP_SDK_LOAD_DATA_CSTR(product_name);
    ROCP_SDK_LOAD_DATA_CSTR(model_name);
    ROCP_SDK_LOAD_DATA_FIELD(node_id);
    ROCP_SDK_LOAD_DATA_FIELD(logical_node_id);
    ROCP_SDK_LOAD_DATA_FIELD(logical_node_type_id);

    auto generate = [&](auto name, const auto*& value, auto& size) {
        using value_type =
            std::remove_const_t<std::remove_pointer_t<std::remove_reference_t<decltype(value)>>>;
        auto vec = std::vector<value_type>{};
        ar(make_nvp(name, vec));
        size          = vec.size();
        auto* value_m = new value_type[size];
        for(uint64_t i = 0; i < size; ++i)
            value_m[i] = vec.at(i);
        value = value_m;
    };

    generate("mem_banks", data.mem_banks, data.mem_banks_count);
    generate("caches", data.caches, data.caches_count);
    generate("io_links", data.io_links, data.io_links_count);
}

template <typename ArchiveT>
void
load(ArchiveT& ar, rocprofiler_address_t& data)
{
    ROCP_SDK_LOAD_DATA_FIELD(handle);
}

template <typename ArchiveT>
void
load(ArchiveT& ar, rocprofiler_callback_tracing_code_object_load_data_t& data)
{
    ROCP_SDK_LOAD_DATA_FIELD(size);
    ROCP_SDK_LOAD_DATA_FIELD(code_object_id);
    ROCP_SDK_LOAD_DATA_FIELD(agent_id);
    ROCP_SDK_LOAD_DATA_CSTR(uri);
    ROCP_SDK_LOAD_DATA_FIELD(load_base);
    ROCP_SDK_LOAD_DATA_FIELD(load_size);
    ROCP_SDK_LOAD_DATA_FIELD(load_delta);
    ROCP_SDK_LOAD_DATA_FIELD(storage_type);
    if(data.storage_type == ROCPROFILER_CODE_OBJECT_STORAGE_TYPE_FILE)
    {
        ROCP_SDK_LOAD_DATA_FIELD(storage_file);
    }
    else if(data.storage_type == ROCPROFILER_CODE_OBJECT_STORAGE_TYPE_MEMORY)
    {
        ROCP_SDK_LOAD_DATA_FIELD(memory_base);
        ROCP_SDK_LOAD_DATA_FIELD(memory_size);
    }
}

template <typename ArchiveT>
void
load(ArchiveT& ar, rocprofiler_callback_tracing_code_object_kernel_symbol_register_data_t& data)
{
    ROCP_SDK_LOAD_DATA_FIELD(size);
    ROCP_SDK_LOAD_DATA_FIELD(kernel_id);
    ROCP_SDK_LOAD_DATA_FIELD(code_object_id);
    ROCP_SDK_LOAD_DATA_CSTR(kernel_name);
    ROCP_SDK_LOAD_DATA_FIELD(kernel_object);
    ROCP_SDK_LOAD_DATA_FIELD(kernarg_segment_size);
    ROCP_SDK_LOAD_DATA_FIELD(kernarg_segment_alignment);
    ROCP_SDK_LOAD_DATA_FIELD(group_segment_size);
    ROCP_SDK_LOAD_DATA_FIELD(private_segment_size);
    ROCP_SDK_LOAD_DATA_FIELD(sgpr_count);
    ROCP_SDK_LOAD_DATA_FIELD(arch_vgpr_count);
    ROCP_SDK_LOAD_DATA_FIELD(accum_vgpr_count);
    ROCP_SDK_LOAD_DATA_FIELD(kernel_code_entry_byte_offset);
    ROCP_SDK_LOAD_DATA_FIELD(kernel_address);
}

ROCPROFILER_SDK_CEREAL_NAMESPACE_END

#undef ROCP_SDK_LOAD_DATA_FIELD
#undef ROCP_SDK_LOAD_DATA_VALUE
#undef ROCP_SDK_LOAD_DATA_CSTR
#undef ROCP_SDK_LOAD_DATA_BITFIELD
