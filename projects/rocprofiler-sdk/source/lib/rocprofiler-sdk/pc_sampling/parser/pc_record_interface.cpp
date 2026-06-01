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

#include "lib/rocprofiler-sdk/pc_sampling/parser/pc_record_interface.hpp"

#include "lib/common/utility.hpp"

#include <type_traits>

template <>
uint64_t
PCSamplingParserContext::alloc<rocprofiler_pc_sampling_record_host_trap_v0_t>(
    rocprofiler_pc_sampling_record_host_trap_v0_t** buffer,
    uint64_t                                        size)
{
    std::unique_lock<std::shared_mutex> lock(mut);
    assert(buffer != nullptr);
    host_trap_data.emplace_back(
        std::make_unique<PCSamplingData<rocprofiler_pc_sampling_record_host_trap_v0_t>>(size));
    *buffer = host_trap_data.back()->samples.data();
    return size;
}

template <>
uint64_t
PCSamplingParserContext::alloc<rocprofiler_pc_sampling_record_stochastic_v0_t>(
    rocprofiler_pc_sampling_record_stochastic_v0_t** buffer,
    uint64_t                                         size)
{
    std::unique_lock<std::shared_mutex> lock(mut);
    assert(buffer != nullptr);
    stochastic_data.emplace_back(
        std::make_unique<PCSamplingData<rocprofiler_pc_sampling_record_stochastic_v0_t>>(size));
    *buffer = stochastic_data.back()->samples.data();
    return size;
}

// --- alloc specializations for v2 record types ---

template <>
uint64_t
PCSamplingParserContext::alloc<rocprofiler_pc_sampling_record_v0_t>(
    rocprofiler_pc_sampling_record_v0_t** buffer,
    uint64_t                              size)
{
    std::unique_lock<std::shared_mutex> lock(mut);
    assert(buffer != nullptr);
    v0_data.emplace_back(
        std::make_unique<PCSamplingData<rocprofiler_pc_sampling_record_v0_t>>(size));
    *buffer = v0_data.back()->samples.data();
    return size;
}

template <>
uint64_t
PCSamplingParserContext::alloc<rocprofiler_pc_sampling_record_v1_t>(
    rocprofiler_pc_sampling_record_v1_t** buffer,
    uint64_t                              size)
{
    std::unique_lock<std::shared_mutex> lock(mut);
    assert(buffer != nullptr);
    v1_data.emplace_back(
        std::make_unique<PCSamplingData<rocprofiler_pc_sampling_record_v1_t>>(size));
    *buffer = v1_data.back()->samples.data();
    return size;
}

template <>
uint64_t
PCSamplingParserContext::alloc<rocprofiler_pc_sampling_record_v2_t>(
    rocprofiler_pc_sampling_record_v2_t** buffer,
    uint64_t                              size)
{
    std::unique_lock<std::shared_mutex> lock(mut);
    assert(buffer != nullptr);
    v2_data.emplace_back(
        std::make_unique<PCSamplingData<rocprofiler_pc_sampling_record_v2_t>>(size));
    *buffer = v2_data.back()->samples.data();
    return size;
}

template <>
uint64_t
PCSamplingParserContext::alloc<rocprofiler_pc_sampling_record_v3_t>(
    rocprofiler_pc_sampling_record_v3_t** buffer,
    uint64_t                              size)
{
    std::unique_lock<std::shared_mutex> lock(mut);
    assert(buffer != nullptr);
    v3_data.emplace_back(
        std::make_unique<PCSamplingData<rocprofiler_pc_sampling_record_v3_t>>(size));
    *buffer = v3_data.back()->samples.data();
    return size;
}

template <>
uint64_t
PCSamplingParserContext::alloc<rocprofiler_pc_sampling_record_v4_t>(
    rocprofiler_pc_sampling_record_v4_t** buffer,
    uint64_t                              size)
{
    std::unique_lock<std::shared_mutex> lock(mut);
    assert(buffer != nullptr);
    v4_data.emplace_back(
        std::make_unique<PCSamplingData<rocprofiler_pc_sampling_record_v4_t>>(size));
    *buffer = v4_data.back()->samples.data();
    return size;
}

/**
 * @brief Get the appropriate parse function based on the GFXIP and sampling method.
 *
 * If the inappropriate sampling method is provided, it returns nullptr.
 */
template <typename GFXIP>
PCSamplingParserContext::parse_funct_ptr_t
PCSamplingParserContext::_get_parse_func_for_method(rocprofiler_pc_sampling_method_t pcs_method)
{
    if(pcs_method == ROCPROFILER_PC_SAMPLING_METHOD_HOST_TRAP)
    {
        return &PCSamplingParserContext::_parse<GFXIP,
                                                rocprofiler_pc_sampling_record_host_trap_v0_t,
                                                ROCPROFILER_PC_SAMPLING_METHOD_HOST_TRAP>;
    }
    else if(pcs_method == ROCPROFILER_PC_SAMPLING_METHOD_STOCHASTIC)
    {
        return &PCSamplingParserContext::_parse<GFXIP,
                                                rocprofiler_pc_sampling_record_stochastic_v0_t,
                                                ROCPROFILER_PC_SAMPLING_METHOD_STOCHASTIC>;
    }
    else
    {
        return nullptr;
    }
}

/**
 * @brief Get the appropriate parse function based on the GFXIP, requested v2 record kind, and
 * negotiated sampling method.
 *
 * The (record_kind, method) pair selects the @ref _parse specialization: the record kind picks
 * the output record type and the method is forwarded as the @c Method template argument, which
 * drives codegen of copySample / is_invalid_sample / add_upcoming_samples.
 *
 * Method compatibility per record kind:
 * - V0/V1/V3 are layout-compatible with BOTH methods (host-trap and stochastic).
 * - V2/V4 are stochastic-only. The (V2, HOST_TRAP) and (V4, HOST_TRAP) combinations are
 *   rejected upstream by validate_record_kinds_against_flags() in pc_sampling.cpp, so they
 *   must never reach here; they fall through to @c nullptr (treated as an internal error).
 *
 * V3/V4 record kinds are only available on GFX1250; on every other @c GFXIP this function
 * returns @c nullptr for them. It also returns @c nullptr for any record kind that is not one
 * of V0/V1/V2/V3/V4 (the @c default case), including
 * ::ROCPROFILER_PC_SAMPLING_RECORD_INVALID_SAMPLE and any future / unknown record kind values.
 *
 * The caller MUST check the return value before dispatching; a @c nullptr return means
 * "no parser available for this (GFXIP, record_kind, method) tuple" and the sample should be
 * skipped (the dispatching code in @ref parse treats this as an internal error).
 */
template <typename GFXIP>
PCSamplingParserContext::parse_funct_ptr_t
PCSamplingParserContext::_get_parse_func_for_record_kind_and_method(
    rocprofiler_pc_sampling_record_kind_t record_kind,
    rocprofiler_pc_sampling_method_t      pcs_method)
{
    const bool is_host_trap = pcs_method == ROCPROFILER_PC_SAMPLING_METHOD_HOST_TRAP;

    switch(record_kind)
    {
        case ROCPROFILER_PC_SAMPLING_RECORD_V0_SAMPLE:
            return is_host_trap
                       ? &PCSamplingParserContext::_parse<GFXIP,
                                                          rocprofiler_pc_sampling_record_v0_t,
                                                          ROCPROFILER_PC_SAMPLING_METHOD_HOST_TRAP>
                       : &PCSamplingParserContext::_parse<GFXIP,
                                                          rocprofiler_pc_sampling_record_v0_t,
                                                          ROCPROFILER_PC_SAMPLING_METHOD_STOCHASTIC>;
        case ROCPROFILER_PC_SAMPLING_RECORD_V1_SAMPLE:
            return is_host_trap
                       ? &PCSamplingParserContext::_parse<GFXIP,
                                                          rocprofiler_pc_sampling_record_v1_t,
                                                          ROCPROFILER_PC_SAMPLING_METHOD_HOST_TRAP>
                       : &PCSamplingParserContext::_parse<GFXIP,
                                                          rocprofiler_pc_sampling_record_v1_t,
                                                          ROCPROFILER_PC_SAMPLING_METHOD_STOCHASTIC>;
        case ROCPROFILER_PC_SAMPLING_RECORD_V2_SAMPLE:
            // V2 is stochastic-only; (V2, HOST_TRAP) is rejected by the validator and must
            // never reach here.
            if(is_host_trap) return nullptr;
            return &PCSamplingParserContext::_parse<GFXIP,
                                                    rocprofiler_pc_sampling_record_v2_t,
                                                    ROCPROFILER_PC_SAMPLING_METHOD_STOCHASTIC>;
        case ROCPROFILER_PC_SAMPLING_RECORD_V3_SAMPLE:
            if constexpr(std::is_same_v<GFXIP, GFX1250>)
                return is_host_trap
                           ? &PCSamplingParserContext::_parse<
                                 GFXIP,
                                 rocprofiler_pc_sampling_record_v3_t,
                                 ROCPROFILER_PC_SAMPLING_METHOD_HOST_TRAP>
                           : &PCSamplingParserContext::_parse<
                                 GFXIP,
                                 rocprofiler_pc_sampling_record_v3_t,
                                 ROCPROFILER_PC_SAMPLING_METHOD_STOCHASTIC>;
            else
                return nullptr;
        case ROCPROFILER_PC_SAMPLING_RECORD_V4_SAMPLE:
            // V4 is stochastic-only; (V4, HOST_TRAP) is rejected by the validator.
            if(is_host_trap) return nullptr;
            if constexpr(std::is_same_v<GFXIP, GFX1250>)
                return &PCSamplingParserContext::_parse<
                    GFXIP,
                    rocprofiler_pc_sampling_record_v4_t,
                    ROCPROFILER_PC_SAMPLING_METHOD_STOCHASTIC>;
            else
                return nullptr;
        default: return nullptr;
    }
}

pcsample_status_t
PCSamplingParserContext::parse(const upcoming_samples_t&             upcoming,
                               const generic_sample_t*               data_,
                               uint32_t                              gfx_target_version,
                               std::condition_variable&              midway_signal,
                               bool                                  bRocrBufferFlip,
                               rocprofiler_pc_sampling_record_kind_t requested_record_kind,
                               bool                                  deliver_invalid)
{
    auto gfxip_major = (gfx_target_version / 10000) % 100;
    auto gfxip_minor = (gfx_target_version / 100) % 100;
    auto pcs_method  = (upcoming.which_sample_type == AMD_HOST_TRAP_V1)
                           ? ROCPROFILER_PC_SAMPLING_METHOD_HOST_TRAP
                           : ROCPROFILER_PC_SAMPLING_METHOD_STOCHASTIC;

    // v2 API when a specific record kind is requested; otherwise legacy (v1) method-based dispatch.
    const bool is_v2 = requested_record_kind != ROCPROFILER_PC_SAMPLING_RECORD_NONE;

    // Template instantiation is faster!
    parse_funct_ptr_t parseSample_func = nullptr;
    if(gfxip_major == 9)
    {
        if(gfxip_minor == 5)
        {
            parseSample_func =
                is_v2 ? _get_parse_func_for_record_kind_and_method<GFX950>(requested_record_kind,
                                                                           pcs_method)
                      : _get_parse_func_for_method<GFX950>(pcs_method);
        }
        else
        {
            parseSample_func =
                is_v2 ? _get_parse_func_for_record_kind_and_method<GFX9>(requested_record_kind,
                                                                         pcs_method)
                      : _get_parse_func_for_method<GFX9>(pcs_method);
        }
    }
    else if(gfxip_major == 11)
    {
        parseSample_func =
            is_v2 ? _get_parse_func_for_record_kind_and_method<GFX11>(requested_record_kind,
                                                                      pcs_method)
                  : _get_parse_func_for_method<GFX11>(pcs_method);
    }
    else if(gfxip_major == 12)
    {
        if(gfxip_minor == 5)
        {
            parseSample_func =
                is_v2 ? _get_parse_func_for_record_kind_and_method<GFX1250>(requested_record_kind,
                                                                            pcs_method)
                      : _get_parse_func_for_method<GFX1250>(pcs_method);
        }
        else
        {
            parseSample_func =
                is_v2 ? _get_parse_func_for_record_kind_and_method<GFX12>(requested_record_kind,
                                                                          pcs_method)
                      : _get_parse_func_for_method<GFX12>(pcs_method);
        }
    }
    else
    {
        return PCSAMPLE_STATUS_INVALID_GFXIP;
    }

    if(parseSample_func == nullptr)
    {
        return PCSAMPLE_STATUS_INVALID_METHOD;
    }

    auto status = (this->*parseSample_func)(upcoming, data_, deliver_invalid);
    midway_signal.notify_all();

    if(!bRocrBufferFlip || status != PCSAMPLE_STATUS_SUCCESS) return status;

    return flushForgetList();
}

void
PCSamplingParserContext::newDispatch(const dispatch_pkt_id_t& pkt)
{
    std::unique_lock<std::shared_mutex> lock(mut);
    corr_map->newDispatch(pkt);
    active_dispatches[pkt.correlation_id.internal] = pkt;
}

void
PCSamplingParserContext::completeDispatch(uint64_t correlation_id)
{
    std::unique_lock<std::shared_mutex> lock(mut);
    forget_list.emplace(correlation_id);
}

pcsample_status_t
PCSamplingParserContext::flushForgetList()
{
    std::unique_lock<std::shared_mutex> lock(mut);
    pcsample_status_t                   status = PCSAMPLE_STATUS_SUCCESS;

    for(uint64_t id : forget_list)
    {
        if(active_dispatches.find(id) == active_dispatches.end())
        {
            status = PCSAMPLE_STATUS_PARSER_ERROR;
            continue;
        }
        const auto& pkt = active_dispatches.at(id);
        generate_id_completion_record(pkt);
        corr_map->forget(pkt);
        active_dispatches.erase(id);
    }
    forget_list.clear();
    return status;
}

bool
PCSamplingParserContext::shouldFlipRocrBuffer(const dispatch_pkt_id_t& pkt) const
{
    std::shared_lock<std::shared_mutex> lock(mut);
    return corr_map->checkDispatch(pkt);
}

// Compile-time mapping from a parsed record type to its public record-kind enum value.
template <typename PcSamplingRecordT>
constexpr rocprofiler_pc_sampling_record_kind_t record_kind_for();

template <>
constexpr rocprofiler_pc_sampling_record_kind_t
record_kind_for<rocprofiler_pc_sampling_record_host_trap_v0_t>()
{
    return ROCPROFILER_PC_SAMPLING_RECORD_HOST_TRAP_V0_SAMPLE;
}
template <>
constexpr rocprofiler_pc_sampling_record_kind_t
record_kind_for<rocprofiler_pc_sampling_record_stochastic_v0_t>()
{
    return ROCPROFILER_PC_SAMPLING_RECORD_STOCHASTIC_V0_SAMPLE;
}
template <>
constexpr rocprofiler_pc_sampling_record_kind_t
record_kind_for<rocprofiler_pc_sampling_record_v0_t>()
{
    return ROCPROFILER_PC_SAMPLING_RECORD_V0_SAMPLE;
}
template <>
constexpr rocprofiler_pc_sampling_record_kind_t
record_kind_for<rocprofiler_pc_sampling_record_v1_t>()
{
    return ROCPROFILER_PC_SAMPLING_RECORD_V1_SAMPLE;
}
template <>
constexpr rocprofiler_pc_sampling_record_kind_t
record_kind_for<rocprofiler_pc_sampling_record_v2_t>()
{
    return ROCPROFILER_PC_SAMPLING_RECORD_V2_SAMPLE;
}
template <>
constexpr rocprofiler_pc_sampling_record_kind_t
record_kind_for<rocprofiler_pc_sampling_record_v3_t>()
{
    return ROCPROFILER_PC_SAMPLING_RECORD_V3_SAMPLE;
}
template <>
constexpr rocprofiler_pc_sampling_record_kind_t
record_kind_for<rocprofiler_pc_sampling_record_v4_t>()
{
    return ROCPROFILER_PC_SAMPLING_RECORD_V4_SAMPLE;
}

/**
 * @brief Emit parsed records into the SDK buffer, applying the deliver/drop policy for invalid
 * samples.
 *
 * Method-aware via is_invalid_sample<RecordT, Method>: for HOST_TRAP V0/V1/V3 the invalid branch
 * is compiled out (the trait is a compile-time false), so this is a plain copy loop. For
 * stochastic records, invalid samples are dropped unless @p deliver_invalid is set, in which case
 * a ROCPROFILER_PC_SAMPLING_RECORD_INVALID_SAMPLE record is emitted instead.
 *
 * @p deliver_invalid is a runtime argument (not a template axis): it is a single branch that is
 * constant per session and perfectly predicted, so templatizing it would only double the
 * instantiation count for no measurable gain.
 */
template <typename PcSamplingRecordT, rocprofiler_pc_sampling_method_t Method>
inline void
emplace_records_in_buffer(rocprofiler::buffer::instance* buff,
                          const PcSamplingRecordT*       samples,
                          size_t                         num_samples,
                          bool                           deliver_invalid)
{
    for(size_t i = 0; i < num_samples; i++)
    {
        if(is_invalid_sample<PcSamplingRecordT, Method>(samples[i]))
        {
            if(!deliver_invalid) continue;
            auto invalid_sample = rocprofiler::common::init_public_api_struct(
                rocprofiler_pc_sampling_record_invalid_t{});
            buff->emplace(ROCPROFILER_BUFFER_CATEGORY_PC_SAMPLING,
                          ROCPROFILER_PC_SAMPLING_RECORD_INVALID_SAMPLE,
                          invalid_sample);
        }
        else
        {
            buff->emplace(ROCPROFILER_BUFFER_CATEGORY_PC_SAMPLING,
                          record_kind_for<PcSamplingRecordT>(),
                          samples[i]);
        }
    }
}

template <typename PcSamplingRecordT, rocprofiler_pc_sampling_method_t Method>
void
PCSamplingParserContext::generate_upcoming_pc_record(uint64_t                 agent_id_handle,
                                                     const PcSamplingRecordT* samples,
                                                     size_t                   num_samples,
                                                     bool                     deliver_invalid)
{
    auto buff_id = _agent_buffers.at(rocprofiler_agent_id_t{agent_id_handle});
    rocprofiler::buffer::instance* buff = rocprofiler::buffer::get_buffer(buff_id);

    if(!buff)
        throw std::runtime_error(fmt::format("Buffer with id: {} does not exists", buff_id.handle));

    emplace_records_in_buffer<PcSamplingRecordT, Method>(
        buff, samples, num_samples, deliver_invalid);
}
