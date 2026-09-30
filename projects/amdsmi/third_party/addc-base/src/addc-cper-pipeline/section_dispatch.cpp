// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
#include "addc/pipeline/section_dispatch.hpp"

#include "addc/pipeline/ainic_decoder.hpp"
#if defined(ADDC_HAS_VENICE) || defined(ADDC_HAS_TURIN) || \
    defined(ADDC_HAS_GENOA) || defined(ADDC_HAS_FIRERANGE)
#include "addc/pipeline/cdd_decoder.hpp"
#include "addc/pipeline/epyc_crashdump_decoder.hpp"
#include "addc/pipeline/ia32x64_decoder.hpp"
#include "addc/pipeline/pcie_decoder.hpp"
#include "addc/pipeline/pmic_decoder.hpp"
#endif
#if defined(ADDC_HAS_VENICE) || defined(ADDC_HAS_TURIN) || \
    defined(ADDC_HAS_GENOA) || defined(ADDC_HAS_FIRERANGE) || \
    defined(ADDC_HAS_MI450)
#include "addc/pipeline/mtl_decoder.hpp"
#endif
#if defined(ADDC_HAS_MI300A) || defined(ADDC_HAS_MI300C) || \
    defined(ADDC_HAS_MI300X) || defined(ADDC_HAS_MI325X) || \
    defined(ADDC_HAS_MI350)
#include "addc/pipeline/gpu_crashdump_decoder.hpp"
#include "addc/pipeline/gpu_runtime_decoder.hpp"
#endif
#ifdef ADDC_HAS_MI450
#include "addc/pipeline/gpu_mi450_crashdump_decoder.hpp"
#include "addc/pipeline/gpu_mi450_runtime_decoder.hpp"
#endif

#include <type_traits>
#include <variant>

namespace addc::pipeline
{

SectionDecodeResult decode_base_section(const SectionDescriptor& descriptor,
                                        const cper::SectionPayload& payload,
                                        const DecodeContext& context,
                                        std::string* error_out)
{
    return std::visit(
        [&](const auto& section) -> SectionDecodeResult {
            using T = std::decay_t<decltype(section)>;
            if constexpr (std::is_same_v<T, cper::OpaqueSection>)
            {
                return {};
            }
            else if constexpr (
                std::is_same_v<T, AinicPlatformContextSection>)
            {
                return {true,
                        decode_ainic(descriptor, section, context, error_out)};
            }
#if defined(ADDC_HAS_VENICE) || defined(ADDC_HAS_TURIN) || \
    defined(ADDC_HAS_GENOA) || defined(ADDC_HAS_FIRERANGE)
            else if constexpr (std::is_same_v<T, AmdEpycCrashdumpSection>)
            {
                return {true, decode_epyc_crashdump(descriptor, section,
                                                    context, error_out)};
            }
            else if constexpr (std::is_same_v<T, Ia32x64Section>)
            {
                return {true, decode_ia32x64(descriptor, section, context,
                                             error_out)};
            }
            else if constexpr (std::is_same_v<T, PcieSection>)
            {
                return {true, decode_pcie_section(descriptor, section, context,
                                                  error_out)};
            }
            else if constexpr (std::is_same_v<T, AmdEpycCddSection>)
            {
                return {true,
                        decode_cdd(descriptor, section, context, error_out)};
            }
            else if constexpr (std::is_same_v<T, AmdEpycPmicSection>)
            {
                return {true,
                        decode_pmic(descriptor, section, context, error_out)};
            }
#endif
#if defined(ADDC_HAS_VENICE) || defined(ADDC_HAS_TURIN) || \
    defined(ADDC_HAS_GENOA) || defined(ADDC_HAS_FIRERANGE) || \
    defined(ADDC_HAS_MI450)
            else if constexpr (std::is_same_v<T, AmdEpycMtlSection>)
            {
                return {true,
                        decode_mtl(descriptor, section, context, error_out)};
            }
#endif
#if defined(ADDC_HAS_MI300A) || defined(ADDC_HAS_MI300C) || \
    defined(ADDC_HAS_MI300X) || defined(ADDC_HAS_MI325X) || \
    defined(ADDC_HAS_MI350)
            else if constexpr (std::is_same_v<T, AmdGpuCrashdumpSection>)
            {
                return {true, decode_gpu_crashdump(descriptor, section,
                                                   context, error_out)};
            }
            else if constexpr (std::is_same_v<T, AmdGpuRuntimeSection>)
            {
                return {true, decode_gpu_runtime(descriptor, section, context,
                                                 error_out)};
            }
#endif
#ifdef ADDC_HAS_MI450
            else if constexpr (
                std::is_same_v<T, AmdGpuMi450CrashdumpSection>)
            {
                return {true, decode_gpu_mi450_crashdump(
                                  descriptor, section, context, error_out)};
            }
            else if constexpr (
                std::is_same_v<T, AmdGpuMi450RuntimeSection>)
            {
                return {true, decode_gpu_mi450_runtime(
                                  descriptor, section, context, error_out)};
            }
#endif
            else
            {
                return {};
            }
        },
        payload);
}

} // namespace addc::pipeline
