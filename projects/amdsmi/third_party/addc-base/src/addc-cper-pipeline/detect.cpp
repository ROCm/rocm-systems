// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
#include "addc/pipeline/detect.hpp"

#include "addc/cper/amd_section_ids.hpp"
#include "addc/product.hpp"

namespace addc::pipeline
{

std::optional<std::string> detect_platform(uint32_t revision_major)
{
    const auto major = static_cast<uint8_t>(revision_major & 0xFFU);
    uint32_t program_rev = cper::program_gen(major);
    const uint32_t gen_rev = cper::ras_gen(major);

    // Some firmware encodes program_rev=0 while gen_rev is valid.
    // program_rev=0 is otherwise unused, so promote it to 1 (EPYC).
    if (program_rev == 0U && gen_rev != 0U)
    {
        program_rev = 1U;
    }

    for (const auto& definition : product::compiled_products())
    {
        for (const auto& route : definition.identity.descriptor_routes)
        {
            if (route.program_revision == program_rev &&
                route.generation_revision == gen_rev)
            {
                return std::string{definition.identity.canonical_name};
            }
        }
    }

    return std::nullopt;
}

std::string_view platform_display_name(std::string_view platform) noexcept
{
    if (const auto* definition = product::find_compiled_product(platform))
    {
        return definition->identity.display_name;
    }
    return "AMD EPYC";
}

} // namespace addc::pipeline
