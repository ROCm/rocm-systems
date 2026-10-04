// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
#include "addc/pipeline/detect.hpp"

#include "addc/product.hpp"

namespace addc::pipeline
{

std::optional<std::string> detect_platform(uint32_t revision_major)
{
    uint32_t program_rev = revision_major & 0xFU;
    const uint32_t gen_rev = (revision_major >> 4U) & 0xFU;

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

    if (program_rev == 5U)
    { // NTSG AINIC (all gen_rev)
        return "ainic";
    }

    return std::nullopt;
}

std::string_view platform_display_name(std::string_view platform) noexcept
{
    if (const auto* definition = product::find_compiled_product(platform))
    {
        return definition->identity.display_name;
    }
    if (platform == "ainic")
    {
        return "AMD NTSG AINIC";
    }
    return "AMD EPYC";
}

} // namespace addc::pipeline
