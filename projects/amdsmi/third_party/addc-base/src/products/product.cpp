// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
#include "addc/product.hpp"

#include <array>

namespace addc::product
{
namespace
{

constexpr std::array<DescriptorRoute, 1> firerange_descriptor_routes{{{4, 1}}};
constexpr std::array<DescriptorRoute, 2> genoa_descriptor_routes{{{1, 0},
                                                                  {1, 1}}};
constexpr std::array<DescriptorRoute, 1> mi300a_descriptor_routes{{{2, 1}}};
constexpr std::array<DescriptorRoute, 1> mi300x_descriptor_routes{{{2, 2}}};
constexpr std::array<DescriptorRoute, 1> mi450_descriptor_routes{{{2, 3}}};
constexpr std::array<DescriptorRoute, 1> turin_descriptor_routes{{{1, 2}}};
constexpr std::array<DescriptorRoute, 1> venice_descriptor_routes{{{1, 3}}};

constexpr std::size_t product_count =
#ifdef ADDC_HAS_FIRERANGE
    1 +
#endif
#ifdef ADDC_HAS_GENOA
    1 +
#endif
#ifdef ADDC_HAS_MI300A
    1 +
#endif
#ifdef ADDC_HAS_MI300C
    1 +
#endif
#ifdef ADDC_HAS_MI300X
    1 +
#endif
#ifdef ADDC_HAS_MI325X
    1 +
#endif
#ifdef ADDC_HAS_MI350
    1 +
#endif
#ifdef ADDC_HAS_MI450
    1 +
#endif
#ifdef ADDC_HAS_TURIN
    1 +
#endif
#ifdef ADDC_HAS_VENICE
    1 +
#endif
    0;

constexpr std::array<ProductDefinition, product_count> products{{
#ifdef ADDC_HAS_FIRERANGE
    {{ProductId::firerange, "firerange", "AMD EPYC Embedded 2005",
      ProductFamily::epyc, {}, firerange_descriptor_routes},
     {true, true, true}},
#endif
#ifdef ADDC_HAS_GENOA
    {{ProductId::genoa, "genoa", "AMD EPYC Genoa", ProductFamily::epyc, {},
      genoa_descriptor_routes},
     {true, true, true}},
#endif
#ifdef ADDC_HAS_MI300A
    {{ProductId::mi300a, "mi300a", "AMD Instinct MI300A/C",
      ProductFamily::instinct, {}, mi300a_descriptor_routes},
     {true, false, false}},
#endif
#ifdef ADDC_HAS_MI300C
    // Preserve the established display fallback for the otherwise supported C SKU.
    {{ProductId::mi300c, "mi300c", "AMD EPYC", ProductFamily::instinct, {}, {}},
     {true, false, false}},
#endif
#ifdef ADDC_HAS_MI300X
    {{ProductId::mi300x, "mi300x", "AMD Instinct MI300X",
      ProductFamily::instinct, {}, mi300x_descriptor_routes},
     {true, false, false}},
#endif
#ifdef ADDC_HAS_MI325X
    {{ProductId::mi325x, "mi325x", "AMD Instinct MI325X",
      ProductFamily::instinct, {}, {}},
     {true, false, false}},
#endif
#ifdef ADDC_HAS_MI350
    {{ProductId::mi350, "mi350", "AMD Instinct MI350X/P",
      ProductFamily::instinct, {}, {}},
     {true, false, false}},
#endif
#ifdef ADDC_HAS_MI450
    {{ProductId::mi450, "mi450", "AMD Instinct MI450X",
      ProductFamily::instinct, {}, mi450_descriptor_routes},
     {true, true, false}},
#endif
#ifdef ADDC_HAS_TURIN
    {{ProductId::turin, "turin", "AMD EPYC Turin", ProductFamily::epyc, {},
      turin_descriptor_routes},
     {true, true, true}},
#endif
#ifdef ADDC_HAS_VENICE
    {{ProductId::venice, "venice", "AMD EPYC Venice", ProductFamily::epyc,
      {}, venice_descriptor_routes},
     {true, true, true}},
#endif
}};

} // namespace

std::span<const ProductDefinition> compiled_products() noexcept
{
    return products;
}

const ProductDefinition* find_compiled_product(std::string_view name) noexcept
{
    for (const auto& product : products)
    {
        if (product.identity.canonical_name == name)
        {
            return &product;
        }
        for (const auto alias : product.identity.aliases)
        {
            if (alias == name)
            {
                return &product;
            }
        }
    }
    return nullptr;
}

const ProductDefinition* find_compiled_product(ProductId id) noexcept
{
    for (const auto& product : products)
    {
        if (product.identity.id == id)
        {
            return &product;
        }
    }
    return nullptr;
}

} // namespace addc::product
