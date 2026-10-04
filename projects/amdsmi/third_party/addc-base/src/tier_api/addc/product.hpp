// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
#pragma once

#include <cstdint>
#include <span>
#include <string_view>

namespace addc::product
{

enum class ProductId : uint8_t
{
    firerange,
    genoa,
    mi300a,
    mi300c,
    mi300x,
    mi325x,
    mi350,
    mi450,
    turin,
    venice,
};

enum class ProductFamily : uint8_t
{
    epyc,
    instinct,
};

struct DescriptorRoute
{
    uint8_t program_revision;
    uint8_t generation_revision;
};

struct ProductIdentity
{
    ProductId id;
    std::string_view canonical_name;
    std::string_view display_name;
    ProductFamily family;
    std::span<const std::string_view> aliases;
    std::span<const DescriptorRoute> descriptor_routes;
};

struct BaseCapabilities
{
    bool mca;
    bool debug_log;
    bool watchdog;
};

struct ProductDefinition
{
    ProductIdentity identity;
    BaseCapabilities capabilities;
};

std::span<const ProductDefinition> compiled_products() noexcept;
const ProductDefinition* find_compiled_product(std::string_view name) noexcept;
const ProductDefinition* find_compiled_product(ProductId id) noexcept;

} // namespace addc::product
