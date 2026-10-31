// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "parser/schema.hpp"
#include "parser/value_provider.hpp"

#include <gtest/gtest.h>

#include <memory>
#include <optional>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

using namespace rocprofsys::parser;

namespace
{
class fixed_provider final : public value_provider_interface
{
public:
    explicit fixed_provider(std::string name)
    : m_name(std::move(name))
    {}

    [[nodiscard]] std::string_view name() const override { return m_name; }

    [[nodiscard]] std::vector<std::string> choices(const schema_param&) const override
    {
        return { "alpha" };
    }

    [[nodiscard]] std::optional<std::string> default_value(
        const schema_param&) const override
    {
        return std::string{ "alpha" };
    }

private:
    std::string m_name;
};

schema_param
domain_param(param_type type = param_type::string)
{
    auto param           = schema_param{};
    param.id             = "domains";
    param.type           = type;
    param.value_provider = "rocm_domains_sdk";
    return param;
}

void
install_builtins(value_provider_registry& registry)
{
    registry.add_builtins();
}
}  // namespace

TEST(parser_value_provider, registry_rejects_duplicate_and_empty_names)
{
    value_provider_registry registry;
    EXPECT_TRUE(registry.add(std::make_unique<fixed_provider>("alpha")));
    EXPECT_FALSE(registry.add(std::make_unique<fixed_provider>("alpha")));
    EXPECT_FALSE(registry.add(std::make_unique<fixed_provider>("")));
    EXPECT_FALSE(registry.add(nullptr));

    const auto* provider = registry.find("alpha");
    ASSERT_NE(provider, nullptr);
    EXPECT_EQ(provider->name(), "alpha");
    EXPECT_EQ(registry.find("missing"), nullptr);
}

TEST(parser_value_provider, builtins_accept_known_names_and_lists)
{
    value_provider_registry registry;
    install_builtins(registry);
    const auto* domains = registry.find("rocm_domains_sdk");
    ASSERT_NE(domains, nullptr);
    const auto param = domain_param();

    const auto fallback = domains->default_value(param).value_or("");
    EXPECT_EQ(fallback, "hip_runtime_api,kernel_dispatch,memory_copy");
    EXPECT_TRUE(domains->accepts(param, fallback));
    EXPECT_TRUE(domains->accepts(param, "hsa_api"));
    EXPECT_FALSE(domains->accepts(param, "not_a_domain"));
    EXPECT_FALSE(domains->accepts(param, "hsa_api,not_a_domain"));

    auto runtimes           = schema_param{};
    runtimes.type           = param_type::enumeration_list;
    runtimes.value_provider = "parallel_runtimes";
    runtimes.env            = env_spec{ .name = "ROCPROFSYS_USE", .separator = ";" };
    const auto* runtime     = registry.find("parallel_runtimes");
    ASSERT_NE(runtime, nullptr);
    EXPECT_TRUE(runtime->accepts(runtimes, "mpi;openmp"));
    EXPECT_TRUE(runtime->accepts(runtimes, ""));
    EXPECT_FALSE(runtime->accepts(runtimes, "mpi;;openmp"));
    EXPECT_NE(registry.find("gpu_metrics"), nullptr);
    EXPECT_NE(registry.find("hw_counters"), nullptr);
}

TEST(parser_value_provider, allow_custom_accepts_unlisted_text)
{
    value_provider_registry registry;
    install_builtins(registry);
    auto param          = domain_param(param_type::enumeration);
    param.allow_custom  = true;
    const auto* domains = registry.find("rocm_domains_sdk");
    ASSERT_NE(domains, nullptr);
    EXPECT_TRUE(domains->accepts(param, "custom_domain"));
    EXPECT_FALSE(domains->accepts(param, ""));
}

TEST(parser_value_provider, missing_provider_is_not_found)
{
    const value_provider_registry registry;
    EXPECT_EQ(registry.find("rocm_domains_sdk"), nullptr);
}

TEST(parser_value_provider, instance_builtins_are_idempotent)
{
    auto& registry = value_provider_registry::instance();
    registry.add_builtins();
    registry.add_builtins();
    EXPECT_EQ(&value_provider_registry::instance(), &registry);
    EXPECT_NE(registry.find("hw_counters"), nullptr);
}

TEST(parser_value_provider, schema_records_the_provider_name)
{
    schema catalog{};
    ASSERT_TRUE(catalog.add_param(domain_param()));

    const auto* param = catalog.find_param("domains");
    ASSERT_NE(param, nullptr);
    EXPECT_EQ(param->value_provider.value_or(""), "rocm_domains_sdk");
}

TEST(parser_value_provider, schema_rejects_a_provider_on_a_numeric_parameter)
{
    schema catalog{};
    EXPECT_FALSE(catalog.add_param(domain_param(param_type::integer)));
    EXPECT_EQ(catalog.find_param("domains"), nullptr);

    auto empty           = domain_param();
    empty.value_provider = "";
    EXPECT_FALSE(catalog.add_param(std::move(empty)));
}

TEST(parser_value_provider, schema_allows_an_enum_whose_choices_come_from_a_provider)
{
    schema catalog{};
    auto   param         = schema_param{};
    param.id             = "counter";
    param.type           = param_type::enumeration;
    param.value_provider = "hw_counters";
    ASSERT_TRUE(catalog.add_param(std::move(param)));

    const auto* stored = catalog.find_param("counter");
    ASSERT_NE(stored, nullptr);
    EXPECT_EQ(stored->value_provider.value_or(""), "hw_counters");
    EXPECT_TRUE(stored->choices.empty());
}
