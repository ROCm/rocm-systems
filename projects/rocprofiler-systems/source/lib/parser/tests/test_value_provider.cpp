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
    EXPECT_EQ(fallback,
              "hip_runtime_api,marker_api,kernel_dispatch,memory_copy,scratch_memory");
    EXPECT_TRUE(domains->accepts(param, fallback));
    EXPECT_TRUE(domains->accepts(param, "hip_api"));
    EXPECT_TRUE(domains->accepts(param, "roctx"));
    EXPECT_TRUE(domains->accepts(param, "rccl_api"));
    EXPECT_FALSE(domains->accepts(param, "hsa_activity"));
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
}

TEST(parser_value_provider, gpu_metrics_use_the_runtime_names)
{
    value_provider_registry registry;
    install_builtins(registry);
    auto metric           = schema_param{};
    metric.type           = param_type::enumeration;
    metric.value_provider = "gpu_metrics";
    const auto* metrics   = registry.find("gpu_metrics");
    ASSERT_NE(metrics, nullptr);
    EXPECT_TRUE(metrics->accepts(metric, "temp"));
    EXPECT_TRUE(metrics->accepts(metric, "busy"));
    EXPECT_TRUE(metrics->accepts(metric, "mem_usage"));
    EXPECT_FALSE(metrics->accepts(metric, "temperature"));
    EXPECT_FALSE(metrics->accepts(metric, "utilization"));
    EXPECT_FALSE(metrics->accepts(metric, "memory"));

    metric.type = param_type::string;
    EXPECT_TRUE(metrics->accepts(metric, "busy, temp, power, mem_usage"));
}

TEST(parser_value_provider, hw_counters_accept_any_non_empty_name)
{
    value_provider_registry registry;
    install_builtins(registry);
    auto counter           = schema_param{};
    counter.type           = param_type::enumeration;
    counter.value_provider = "hw_counters";
    const auto* counters   = registry.find("hw_counters");
    ASSERT_NE(counters, nullptr);
    EXPECT_TRUE(counters->choices(counter).empty());
    EXPECT_TRUE(counters->accepts(counter, "SQ_WAVES"));
    EXPECT_TRUE(counters->accepts(counter, "PAPI_TOT_CYC"));
    EXPECT_FALSE(counters->accepts(counter, ""));
    EXPECT_FALSE(counters->accepts(counter, "   "));
}

TEST(parser_value_provider, provider_backed_enumeration_uses_the_catalog)
{
    value_provider_registry registry;
    install_builtins(registry);
    const auto  param   = domain_param(param_type::enumeration);
    const auto* domains = registry.find("rocm_domains_sdk");
    ASSERT_NE(domains, nullptr);
    EXPECT_TRUE(param.choices.empty());
    EXPECT_TRUE(domains->accepts(param, "scratch_memory"));
    EXPECT_FALSE(domains->accepts(param, "hsa_activity"));
    EXPECT_FALSE(domains->accepts(param, ""));
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

TEST(parser_value_provider, builtins_are_idempotent)
{
    value_provider_registry registry;
    registry.add_builtins();
    const auto* counters = registry.find("hw_counters");
    ASSERT_NE(counters, nullptr);
    registry.add_builtins();
    EXPECT_EQ(registry.find("hw_counters"), counters);
    EXPECT_NE(registry.find("parallel_runtimes"), nullptr);
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
