// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "parser/backend.hpp"
#include "parser/schema.hpp"

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
class named_backend final : public backend
{
public:
    explicit named_backend(std::string name)
    : m_name(std::move(name))
    {}

    [[nodiscard]] std::string_view name() const override { return m_name; }

    [[nodiscard]] std::vector<std::string> choices(const schema_param&) const override
    {
        return {};
    }

    [[nodiscard]] std::optional<std::string> default_value(
        const schema_param&) const override
    {
        return std::nullopt;
    }

private:
    std::string m_name;
};

class fixed_backend final : public backend
{
public:
    [[nodiscard]] std::string_view name() const override { return "domains"; }

    [[nodiscard]] std::vector<std::string> choices(
        const schema_param& param) const override
    {
        if(param.id == "counters")
        {
            return { "SQ_WAVES" };
        }
        return { "kernel_dispatch", "hip_api" };
    }

    [[nodiscard]] std::optional<std::string> default_value(
        const schema_param& param) const override
    {
        if(param.id == "counters")
        {
            return std::nullopt;
        }
        return std::string{ "kernel_dispatch" };
    }
};

schema_param
make_param(std::string param_id, param_type type)
{
    auto param    = schema_param{};
    param.id      = std::move(param_id);
    param.type    = type;
    param.backend = "domains";
    return param;
}

void
add_domains(backend_registry& registry)
{
    EXPECT_TRUE(registry.add(std::make_unique<fixed_backend>()));
}
}  // namespace

TEST(parser_backend, registry_rejects_duplicate_and_empty_names)
{
    backend_registry registry;
    EXPECT_TRUE(registry.add(std::make_unique<fixed_backend>()));
    EXPECT_FALSE(registry.add(std::make_unique<fixed_backend>()));
    EXPECT_FALSE(registry.add(std::make_unique<named_backend>("")));
    EXPECT_FALSE(registry.add(nullptr));
    EXPECT_EQ(registry.find("missing"), nullptr);

    const auto* found = registry.find("domains");
    ASSERT_NE(found, nullptr);
    EXPECT_EQ(found->name(), "domains");
}

TEST(parser_backend, an_empty_registry_has_no_backends)
{
    const backend_registry registry;
    EXPECT_EQ(registry.find("domains"), nullptr);
    EXPECT_EQ(registry.find("rocprofiler-sdk"), nullptr);
    EXPECT_EQ(registry.find("papi"), nullptr);
}

TEST(parser_backend, choices_and_the_default_follow_the_parameter)
{
    backend_registry registry;
    add_domains(registry);
    const auto* domains = registry.find("domains");
    ASSERT_NE(domains, nullptr);

    const auto domain_row  = make_param("domains", param_type::enumeration);
    const auto counter_row = make_param("counters", param_type::enumeration);
    EXPECT_EQ(domains->choices(domain_row),
              (std::vector<std::string>{ "kernel_dispatch", "hip_api" }));
    EXPECT_EQ(domains->default_value(domain_row).value_or(""), "kernel_dispatch");
    EXPECT_EQ(domains->choices(counter_row), (std::vector<std::string>{ "SQ_WAVES" }));
    EXPECT_FALSE(domains->default_value(counter_row).has_value());
    EXPECT_TRUE(domains->accepts(domain_row, "kernel_dispatch"));
    EXPECT_FALSE(domains->accepts(domain_row, "SQ_WAVES"));
    EXPECT_TRUE(domains->accepts(counter_row, "SQ_WAVES"));
    EXPECT_FALSE(domains->accepts(counter_row, "kernel_dispatch"));
    EXPECT_FALSE(domains->accepts(domain_row, "HIP_API"));
    EXPECT_FALSE(domains->accepts(domain_row, ""));
}

TEST(parser_backend, a_list_accepts_each_delimiter)
{
    backend_registry registry;
    add_domains(registry);
    const auto* domains = registry.find("domains");
    ASSERT_NE(domains, nullptr);
    const auto param = make_param("domains", param_type::enumeration_list);

    EXPECT_TRUE(domains->accepts(param, "kernel_dispatch hip_api"));
    EXPECT_TRUE(domains->accepts(param, "kernel_dispatch,hip_api"));
    EXPECT_TRUE(domains->accepts(param, "kernel_dispatch;hip_api"));
    EXPECT_TRUE(domains->accepts(param, "kernel_dispatch:hip_api"));
    EXPECT_TRUE(domains->accepts(param, "kernel_dispatch\thip_api"));
    EXPECT_TRUE(domains->accepts(param, "kernel_dispatch\nhip_api"));
    EXPECT_TRUE(domains->accepts(param, "kernel_dispatch,,hip_api"));
    EXPECT_FALSE(domains->accepts(param, "kernel_dispatch, not_a_domain"));
    EXPECT_TRUE(domains->accepts(param, ""));
}

TEST(parser_backend, a_string_with_a_delimiter_is_a_list)
{
    backend_registry registry;
    add_domains(registry);
    const auto* domains = registry.find("domains");
    ASSERT_NE(domains, nullptr);
    const auto param = make_param("domains", param_type::string);

    EXPECT_TRUE(domains->accepts(param, "kernel_dispatch"));
    EXPECT_TRUE(domains->accepts(param, "kernel_dispatch,hip_api"));
    EXPECT_FALSE(domains->accepts(param, "kernel_dispatch,not_a_domain"));
    EXPECT_TRUE(domains->accepts(param, ""));
}

TEST(parser_backend, allow_custom_accepts_unlisted_text)
{
    backend_registry registry;
    add_domains(registry);
    const auto* domains = registry.find("domains");
    ASSERT_NE(domains, nullptr);

    auto enumeration         = make_param("domains", param_type::enumeration);
    enumeration.allow_custom = true;
    EXPECT_TRUE(domains->accepts(enumeration, "custom_domain"));
    EXPECT_FALSE(domains->accepts(enumeration, ""));

    auto listed         = make_param("domains", param_type::enumeration_list);
    listed.allow_custom = true;
    EXPECT_TRUE(domains->accepts(listed, "one;two"));
    EXPECT_TRUE(domains->accepts(listed, ""));
    EXPECT_FALSE(domains->accepts(listed, " ,;"));
}
