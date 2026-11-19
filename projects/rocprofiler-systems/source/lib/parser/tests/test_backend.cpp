// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "parser/backend.hpp"
#include "parser/schema.hpp"
#include "parser/value_text.hpp"

#include <gtest/gtest.h>

#include <memory>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

using namespace rocprofsys::parser;

namespace
{
class named_backend final : public backend_interface
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

private:
    std::string m_name;
};

class fixed_backend final : public backend_interface
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

TEST(parser_backend, choices_follow_the_parameter)
{
    backend_registry registry;
    add_domains(registry);
    const auto* domains = registry.find("domains");
    ASSERT_NE(domains, nullptr);

    const auto domain_row  = make_param("domains", param_type::enumeration);
    const auto counter_row = make_param("counters", param_type::enumeration);
    EXPECT_EQ(domains->choices(domain_row),
              (std::vector<std::string>{ "kernel_dispatch", "hip_api" }));
    EXPECT_EQ(domains->choices(counter_row), (std::vector<std::string>{ "SQ_WAVES" }));
    EXPECT_TRUE(domains->accepts(domain_row, "kernel_dispatch"));
    EXPECT_FALSE(domains->accepts(domain_row, "SQ_WAVES"));
    EXPECT_TRUE(domains->accepts(counter_row, "SQ_WAVES"));
    EXPECT_FALSE(domains->accepts(counter_row, "kernel_dispatch"));
    EXPECT_FALSE(domains->accepts(domain_row, "HIP_API"));
    EXPECT_FALSE(domains->accepts(domain_row, ""));
}

TEST(parser_backend, a_list_follows_the_list_rules_of_every_source)
{
    backend_registry registry;
    add_domains(registry);
    const auto* domains = registry.find("domains");
    ASSERT_NE(domains, nullptr);
    const auto param = make_param("domains", param_type::enumeration_list);

    EXPECT_TRUE(domains->accepts(param, "kernel_dispatch,hip_api"));
    EXPECT_TRUE(domains->accepts(param, " kernel_dispatch , hip_api "));
    EXPECT_TRUE(domains->accepts(param, "kernel_dispatch,"));
    EXPECT_TRUE(domains->accepts(param, ""));
    EXPECT_FALSE(domains->accepts(param, "kernel_dispatch hip_api"));
    EXPECT_FALSE(domains->accepts(param, "kernel_dispatch;hip_api"));
    EXPECT_FALSE(domains->accepts(param, "kernel_dispatch,,hip_api"));
    EXPECT_FALSE(domains->accepts(param, ","));
    EXPECT_FALSE(domains->accepts(param, "kernel_dispatch, not_a_domain"));
}

TEST(parser_backend, adds_its_names_to_the_rows_own_choices)
{
    backend_registry registry;
    add_domains(registry);
    const auto* domains = registry.find("domains");
    ASSERT_NE(domains, nullptr);
    auto param    = make_param("domains", param_type::enumeration_list);
    param.choices = { "all" };

    EXPECT_TRUE(domains->accepts(param, "all"));
    EXPECT_TRUE(domains->accepts(param, "all,hip_api"));
    EXPECT_FALSE(domains->accepts(param, "all,bogus"));
}

TEST(parser_backend, a_backend_row_gets_the_verdict_of_a_row_that_lists_the_names)
{
    backend_registry registry;
    add_domains(registry);
    const auto* domains = registry.find("domains");
    ASSERT_NE(domains, nullptr);
    const auto from_backend = make_param("domains", param_type::enumeration_list);
    auto       listed       = from_backend;
    listed.backend.reset();
    listed.choices = { "kernel_dispatch", "hip_api" };

    for(const auto* text : { "hip_api", " hip_api , kernel_dispatch ", "hip_api,", ",",
                             "hip_api,,kernel_dispatch", "hip_api bogus", "bogus", "" })
    {
        SCOPED_TRACE(text);
        const bool stored_and_accepted = parse_value(from_backend, text).has_value() &&
                                         domains->accepts(from_backend, text);
        EXPECT_EQ(stored_and_accepted, accepts_text(listed, text));
    }
}

TEST(parser_backend, accepts_throws_for_a_row_that_names_another_backend_or_none)
{
    backend_registry registry;
    add_domains(registry);
    const auto* domains = registry.find("domains");
    ASSERT_NE(domains, nullptr);

    auto other    = make_param("domains", param_type::enumeration);
    other.backend = "amd-smi";
    auto none     = make_param("domains", param_type::enumeration);
    none.backend.reset();
    EXPECT_THROW(static_cast<void>(domains->accepts(other, "kernel_dispatch")),
                 std::logic_error);
    EXPECT_THROW(static_cast<void>(domains->accepts(none, "kernel_dispatch")),
                 std::logic_error);
}
