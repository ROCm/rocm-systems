// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "parser/value_provider.hpp"
#include "parser/schema.hpp"

#include "common/string_utility.hpp"

#if defined(ROCPROFSYS_PARSER_HAS_SDK_BACKEND)
#    include "backends/rocprofiler_sdk/backend.hpp"
#    include "backends/rocprofiler_sdk/wrapper.hpp"
#    include "core/sdk/tracing-config.hpp"
#endif

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <mutex>
#include <optional>
#include <shared_mutex>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace rocprofsys::parser
{
namespace
{
// Same delimiter set tracing_config uses for ROCPROFSYS_ROCM_DOMAINS.
constexpr std::string_view k_list_delimiters = " ,;:\t\n";

bool
is_listed(const std::vector<std::string>& choices, std::string_view value)
{
    return std::ranges::find(choices, value) != choices.end();
}

[[nodiscard]] bool
contains_delimiter(std::string_view value)
{
    return value.find_first_of(k_list_delimiters) != std::string_view::npos;
}

std::vector<std::string>
split_provider_list(std::string_view value)
{
    std::vector<std::string> parts;
    std::size_t              begin = 0;
    while(begin < value.size())
    {
        begin = value.find_first_not_of(k_list_delimiters, begin);
        if(begin == std::string_view::npos)
        {
            break;
        }
        const auto end = value.find_first_of(k_list_delimiters, begin);
        const auto count =
            end == std::string_view::npos ? std::string_view::npos : end - begin;
        const auto token = value.substr(begin, count);
        if(!token.empty())
        {
            parts.emplace_back(token);
        }
        if(end == std::string_view::npos)
        {
            break;
        }
        begin = end;
    }
    return parts;
}

bool
every_element_present(std::string_view value)
{
    if(value.empty())
    {
        return true;
    }
    return !split_provider_list(value).empty();
}

bool
elements_are_listed(const std::vector<std::string>& choices, std::string_view value)
{
    const auto parts = split_provider_list(value);
    if(parts.empty())
    {
        return is_listed(choices, value);
    }
    return std::ranges::all_of(parts, [&choices](const std::string& element) {
        return is_listed(choices, element);
    });
}
}  // namespace

bool
value_provider_interface::accepts(const schema_param& param, std::string_view value) const
{
    const bool is_list = param.type == param_type::enumeration_list ||
                         (param.type == param_type::string && contains_delimiter(value));
    if(param.allow_custom)
    {
        if(is_list)
        {
            return every_element_present(value);
        }
        return param.type == param_type::string || !value.empty();
    }
    if(value.empty())
    {
        return param.type != param_type::enumeration;
    }
    const auto available = choices(param);
    return is_list ? elements_are_listed(available, value) : is_listed(available, value);
}

bool
value_provider_registry::add(std::unique_ptr<value_provider_interface> provider)
{
    if(!provider || provider->name().empty())
    {
        return false;
    }
    const auto                                name = std::string{ provider->name() };
    const std::unique_lock<std::shared_mutex> lock{ m_mutex };
    if(m_providers.contains(name))
    {
        return false;
    }
    m_providers.emplace(name, std::move(provider));
    return true;
}

const value_provider_interface*
value_provider_registry::find(std::string_view name) const
{
    const std::shared_lock<std::shared_mutex> lock{ m_mutex };
    const auto                                found = m_providers.find(name);
    return found == m_providers.end() ? nullptr : found->second.get();
}

namespace
{
/// Accepts any non-empty value. The choice list arrives with a later backend.
class stub_provider final : public value_provider_interface
{
public:
    explicit stub_provider(std::string name)
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

    [[nodiscard]] bool accepts(const schema_param& param,
                               std::string_view    value) const override
    {
        if(utility::string::trim(value).empty())
        {
            return param.type != param_type::enumeration;
        }
        return true;
    }

private:
    std::string m_name;
};

#if defined(ROCPROFSYS_PARSER_HAS_SDK_BACKEND)
/// Satisfies tracing_config's externals concept. Domain queries do not read it.
struct domain_externals
{
    struct process_state
    {
        enum class state : std::uint8_t
        {
            finalized,
        };

        // Names required by policies::tracing_config::externals.
        using State = state;  // NOLINT(readability-identifier-naming)
        static constexpr State Finalized =
            State::finalized;  // NOLINT(readability-identifier-naming)

        static void set(State) {}
    };

    using ProcessState = process_state;  // NOLINT(readability-identifier-naming)

    static bool get_use_rcclp() { return false; }

    static bool get_use_ompt() { return false; }

    static bool get_use_unified_memory_profiling() { return false; }

    static std::string get_rocm_domains() { return {}; }

    static std::optional<std::string> get_setting_value(std::string_view)
    {
        return std::nullopt;
    }
};

using sdk_backend =
    rocprofsys::backends::rocprofiler_sdk::backend<rocprofsys::rocprofiler_sdk::wrapper>;
using domain_config =
    rocprofsys::rocprofiler_sdk::tracing_config<sdk_backend, domain_externals>;

class sdk_domain_provider final : public value_provider_interface
{
public:
    [[nodiscard]] std::string_view name() const override { return k_name; }

    [[nodiscard]] std::vector<std::string> choices(const schema_param&) const override
    {
        return domain_config::get_domain_choices();
    }

    [[nodiscard]] std::optional<std::string> default_value(
        const schema_param&) const override
    {
        return domain_config::get_domain_defaults();
    }

private:
    static constexpr std::string_view k_name{ "rocm_domains_sdk" };
};
#endif

void
register_stub(value_provider_registry& registry, std::string name)
{
    static_cast<void>(registry.add(std::make_unique<stub_provider>(std::move(name))));
}
}  // namespace

void
value_provider_registry::add_builtins()
{
#if defined(ROCPROFSYS_PARSER_HAS_SDK_BACKEND)
    static_cast<void>(add(std::make_unique<sdk_domain_provider>()));
#endif
    register_stub(*this, "gpu_metrics");
    register_stub(*this, "parallel_runtimes");
    register_stub(*this, "hw_counters");
}

}  // namespace rocprofsys::parser
