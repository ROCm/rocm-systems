// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "parser/schema.hpp"
#include "parser/value_provider.hpp"

#include <memory>
#include <optional>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace rocprofsys::parser
{
namespace
{
class catalog_provider final : public value_provider_interface
{
public:
    catalog_provider(std::string name, std::vector<std::string> choices,
                     std::optional<std::string> fallback)
    : m_name(std::move(name))
    , m_choices(std::move(choices))
    , m_default(std::move(fallback))
    {}

    [[nodiscard]] std::string_view name() const override { return m_name; }

    [[nodiscard]] std::vector<std::string> choices(const schema_param&) const override
    {
        return m_choices;
    }

    [[nodiscard]] std::optional<std::string> default_value(
        const schema_param&) const override
    {
        return m_default;
    }

private:
    std::string                m_name;
    std::vector<std::string>   m_choices;
    std::optional<std::string> m_default;
};

std::unique_ptr<value_provider_interface>
catalog(std::string name, std::vector<std::string> choices,
        std::optional<std::string> fallback = std::nullopt)
{
    return std::make_unique<catalog_provider>(std::move(name), std::move(choices),
                                              std::move(fallback));
}

std::vector<std::string>
rocm_domains()
{
    return { "hip_runtime_api", "hip_compiler_api", "hsa_api",        "hsa_activity",
             "marker_api",      "memory_copy",      "kernel_dispatch" };
}
}  // namespace

void
value_provider_registry::add_builtins()
{
    static_cast<void>(
        add(catalog("rocm_domains_sdk", rocm_domains(),
                    std::string{ "hip_runtime_api,kernel_dispatch,memory_copy" })));
    static_cast<void>(add(catalog("parallel_runtimes", { "mpi", "openmp", "pthread" })));
    static_cast<void>(
        add(catalog("gpu_metrics", { "temperature", "power", "utilization", "memory" })));
    static_cast<void>(add(catalog("hw_counters", { "PAPI_TOT_CYC", "PAPI_TOT_INS" })));
}

}  // namespace rocprofsys::parser
