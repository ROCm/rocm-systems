// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "common/env_vars.hpp"
#include "options/avail_traces.hpp"
#include "options/context.hpp"
#include "trace/catalog.hpp"

#include <gtest/gtest.h>

#include <algorithm>
#include <sstream>
#include <string>
#include <string_view>
#include <vector>

namespace
{
namespace env_vars = rocprofsys::env_vars;
using rocprofsys::rocsys::options::domain_query;
using rocprofsys::rocsys::options::write_avail_traces;
using rocprofsys::rocsys::trace::trace_catalog;
using rocprofsys::rocsys::trace::trace_selection;

std::vector<std::string> g_choices;
std::vector<std::string> g_supported;
std::vector<std::string> g_compiled;
int                      g_choice_calls = 0;

std::vector<std::string>
choices()
{
    ++g_choice_calls;
    return g_choices;
}

std::vector<std::string>
supported()
{
    return g_supported;
}

bool
compiled(std::string_view name)
{
    return std::ranges::find(g_compiled, std::string{ name }) != g_compiled.end();
}

domain_query
query()
{
    return { .choices = choices, .supported = supported, .runtime_compiled = compiled };
}

void
reset(std::vector<std::string> domain_choices, std::vector<std::string> domain_supported,
      std::vector<std::string> compiled_runtimes)
{
    g_choices      = std::move(domain_choices);
    g_supported    = std::move(domain_supported);
    g_compiled     = std::move(compiled_runtimes);
    g_choice_calls = 0;
}

[[nodiscard]] bool
enabled(const trace_selection& selection, std::string_view key)
{
    for(const auto& runtime : selection.runtimes)
    {
        if(runtime.env_key == key)
        {
            return runtime.enabled;
        }
    }
    return false;
}

[[nodiscard]] const rocprofsys::rocsys::trace::catalog_entry*
find_entry(const std::vector<rocprofsys::rocsys::trace::catalog_entry>& rows,
           std::string_view                                             name)
{
    for(const auto& row : rows)
    {
        if(row.name == name || row.name.starts_with(std::string{ name } + " ("))
        {
            return &row;
        }
    }
    return nullptr;
}

}  // namespace

TEST(catalog_test, construction_does_not_query_the_sdk)
{
    reset({ "hip_api" }, { "hip_api" }, { "mpi" });
    const trace_catalog catalog{ query() };
    EXPECT_EQ(g_choice_calls, 0);
    EXPECT_EQ(catalog.sdk_domains().size(), 1U);
    EXPECT_EQ(catalog.sdk_domains().size(), 1U);
    EXPECT_EQ(g_choice_calls, 1);
}

TEST(catalog_test, friendly_name_expands_to_its_domain)
{
    reset({ "hip_api", "marker_api", "memory_copy", "memory_allocation" },
          { "hip_api", "marker_api", "memory_copy", "memory_allocation" }, {});
    const trace_catalog catalog{ query() };

    const auto hip = catalog.expand(std::vector<std::string>{ "hip" });
    ASSERT_TRUE(hip);
    EXPECT_EQ(hip.value().rocm_domains, (std::vector<std::string>{ "hip_api" }));
    EXPECT_FALSE(enabled(hip.value(), env_vars::USE_MPIP));

    const auto roctx = catalog.expand(std::vector<std::string>{ "roctx" });
    ASSERT_TRUE(roctx);
    EXPECT_EQ(roctx.value().rocm_domains, (std::vector<std::string>{ "marker_api" }));

    const auto memory = catalog.expand(std::vector<std::string>{ "memory" });
    ASSERT_TRUE(memory);
    EXPECT_EQ(memory.value().rocm_domains,
              (std::vector<std::string>{ "memory_allocation", "memory_copy" }));
}

TEST(catalog_test, repeated_domains_collapse_and_runtimes_do_not_add_domains)
{
    reset({ "hip_api", "kernel_dispatch" }, { "hip_api", "kernel_dispatch" }, { "mpi" });
    const trace_catalog catalog{ query() };
    const auto          selected = catalog.expand(
        std::vector<std::string>{ "hip", "hip_api", "kernel_dispatch", "mpi" });
    ASSERT_TRUE(selected);
    EXPECT_EQ(selected.value().rocm_domains,
              (std::vector<std::string>{ "hip_api", "kernel_dispatch" }));
    EXPECT_TRUE(enabled(selected.value(), env_vars::USE_MPIP));
    EXPECT_FALSE(enabled(selected.value(), env_vars::USE_OMPT));
    EXPECT_FALSE(enabled(selected.value(), env_vars::USE_KOKKOSP));
}

TEST(catalog_test, ompt_and_rccl_api_select_runtime_rows)
{
    reset({ "ompt", "rccl_api" }, { "ompt", "rccl_api" }, { "openmp", "rccl" });
    const trace_catalog catalog{ query() };

    const auto ompt = catalog.expand(std::vector<std::string>{ "ompt" });
    ASSERT_TRUE(ompt);
    EXPECT_TRUE(ompt.value().rocm_domains.empty());
    EXPECT_TRUE(enabled(ompt.value(), env_vars::USE_OMPT));

    const auto rccl = catalog.expand(std::vector<std::string>{ "rccl_api" });
    ASSERT_TRUE(rccl);
    EXPECT_TRUE(rccl.value().rocm_domains.empty());
    EXPECT_TRUE(enabled(rccl.value(), env_vars::USE_RCCLP));
}

TEST(catalog_test, unavailable_names_explain_why)
{
    reset({ "hip_api", "ompt" }, { "ompt" }, { "mpi" });
    const trace_catalog catalog{ query() };

    const auto headers = catalog.expand(std::vector<std::string>{ "hip" });
    ASSERT_FALSE(headers);
    EXPECT_NE(headers.error().hint.find("headers"), std::string::npos);

    reset({}, {}, { "mpi" });
    const trace_catalog missing{ query() };
    const auto          sdk = missing.expand(std::vector<std::string>{ "kernel" });
    ASSERT_FALSE(sdk);
    EXPECT_NE(sdk.error().hint.find("loaded rocprofiler-sdk"), std::string::npos);

    reset({ "ompt" }, { "ompt" }, {});
    const trace_catalog unbuilt{ query() };
    const auto          openmp = unbuilt.expand(std::vector<std::string>{ "openmp" });
    ASSERT_FALSE(openmp);
    EXPECT_NE(openmp.error().hint.find("not compiled"), std::string::npos);
}

TEST(catalog_test, unknown_names_suggest_a_canonical_name_and_kokkos_is_unknown)
{
    reset({ "hip_api" }, { "hip_api" }, { "mpi", "openmp" });
    const trace_catalog catalog{ query() };

    const auto typo = catalog.expand(std::vector<std::string>{ "hpp" });
    ASSERT_FALSE(typo);
    EXPECT_NE(typo.error().error.find("unknown trace name 'hpp'"), std::string::npos);
    EXPECT_NE(typo.error().hint.find("did you mean 'hip'"), std::string::npos);

    const auto kokkos = catalog.expand(std::vector<std::string>{ "kokkos" });
    ASSERT_FALSE(kokkos);
    EXPECT_NE(kokkos.error().error.find("unknown trace name 'kokkos'"),
              std::string::npos);

    EXPECT_EQ(catalog.suggest("kernl"), "kernel");
}

TEST(catalog_test, entries_skip_kokkos_and_overridden_domain_names)
{
    reset({ "hip_api", "ompt", "rccl_api", "kernel_dispatch" }, { "kernel_dispatch" },
          { "mpi", "ucx" });
    const trace_catalog catalog{ query() };
    const auto          rows = catalog.entries();

    const auto* hip = find_entry(rows, "hip");
    ASSERT_NE(hip, nullptr);
    EXPECT_FALSE(hip->available);
    EXPECT_EQ(hip->expands_to, "hip_api");

    const auto* openmp = find_entry(rows, "openmp");
    ASSERT_NE(openmp, nullptr);
    EXPECT_NE(openmp->name.find("ompt"), std::string::npos);
    EXPECT_FALSE(openmp->available);

    const auto* mpi = find_entry(rows, "mpi");
    ASSERT_NE(mpi, nullptr);
    EXPECT_TRUE(mpi->available);
    EXPECT_EQ(mpi->expands_to, env_vars::USE_MPIP);

    EXPECT_EQ(find_entry(rows, "kokkos"), nullptr);
    EXPECT_NE(find_entry(rows, "kernel_dispatch"), nullptr);
    for(const auto& row : rows)
    {
        EXPECT_NE(row.name, "ompt");
        EXPECT_NE(row.name, "rccl_api");
    }

    std::ostringstream out;
    write_avail_traces(out, catalog);
    const auto text = out.str();
    EXPECT_NE(text.find("NAME"), std::string::npos);
    EXPECT_NE(text.find("WHAT IT TRACES"), std::string::npos);
    EXPECT_NE(text.find("EXPANDS TO"), std::string::npos);
    EXPECT_NE(text.find("AVAILABLE"), std::string::npos);
    EXPECT_NE(text.find("ROCPROFSYS_USE_UNIFIED_MEMORY_PROFILING"), std::string::npos);
    EXPECT_NE(text.find("SDK domains for this install:"), std::string::npos);
    EXPECT_EQ(text.find("kokkos"), std::string::npos);
}
