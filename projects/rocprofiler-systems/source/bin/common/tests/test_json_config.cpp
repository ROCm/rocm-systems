// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "common/env_vars.hpp"
#include "common/json_config.hpp"

#include <gtest/gtest.h>

#include <fstream>
#include <string>

using namespace rocprofsys::json_config;
namespace env_vars = rocprofsys::env_vars;

class json_config_test : public ::testing::Test
{};

TEST_F(json_config_test, resolves_tracing_section)
{
    auto j = nlohmann::json::parse(R"({
        "tracing": {
            "enabled": true,
            "buffer_size_kb": {"value": 2048},
            "fill_policy": {"value": "ring_buffer"}
        }
    })");

    auto result = resolve_config(j);

    EXPECT_EQ(result.at(env_vars::TRACE), "true");
    EXPECT_EQ(result.at(env_vars::PERFETTO_BUFFER_SIZE_KB), "2048");
    EXPECT_EQ(result.at(env_vars::PERFETTO_FILL_POLICY), "ring_buffer");
}

TEST_F(json_config_test, resolves_profiling_section)
{
    auto j = nlohmann::json::parse(R"({
        "profiling": {
            "enabled": true,
            "flat_profile": {"enabled": true}
        }
    })");

    auto result = resolve_config(j);

    EXPECT_EQ(result.at(env_vars::PROFILE), "true");
    EXPECT_EQ(result.at(env_vars::FLAT_PROFILE), "true");
}

TEST_F(json_config_test, resolves_sampling_section)
{
    auto j = nlohmann::json::parse(R"({
        "sampling": {
            "enabled": true,
            "timer": {"value": "realtime"},
            "frequency_hz": {"value": 200},
            "cpus": {"value": "0-3"},
            "delay_sec": {"value": 1.5}
        }
    })");

    auto result = resolve_config(j);

    EXPECT_EQ(result.at(env_vars::USE_SAMPLING), "true");
    EXPECT_EQ(result.at(env_vars::SAMPLING_TIMER), "realtime");
    EXPECT_EQ(result.at(env_vars::SAMPLING_FREQ), "200");
    EXPECT_EQ(result.at(env_vars::SAMPLING_CPUS), "0-3");
    EXPECT_EQ(result.at(env_vars::SAMPLING_DELAY), std::to_string(1.5));
}

// Test new schema format - domains.gpu section
TEST_F(json_config_test, resolves_gpu_domain)
{
    auto j = nlohmann::json::parse(R"({
        "domains": {
            "gpu": {
                "enabled": true,
                "metrics": {
                    "temp": {"enabled": true},
                    "power": {"enabled": true},
                    "busy": {"enabled": false}
                },
                "sampling_rate_hz": {"value": 10}
            }
        }
    })");

    auto result = resolve_config(j);

    EXPECT_EQ(result.at(env_vars::USE_AMD_SMI), "true");
    EXPECT_EQ(result.at(env_vars::USE_PROCESS_SAMPLING), "true");
    // Order might vary, but should contain temp and power
    auto metrics = result.at(env_vars::AMD_SMI_METRICS);
    EXPECT_NE(metrics.find("temp"), std::string::npos);
    EXPECT_NE(metrics.find("power"), std::string::npos);
    EXPECT_EQ(metrics.find("busy"), std::string::npos);  // busy is disabled
    EXPECT_EQ(result.at(env_vars::AMD_SMI_FREQ), "10");
}

// Test new schema format - domains.rocm section
TEST_F(json_config_test, resolves_rocm_domain)
{
    auto j = nlohmann::json::parse(R"({
        "domains": {
            "rocm": {
                "api_domains": {
                    "hip_runtime_api": {"enabled": true},
                    "kernel_dispatch": {"enabled": true},
                    "memory_copy": {"enabled": false}
                }
            }
        }
    })");

    auto result = resolve_config(j);

    auto domains = result.at(env_vars::ROCM_DOMAINS);
    EXPECT_NE(domains.find("hip_runtime_api"), std::string::npos);
    EXPECT_NE(domains.find("kernel_dispatch"), std::string::npos);
    EXPECT_EQ(domains.find("memory_copy"), std::string::npos);
}

// Test new schema format - domains.parallel section
TEST_F(json_config_test, resolves_parallel_domain)
{
    auto j = nlohmann::json::parse(R"({
        "domains": {
            "parallel": {
                "runtimes": {
                    "mpi": {"enabled": true},
                    "openmp": {"enabled": true},
                    "kokkos": {"enabled": false}
                }
            }
        }
    })");

    auto result = resolve_config(j);

    EXPECT_EQ(result.at(env_vars::USE_MPIP), "true");
    EXPECT_EQ(result.at(env_vars::USE_OMPT), "true");
    EXPECT_EQ(result.count(env_vars::USE_KOKKOSP), 0u);
}

// Test new schema format - output section
TEST_F(json_config_test, resolves_output_section)
{
    auto j = nlohmann::json::parse(R"({
        "output": {
            "path": {"value": "/tmp/my-traces"},
            "time_output": {"enabled": false},
            "file_output": {"enabled": true}
        }
    })");

    auto result = resolve_config(j);

    EXPECT_EQ(result.at(env_vars::OUTPUT_PATH), "/tmp/my-traces");
    // time_output and file_output are resolved when the enabled field is present
    EXPECT_EQ(result.count(env_vars::TIME_OUTPUT), 1u);
    EXPECT_EQ(result.count(env_vars::FILE_OUTPUT), 1u);
}

// Test new schema format - hardware_counters section
TEST_F(json_config_test, rasolves_hw_counters_section)
{
    auto j = nlohmann::json::parse(R"({
        "hardware_counters": {
            "enabled": true,
            "rocm_events": {"value": ["VALUUtilization", "Occupancy"]},
            "papi_events": {"value": ["PAPI_TOT_CYC", "PAPI_TOT_INS"]}
        }
    })");

    auto result = resolve_config(j);

    EXPECT_EQ(result.at(env_vars::ROCM_EVENTS), "VALUUtilization,Occupancy");
    EXPECT_EQ(result.at(env_vars::PAPI_EVENTS), "PAPI_TOT_CYC,PAPI_TOT_INS");
}

// Test new schema format - causal section
TEST_F(json_config_test, resolves_causal_section)
{
    auto j = nlohmann::json::parse(R"({
        "causal": {
            "enabled": true,
            "mode": {"value": "function"},
            "backend": {"value": "perf"},
            "binary_scope": {"value": "%MAIN%"}
        }
    })");

    auto result = resolve_config(j);

    EXPECT_EQ(result.at(env_vars::USE_CAUSAL), "true");
    EXPECT_EQ(result.at(env_vars::CAUSAL_MODE), "function");
    EXPECT_EQ(result.at(env_vars::CAUSAL_BACKEND), "perf");
    EXPECT_EQ(result.at(env_vars::CAUSAL_BINARY_SCOPE), "%MAIN%");
}

// Test new schema format - advanced section
TEST_F(json_config_test, resolves_advanced_configuration_section)
{
    auto j = nlohmann::json::parse(R"({
        "advanced": {
            "max_depth": {"value": 100},
            "verbose": {"value": 2},
            "debug": {"enabled": true},
            "collapse_threads": {"enabled": false}
        }
    })");

    auto result = resolve_config(j);

    EXPECT_EQ(result.at(env_vars::MAX_DEPTH), "100");
    EXPECT_EQ(result.at(env_vars::VERBOSE), "2");
    EXPECT_EQ(result.at(env_vars::DEBUG_MODE), "true");
    // collapse_threads is resolved when the enabled field is present
    EXPECT_EQ(result.count(env_vars::COLLAPSE_THREADS), 1u);
}

// Test combined sections
TEST_F(json_config_test, combines_multiple_sections)
{
    auto j = nlohmann::json::parse(R"({
        "metadata": {"name": "test-config"},
        "tracing": {"enabled": true},
        "profiling": {"enabled": true},
        "sampling": {"enabled": true, "frequency_hz": {"value": 50}},
        "domains": {
            "gpu": {"enabled": true}
        }
    })");

    auto result = resolve_config(j);

    EXPECT_EQ(result.at(env_vars::TRACE), "true");
    EXPECT_EQ(result.at(env_vars::PROFILE), "true");
    EXPECT_EQ(result.at(env_vars::USE_SAMPLING), "true");
    EXPECT_EQ(result.at(env_vars::SAMPLING_FREQ), "50");
    EXPECT_EQ(result.at(env_vars::USE_AMD_SMI), "true");
}

// Test empty JSON returns empty map
TEST_F(json_config_test, empty_json_returns_empty_map)
{
    auto j      = nlohmann::json::parse("{}");
    auto result = resolve_config(j);

    EXPECT_TRUE(result.empty());
}

// Test get_config_metadata extraction
TEST_F(json_config_test, extract_configuration_metadata)
{
    auto j = nlohmann::json::parse(R"({
        "metadata": {
            "name": "balanced",
            "description": "Balanced profiling mode",
            "use_case": "General-purpose profiling",
            "category": "general",
            "cli_flag": "--balanced"
        }
    })");

    auto meta = get_config_metadata(j);

    ASSERT_TRUE(meta.has_value());
    EXPECT_EQ(meta->name, "balanced");
    EXPECT_EQ(meta->description, "Balanced profiling mode");
    EXPECT_EQ(meta->use_case, "General-purpose profiling");
    EXPECT_EQ(meta->category, "general");
    EXPECT_EQ(meta->cli_flag, "--balanced");
}

// Test json_value_to_string helper
TEST_F(json_config_test, json_values_to_sting_types)
{
    EXPECT_EQ(json_value_to_string(nlohmann::json("hello")), "hello");
    EXPECT_EQ(json_value_to_string(nlohmann::json(true)), "true");
    EXPECT_EQ(json_value_to_string(nlohmann::json(false)), "false");
    EXPECT_EQ(json_value_to_string(nlohmann::json(42)), "42");

    auto arr = nlohmann::json::array({ "a", "b", "c" });
    EXPECT_EQ(json_value_to_string(arr), "a,b,c");
}

// Test output.rocpd_output resolution
TEST_F(json_config_test, resolves_rocpd_output)
{
    auto j = nlohmann::json::parse(R"({
        "output": {
            "rocpd_output": {"enabled": true}
        }
    })");

    auto result = resolve_config(j);

    EXPECT_EQ(result.at(env_vars::USE_ROCPD), "true");
}

// Test advanced.network_interface resolution
TEST_F(json_config_test, resolves_network_interface)
{
    auto j = nlohmann::json::parse(R"({
        "advanced": {
            "network_interface": {"value": "eth0"}
        }
    })");

    auto result = resolve_config(j);

    EXPECT_EQ(result.at(env_vars::NETWORK_INTERFACE), "eth0");
}

// Test advanced.trace_periods resolution
TEST_F(json_config_test, resolves_trace_periods)
{
    auto j = nlohmann::json::parse(R"({
        "advanced": {
            "trace_periods": {"value": "0:10,20:30"}
        }
    })");

    auto result = resolve_config(j);

    EXPECT_EQ(result.at(env_vars::TRACE_PERIODS), "0:10,20:30");
}

// Test hardware_counters.papi_multiplexing resolution
TEST_F(json_config_test, resolves_papi_multiplexing)
{
    auto j = nlohmann::json::parse(R"({
        "hardware_counters": {
            "enabled": true,
            "papi_multiplexing": {"enabled": true}
        }
    })");

    auto result = resolve_config(j);

    EXPECT_EQ(result.at(env_vars::PAPI_MULTIPLEXING_ENABLED), "true");
}

// Test domains.rocm.enabled top-level flag
TEST_F(json_config_test, resolves_rocm_enabled_flag)
{
    auto j = nlohmann::json::parse(R"({
        "domains": {
            "rocm": {
                "enabled": true
            }
        }
    })");

    auto result = resolve_config(j);

    EXPECT_EQ(result.at(env_vars::TRACE), "true");
}

// Test env_vars_to_json_schema with non-numeric env var values
TEST_F(json_config_test, handling_non_numeric_values_for_json_schema)
{
    std::map<std::string, std::string> env_vars = {
        { "ROCPROFSYS_PERFETTO_BUFFER_SIZE_KB", "not_a_number" },
        { "ROCPROFSYS_SAMPLING_FREQ", "" },
        { "ROCPROFSYS_VERBOSE", "abc" },
        { "ROCPROFSYS_TRACE", "true" },
    };

    // Should not throw - values stored as strings when conversion fails
    nlohmann::json j;
    EXPECT_NO_THROW(j = env_vars_to_json_schema(env_vars));

    EXPECT_EQ(j["tracing"]["enabled"], true);
    // Non-numeric values should be stored as strings instead of crashing
    EXPECT_EQ(j["tracing"]["buffer_size_kb"]["value"], "not_a_number");
    EXPECT_EQ(j["sampling"]["frequency_hz"]["value"], "");
    EXPECT_EQ(j["advanced"]["verbose"]["value"], "abc");
}

// Test env_vars_to_json_schema round-trip for new fields
TEST_F(json_config_test, handling_round_trip_for_new_values_in_json_schema)
{
    std::map<std::string, std::string> env_vars = {
        { "ROCPROFSYS_USE_ROCPD", "true" },
        { "ROCPROFSYS_NETWORK_INTERFACE", "ib0" },
        { "ROCPROFSYS_TRACE_PERIODS", "1:5,10:20" },
        { "ROCPROFSYS_PAPI_MULTIPLEXING", "true" },
    };

    auto j = env_vars_to_json_schema(env_vars);

    EXPECT_EQ(j["output"]["rocpd_output"]["enabled"], true);
    EXPECT_EQ(j["advanced"]["network_interface"]["value"], "ib0");
    EXPECT_EQ(j["advanced"]["trace_periods"]["value"], "1:5,10:20");
    EXPECT_EQ(j["hardware_counters"]["papi_multiplexing"]["enabled"], true);
}

// Test env_vars constants match expected string values
TEST_F(json_config_test, validate_env_var_constants)
{
    EXPECT_STREQ(rocprofsys::env_vars::TRACE, "ROCPROFSYS_TRACE");
    EXPECT_STREQ(rocprofsys::env_vars::PROFILE, "ROCPROFSYS_PROFILE");
    EXPECT_STREQ(rocprofsys::env_vars::USE_SAMPLING, "ROCPROFSYS_USE_SAMPLING");
    EXPECT_STREQ(rocprofsys::env_vars::USE_AMD_SMI, "ROCPROFSYS_USE_AMD_SMI");
    EXPECT_STREQ(rocprofsys::env_vars::ROCM_DOMAINS, "ROCPROFSYS_ROCM_DOMAINS");
    EXPECT_STREQ(rocprofsys::env_vars::USE_MPIP, "ROCPROFSYS_USE_MPIP");
    EXPECT_STREQ(rocprofsys::env_vars::OUTPUT_PATH, "ROCPROFSYS_OUTPUT_PATH");
    EXPECT_STREQ(rocprofsys::env_vars::USE_CAUSAL, "ROCPROFSYS_USE_CAUSAL");
    EXPECT_STREQ(rocprofsys::env_vars::VERBOSE, "ROCPROFSYS_VERBOSE");
}
