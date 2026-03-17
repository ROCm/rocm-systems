// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier:  MIT

#include "common/json_config.hpp"

#include <gtest/gtest.h>

#include <fstream>
#include <string>

using namespace rocprofsys::json_config;

class JsonConfigTest : public ::testing::Test
{};

// Test new schema format - tracing section
TEST_F(JsonConfigTest, ResolvesTracingSection)
{
    auto j = nlohmann::json::parse(R"({
        "tracing": {
            "enabled": true,
            "buffer_size_kb": {"value": 2048},
            "fill_policy": {"value": "ring_buffer"}
        }
    })");

    auto result = resolve_config(j);

    EXPECT_EQ(result.at("ROCPROFSYS_TRACE"), "true");
    EXPECT_EQ(result.at("ROCPROFSYS_PERFETTO_BUFFER_SIZE_KB"), "2048");
    EXPECT_EQ(result.at("ROCPROFSYS_PERFETTO_FILL_POLICY"), "ring_buffer");
}

// Test new schema format - profiling section
TEST_F(JsonConfigTest, ResolvesProfilingSection)
{
    auto j = nlohmann::json::parse(R"({
        "profiling": {
            "enabled": true,
            "flat_profile": {"enabled": true}
        }
    })");

    auto result = resolve_config(j);

    EXPECT_EQ(result.at("ROCPROFSYS_PROFILE"), "true");
    EXPECT_EQ(result.at("ROCPROFSYS_FLAT_PROFILE"), "true");
}

// Test new schema format - sampling section
TEST_F(JsonConfigTest, ResolvesSamplingSection)
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

    EXPECT_EQ(result.at("ROCPROFSYS_USE_SAMPLING"), "true");
    EXPECT_EQ(result.at("ROCPROFSYS_SAMPLING_TIMER"), "realtime");
    EXPECT_EQ(result.at("ROCPROFSYS_SAMPLING_FREQ"), "200");
    EXPECT_EQ(result.at("ROCPROFSYS_SAMPLING_CPUS"), "0-3");
    EXPECT_EQ(result.at("ROCPROFSYS_SAMPLING_DELAY"), std::to_string(1.5));
}

// Test new schema format - domains.gpu section
TEST_F(JsonConfigTest, ResolvesGpuDomain)
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

    EXPECT_EQ(result.at("ROCPROFSYS_USE_AMD_SMI"), "true");
    EXPECT_EQ(result.at("ROCPROFSYS_USE_PROCESS_SAMPLING"), "true");
    // Order might vary, but should contain temp and power
    auto metrics = result.at("ROCPROFSYS_AMD_SMI_METRICS");
    EXPECT_NE(metrics.find("temp"), std::string::npos);
    EXPECT_NE(metrics.find("power"), std::string::npos);
    EXPECT_EQ(metrics.find("busy"), std::string::npos);  // busy is disabled
    EXPECT_EQ(result.at("ROCPROFSYS_AMD_SMI_FREQ"), "10");
}

// Test new schema format - domains.rocm section
TEST_F(JsonConfigTest, ResolvesRocmDomain)
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

    auto domains = result.at("ROCPROFSYS_ROCM_DOMAINS");
    EXPECT_NE(domains.find("hip_runtime_api"), std::string::npos);
    EXPECT_NE(domains.find("kernel_dispatch"), std::string::npos);
    EXPECT_EQ(domains.find("memory_copy"), std::string::npos);
}

// Test new schema format - domains.parallel section
TEST_F(JsonConfigTest, ResolvesParallelDomain)
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

    EXPECT_EQ(result.at("ROCPROFSYS_USE_MPIP"), "true");
    EXPECT_EQ(result.at("ROCPROFSYS_USE_OMPT"), "true");
    EXPECT_EQ(result.count("ROCPROFSYS_USE_KOKKOSP"), 0u);
}

// Test new schema format - output section
TEST_F(JsonConfigTest, ResolvesOutputSection)
{
    auto j = nlohmann::json::parse(R"({
        "output": {
            "path": {"value": "/tmp/my-traces"},
            "time_output": {"enabled": false},
            "file_output": {"enabled": true}
        }
    })");

    auto result = resolve_config(j);

    EXPECT_EQ(result.at("ROCPROFSYS_OUTPUT_PATH"), "/tmp/my-traces");
    // time_output and file_output are resolved when the enabled field is present
    EXPECT_EQ(result.count("ROCPROFSYS_TIME_OUTPUT"), 1u);
    EXPECT_EQ(result.count("ROCPROFSYS_FILE_OUTPUT"), 1u);
}

// Test new schema format - hardware_counters section
TEST_F(JsonConfigTest, ResolvesHardwareCountersSection)
{
    auto j = nlohmann::json::parse(R"({
        "hardware_counters": {
            "enabled": true,
            "rocm_events": {"value": ["VALUUtilization", "Occupancy"]},
            "papi_events": {"value": ["PAPI_TOT_CYC", "PAPI_TOT_INS"]}
        }
    })");

    auto result = resolve_config(j);

    EXPECT_EQ(result.at("ROCPROFSYS_ROCM_EVENTS"), "VALUUtilization,Occupancy");
    EXPECT_EQ(result.at("ROCPROFSYS_PAPI_EVENTS"), "PAPI_TOT_CYC,PAPI_TOT_INS");
}

// Test new schema format - causal section
TEST_F(JsonConfigTest, ResolvesCausalSection)
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

    EXPECT_EQ(result.at("ROCPROFSYS_USE_CAUSAL"), "true");
    EXPECT_EQ(result.at("ROCPROFSYS_CAUSAL_MODE"), "function");
    EXPECT_EQ(result.at("ROCPROFSYS_CAUSAL_BACKEND"), "perf");
    EXPECT_EQ(result.at("ROCPROFSYS_CAUSAL_BINARY_SCOPE"), "%MAIN%");
}

// Test new schema format - advanced section
TEST_F(JsonConfigTest, ResolvesAdvancedSection)
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

    EXPECT_EQ(result.at("ROCPROFSYS_MAX_DEPTH"), "100");
    EXPECT_EQ(result.at("ROCPROFSYS_VERBOSE"), "2");
    EXPECT_EQ(result.at("ROCPROFSYS_DEBUG"), "true");
    // collapse_threads is resolved when the enabled field is present
    EXPECT_EQ(result.count("ROCPROFSYS_COLLAPSE_THREADS"), 1u);
}

// Test combined sections
TEST_F(JsonConfigTest, CombinesMultipleSections)
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

    EXPECT_EQ(result.at("ROCPROFSYS_TRACE"), "true");
    EXPECT_EQ(result.at("ROCPROFSYS_PROFILE"), "true");
    EXPECT_EQ(result.at("ROCPROFSYS_USE_SAMPLING"), "true");
    EXPECT_EQ(result.at("ROCPROFSYS_SAMPLING_FREQ"), "50");
    EXPECT_EQ(result.at("ROCPROFSYS_USE_AMD_SMI"), "true");
}

// Test empty JSON returns empty map
TEST_F(JsonConfigTest, EmptyJsonReturnsEmptyMap)
{
    auto j      = nlohmann::json::parse("{}");
    auto result = resolve_config(j);

    EXPECT_TRUE(result.empty());
}

// Test load_and_resolve for schema format file
TEST_F(JsonConfigTest, LoadAndResolveSchemaFormat)
{
    auto content = R"({
        "metadata": {"name": "test-preset"},
        "tracing": {"enabled": true},
        "profiling": {"enabled": false}
    })";

    char tmpl[]   = "/tmp/config_test_XXXXXX";
    auto tmpdir   = std::string{ ::mkdtemp(tmpl) };
    auto filepath = tmpdir + "/preset.json";
    {
        std::ofstream ofs{ filepath };
        ofs << content;
    }

    auto result = load_and_resolve(filepath);

    std::remove(filepath.c_str());
    ::rmdir(tmpdir.c_str());

    ASSERT_TRUE(result.has_value());
    EXPECT_EQ(result->at("ROCPROFSYS_TRACE"), "true");
    EXPECT_EQ(result->at("ROCPROFSYS_PROFILE"), "false");
}

// Test load_and_resolve returns nullopt for missing file
TEST_F(JsonConfigTest, LoadAndResolveReturnsNulloptForMissing)
{
    auto result = load_and_resolve("/nonexistent/config.json");
    EXPECT_FALSE(result.has_value());
}

// Test load_and_resolve returns nullopt for invalid JSON
TEST_F(JsonConfigTest, LoadAndResolveReturnsNulloptForInvalidJson)
{
    char tmpl[]   = "/tmp/config_test_XXXXXX";
    auto tmpdir   = std::string{ ::mkdtemp(tmpl) };
    auto filepath = tmpdir + "/bad.json";
    {
        std::ofstream ofs{ filepath };
        ofs << "{ not valid json }";
    }

    auto result = load_and_resolve(filepath);

    std::remove(filepath.c_str());
    ::rmdir(tmpdir.c_str());

    EXPECT_FALSE(result.has_value());
}

// Test get_config_metadata extraction
TEST_F(JsonConfigTest, ExtractsConfigMetadata)
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
TEST_F(JsonConfigTest, JsonValueToStringHandlesTypes)
{
    EXPECT_EQ(json_value_to_string(nlohmann::json("hello")), "hello");
    EXPECT_EQ(json_value_to_string(nlohmann::json(true)), "true");
    EXPECT_EQ(json_value_to_string(nlohmann::json(false)), "false");
    EXPECT_EQ(json_value_to_string(nlohmann::json(42)), "42");

    auto arr = nlohmann::json::array({ "a", "b", "c" });
    EXPECT_EQ(json_value_to_string(arr), "a,b,c");
}

// Test extract_setting_value helper
TEST_F(JsonConfigTest, ExtractSettingValueHandlesFormats)
{
    auto val_obj = nlohmann::json::parse(R"({"value": 100})");
    EXPECT_EQ(extract_setting_value(val_obj), "100");

    auto enabled_obj = nlohmann::json::parse(R"({"enabled": true})");
    EXPECT_EQ(extract_setting_value(enabled_obj), "true");

    auto disabled_obj = nlohmann::json::parse(R"({"enabled": false})");
    EXPECT_EQ(extract_setting_value(disabled_obj), "false");

    auto bool_val = nlohmann::json(true);
    EXPECT_EQ(extract_setting_value(bool_val), "true");
}
