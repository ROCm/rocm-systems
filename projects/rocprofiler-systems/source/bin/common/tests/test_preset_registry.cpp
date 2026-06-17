// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "common/preset_registry.hpp"

#include "common/env_vars.hpp"

#include <gtest/gtest.h>

#include <cstdlib>
#include <fstream>
#include <sstream>
#include <string>

using rocprofsys::preset_registry;
namespace env_vars = rocprofsys::env_vars;

namespace
{
class temp_dir
{
public:
    temp_dir()
    {
        char tmpl[] = "/tmp/preset_test_XXXXXX";
        m_path      = ::mkdtemp(tmpl);
    }

    ~temp_dir()
    {
        if(!m_path.empty())
        {
            for(const auto& f : m_files)
                std::remove(f.c_str());
            ::rmdir(m_path.c_str());
        }
    }

    temp_dir(const temp_dir&)            = delete;
    temp_dir& operator=(const temp_dir&) = delete;
    temp_dir(temp_dir&&)                 = delete;
    temp_dir& operator=(temp_dir&&)      = delete;

    const std::string& path() const noexcept { return m_path; }

    std::string write_file(const std::string& name, const std::string& content)
    {
        auto          filepath = m_path + "/" + name;
        std::ofstream ofs{ filepath };
        ofs << content;
        m_files.push_back(filepath);
        return filepath;
    }

private:
    std::string              m_path;
    std::vector<std::string> m_files;
};

constexpr auto balanced_json = R"({
    "metadata": {
        "name": "balanced",
        "cli_flag": "--balanced",
        "description": "Balanced profiling mode",
        "use_case": "General-purpose profiling",
        "category": "general"
    },
    "tracing": {"enabled": true},
    "profiling": {"enabled": true},
    "sampling": {
        "enabled": true,
        "frequency_hz": {"value": 50}
    }
})";

constexpr auto gpu_preset_json = R"({
    "metadata": {
        "name": "gpu-trace",
        "cli_flag": "--gpu-trace",
        "description": "GPU tracing preset"
    },
    "tracing": {"enabled": true},
    "domains": {
        "gpu": {
            "enabled": true,
            "metrics": {
                "temp": {"enabled": true},
                "power": {"enabled": true}
            }
        }
    }
})";

constexpr auto invalid_json = R"({ this is not valid json })";

}  // namespace

class preset_registry_test : public ::testing::Test
{};

TEST_F(preset_registry_test, get_settings_loads_metadata_and_settings)
{
    temp_dir dir;
    auto     filepath = dir.write_file("balanced.json", balanced_json);

    preset_registry registry;
    auto            settings = registry.get_settings(filepath);

    ASSERT_TRUE(settings.has_value());
    EXPECT_EQ(settings->at(std::string{ env_vars::TRACE }), "true");
    EXPECT_EQ(settings->at(std::string{ env_vars::PROFILE }), "true");
    EXPECT_EQ(settings->at(std::string{ env_vars::USE_SAMPLING }), "true");
    EXPECT_EQ(settings->at(std::string{ env_vars::SAMPLING_FREQ }), "50");

    // Verify metadata via explain
    std::ostringstream oss;
    EXPECT_TRUE(registry.explain("balanced", "run", oss));
    auto output = oss.str();
    EXPECT_NE(output.find("balanced"), std::string::npos);
    EXPECT_NE(output.find("Balanced profiling mode"), std::string::npos);
    EXPECT_NE(output.find("General-purpose profiling"), std::string::npos);
}

TEST_F(preset_registry_test, get_settings_resolves_gpu_domain)
{
    temp_dir dir;
    auto     filepath = dir.write_file("gpu-trace.json", gpu_preset_json);

    preset_registry registry;
    auto            settings = registry.get_settings(filepath);

    ASSERT_TRUE(settings.has_value());
    EXPECT_EQ(settings->at(std::string{ env_vars::USE_AMD_SMI }), "true");
    EXPECT_EQ(settings->at(std::string{ env_vars::USE_PROCESS_SAMPLING }), "true");
    auto metrics = settings->at(std::string{ env_vars::AMD_SMI_METRICS });
    EXPECT_NE(metrics.find("temp"), std::string::npos);
    EXPECT_NE(metrics.find("power"), std::string::npos);
}

TEST_F(preset_registry_test, get_settings_returns_nullopt_for_missing_file)
{
    preset_registry registry;
    auto            settings = registry.get_settings("/nonexistent/path/missing.json");
    EXPECT_FALSE(settings.has_value());
}

TEST_F(preset_registry_test, get_settings_returns_nullopt_for_invalid_json)
{
    temp_dir dir;
    auto     filepath = dir.write_file("invalid.json", invalid_json);

    preset_registry registry;
    auto            settings = registry.get_settings(filepath);
    EXPECT_FALSE(settings.has_value());
}

TEST_F(preset_registry_test, explain_finds_by_name)
{
    temp_dir dir;
    dir.write_file("balanced.json", balanced_json);

    ::setenv(env_vars::PRESET_DIR, dir.path().c_str(), 1);
    preset_registry    registry;
    std::ostringstream oss;
    bool               found = registry.explain("balanced", "run", oss);
    ::unsetenv(env_vars::PRESET_DIR);

    EXPECT_TRUE(found);
    EXPECT_NE(oss.str().find("balanced"), std::string::npos);
}

TEST_F(preset_registry_test, explain_returns_false_for_unknown_preset)
{
    temp_dir dir;

    ::setenv(env_vars::PRESET_DIR, dir.path().c_str(), 1);
    preset_registry    registry;
    std::ostringstream oss;
    bool               found = registry.explain("nonexistent-preset", "run", oss);
    ::unsetenv(env_vars::PRESET_DIR);

    EXPECT_FALSE(found);
}

TEST_F(preset_registry_test, load_all_from_directory)
{
    temp_dir dir;
    dir.write_file("balanced.json", balanced_json);

    ::setenv(env_vars::PRESET_DIR, dir.path().c_str(), 1);
    preset_registry    registry;
    std::ostringstream oss;
    registry.list("run", oss);
    ::unsetenv(env_vars::PRESET_DIR);

    auto output = oss.str();
    EXPECT_NE(output.find("balanced"), std::string::npos);
}

TEST_F(preset_registry_test, get_settings_returns_consistent_results)
{
    temp_dir dir;
    auto     filepath = dir.write_file("balanced.json", balanced_json);

    preset_registry registry;
    auto            first  = registry.get_settings(filepath);
    auto            second = registry.get_settings(filepath);

    ASSERT_TRUE(first.has_value());
    ASSERT_TRUE(second.has_value());
    EXPECT_EQ(*first, *second);
}

TEST_F(preset_registry_test, is_section_enabled_checks)
{
    temp_dir dir;
    auto     filepath = dir.write_file("balanced.json", balanced_json);

    preset_registry registry;
    // Trigger load via get_settings
    (void) registry.get_settings(filepath);

    EXPECT_TRUE(registry.is_section_enabled(filepath, "tracing"));
    EXPECT_TRUE(registry.is_section_enabled(filepath, "profiling"));
    // Non-existent section returns default
    EXPECT_TRUE(registry.is_section_enabled(filepath, "nonexistent", true));
    EXPECT_FALSE(registry.is_section_enabled(filepath, "nonexistent", false));
    // Non-existent preset returns default
    EXPECT_TRUE(registry.is_section_enabled("missing", "tracing", true));
}

TEST_F(preset_registry_test, get_settings_handles_empty_metadata)
{
    constexpr auto minimal_json = R"({
        "tracing": {"enabled": true}
    })";

    temp_dir dir;
    auto     filepath = dir.write_file("minimal.json", minimal_json);

    preset_registry registry;
    auto            settings = registry.get_settings(filepath);

    ASSERT_TRUE(settings.has_value());
    EXPECT_EQ(settings->at(std::string{ env_vars::TRACE }), "true");
}

TEST_F(preset_registry_test, list_output_content)
{
    temp_dir dir;
    dir.write_file("balanced.json", balanced_json);

    ::setenv(env_vars::PRESET_DIR, dir.path().c_str(), 1);
    preset_registry    registry;
    std::ostringstream oss;
    registry.list("run", oss);
    ::unsetenv(env_vars::PRESET_DIR);

    auto output = oss.str();
    EXPECT_NE(output.find("Available Presets:"), std::string::npos);
    EXPECT_NE(output.find("balanced"), std::string::npos);
    EXPECT_NE(output.find("rocprof-sys-run"), std::string::npos);
}

TEST_F(preset_registry_test, explain_output_content)
{
    temp_dir dir;
    dir.write_file("balanced.json", balanced_json);

    ::setenv(env_vars::PRESET_DIR, dir.path().c_str(), 1);
    preset_registry    registry;
    std::ostringstream oss;
    bool               result = registry.explain("balanced", "run", oss);
    ::unsetenv(env_vars::PRESET_DIR);

    EXPECT_TRUE(result);
    auto output = oss.str();
    EXPECT_NE(output.find("Preset: balanced"), std::string::npos);
    EXPECT_NE(output.find(std::string{ env_vars::TRACE }), std::string::npos);
}

TEST_F(preset_registry_test, explain_return_false_for_missing_preset)
{
    preset_registry    registry;
    std::ostringstream oss;
    bool               result = registry.explain("nonexistent", "run", oss);
    EXPECT_FALSE(result);
}

TEST_F(preset_registry_test, describe_generates_output_tree)
{
    temp_dir dir;
    dir.write_file("balanced.json", balanced_json);

    ::setenv(env_vars::PRESET_DIR, dir.path().c_str(), 1);
    preset_registry registry;
    auto            desc = registry.describe("balanced");
    ::unsetenv(env_vars::PRESET_DIR);

    EXPECT_NE(desc.find("Tracing:"), std::string::npos);
    EXPECT_NE(desc.find("Profiling:"), std::string::npos);
    EXPECT_NE(desc.find("CPU Sampling:"), std::string::npos);
}
