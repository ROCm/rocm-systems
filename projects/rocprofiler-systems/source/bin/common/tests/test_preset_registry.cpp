// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier:  MIT

#include "common/preset_registry.hpp"

#include <gtest/gtest.h>

#include <cstdlib>
#include <fstream>
#include <sstream>
#include <string>

using rocprofsys::preset_registry;

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

TEST_F(preset_registry_test, find_file_by_path)
{
    temp_dir dir;
    auto     filepath = dir.write_file("balanced.json", balanced_json);

    preset_registry registry;
    const auto*     info = registry.find(filepath);

    ASSERT_NE(info, nullptr);
    EXPECT_EQ(info->name, "balanced");
    EXPECT_EQ(info->cli_flag, "--balanced");
    EXPECT_EQ(info->description, "Balanced profiling mode");
    EXPECT_EQ(info->use_case, "General-purpose profiling");
    EXPECT_EQ(info->category, "general");
}

TEST_F(preset_registry_test, find_resolves_settings)
{
    temp_dir dir;
    auto     filepath = dir.write_file("balanced.json", balanced_json);

    preset_registry registry;
    const auto*     info = registry.find(filepath);

    ASSERT_NE(info, nullptr);
    EXPECT_EQ(info->settings.at("ROCPROFSYS_TRACE"), "true");
    EXPECT_EQ(info->settings.at("ROCPROFSYS_PROFILE"), "true");
    EXPECT_EQ(info->settings.at("ROCPROFSYS_USE_SAMPLING"), "true");
    EXPECT_EQ(info->settings.at("ROCPROFSYS_SAMPLING_FREQ"), "50");
}

TEST_F(preset_registry_test, find_resolves_gpu_domain)
{
    temp_dir dir;
    auto     filepath = dir.write_file("gpu-trace.json", gpu_preset_json);

    preset_registry registry;
    const auto*     info = registry.find(filepath);

    ASSERT_NE(info, nullptr);
    EXPECT_EQ(info->settings.at("ROCPROFSYS_USE_AMD_SMI"), "true");
    EXPECT_EQ(info->settings.at("ROCPROFSYS_USE_PROCESS_SAMPLING"), "true");
    auto metrics = info->settings.at("ROCPROFSYS_AMD_SMI_METRICS");
    EXPECT_NE(metrics.find("temp"), std::string::npos);
    EXPECT_NE(metrics.find("power"), std::string::npos);
}

TEST_F(preset_registry_test, find_returns_nullptr_for_missing_file)
{
    preset_registry registry;
    const auto*     info = registry.find("/nonexistent/path/missing.json");
    EXPECT_EQ(info, nullptr);
}

TEST_F(preset_registry_test, find_returns_nullptr_for_invalid_json)
{
    temp_dir dir;
    auto     filepath = dir.write_file("invalid.json", invalid_json);

    preset_registry registry;
    const auto*     info = registry.find(filepath);
    EXPECT_EQ(info, nullptr);
}

TEST_F(preset_registry_test, find_by_name)
{
    temp_dir dir;
    dir.write_file("balanced.json", balanced_json);

    ::setenv("ROCPROFSYS_PRESET_DIR", dir.path().c_str(), 1);
    preset_registry registry;
    const auto*     info = registry.find("balanced");
    ::unsetenv("ROCPROFSYS_PRESET_DIR");

    ASSERT_NE(info, nullptr);
    EXPECT_EQ(info->name, "balanced");
}

TEST_F(preset_registry_test, find_by_name_returns_nullptr_for_unknown_preset)
{
    temp_dir dir;

    ::setenv("ROCPROFSYS_PRESET_DIR", dir.path().c_str(), 1);
    preset_registry registry;
    const auto*     info = registry.find("nonexistent-preset");
    ::unsetenv("ROCPROFSYS_PRESET_DIR");

    EXPECT_EQ(info, nullptr);
}

TEST_F(preset_registry_test, directory_from_environment)
{
    temp_dir dir;

    ::setenv("ROCPROFSYS_PRESET_DIR", dir.path().c_str(), 1);
    preset_registry registry;
    ::unsetenv("ROCPROFSYS_PRESET_DIR");

    EXPECT_EQ(registry.directory(), dir.path());
}

TEST_F(preset_registry_test, load_all_from_directory)
{
    temp_dir dir;
    dir.write_file("balanced.json", balanced_json);

    ::setenv("ROCPROFSYS_PRESET_DIR", dir.path().c_str(), 1);
    preset_registry registry;
    const auto&     presets = registry.all();
    ::unsetenv("ROCPROFSYS_PRESET_DIR");

    ASSERT_EQ(presets.count("balanced"), 1u);
    EXPECT_EQ(presets.at("balanced").name, "balanced");
}

TEST_F(preset_registry_test, caching_found_presets)
{
    temp_dir dir;
    auto     filepath = dir.write_file("balanced.json", balanced_json);

    preset_registry registry;
    const auto*     first  = registry.find(filepath);
    const auto*     second = registry.find(filepath);

    ASSERT_NE(first, nullptr);
    EXPECT_EQ(first, second);  // Same pointer, but cached
}

TEST_F(preset_registry_test, apply_calls_callbacks)
{
    temp_dir dir;
    auto     filepath = dir.write_file("balanced.json", balanced_json);

    preset_registry                    registry;
    std::map<std::string, std::string> applied;
    bool                               result = registry.apply(
        filepath, [&](const std::string& k, const std::string& v) { applied[k] = v; });

    EXPECT_TRUE(result);
    EXPECT_EQ(applied.at("ROCPROFSYS_TRACE"), "true");
    EXPECT_EQ(applied.at("ROCPROFSYS_PROFILE"), "true");
}

TEST_F(preset_registry_test, apply_retuns_false_for_missing_file)
{
    preset_registry registry;
    bool            result = registry.apply("/nonexistent.json", [](auto&, auto&) {});
    EXPECT_FALSE(result);
}

TEST_F(preset_registry_test, caching_raw_json)
{
    temp_dir dir;
    auto     filepath = dir.write_file("balanced.json", balanced_json);

    preset_registry registry;
    // triggers load + JSON cache. Voiding it since we
    // wnat to test caching of found json
    (void) registry.find(filepath);

    const auto* j = registry.raw_json(filepath);
    ASSERT_NE(j, nullptr);
    EXPECT_TRUE(j->contains("metadata"));
    EXPECT_TRUE(j->contains("tracing"));
}

TEST_F(preset_registry_test, handling_empty_metadata)
{
    constexpr auto minimal_json = R"({
        "tracing": {"enabled": true}
    })";

    temp_dir dir;
    auto     filepath = dir.write_file("minimal.json", minimal_json);

    preset_registry registry;
    const auto*     info = registry.find(filepath);

    ASSERT_NE(info, nullptr);
    EXPECT_TRUE(info->name.empty());
    EXPECT_TRUE(info->cli_flag.empty());
    EXPECT_EQ(info->settings.at("ROCPROFSYS_TRACE"), "true");
}

TEST_F(preset_registry_test, list_output_content)
{
    temp_dir dir;
    dir.write_file("balanced.json", balanced_json);

    ::setenv("ROCPROFSYS_PRESET_DIR", dir.path().c_str(), 1);
    preset_registry    registry;
    std::ostringstream oss;
    registry.list("run", oss);
    ::unsetenv("ROCPROFSYS_PRESET_DIR");

    auto output = oss.str();
    EXPECT_NE(output.find("Available Presets:"), std::string::npos);
    EXPECT_NE(output.find("balanced"), std::string::npos);
    EXPECT_NE(output.find("rocprof-sys-run"), std::string::npos);
}

TEST_F(preset_registry_test, explain_output_content)
{
    temp_dir dir;
    dir.write_file("balanced.json", balanced_json);

    ::setenv("ROCPROFSYS_PRESET_DIR", dir.path().c_str(), 1);
    preset_registry    registry;
    std::ostringstream oss;
    bool               result = registry.explain("balanced", "run", oss);
    ::unsetenv("ROCPROFSYS_PRESET_DIR");

    EXPECT_TRUE(result);
    auto output = oss.str();
    EXPECT_NE(output.find("Preset: balanced"), std::string::npos);
    EXPECT_NE(output.find("ROCPROFSYS_TRACE"), std::string::npos);
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

    ::setenv("ROCPROFSYS_PRESET_DIR", dir.path().c_str(), 1);
    preset_registry registry;
    auto            desc = registry.describe("balanced");
    ::unsetenv("ROCPROFSYS_PRESET_DIR");

    EXPECT_NE(desc.find("Tracing:"), std::string::npos);
    EXPECT_NE(desc.find("Profiling:"), std::string::npos);
    EXPECT_NE(desc.find("CPU Sampling:"), std::string::npos);
}
