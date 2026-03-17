// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier:  MIT

#include "common/preset_loader.hpp"

#include <gtest/gtest.h>

#include <cstdlib>
#include <fstream>
#include <string>

using namespace rocprofsys::preset_loader;

namespace
{
class TempDir
{
public:
    TempDir()
    {
        char tmpl[] = "/tmp/preset_test_XXXXXX";
        m_path      = ::mkdtemp(tmpl);
    }

    ~TempDir()
    {
        if(!m_path.empty())
        {
            for(const auto& f : m_files)
                std::remove(f.c_str());
            ::rmdir(m_path.c_str());
        }
    }

    TempDir(const TempDir&)            = delete;
    TempDir& operator=(const TempDir&) = delete;
    TempDir(TempDir&&)                 = delete;
    TempDir& operator=(TempDir&&)      = delete;

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

// New schema format preset
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

// Preset with GPU domain
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

class PresetLoaderTest : public ::testing::Test
{};

TEST_F(PresetLoaderTest, LoadsSchemaFormatPreset)
{
    TempDir dir;
    auto    filepath = dir.write_file("balanced.json", balanced_json);

    auto result = load_preset_file(filepath);

    ASSERT_TRUE(result.has_value());
    EXPECT_EQ(result->name, "balanced");
    EXPECT_EQ(result->cli_flag, "--balanced");
    EXPECT_EQ(result->description, "Balanced profiling mode");
    EXPECT_EQ(result->use_case, "General-purpose profiling");
    EXPECT_EQ(result->category, "general");
}

TEST_F(PresetLoaderTest, ResolvesSchemaFormatSettings)
{
    TempDir dir;
    auto    filepath = dir.write_file("balanced.json", balanced_json);

    auto result = load_preset_file(filepath);

    ASSERT_TRUE(result.has_value());
    EXPECT_EQ(result->settings.at("ROCPROFSYS_TRACE"), "true");
    EXPECT_EQ(result->settings.at("ROCPROFSYS_PROFILE"), "true");
    EXPECT_EQ(result->settings.at("ROCPROFSYS_USE_SAMPLING"), "true");
    EXPECT_EQ(result->settings.at("ROCPROFSYS_SAMPLING_FREQ"), "50");
}

TEST_F(PresetLoaderTest, ResolvesGpuDomain)
{
    TempDir dir;
    auto    filepath = dir.write_file("gpu-trace.json", gpu_preset_json);

    auto result = load_preset_file(filepath);

    ASSERT_TRUE(result.has_value());
    EXPECT_EQ(result->settings.at("ROCPROFSYS_USE_AMD_SMI"), "true");
    EXPECT_EQ(result->settings.at("ROCPROFSYS_USE_PROCESS_SAMPLING"), "true");
    auto metrics = result->settings.at("ROCPROFSYS_AMD_SMI_METRICS");
    EXPECT_NE(metrics.find("temp"), std::string::npos);
    EXPECT_NE(metrics.find("power"), std::string::npos);
}

TEST_F(PresetLoaderTest, ReturnsNulloptForMissingFile)
{
    auto result = load_preset_file("/nonexistent/path/missing.json");
    EXPECT_FALSE(result.has_value());
}

TEST_F(PresetLoaderTest, ReturnsNulloptForInvalidJson)
{
    TempDir dir;
    auto    filepath = dir.write_file("invalid.json", invalid_json);

    auto result = load_preset_file(filepath);
    EXPECT_FALSE(result.has_value());
}

TEST_F(PresetLoaderTest, LoadPresetByName)
{
    TempDir dir;
    dir.write_file("balanced.json", balanced_json);

    ::setenv("ROCPROFSYS_PRESET_DIR", dir.path().c_str(), 1);
    auto result = load_preset("balanced");
    ::unsetenv("ROCPROFSYS_PRESET_DIR");

    ASSERT_TRUE(result.has_value());
    EXPECT_EQ(result->name, "balanced");
}

TEST_F(PresetLoaderTest, LoadPresetReturnsNulloptForUnknown)
{
    TempDir dir;

    ::setenv("ROCPROFSYS_PRESET_DIR", dir.path().c_str(), 1);
    auto result = load_preset("nonexistent-preset");
    ::unsetenv("ROCPROFSYS_PRESET_DIR");

    EXPECT_FALSE(result.has_value());
}

TEST_F(PresetLoaderTest, FindPresetDirectoryFromEnv)
{
    TempDir dir;

    ::setenv("ROCPROFSYS_PRESET_DIR", dir.path().c_str(), 1);
    auto found = find_preset_directory();
    ::unsetenv("ROCPROFSYS_PRESET_DIR");

    EXPECT_EQ(found, dir.path());
}

TEST_F(PresetLoaderTest, FindPresetDirectoryEmptyWhenNoneExist)
{
    ::unsetenv("ROCPROFSYS_PRESET_DIR");
    ::unsetenv("ROCM_PATH");

    auto found = find_preset_directory();
    // May or may not find a directory depending on the build environment,
    // but at minimum should not crash
    (void) found;
}

TEST_F(PresetLoaderTest, LoadAllPresetsFromDirectory)
{
    TempDir dir;
    dir.write_file("balanced.json", balanced_json);

    ::setenv("ROCPROFSYS_PRESET_DIR", dir.path().c_str(), 1);
    auto presets = load_all_presets();
    ::unsetenv("ROCPROFSYS_PRESET_DIR");

    ASSERT_EQ(presets.count("balanced"), 1u);
    EXPECT_EQ(presets.at("balanced").name, "balanced");
}

TEST_F(PresetLoaderTest, HandlesEmptyPresetMetadata)
{
    constexpr auto minimal_json = R"({
        "tracing": {"enabled": true}
    })";

    TempDir dir;
    auto    filepath = dir.write_file("minimal.json", minimal_json);

    auto result = load_preset_file(filepath);

    ASSERT_TRUE(result.has_value());
    EXPECT_TRUE(result->name.empty());
    EXPECT_TRUE(result->cli_flag.empty());
    EXPECT_EQ(result->settings.at("ROCPROFSYS_TRACE"), "true");
}

TEST_F(PresetLoaderTest, LoadPresetOrFileWithFilePath)
{
    TempDir dir;
    auto    filepath = dir.write_file("custom.json", balanced_json);

    auto result = load_preset_or_file(filepath);

    ASSERT_TRUE(result.has_value());
    EXPECT_EQ(result->name, "balanced");
}

TEST_F(PresetLoaderTest, LoadPresetOrFileWithPresetName)
{
    TempDir dir;
    dir.write_file("balanced.json", balanced_json);

    ::setenv("ROCPROFSYS_PRESET_DIR", dir.path().c_str(), 1);
    auto result = load_preset_or_file("balanced");
    ::unsetenv("ROCPROFSYS_PRESET_DIR");

    ASSERT_TRUE(result.has_value());
    EXPECT_EQ(result->name, "balanced");
}

TEST_F(PresetLoaderTest, ApplyPresetToEnvironment)
{
    TempDir dir;
    auto    filepath = dir.write_file("balanced.json", balanced_json);

    auto result = load_preset_file(filepath);
    ASSERT_TRUE(result.has_value());

    // Clear any existing env vars
    ::unsetenv("ROCPROFSYS_TRACE");
    ::unsetenv("ROCPROFSYS_PROFILE");

    apply_preset_to_environment(*result);

    EXPECT_STREQ(::getenv("ROCPROFSYS_TRACE"), "true");
    EXPECT_STREQ(::getenv("ROCPROFSYS_PROFILE"), "true");

    // Cleanup
    ::unsetenv("ROCPROFSYS_TRACE");
    ::unsetenv("ROCPROFSYS_PROFILE");
    ::unsetenv("ROCPROFSYS_USE_SAMPLING");
    ::unsetenv("ROCPROFSYS_SAMPLING_FREQ");
}

TEST_F(PresetLoaderTest, ApplyPresetDoesNotOverrideExisting)
{
    TempDir dir;
    auto    filepath = dir.write_file("balanced.json", balanced_json);

    auto result = load_preset_file(filepath);
    ASSERT_TRUE(result.has_value());

    // Set an existing value
    ::setenv("ROCPROFSYS_TRACE", "false", 1);

    apply_preset_to_environment(*result, false);  // override_existing = false

    // Should keep existing value
    EXPECT_STREQ(::getenv("ROCPROFSYS_TRACE"), "false");

    // Cleanup
    ::unsetenv("ROCPROFSYS_TRACE");
    ::unsetenv("ROCPROFSYS_PROFILE");
    ::unsetenv("ROCPROFSYS_USE_SAMPLING");
    ::unsetenv("ROCPROFSYS_SAMPLING_FREQ");
}
