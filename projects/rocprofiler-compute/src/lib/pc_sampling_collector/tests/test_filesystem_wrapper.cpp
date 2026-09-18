// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier:  MIT
#include "filesystem_wrapper.h"
#include "gtest/gtest.h"

#include <unistd.h>

#include <filesystem>
#include <fstream>
#include <iterator>
#include <string>
#include <string_view>
#include <system_error>

using namespace rocprofiler_compute_tool;

namespace
{
std::filesystem::path test_output_path(std::string_view label)
{
    return std::filesystem::temp_directory_path() /
           ("filesystem_wrapper_test_" + std::to_string(::getpid()) + "_" + std::string{label});
}

std::string read_file(const std::filesystem::path& path)
{
    std::ifstream input(path);
    return std::string(std::istreambuf_iterator<char>(input), std::istreambuf_iterator<char>{});
}
}  // namespace

TEST(filesystem_wrapper_impl_test, ProvidedStaleError_ClearsItOnSuccessfulWrite)
{
    const auto path = test_output_path("successful_write.json");
    std::filesystem::remove(path);
    std::error_code           error = std::make_error_code(std::errc::permission_denied);
    filesystem_wrapper_impl_t filesystem;

    filesystem.write_file(path, "source map", error);

    EXPECT_FALSE(error);
    EXPECT_EQ(read_file(path), "source map");
    std::filesystem::remove(path);
}

TEST(filesystem_wrapper_impl_test, ProvidedExistingLongerFile_TruncatesIt)
{
    const auto path = test_output_path("truncate.json");
    std::filesystem::remove(path);
    std::error_code           error;
    filesystem_wrapper_impl_t filesystem;

    filesystem.write_file(path, "a much longer previous source map", error);
    ASSERT_FALSE(error);
    filesystem.write_file(path, "short", error);

    EXPECT_FALSE(error);
    EXPECT_EQ(read_file(path), "short");
    std::filesystem::remove(path);
}

TEST(filesystem_wrapper_impl_test, ProvidedMissingParent_SetsError)
{
    const auto missing_directory = test_output_path("missing_parent");
    const auto path              = missing_directory / "source_map.json";
    std::filesystem::remove_all(missing_directory);
    std::error_code           error;
    filesystem_wrapper_impl_t filesystem;

    filesystem.write_file(path, "source map", error);

    EXPECT_TRUE(error);
}

TEST(filesystem_wrapper_impl_test, ProvidedFullDevice_SetsError)
{
    const std::filesystem::path full_device = "/dev/full";
    if (!std::filesystem::exists(full_device))
        GTEST_SKIP() << full_device << " is unavailable";

    std::error_code           error;
    filesystem_wrapper_impl_t filesystem;

    filesystem.write_file(full_device, "source map", error);

    EXPECT_TRUE(error);
}
