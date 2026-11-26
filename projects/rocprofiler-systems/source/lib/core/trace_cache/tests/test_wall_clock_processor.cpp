// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "core/trace_cache/metadata_registry.hpp"
#include "core/trace_cache/sample_type.hpp"
#include "core/trace_cache/wall_clock_processor.hpp"

#include <gtest/gtest.h>

#include <sys/types.h>
#include <unistd.h>

#include <cstdint>
#include <filesystem>
#include <fstream>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>

using rocprofsys::trace_cache::kernel_dispatch_sample;
using rocprofsys::trace_cache::metadata_registry;
using rocprofsys::trace_cache::region_sample;
using rocprofsys::trace_cache::scratch_memory_sample;
using rocprofsys::trace_cache::wall_clock_options;
using rocprofsys::trace_cache::wall_clock_processor_t;

namespace
{
constexpr std::uint64_t k_begin_ns  = 1'000'000;
constexpr std::uint64_t k_end_ns    = 3'000'000;
constexpr std::uint64_t k_thread    = 1;
constexpr std::uint64_t k_kernel_id = 7;
constexpr pid_t         k_pid       = 1;

[[nodiscard]] region_sample
make_region(const char* name, std::uint64_t begin = k_begin_ns,
            std::uint64_t end = k_end_ns)
{
    return region_sample{ k_thread, name, 0, 0, begin, end, "{}", "", "" };
}

[[nodiscard]] kernel_dispatch_sample
make_kernel_dispatch(std::uint64_t kernel_id)
{
    return kernel_dispatch_sample{ k_begin_ns, k_end_ns, k_thread, 0, kernel_id, 0,
                                   0,          0,        0,        0, 0,         1,
                                   1,          1,        1,        1, 1,         0 };
}

[[nodiscard]] wall_clock_processor_t
make_processor(
    std::shared_ptr<metadata_registry> metadata = std::make_shared<metadata_registry>(),
    wall_clock_options                 options  = {})
{
    return wall_clock_processor_t{ std::move(metadata), nullptr, k_pid,
                                   std::move(options) };
}

[[nodiscard]] std::shared_ptr<metadata_registry>
make_metadata_with_kernel(std::uint64_t kernel_id, const char* kernel_name)
{
    auto metadata = std::make_shared<metadata_registry>();
    rocprofiler_callback_tracing_code_object_kernel_symbol_register_data_t symbol{};
    symbol.kernel_id   = kernel_id;
    symbol.kernel_name = kernel_name;
    metadata->add_kernel_symbol(symbol);
    return metadata;
}

[[nodiscard]] std::string
read_file(const std::filesystem::path& path)
{
    const std::ifstream stream{ path };
    std::ostringstream  content;
    content << stream.rdbuf();
    return content.str();
}

class wall_clock_processor_files : public ::testing::Test
{
protected:
    void SetUp() override
    {
        m_directory = std::filesystem::temp_directory_path() /
                      ("wall_clock_processor_" + std::to_string(::getpid()));
        std::filesystem::remove_all(m_directory);
    }

    void TearDown() override { std::filesystem::remove_all(m_directory); }

    [[nodiscard]] wall_clock_options options() const
    {
        return { .text_path = (m_directory / "wall_clock.txt").string(),
                 .json_path = (m_directory / "wall_clock.json").string() };
    }

    std::filesystem::path m_directory;
};
}  // namespace

TEST(wall_clock_processor, kernel_with_unregistered_symbol_throws)
{
    constexpr std::uint64_t k_unknown_kernel = 999;
    auto                    processor        = make_processor();
    EXPECT_THROW(processor.handle(make_kernel_dispatch(k_unknown_kernel)),
                 std::runtime_error);
}

TEST(wall_clock_processor, region_ending_before_it_begins_is_skipped)
{
    auto processor = make_processor();
    EXPECT_NO_THROW(processor.handle(make_region("reversed", k_end_ns, k_begin_ns)));
}

TEST_F(wall_clock_processor_files, finalize_writes_the_text_and_json_files)
{
    auto processor = make_processor(std::make_shared<metadata_registry>(), options());
    processor.handle(make_region("work"));
    processor.finalize_processing();

    EXPECT_NE(read_file(m_directory / "wall_clock.txt").find("work"), std::string::npos);
    EXPECT_NE(read_file(m_directory / "wall_clock.json").find("|0>>> work"),
              std::string::npos);
}

TEST_F(wall_clock_processor_files, finalize_without_samples_writes_nothing)
{
    auto processor = make_processor(std::make_shared<metadata_registry>(), options());
    processor.finalize_processing();

    EXPECT_FALSE(std::filesystem::exists(m_directory));
}

TEST_F(wall_clock_processor_files, a_file_that_cannot_be_written_does_not_stop_the_others)
{
    auto unwritable      = options();
    unwritable.text_path = "/dev/null/wall_clock.txt";
    auto processor = make_processor(std::make_shared<metadata_registry>(), unwritable);
    processor.handle(make_region("work"));

    EXPECT_NO_THROW(processor.finalize_processing());
    EXPECT_TRUE(std::filesystem::exists(m_directory / "wall_clock.json"));
}

TEST_F(wall_clock_processor_files, kernels_and_scratch_memory_are_reported)
{
    constexpr std::uint64_t k_flags_none = 0;
    auto                    processor =
        make_processor(make_metadata_with_kernel(k_kernel_id, "my_kernel"), options());
    processor.handle(make_kernel_dispatch(k_kernel_id));
    processor.handle(scratch_memory_sample{ k_begin_ns, k_end_ns, k_thread, 0, 0,
                                            "my_scratch", 0, 0, k_flags_none, 0, 0, 0 });
    processor.finalize_processing();

    auto const text = read_file(m_directory / "wall_clock.txt");
    EXPECT_NE(text.find("my_kernel"), std::string::npos);
    EXPECT_NE(text.find("my_scratch"), std::string::npos);
}

TEST_F(wall_clock_processor_files, dispatches_of_one_kernel_merge_into_one_row)
{
    auto processor =
        make_processor(make_metadata_with_kernel(k_kernel_id, "my_kernel"), options());
    processor.handle(make_kernel_dispatch(k_kernel_id));
    processor.handle(make_kernel_dispatch(k_kernel_id));
    processor.finalize_processing();

    auto const json = read_file(m_directory / "wall_clock.json");
    EXPECT_NE(json.find("\"graph_size\": 1"), std::string::npos);
    EXPECT_NE(json.find("\"laps\": 2"), std::string::npos);
}

TEST(wall_clock_processor, stdout_option_prints_the_table)
{
    auto processor = make_processor(std::make_shared<metadata_registry>(),
                                    { .should_print_to_stdout = true });
    processor.handle(make_region("work"));

    ::testing::internal::CaptureStdout();
    processor.finalize_processing();
    EXPECT_NE(::testing::internal::GetCapturedStdout().find("work"), std::string::npos);
}
