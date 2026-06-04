// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "common/common_utils.hpp"

#include <gtest/gtest.h>

#include <sstream>
#include <string>

using namespace rocprofsys::common_utils;

class help_system_test : public ::testing::Test
{};

// Synthetic help output that mimics print_help() format (no ANSI codes).
// ANSI stripping is tested separately below.
static const std::string synthetic_help =
    R"([tool] Usage: rocprof-sys-run [ --help --version ]

Options:

    [DEBUG OPTIONS]

    --monochrome                   Disable colorized output (max: 1, dtype: bool)
    -v, --verbose                  Increase verbosity (count: 1, dtype: integral)

    [GENERAL OPTIONS]  These are options which are ubiquitously applied

    -o, --output                   Set the output path (min: 1, dtype: path)
    -T, --trace                    Enable tracing (max: 1, dtype: bool)
    -P, --profile                  Enable profiling (max: 1, dtype: bool)
    -S, --sample                   Enable sampling (min: 0, dtype: timer-type)

    [TRACING OPTIONS]  Specific options controlling tracing

    --trace-file                   Set the trace output file (min: 1, dtype: filepath)
    --trace-buffer-size            Set buffer size in KB (min: 1, dtype: KB)

    [PRESET OPTIONS]   Load a profiling preset

    --preset                       Load a preset configuration (max: 1, dtype: string)
    --list-presets                 List all available presets (max: 0)

    [DOMAIN OPTIONS]   High-level domain flags

    --gpu                          Enable GPU metrics (min: 0, dtype: string)
    --rocm                         Enable ROCm tracing (min: 0, dtype: string)
    --cpu                          Enable CPU sampling (min: 0, dtype: string)

    [EXPORT OPTIONS]   Export resolved configuration

    --export-config                Export config as JSON (min: 0, dtype: filepath)

    [HARDWARE COUNTER OPTIONS] See also: rocprof-sys-avail -H

    -C, --cpu-events               Set CPU hardware counter events (min: 1, dtype: [EVENT])
    -G, --gpu-events               Set GPU hardware counter events (min: 1, dtype: [EVENT])

    [HOST/DEVICE (PROCESS SAMPLING) OPTIONS]

    --process-freq                 Set the host/device sampling frequency (count: 1, dtype: float)
    --gpus                         GPU IDs for SMI queries (count: unlimited, dtype: int)
    -D, --device                   Enable device-based metrics (max: 1, dtype: bool)
    --ai-nics                      AI NIC IDs for SMI queries (count: unlimited, dtype: list)

    [GENERAL SAMPLING OPTIONS] General sampling options

    -f, --sampling-freq            Set sampling frequency (count: 1, dtype: float)
    -t, --tids                     Thread IDs for sampling (min: 1, dtype: int)

    [SAMPLING TIMER OPTIONS] Timer heuristics

    --sample-cputime               Sample based on CPU-clock timer
                                       Accepts zero or more arguments (min: 0, dtype: args)
    --sample-realtime              Sample based on real-clock timer (min: 0, dtype: args)
)";

// Synthetic help with ANSI escape codes for testing ANSI stripping.
static const std::string ansi_help =
    "[tool] Usage: test\n\nOptions:\n\n"
    "    \033[01;34m[DEBUG OPTIONS]\033[0m\n\n"
    "    --monochrome                   Disable colorized output\n"
    "    -v, --verbose                  Increase verbosity\n\n"
    "    \033[01;34m[TRACING OPTIONS]\033[0m  Tracing controls\n\n"
    "    --trace-file                   Set trace output file\n";

// ============================================================================
// Topic map tests
// ============================================================================

TEST_F(help_system_test, topic_map_is_not_empty)
{
    const auto& map = get_help_topic_map();
    EXPECT_FALSE(map.empty());
    EXPECT_NE(map.count("preset"), 0u);
    EXPECT_NE(map.count("sampling"), 0u);
    EXPECT_NE(map.count("tracing"), 0u);
    EXPECT_NE(map.count("general"), 0u);
    EXPECT_NE(map.count("debug"), 0u);
    EXPECT_NE(map.count("counters"), 0u);
}

TEST_F(help_system_test, domain_map_is_not_empty)
{
    const auto& map = get_domain_help_map();
    EXPECT_FALSE(map.empty());
    EXPECT_NE(map.count("gpu"), 0u);
    EXPECT_NE(map.count("cpu"), 0u);
    EXPECT_NE(map.count("rocm"), 0u);
    EXPECT_NE(map.count("parallel"), 0u);
}

TEST_F(help_system_test, domain_entries_have_flags)
{
    const auto& map = get_domain_help_map();
    for(const auto& [name, entry] : map)
    {
        EXPECT_FALSE(entry.description.empty()) << "Domain '" << name << "' has no desc";
        EXPECT_FALSE(entry.flag_patterns.empty())
            << "Domain '" << name << "' has no flags";
    }
}

// ============================================================================
// Compact help tests
// ============================================================================

TEST_F(help_system_test, compact_help_contains_essential_info)
{
    std::ostringstream oss;
    print_compact_help("run", oss);
    auto output = oss.str();

    EXPECT_NE(output.find("--preset"), std::string::npos);
    EXPECT_NE(output.find("--list-presets"), std::string::npos);
    EXPECT_NE(output.find("--gpu"), std::string::npos);
    EXPECT_NE(output.find("--rocm"), std::string::npos);
    EXPECT_NE(output.find("--cpu"), std::string::npos);
    EXPECT_NE(output.find("--help="), std::string::npos);
    EXPECT_NE(output.find("QUICK START"), std::string::npos);
    EXPECT_NE(output.find("HELP TOPICS"), std::string::npos);
    EXPECT_NE(output.find("rocprof-sys-run"), std::string::npos);
}

TEST_F(help_system_test, compact_help_uses_tool_name)
{
    std::ostringstream oss_run, oss_sample;
    print_compact_help("run", oss_run);
    print_compact_help("sample", oss_sample);

    EXPECT_NE(oss_run.str().find("rocprof-sys-run"), std::string::npos);
    EXPECT_NE(oss_sample.str().find("rocprof-sys-sample"), std::string::npos);
}

// ============================================================================
// Topic-based help extraction tests
// ============================================================================

TEST_F(help_system_test, topic_filter_extracts_matching_section)
{
    std::ostringstream oss;
    bool result = print_help_for_topic(synthetic_help, "tracing", "run", oss);
    auto output = oss.str();

    EXPECT_TRUE(result);
    EXPECT_NE(output.find("--trace-file"), std::string::npos);
    EXPECT_NE(output.find("--trace-buffer-size"), std::string::npos);
    // Should NOT include unrelated sections
    EXPECT_EQ(output.find("--monochrome"), std::string::npos);
    EXPECT_EQ(output.find("--preset"), std::string::npos);
}

TEST_F(help_system_test, topic_filter_extracts_multiple_groups)
{
    std::ostringstream oss;
    bool result = print_help_for_topic(synthetic_help, "preset", "run", oss);
    auto output = oss.str();

    EXPECT_TRUE(result);
    // "preset" topic maps to PRESET, DOMAIN, and EXPORT groups
    EXPECT_NE(output.find("--preset"), std::string::npos);
    EXPECT_NE(output.find("--list-presets"), std::string::npos);
    EXPECT_NE(output.find("--gpu"), std::string::npos);
    EXPECT_NE(output.find("--rocm"), std::string::npos);
    EXPECT_NE(output.find("--export-config"), std::string::npos);
    // Should NOT include unrelated
    EXPECT_EQ(output.find("--trace-file"), std::string::npos);
}

TEST_F(help_system_test, topic_filter_sampling_extracts_timer_options)
{
    std::ostringstream oss;
    bool result = print_help_for_topic(synthetic_help, "sampling", "run", oss);
    auto output = oss.str();

    EXPECT_TRUE(result);
    EXPECT_NE(output.find("--sampling-freq"), std::string::npos);
    EXPECT_NE(output.find("--sample-cputime"), std::string::npos);
    EXPECT_NE(output.find("--sample-realtime"), std::string::npos);
}

TEST_F(help_system_test, topic_filter_returns_false_for_unknown_topic)
{
    std::ostringstream oss;
    bool result = print_help_for_topic(synthetic_help, "nonexistent", "run", oss);
    EXPECT_FALSE(result);
}

TEST_F(help_system_test, topic_filter_debug_section)
{
    std::ostringstream oss;
    bool               result = print_help_for_topic(synthetic_help, "debug", "run", oss);
    auto               output = oss.str();

    EXPECT_TRUE(result);
    EXPECT_NE(output.find("--monochrome"), std::string::npos);
    EXPECT_NE(output.find("--verbose"), std::string::npos);
    // Should not include other sections
    EXPECT_EQ(output.find("--trace-file"), std::string::npos);
}

// ============================================================================
// Domain-based help extraction tests
// ============================================================================

TEST_F(help_system_test, domain_gpu_extracts_related_options)
{
    std::ostringstream oss;
    bool               result = print_help_for_domain(synthetic_help, "gpu", "run", oss);
    auto               output = oss.str();

    EXPECT_TRUE(result);
    EXPECT_NE(output.find("GPU OPTIONS"), std::string::npos);
    EXPECT_NE(output.find("--gpu"), std::string::npos);
    EXPECT_NE(output.find("--gpus"), std::string::npos);
    EXPECT_NE(output.find("--process-freq"), std::string::npos);
    EXPECT_NE(output.find("--device"), std::string::npos);
    EXPECT_NE(output.find("--gpu-events"), std::string::npos);
    EXPECT_NE(output.find("--ai-nics"), std::string::npos);
    // Should NOT include CPU-only options
    EXPECT_EQ(output.find("--cpu-events"), std::string::npos);
    EXPECT_EQ(output.find("--sampling-freq"), std::string::npos);
}

TEST_F(help_system_test, domain_cpu_extracts_related_options)
{
    std::ostringstream oss;
    bool               result = print_help_for_domain(synthetic_help, "cpu", "run", oss);
    auto               output = oss.str();

    EXPECT_TRUE(result);
    EXPECT_NE(output.find("CPU OPTIONS"), std::string::npos);
    EXPECT_NE(output.find("--cpu"), std::string::npos);
    EXPECT_NE(output.find("--sampling-freq"), std::string::npos);
    EXPECT_NE(output.find("--sample-cputime"), std::string::npos);
    EXPECT_NE(output.find("--sample-realtime"), std::string::npos);
    EXPECT_NE(output.find("--cpu-events"), std::string::npos);
    EXPECT_NE(output.find("--sample"), std::string::npos);
}

TEST_F(help_system_test, domain_cpu_include_continuation_lines)
{
    std::ostringstream    oss;
    [[maybe_unused]] auto matched =
        print_help_for_domain(synthetic_help, "cpu", "run", oss);
    auto output = oss.str();

    // --sample-cputime has a continuation line "Accepts zero or more arguments"
    EXPECT_NE(output.find("Accepts zero or more arguments"), std::string::npos);
}

TEST_F(help_system_test, domain_return_false_for_unknown_domain)
{
    std::ostringstream oss;
    bool result = print_help_for_domain(synthetic_help, "nonexistent", "run", oss);
    EXPECT_FALSE(result);
}

TEST_F(help_system_test, domain_rocm_extracts_related_options)
{
    std::ostringstream oss;
    bool               result = print_help_for_domain(synthetic_help, "rocm", "run", oss);
    auto               output = oss.str();

    EXPECT_TRUE(result);
    EXPECT_NE(output.find("ROCM OPTIONS"), std::string::npos);
    EXPECT_NE(output.find("--rocm"), std::string::npos);
    EXPECT_NE(output.find("--trace"), std::string::npos);
}

// ============================================================================
// ANSI escape code handling
// ============================================================================

TEST_F(help_system_test, topic_filter_works_with_ansi_codes)
{
    std::ostringstream oss;
    bool               result = print_help_for_topic(ansi_help, "debug", "run", oss);
    auto               output = oss.str();

    EXPECT_TRUE(result);
    EXPECT_NE(output.find("--monochrome"), std::string::npos);
    EXPECT_NE(output.find("--verbose"), std::string::npos);
    // Should not include tracing section
    EXPECT_EQ(output.find("--trace-file"), std::string::npos);
}

TEST_F(help_system_test, topic_filter_ansi_tracing_section)
{
    std::ostringstream oss;
    bool               result = print_help_for_topic(ansi_help, "tracing", "run", oss);
    auto               output = oss.str();

    EXPECT_TRUE(result);
    EXPECT_NE(output.find("--trace-file"), std::string::npos);
    // Should not include debug section
    EXPECT_EQ(output.find("--monochrome"), std::string::npos);
}
