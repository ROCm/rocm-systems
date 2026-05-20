// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "core_flags.hpp"

#include "common/env_vars.hpp"
#include "detail/parser_engine.hpp"
#include "exception.hpp"
#include "flag_descriptor.hpp"
#include "interpreter.hpp"

#include <spdlog/fmt/fmt.h>
#include <spdlog/fmt/ranges.h>

#include <algorithm>
#include <array>
#include <set>
#include <string>
#include <vector>

namespace rocprofsys
{
namespace argparse
{
namespace
{
namespace env = rocprofsys::env_vars;
using common::update_mode;

// --- Help text (each used exactly once; kept inline to avoid spreading a
//     descriptor's literal data across two files) ---

constexpr std::string_view CPUTIME_HELP =
    R"(Sample based on a CPU-clock timer (default). Accepts zero or more arguments:
    %{INDENT}%0. Enables sampling based on CPU-clock timer.
    %{INDENT}%1. Interrupts per second. E.g., 100 == sample every 10 milliseconds of CPU-time.
    %{INDENT}%2. Delay (in seconds of CPU-clock time). I.e., how long each thread should wait before taking first sample.
    %{INDENT}%3+ Thread IDs to target for sampling, starting at 0 (the main thread).
    %{INDENT}%   May be specified as index or range, e.g., '0 2-4' will be interpreted as:
    %{INDENT}%      sample the main thread (0), do not sample the first child thread but sample the 2nd, 3rd, and 4th child threads)";

constexpr std::string_view REALTIME_HELP =
    R"(Sample based on a real-clock timer. Accepts zero or more arguments:
    %{INDENT}%0. Enables sampling based on real-clock timer.
    %{INDENT}%1. Interrupts per second. E.g., 100 == sample every 10 milliseconds of realtime.
    %{INDENT}%2. Delay (in seconds of real-clock time). I.e., how long each thread should wait before taking first sample.
    %{INDENT}%3+ Thread IDs to target for sampling, starting at 0 (the main thread).
    %{INDENT}%   May be specified as index or range, e.g., '0 2-4' will be interpreted as:
    %{INDENT}%      sample the main thread (0), do not sample the first child thread but sample the 2nd, 3rd, and 4th child threads
    %{INDENT}%   When sampling with a real-clock timer, please note that enabling this will cause threads which are typically "idle"
    %{INDENT}%   to consume more resources since, while idle, the real-clock time increases (and therefore triggers taking samples)
    %{INDENT}%   whereas the CPU-clock time does not.)";

constexpr std::string_view OVERFLOW_HELP =
    R"(Sample based on an overflow event. Accepts zero or more arguments:
    %{INDENT}%0. Enables sampling based on overflow.
    %{INDENT}%1. Overflow metric, e.g. PERF_COUNT_HW_INSTRUCTIONS
    %{INDENT}%2. Overflow value. E.g., if metric == PERF_COUNT_HW_INSTRUCTIONS, then 10000000 == sample every 10,000,000 instructions.
    %{INDENT}%3+ Thread IDs to target for sampling, starting at 0 (the main thread).
    %{INDENT}%   May be specified as index or range, e.g., '0 2-4' will be interpreted as:
    %{INDENT}%      sample the main thread (0), do not sample the first child thread but sample the 2nd, 3rd, and 4th child threads)";

constexpr std::string_view HSA_INTERRUPT_HELP =
    R"(Set the value of the HSA_ENABLE_INTERRUPT environment variable.
%{INDENT}%  ROCm version 5.2 and older have a bug which will cause a deadlock if a sample is taken while waiting for the signal
%{INDENT}%  that a kernel completed -- which happens when sampling with a real-clock timer. We require this option to be set to
%{INDENT}%  when --sample-realtime is specified to make users aware that, while this may fix the bug, it can have a negative impact on
%{INDENT}%  performance.
%{INDENT}%  Values:
%{INDENT}%    0     avoid triggering the bug, potentially at the cost of reduced performance
%{INDENT}%    1     do not modify how ROCm is notified about kernel completion)";

// --- Custom actions: flags whose behavior cannot be expressed as pure data ---

void
monochrome_action(parser_t& parser, parser_data& data)
{
    auto enabled        = parser.get<bool>("monochrome");
    data.out.monochrome = enabled;
    parser.set_use_color(!enabled);
    data.env.set(env::MONOCHROME, enabled ? "1" : "0");
    // Bare MONOCHROME (non-ROCPROFSYS_) is consumed by other tools; intentionally
    // not in env_vars.hpp because it isn't part of our prefixed namespace.
    data.env.set("MONOCHROME", enabled ? "1" : "0");
}

void
debug_action(parser_t& /*parser*/, parser_data& data)
{
    data.env.set(env::LOG_LEVEL, "debug");
}

void
verbose_action(parser_t& parser, parser_data& data)
{
    auto level       = parser.get<int>("verbose");
    data.out.verbose = level;
    data.env.set(env::VERBOSE_OUTPUT, level);
    constexpr std::array<const char*, 5> log_levels = { "off", "info", "debug", "debug",
                                                        "trace" };
    auto index = std::clamp(level + 1, 0, static_cast<int>(log_levels.size() - 1));
    data.env.set(env::LOG_LEVEL, log_levels[index]);
}

void
output_action(parser_t& parser, parser_data& data)
{
    auto values = parser.get<std::vector<std::string>>("output");
    data.env.set(env::OUTPUT_PATH, values.at(0));
    if(values.size() > 1) data.env.set(env::OUTPUT_PREFIX, values.at(1));
}

void
sample_action(parser_t& parser, parser_data& data)
{
    data.env.set(env::USE_SAMPLING, true);
    auto modes = parser.get<std::set<std::string>>("sample");
    if(modes.empty()) return;
    data.env.set(env::SAMPLING_CPUTIME, modes.count("cputime") > 0, update_mode::WEAK);
    data.env.set(env::SAMPLING_REALTIME, modes.count("realtime") > 0, update_mode::WEAK);
}

// --host and --device share the same three writes but differ in which secondary
// write is gated by the *primary* flag's enabled state. This asymmetry is
// inherited from the legacy code; preserved here.
void
apply_host_device(parser_data& data, bool host_enabled, bool device_enabled,
                  bool driving_flag_is_host)
{
    data.env.set(env::USE_PROCESS_SAMPLING, host_enabled || device_enabled);
    if(driving_flag_is_host)
    {
        data.env.set(env::CPU_FREQ_ENABLED, host_enabled);
        if(host_enabled) data.env.set(env::USE_AMD_SMI, device_enabled);
    }
    else
    {
        data.env.set(env::USE_AMD_SMI, device_enabled);
        if(device_enabled) data.env.set(env::CPU_FREQ_ENABLED, host_enabled);
    }
}

void
host_action(parser_t& parser, parser_data& data)
{
    apply_host_device(data, parser.get<bool>("host"), parser.get<bool>("device"),
                      /*driving_flag_is_host=*/true);
}

void
device_action(parser_t& parser, parser_data& data)
{
    apply_host_device(data, parser.get<bool>("host"), parser.get<bool>("device"),
                      /*driving_flag_is_host=*/false);
}

void
tids_action(parser_t& parser, parser_data& data)
{
    // Comma-space joiner is intentional and matches legacy behavior;
    // `--sample-cputime`/`--sample-realtime`/`--sample-overflow` use comma-only.
    auto tids = parser.get<std::vector<int64_t>>("tids");
    data.env.set(env::SAMPLING_TIDS, fmt::format("{}", fmt::join(tids, ", ")));
}

struct sample_consume_spec
{
    std::string_view flag_key;
    std::string_view enable_env;
    std::string_view first_env;
    std::string_view second_env;
    std::string_view tail_env;
    bool             check_overflow_event_conflict = false;
};

void
consume_sample_args(parser_t& parser, parser_data& data, const sample_consume_spec& spec)
{
    auto values = parser.get<std::deque<std::string>>(std::string{ spec.flag_key });
    data.env.set(spec.enable_env, true);

    if(!values.empty())
    {
        if(spec.check_overflow_event_conflict &&
           parser.exists("sampling-overflow-event") &&
           values.front() != parser.get<std::string>("sampling-overflow-event"))
        {
            throw exception<std::runtime_error>(fmt::format(
                "'--sample-overflow {} ...' conflicts with "
                "'--sampling-overflow-event {}' option",
                values.front(), parser.get<std::string>("sampling-overflow-event")));
        }
        data.env.set(spec.first_env, values.front());
        values.pop_front();
    }
    if(!values.empty())
    {
        data.env.set(spec.second_env, values.front());
        values.pop_front();
    }
    if(!values.empty())
        data.env.set(spec.tail_env, fmt::format("{}", fmt::join(values, ",")));
}

void
sample_cputime_action(parser_t& parser, parser_data& data)
{
    consume_sample_args(parser, data,
                        { "sample-cputime", env::SAMPLING_CPUTIME,
                          env::SAMPLING_CPUTIME_FREQ, env::SAMPLING_CPUTIME_DELAY,
                          env::SAMPLING_CPUTIME_TIDS });
}

void
sample_realtime_action(parser_t& parser, parser_data& data)
{
    consume_sample_args(parser, data,
                        { "sample-realtime", env::SAMPLING_REALTIME,
                          env::SAMPLING_REALTIME_FREQ, env::SAMPLING_REALTIME_DELAY,
                          env::SAMPLING_REALTIME_TIDS });
}

void
sample_overflow_action(parser_t& parser, parser_data& data)
{
    consume_sample_args(parser, data,
                        { "sample-overflow", env::SAMPLING_OVERFLOW,
                          env::SAMPLING_OVERFLOW_EVENT, env::SAMPLING_OVERFLOW_FREQ,
                          env::SAMPLING_OVERFLOW_TIDS,
                          /*check_overflow_event_conflict=*/true });
}

void
profile_format_action(parser_t& parser, parser_data& data)
{
    auto formats = parser.get<std::set<std::string>>("profile-format");
    data.env.set(env::PROFILE, true);
    if(formats.empty()) return;
    data.env.set(env::TEXT_OUTPUT, formats.count("text") != 0);
    data.env.set(env::JSON_OUTPUT, formats.count("json") != 0);
    data.env.set(env::COUT_OUTPUT, formats.count("console") != 0);
}

void
profile_diff_action(parser_t& parser, parser_data& data)
{
    auto values = parser.get<std::vector<std::string>>("profile-diff");
    data.env.set(env::DIFF_OUTPUT, true);
    data.env.set(env::INPUT_PATH, values.at(0));
    if(values.size() > 1) data.env.set(env::INPUT_PREFIX, values.at(1));
}

}  // namespace

// --- Group definitions (positional aggregate init; will switch to designated
//     initializers when project moves to C++20) ---

const flag_group&
debug_group()
{
    static const flag_group group{
        "DEBUG OPTIONS",
        "",
        {
            flag_descriptor{
                /* names       */ { "--log-level" },
                /* help        */ "Log level",
                /* dtype       */ "string",
                /* count       */ count_spec::at_most(1),
                /* kind        */ value_kind::scalar,
                /* join        */ join_with::none,
                /* env_vars    */ { env::LOG_LEVEL },
                /* mode        */ update_mode::REPLACE,
                /* dedup_keys  */ {},
                /* choices     */
                { "trace", "debug", "info", "warn", "error", "critical", "off" },
            },
            flag_descriptor{
                /* names       */ { "--monochrome" },
                /* help        */ "Disable colorized output",
                /* dtype       */ "bool",
                /* count       */ count_spec::at_most(1),
                /* kind        */ value_kind::flag,
                /* join        */ join_with::none,
                /* env_vars    */ {},
                /* mode        */ update_mode::REPLACE,
                /* dedup_keys  */ {},
                /* choices     */ {},
                /* conflicts   */ {},
                /* requires_   */ {},
                /* custom      */ &monochrome_action,
            },
            flag_descriptor{
                /* names       */ { "--debug" },
                /* help        */ "[DEPRECATED Use --log-level=debug] Debug output",
                /* dtype       */ {},
                /* count       */ count_spec::at_most(1),
                /* kind        */ value_kind::flag,
                /* join        */ join_with::none,
                /* env_vars    */ {},
                /* mode        */ update_mode::REPLACE,
                /* dedup_keys  */ {},
                /* choices     */ {},
                /* conflicts   */ {},
                /* requires_   */ {},
                /* custom      */ &debug_action,
            },
            flag_descriptor{
                /* names       */ { "-v", "--verbose" },
                /* help        */ "[DEPRECATED Use --log-level=trace] Verbose output",
                /* dtype       */ "integral",
                /* count       */ count_spec::exactly(1),
                /* kind        */ value_kind::scalar,
                /* join        */ join_with::none,
                /* env_vars    */ {},
                /* mode        */ update_mode::REPLACE,
                /* dedup_keys  */ {},
                /* choices     */ {},
                /* conflicts   */ {},
                /* requires_   */ {},
                /* custom      */ &verbose_action,
            },
        },
    };
    return group;
}

const flag_group&
general_group()
{
    static const flag_group group{
        "GENERAL OPTIONS",
        "These are options which are ubiquitously applied",
        {
            flag_descriptor{
                /* names       */ { "-c", "--config" },
                /* help        */ "Configuration file",
                /* dtype       */ "filepath",
                /* count       */ count_spec::at_least(1),
                /* kind        */ value_kind::list,
                /* join        */ join_with::colon,
                /* env_vars    */ { env::CONFIG_FILE },
                /* mode        */ update_mode::REPLACE,
                /* dedup_keys  */ { "config_file" },
            },
            flag_descriptor{
                /* names       */ { "-o", "--output" },
                /* help        */
                "Output path. Accepts 1-2 parameters corresponding to "
                "the output path and the output prefix",
                /* dtype       */ "path [prefix]",
                /* count       */ count_spec::range(1, 2),
                /* kind        */ value_kind::list,
                /* join        */ join_with::none,
                /* env_vars    */ {},
                /* mode        */ update_mode::REPLACE,
                /* dedup_keys  */ { "output_path", "output_prefix" },
                /* choices     */ {},
                /* conflicts   */ {},
                /* requires_   */ {},
                /* custom      */ &output_action,
            },
            flag_descriptor{
                /* names       */ { "-T", "--trace" },
                /* help        */ "Generate a detailed trace (perfetto output)",
                /* dtype       */ {},
                /* count       */ count_spec::at_most(1),
                /* kind        */ value_kind::flag,
                /* join        */ join_with::none,
                /* env_vars    */ { env::TRACE },
            },
            flag_descriptor{
                /* names       */ { "-L", "--trace-legacy" },
                /* help        */
                "Use legacy direct mode for tracing instead of "
                "deferred trace generation (higher overhead)",
                /* dtype       */ {},
                /* count       */ count_spec::at_most(1),
                /* kind        */ value_kind::flag,
                /* join        */ join_with::none,
                /* env_vars    */ { env::TRACE_LEGACY },
            },
            flag_descriptor{
                /* names       */ { "-P", "--profile" },
                /* help        */
                "Generate a call-stack-based profile (conflicts "
                "with --flat-profile)",
                /* dtype       */ {},
                /* count       */ count_spec::at_most(1),
                /* kind        */ value_kind::flag,
                /* join        */ join_with::none,
                /* env_vars    */ { env::PROFILE },
                /* mode        */ update_mode::REPLACE,
                /* dedup_keys  */ {},
                /* choices     */ {},
                /* conflicts   */ { "flat-profile" },
            },
            flag_descriptor{
                /* names       */ { "-F", "--flat-profile" },
                /* help        */ "Generate a flat profile (conflicts with --profile)",
                /* dtype       */ {},
                /* count       */ count_spec::at_most(1),
                /* kind        */ value_kind::flag,
                /* join        */ join_with::none,
                /* env_vars    */ { env::PROFILE, env::FLAT_PROFILE },
                /* mode        */ update_mode::REPLACE,
                /* dedup_keys  */ {},
                /* choices     */ {},
                /* conflicts   */ { "profile" },
            },
            flag_descriptor{
                /* names       */ { "-S", "--sample" },
                /* help        */ "Enable statistical sampling of call-stack",
                /* dtype       */ "timer-type",
                /* count       */ count_spec::range(0, 2),
                /* kind        */ value_kind::list,
                /* join        */ join_with::none,
                /* env_vars    */ {},
                /* mode        */ update_mode::REPLACE,
                /* dedup_keys  */ { "cpu_freq" },
                /* choices     */ { "cputime", "realtime" },
                /* conflicts   */ {},
                /* requires_   */ {},
                /* custom      */ &sample_action,
            },
            flag_descriptor{
                /* names       */ { "-H", "--host" },
                /* help        */
                "Enable sampling host-based metrics for the process. "
                "E.g. CPU frequency, memory usage, etc.",
                /* dtype       */ {},
                /* count       */ count_spec::at_most(1),
                /* kind        */ value_kind::flag,
                /* join        */ join_with::none,
                /* env_vars    */ {},
                /* mode        */ update_mode::REPLACE,
                /* dedup_keys  */ { "cpu_freq" },
                /* choices     */ {},
                /* conflicts   */ {},
                /* requires_   */ {},
                /* custom      */ &host_action,
            },
            flag_descriptor{
                /* names       */ { "-D", "--device" },
                /* help        */
                "Enable sampling device-based metrics for the "
                "process. E.g. GPU temperature, memory usage, etc.",
                /* dtype       */ {},
                /* count       */ count_spec::at_most(1),
                /* kind        */ value_kind::flag,
                /* join        */ join_with::none,
                /* env_vars    */ {},
                /* mode        */ update_mode::REPLACE,
                /* dedup_keys  */ { "amd_smi" },
                /* choices     */ {},
                /* conflicts   */ {},
                /* requires_   */ {},
                /* custom      */ &device_action,
            },
            flag_descriptor{
                /* names       */ { "-w", "--wait" },
                /* help        */
                "This option is a combination of '--trace-wait' and "
                "'--sampling-wait'. See the descriptions for those "
                "two options.",
                /* dtype       */ "seconds",
                /* count       */ count_spec::exactly(1),
                /* kind        */ value_kind::scalar_double,
                /* join        */ join_with::none,
                /* env_vars    */
                { env::TRACE_DELAY, env::SAMPLING_DELAY, env::CAUSAL_DELAY },
                /* mode        */ update_mode::WEAK,
            },
            flag_descriptor{
                /* names       */ { "-d", "--duration" },
                /* help        */
                "This option is a combination of '--trace-duration' "
                "and '--sampling-duration'. See the descriptions for "
                "those two options.",
                /* dtype       */ "seconds",
                /* count       */ count_spec::exactly(1),
                /* kind        */ value_kind::scalar_double,
                /* join        */ join_with::none,
                /* env_vars    */
                { env::TRACE_DURATION, env::SAMPLING_DURATION, env::CAUSAL_DURATION },
                /* mode        */ update_mode::WEAK,
            },
            flag_descriptor{
                /* names       */ { "--periods" },
                /* help        */
                "Similar to specifying delay and/or duration except "
                "in the form <DELAY>:<DURATION>, "
                "<DELAY>:<DURATION>:<REPEAT>, and/or "
                "<DELAY>:<DURATION>:<REPEAT>:<CLOCK_ID>",
                /* dtype       */ "period-spec(s)",
                /* count       */ count_spec::at_least(1),
                /* kind        */ value_kind::list,
                /* join        */ join_with::space,
                /* env_vars    */ { env::TRACE_PERIODS },
                /* mode        */ update_mode::WEAK,
            },
            flag_descriptor{
                /* names       */ { "--selected-regions" },
                /* help        */
                "Comma-separated list of roctx region names. When "
                "set, only activity inside matching roctx regions is "
                "traced (matched against roctxRangeStartA message)",
                /* dtype       */ "string",
                /* count       */ count_spec::exactly(1),
                /* kind        */ value_kind::scalar,
                /* join        */ join_with::none,
                /* env_vars    */ { env::SELECTED_REGIONS },
            },
            flag_descriptor{
                /* names       */ { "--rank-filter-id" },
                /* help        */
                "Sets the name of environment variable to read rank "
                "from for MPI output filtering",
                /* dtype       */ "string",
                /* count       */ count_spec::at_most(1),
                /* kind        */ value_kind::scalar,
                /* join        */ join_with::none,
                /* env_vars    */ { env::RANK_FILTER_ID },
                /* mode        */ update_mode::REPLACE,
                /* dedup_keys  */ {},
                /* choices     */ {},
                /* conflicts   */ {},
                /* requires_   */ { "rank-filter-output" },
            },
            flag_descriptor{
                /* names       */ { "--rank-filter-output" },
                /* help        */
                "Ranks for which file output is generated. Values "
                "should be separated by commas and can be explicit or "
                "ranges, e.g. 0,1,5-8. An empty value enables output "
                "for all ranks",
                /* dtype       */ "int and/or range",
                /* count       */ count_spec::at_most(1),
                /* kind        */ value_kind::list,
                /* join        */ join_with::comma,
                /* env_vars    */ { env::RANK_FILTER_OUTPUT },
            },
        },
    };
    return group;
}

const flag_group&
tracing_group()
{
    static const flag_group group{
        "TRACING OPTIONS",
        "Specific options controlling tracing (i.e. deterministic measurements of "
        "every event)",
        {
            flag_descriptor{
                /* names       */ { "--trace-file" },
                /* help        */
                "Specify the trace output filename. Relative filepath "
                "will be with respect to output path and output prefix.",
                /* dtype       */ "filepath",
                /* count       */ count_spec::exactly(1),
                /* kind        */ value_kind::scalar,
                /* join        */ join_with::none,
                /* env_vars    */ { env::PERFETTO_FILE },
                /* mode        */ update_mode::REPLACE,
                /* dedup_keys  */ { "perfetto_file" },
            },
            flag_descriptor{
                /* names       */ { "--trace-buffer-size" },
                /* help        */ "Size limit for the trace output (in KB)",
                /* dtype       */ "KB",
                /* count       */ count_spec::exactly(1),
                /* kind        */ value_kind::scalar_int,
                /* join        */ join_with::none,
                /* env_vars    */ { env::PERFETTO_BUFFER_SIZE_KB },
                /* mode        */ update_mode::REPLACE,
                /* dedup_keys  */ { "perfetto_buffer_size_kb" },
            },
            flag_descriptor{
                /* names       */ { "--trace-fill-policy" },
                /* help        */
                "Policy for new data when the buffer size limit is "
                "reached:\n"
                "    %{INDENT}%- discard     : new data is ignored\n"
                "    %{INDENT}%- ring_buffer : new data overwrites "
                "oldest data",
                /* dtype       */ "policy",
                /* count       */ count_spec::exactly(1),
                /* kind        */ value_kind::scalar,
                /* join        */ join_with::none,
                /* env_vars    */ { env::PERFETTO_FILL_POLICY },
                /* mode        */ update_mode::REPLACE,
                /* dedup_keys  */ { "perfetto_fill_policy" },
                /* choices     */ { "discard", "ring_buffer" },
            },
            flag_descriptor{
                /* names       */ { "--trace-wait" },
                /* help        */
                "Set the wait time (in seconds) before collecting trace "
                "and/or profiling data(in seconds). By default, the "
                "duration is in seconds of realtime but that can changed "
                "via --trace-clock-id.",
                /* dtype       */ "seconds",
                /* count       */ count_spec::exactly(1),
                /* kind        */ value_kind::scalar_double,
                /* join        */ join_with::none,
                /* env_vars    */ { env::TRACE_DELAY },
            },
            flag_descriptor{
                /* names       */ { "--trace-duration" },
                /* help        */
                "Set the duration of the trace and/or profile data "
                "collection (in seconds). By default, the duration is in "
                "seconds of realtime but that can changed via "
                "--trace-clock-id.",
                /* dtype       */ "seconds",
                /* count       */ count_spec::exactly(1),
                /* kind        */ value_kind::scalar_double,
                /* join        */ join_with::none,
                /* env_vars    */ { env::TRACE_DURATION },
            },
            flag_descriptor{
                /* names       */ { "--trace-periods" },
                /* help        */
                "More powerful version of specifying trace delay and/or "
                "duration. Format is one or more groups of: "
                "<DELAY>:<DURATION>, <DELAY>:<DURATION>:<REPEAT>, and/or "
                "<DELAY>:<DURATION>:<REPEAT>:<CLOCK_ID>.",
                /* dtype       */ "period-spec(s)",
                /* count       */ count_spec::at_least(1),
                /* kind        */ value_kind::list,
                /* join        */ join_with::comma,
                /* env_vars    */ { env::TRACE_PERIODS },
            },
        },
    };
    return group;
}

const flag_group&
profile_group()
{
    static const flag_group group{
        "PROFILE OPTIONS",
        "Specific options controlling profiling (i.e. deterministic measurements "
        "which are aggregated into a summary)",
        {
            flag_descriptor{
                /* names       */ { "--profile-format" },
                /* help        */ "Data formats for profiling results",
                /* dtype       */ "string",
                /* count       */ count_spec::range(1, 3),
                /* kind        */ value_kind::list,
                /* join        */ join_with::none,
                /* env_vars    */ {},
                /* mode        */ update_mode::REPLACE,
                /* dedup_keys  */ { "text_output", "json_output", "cout_output" },
                /* choices     */ { "text", "json", "console" },
                /* conflicts   */ {},
                /* requires_   */ { "profile|flat-profile" },
                /* custom      */ &profile_format_action,
            },
            flag_descriptor{
                /* names       */ { "--profile-diff" },
                /* help        */
                "Generate a diff output b/t the profile collected and "
                "an existing profile from another run Accepts 1-2 "
                "parameters corresponding to the input path and the "
                "input prefix",
                /* dtype       */ "path [prefix]",
                /* count       */ count_spec::range(1, 2),
                /* kind        */ value_kind::list,
                /* join        */ join_with::none,
                /* env_vars    */ {},
                /* mode        */ update_mode::REPLACE,
                /* dedup_keys  */ { "diff_output", "input_path", "input_prefix" },
                /* choices     */ {},
                /* conflicts   */ {},
                /* requires_   */ {},
                /* custom      */ &profile_diff_action,
            },
        },
    };
    return group;
}

const flag_group&
process_sampling_group()
{
    static const flag_group group{
        "HOST/DEVICE (PROCESS SAMPLING) OPTIONS",
        "Process sampling is background measurements for resources available to the "
        "entire process. These samples are not tied to specific lines/regions of code",
        {
            flag_descriptor{
                /* names       */ { "--process-freq" },
                /* help        */
                "Set the default host/device sampling frequency "
                "(number of interrupts per second)",
                /* dtype       */ "floating-point",
                /* count       */ count_spec::exactly(1),
                /* kind        */ value_kind::scalar_double,
                /* join        */ join_with::none,
                /* env_vars    */ { env::PROCESS_SAMPLING_FREQ },
                /* mode        */ update_mode::REPLACE,
                /* dedup_keys  */ { "process_sampling_freq" },
            },
            flag_descriptor{
                /* names       */ { "--process-wait" },
                /* help        */
                "Set the default wait time (i.e. delay) before taking "
                "first host/device sample (in seconds of realtime)",
                /* dtype       */ "seconds",
                /* count       */ count_spec::exactly(1),
                /* kind        */ value_kind::scalar_double,
                /* join        */ join_with::none,
                /* env_vars    */ { env::PROCESS_SAMPLING_DELAY },
                /* mode        */ update_mode::REPLACE,
                /* dedup_keys  */ { "process_sampling_delay" },
            },
            flag_descriptor{
                /* names       */ { "--process-duration" },
                /* help        */
                "Set the duration of the host/device sampling (in "
                "seconds of realtime)",
                /* dtype       */ "seconds",
                /* count       */ count_spec::exactly(1),
                /* kind        */ value_kind::scalar_double,
                /* join        */ join_with::none,
                /* env_vars    */ { env::SAMPLING_PROCESS_DURATION },
                /* mode        */ update_mode::REPLACE,
                /* dedup_keys  */ { "process_sampling_duration" },
            },
            flag_descriptor{
                /* names       */ { "--cpus" },
                /* help        */
                "CPU IDs for frequency sampling. Supports integers "
                "and/or ranges",
                /* dtype       */ "int and/or range",
                /* count       */ count_spec::any(),
                /* kind        */ value_kind::list,
                /* join        */ join_with::comma,
                /* env_vars    */ { env::SAMPLING_CPUS },
                /* mode        */ update_mode::REPLACE,
                /* dedup_keys  */ { "sampling_cpus" },
                /* choices     */ {},
                /* conflicts   */ {},
                /* requires_   */ { "host" },
            },
            flag_descriptor{
                /* names       */ { "--gpus" },
                /* help        */
                "GPU IDs for SMI queries. Supports integers and/or "
                "ranges",
                /* dtype       */ "int and/or range",
                /* count       */ count_spec::any(),
                /* kind        */ value_kind::list,
                /* join        */ join_with::comma,
                /* env_vars    */ { env::SAMPLING_GPUS },
                /* mode        */ update_mode::REPLACE,
                /* dedup_keys  */ { "sampling_gpus" },
                /* choices     */ {},
                /* conflicts   */ {},
                /* requires_   */ { "device" },
            },
            flag_descriptor{
                /* names       */ { "--ai-nics" },
                /* help        */
                "AI NIC IDs for SMI queries. Supports comma-separated "
                "list",
                /* dtype       */ "list of strings",
                /* count       */ count_spec::any(),
                /* kind        */ value_kind::list,
                /* join        */ join_with::comma,
                /* env_vars    */ { env::SAMPLING_AINICS },
                /* mode        */ update_mode::REPLACE,
                /* dedup_keys  */ { "sampling_ai-nics" },
                /* choices     */ {},
                /* conflicts   */ {},
                /* requires_   */ { "device" },
            },
        },
    };
    return group;
}

const flag_group&
general_sampling_group()
{
    static const flag_group group{
        "GENERAL SAMPLING OPTIONS",
        "General options for timer-based sampling per-thread",
        {
            flag_descriptor{
                /* names       */ { "-f", "--sampling-freq" },
                /* help        */
                "Set the default sampling frequency "
                "(number of interrupts per second)",
                /* dtype       */ "floating-point",
                /* count       */ count_spec::exactly(1),
                /* kind        */ value_kind::scalar_double,
                /* join        */ join_with::none,
                /* env_vars    */ { env::SAMPLING_FREQ },
            },
            flag_descriptor{
                /* names       */ { "-t", "--tids" },
                /* help        */
                "Specify the default thread IDs for sampling, where 0 "
                "(zero) is the main thread and each thread created by "
                "the target application is assigned an atomically "
                "incrementing value.",
                /* dtype       */ "int and/or range",
                /* count       */ count_spec::at_least(1),
                /* kind        */ value_kind::list,
                /* join        */ join_with::none,
                /* env_vars    */ {},
                /* mode        */ update_mode::REPLACE,
                /* dedup_keys  */ { "sampling_tids" },
                /* choices     */ {},
                /* conflicts   */ {},
                /* requires_   */ {},
                /* custom      */ &tids_action,
            },
            flag_descriptor{
                /* names       */ { "--sampling-wait" },
                /* help        */
                "Set the default wait time (i.e. delay) before taking "
                "first sample (in seconds). This delay time is based "
                "on the clock of the sampler, i.e., a delay of 1 "
                "second for CPU-clock sampler may not equal 1 second "
                "of realtime",
                /* dtype       */ "seconds",
                /* count       */ count_spec::exactly(1),
                /* kind        */ value_kind::scalar_double,
                /* join        */ join_with::none,
                /* env_vars    */ { env::SAMPLING_DELAY },
                /* mode        */ update_mode::REPLACE,
                /* dedup_keys  */ { "sampling_delay" },
            },
            flag_descriptor{
                /* names       */ { "--sampling-duration" },
                /* help        */
                "Set the duration of the sampling (in seconds of "
                "realtime). I.e., it is possible (currently) to set a "
                "CPU-clock time delay that exceeds the real-time "
                "duration... resulting in zero samples being taken",
                /* dtype       */ "seconds",
                /* count       */ count_spec::exactly(1),
                /* kind        */ value_kind::scalar_double,
                /* join        */ join_with::none,
                /* env_vars    */ { env::SAMPLING_DURATION },
            },
        },
    };
    return group;
}

const flag_group&
sampling_timer_group()
{
    static const flag_group group{
        "SAMPLING TIMER OPTIONS",
        "These options determine the heuristic for deciding when to take a sample",
        {
            flag_descriptor{
                /* names       */ { "--sample-cputime" },
                /* help        */ CPUTIME_HELP,
                /* dtype       */ "[freq] [delay] [tids...]",
                /* count       */ count_spec::at_least(0),
                /* kind        */ value_kind::list,
                /* join        */ join_with::none,
                /* env_vars    */ {},
                /* mode        */ update_mode::REPLACE,
                /* dedup_keys  */ {},
                /* choices     */ {},
                /* conflicts   */ {},
                /* requires_   */ {},
                /* custom      */ &sample_cputime_action,
            },
            flag_descriptor{
                /* names       */ { "--sample-realtime" },
                /* help        */ REALTIME_HELP,
                /* dtype       */ "[freq] [delay] [tids...]",
                /* count       */ count_spec::at_least(0),
                /* kind        */ value_kind::list,
                /* join        */ join_with::none,
                /* env_vars    */ {},
                /* mode        */ update_mode::REPLACE,
                /* dedup_keys  */ {},
                /* choices     */ {},
                /* conflicts   */ {},
                /* requires_   */ {},
                /* custom      */ &sample_realtime_action,
            },
            flag_descriptor{
                /* names       */ { "--sample-overflow" },
                /* help        */ OVERFLOW_HELP,
                /* dtype       */ "[event] [freq] [tids...]",
                /* count       */ count_spec::at_least(0),
                /* kind        */ value_kind::list,
                /* join        */ join_with::none,
                /* env_vars    */ {},
                /* mode        */ update_mode::REPLACE,
                /* dedup_keys  */ {},
                /* choices     */ {},
                /* conflicts   */ {},
                /* requires_   */ {},
                /* custom      */ &sample_overflow_action,
            },
        },
    };
    return group;
}

const flag_group&
hw_counter_group()
{
    static const flag_group group{
        "HARDWARE COUNTER OPTIONS",
        "See also: rocprof-sys-avail -H",
        {
            flag_descriptor{
                /* names       */ { "-C", "--cpu-events" },
                /* help        */
                "Set the CPU hardware counter events to record (ref: "
                "`rocprof-sys-avail -H -c CPU`)",
                /* dtype       */ "[EVENT ...]",
                /* count       */ count_spec::at_least(1),
                /* kind        */ value_kind::list,
                /* join        */ join_with::comma,
                /* env_vars    */ { env::PAPI_EVENTS },
                /* mode        */ update_mode::REPLACE,
                /* dedup_keys  */ { "papi_events" },
            },
            flag_descriptor{
                /* names       */ { "-G", "--gpu-events" },
                /* help        */
                "Set the GPU hardware counter events to record (ref: "
                "`rocprof-sys-avail -H -c GPU`)",
                /* dtype       */ "[EVENT ...]",
                /* count       */ count_spec::at_least(1),
                /* kind        */ value_kind::list,
                /* join        */ join_with::comma,
                /* env_vars    */ { env::ROCM_EVENTS },
                /* mode        */ update_mode::REPLACE,
                /* dedup_keys  */ { "rocm_events" },
            },
        },
    };
    return group;
}

const flag_group&
misc_group()
{
    static const flag_group group{
        "MISCELLANEOUS OPTIONS",
        "",
        {
            flag_descriptor{
                /* names       */ { "-i", "--inlines" },
                /* help        */ "Include inline info in output when available",
                /* dtype       */ {},
                /* count       */ count_spec::at_most(1),
                /* kind        */ value_kind::flag,
                /* join        */ join_with::none,
                /* env_vars    */ { env::SAMPLING_INCLUDE_INLINES },
                /* mode        */ update_mode::REPLACE,
                /* dedup_keys  */ { "sampling_include_inlines" },
            },
            // HSA_ENABLE_INTERRUPT is a non-ROCPROFSYS_ env var consumed by ROCm.
            flag_descriptor{
                /* names       */ { "--hsa-interrupt" },
                /* help        */ HSA_INTERRUPT_HELP,
                /* dtype       */ "int",
                /* count       */ count_spec::exactly(1),
                /* kind        */ value_kind::scalar_int,
                /* join        */ join_with::none,
                /* env_vars    */ { "HSA_ENABLE_INTERRUPT" },
                /* mode        */ update_mode::REPLACE,
                /* dedup_keys  */ {},
                /* choices     */ { "0", "1" },
            },
        },
    };
    return group;
}

}  // namespace argparse
}  // namespace rocprofsys
