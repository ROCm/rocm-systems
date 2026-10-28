#pragma once

/**
 * @file launch_plan.hpp
 * @brief Turns cpu-snap-run's command line into the environment and command
 *        to exec. Pure logic, no I/O, so it can be unit tested.
 *
 * cpu-snap-run samples a program by preloading the autoload library into it;
 * the library is configured through CALLSTACK_* environment variables, which
 * this file derives from the command-line options.
 */

#include <algorithm>
#include <bit>
#include <charconv>
#include <cstdint>
#include <span>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace cpu_snap::run
{

/** Everything cpu-snap-run needs to launch the target. */
struct launch_plan
{
    /** Variables to set before exec, in order. */
    std::vector<std::pair<std::string, std::string>> environment;
    /** The program and its arguments. */
    std::vector<std::string> command;
    /** Library that is preloaded into the target. */
    std::string preload;
    /** Non-empty when the arguments are invalid. */
    std::string error;
    /** True when --help was requested. */
    bool help = false;
};

/** @return The help text. */
[[nodiscard]] inline std::string_view
usage() noexcept
{
    return R"(Usage: cpu-snap-run [options] [--] <command> [args...]

Runs <command> with the CPU sampler preloaded and writes one Perfetto trace per
process (the program and every child it starts). Processes with no samples
write nothing.

Options:
  -f, --freq=HZ         sample frequency per thread (default 1000)
  -b, --backend=NAME    auto | perf | signal (default auto: perf when permitted, else signal)
  -u, --unwind=MODE     dwarf | fp | auto (default dwarf; fp fails on binaries
                        built without frame pointers)
      --stack-bytes=N   user-stack snapshot per sample for DWARF (default 32768)
      --pages=N         perf ring pages per thread (default: derived from the snapshot)
      --depth=N         maximum frames per sample (default 256)
  -o, --output=PATH     trace path, %p = pid, %e = process name
                        (default cpu-snap.%e.%p.pftrace)
      --no-merge        one slice stack per sample instead of merging consecutive samples
      --filter-stdlib   drop C++ standard library frames
      --no-children     sample only the program itself, not the processes it starts
      --preload=PATH    sampler library to preload (default: CPU_SNAP_PRELOAD or the built one)
  -h, --help            show this help

Examples:
  cpu-snap-run -- ./my_app --its-flag
  cpu-snap-run --backend signal --freq 2000 -o /tmp/run.%p.pftrace -- python3 script.py
  cpu-snap-run -- rocprof-sys-run -- ./transpose
)";
}

namespace detail
{

struct parsed_options
{
    std::uint32_t freq_hz     = 1000;
    std::string   backend     = "auto";
    std::string   unwind      = "dwarf";
    std::uint32_t stack_bytes = 32768;
    std::uint32_t depth       = 256;
    std::string   output      = "cpu-snap.%e.%p.pftrace";
    bool          merge       = true;
    bool          filter_stdlib = false;
    bool          children    = true;
    std::string   preload;
    std::uint32_t pages       = 0;
};

[[nodiscard]] inline bool
parse_u32(std::string_view text, std::uint32_t& out) noexcept
{
    const auto [ptr, ec] = std::from_chars(text.data(), text.data() + text.size(), out);
    return ec == std::errc{} && ptr == text.data() + text.size() && !text.empty();
}

/** Perf ring pages able to hold several DWARF snapshots (power of two, at least 8). */
[[nodiscard]] constexpr std::uint32_t
pages_for_snapshot(std::uint32_t stack_bytes) noexcept
{
    constexpr std::uint32_t k_page_bytes        = 4096;
    constexpr std::uint32_t k_snapshots_in_ring = 32;
    const std::uint64_t     wanted = std::uint64_t{ stack_bytes } * k_snapshots_in_ring / k_page_bytes;
    return std::bit_ceil(static_cast<std::uint32_t>(std::max<std::uint64_t>(8, wanted)));
}

struct parse_result
{
    std::size_t command_begin = 0;
    std::string error;
    bool        help = false;
};

/** Applies the option at args[i] to @p opt, consuming a following value word by advancing @p i. */
[[nodiscard]] inline std::string
apply_option(parsed_options& opt, std::span<const std::string_view> args, std::size_t& i)
{
    const std::string_view arg = args[i];
    if(arg == "--no-merge") { opt.merge = false; return {}; }
    if(arg == "--filter-stdlib") { opt.filter_stdlib = true; return {}; }
    if(arg == "--no-children") { opt.children = false; return {}; }

    std::string_view name = arg;
    std::string_view value;
    bool             has_value = false;
    if(const auto eq = arg.find('='); eq != std::string_view::npos && arg.starts_with("--"))
    {
        name      = arg.substr(0, eq);
        value     = arg.substr(eq + 1);
        has_value = true;
    }
    const auto take_value = [&]() -> bool {
        if(has_value) return true;
        if(i + 1 >= args.size()) return false;
        value     = args[++i];
        has_value = true;
        return true;
    };
    const auto number = [&](std::uint32_t& target) -> bool {
        return take_value() && parse_u32(value, target);
    };

    if(name == "-f" || name == "--freq")
    {
        if(!number(opt.freq_hz) || opt.freq_hz == 0) return "--freq needs a positive number";
    }
    else if(name == "--stack-bytes")
    {
        if(!number(opt.stack_bytes) || opt.stack_bytes == 0 || opt.stack_bytes % 8 != 0)
            return "--stack-bytes needs a positive multiple of 8";
    }
    else if(name == "--pages")
    {
        if(!number(opt.pages) || !std::has_single_bit(opt.pages)) return "--pages needs a power of two";
    }
    else if(name == "--depth")
    {
        if(!number(opt.depth) || opt.depth == 0) return "--depth needs a positive number";
    }
    else if(name == "-b" || name == "--backend")
    {
        if(!take_value() || (value != "auto" && value != "perf" && value != "signal"))
            return "--backend must be auto, perf or signal";
        opt.backend = std::string(value);
    }
    else if(name == "-u" || name == "--unwind")
    {
        if(!take_value() || (value != "dwarf" && value != "fp" && value != "auto"))
            return "--unwind must be dwarf, fp or auto";
        opt.unwind = std::string(value);
    }
    else if(name == "-o" || name == "--output")
    {
        if(!take_value() || value.empty()) return "--output needs a path";
        opt.output = std::string(value);
    }
    else if(name == "--preload")
    {
        if(!take_value() || value.empty()) return "--preload needs a path";
        opt.preload = std::string(value);
    }
    else
    {
        return "unknown option: " + std::string(arg);
    }
    return {};
}

[[nodiscard]] inline parse_result
parse_options(std::span<const std::string_view> args, parsed_options& opt)
{
    parse_result result;
    std::size_t& i = result.command_begin;
    for(; i < args.size(); ++i)
    {
        const std::string_view arg = args[i];
        if(arg == "--")
        {
            ++i;
            break;
        }
        if(arg.empty() || arg.front() != '-')
        {
            break;
        }
        if(arg == "-h" || arg == "--help")
        {
            result.help = true;
            return result;
        }
        result.error = apply_option(opt, args, i);
        if(!result.error.empty())
        {
            return result;
        }
    }
    return result;
}

[[nodiscard]] inline std::vector<std::pair<std::string, std::string>>
build_environment(const parsed_options& opt, std::uint32_t launcher_pid, std::string_view existing_ld_preload)
{
    std::vector<std::pair<std::string, std::string>> env;
    const auto set = [&](std::string key, std::string value) {
        env.emplace_back(std::move(key), std::move(value));
    };
    std::string ld_preload = opt.preload;
    if(!existing_ld_preload.empty())
    {
        ld_preload += ':';
        ld_preload += existing_ld_preload;
    }
    set("LD_PRELOAD", std::move(ld_preload));
    set("CALLSTACK_FREQ_HZ", std::to_string(opt.freq_hz));
    set("CALLSTACK_BACKEND", opt.backend);
    set("CALLSTACK_UNWIND", opt.unwind);
    set("CALLSTACK_STACK_BYTES", std::to_string(opt.stack_bytes));
    set("CALLSTACK_DEPTH", std::to_string(opt.depth));
    set("CALLSTACK_TRACE", opt.output);
    set("CALLSTACK_MERGE", opt.merge ? "1" : "0");
    set("CALLSTACK_FILTER_STDLIB", opt.filter_stdlib ? "1" : "0");
    if(opt.pages != 0)
    {
        set("CALLSTACK_PAGES", std::to_string(opt.pages));
    }
    else if(opt.backend != "signal" && opt.unwind == "dwarf")
    {
        set("CALLSTACK_PAGES", std::to_string(pages_for_snapshot(opt.stack_bytes)));
    }
    if(!opt.children)
    {
        set("CALLSTACK_ONLY_PID", std::to_string(launcher_pid));
    }
    return env;
}

}  // namespace detail

/**
 * @brief Parse @p args (without argv[0]) into a launch_plan.
 *
 * @param args                Command-line arguments after the program name.
 * @param launcher_pid        Pid of the process that will exec the target (same pid
 *                            after exec); used by --no-children.
 * @param default_preload     Sampler library used when --preload is not given.
 * @param existing_ld_preload Current LD_PRELOAD; kept after ours.
 */
[[nodiscard]] inline launch_plan
make_launch_plan(std::span<const std::string_view> args, std::uint32_t launcher_pid,
                 std::string_view default_preload, std::string_view existing_ld_preload)
{
    launch_plan            plan;
    detail::parsed_options opt;
    opt.preload = std::string(default_preload);

    const auto parsed = detail::parse_options(args, opt);
    plan.help         = parsed.help;
    plan.error        = parsed.error;
    if(plan.help || !plan.error.empty())
    {
        return plan;
    }

    for(std::size_t i = parsed.command_begin; i < args.size(); ++i)
    {
        plan.command.emplace_back(args[i]);
    }
    if(plan.command.empty())
    {
        plan.error = "no command given";
    }
    else if(opt.preload.empty())
    {
        plan.error = "no sampler library found: pass --preload or set CPU_SNAP_PRELOAD";
    }
    if(!plan.error.empty())
    {
        return plan;
    }

    plan.preload     = opt.preload;
    plan.environment = detail::build_environment(opt, launcher_pid, existing_ld_preload);
    return plan;
}

}  // namespace cpu_snap::run
