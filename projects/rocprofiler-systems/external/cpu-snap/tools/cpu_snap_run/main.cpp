// cpu-snap-run: run any program under the CPU sampler.
//
//   cpu-snap-run [options] -- <command> [args...]
//
// Sets LD_PRELOAD to the sampler library plus the CALLSTACK_* configuration and
// execs the command, so the program keeps its pid, exit code and signals. Every
// process of the tree that loads the library writes its own Perfetto trace.

#include <cerrno>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <string>
#include <string_view>
#include <vector>

#include <unistd.h>

#include "launch_plan.hpp"

#ifndef CPU_SNAP_RUN_DEFAULT_PRELOAD
#    define CPU_SNAP_RUN_DEFAULT_PRELOAD ""
#endif

int
main(int argc, char** argv)
{
    const std::vector<std::string_view> args(argv + 1, argv + argc);

    // NOLINTBEGIN(concurrency-mt-unsafe): single-threaded launcher
    const char* env_preload = std::getenv("CPU_SNAP_PRELOAD");
    const char* existing    = std::getenv("LD_PRELOAD");
    // NOLINTEND(concurrency-mt-unsafe)

    const auto plan = cpu_snap::run::make_launch_plan(
        args, static_cast<std::uint32_t>(::getpid()),
        env_preload != nullptr ? env_preload : CPU_SNAP_RUN_DEFAULT_PRELOAD,
        existing != nullptr ? existing : "");

    if(plan.help)
    {
        std::fputs(cpu_snap::run::usage().data(), stdout);
        return EXIT_SUCCESS;
    }
    if(!plan.error.empty())
    {
        std::fprintf(stderr, "cpu-snap-run: %s\n\n%s", plan.error.c_str(),
                     cpu_snap::run::usage().data());
        return 2;
    }
    if(!std::filesystem::exists(plan.preload))
    {
        std::fprintf(stderr, "cpu-snap-run: sampler library not found: %s\n", plan.preload.c_str());
        return 2;
    }

    for(const auto& [key, value] : plan.environment)
    {
        ::setenv(key.c_str(), value.c_str(), 1);  // NOLINT(concurrency-mt-unsafe)
    }

    std::vector<char*> command;
    command.reserve(plan.command.size() + 1);
    for(const auto& word : plan.command)
    {
        command.push_back(const_cast<char*>(word.c_str()));  // NOLINT(cppcoreguidelines-pro-type-const-cast): execvp takes char* const[]
    }
    command.push_back(nullptr);

    ::execvp(command[0], command.data());
    std::fprintf(stderr, "cpu-snap-run: cannot run '%s': %s\n", command[0], std::strerror(errno));
    return 127;
}
