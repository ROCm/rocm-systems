/* Copyright (c) Advanced Micro Devices, Inc. All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 */

// Runs a stress test program unless fastpath is unavailable and the caller
// allows skipping it
//
//   hipfile-fastpath-gate [--allow-skip-fastpath] DIRECTORY -- PROGRAM [ARGS...]
//
// Without --allow-skip-fastpath, the gate runs PROGRAM right away. With it, a
// child process first runs the system tests' fastpath check on a temporary
// file in DIRECTORY. If fastpath is unavailable, the gate prints a message
// that ctest's SKIP_REGULAR_EXPRESSION matches, and exits without running
// PROGRAM. Otherwise PROGRAM replaces the gate's process, so the test's exit
// status is PROGRAM's.

#include "ais-capability.h"
#include "args.h"
#include "hipfile-literals.h"
#include "hipfile-warnings.h"
#include "hipfile.h"
#include "test-common.h"
#include "test-options.h"

#include <algorithm>
#include <cerrno>
#include <cstdlib>
#include <cstring>
#include <hip/hip_runtime_api.h>
#include <iostream>
#include <iterator>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <sys/types.h>
#include <sys/wait.h>
#include <unistd.h>
#include <vector>

namespace {

/// @brief Size of the GPU buffer the fastpath check uses
constexpr size_t PROBE_BUFFER_SIZE{1_MiB};

/// @brief Exit status of the check process when fastpath is unavailable
constexpr int FASTPATH_UNAVAILABLE{2};

/// @brief Exit status when PROGRAM cannot be run, as shells report it
constexpr int COMMAND_NOT_RUN{127};

/// @brief Owns the registered file handle and the GPU buffer of the fastpath check
struct ProbeResources {
    /// @brief Registered handle for the temporary file, or nullptr
    hipFileHandle_t handle{nullptr};
    /// @brief GPU buffer from hipMalloc, or nullptr
    void *buffer{nullptr};

    /// @brief Creates an object that owns nothing yet
    ProbeResources() = default;
    /// @brief Not copyable, since the object owns the handle and buffer
    ProbeResources(const ProbeResources &) = delete;
    /// @brief Not copyable, since the object owns the handle and buffer
    ProbeResources &operator=(const ProbeResources &) = delete;

    /// @brief Frees the GPU buffer and deregisters the file handle
    ~ProbeResources()
    {
        if (buffer) {
            (void)hipFree(buffer);
        }
        if (handle) {
            hipFileHandleDeregister(handle);
        }
    }
};

/// @brief Joins a command and its arguments with spaces, for messages
///
/// @param command The command and its arguments
///
/// @return The command line
std::string
commandText(std::span<char *const> command)
{
    std::string text;
    for (const char *arg : command) {
        if (!text.empty()) {
            text += ' ';
        }
        text += arg;
    }
    return text;
}

/// @brief Checks whether fastpath is available for files in a directory
///
/// Runs the system tests' check: AisCapability examines the kernel, the HIP
/// runtime, and amdgpu, then reads zero bytes from a registered temporary file
/// in the directory. Prints the skip message when fastpath is unavailable.
///
/// @param directory Directory for the temporary file
/// @param command PROGRAM and its arguments, for the skip message
///
/// @return EXIT_SUCCESS if fastpath is available, FASTPATH_UNAVAILABLE if it
///         is not, or EXIT_FAILURE if the check cannot run
int
checkFastpath(const std::string &directory, std::span<char *const> command)
{
    // The probe read must fail when fastpath can't serve it, whatever backends
    // the caller's environment allows
    if (setenv("HIPFILE_ALLOW_COMPAT_MODE", "false", 1) != 0 || unsetenv("HIPFILE_FORCE_COMPAT_MODE") != 0) {
        std::cerr << "Unable to set up the environment for the fastpath check\n";
        return EXIT_FAILURE;
    }

    try {
        Tmpfile        tmpfile{directory};
        hipFileDescr_t descr{};
        descr.type      = hipFileHandleTypeOpaqueFD;
        descr.handle.fd = tmpfile.fd;

        ProbeResources resources;
        if (hipFileHandleRegister(&resources.handle, &descr).err != hipFileSuccess) {
            resources.handle = nullptr;
            std::cerr << "hipFileHandleRegister failed for a temporary file in " << directory << '\n';
            return EXIT_FAILURE;
        }
        if (hipMalloc(&resources.buffer, PROBE_BUFFER_SIZE) != hipSuccess) {
            resources.buffer = nullptr;
            std::cerr << "hipMalloc failed\n";
            return EXIT_FAILURE;
        }

        hipFile::test::AisCapability capability{true};
        if (capability.populate(resources.handle, resources.buffer) ==
            hipFile::test::AisCapability::GateDecision::Run) {
            return EXIT_SUCCESS;
        }
        // Keep this marker synchronized with SKIP_REGULAR_EXPRESSION in test/amd_detail/CMakeLists.txt.
        std::cout << "Fast path is not available on " << directory << ", skipping " << commandText(command)
                  << '\n'
                  << capability.report() << '\n';
        return FASTPATH_UNAVAILABLE;
    }
    catch (const std::runtime_error &e) {
        std::cerr << e.what() << '\n';
        return EXIT_FAILURE;
    }
}

/// @brief Runs the fastpath check in a child process
///
/// The process that later runs PROGRAM never initializes HIP or hipFile.
///
/// @param directory Directory for the check's temporary file
/// @param command PROGRAM and its arguments, for the skip message
///
/// @return The check's result, as checkFastpath() returns it, or EXIT_FAILURE
///         if the child process could not run or did not exit normally
int
checkFastpathInChild(const std::string &directory, std::span<char *const> command)
{
    std::cout.flush();
    std::cerr.flush();
    const pid_t pid{fork()};
    if (pid == -1) {
        const int err{errno};
        std::cerr << "fork failed: " << std::strerror(err) << '\n';
        return EXIT_FAILURE;
    }
    if (pid == 0) {
        std::exit(checkFastpath(directory, command));
    }

    int status{};
    while (waitpid(pid, &status, 0) == -1) {
        if (errno != EINTR) {
            const int err{errno};
            std::cerr << "waitpid failed: " << std::strerror(err) << '\n';
            return EXIT_FAILURE;
        }
    }
    if (!WIFEXITED(status)) {
        std::cerr << "The fastpath check did not exit normally\n";
        return EXIT_FAILURE;
    }
    return WEXITSTATUS(status);
}

}

/// @brief Parses the options, runs the fastpath check if asked, and runs PROGRAM
///
/// @param argc Number of command-line arguments
/// @param argv Command-line arguments
///
/// @return EXIT_SUCCESS if PROGRAM was skipped, EXIT_FAILURE if the
///         arguments or the fastpath check failed, or COMMAND_NOT_RUN if
///         PROGRAM could not be run. Otherwise PROGRAM replaces this process.
int
main(int argc, char **argv)
{
    // main's argv is a C array, so a span of it can only come from a pointer and a length
    HIPFILE_WARN_UNSAFE_BUFFER_USAGE_OFF
    const std::span<char *> arguments{argv, static_cast<size_t>(argc)};
    HIPFILE_WARN_UNSAFE_BUFFER_USAGE_ON
    const auto separator{std::find_if(arguments.begin(), arguments.end(), [](const char *argument) {
        return std::string_view{argument} == "--";
    })};
    const auto options_end{separator - arguments.begin()};

    args::ArgumentParser parser{
        "Runs PROGRAM, unless --allow-skip-fastpath is given and fastpath is unavailable for files in "
        "DIRECTORY",
        "Usage: hipfile-fastpath-gate [--allow-skip-fastpath] DIRECTORY -- PROGRAM [ARGS...]"};
    [[maybe_unused]] args::HelpFlag help_arg(parser, "help", "Show this help message",
                                             args::Matcher{'h', "help"});

    args::Flag allow_skip_arg(parser, "allow-skip-fastpath", "Skip PROGRAM when fastpath is unavailable",
                              args::Matcher{"allow-skip-fastpath"});
    args::Positional<std::string> directory_arg(parser, "DIRECTORY", "Directory for the fastpath check",
                                                args::Options::Required);
    parseCli(parser, static_cast<int>(options_end), argv);

    if (separator == arguments.end() || std::next(separator) == arguments.end()) {
        std::cerr << "Expected -- PROGRAM [ARGS...] after the options\n";
        return EXIT_FAILURE;
    }
    const auto command{arguments.subspan(static_cast<size_t>(options_end) + 1)};

    if (allow_skip_arg) {
        const int result{checkFastpathInChild(args::get(directory_arg), command)};
        if (result == FASTPATH_UNAVAILABLE) {
            return EXIT_SUCCESS;
        }
        if (result != EXIT_SUCCESS) {
            return EXIT_FAILURE;
        }
    }

    std::vector<char *> exec_arguments{command.begin(), command.end()};
    exec_arguments.push_back(nullptr);
    std::cout.flush();
    std::cerr.flush();
    execvp(exec_arguments.front(), exec_arguments.data());
    const int err{errno};
    std::cerr << "Unable to run " << exec_arguments.front() << ": " << std::strerror(err) << '\n';
    return COMMAND_NOT_RUN;
}
