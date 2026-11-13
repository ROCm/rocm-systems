/*************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#ifndef RCCL_TEST_HOST_EXECUTABLE_SCRATCH_H_
#define RCCL_TEST_HOST_EXECUTABLE_SCRATCH_H_

#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <string>

#include <unistd.h>

// Where a test may write a file and then RUN it.
//
// TMPDIR is the obvious answer and the wrong one: it is mounted noexec on some developer
// machines and in many containers, and a test that execs a fixture it wrote there fails for a
// reason that has nothing to do with the code under test. The candidates below are tried in
// order and the first that actually works is used, so such a machine runs these tests out of the
// build tree instead of skipping them:
//
//   1. RCCL_HOST_TEST_SCRATCH_DIR, for a caller that knows where its writable-and-executable
//      space is (a CI image with an unusual mount layout, say);
//   2. the build directory this binary was built in, which must allow execution -- the compiler
//      just wrote and the test runner just ran binaries there; and
//   3. TMPDIR, which is the right answer wherever it is not mounted noexec.
//
// Empty when none of them allows it, which callers report as a skip rather than a failure.
namespace rccl_host_test {

namespace detail {

// Whether a file written into *directory* can be executed.
inline bool allowsExecution(const std::filesystem::path &directory)
{
    std::error_code ec;
    if (directory.empty() || !std::filesystem::is_directory(directory, ec)) {
        return false;
    }
    const std::filesystem::path probe =
        directory / ("rccl-exec-probe-" + std::to_string(::getpid()));
    {
        std::ofstream script(probe);
        if (!script) {
            return false;
        }
        script << "#!/bin/sh\nexit 0\n";
    }
    std::filesystem::permissions(probe, std::filesystem::perms::owner_all,
                                 std::filesystem::perm_options::add, ec);
    // access(X_OK) and not an exec: it answers the mount-option question without a fork, and
    // the kernel applies the same rule to both.
    const bool executable = ::access(probe.c_str(), X_OK) == 0;
    std::filesystem::remove(probe, ec);
    return executable;
}

}  // namespace detail

// The first candidate directory that allows execution, or empty when none does.
inline const std::string &executableScratchDirectory()
{
    static const std::string chosen = [] {
        std::error_code ec;
        const char *configured = std::getenv("RCCL_HOST_TEST_SCRATCH_DIR");
        if (configured != nullptr && *configured != '\0' &&
            detail::allowsExecution(configured)) {
            return std::string(configured);
        }
#ifdef RCCL_HOST_TEST_BUILD_DIR
        if (detail::allowsExecution(RCCL_HOST_TEST_BUILD_DIR)) {
            return std::string(RCCL_HOST_TEST_BUILD_DIR);
        }
#endif
        const std::filesystem::path temp = std::filesystem::temp_directory_path(ec);
        if (!ec && detail::allowsExecution(temp)) {
            return temp.string();
        }
        return std::string();
    }();
    return chosen;
}

}  // namespace rccl_host_test

#endif  // RCCL_TEST_HOST_EXECUTABLE_SCRATCH_H_
