/*************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#ifndef RCCL_TEST_HOST_SCOPED_DIR_H_
#define RCCL_TEST_HOST_SCOPED_DIR_H_

#include <fmt/format.h>
#include <stdlib.h>

#include <cerrno>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <string>
#include <string_view>
#include <utility>

// A private temporary directory that removes itself and everything under it, including after a
// fatal ASSERT_* (which returns before any cleanup written after it). Prefer this over a
// hand-rolled mkdtemp: those leave the directory behind on a failing assertion, which is exactly
// the run whose leftovers nobody wanted.
//
// *prefix* names the directory, so a leftover after a crash, or a failure mentioning the path,
// points back at the suite that made it; the mkdtemp template suffix belongs to ScopedDir and is
// not the caller's to supply. Deliberately has no default: a shared helper with one is a helper
// every caller leaves unnamed.
//
// The directory is created under std::filesystem::temp_directory_path() (honouring TMPDIR), not
// a hardcoded /tmp, so a test run with a TMPDIR of its own keeps its files there.
//
// *base* overrides that. A test that has to EXECUTE something it wrote needs it: TMPDIR is
// mounted noexec on some developer machines and in some containers, so such tests pass the build
// directory (RCCL_HOST_TEST_BUILD_DIR) instead.
class ScopedDir {
public:
    explicit ScopedDir(std::string_view prefix, std::string_view base = {})
    {
        const std::filesystem::path parent =
            base.empty() ? std::filesystem::temp_directory_path() : std::filesystem::path(base);
        const std::string tmpl = (parent / (std::string(prefix) + "-XXXXXX")).string();
        std::string buffer = tmpl;
        // Dies rather than hands back an empty path, which would mean remove_all("") in the
        // destructor and silently working-directory-relative paths in the test body. Reported
        // from *tmpl*: mkdtemp leaves *buffer* undefined, so its name is not a path anyone tried.
        if (mkdtemp(buffer.data()) == nullptr) {
            fmt::print(stderr, "ScopedDir: mkdtemp(\"{}\") failed: {}\n", tmpl,
                       std::strerror(errno));
            std::abort();
        }
        path_ = std::move(buffer);
    }
    ~ScopedDir()
    {
        std::error_code ec;
        std::filesystem::remove_all(path_, ec);
    }

    // Non-copyable and non-movable: two objects owning one path would remove it twice, the
    // second time after something else may have taken the name.
    ScopedDir(const ScopedDir &)            = delete;
    ScopedDir &operator=(const ScopedDir &) = delete;
    ScopedDir(ScopedDir &&)                 = delete;
    ScopedDir &operator=(ScopedDir &&)      = delete;

    // Never empty: the constructor does not return without a directory.
    const std::string &path() const { return path_; }

private:
    std::string path_;
};

// Skip the current test when *dir* does not exist. A ScopedDir under a missing *base* aborts the
// whole binary, and RCCL_HOST_TEST_BUILD_DIR is missing wherever the tests run from an installed
// package rather than the build tree.
#define RCCL_SKIP_WITHOUT_DIRECTORY(dir)                                \
    do {                                                                \
        if (!std::filesystem::is_directory(dir)) {                      \
            GTEST_SKIP() << (dir) << " does not exist: no build tree";  \
        }                                                               \
    } while (false)

#endif  // RCCL_TEST_HOST_SCOPED_DIR_H_
