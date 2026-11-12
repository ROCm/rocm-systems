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
class ScopedDir {
public:
    explicit ScopedDir(std::string_view prefix)
    {
        const std::string tmpl =
            (std::filesystem::temp_directory_path() / (std::string(prefix) + "-XXXXXX")).string();
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

#endif  // RCCL_TEST_HOST_SCOPED_DIR_H_
