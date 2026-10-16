/* Copyright (c) Advanced Micro Devices, Inc. All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 */

/* Verify hipFile reports hipFileGetNewFDFailed when the process runs out of
 * file descriptors.
 *
 * These tests lower this process's soft RLIMIT_NOFILE, restoring it when each
 * test finishes. They assume each test runs in its own process, as it does
 * under ctest: running the whole binary directly lets earlier tests initialize
 * hipFile first, which invalidates the fd limits chosen below.
 */

#include "hipfile-warnings.h"
#include "hipfile.h"

#include "test-common.h"
#include "test-options.h"
#include "test-shared-fixtures.h"

#include <cerrno>
#include <memory>
#include <ostream>
#include <sys/resource.h>
#include <system_error>
#include <vector>

extern SystemTestOptions test_env;

namespace {

// Lowers the soft RLIMIT_NOFILE, restoring the original limits on destruction
class ScopedFdLimit {
public:
    explicit ScopedFdLimit(rlim_t soft_limit)
    {
        if (getrlimit(RLIMIT_NOFILE, &m_saved) != 0) {
            throw std::system_error(errno, std::generic_category(), "getrlimit");
        }
        rlimit lowered{m_saved};
        lowered.rlim_cur = soft_limit;
        if (setrlimit(RLIMIT_NOFILE, &lowered) != 0) {
            throw std::system_error(errno, std::generic_category(), "setrlimit");
        }
    }

    ~ScopedFdLimit()
    {
        setrlimit(RLIMIT_NOFILE, &m_saved);
    }

    ScopedFdLimit(const ScopedFdLimit &)            = delete;
    ScopedFdLimit &operator=(const ScopedFdLimit &) = delete;

private:
    rlimit m_saved{};
};

struct FdLimitParams {
    const char    *name;
    int            file_count;
    rlim_t         fd_limit;
    hipFileError_t expected_err;
};

// Readable test parameter in gtest/ctest output, also used as the test name
void
PrintTo(const FdLimitParams &params, std::ostream *os)
{
    *os << params.name;
}

}

HIPFILE_WARN_NO_GLOBAL_CTOR_OFF

struct HipFileFdLimit : public DriverInit, public testing::WithParamInterface<FdLimitParams> {};

// Registration needs an extra fd per file (hipFile reopens each file), so the
// parameters are chosen to be well clear of the exact point of exhaustion.
TEST_P(HipFileFdLimit, RegisterFiles)
{
    const FdLimitParams &params{GetParam()};
    ScopedFdLimit        fd_limit{params.fd_limit};

    std::vector<std::unique_ptr<Tmpfile>> files;
    for (int i{0}; i < params.file_count; i++) {
        files.push_back(std::make_unique<Tmpfile>(test_env.ais_capable_dir));
    }

    ASSERT_EQ(hipFileDriverOpen(), HIPFILE_SUCCESS);

    std::vector<hipFileHandle_t> handles;
    hipFileError_t               err{HIPFILE_SUCCESS};
    for (const auto &file : files) {
        hipFileDescr_t descr{};
        descr.type      = hipFileHandleTypeOpaqueFD;
        descr.handle.fd = file->fd;

        hipFileHandle_t handle{};
        err = hipFileHandleRegister(&handle, &descr);
        if (!(err == HIPFILE_SUCCESS)) {
            break;
        }
        handles.push_back(handle);
    }

    for (hipFileHandle_t handle : handles) {
        hipFileHandleDeregister(handle);
    }

    EXPECT_EQ(err, params.expected_err);
}

// On file registration, hipFile will open up a new fd to the same file that
// is being registered, just with slightly different flags. As such, to
// register N files with hipfile, there must be at least another N fds
// available.
// This does not consider any other file descriptors that hipFile may have
// opened internally.
const FdLimitParams fd_limit_params[] = {
    {"Files10Limit50", 10, 50, HIPFILE_SUCCESS},
    {"Files30Limit50", 30, 50, HipFileOpError(hipFileGetNewFDFailed)},
};

INSTANTIATE_TEST_SUITE_P(, HipFileFdLimit, testing::ValuesIn(fd_limit_params),
                         testing::PrintToStringParamName());

// A limit of 0 leaves no fds for the stats server hipFile creates on first use
TEST(HipFileFdLimitDriverOpen, DriverOpenOutOfFds)
{
    {
        ScopedFdLimit fd_limit{0};
        ASSERT_EQ(hipFileDriverOpen(), HipFileOpError(hipFileGetNewFDFailed));
    }

    // With fds available again the driver opens normally
    ASSERT_EQ(hipFileDriverOpen(), HIPFILE_SUCCESS);
    ASSERT_EQ(hipFileDriverClose(), HIPFILE_SUCCESS);
}

HIPFILE_WARN_NO_GLOBAL_CTOR_ON
