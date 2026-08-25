/* Copyright (c) Advanced Micro Devices, Inc. All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 */

#include "context.h"
#include "configuration.h"
#include "hipfile-warnings.h"
#include "hipfile.h"

#include "io-scenario.h"
#include "io-test.h"
#include "io-verify.h"
#include "test-common.h"
#include "test-options.h"

#include <cstdint>
#include <gtest/gtest.h>
#include <hip/hip_runtime_api.h>
#include <string>
#include <tuple>
#include <unistd.h>
#include <vector>

extern SystemTestOptions test_env;

using namespace hipFile;
using namespace hipFileTest;

HIPFILE_WARN_NO_GLOBAL_CTOR_OFF

// ---------------------------------------------------------------------------
// This test suite exercises the ability of hipFileRead and hipFileWrite while varying the size of I/O and the
// backend that is used to fulfill the I/O request. Additionally, we test the ability of hipFileRead and
// hipFileWrite to correctly transfer data when data is modified at a byte granularity.
//
// We verify hipFileRead and hipFileWrite behave as expected by guarding the targeted regions of the
// device memory allocation and the region of the file that data will be read from and written to, surrounding
// them with poisoned memory containing sentinel values that would tell us if hipFile ever read or wrote data
// to a location that the user did not specify.
// ---------------------------------------------------------------------------
struct HipFileVerifyBytes : public DataModificationBase<ByteElementPolicy> {
    void SetUp() override
    {
        // File layout (each sentinel region 4_KiB, data io_bytes; data begins at file
        // offset GetParam().file_off):
        // [head file sentinel region][data][tail file sentinel region]
        const hoff_t tail_off = GetParam().file_off + static_cast<hoff_t>(io_bytes);
        ASSERT_EQ(0, ftruncate(tmpfile.fd, tail_off + static_cast<hoff_t>(4_KiB)));
        DataModificationBase::SetUp();
    }
};

TEST_P(HipFileVerifyBytes, RoundTripGuardsAllRegions)
{
    ASSERT_NO_FATAL_FAILURE(runAllRegionsTest(*this));
}

INSTANTIATE_TEST_SUITE_P(, HipFileVerifyBytes,
                         testing::ValuesIn(IoTestScenarioSet{IoTestScenario{.file_off = kCombinedFileOff,
                                                                            .buf_off  = kFourKiBOff,
                                                                            .stride   = 2}}
                                               .over(&IoTestScenario::backend, kBackends)
                                               .over(&IoTestScenario::io_bytes, kCombinedSizes)
                                               .build()),
                         ioTestScenarioName);

HIPFILE_WARN_NO_GLOBAL_CTOR_ON
