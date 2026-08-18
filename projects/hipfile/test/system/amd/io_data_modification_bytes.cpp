/* Copyright (c) Advanced Micro Devices, Inc. All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 */

#include "context.h"
#include "configuration.h"
#include "hipfile-warnings.h"
#include "hipfile.h"

#include "io.hpp"
#include "io-verify.hpp"
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
struct HipFileVerifyBytes : public testing::TestWithParam<std::tuple<IoTestParam, SizeParam>> {

    Tmpfile         tmpfile;
    hipFileHandle_t tmpfile_handle{nullptr};
    void           *device_buffer{nullptr};
    size_t          io_bytes{0};     // bytes transferred using the hipFile API
    size_t          buffer_bytes{0}; // total device buffer size

    HipFileVerifyBytes() : tmpfile{test_env.ais_capable_dir}
    {
    }

    IoTestBackend backend() const
    {
        return std::get<0>(GetParam()).backend;
    }

    void SetUp() override
    {
        Context<Configuration>::get()->fastpath(false);
        Context<Configuration>::get()->fallback(false);

        switch (backend()) {
            case IoTestBackend::Fastpath:
                Context<Configuration>::get()->fastpath(true);
                break;
            case IoTestBackend::Fallback:
                Context<Configuration>::get()->fallback(true);
                break;
            default:
                FAIL() << "Unsupported IoTestBackend";
        }

        io_bytes = std::get<1>(GetParam()).bytes;
        // The data modification kernel will also verify that the sentinel regions of the device buffer are
        // unmodified. Device layout (each sentinel region 4_KiB, data io_bytes):
        // [head device sentinel region][data][tail device sentinel region]
        buffer_bytes = io_bytes + 2 * 4_KiB;

        // File layout (each sentinel region 4_KiB, data io_bytes; data begins at file
        // offset kCombinedFileOff):
        // [head file sentinel region][data][tail file sentinel region]
        const hoff_t tail_off = kCombinedFileOff + static_cast<hoff_t>(io_bytes);
        ASSERT_EQ(0, ftruncate(tmpfile.fd, tail_off + static_cast<hoff_t>(4_KiB)));

        hipFileDescr_t descr{};
        descr.type      = hipFileHandleTypeOpaqueFD;
        descr.handle.fd = tmpfile.fd;
        ASSERT_EQ(HIPFILE_SUCCESS, hipFileHandleRegister(&tmpfile_handle, &descr));

        ASSERT_EQ(hipSuccess, hipMalloc(&device_buffer, buffer_bytes));
        ASSERT_EQ(hipSuccess, hipMemset(device_buffer, kByteDevSlack, buffer_bytes));
        // hipMemset is not synchronous w.r.t. the host, and hipFileRead is not ordered w.r.t. the stream.
        ASSERT_EQ(hipSuccess, hipDeviceSynchronize());

        if (backend() == IoTestBackend::Fastpath) {
            enforceFastpathGate(tmpfile_handle, device_buffer);
        }
    }

    void TearDown() override
    {
        if (device_buffer) {
            ASSERT_EQ(hipSuccess, hipFree(device_buffer));
        }
        hipFileHandleDeregister(tmpfile_handle);
    }

    uint8_t *bufferStart() const
    {
        return static_cast<uint8_t *>(device_buffer);
    }
};

TEST_P(HipFileVerifyBytes, RoundTripGuardsAllRegions)
{
    const size_t n           = io_bytes; // one byte per element
    const size_t slack_n     = 4_KiB;    // file sentinel region bracket size in bytes
    const hoff_t buf_off     = static_cast<hoff_t>(4_KiB);
    const hoff_t file_off    = kCombinedFileOff;
    const hoff_t head_off    = file_off - static_cast<hoff_t>(4_KiB); // head file sentinel region byte offset
    const hoff_t tail_off    = file_off + static_cast<hoff_t>(io_bytes);
    constexpr size_t kStride = 2; // every-other byte

    // File layout (each sentinel region slack_n bytes, data n bytes; data begins at
    // file offset file_off past the chunk boundary):
    // [unwritten sparse hole = 0][head file sentinel region = 0x55][data = 0xFF][tail file sentinel region =
    // 0x55].
    seedFileBytesConstant(tmpfile.fd, head_off, slack_n, kByteFileSlack);
    seedFileBytesConstant(tmpfile.fd, file_off, n, kByteEntry);
    seedFileBytesConstant(tmpfile.fd, tail_off, slack_n, kByteFileSlack);

    ASSERT_EQ(static_cast<ssize_t>(io_bytes),
              hipFileRead(tmpfile_handle, device_buffer, io_bytes, file_off, buf_off));

    // Device layout (each sentinel region 4_KiB, data n bytes; data begins at buffer
    // offset buf_off = 4_KiB):
    // [head device sentinel region][data][tail device sentinel region]
    uint8_t *data = bufferStart() + 4_KiB;
    assertVerifyAndModifyBytes(bufferStart(), buffer_bytes, data, n, defaultGrid(n),
                               dim3(kDefaultWorkgroupSize), kStride);

    ASSERT_EQ(static_cast<ssize_t>(io_bytes),
              hipFileWrite(tmpfile_handle, device_buffer, io_bytes, file_off, buf_off));

    assertHoleZero(tmpfile.fd, 0, head_off);
    std::vector<uint8_t> head = readFileBytes(tmpfile.fd, head_off, slack_n);
    assertBytesConstant(head.data(), 0, slack_n, kByteFileSlack);
    std::vector<uint8_t> body = readFileBytes(tmpfile.fd, file_off, n);
    assertBytesModified(body.data(), n, kStride);
    std::vector<uint8_t> tail = readFileBytes(tmpfile.fd, tail_off, slack_n);
    assertBytesConstant(tail.data(), 0, slack_n, kByteFileSlack);
}

static std::string
byteName(const testing::TestParamInfo<HipFileVerifyBytes::ParamType> &info)
{
    return std::get<0>(info.param).name + "_" + std::get<1>(info.param).name;
}

INSTANTIATE_TEST_SUITE_P(, HipFileVerifyBytes,
                         testing::Combine(testing::ValuesIn(io_test_params),
                                          testing::ValuesIn(combined_sizes)),
                         byteName);

HIPFILE_WARN_NO_GLOBAL_CTOR_ON
