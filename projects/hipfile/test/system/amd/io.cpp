/* Copyright (c) Advanced Micro Devices, Inc. All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 */

#include "context.h"
#include "configuration.h"
#include "hipfile-warnings.h"
#include "hipfile.h"

#include "test-common.h"
#include "test-options.h"

#include <array>
#include <cstdlib>
#include <gtest/gtest.h>
#include <hip/hip_runtime_api.h>
#include <string>
#include <unistd.h>

extern SystemTestOptions test_env;

using namespace hipFile;

HIPFILE_WARN_NO_GLOBAL_CTOR_OFF

enum class IoTestBackend {
    Fastpath,
    Fallback,
};

struct IoTestParam {
    IoTestBackend backend;
    std::string   name;
};

HIPFILE_WARN_NO_EXIT_DTOR_OFF
static std::array<IoTestParam, 2> io_test_params{
    {{IoTestBackend::Fastpath, "Fastpath"}, {IoTestBackend::Fallback, "Fallback"}}};
HIPFILE_WARN_NO_EXIT_DTOR_ON

struct HipFileIo : public testing::TestWithParam<IoTestParam> {

    Tmpfile         tmpfile;
    size_t          tmpfile_size;
    hipFileHandle_t tmpfile_handle;
    void           *unregistered_device_buffer;
    size_t          unregistered_device_buffer_size;

    HipFileIo()
        : tmpfile{test_env.ais_capable_dir}, tmpfile_size{1024 * 1024}, tmpfile_handle{nullptr},
          unregistered_device_buffer{nullptr}, unregistered_device_buffer_size{1024 * 1024}
    {
    }

    void SetUp() override
    {
        // Disable all backends
        Context<Configuration>::get()->fastpath(false);
        Context<Configuration>::get()->fallback(false);

        // Enable the desired backend
        switch (GetParam().backend) {
            case IoTestBackend::Fastpath:
                Context<Configuration>::get()->fastpath(true);
                break;

            case IoTestBackend::Fallback:
                Context<Configuration>::get()->fallback(true);
                break;

            default:
                FAIL() << "Unsupported IoTestBackend";
        }

        ASSERT_EQ(0, ftruncate(tmpfile.fd, static_cast<off_t>(tmpfile_size)));

        hipFileDescr_t descr{};
        descr.type      = hipFileHandleTypeOpaqueFD;
        descr.handle.fd = tmpfile.fd;

        ASSERT_EQ(HIPFILE_SUCCESS, hipFileHandleRegister(&tmpfile_handle, &descr));

        ASSERT_EQ(hipSuccess, hipMalloc(&unregistered_device_buffer, unregistered_device_buffer_size));
    }

    void TearDown() override
    {
        ASSERT_EQ(hipSuccess, hipFree(unregistered_device_buffer));

        hipFileHandleDeregister(tmpfile_handle);
    }
};

TEST_P(HipFileIo, ReadToUnregisteredBufferAtOffset)
{
    hoff_t io_buffer_offset{4096};
    size_t io_size{unregistered_device_buffer_size - static_cast<size_t>(io_buffer_offset)};

    ASSERT_EQ(io_size, hipFileRead(tmpfile_handle, unregistered_device_buffer, io_size, 0, io_buffer_offset));
}

TEST_P(HipFileIo, ReadToUnregisteredBufferAtOffsetReturnsErrorIfOverflow)
{
    hoff_t io_buffer_offset{4096};
    size_t io_size{unregistered_device_buffer_size};

    ASSERT_EQ(-hipFileInvalidValue,
              hipFileRead(tmpfile_handle, unregistered_device_buffer, io_size, 0, io_buffer_offset));
}

INSTANTIATE_TEST_SUITE_P(, HipFileIo, testing::ValuesIn(io_test_params),
                         [](const testing::TestParamInfo<HipFileIo::ParamType> &param_info) {
                             return param_info.param.name;
                         });

HIPFILE_WARN_NO_GLOBAL_CTOR_ON
