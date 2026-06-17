/* Copyright (c) Advanced Micro Devices, Inc. All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 */

#include "io.h"
#include "hipfile-test.h"
#include "hipfile-warnings.h"
#include "mbackend.h"
#include "mbuffer.h"
#include "mfile.h"

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include <memory>

using namespace hipFile;

using ::testing::AnyNumber;
using ::testing::Return;
using ::testing::StrictMock;
using ::testing::Throw;

// Put tests inside the macros to suppress the global constructor
// warnings
HIPFILE_WARN_NO_GLOBAL_CTOR_OFF

struct DummyBackendWithFallback : public MBackendWithFallback {};

struct DummyFallbackBackend : MBackend {};

struct HipFileBackendWithFallback : public HipFileUnopened, ::testing::WithParamInterface<IoType> {
    std::shared_ptr<StrictMock<MBuffer>> mock_buffer;
    std::shared_ptr<StrictMock<MFile>>   mock_file;

    std::shared_ptr<StrictMock<DummyBackendWithFallback>> default_backend;
    std::shared_ptr<StrictMock<DummyFallbackBackend>>     fallback_backend;

    IoType  io_type{IoType::Read};
    ssize_t successful_io_size{0x1234};

    void SetUp() override
    {
        mock_buffer = std::make_shared<StrictMock<MBuffer>>();
        mock_file   = std::make_shared<StrictMock<MFile>>();

        default_backend = std::make_shared<StrictMock<DummyBackendWithFallback>>();
        EXPECT_CALL(*default_backend, _io_impl).Times(AnyNumber());
        EXPECT_CALL(*default_backend, is_fallback_eligible).WillRepeatedly(Return(true));

        fallback_backend = std::make_shared<StrictMock<DummyFallbackBackend>>();
        EXPECT_CALL(*fallback_backend, io).Times(AnyNumber());
        EXPECT_CALL(*fallback_backend, score).WillRepeatedly(Return(0));

        io_type = GetParam();
    }

    HipFileBackendWithFallback()
    {
    }
};

TEST_P(HipFileBackendWithFallback, IOSuccess)
{
    EXPECT_CALL(*default_backend, _io_impl).WillOnce(Return(successful_io_size));

    ssize_t nbytes = default_backend->io(io_type, mock_file, mock_buffer, 0, 0, 0);

    ASSERT_EQ(nbytes, successful_io_size);
}

TEST_P(HipFileBackendWithFallback, IOFailureWithNoRegisteredFallback)
{
    EXPECT_CALL(*default_backend, _io_impl).WillOnce(Throw(std::runtime_error("IO failure")));

    EXPECT_THROW(default_backend->io(io_type, mock_file, mock_buffer, 0, 0, 0), std::runtime_error);
}

TEST_P(HipFileBackendWithFallback, IOFailureWithRegisteredFallbackButNotFallbackEligible)
{
    // The Backend has registered a fallback, but the BackendWithFallback has
    // determined based on the error that this IO should not be retried.
    default_backend->register_fallback_backend(fallback_backend);
    EXPECT_CALL(*default_backend, _io_impl).WillOnce(Throw(std::runtime_error("IO failure")));
    EXPECT_CALL(*default_backend, is_fallback_eligible).WillOnce(Return(false));

    EXPECT_THROW(default_backend->io(io_type, mock_file, mock_buffer, 0, 0, 0), std::runtime_error);
}

TEST_P(HipFileBackendWithFallback, IOFailureWithRegisteredFallbackRefusingTheIO)
{
    // The Backend has registered a fallback, but the fallback backend has
    // determined that it cannot process the IO.
    default_backend->register_fallback_backend(fallback_backend);
    EXPECT_CALL(*default_backend, _io_impl).WillOnce(Throw(std::runtime_error("IO failure")));
    EXPECT_CALL(*fallback_backend, score).WillOnce(Return(-1));

    EXPECT_THROW(default_backend->io(io_type, mock_file, mock_buffer, 0, 0, 0), std::runtime_error);
}

TEST_P(HipFileBackendWithFallback, IOFailureWithGoodFallback)
{
    default_backend->register_fallback_backend(fallback_backend);
    EXPECT_CALL(*default_backend, _io_impl).WillOnce(Throw(std::runtime_error("IO failure")));
    EXPECT_CALL(*fallback_backend, io).WillOnce(Return(successful_io_size));

    ssize_t nbytes = default_backend->io(io_type, mock_file, mock_buffer, 0, 0, 0);
    ASSERT_EQ(nbytes, successful_io_size);
}

INSTANTIATE_TEST_SUITE_P(, HipFileBackendWithFallback, ::testing::Values(IoType::Read, IoType::Write));

struct HipFileBackendWithFallbackRegisterFallback : public HipFileUnopened {};

TEST_F(HipFileBackendWithFallbackRegisterFallback, RegisterNullptrFallback)
{
    auto backend = std::make_shared<DummyBackendWithFallback>();
    ASSERT_THROW(backend->register_fallback_backend(nullptr), std::invalid_argument);
}

TEST_F(HipFileBackendWithFallbackRegisterFallback, RegisterSelfAsAFallback)
{
    auto backend = std::make_shared<DummyBackendWithFallback>();
    ASSERT_THROW(backend->register_fallback_backend(backend), std::invalid_argument);
}

HIPFILE_WARN_NO_GLOBAL_CTOR_ON
