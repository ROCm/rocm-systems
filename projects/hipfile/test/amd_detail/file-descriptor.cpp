/* Copyright (c) Advanced Micro Devices, Inc. All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 */

#include "file-descriptor.h"
#include "hipfile-warnings.h"
#include "msys.h"

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include <memory>
#include <utility>

using namespace hipFile;
using namespace testing;

HIPFILE_WARN_NO_GLOBAL_CTOR_OFF

struct HipFileFileDescriptor : public Test {
    StrictMock<MSys> msys;
};

TEST_F(HipFileFileDescriptor, ManagedFileDescriptorClosesOnDestruction)
{
    auto fd{FileDescriptor::make_managed(42)};
    EXPECT_CALL(msys, close(42));
}

TEST_F(HipFileFileDescriptor, ManagedFileDescriptorDoesNotCloseOnDestructionIfFdIsNegativeOne)
{
    auto fd{FileDescriptor::make_managed(-1)};
}

TEST_F(HipFileFileDescriptor, UnmanagedFileDescriptorDoesNotCloseOnDestruction)
{
    auto fd{FileDescriptor::make_unmanaged(42)};
}

TEST_F(HipFileFileDescriptor, ManagedFileDescriptorMoveCopyConstructor)
{
    auto fd_a{FileDescriptor::make_managed(42)};
    auto fd_b{std::move(fd_a)};

    // coverity[USE_AFTER_MOVE]
    ASSERT_EQ(fd_a.get(), -1); // NOLINT(clang-analyzer-cplusplus.Move)
    ASSERT_EQ(fd_b.get(), 42);
    EXPECT_CALL(msys, close(42));
}

TEST_F(HipFileFileDescriptor, UnmanagedFileDescriptorMoveCopyConstructor)
{
    auto fd_a{FileDescriptor::make_unmanaged(42)};
    auto fd_b{std::move(fd_a)};

    // coverity[USE_AFTER_MOVE]
    ASSERT_EQ(fd_a.get(), -1); // NOLINT(clang-analyzer-cplusplus.Move)
    ASSERT_EQ(fd_b.get(), 42);
}

TEST_F(HipFileFileDescriptor, ManagedFileDescriptorToManagedFileDescriptorMoveAssignment)
{
    auto fd_a{FileDescriptor::make_managed(42)};
    auto fd_b{FileDescriptor::make_managed(12)};
    EXPECT_CALL(msys, close(12));
    fd_b = std::move(fd_a);

    // coverity[USE_AFTER_MOVE]
    ASSERT_EQ(fd_a.get(), -1); // NOLINT(clang-analyzer-cplusplus.Move)
    ASSERT_EQ(fd_b.get(), 42);
    EXPECT_CALL(msys, close(42));
}

TEST_F(HipFileFileDescriptor, ManagedFileDescriptorToUnmanagedFileDescriptorMoveAssignment)
{
    auto fd_a{FileDescriptor::make_managed(42)};
    auto fd_b{FileDescriptor::make_unmanaged(12)};
    fd_b = std::move(fd_a);

    // coverity[USE_AFTER_MOVE]
    ASSERT_EQ(fd_a.get(), -1); // NOLINT(clang-analyzer-cplusplus.Move)
    ASSERT_EQ(fd_b.get(), 42);
    EXPECT_CALL(msys, close(42));
}

TEST_F(HipFileFileDescriptor, UnmanagedFileDescriptorToManagedFileDescriptorMoveAssignment)
{
    auto fd_a{FileDescriptor::make_unmanaged(42)};
    auto fd_b{FileDescriptor::make_managed(12)};
    EXPECT_CALL(msys, close(12));
    fd_b = std::move(fd_a);

    // coverity[USE_AFTER_MOVE]
    ASSERT_EQ(fd_a.get(), -1); // NOLINT(clang-analyzer-cplusplus.Move)
    ASSERT_EQ(fd_b.get(), 42);
}

TEST_F(HipFileFileDescriptor, UnmanagedFileDescriptorToUnmanagedFileDescriptorMoveAssignment)
{
    auto fd_a{FileDescriptor::make_unmanaged(42)};
    auto fd_b{FileDescriptor::make_unmanaged(12)};
    fd_b = std::move(fd_a);

    // coverity[USE_AFTER_MOVE]
    ASSERT_EQ(fd_a.get(), -1); // NOLINT(clang-analyzer-cplusplus.Move)
    ASSERT_EQ(fd_b.get(), 42);
}

HIPFILE_WARN_NO_GLOBAL_CTOR_ON
