/* Copyright (c) Advanced Micro Devices, Inc. All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 */

#include "buffer.h"
#include "context.h"
#include "hip.h"
#include "hipfile.h"
#include "hipfile-test.h"
#include "hipfile-warnings.h"
#include "mhip.h"
#include "state.h"

#include <array>
#include <cstddef>
#include <cstdint>
#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include <hip/hip_runtime_api.h>
#include <limits>
#include <memory>
#include <stdexcept>

using namespace hipFile;

using ::testing::StrictMock;

// Put tests inside the macros to suppress the global constructor
// warnings
HIPFILE_WARN_NO_GLOBAL_CTOR_OFF

void
expect_buffer_registration(MHip &mhip, hipMemoryType memory_type)
{
    hipPointerAttribute_t attrs{};
    attrs.type = memory_type;
    HipMemAddressRange range{reinterpret_cast<void *>(0x1), UINT64_MAX - 1};
    EXPECT_CALL(mhip, hipPointerGetAttributes).WillOnce(testing::Return(attrs));
    EXPECT_CALL(mhip, hipMemGetAddressRange).WillOnce(testing::Return(range));
}

struct HipFileBuffer : public HipFileOpened {
    HipFileBuffer() : nonnull_ptr(reinterpret_cast<void *>(0x1))
    {
    }
    void *nonnull_ptr;
};

TEST_F(HipFileBuffer, register_internal_supported_hip_memory)
{
    for (const auto memoryType : SupportedHipMemoryTypes) {
        StrictMock<MHip> mhip;
        expect_buffer_registration(mhip, memoryType);
        Context<DriverState>::get()->registerBuffer(nonnull_ptr, 0, 0);
    }
}

TEST_F(HipFileBuffer, register_supported_hip_memory)
{
    for (const auto memoryType : SupportedHipMemoryTypes) {
        StrictMock<MHip> mhip;
        expect_buffer_registration(mhip, memoryType);
        ASSERT_EQ(hipFileBufRegister(nonnull_ptr, 0, 0), HIPFILE_SUCCESS);
    }
}

TEST_F(HipFileBuffer, register_internal_unsupported_hip_memory)
{
    for (const auto memoryType : UnsupportedHipMemoryTypes) {
        StrictMock<MHip>      mhip;
        hipPointerAttribute_t attrs{};
        attrs.type = memoryType;
        EXPECT_CALL(mhip, hipPointerGetAttributes).WillOnce(testing::Return(attrs));
        ASSERT_THROW(Context<DriverState>::get()->registerBuffer(nonnull_ptr, 0, 0), InvalidMemoryType);
    }
}

TEST_F(HipFileBuffer, register_unsupported_hip_memory)
{
    for (const auto memoryType : UnsupportedHipMemoryTypes) {
        StrictMock<MHip>      mhip;
        hipPointerAttribute_t attrs{};
        attrs.type = memoryType;
        EXPECT_CALL(mhip, hipPointerGetAttributes).WillOnce(testing::Return(attrs));
        ASSERT_EQ(hipFileBufRegister(nonnull_ptr, 0, 0), HipFileOpError(hipFileHipMemoryTypeInvalid));
    }
}

TEST_F(HipFileBuffer, register_internal_hip_pointer_get_attributes_error)
{
    StrictMock<MHip> mhip;
    EXPECT_CALL(mhip, hipPointerGetAttributes).WillOnce(testing::Throw(Hip::RuntimeError(hipErrorUnknown)));
    ASSERT_THROW(Context<DriverState>::get()->registerBuffer(nonnull_ptr, 0, 0), Hip::RuntimeError);
}

TEST_F(HipFileBuffer, register_hip_pointer_get_attributes_error)
{
    StrictMock<MHip> mhip;
    EXPECT_CALL(mhip, hipPointerGetAttributes).WillOnce(testing::Throw(Hip::RuntimeError(hipErrorUnknown)));
    ASSERT_EQ(hipFileBufRegister(nonnull_ptr, 0, 0), HipFileHipError(hipErrorUnknown));

    // hipErrorInvalidValue is handled differently to match the behaviour of cufile
    EXPECT_CALL(mhip, hipPointerGetAttributes)
        .WillOnce(testing::Throw(Hip::RuntimeError(hipErrorInvalidValue)));
    ASSERT_EQ(hipFileBufRegister(nonnull_ptr, 0, 0), HipFileOpError(hipFileInvalidValue));
}

TEST_F(HipFileBuffer, register_internal_already_registered)
{
    StrictMock<MHip> mhip;
    expect_buffer_registration(mhip, hipMemoryTypeDevice);
    ASSERT_EQ(hipFileBufRegister(nonnull_ptr, 0, 0), HIPFILE_SUCCESS);
    ASSERT_THROW(Context<DriverState>::get()->registerBuffer(nonnull_ptr, 0, 0), BufferAlreadyRegistered);
}

TEST_F(HipFileBuffer, register_already_registered)
{
    StrictMock<MHip> mhip;
    expect_buffer_registration(mhip, hipMemoryTypeDevice);
    ASSERT_EQ(hipFileBufRegister(nonnull_ptr, 0, 0), HIPFILE_SUCCESS);
    ASSERT_EQ(hipFileBufRegister(nonnull_ptr, 0, 0), HipFileOpError(hipFileMemoryAlreadyRegistered));
}

TEST_F(HipFileBuffer, registerNullPointerReturnsError)
{
    ASSERT_EQ(hipFileBufRegister(nullptr, 0x10000, 0), HipFileOpError(hipFileInvalidValue));
}

TEST_F(HipFileBuffer, registerOversizeRangeReturnsError)
{
    StrictMock<MHip>   mhip;
    HipMemAddressRange range{nonnull_ptr, 100};
    EXPECT_CALL(mhip, hipMemGetAddressRange).WillOnce(testing::Return(range));
    hipPointerAttribute_t attrs{};
    attrs.type = hipMemoryTypeDevice;
    EXPECT_CALL(mhip, hipPointerGetAttributes).WillOnce(testing::Return(attrs));
    ASSERT_EQ(hipFileBufRegister(reinterpret_cast<void *>(0x1), 101, 0),
              HipFileOpError(hipFileHipPointerRangeError));
}

TEST_F(HipFileBuffer, registerRangeEndingAtUintptrMaxSucceeds)
{
    StrictMock<MHip>   mhip;
    const uintptr_t    base_addr = std::numeric_limits<uintptr_t>::max() - 99;
    void              *buffer    = reinterpret_cast<void *>(base_addr);
    HipMemAddressRange range{buffer, 100};
    EXPECT_CALL(mhip, hipMemGetAddressRange).WillOnce(testing::Return(range));
    hipPointerAttribute_t attrs{};
    attrs.type = hipMemoryTypeDevice;
    EXPECT_CALL(mhip, hipPointerGetAttributes).WillOnce(testing::Return(attrs));
    ASSERT_EQ(hipFileBufRegister(buffer, 100, 0), HIPFILE_SUCCESS);
}

TEST_F(HipFileBuffer, registerPointerBeforeAddressRangeBaseReturnsError)
{
    StrictMock<MHip>   mhip;
    void              *buffer = reinterpret_cast<void *>(0x1FFF);
    HipMemAddressRange range{reinterpret_cast<void *>(0x2000), 0x100};
    EXPECT_CALL(mhip, hipMemGetAddressRange).WillOnce(testing::Return(range));
    hipPointerAttribute_t attrs{};
    attrs.type = hipMemoryTypeDevice;
    EXPECT_CALL(mhip, hipPointerGetAttributes).WillOnce(testing::Return(attrs));
    ASSERT_EQ(hipFileBufRegister(buffer, 1, 0), HipFileOpError(hipFileHipPointerRangeError));
}

TEST_F(HipFileBuffer, registerPointerAtOffsetPastEndReturnsError)
{
    StrictMock<MHip>   mhip;
    void              *buffer = reinterpret_cast<void *>(0x2100);
    HipMemAddressRange range{reinterpret_cast<void *>(0x2000), 0x100};
    EXPECT_CALL(mhip, hipMemGetAddressRange).WillOnce(testing::Return(range));
    hipPointerAttribute_t attrs{};
    attrs.type = hipMemoryTypeDevice;
    EXPECT_CALL(mhip, hipPointerGetAttributes).WillOnce(testing::Return(attrs));
    ASSERT_EQ(hipFileBufRegister(buffer, 1, 0), HipFileOpError(hipFileHipPointerRangeError));
}

TEST_F(HipFileBuffer, registerHipMemGetAddressRangeThrowReturnsError)
{
    StrictMock<MHip> mhip;
    EXPECT_CALL(mhip, hipMemGetAddressRange).WillOnce(testing::Throw(Hip::RuntimeError(hipErrorNotFound)));
    hipPointerAttribute_t attrs{};
    attrs.type = hipMemoryTypeDevice;
    EXPECT_CALL(mhip, hipPointerGetAttributes).WillOnce(testing::Return(attrs));
    ASSERT_EQ(hipFileBufRegister(reinterpret_cast<void *>(0x1), 100, 0), HipFileOpError(hipFileInvalidValue));
}

TEST_F(HipFileBuffer, registerOverflowingRangeReturnsError)
{
    StrictMock<MHip>   mhip;
    HipMemAddressRange range{nonnull_ptr, 100};
    EXPECT_CALL(mhip, hipMemGetAddressRange).WillOnce(testing::Return(range));
    hipPointerAttribute_t attrs{};
    attrs.type = hipMemoryTypeDevice;
    EXPECT_CALL(mhip, hipPointerGetAttributes).WillOnce(testing::Return(attrs));
    ASSERT_EQ(hipFileBufRegister(reinterpret_cast<void *>(0xFFFF0000), 0x10000, 0),
              HipFileOpError(hipFileHipPointerRangeError));
}

TEST_F(HipFileBuffer, deregister_internal_not_registered)
{
    ASSERT_THROW(Context<DriverState>::get()->deregisterBuffer(nonnull_ptr), BufferNotRegistered);
}

TEST_F(HipFileBuffer, deregister_not_registered)
{
    ASSERT_EQ(hipFileBufDeregister(nonnull_ptr), HipFileOpError(hipFileMemoryNotRegistered));
}

TEST_F(HipFileBuffer, deregister_internal)
{
    StrictMock<MHip> mhip;
    expect_buffer_registration(mhip, hipMemoryTypeDevice);
    Context<DriverState>::get()->registerBuffer(nonnull_ptr, 0, 0);
    Context<DriverState>::get()->deregisterBuffer(nonnull_ptr);
}

TEST_F(HipFileBuffer, deregister)
{
    StrictMock<MHip> mhip;
    expect_buffer_registration(mhip, hipMemoryTypeDevice);
    ASSERT_EQ(hipFileBufRegister(nonnull_ptr, 0, 0), HIPFILE_SUCCESS);
    ASSERT_EQ(hipFileBufDeregister(nonnull_ptr), HIPFILE_SUCCESS);
}

TEST_F(HipFileBuffer, deregister_internal_duplicate_deregister)
{
    StrictMock<MHip> mhip;
    expect_buffer_registration(mhip, hipMemoryTypeDevice);
    Context<DriverState>::get()->registerBuffer(nonnull_ptr, 0, 0);
    Context<DriverState>::get()->deregisterBuffer(nonnull_ptr);
    ASSERT_THROW(Context<DriverState>::get()->deregisterBuffer(nonnull_ptr), BufferNotRegistered);
}

TEST_F(HipFileBuffer, deregister_duplicate_deregister)
{
    StrictMock<MHip> mhip;
    expect_buffer_registration(mhip, hipMemoryTypeDevice);
    ASSERT_EQ(hipFileBufRegister(nonnull_ptr, 0, 0), HIPFILE_SUCCESS);
    ASSERT_EQ(hipFileBufDeregister(nonnull_ptr), HIPFILE_SUCCESS);
    ASSERT_EQ(hipFileBufDeregister(nonnull_ptr), HipFileOpError(hipFileMemoryNotRegistered));
}

TEST_F(HipFileBuffer, deregister_internal_get_prevents_deregister)
{
    StrictMock<MHip> mhip;
    expect_buffer_registration(mhip, hipMemoryTypeDevice);
    Context<DriverState>::get()->registerBuffer(nonnull_ptr, 0, 0);
    {
        auto buffer = Context<DriverState>::get()->getRegisteredBuffer(nonnull_ptr);
        ASSERT_THROW(Context<DriverState>::get()->deregisterBuffer(nonnull_ptr), BufferOperationsOutstanding);
    }
    Context<DriverState>::get()->deregisterBuffer(nonnull_ptr);
}

TEST_F(HipFileBuffer, deregister_get_prevents_deregister)
{
    StrictMock<MHip> mhip;
    expect_buffer_registration(mhip, hipMemoryTypeDevice);
    ASSERT_EQ(hipFileBufRegister(nonnull_ptr, 0, 0), HIPFILE_SUCCESS);
    {
        auto buffer = Context<DriverState>::get()->getRegisteredBuffer(nonnull_ptr);
        ASSERT_EQ(hipFileBufDeregister(nonnull_ptr), HipFileOpError(hipFileInternalError));
    }
    ASSERT_EQ(hipFileBufDeregister(nonnull_ptr), HIPFILE_SUCCESS);
}

TEST_F(HipFileBuffer, get_not_registered)
{
    ASSERT_THROW(Context<DriverState>::get()->getRegisteredBuffer(nonnull_ptr), BufferNotRegistered);
}

TEST_F(HipFileBuffer, get_internal_after_register)
{
    StrictMock<MHip> mhip;
    expect_buffer_registration(mhip, hipMemoryTypeDevice);
    Context<DriverState>::get()->registerBuffer(nonnull_ptr, 0, 0);
    auto buffer = Context<DriverState>::get()->getRegisteredBuffer(nonnull_ptr);
}

TEST_F(HipFileBuffer, get_after_register)
{
    StrictMock<MHip> mhip;
    expect_buffer_registration(mhip, hipMemoryTypeDevice);
    ASSERT_EQ(hipFileBufRegister(nonnull_ptr, 0, 0), HIPFILE_SUCCESS);
    auto buffer = Context<DriverState>::get()->getRegisteredBuffer(nonnull_ptr);
}

TEST_F(HipFileBuffer, get_internal_after_deregister)
{
    StrictMock<MHip> mhip;
    expect_buffer_registration(mhip, hipMemoryTypeDevice);
    Context<DriverState>::get()->registerBuffer(nonnull_ptr, 0, 0);
    Context<DriverState>::get()->deregisterBuffer(nonnull_ptr);
    ASSERT_THROW(Context<DriverState>::get()->getRegisteredBuffer(nonnull_ptr), BufferNotRegistered);
}

TEST_F(HipFileBuffer, get_after_deregister)
{
    StrictMock<MHip> mhip;
    expect_buffer_registration(mhip, hipMemoryTypeDevice);
    ASSERT_EQ(hipFileBufRegister(nonnull_ptr, 0, 0), HIPFILE_SUCCESS);
    ASSERT_EQ(hipFileBufDeregister(nonnull_ptr), HIPFILE_SUCCESS);
    ASSERT_THROW(Context<DriverState>::get()->getRegisteredBuffer(nonnull_ptr), BufferNotRegistered);
}

TEST_F(HipFileBuffer, get_buffer_makes_temporary_buffer)
{
    StrictMock<MHip> mhip;
    expect_buffer_registration(mhip, hipMemoryTypeDevice);
    auto buffer = Context<DriverState>::get()->getBuffer(nonnull_ptr);
    ASSERT_EQ(buffer.use_count(), 1);
}

TEST_F(HipFileBuffer, get_buffer_returns_registered_buffer)
{
    StrictMock<MHip> mhip;
    expect_buffer_registration(mhip, hipMemoryTypeDevice);
    Context<DriverState>::get()->registerBuffer(nonnull_ptr, 0, 0);
    ASSERT_EQ(Context<DriverState>::get()->getBuffer(nonnull_ptr),
              Context<DriverState>::get()->getRegisteredBuffer(nonnull_ptr));
}

TEST_F(HipFileBuffer, get_buffer_throws_on_getPointerAttributes_error)
{
    StrictMock<MHip> mhip;
    EXPECT_CALL(mhip, hipPointerGetAttributes).WillOnce(testing::Throw(Hip::RuntimeError(hipErrorUnknown)));
    ASSERT_THROW(Context<DriverState>::get()->getBuffer(nonnull_ptr), Hip::RuntimeError);
}

HIPFILE_WARN_NO_GLOBAL_CTOR_ON
