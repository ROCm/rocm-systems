// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include <gtest/gtest.h>

#include <array>
#include <cstdint>

#include "amd_smi/impl/amd_smi_common.h"
#include "amd_smi/impl/amd_smi_socket.h"
#include "amd_smi/impl/amd_smi_system.h"

namespace {

using amd::smi::AMDSmiProcessor;
using amd::smi::AMDSmiSocket;

TEST(SystemUnit, ProcessorTypeFilterKeepsSupportedTypesSeparate) {
  AMDSmiSocket socket(std::string("test"));
  const std::array<amdsmi_processor_type_t, 4> types = {
      AMDSMI_PROCESSOR_TYPE_AMD_GPU, AMDSMI_PROCESSOR_TYPE_AMD_CPU,
      AMDSMI_PROCESSOR_TYPE_AMD_CPU_CORE, AMDSMI_PROCESSOR_TYPE_AMD_NIC};
  for (auto type : types) {
    socket.add_processor(new AMDSmiProcessor(type));
  }
  for (auto type : types) {
    uint32_t count = UINT32_MAX;
    ASSERT_EQ(socket.get_processor_count(type, &count), AMDSMI_STATUS_SUCCESS);
    ASSERT_EQ(count, 1u);
    const auto& processors = socket.get_processors(type);
    ASSERT_EQ(processors.size(), count);
    EXPECT_EQ(processors.front()->get_processor_type(), type);
  }
  ASSERT_EQ(socket.get_processors().size(), 1u);
  EXPECT_EQ(socket.get_processors().front()->get_processor_type(), AMDSMI_PROCESSOR_TYPE_AMD_GPU);
}

TEST(SystemUnit, ProcessorTypeFilterUnimplementedTypesNeverReturnGpus) {
  AMDSmiSocket socket(std::string("test"));
  socket.add_processor(new AMDSmiProcessor(AMDSMI_PROCESSOR_TYPE_AMD_GPU));
  const std::array<amdsmi_processor_type_t, 5> types = {
      AMDSMI_PROCESSOR_TYPE_NON_AMD_GPU, AMDSMI_PROCESSOR_TYPE_NON_AMD_CPU,
      AMDSMI_PROCESSOR_TYPE_AMD_APU, AMDSMI_PROCESSOR_TYPE_BRCM_NIC,
      AMDSMI_PROCESSOR_TYPE_BRCM_SWITCH};
  for (auto type : types) {
    uint32_t count = UINT32_MAX;
    EXPECT_TRUE(socket.get_processors(type).empty()) << "type=" << type;
    EXPECT_EQ(socket.get_processor_count(type, &count), AMDSMI_STATUS_SUCCESS);
    EXPECT_EQ(count, 0u);
  }
}

TEST(SystemUnit, ProcessorTypeFilterInvalidTypeIsRejected) {
  AMDSmiSocket socket(std::string("test"));
  socket.add_processor(new AMDSmiProcessor(AMDSMI_PROCESSOR_TYPE_AMD_GPU));
  // One past AMDSMI_PROCESSOR_TYPE_BRCM_SWITCH, still inside the enum's representable range.
  const auto invalid = static_cast<amdsmi_processor_type_t>(10);
  for (auto type : {AMDSMI_PROCESSOR_TYPE_UNKNOWN, invalid}) {
    SCOPED_TRACE(type);
    uint32_t count = UINT32_MAX;
    EXPECT_EQ(socket.get_processor_count(type, &count), AMDSMI_STATUS_INVAL);
    EXPECT_EQ(count, 0u);
    EXPECT_TRUE(socket.get_processors(type).empty());
  }
}

class RegisteredTestSocket {
 public:
  RegisteredTestSocket() {
    for (int i = 0; i < 3; ++i) {
      socket_.add_processor(new AMDSmiProcessor(AMDSMI_PROCESSOR_TYPE_AMD_GPU));
    }
    socket_.add_processor(new AMDSmiProcessor(AMDSMI_PROCESSOR_TYPE_AMD_CPU));
    amd::smi::AMDSmiSystem::getInstance().get_sockets().push_back(&socket_);
    amd::smi::amdsmi_library_init_ref_acquire();
  }

  ~RegisteredTestSocket() {
    amd::smi::AMDSmiSystem::getInstance().get_sockets().pop_back();
    amd::smi::amdsmi_library_init_ref_release();
  }

  AMDSmiSocket socket_{std::string("test")};
};

TEST(SystemUnit, ProcessorTypeBufferCountQuery) {
  RegisteredTestSocket registered;
  uint32_t count = 0;
  ASSERT_EQ(amdsmi_get_processor_handles_by_type(&registered.socket_, AMDSMI_PROCESSOR_TYPE_AMD_GPU,
                                                 nullptr, &count),
            AMDSMI_STATUS_SUCCESS);
  EXPECT_EQ(count, 3u);
}

TEST(SystemUnit, ProcessorTypeInvalidQueryPreservesOutput) {
  RegisteredTestSocket registered;
  const auto invalid = static_cast<amdsmi_processor_type_t>(10);
  const auto guard = reinterpret_cast<amdsmi_processor_handle>(uintptr_t{1});
  for (auto type : {AMDSMI_PROCESSOR_TYPE_UNKNOWN, invalid}) {
    for (uint32_t capacity : {0u, 1u, 3u}) {
      std::array<amdsmi_processor_handle, 4> handles;
      handles.fill(guard);
      for (auto* buffer : {static_cast<amdsmi_processor_handle*>(nullptr), handles.data()}) {
        SCOPED_TRACE(type);
        SCOPED_TRACE(capacity);
        uint32_t count = capacity;
        EXPECT_EQ(amdsmi_get_processor_handles_by_type(&registered.socket_, type, buffer, &count),
                  AMDSMI_STATUS_INVAL);
        EXPECT_EQ(count, capacity);
        for (auto handle : handles) {
          EXPECT_EQ(handle, guard);
        }
      }
    }
  }
}

TEST(SystemUnit, ProcessorTypeBufferPartialFillAndBounds) {
  RegisteredTestSocket registered;
  const auto& expected = registered.socket_.get_processors();
  for (uint32_t capacity : {0u, 1u, 2u, 3u, 5u}) {
    SCOPED_TRACE(capacity);
    const auto guard = reinterpret_cast<amdsmi_processor_handle>(uintptr_t{1});
    std::array<amdsmi_processor_handle, 6> handles;
    handles.fill(guard);
    uint32_t count = capacity;
    ASSERT_EQ(amdsmi_get_processor_handles_by_type(
                  &registered.socket_, AMDSMI_PROCESSOR_TYPE_AMD_GPU, handles.data(), &count),
              AMDSMI_STATUS_SUCCESS);
    const uint32_t expected_count = capacity < 3 ? capacity : 3;
    ASSERT_EQ(count, expected_count);
    for (uint32_t i = 0; i < count; ++i) {
      EXPECT_EQ(handles[i], reinterpret_cast<amdsmi_processor_handle>(expected[i]));
    }
    for (size_t i = count; i < handles.size(); ++i) {
      EXPECT_EQ(handles[i], guard);
    }
  }
}

}  // namespace
