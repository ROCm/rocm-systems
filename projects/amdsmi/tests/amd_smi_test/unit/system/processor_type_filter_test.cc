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
  const std::array<amdsmi_processor_type_t, 5> types = {
      AMDSMI_PROCESSOR_TYPE_AMD_GPU, AMDSMI_PROCESSOR_TYPE_AMD_CPU,
      AMDSMI_PROCESSOR_TYPE_AMD_CPU_CORE, AMDSMI_PROCESSOR_TYPE_BRCM_NIC,
      AMDSMI_PROCESSOR_TYPE_BRCM_SWITCH};
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
  EXPECT_TRUE(socket.get_processors(AMDSMI_PROCESSOR_TYPE_AMD_NIC).empty());
  ASSERT_EQ(socket.get_processors().size(), 1u);
  EXPECT_EQ(socket.get_processors().front()->get_processor_type(), AMDSMI_PROCESSOR_TYPE_AMD_GPU);
}

TEST(SystemUnit, ProcessorTypeFilterUnimplementedTypesNeverReturnGpus) {
  AMDSmiSocket socket(std::string("test"));
  socket.add_processor(new AMDSmiProcessor(AMDSMI_PROCESSOR_TYPE_AMD_GPU));
  const std::array<amdsmi_processor_type_t, 4> types = {
      AMDSMI_PROCESSOR_TYPE_UNKNOWN, AMDSMI_PROCESSOR_TYPE_NON_AMD_GPU,
      AMDSMI_PROCESSOR_TYPE_NON_AMD_CPU, AMDSMI_PROCESSOR_TYPE_AMD_APU};
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
  // Value 10 is unnamed but inside the enum's representable range.
  const auto invalid = static_cast<amdsmi_processor_type_t>(10);
  uint32_t count = UINT32_MAX;
  EXPECT_EQ(socket.get_processor_count(invalid, &count), AMDSMI_STATUS_INVAL);
  EXPECT_EQ(count, 0u);
  EXPECT_TRUE(socket.get_processors(invalid).empty());
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

class CountedProcessor : public AMDSmiProcessor {
 public:
  CountedProcessor(amdsmi_processor_type_t type, int& destroyed)
      : AMDSmiProcessor(type), destroyed_(destroyed) {}
  ~CountedProcessor() override { ++destroyed_; }

 private:
  int& destroyed_;
};

TEST(SystemUnit, ProcessorTypeUnreportedObjectsAreReleased) {
  int destroyed = 0;
  {
    AMDSmiSocket socket(std::string("test"));
    for (auto type : {AMDSMI_PROCESSOR_TYPE_UNKNOWN, AMDSMI_PROCESSOR_TYPE_AMD_APU,
                      AMDSMI_PROCESSOR_TYPE_NON_AMD_GPU, AMDSMI_PROCESSOR_TYPE_NON_AMD_CPU}) {
      socket.add_processor(new CountedProcessor(type, destroyed));
      EXPECT_TRUE(socket.get_processors(type).empty());
    }
    socket.add_processor(new CountedProcessor(AMDSMI_PROCESSOR_TYPE_AMD_NIC, destroyed));
    EXPECT_EQ(socket.get_processors(AMDSMI_PROCESSOR_TYPE_AMD_NIC).size(), 1u);
    EXPECT_TRUE(socket.get_processors().empty());
    EXPECT_EQ(destroyed, 0);
  }
  EXPECT_EQ(destroyed, 5);
}

TEST(SystemUnit, ProcessorTypeCleanupWithoutGpuBackend) {
  auto& system = amd::smi::AMDSmiSystem::getInstance();
  ASSERT_FALSE(amd::smi::amdsmi_library_initialized());
  ASSERT_TRUE(system.get_sockets().empty());
  ASSERT_EQ(system.init(0), AMDSMI_STATUS_SUCCESS);
  int destroyed = 0;
  auto* socket = new AMDSmiSocket(std::string("test"));
  socket->add_processor(new CountedProcessor(AMDSMI_PROCESSOR_TYPE_AMD_NIC, destroyed));
  socket->add_processor(new CountedProcessor(AMDSMI_PROCESSOR_TYPE_UNKNOWN, destroyed));
  system.get_sockets().push_back(socket);
  EXPECT_EQ(system.cleanup(), AMDSMI_STATUS_SUCCESS);
  EXPECT_EQ(destroyed, 2);
  EXPECT_TRUE(system.get_sockets().empty());
  ASSERT_EQ(system.init(0), AMDSMI_STATUS_SUCCESS);
  EXPECT_TRUE(system.get_sockets().empty());
  EXPECT_EQ(system.cleanup(), AMDSMI_STATUS_SUCCESS);
}

}  // namespace
