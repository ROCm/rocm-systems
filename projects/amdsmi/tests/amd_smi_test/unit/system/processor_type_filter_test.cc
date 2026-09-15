// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include <gtest/gtest.h>

#include <array>
#include <cstdint>

#include "amd_smi/impl/amd_smi_socket.h"

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
  // An unnamed value inside the enum's representable range avoids an
  // out-of-range C++ enum conversion while still testing invalid input.
  const auto invalid = static_cast<amdsmi_processor_type_t>(10);
  uint32_t count = UINT32_MAX;
  EXPECT_EQ(socket.get_processor_count(invalid, &count), AMDSMI_STATUS_INVAL);
  EXPECT_EQ(count, 0u);
  EXPECT_TRUE(socket.get_processors(invalid).empty());
}

}  // namespace
