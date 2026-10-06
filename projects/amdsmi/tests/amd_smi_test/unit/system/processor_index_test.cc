// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// Processor index defaults. Only the CPU socket and core constructors pass an
// index. The CPU/Core APIs reuse that index as an ESMI socket or core number.
// An unset index must therefore stay an obvious out-of-range selection.

#include <gtest/gtest.h>

#include <cstdint>
#include <string>

#include "amd_smi/impl/amd_smi_processor.h"

namespace {

using amd::smi::AMDSmiProcessor;

// ESMI limits from e_smi.h. Both are counts, so the last valid index is one below.
constexpr uint32_t kMaxSockets = 8;           // MAX_SOCKET
constexpr uint32_t kMaxCoresPerSocket = 512;  // CPU_MAX_CORES_PER_SOCKET
constexpr uint32_t kMaxCoresInSystem = kMaxSockets * kMaxCoresPerSocket;
constexpr uint32_t kLastCoreInSocket = kMaxCoresPerSocket - 1;
constexpr uint32_t kLastCoreInSystem = kMaxCoresInSystem - 1;

// Spelled out rather than read from the header, so changing the sentinel fails here.
constexpr uint32_t kUnsetIndex = UINT32_MAX;
constexpr uint32_t kUnsetIndexLowByte = 255;

// Arbitrary in-range values. They only have to survive the round trip.
constexpr uint32_t kSampleSocketIndex = 3;
constexpr uint32_t kSampleCoreIndex = 191;

TEST(SystemUnit, ProcessorIndexUnsetDefaultsToInvalid) {
  AMDSmiProcessor gpu(AMDSMI_PROCESSOR_TYPE_AMD_GPU);
  EXPECT_EQ(gpu.get_processor_index(), kUnsetIndex)
      << "pindex_ lost its default initializer in amd_smi_processor.h. GPU, NIC and\n"
         "switch processors never pass an index, so they would hold indeterminate\n"
         "values again. Fix the initializer, not this expectation. See AILITOOLS-304.";
}

TEST(SystemUnit, ProcessorIndexIdentifierCtorDefaultsToInvalid) {
  AMDSmiProcessor by_id(std::string("0000:0c:00.0"));
  EXPECT_EQ(by_id.get_processor_index(), kUnsetIndex)
      << "The identifier constructor sets neither the type nor the index, so it\n"
         "depends entirely on the default initializer. See AILITOOLS-304.";
}

TEST(SystemUnit, ProcessorIndexPreservedWhenProvided) {
  AMDSmiProcessor socket(AMDSMI_PROCESSOR_TYPE_AMD_CPU, kSampleSocketIndex);
  EXPECT_EQ(socket.get_processor_index(), kSampleSocketIndex)
      << "CPU sockets must keep the index they were built with. Any other value\n"
         "means every CPU API would target the wrong socket.";

  AMDSmiProcessor core(AMDSMI_PROCESSOR_TYPE_AMD_CPU_CORE, kSampleCoreIndex);
  EXPECT_EQ(core.get_processor_index(), kSampleCoreIndex)
      << "CPU cores must keep the index they were built with.";
}

// ESMI allows CPU_MAX_CORES_PER_SOCKET (512) cores across MAX_SOCKET (8)
// sockets, so the index has to hold values well beyond a single byte. amd-smi
// assigns per-socket core indices today, so the system-wide ceiling is headroom
// rather than a value it currently builds.
TEST(SystemUnit, ProcessorIndexHoldsFullEsmiCoreRange) {
  AMDSmiProcessor last_in_socket(AMDSMI_PROCESSOR_TYPE_AMD_CPU_CORE, kLastCoreInSocket);
  EXPECT_EQ(last_in_socket.get_processor_index(), kLastCoreInSocket)
      << "Core " << kLastCoreInSocket << " is the last of " << kMaxCoresPerSocket
      << " cores ESMI allows on one socket. Narrowing pindex_ to uint8_t would\n"
         "fold it onto core "
      << (kLastCoreInSocket & 0xFFu) << ".";

  AMDSmiProcessor last_in_system(AMDSMI_PROCESSOR_TYPE_AMD_CPU_CORE, kLastCoreInSystem);
  EXPECT_EQ(last_in_system.get_processor_index(), kLastCoreInSystem)
      << "Core " << kLastCoreInSystem
      << " is the highest index ESMI can address: " << kMaxCoresPerSocket << " cores on each of "
      << kMaxSockets << " sockets.";
}

// The CPU/Core APIs narrow the index to uint8_t before passing it to ESMI.
// A low byte inside the socket range selects a real socket and returns its
// data. The previous uninitialized value did this non-deterministically.
TEST(SystemUnit, ProcessorIndexSentinelSurvivesUint8Narrowing) {
  AMDSmiProcessor gpu(AMDSMI_PROCESSOR_TYPE_AMD_GPU);
  const uint32_t narrowed = static_cast<uint8_t>(gpu.get_processor_index());
  EXPECT_EQ(narrowed, kUnsetIndexLowByte)
      << "The sentinel's low byte must stay outside the range of " << kMaxSockets
      << " sockets. UINT32_MAX narrows to " << kUnsetIndexLowByte
      << " and is rejected. A tidier\n"
         "looking value such as 0x100 narrows to 0 and would silently select socket\n"
         "0. Change the sentinel in amd_smi_processor.h, not this expectation.\n"
         "See AILITOOLS-304.";
}

}  // namespace
