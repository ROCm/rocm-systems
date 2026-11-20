// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include <gtest/gtest.h>

#include <algorithm>
#include <cstdint>
#include <string>
#include <vector>

#include "amd_smi/amdsmi.h"
#include "amd_smi/impl/amd_smi_socket.h"
#include "amd_smi/impl/amd_smi_system.h"
#include "amd_smi/impl/nic/amd_smi_ainic_device.h"
#include "test_fixture_utils.h"

namespace {

using amd::smi::AMDSmiAINICDevice;
using amd::smi::AMDSmiSocket;
using amd::smi::AMDSmiSystem;
using amd::smi::testing::ScopedProcessorRegistration;

// Counts its destructions in a static, so a late one never writes into a finished test.
class TrackedAiNic : public AMDSmiAINICDevice {
 public:
  TrackedAiNic() : AMDSmiAINICDevice(AINICInfo{}) {}
  ~TrackedAiNic() override { ++destructions; }

  static inline int destructions = 0;
};

struct Topology {
  std::vector<std::string> socket_ids;
  uint32_t ai_nics = 0;
};

Topology ReadTopology() {
  Topology topology;
  uint32_t count = 0;
  EXPECT_EQ(amdsmi_get_socket_handles(&count, nullptr), AMDSMI_STATUS_SUCCESS);
  std::vector<amdsmi_socket_handle> sockets(count);
  EXPECT_EQ(amdsmi_get_socket_handles(&count, sockets.data()), AMDSMI_STATUS_SUCCESS);
  for (uint32_t i = 0; i < count; i++) {
    char id[AMDSMI_MAX_STRING_LENGTH] = {};
    EXPECT_EQ(amdsmi_get_socket_info(sockets[i], sizeof(id), id), AMDSMI_STATUS_SUCCESS);
    topology.socket_ids.emplace_back(id);
    uint32_t nics = 0;
    EXPECT_EQ(amdsmi_get_processor_handles_by_type(sockets[i], AMDSMI_PROCESSOR_TYPE_AMD_NIC,
                                                   nullptr, &nics),
              AMDSMI_STATUS_SUCCESS);
    topology.ai_nics += nics;
  }
  std::sort(topology.socket_ids.begin(), topology.socket_ids.end());
  return topology;
}

void ExpectNotInitialized() {
  uint32_t count = 0;
  ASSERT_EQ(amdsmi_get_socket_handles(&count, nullptr), AMDSMI_STATUS_NOT_INIT)
      << "An earlier test left the library initialized.";
}

// A NIC-only init has no GPU or CPU teardown to free its sockets. Its shut down must still free
// every socket and AI-NIC and forget their handles, or the next init returns them again.
TEST(SystemUnit, NicOnlyShutDownReleasesSocketsAndHandles) {
  ASSERT_NO_FATAL_FAILURE(ExpectNotInitialized());
  ASSERT_EQ(amdsmi_init(AMDSMI_INIT_AMD_NICS), AMDSMI_STATUS_SUCCESS);
  const Topology discovered = ReadTopology();

  // An AI-NIC on its own socket, owned by the system the way discovery leaves one, so a host
  // without NIC hardware still has something to tear down.
  TrackedAiNic::destructions = 0;
  auto* socket = new AMDSmiSocket("ffff:ff:1f");
  socket->add_processor(new TrackedAiNic());
  AMDSmiSystem::getInstance().get_sockets().push_back(socket);

  // A handle the test keeps alive, to check that shut down forgets it.
  AMDSmiAINICDevice kept_nic(AMDSmiAINICDevice::AINICInfo{});
  ScopedProcessorRegistration registration(&kept_nic);
  amdsmi_processor_type_t type = AMDSMI_PROCESSOR_TYPE_UNKNOWN;
  EXPECT_EQ(amdsmi_get_processor_type(registration.handle(), &type), AMDSMI_STATUS_SUCCESS);

  // Shut down even after a failed check, so the synthetic socket is not left to later tests.
  EXPECT_EQ(amdsmi_shut_down(), AMDSMI_STATUS_SUCCESS);
  EXPECT_EQ(TrackedAiNic::destructions, 1);

  ASSERT_EQ(amdsmi_init(AMDSMI_INIT_AMD_NICS), AMDSMI_STATUS_SUCCESS);
  const Topology rediscovered = ReadTopology();
  EXPECT_EQ(rediscovered.socket_ids, discovered.socket_ids);
  EXPECT_EQ(rediscovered.ai_nics, discovered.ai_nics);
  EXPECT_EQ(amdsmi_get_processor_type(registration.handle(), &type), AMDSMI_STATUS_NOT_FOUND);
  EXPECT_EQ(amdsmi_shut_down(), AMDSMI_STATUS_SUCCESS);
}

// Every native GPU init starts with a cleanup() of an already shut down system, which must not
// destroy the NIC context again: ASan reports that as a use-after-free, other builds can hang.
TEST(SystemUnit, CleanupAfterShutDownLeavesNicContextAlone) {
  ASSERT_NO_FATAL_FAILURE(ExpectNotInitialized());
  ASSERT_EQ(amdsmi_init(AMDSMI_INIT_AMD_NICS), AMDSMI_STATUS_SUCCESS);
  ASSERT_EQ(amdsmi_shut_down(), AMDSMI_STATUS_SUCCESS);
  EXPECT_EQ(AMDSmiSystem::getInstance().cleanup(), AMDSMI_STATUS_SUCCESS);
}

}  // namespace
