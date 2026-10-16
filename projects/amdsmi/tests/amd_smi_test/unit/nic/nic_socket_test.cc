// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

/**
 * Ownership of AI-NIC processors: AMDSmiSocket holds raw pointers and frees them in its
 * destructor, as it does for GPU and CPU processors. A device a socket does not free is
 * leaked, and a NIC-only init/shutdown/init cycle then counts it twice.
 */

#include <gtest/gtest.h>

#include <cstdint>
#include <limits>

#include "amd_smi/impl/amd_smi_processor.h"
#include "amd_smi/impl/amd_smi_socket.h"
#include "nic_unit_fixture.h"

namespace {

using amd::smi::AMDSmiProcessor;
using amd::smi::AMDSmiSocket;

// Counts its own destruction, so ownership is observable without a sanitizer.
class CountedNic_t : public AMDSmiProcessor {
 public:
  explicit CountedNic_t(int* num_destroyed)
      : AMDSmiProcessor(AMDSMI_PROCESSOR_TYPE_AMD_NIC), m_num_destroyed(num_destroyed) {}
  ~CountedNic_t() override { ++(*m_num_destroyed); }

 private:
  int* m_num_destroyed;
};

TEST_F(NicUnit, SocketDestroysItsNicProcessors) {
  auto num_destroyed = int{0};
  auto* socket = new AMDSmiSocket("0000:c1:00");
  socket->add_processor(new CountedNic_t(&num_destroyed));
  socket->add_processor(new CountedNic_t(&num_destroyed));
  ASSERT_EQ(socket->get_processors(AMDSMI_PROCESSOR_TYPE_AMD_NIC).size(), size_t{2});

  delete socket;

  EXPECT_EQ(num_destroyed, 2);
}

// Stands in for a GPU: the socket's default list, which an unhandled processor type falls back to.
class Gpu_t : public AMDSmiProcessor {
 public:
  Gpu_t() : AMDSmiProcessor(AMDSMI_PROCESSOR_TYPE_AMD_GPU) {}
};

// A binary built against the previous release can still ask for the Broadcom types. They must read
// as "none present", not as the GPU list.
TEST_F(NicUnit, DeprecatedBrcmTypesReportNoProcessors) {
  AMDSmiSocket socket("0000:c1:00");
  socket.add_processor(new Gpu_t());

  for (const auto type : {AMDSMI_PROCESSOR_TYPE_BRCM_NIC, AMDSMI_PROCESSOR_TYPE_BRCM_SWITCH}) {
    auto num_processors = std::numeric_limits<uint32_t>::max();
    EXPECT_EQ(socket.get_processor_count(type, &num_processors), AMDSMI_STATUS_SUCCESS);
    EXPECT_EQ(num_processors, uint32_t{0});
    EXPECT_TRUE(socket.get_processors(type).empty());
  }
}

TEST_F(NicUnit, SocketWithoutNicProcessorsDestroysCleanly) {
  auto num_destroyed = int{0};
  auto* socket = new AMDSmiSocket("0000:c1:00");
  ASSERT_TRUE(socket->get_processors(AMDSMI_PROCESSOR_TYPE_AMD_NIC).empty());

  delete socket;

  EXPECT_EQ(num_destroyed, 0);
}

}  // namespace
