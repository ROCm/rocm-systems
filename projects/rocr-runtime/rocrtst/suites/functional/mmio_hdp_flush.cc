/*
 * Copyright © Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

#include "suites/functional/mmio_hdp_flush.h"

#include <cstdint>
#include <cstdio>

#include "common/base_rocr_utils.h"
#include "common/common.h"
#include "gtest/gtest.h"
#include "hsa/hsa.h"
#include "hsa/hsa_ext_amd.h"

MMIOHdpFlushTest::MMIOHdpFlushTest(void) : TestBase() {
  set_num_iteration(1);
  set_title("MMIO HDP Flush Test");
  set_description(
      "Verifies the MMIO_REMAP allocation is usable by checking that the "
      "GPU agent's HSA_AMD_AGENT_INFO_HDP_FLUSH pointer is non-null and "
      "writable. Catches the case where the MMIO BO allocation reports "
      "success but the HDP flush register is never mapped (observed in DRM "
      "user-queue mode).");
}

MMIOHdpFlushTest::~MMIOHdpFlushTest(void) {
}

void MMIOHdpFlushTest::SetUp(void) {
  if (!checkPlatformFiltering()) return;

  TestBase::SetUp();  // hsa_init()

  hsa_agent_t gpu_agent = {0};
  hsa_status_t err = hsa_iterate_agents(rocrtst::FindGPUDevice, &gpu_agent);
  // FindGPUDevice returns HSA_STATUS_INFO_BREAK when a GPU is found.
  if (err != HSA_STATUS_INFO_BREAK || gpu_agent.handle == 0) {
    fprintf(stdout, "[ SKIPPED ] No GPU agent visible\n");
    test_skipped_ = true;
    return;
  }
  set_gpu_device1(gpu_agent);
}

void MMIOHdpFlushTest::Run(void) {
  if (test_skipped_) return;
  TestBase::Run();
}

void MMIOHdpFlushTest::DisplayTestInfo(void) {
  TestBase::DisplayTestInfo();
}

void MMIOHdpFlushTest::DisplayResults(void) const {
  TestBase::DisplayResults();
}

void MMIOHdpFlushTest::Close(void) {
  TestBase::Close();
}

void MMIOHdpFlushTest::TestHdpFlushMapped(void) {
  if (test_skipped_) return;

  hsa_agent_t gpu_agent = *gpu_device1();
  ASSERT_NE(gpu_agent.handle, 0u);

  hsa_amd_hdp_flush_t hdp = {};
  hsa_status_t err = hsa_agent_get_info(
      gpu_agent, static_cast<hsa_agent_info_t>(HSA_AMD_AGENT_INFO_HDP_FLUSH),
      &hdp);
  ASSERT_EQ(HSA_STATUS_SUCCESS, err)
      << "HSA_AMD_AGENT_INFO_HDP_FLUSH query failed";

  // The MMIO_REMAP page backs these registers; a NULL pointer means the
  // MMIO allocation/mapping did not complete, even though the underlying BO
  // allocation may have reported success.
  ASSERT_NE(hdp.HDP_MEM_FLUSH_CNTL, nullptr)
      << "HDP_MEM_FLUSH_CNTL is NULL - MMIO_REMAP page not mapped "
         "(MMIO allocate not working)";
  ASSERT_NE(hdp.HDP_REG_FLUSH_CNTL, nullptr)
      << "HDP_REG_FLUSH_CNTL is NULL - MMIO_REMAP page not mapped";

  // Perform the HDP flush write the runtime itself issues. If the page is
  // not properly mapped this faults (SIGSEGV/SIGBUS), failing the test.
  *hdp.HDP_MEM_FLUSH_CNTL = 1;
}
