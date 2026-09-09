// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "tray_info_read.h"

#include <gtest/gtest.h>

#include <cstdint>
#include <iostream>
#include <limits>

#include "amd_smi/amdsmi.h"
#include "test_common.h"

TestTrayInfoRead::TestTrayInfoRead() : TestBase() {
  set_title("AMDSMI Tray Info (UALoE) Read Test");
  set_description(
      "This test verifies that node-scoped compute tray type and "
      "accelerator count can be read properly via amdsmi_get_tray_info().");
}

TestTrayInfoRead::~TestTrayInfoRead(void) {}

void TestTrayInfoRead::SetUp(void) {
  TestBase::SetUp();
  return;
}

void TestTrayInfoRead::DisplayTestInfo(void) { TestBase::DisplayTestInfo(); }

void TestTrayInfoRead::DisplayResults(void) const {
  TestBase::DisplayResults();
  return;
}

void TestTrayInfoRead::Close() {
  // This will close handles opened within rsmitst utility calls and call
  // amdsmi_shut_down(), so it should be done after other hsa cleanup
  TestBase::Close();
}

static const char* TrayTypeName(amdsmi_compute_tray_type_t tray_type) {
  switch (tray_type) {
    case AMDSMI_COMPUTE_TRAY_TYPE_HELIOS_P:
      return "HELIOS_P";
    case AMDSMI_COMPUTE_TRAY_TYPE_HELIOS_R:
      return "HELIOS_R";
    case AMDSMI_COMPUTE_TRAY_TYPE_TITAN:
      return "TITAN";
    default:
      return "UNKNOWN";
  }
}

void TestTrayInfoRead::Run(void) {
  amdsmi_status_t err;

  TestBase::Run();
  if (setup_failed_) {
    IF_VERB(STANDARD) { std::cout << "** SetUp Failed for this test. Skipping.**" << std::endl; }
    return;
  }

  // amdsmi_get_tray_info() is node-scoped (not per-processor); node_handle is
  // reserved for future use and must be passed as NULL.
  amdsmi_tray_info_t tray_info = {};
  DISPLAY_AMDSMI_API("amdsmi_get_tray_info", "node=NULL", VERB(STANDARD));
  err = amdsmi_get_tray_info(nullptr, &tray_info);
  DISPLAY_AMDSMI_STATUS(VERB(STANDARD), __FILE__, __LINE__, err, AMDSMI_STATUS_SUCCESS,
                        AMDSMI_STATUS_NOT_SUPPORTED);
  ASSERT_TRUE(err == AMDSMI_STATUS_SUCCESS || err == AMDSMI_STATUS_NOT_SUPPORTED);

  if (err == AMDSMI_STATUS_NOT_SUPPORTED) {
    IF_VERB(STANDARD) {
      std::cout << "\t**amdsmi_get_tray_info() is not supported on this system" << std::endl;
    }
    ASSERT_EQ(tray_info.max_acc_per_tray, std::numeric_limits<uint32_t>::max());
    ASSERT_EQ(tray_info.tray_type, AMDSMI_COMPUTE_TRAY_TYPE_UNKNOWN);
  } else {
    IF_VERB(STANDARD) {
      std::cout << "\t\tmax_acc_per_tray: " << tray_info.max_acc_per_tray << "\n"
                << "\t\ttray_type:        " << TrayTypeName(tray_info.tray_type) << std::endl;
    }
    ASSERT_NE(tray_info.max_acc_per_tray, std::numeric_limits<uint32_t>::max());
  }

  // Null-pointer validation
  DISPLAY_AMDSMI_API("amdsmi_get_tray_info(info=nullptr check)", "node=NULL", VERB(STANDARD));
  err = amdsmi_get_tray_info(nullptr, nullptr);
  DISPLAY_AMDSMI_STATUS(VERB(STANDARD), __FILE__, __LINE__, err, AMDSMI_STATUS_INVAL);
  ASSERT_EQ(err, AMDSMI_STATUS_INVAL);

  // node_handle must be NULL; a non-NULL handle must be rejected.
  amdsmi_node_handle bogus_handle = reinterpret_cast<amdsmi_node_handle>(0x1);
  DISPLAY_AMDSMI_API("amdsmi_get_tray_info(node_handle!=NULL check)", "node=0x1", VERB(STANDARD));
  err = amdsmi_get_tray_info(bogus_handle, &tray_info);
  DISPLAY_AMDSMI_STATUS(VERB(STANDARD), __FILE__, __LINE__, err, AMDSMI_STATUS_INVAL);
  ASSERT_EQ(err, AMDSMI_STATUS_INVAL);
}
