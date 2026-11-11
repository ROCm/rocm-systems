// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// amdsmi_fabric_telem_id_to_string() is a pure lookup over a static id->name
// table generated from the vendored IFOE_TELEM_ID_* headers, so these run
// without a device and catch id->name drift when the headers are re-synced.

#include <gtest/gtest.h>

#include <cstdint>

#include "amd_smi/amdsmi.h"
#include "test_common.h"

namespace {

// Literals (independent of the vendored macros) spanning the table — first,
// mid-table, and last entry — so a reorder or drop anywhere is caught.
static constexpr uint64_t kFirstTelemId = 0x1;
static constexpr uint64_t kMidTelemId = 0x6000001;
static constexpr uint64_t kLastTelemId = 0x6001011;

TEST(IfoeUnit, FabricTelemIdToStringMapsKnownIds) {
  const char* name = nullptr;

  DISPLAY_AMDSMI_API("amdsmi_fabric_telem_id_to_string", "id=kFirstTelemId",
                     GetTestVerbosity() >= 1);
  amdsmi_status_t err = amdsmi_fabric_telem_id_to_string(kFirstTelemId, &name);
  DISPLAY_AMDSMI_STATUS(GetTestVerbosity() >= 1, __FILE__, __LINE__, err, AMDSMI_STATUS_SUCCESS);
  ASSERT_EQ(err, AMDSMI_STATUS_SUCCESS);
  ASSERT_STREQ(name, "IFOE_SDP_TX_PACK_WR_REQ");

  DISPLAY_AMDSMI_API("amdsmi_fabric_telem_id_to_string", "id=kMidTelemId", GetTestVerbosity() >= 1);
  err = amdsmi_fabric_telem_id_to_string(kMidTelemId, &name);
  DISPLAY_AMDSMI_STATUS(GetTestVerbosity() >= 1, __FILE__, __LINE__, err, AMDSMI_STATUS_SUCCESS);
  ASSERT_EQ(err, AMDSMI_STATUS_SUCCESS);
  ASSERT_STREQ(name, "NETPORT_LINK_STATUS");

  DISPLAY_AMDSMI_API("amdsmi_fabric_telem_id_to_string", "id=kLastTelemId",
                     GetTestVerbosity() >= 1);
  err = amdsmi_fabric_telem_id_to_string(kLastTelemId, &name);
  DISPLAY_AMDSMI_STATUS(GetTestVerbosity() >= 1, __FILE__, __LINE__, err, AMDSMI_STATUS_SUCCESS);
  ASSERT_EQ(err, AMDSMI_STATUS_SUCCESS);
  ASSERT_STREQ(name, "NETPORT_FEC_CW_SYMBOL_ERRS_UNCORRECTABLE");
}

TEST(IfoeUnit, FabricTelemIdToStringRejectsNullName) {
  DISPLAY_AMDSMI_API("amdsmi_fabric_telem_id_to_string", "name=nullptr", GetTestVerbosity() >= 1);
  amdsmi_status_t err = amdsmi_fabric_telem_id_to_string(kFirstTelemId, nullptr);
  DISPLAY_AMDSMI_STATUS(GetTestVerbosity() >= 1, __FILE__, __LINE__, err, AMDSMI_STATUS_INVAL);
  ASSERT_EQ(err, AMDSMI_STATUS_INVAL);
}

TEST(IfoeUnit, FabricTelemIdToStringUnknownIdReportsUnknown) {
  const char* name = nullptr;
  DISPLAY_AMDSMI_API("amdsmi_fabric_telem_id_to_string", "id=UINT64_MAX", GetTestVerbosity() >= 1);
  amdsmi_status_t err = amdsmi_fabric_telem_id_to_string(UINT64_MAX, &name);
  DISPLAY_AMDSMI_STATUS(GetTestVerbosity() >= 1, __FILE__, __LINE__, err, AMDSMI_STATUS_NOT_FOUND);
  ASSERT_EQ(err, AMDSMI_STATUS_NOT_FOUND);
  ASSERT_STREQ(name, "UNKNOWN");
}

}  // namespace
