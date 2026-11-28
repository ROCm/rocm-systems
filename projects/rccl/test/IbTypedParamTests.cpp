/*************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#include <gtest/gtest.h>
#include <rccl/rccl.h>

#include <cstdint>
#include <string>

#include "common/ProcessIsolatedTestRunner.hpp"

namespace RcclUnitTesting
{
  // NCCL_IB_* connection settings (src/transport/net_ib/connect.cc) are registered typed
  // params: AUTO and its legacy -1 alias are named values beside a bounded range, integers
  // outside the range or the destination type are rejected (the value falls back to the
  // default instead of being accepted or wrapped), and booleans take TRUE/FALSE spellings.
  // Driven through the exported ncclParam* API, so no IB device or GPU is needed.
  namespace
  {
    // Returns the ncclParamDumpAll() entry for `key` (header and "Current value" line), or "".
    std::string DumpEntry(const char* key)
    {
      testing::internal::CaptureStdout();
      ncclParamDumpAll();
      std::string out = testing::internal::GetCapturedStdout();
      size_t pos = out.find(std::string("\n") + key + " (");
      if (pos == std::string::npos) return "";
      size_t end = out.find("Accepted value", pos);
      return out.substr(pos + 1, end == std::string::npos ? std::string::npos : end - pos - 1);
    }
  }

  TEST(IbTypedParam, AutoNamedValueAndLegacyAlias)
  {
    RUN_ISOLATED_TEST_WITH_ENV(
      "IbTypedParam.AutoNamedValueAndLegacyAlias",
      []() {
        ncclParamHandle_t h = nullptr;
        int32_t v = 0;
        ASSERT_EQ(ncclParamBind(&h, "NCCL_IB_TC"), ncclSuccess) << "NCCL_IB_TC must be a typed param";
        ASSERT_EQ(ncclParamGetI32(h, &v), ncclSuccess);
        EXPECT_EQ(v, -1) << "auto (any case) resolves to the AUTO value";
        std::string tc = DumpEntry("NCCL_IB_TC");
        EXPECT_NE(tc.find("Current value=AUTO set_by=EnvPlugin"), std::string::npos) << tc;

        ASSERT_EQ(ncclParamBind(&h, "NCCL_IB_SL"), ncclSuccess) << "NCCL_IB_SL must be a typed param";
        ASSERT_EQ(ncclParamGetI32(h, &v), ncclSuccess);
        EXPECT_EQ(v, -1) << "legacy -1 is accepted as AUTO";
        std::string sl = DumpEntry("NCCL_IB_SL");
        EXPECT_NE(sl.find("Current value=AUTO set_by=EnvPlugin"), std::string::npos) << sl;

        ASSERT_EQ(ncclParamBind(&h, "NCCL_IB_GID_INDEX"), ncclSuccess);
        ASSERT_EQ(ncclParamGetI32(h, &v), ncclSuccess);
        EXPECT_EQ(v, 3) << "plain integers still go through the bounded base parser";
      },
      {{"NCCL_IB_TC", "auto"}, {"NCCL_IB_SL", "-1"}, {"NCCL_IB_GID_INDEX", "3"}});
  }

  TEST(IbTypedParam, OutOfRangeFallsBackToDefault)
  {
    RUN_ISOLATED_TEST_WITH_ENV(
      "IbTypedParam.OutOfRangeFallsBackToDefault",
      []() {
        ncclParamHandle_t h = nullptr;
        int32_t i32 = 0;
        uint8_t u8 = 0;
        ASSERT_EQ(ncclParamBind(&h, "NCCL_IB_FIFO_TC"), ncclSuccess) << "NCCL_IB_FIFO_TC must be a typed param";
        ASSERT_EQ(ncclParamGetI32(h, &i32), ncclSuccess);
        EXPECT_EQ(i32, -1) << "TC 256 is above 0xFF -> default AUTO";
        std::string fifo = DumpEntry("NCCL_IB_FIFO_TC");
        EXPECT_NE(fifo.find("set_by=Default"), std::string::npos) << fifo;

        ASSERT_EQ(ncclParamBind(&h, "NCCL_IB_RETRY_CNT"), ncclSuccess);
        ASSERT_EQ(ncclParamGetU8(h, &u8), ncclSuccess);
        EXPECT_EQ(u8, 7u) << "retry count 8 is above 7 -> default";

        ASSERT_EQ(ncclParamBind(&h, "NCCL_IB_TIMEOUT"), ncclSuccess);
        ASSERT_EQ(ncclParamGetU8(h, &u8), ncclSuccess);
        EXPECT_EQ(u8, 20u) << "300 does not fit uint8_t -> default, not wrapped to 44";

        ASSERT_EQ(ncclParamBind(&h, "NCCL_IB_QPS_PER_CONNECTION"), ncclSuccess);
        ASSERT_EQ(ncclParamGetI32(h, &i32), ncclSuccess);
        EXPECT_EQ(i32, 1) << "0 QPs is below 1 -> default";

        ASSERT_EQ(ncclParamBind(&h, "NCCL_IB_USE_INLINE"), ncclSuccess);
        bool b = false;
        int len = 0;
        ASSERT_EQ(ncclParamGet(h, &b, sizeof(b), &len), ncclSuccess);
        EXPECT_EQ(len, static_cast<int>(sizeof(bool)));
        EXPECT_TRUE(b) << "boolean spelling TRUE is accepted";
      },
      {{"NCCL_IB_FIFO_TC", "256"},
       {"NCCL_IB_RETRY_CNT", "8"},
       {"NCCL_IB_TIMEOUT", "300"},
       {"NCCL_IB_QPS_PER_CONNECTION", "0"},
       {"NCCL_IB_USE_INLINE", "TRUE"}});
  }

  // Control: a typed param that predates this change binds and reads its env value the
  // same way, so a failure above is the IB settings, not the param API or the harness.
  TEST(IbTypedParam, ControlPreexistingTypedParamBinds)
  {
    RUN_ISOLATED_TEST_WITH_ENV(
      "IbTypedParam.ControlPreexistingTypedParamBinds",
      []() {
        ncclParamHandle_t h = nullptr;
        ASSERT_EQ(ncclParamBind(&h, "NCCL_PARAM_DUMP_ALL"), ncclSuccess);
        bool b = false;
        int len = 0;
        ASSERT_EQ(ncclParamGet(h, &b, sizeof(b), &len), ncclSuccess);
        EXPECT_EQ(len, static_cast<int>(sizeof(bool)));
        EXPECT_TRUE(b);
      },
      {{"NCCL_PARAM_DUMP_ALL", "TRUE"}});
  }
}
