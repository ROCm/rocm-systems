/*************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

// ncclEnqueueCommCheck -- the comm gate every collective entry point calls first.

TEST_F(EnqueueMicrotest, EnqueueCommCheck_ReadyComm_Succeeds) {
  RedOpComm rc;
  EXPECT_EQ(ncclSuccess, ncclEnqueueCommCheck(rc.get(), "AllReduce"));
}

TEST_F(EnqueueMicrotest, EnqueueCommCheck_RevokedComm_IsInvalidUsage) {
  RedOpComm rc;
  rc.get()->revokedFlag = 1;
  ncclResult_t r = ncclSuccess;
  const std::string log = RcclUnitTesting::CaptureLog([&] {
    r = ncclEnqueueCommCheck(rc.get(), "AllReduce");
  });
  EXPECT_EQ(ncclInvalidUsage, r);
  EXPECT_TRUE(RcclUnitTesting::LogHas(log, "communicator was revoked")) << "actual log:\n" << log;
}

TEST_F(EnqueueMicrotest, EnqueueCommCheck_InitInProgress_IsInvalidArgument) {
  RedOpComm rc;
  rc.get()->initState = ncclInProgress;
  ncclResult_t r = ncclSuccess;
  const std::string log = RcclUnitTesting::CaptureLog([&] {
    r = ncclEnqueueCommCheck(rc.get(), "AllReduce");
  });
  EXPECT_EQ(ncclInvalidArgument, r);
  EXPECT_TRUE(RcclUnitTesting::LogHas(log, "is not ready")) << "actual log:\n" << log;
}

TEST_F(EnqueueMicrotest, EnqueueCommCheck_AsyncResultInProgress_IsInvalidArgument) {
  RedOpComm rc;
  rc.get()->asyncResult = ncclInProgress;
  ncclResult_t r = ncclSuccess;
  const std::string log = RcclUnitTesting::CaptureLog([&] {
    r = ncclEnqueueCommCheck(rc.get(), "AllReduce");
  });
  EXPECT_EQ(ncclInvalidArgument, r);
  EXPECT_TRUE(RcclUnitTesting::LogHas(log, "is not ready")) << "actual log:\n" << log;
}
