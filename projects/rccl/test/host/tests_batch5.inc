// ===========================================================================
// ncclRedOpCreatePreMulSum_impl (enqueue.cc:4104) / ncclRedOpDestroy_impl (:4148)
//
// A create/destroy PAIR over a free list embedded in comm->userRedOps, with
// capacity doubling. Testing them together is what makes the free list
// observable: create -> destroy -> create must REUSE the slot, which no
// single-call test can show.
//
// Public API surface (ncclRedOpCreatePreMulSum / ncclRedOpDestroy).
// ===========================================================================

namespace {
// A comm that passes CommCheck + ncclCommEnsureReady. The real argcheck.cc is
// compiled in as an oracle, so these fields must genuinely satisfy it.
struct RedOpComm {
  std::unique_ptr<ncclComm> storage{new ncclComm{}};
  RedOpComm() {
    ncclComm& c = *storage;
    // CommCheck (argcheck.cc:40) validates startMagic AND endMagic -- the
    // sentinels bracketing the struct, not a single `magic` field. The real
    // argcheck.cc is compiled in as an oracle, so both must be set or every
    // create is rejected with "corrupted comm object detected".
    c.startMagic = NCCL_MAGIC;
    c.endMagic = NCCL_MAGIC;
    c.rank = 0;
    c.nRanks = 8;
    c.userRedOps = nullptr;
    c.userRedOpCapacity = 0;
    c.userRedOpFreeHead = 0;
    c.initState = ncclSuccess;   // ncclCommEnsureReady: already initialised
    c.finalizeCalled = 0;
  }
  ~RedOpComm() { delete[] storage->userRedOps; }
  ncclComm* get() { return storage.get(); }
};
}  // namespace

TEST_F(EnqueueMicrotest, RedOpCreate_FirstCall_GrowsCapacityToFour) {
  // `if (cap < 4) cap = 4` -- the first allocation jumps straight to 4, it does
  // not double from 0.
  RedOpComm rc;
  ncclRedOp_t op = ncclSum;
  float scalar = 2.0f;
  ASSERT_EQ(ncclSuccess, ncclRedOpCreatePreMulSum_impl(&op, &scalar, ncclFloat32,
                                                       ncclScalarHostImmediate, rc.get()));
  EXPECT_EQ(4, rc.get()->userRedOpCapacity);
  EXPECT_NE(nullptr, rc.get()->userRedOps);
}

TEST_F(EnqueueMicrotest, RedOpCreate_ReturnsOpAboveBuiltinRange) {
  // The handle is ncclNumOps + slot index, so it must never collide with a
  // builtin op -- that separation is what ncclRedOpDestroy's first guard relies on.
  RedOpComm rc;
  ncclRedOp_t op = ncclSum;
  float scalar = 2.0f;
  ASSERT_EQ(ncclSuccess, ncclRedOpCreatePreMulSum_impl(&op, &scalar, ncclFloat32,
                                                       ncclScalarHostImmediate, rc.get()));
  EXPECT_GE(int(op), int(ncclNumOps));
}

TEST_F(EnqueueMicrotest, RedOpCreate_HostImmediate_CopiesScalarBytesInline) {
  // residence == HostImmediate memcpy's ncclTypeSize(datatype) bytes INTO
  // scalarArg. Compare the bit pattern, and assert the pointer flag is clear --
  // a mutant that stored the pointer instead would otherwise look similar.
  RedOpComm rc;
  ncclRedOp_t op = ncclSum;
  float scalar = 0.25f;
  ASSERT_EQ(ncclSuccess, ncclRedOpCreatePreMulSum_impl(&op, &scalar, ncclFloat32,
                                                       ncclScalarHostImmediate, rc.get()));
  const int ix = int(ncclUserRedOpMangle(rc.get(), op)) - int(ncclNumOps);
  const ncclUserRedOp& u = rc.get()->userRedOps[ix];
  EXPECT_EQ(ncclDevPreMulSum, u.opFull.op);
  EXPECT_FALSE(u.opFull.scalarArgIsPtr);
  uint32_t want;
  std::memcpy(&want, &scalar, sizeof(want));
  EXPECT_EQ(want, uint32_t(u.opFull.scalarArg & 0xFFFFFFFFull));
  EXPECT_EQ(ncclFloat32, u.datatype);
}

TEST_F(EnqueueMicrotest, RedOpCreate_DeviceResidence_StoresPointerNotBytes) {
  // The other residence arm: scalarArg holds the POINTER and the flag is set.
  RedOpComm rc;
  ncclRedOp_t op = ncclSum;
  double scalar = 3.5;
  ASSERT_EQ(ncclSuccess, ncclRedOpCreatePreMulSum_impl(&op, &scalar, ncclFloat64,
                                                       ncclScalarDevice, rc.get()));
  const int ix = int(ncclUserRedOpMangle(rc.get(), op)) - int(ncclNumOps);
  const ncclUserRedOp& u = rc.get()->userRedOps[ix];
  EXPECT_TRUE(u.opFull.scalarArgIsPtr);
  EXPECT_EQ(reinterpret_cast<uint64_t>(&scalar), u.opFull.scalarArg);
}

TEST_F(EnqueueMicrotest, RedOpCreate_MarksSlotAllocatedViaFreeNext) {
  // freeNext == -1 is the "allocated" marker Destroy checks. Pin it explicitly.
  RedOpComm rc;
  ncclRedOp_t op = ncclSum;
  float s = 1.0f;
  ASSERT_EQ(ncclSuccess, ncclRedOpCreatePreMulSum_impl(&op, &s, ncclFloat32,
                                                       ncclScalarHostImmediate, rc.get()));
  const int ix = int(ncclUserRedOpMangle(rc.get(), op)) - int(ncclNumOps);
  EXPECT_EQ(-1, rc.get()->userRedOps[ix].freeNext);
}

TEST_F(EnqueueMicrotest, RedOpCreate_FourOpsFitThenCapacityDoubles) {
  // Exhaust the initial capacity of 4, then the 5th create must double to 8 and
  // PRESERVE the existing four (the memcpy at :4116). Asserting the handles stay
  // distinct is what catches a dropped or short memcpy.
  RedOpComm rc;
  float s = 1.0f;
  std::vector<ncclRedOp_t> ops;
  for (int i = 0; i < 4; ++i) {
    ncclRedOp_t op = ncclSum;
    ASSERT_EQ(ncclSuccess, ncclRedOpCreatePreMulSum_impl(&op, &s, ncclFloat32,
                                                         ncclScalarHostImmediate, rc.get()))
        << "i=" << i;
    ops.push_back(op);
  }
  EXPECT_EQ(4, rc.get()->userRedOpCapacity);

  ncclRedOp_t fifth = ncclSum;
  ASSERT_EQ(ncclSuccess, ncclRedOpCreatePreMulSum_impl(&fifth, &s, ncclFloat32,
                                                       ncclScalarHostImmediate, rc.get()));
  EXPECT_EQ(8, rc.get()->userRedOpCapacity) << "capacity must DOUBLE, not reset";
  ops.push_back(fifth);

  // Every handle distinct, and every slot still marked allocated.
  std::vector<int> seen;
  for (auto op : ops) {
    int ix = int(ncclUserRedOpMangle(rc.get(), op)) - int(ncclNumOps);
    EXPECT_EQ(-1, rc.get()->userRedOps[ix].freeNext) << "slot " << ix << " lost its marker";
    seen.push_back(ix);
  }
  std::sort(seen.begin(), seen.end());
  EXPECT_EQ(seen.end(), std::unique(seen.begin(), seen.end())) << "handles must be distinct";
}

TEST_F(EnqueueMicrotest, RedOpCreateDestroy_RoundTrip_ReusesTheFreedSlot) {
  // THE point of the pair: destroy pushes the slot onto the free list, so the
  // next create pops the SAME slot. A destroy that forgot to update
  // userRedOpFreeHead would leak the slot and hand back a different index.
  RedOpComm rc;
  float s = 1.0f;
  ncclRedOp_t first = ncclSum;
  ASSERT_EQ(ncclSuccess, ncclRedOpCreatePreMulSum_impl(&first, &s, ncclFloat32,
                                                       ncclScalarHostImmediate, rc.get()));
  const int ixFirst = int(ncclUserRedOpMangle(rc.get(), first)) - int(ncclNumOps);

  ASSERT_EQ(ncclSuccess, ncclRedOpDestroy_impl(first, rc.get()));
  EXPECT_NE(-1, rc.get()->userRedOps[ixFirst].freeNext) << "destroy must clear the marker";

  ncclRedOp_t second = ncclSum;
  ASSERT_EQ(ncclSuccess, ncclRedOpCreatePreMulSum_impl(&second, &s, ncclFloat32,
                                                       ncclScalarHostImmediate, rc.get()));
  const int ixSecond = int(ncclUserRedOpMangle(rc.get(), second)) - int(ncclNumOps);
  EXPECT_EQ(ixFirst, ixSecond) << "the freed slot must be reused";
  EXPECT_EQ(4, rc.get()->userRedOpCapacity) << "and no growth should have been needed";
}

TEST_F(EnqueueMicrotest, RedOpDestroy_BuiltinOp_RejectedWithWarning) {
  // Guard 1: builtin ops are not destroyable. Assert the WARN too -- the return
  // code alone cannot distinguish this guard from the three below it.
  RedOpComm rc;
  for (auto op : {ncclSum, ncclProd, ncclMin, ncclMax, ncclAvg}) {
    ncclResult_t r = ncclSuccess;
    const std::string log = RcclUnitTesting::CaptureLog([&] {
      r = ncclRedOpDestroy_impl(op, rc.get());
    });
    EXPECT_EQ(ncclInvalidArgument, r) << "op=" << int(op);
    EXPECT_TRUE(RcclUnitTesting::LogHas(log, "operator is a NCCL builtin"))
        << "op=" << int(op) << " actual log:\n" << log;
  }
}

TEST_F(EnqueueMicrotest, RedOpDestroy_NullComm_RejectedWithWarning) {
  // Guard 3. Reached only by an op ABOVE the builtin range, so this also proves
  // guard 1 did not swallow it.
  RedOpComm rc;
  float s = 1.0f;
  ncclRedOp_t op = ncclSum;
  ASSERT_EQ(ncclSuccess, ncclRedOpCreatePreMulSum_impl(&op, &s, ncclFloat32,
                                                       ncclScalarHostImmediate, rc.get()));
  ncclResult_t r = ncclSuccess;
  const std::string log = RcclUnitTesting::CaptureLog([&] {
    r = ncclRedOpDestroy_impl(op, nullptr);
  });
  EXPECT_EQ(ncclInvalidArgument, r);
  EXPECT_TRUE(RcclUnitTesting::LogHas(log, "invalid communicator")) << "actual log:\n" << log;
}

TEST_F(EnqueueMicrotest, RedOpDestroy_UnknownToThisComm_RejectedWithWarning) {
  // Guard 4, first half: an index past this comm's capacity.
  RedOpComm rc;
  float s = 1.0f;
  ncclRedOp_t op = ncclSum;
  ASSERT_EQ(ncclSuccess, ncclRedOpCreatePreMulSum_impl(&op, &s, ncclFloat32,
                                                       ncclScalarHostImmediate, rc.get()));
  // Fabricate a handle far beyond the 4 allocated slots.
  auto bogus = static_cast<ncclRedOp_t>(int(op) + 1000);
  ncclResult_t r = ncclSuccess;
  const std::string log = RcclUnitTesting::CaptureLog([&] {
    r = ncclRedOpDestroy_impl(bogus, rc.get());
  });
  EXPECT_EQ(ncclInvalidArgument, r);
  EXPECT_TRUE(RcclUnitTesting::LogHas(log, "operator unknown to this communicator"))
      << "actual log:\n" << log;
}

TEST_F(EnqueueMicrotest, RedOpDestroy_DoubleFree_RejectedWithWarning) {
  // Guard 4, second half: freeNext != -1 means the slot is already on the free
  // list. This is the double-free rejection and it shares a message with the
  // out-of-range case, so the two tests together pin both halves of the ||.
  RedOpComm rc;
  float s = 1.0f;
  ncclRedOp_t op = ncclSum;
  ASSERT_EQ(ncclSuccess, ncclRedOpCreatePreMulSum_impl(&op, &s, ncclFloat32,
                                                       ncclScalarHostImmediate, rc.get()));
  ASSERT_EQ(ncclSuccess, ncclRedOpDestroy_impl(op, rc.get()));

  ncclResult_t r = ncclSuccess;
  const std::string log = RcclUnitTesting::CaptureLog([&] {
    r = ncclRedOpDestroy_impl(op, rc.get());
  });
  EXPECT_EQ(ncclInvalidArgument, r) << "second destroy must be rejected";
  EXPECT_TRUE(RcclUnitTesting::LogHas(log, "operator unknown to this communicator"))
      << "actual log:\n" << log;
}

TEST_F(EnqueueMicrotest, RedOpCreate_NullComm_RejectedByCommCheck) {
  // The real argcheck.cc oracle is compiled in, so this exercises production's
  // own validation rather than a fake's.
  ncclRedOp_t op = ncclSum;
  float s = 1.0f;
  EXPECT_EQ(ncclInvalidArgument,
            ncclRedOpCreatePreMulSum_impl(&op, &s, ncclFloat32,
                                          ncclScalarHostImmediate, nullptr));
}

// ===========================================================================
// rcclEffectiveP2pBatchEnable (enqueue.cc:1170)
// Four arms gating P2P batching for the whole file. The user override is
// checked FIRST, so it must win over both the node count and the arch.
// ===========================================================================

namespace {
struct BatchComm {
  std::unique_ptr<ncclTopoSystem> topo{new ncclTopoSystem{}};
  std::unique_ptr<ncclComm> comm{new ncclComm{}};
  BatchComm(int nNodes, const char* gcn) {
    comm->nNodes = nNodes;
    comm->topo = topo.get();
    topo->nodes[GPU].count = 1;
    std::snprintf(topo->nodes[GPU].nodes[0].gpu.gcn,
                  sizeof(topo->nodes[GPU].nodes[0].gpu.gcn), "%s", gcn);
  }
  ncclComm* get() { return comm.get(); }
};

void SetBatchParam(int64_t v) {
  g_loadParam = [v](const char* env, int64_t deft) {
    return std::strcmp(env, "RCCL_P2P_BATCH_ENABLE") == 0 ? v : deft;
  };
}
}  // namespace

TEST_F(EnqueueMicrotest, EffectiveP2pBatchEnable_UserOverrideWinsOverEverything) {
  // userInput >= 0 returns immediately -- single node and a non-gfx950 arch,
  // both of which would otherwise force 0.
  BatchComm bc(/*nNodes=*/1, "gfx942");
  SetBatchParam(1);
  EXPECT_EQ(1, rcclEffectiveP2pBatchEnable(bc.get()));
  SetBatchParam(0);
  EXPECT_EQ(0, rcclEffectiveP2pBatchEnable(bc.get()));
  SetBatchParam(7);
  EXPECT_EQ(7, rcclEffectiveP2pBatchEnable(bc.get())) << "returned verbatim, not clamped";
}

TEST_F(EnqueueMicrotest, EffectiveP2pBatchEnable_SingleNode_IsDisabled) {
  // With the override off (-1), nNodes <= 1 short-circuits to 0 before the arch
  // check -- so even gfx950 gives 0 here.
  BatchComm bc(/*nNodes=*/1, "gfx950");
  SetBatchParam(-1);
  EXPECT_EQ(0, rcclEffectiveP2pBatchEnable(bc.get()));
}

TEST_F(EnqueueMicrotest, EffectiveP2pBatchEnable_MultiNodeGfx950_IsEnabled) {
  // The real gfx950 arm. IsArchMatch comes from the archinfo.cc oracle, so this
  // is production's own matching, not a fake's.
  BatchComm bc(/*nNodes=*/2, "gfx950");
  SetBatchParam(-1);
  EXPECT_EQ(1, rcclEffectiveP2pBatchEnable(bc.get()));
}

TEST_F(EnqueueMicrotest, EffectiveP2pBatchEnable_MultiNodeOtherArch_IsDisabled) {
  // Differential against the test above: same node count, different arch.
  for (const char* gcn : {"gfx942", "gfx90a", "gfx1100"}) {
    BatchComm bc(/*nNodes=*/2, gcn);
    SetBatchParam(-1);
    EXPECT_EQ(0, rcclEffectiveP2pBatchEnable(bc.get())) << "gcn=" << gcn;
  }
}

// ===========================================================================
// getImplicitOrder (enqueue.cc:1997)
// On AMD the CUDA driver-version arm is #if'd out entirely, so only two arms
// are reachable: param-on -> Serial, param-off -> None. Pinning that the AMD
// build cannot return ncclImplicitOrderLaunch is the useful assertion.
// ===========================================================================

TEST_F(EnqueueMicrotest, GetImplicitOrder_ParamDisabled_IsNone) {
  g_loadParam = [](const char* env, int64_t deft) {
    return std::strcmp(env, "LAUNCH_ORDER_IMPLICIT") == 0 ? int64_t(0) : deft;
  };
  auto mode = ncclImplicitOrderLaunch;  // poison
  ASSERT_EQ(ncclSuccess, getImplicitOrder(&mode, /*capturing=*/false));
  EXPECT_EQ(ncclImplicitOrderNone, mode);
}

TEST_F(EnqueueMicrotest, GetImplicitOrder_ParamEnabled_IsSerialOnAmd) {
  g_loadParam = [](const char* env, int64_t deft) {
    return std::strcmp(env, "LAUNCH_ORDER_IMPLICIT") == 0 ? int64_t(1) : deft;
  };
  auto mode = ncclImplicitOrderNone;  // poison
  ASSERT_EQ(ncclSuccess, getImplicitOrder(&mode, /*capturing=*/false));
  EXPECT_EQ(ncclImplicitOrderSerial, mode);
}

TEST_F(EnqueueMicrotest, GetImplicitOrder_CapturingIsIrrelevantOnAmd) {
  // The `capturing` parameter only matters inside the #if'd-out CUDA block, so
  // on AMD both values must give the same answer. Pins that the AMD arm really
  // is unconditional -- if the #if ever changes, this fails.
  g_loadParam = [](const char* env, int64_t deft) {
    return std::strcmp(env, "LAUNCH_ORDER_IMPLICIT") == 0 ? int64_t(1) : deft;
  };
  auto a = ncclImplicitOrderNone;
  auto b = ncclImplicitOrderNone;
  ASSERT_EQ(ncclSuccess, getImplicitOrder(&a, /*capturing=*/true));
  ASSERT_EQ(ncclSuccess, getImplicitOrder(&b, /*capturing=*/false));
  EXPECT_EQ(a, b);
  EXPECT_EQ(ncclImplicitOrderSerial, a);
}
