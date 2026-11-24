/*************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

// Reduction ops: hostToDevRedOp, ncclRedOpCreatePreMulSum_impl and ncclRedOpDestroy_impl.
// Included into enqueue-test.cc's translation unit; see README.md.

// ===========================================================================
// hostToDevRedOp (enqueue.cc:3186) -- maps a host ncclRedOp_t + datatype onto
// the device-side op and its scalar argument.
//
// The scalarArg for Min/Max is a XOR MASK, not a value, computed from the
// datatype's bit width:
//     allBits = ~0 >> (64 - nbits)
//     signBit = allBits ^ (allBits >> 1)
//     mask    = (signed ? signBit : 0) ^ (op==Max ? allBits : 0)
// A return-code assertion cannot see any of that, so every test below asserts
// the exact mask. This is the function where mutation earns its keep.
// ===========================================================================

namespace {
// Poison, not zero: scalarArg is written by every arm, so a zero-initialised
// out-param cannot distinguish "written 0" from "not written".
constexpr uint64_t kRedOpPoison = 0xDEADBEEFDEADBEEFull;

ncclDevRedOpFull MakeRedOpOut() {
  ncclDevRedOpFull o{};
  o.scalarArg = kRedOpPoison;
  o.scalarArgIsPtr = true;   // every path must clear this
  return o;
}

// Expected mask, recomputed independently of the production expression so a
// mutated production formula does not also mutate the oracle.
uint64_t ExpectMinMaxMask(int nbits, bool isSigned, bool isMax) {
  uint64_t allBits = (nbits >= 64) ? ~0ull : ((1ull << nbits) - 1);
  uint64_t signBit = allBits ^ (allBits >> 1);
  uint64_t m = 0;
  if (isSigned) m ^= signBit;
  if (isMax) m ^= allBits;
  return m;
}
// MEASURED: sizeof(ncclComm) is ~3.8 MB. These tests only need nRanks set, but a
// stack instance is still 3.8 MB of frame -- and AvgScalesWithRankCount needs two
// at once, which alone exceeds a default 8 MB stack. Heap-allocate, as
// CostComm/ChunkComm/RankComm/BatchPlanComm/FinishComm/RedOpComm all do.
class AvgComm {
 public:
  explicit AvgComm(int nRanks) : storage_(new ncclComm{}) { storage_->nRanks = nRanks; }
  ncclComm* get() { return storage_.get(); }
 private:
  std::unique_ptr<ncclComm> storage_;
};
}  // namespace

TEST_F(EnqueueMicrotest, HostToDevRedOp_Sum_MapsToDevSumAndClearsPtrFlag) {
  auto out = MakeRedOpOut();
  ASSERT_EQ(ncclSuccess, hostToDevRedOp(&out, ncclSum, ncclFloat32, nullptr));
  EXPECT_EQ(ncclDevSum, out.op);
  EXPECT_FALSE(out.scalarArgIsPtr);
  EXPECT_EQ(ncclSum, out.proxyOp) << "proxyOp must carry the ORIGINAL host op";
}

TEST_F(EnqueueMicrotest, HostToDevRedOp_Prod_MapsToDevProd) {
  auto out = MakeRedOpOut();
  ASSERT_EQ(ncclSuccess, hostToDevRedOp(&out, ncclProd, ncclFloat32, nullptr));
  EXPECT_EQ(ncclDevProd, out.op);
  EXPECT_EQ(ncclProd, out.proxyOp);
}

TEST_F(EnqueueMicrotest, HostToDevRedOp_MinUnsigned_MaskIsZero) {
  // Unsigned + Min: neither XOR term applies, so the mask is exactly 0. This is
  // the one case where poisoning the out-param matters -- without it, "wrote 0"
  // and "wrote nothing" look identical.
  for (auto dt : {ncclUint8, ncclUint32, ncclUint64}) {
    auto out = MakeRedOpOut();
    ASSERT_EQ(ncclSuccess, hostToDevRedOp(&out, ncclMin, dt, nullptr));
    EXPECT_EQ(ncclDevMinMax, out.op);
    EXPECT_EQ(0u, out.scalarArg) << "dtype=" << int(dt);
  }
}

TEST_F(EnqueueMicrotest, HostToDevRedOp_MaxUnsigned_MaskIsAllBitsOfThatWidth) {
  // Width-dependent: 8/32/64-bit types must give DIFFERENT masks. A hardcoded
  // 64-bit allBits would pass ncclUint64 and fail the other two.
  const struct { ncclDataType_t dt; int nbits; } kCases[] = {
      {ncclUint8, 8}, {ncclUint32, 32}, {ncclUint64, 64}};
  for (auto c : kCases) {
    auto out = MakeRedOpOut();
    ASSERT_EQ(ncclSuccess, hostToDevRedOp(&out, ncclMax, c.dt, nullptr));
    EXPECT_EQ(ExpectMinMaxMask(c.nbits, /*signed=*/false, /*max=*/true), out.scalarArg)
        << "dtype=" << int(c.dt) << " nbits=" << c.nbits;
  }
}

TEST_F(EnqueueMicrotest, HostToDevRedOp_MinSigned_MaskIsSignBitOnly) {
  const struct { ncclDataType_t dt; int nbits; } kCases[] = {
      {ncclInt8, 8}, {ncclInt32, 32}, {ncclInt64, 64}};
  for (auto c : kCases) {
    auto out = MakeRedOpOut();
    ASSERT_EQ(ncclSuccess, hostToDevRedOp(&out, ncclMin, c.dt, nullptr));
    EXPECT_EQ(ExpectMinMaxMask(c.nbits, /*signed=*/true, /*max=*/false), out.scalarArg)
        << "dtype=" << int(c.dt);
    // Differential vs the unsigned case: the signed mask must NOT be zero.
    EXPECT_NE(0u, out.scalarArg) << "signed Min must set the sign bit";
  }
}

TEST_F(EnqueueMicrotest, HostToDevRedOp_MaxSigned_MaskIsSignBitXorAllBits) {
  const struct { ncclDataType_t dt; int nbits; } kCases[] = {
      {ncclInt8, 8}, {ncclInt32, 32}, {ncclInt64, 64}};
  for (auto c : kCases) {
    auto out = MakeRedOpOut();
    ASSERT_EQ(ncclSuccess, hostToDevRedOp(&out, ncclMax, c.dt, nullptr));
    EXPECT_EQ(ExpectMinMaxMask(c.nbits, /*signed=*/true, /*max=*/true), out.scalarArg)
        << "dtype=" << int(c.dt);
  }
}

TEST_F(EnqueueMicrotest, HostToDevRedOp_SignedMaxDiffersFromUnsignedMax) {
  // Pins that the `datatype == ncclInt8|Int32|Int64` guard actually gates the
  // signBit XOR: deleting it makes these two identical.
  auto s = MakeRedOpOut();
  auto u = MakeRedOpOut();
  ASSERT_EQ(ncclSuccess, hostToDevRedOp(&s, ncclMax, ncclInt32, nullptr));
  ASSERT_EQ(ncclSuccess, hostToDevRedOp(&u, ncclMax, ncclUint32, nullptr));
  EXPECT_NE(s.scalarArg, u.scalarArg);
}

TEST_F(EnqueueMicrotest, HostToDevRedOp_MinAndMaxDifferByAllBits) {
  // Pins the `(op == ncclMax) ? allBits : 0` term specifically.
  auto mn = MakeRedOpOut();
  auto mx = MakeRedOpOut();
  ASSERT_EQ(ncclSuccess, hostToDevRedOp(&mn, ncclMin, ncclUint32, nullptr));
  ASSERT_EQ(ncclSuccess, hostToDevRedOp(&mx, ncclMax, ncclUint32, nullptr));
  EXPECT_EQ(0xFFFFFFFFull, mn.scalarArg ^ mx.scalarArg);
}

TEST_F(EnqueueMicrotest, HostToDevRedOp_ProxyOpPreservesMinVsMax) {
  // Min and Max share one device op (ncclDevMinMax); only proxyOp keeps them
  // apart downstream. A mutant that assigns proxyOp = opFull->op survives every
  // scalarArg assertion but dies here.
  auto mn = MakeRedOpOut();
  auto mx = MakeRedOpOut();
  ASSERT_EQ(ncclSuccess, hostToDevRedOp(&mn, ncclMin, ncclInt32, nullptr));
  ASSERT_EQ(ncclSuccess, hostToDevRedOp(&mx, ncclMax, ncclInt32, nullptr));
  EXPECT_EQ(mn.op, mx.op) << "both map to ncclDevMinMax";
  EXPECT_NE(mn.proxyOp, mx.proxyOp) << "but proxyOp must still distinguish them";
}

TEST_F(EnqueueMicrotest, HostToDevRedOp_AvgIntegral_UsesSumPostDivWithRankCountAndSignFlag) {
  // The Avg integral arm packs TWO things into scalarArg:
  //     u64 = comm->nRanks << 1 | datatype_signed
  // so the low bit is the signedness and the rest is the rank count.
  AvgComm comm(8);

  auto si = MakeRedOpOut();
  ASSERT_EQ(ncclSuccess, hostToDevRedOp(&si, ncclAvg, ncclInt32, comm.get()));
  EXPECT_EQ(ncclDevSumPostDiv, si.op);
  EXPECT_EQ((8ull << 1) | 1ull, si.scalarArg) << "signed -> low bit set";

  auto ui = MakeRedOpOut();
  ASSERT_EQ(ncclSuccess, hostToDevRedOp(&ui, ncclAvg, ncclUint32, comm.get()));
  EXPECT_EQ(ncclDevSumPostDiv, ui.op);
  EXPECT_EQ((8ull << 1) | 0ull, ui.scalarArg) << "unsigned -> low bit clear";
}

TEST_F(EnqueueMicrotest, HostToDevRedOp_AvgIntegral_SignedFallthroughCoversAllIntWidths) {
  // The signed cases deliberately FALL THROUGH into the unsigned ones (:3240
  // "no break, we want to fall through"). If a `break` were added, the signed
  // types would leave op unassigned. Sweep all six integral types.
  AvgComm comm(4);
  const struct { ncclDataType_t dt; bool sgn; } kCases[] = {
      {ncclInt8, true},  {ncclInt32, true},  {ncclInt64, true},
      {ncclUint8, false}, {ncclUint32, false}, {ncclUint64, false}};
  for (auto c : kCases) {
    auto out = MakeRedOpOut();
    ASSERT_EQ(ncclSuccess, hostToDevRedOp(&out, ncclAvg, c.dt, comm.get())) << "dtype=" << int(c.dt);
    EXPECT_EQ(ncclDevSumPostDiv, out.op) << "dtype=" << int(c.dt);
    EXPECT_EQ((4ull << 1) | (c.sgn ? 1ull : 0ull), out.scalarArg) << "dtype=" << int(c.dt);
  }
}

TEST_F(EnqueueMicrotest, HostToDevRedOp_AvgFloat32_UsesPreMulSumWithReciprocal) {
  // Float Avg becomes PreMulSum with scalar 1/nRanks, bit-pattern compared.
  AvgComm comm(4);
  auto out = MakeRedOpOut();
  ASSERT_EQ(ncclSuccess, hostToDevRedOp(&out, ncclAvg, ncclFloat32, comm.get()));
  EXPECT_EQ(ncclDevPreMulSum, out.op);
  float want = float(1.0 / 4);
  uint32_t wantBits;
  std::memcpy(&wantBits, &want, sizeof(wantBits));
  EXPECT_EQ(wantBits, uint32_t(out.scalarArg & 0xFFFFFFFFull));
  EXPECT_FALSE(out.scalarArgIsPtr) << "the scalar is inline, not a pointer";
}

TEST_F(EnqueueMicrotest, HostToDevRedOp_AvgFloat64_UsesPreMulSumWithReciprocal) {
  AvgComm comm(8);
  auto out = MakeRedOpOut();
  ASSERT_EQ(ncclSuccess, hostToDevRedOp(&out, ncclAvg, ncclFloat64, comm.get()));
  EXPECT_EQ(ncclDevPreMulSum, out.op);
  double want = 1.0 / 8;
  uint64_t wantBits;
  std::memcpy(&wantBits, &want, sizeof(wantBits));
  EXPECT_EQ(wantBits, out.scalarArg);
}

TEST_F(EnqueueMicrotest, HostToDevRedOp_AvgScalesWithRankCount) {
  // Differential: a mutant hardcoding a divisor passes one rank count only.
  AvgComm c4(4);
  AvgComm c8(8);
  auto a = MakeRedOpOut();
  auto b = MakeRedOpOut();
  ASSERT_EQ(ncclSuccess, hostToDevRedOp(&a, ncclAvg, ncclFloat32, c4.get()));
  ASSERT_EQ(ncclSuccess, hostToDevRedOp(&b, ncclAvg, ncclFloat32, c8.get()));
  EXPECT_NE(a.scalarArg, b.scalarArg);
}

TEST_F(EnqueueMicrotest, HostToDevRedOp_AvgFloat8_PacksFloatReciprocalNotFp8Bits) {
  // AICOMRCCL-1945. The fp8 arms used to pack 1/nRanks as a host (always OCP) fp8 byte,
  // which gfx942 device code decodes as FNUZ, i.e. at half the value. The rank counts are
  // ones an fp8 scalar cannot carry at all: 1/3 and 1/384 are not representable, and
  // 1/1024 rounds to zero in OCP e4m3. The full 64 bits are compared, so no stray fp8 byte
  // or poison may survive above the float.
  for (int nRanks : {3, 8, 384, 1024}) {
    AvgComm comm(nRanks);
    float want = float(1.0 / nRanks);
    uint32_t wantBits;
    std::memcpy(&wantBits, &want, sizeof(wantBits));
    for (auto dt : {ncclFloat8e4m3, ncclFloat8e5m2}) {
      auto out = MakeRedOpOut();
      ASSERT_EQ(ncclSuccess, hostToDevRedOp(&out, ncclAvg, dt, comm.get())) << "dtype=" << int(dt);
      EXPECT_EQ(ncclDevPreMulSum, out.op) << "dtype=" << int(dt);
      EXPECT_EQ(uint64_t(wantBits), out.scalarArg) << "dtype=" << int(dt) << " nRanks=" << nRanks;
      EXPECT_FALSE(out.scalarArgIsPtr) << "dtype=" << int(dt);
    }
  }
}

TEST_F(EnqueueMicrotest, Fp8DeviceIsFnuz_MirrorsTheTypedefSelection) {
  // Mirrors rccl_float8.h: only these take the OCP typedef in device code.
  EXPECT_FALSE(rcclFp8DeviceIsFnuz("gfx950"));
  EXPECT_FALSE(rcclFp8DeviceIsFnuz("gfx1170"));
  EXPECT_FALSE(rcclFp8DeviceIsFnuz("gfx1171"));
  EXPECT_FALSE(rcclFp8DeviceIsFnuz("gfx1200"));
  EXPECT_FALSE(rcclFp8DeviceIsFnuz("gfx1201"));
  EXPECT_FALSE(rcclFp8DeviceIsFnuz("gfx1250"));
  EXPECT_TRUE(rcclFp8DeviceIsFnuz("gfx942")) << "explicit FNUZ typedef";
  EXPECT_TRUE(rcclFp8DeviceIsFnuz("gfx90a")) << "software fallback, which is FNUZ";
  EXPECT_TRUE(rcclFp8DeviceIsFnuz(nullptr)) << "an unknown arch lands on the software fallback";
  // comm->archName carries target features, and IsArchMatch is a prefix compare.
  EXPECT_FALSE(rcclFp8DeviceIsFnuz("gfx950:sramecc+:xnack-"));
  EXPECT_TRUE(rcclFp8DeviceIsFnuz("gfx942:sramecc+:xnack-"));
}

TEST_F(EnqueueMicrotest, Fp8ToFloat_DecodesInTheRequestedEncoding) {
  // 0x38 separates all four encodings: OCP and FNUZ differ by one in exponent bias, and
  // e4m3 and e5m2 place the exponent differently.
  EXPECT_EQ(1.0f, rcclFp8ToFloat(0x38, /*isE5m2=*/false, /*fnuz=*/false));
  EXPECT_EQ(0.5f, rcclFp8ToFloat(0x38, /*isE5m2=*/false, /*fnuz=*/true));
  EXPECT_EQ(0.5f, rcclFp8ToFloat(0x38, /*isE5m2=*/true, /*fnuz=*/false));
  EXPECT_EQ(0.25f, rcclFp8ToFloat(0x38, /*isE5m2=*/true, /*fnuz=*/true));
}

// ---------------------------------------------------------------------------
// NOT A BUG -- investigated and cleared. The ncclAvg inner datatype switch
// (:3235-3275) has no `default:` arm, which reads like an uninitialised-`op`
// hazard. It is not reachable:
//
//   ncclTypeSize (collectives.h:61-82) returns -1 for any type outside
//   {Int8,Uint8,Float8e4m3,Float8e5m2,Float16,Bfloat16,Int32,Uint32,Float32,
//    Int64,Uint64,Float64} -- and the Avg switch names EXACTLY that same set.
//   So for any type the switch would miss, `nbits = 8*ncclTypeSize(dt)` is
//   negative and the guard at :3211 returns ncclInvalidArgument first.
//
// The guard is therefore load-bearing for two reasons, and the test below pins
// it: it prevents UB in the `>> (64 - nbits)` shift AND it is what makes the
// missing default unreachable. If ncclTypeSize ever gains a type the Avg switch
// does not handle, this coupling breaks silently -- which is worth knowing, and
// is reported in the PR body rather than pinned as a fake bug.
// ---------------------------------------------------------------------------
TEST_F(EnqueueMicrotest, HostToDevRedOp_UnknownDatatype_RejectedByWidthGuard) {
  // :3211 `if (nbits <= 0) return ncclInvalidArgument`. Without it, :3212's
  // `uint64_t(-1) >> (64 - nbits)` is UB for nbits <= 0.
  const auto kBogus = static_cast<ncclDataType_t>(ncclNumTypes + 50);
  ASSERT_LE(ncclTypeSize(kBogus), 0) << "precondition: this id must be unknown";

  for (auto op : {ncclSum, ncclProd, ncclMin, ncclMax, ncclAvg}) {
    auto out = MakeRedOpOut();
    EXPECT_EQ(ncclInvalidArgument, hostToDevRedOp(&out, op, kBogus, nullptr))
        << "op=" << int(op);
  }
}

TEST_F(EnqueueMicrotest, HostToDevRedOp_AvgSwitchCoversEveryTypeNcclTypeSizeAccepts) {
  // The coupling described above, as an executable assertion: for EVERY datatype
  // that ncclTypeSize accepts, the Avg arm must assign a real op. If someone adds
  // a datatype to ncclTypeSize but forgets the Avg switch, this fails here rather
  // than shipping an uninitialised op.
  AvgComm comm(4);
  const auto kSentinel = static_cast<ncclDevRedOp_t>(0x7E);
  for (int i = 0; i < int(ncclNumTypes); ++i) {
    const auto dt = static_cast<ncclDataType_t>(i);
    if (ncclTypeSize(dt) <= 0) continue;  // rejected by the width guard anyway
    auto out = MakeRedOpOut();
    out.op = kSentinel;
    ASSERT_EQ(ncclSuccess, hostToDevRedOp(&out, ncclAvg, dt, comm.get())) << "dtype=" << i;
    EXPECT_NE(kSentinel, out.op)
        << "dtype=" << i << " is accepted by ncclTypeSize but not handled by the "
           "ncclAvg switch at enqueue.cc:3235-3275, leaving op uninitialised";
  }
}

// ===========================================================================
// ncclRedOpCreatePreMulSum_impl (enqueue.cc:4103) / ncclRedOpDestroy_impl (:4148)
//
// A create/destroy PAIR over a free list embedded in comm->userRedOps, with
// capacity doubling. Testing them together is what makes the free list
// observable: create -> destroy -> create must REUSE the slot, which no
// single-call test can show.
//
// Public API surface (ncclRedOpCreatePreMulSum / ncclRedOpDestroy).
// ===========================================================================

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

TEST_F(EnqueueMicrotest, RedOpCreate_HostImmediateFp8_DecodesToFloatWithTheDeviceEncoding) {
  // FuncPreMulSum<fp8> reads float bits, so the user's byte is promoted here. It must be
  // read in the device's encoding, as the device reads the payload and an ncclScalarDevice
  // scalar; the host typedef would be off by 2x wherever the device is FNUZ.
  const uint8_t byte = 0x38;
  const struct { ncclDataType_t dt; const char* arch; float want; } kCases[] = {
      {ncclFloat8e4m3, "gfx950", 1.0f},  {ncclFloat8e4m3, "gfx942", 0.5f},
      {ncclFloat8e4m3, "gfx90a", 0.5f},  {ncclFloat8e4m3, "gfx1201:xnack-", 1.0f},
      {ncclFloat8e5m2, "gfx950", 0.5f},  {ncclFloat8e5m2, "gfx942", 0.25f}};
  for (auto c : kCases) {
    RedOpComm rc;
    rc.get()->archName = const_cast<char*>(c.arch);
    ncclRedOp_t op = ncclSum;
    ASSERT_EQ(ncclSuccess, ncclRedOpCreatePreMulSum_impl(&op, const_cast<uint8_t*>(&byte), c.dt,
                                                         ncclScalarHostImmediate, rc.get()))
        << "arch=" << c.arch;
    const int ix = int(ncclUserRedOpMangle(rc.get(), op)) - int(ncclNumOps);
    const ncclUserRedOp& u = rc.get()->userRedOps[ix];
    EXPECT_FALSE(u.opFull.scalarArgIsPtr) << "arch=" << c.arch;
    uint32_t want;
    std::memcpy(&want, &c.want, sizeof(want));
    EXPECT_EQ(uint64_t(want), u.opFull.scalarArg) << "dtype=" << int(c.dt) << " arch=" << c.arch;
    rc.get()->archName = nullptr;  // not owned here
  }
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
  // PRESERVE the existing four (the memcpy at :4115). Asserting the handles stay
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
// Seams whose comments promised a tested rejection path. Driving them here so
// the promise holds; the remaining declared-but-undriven seams are marked in
// enqueue_test_deps.h as link-floor-only rather than left to imply coverage.
// ===========================================================================

TEST_F(EnqueueMicrotest, RedOpCreate_CommNotReady_IsRejected) {
  // g_commEnsureReadyResult is documented as "controllable so a test can
  // exercise the not-ready rejection" -- this is that test. RedOpCreate joins
  // the init thread through ncclCommEnsureReady before touching the free list,
  // so a not-ready comm must be rejected with nothing allocated.
  // The code is ncclInternalError, not the ncclInvalidArgument the preceding
  // CommCheck also returns, so a regressed RedOpComm magic cannot fake a pass.
  RedOpComm rc;
  g_commEnsureReadyResult = ncclInternalError;
  ncclRedOp_t op = ncclSum;
  float s = 1.0f;
  EXPECT_EQ(ncclInternalError,
            ncclRedOpCreatePreMulSum_impl(&op, &s, ncclFloat32,
                                          ncclScalarHostImmediate, rc.get()));
  EXPECT_EQ(0, rc.get()->userRedOpCapacity) << "must reject before growing the free list";
}

TEST_F(EnqueueMicrotest, RedOpCreate_RecorderFailure_Propagates) {
  // Production NCCLCHECKs the recorder call at the end of RedOpCreate, which is
  // why the seam exists. Note the contrast with init.cc:369, where the identical
  // call is NOT checked -- pinning this one keeps that asymmetry visible.
  RedOpComm rc;
  g_recorderResult = ncclInternalError;
  ncclRedOp_t op = ncclSum;
  float s = 1.0f;
  EXPECT_EQ(ncclInternalError,
            ncclRedOpCreatePreMulSum_impl(&op, &s, ncclFloat32,
                                          ncclScalarHostImmediate, rc.get()));
}

TEST_F(EnqueueMicrotest, HostToDevRedOp_UserOpWithAnotherDatatype_IsRejected) {
  RedOpComm rc;
  ncclRedOp_t op;
  float scalar = 2.0f;
  ASSERT_EQ(ncclSuccess, ncclRedOpCreatePreMulSum_impl(&op, &scalar, ncclFloat32, ncclScalarHostImmediate, rc.get()));
  ncclDevRedOpFull full = MakeRedOpOut();
  const std::string log = RcclUnitTesting::CaptureLog(
      [&] { EXPECT_EQ(ncclInvalidArgument, hostToDevRedOp(&full, op, ncclFloat64, rc.get())); });
  EXPECT_TRUE(RcclUnitTesting::LogHas(log, "Data type supplied to user-created ncclRedOp_t does not match")) << log;
  ASSERT_EQ(ncclSuccess, hostToDevRedOp(&full, op, ncclFloat32, rc.get()));
  EXPECT_EQ(ncclDevPreMulSum, full.op);
}
