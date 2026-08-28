/*************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

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
      {ncclUint8, false},{ncclUint32, false},{ncclUint64, false}};
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

// ---------------------------------------------------------------------------
// NOT A BUG -- investigated and cleared. The ncclAvg inner datatype switch
// (:3232-3274) has no `default:` arm, which reads like an uninitialised-`op`
// hazard. It is not reachable:
//
//   ncclTypeSize (collectives.h:61-82) returns -1 for any type outside
//   {Int8,Uint8,Float8e4m3,Float8e5m2,Float16,Bfloat16,Int32,Uint32,Float32,
//    Int64,Uint64,Float64} -- and the Avg switch names EXACTLY that same set.
//   So for any type the switch would miss, `nbits = 8*ncclTypeSize(dt)` is
//   negative and the guard at :3210 returns ncclInvalidArgument first.
//
// The guard is therefore load-bearing for two reasons, and the test below pins
// it: it prevents UB in the `>> (64 - nbits)` shift AND it is what makes the
// missing default unreachable. If ncclTypeSize ever gains a type the Avg switch
// does not handle, this coupling breaks silently -- which is worth knowing, and
// is reported in the PR body rather than pinned as a fake bug.
// ---------------------------------------------------------------------------
TEST_F(EnqueueMicrotest, HostToDevRedOp_UnknownDatatype_RejectedByWidthGuard) {
  // :3210 `if (nbits <= 0) return ncclInvalidArgument`. Without it, :3211's
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
           "ncclAvg switch at enqueue.cc:3232-3274, leaving op uninitialised";
  }
}
