/*************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

// Host-only tests for device/ll128RecvWait.h, the windowed LL128 receive wait that
// Primitives<ProtoLL128>::recvWaitLoad runs when RCCL_LL128_RECV_WINDOW_ENABLE is set.
// No GPU needed: the wait is driven against a simulated wire whose rows turn valid
// at chosen times.
//
//   AllRows* / ReloadWindow* / LoadRows* / VoteRows* / ValidRows* - each helper alone
//   FastPath*         - a slice valid on the first load is loaded once (twice with confirm)
//   NoConfirm*        - rows valid at a vote are not reloaded unless the window spans them
//   Confirm*          - every row's final load is issued after a vote saw it valid
//   RandomSchedules*  - both modes end with every row valid, over many arrival orders
//   KeepWaitingFalse* - a false keepWaiting (abort) ends the wait

#include <algorithm>
#include <array>
#include <cstdint>
#include <random>
#include <vector>

#include "gtest/gtest.h"
#include "ll128RecvWait.h"

namespace RcclUnitTesting
{

namespace
{

constexpr uint64_t kWave64 = ~0ull;
constexpr uint64_t kWave32 = 0xffffffffull;
// One flag lane per 4 lanes, as on gfx9 LL128 (lane % 4 == 3).
constexpr uint64_t kFlagLanes64 = 0x8888888888888888ull;

// One peer's slice as the wait sees it. Every row load is one tick; row r's flags
// read valid from tick validAt[r] on.
template <int Rows>
struct FakeSlice
{
  std::array<int, Rows> validAt{};
  int tick = 0;
  // Registers after the last load of each row.
  std::array<bool, Rows> regValid{};
  std::array<int, Rows> regLoadedAt{};
  // Tick of the first vote that saw each row valid, or -1.
  std::array<int, Rows> seenValidAt{};
  std::vector<int> loads;
  int spins = 0;

  FakeSlice() { seenValidAt.fill(-1); }

  void load(int r)
  {
    regValid[r] = tick >= validAt[r];
    regLoadedAt[r] = tick;
    loads.push_back(r);
    tick++;
  }

  uint64_t ballot(int r)
  {
    if (!regValid[r]) return kWave64 & ~(1ull << 7);
    if (seenValidAt[r] < 0) seenValidAt[r] = tick;
    return kWave64;
  }

  template <bool Confirm>
  void wait(int maxSpins = 1000)
  {
    ll128RecvWaitRows<Rows, Confirm>(
        kWave64, [&](int r) { load(r); }, [&](int r) { return ballot(r); },
        [&](bool needReload) { return needReload && ++spins < maxSpins; });
  }

  int loadsOf(int r) const
  {
    int n = 0;
    for (int x : loads) n += x == r;
    return n;
  }
};

template <int Rows>
std::vector<int> windowRows(uint32_t skip)
{
  std::vector<int> rows;
  ll128LoadRows<Rows>(skip, [&](int r) { rows.push_back(r); });
  return rows;
}

// The reload window of skip, or (-1, -1) when it is empty.
template <int Rows>
std::pair<int, int> window(uint32_t skip)
{
  int lo = -1, hi = -1;
  if (!ll128ReloadWindow<Rows>(skip, lo, hi)) return {-1, -1};
  return {lo, hi};
}

}  // namespace

TEST(Ll128RecvWait, AllRowsHasOneBitPerRow)
{
  EXPECT_EQ(ll128AllRows<1>(), 0x1u);
  EXPECT_EQ(ll128AllRows<4>(), 0xfu);
  EXPECT_EQ(ll128AllRows<8>(), 0xffu);
  EXPECT_EQ(ll128AllRows<31>(), 0x7fffffffu);
  EXPECT_EQ(ll128AllRows<32>(), 0xffffffffu);
}

TEST(Ll128RecvWait, ReloadWindowSpansLowestToHighestPendingRow)
{
  EXPECT_EQ(window<8>(0x00), std::make_pair(0, 8));
  EXPECT_EQ(window<8>(0x0f), std::make_pair(4, 8));
  EXPECT_EQ(window<8>(0xf0), std::make_pair(0, 4));
  // Pending rows 2 and 5: the skipped rows between them are inside the window.
  EXPECT_EQ(window<8>(0xdb), std::make_pair(2, 6));
  // Pending rows 0 and 7 make the window the whole slice.
  EXPECT_EQ(window<8>(0x7e), std::make_pair(0, 8));
  EXPECT_EQ(window<8>(0xef), std::make_pair(4, 5));
}

TEST(Ll128RecvWait, ReloadWindowEmptyWhenEveryRowSkipped)
{
  EXPECT_EQ(window<8>(0xff), std::make_pair(-1, -1));
  // Bits above the slice do not count as pending rows.
  EXPECT_EQ(window<4>(0xffffffffu), std::make_pair(-1, -1));
  EXPECT_EQ(window<4>(0xfffffff0u), std::make_pair(0, 4));
}

TEST(Ll128RecvWait, ReloadWindowHandlesRow31)
{
  EXPECT_EQ(window<32>(0x7fffffffu), std::make_pair(31, 32));
  EXPECT_EQ(window<32>(0x0u), std::make_pair(0, 32));
  EXPECT_EQ(window<32>(0xfffffffeu), std::make_pair(0, 1));
}

TEST(Ll128RecvWait, LoadRowsLoadsTheWindowLowestFirst)
{
  EXPECT_EQ(windowRows<8>(0xdb), (std::vector<int>{2, 3, 4, 5}));
  EXPECT_EQ(windowRows<8>(0x00), (std::vector<int>{0, 1, 2, 3, 4, 5, 6, 7}));
  EXPECT_TRUE(windowRows<8>(0xff).empty());
}

TEST(Ll128RecvWait, VoteRowsTrueOnlyWhenEveryRowVotesTheWholeWarp)
{
  uint64_t ok[4];
  EXPECT_TRUE(ll128VoteRows<4>(ok, kWave64, [](int) { return kWave64; }));
  for (uint64_t v : ok) EXPECT_EQ(v, kWave64);

  // One flag lane of row 2 still stale.
  EXPECT_FALSE(ll128VoteRows<4>(ok, kWave64, [](int r) { return r == 2 ? kWave64 & ~(1ull << 11) : kWave64; }));
  EXPECT_EQ(ok[2], kWave64 & ~(1ull << 11));
  EXPECT_EQ(ok[1], kWave64);
}

TEST(Ll128RecvWait, VoteRowsUsesTheWarpMask)
{
  uint64_t ok[2];
  // On wave32 the ballot has no upper half; a full lower half is a full vote.
  EXPECT_TRUE(ll128VoteRows<2>(ok, kWave32, [](int) { return kWave32; }));
  EXPECT_FALSE(ll128VoteRows<2>(ok, kWave32, [](int r) { return r ? kWave32 : kWave32 >> 1; }));
}

TEST(Ll128RecvWait, ValidRowsMarksRowsWithEveryFlagValid)
{
  const uint64_t notFlagLanes = kWave64 & ~kFlagLanes64;
  uint64_t ok[4] = {kWave64, notFlagLanes, kWave64, kWave64 & ~(1ull << 63)};
  EXPECT_EQ(ll128ValidRows<4>(ok, kWave64), 0x5u);
  uint64_t all[4] = {kWave64, kWave64, kWave64, kWave64};
  EXPECT_EQ(ll128ValidRows<4>(all, kWave64), 0xfu);
}

TEST(Ll128RecvWait, FastPathLoadsOnceWithoutConfirm)
{
  FakeSlice<8> s;
  s.wait<false>();
  EXPECT_EQ(s.loads, (std::vector<int>{0, 1, 2, 3, 4, 5, 6, 7}));
  EXPECT_EQ(s.spins, 0);
}

TEST(Ll128RecvWait, FastPathReloadsEveryRowOnceWithConfirm)
{
  FakeSlice<8> s;
  s.wait<true>();
  ASSERT_EQ(s.loads.size(), 16u);
  for (int r = 0; r < 8; r++) {
    EXPECT_EQ(s.loadsOf(r), 2) << "row " << r;
    EXPECT_GE(s.regLoadedAt[r], s.seenValidAt[r]) << "row " << r;
  }
  EXPECT_EQ(s.spins, 0);
}

TEST(Ll128RecvWait, NoConfirmReloadsOnlyTheLateRows)
{
  // Rows 0-3 valid on the first load; rows 4-7 arrive later.
  FakeSlice<8> s;
  for (int r = 4; r < 8; r++) s.validAt[r] = 40;
  s.wait<false>();
  for (int r = 0; r < 4; r++) EXPECT_EQ(s.loadsOf(r), 1) << "row " << r;
  for (int r = 0; r < 8; r++) EXPECT_TRUE(s.regValid[r]) << "row " << r;
  EXPECT_GT(s.spins, 0);
}

TEST(Ll128RecvWait, NoConfirmReloadsJustTheOneLateRow)
{
  FakeSlice<8> s;
  s.validAt[3] = 20;
  s.wait<false>();
  for (int r = 0; r < 8; r++) {
    if (r != 3) EXPECT_EQ(s.loadsOf(r), 1) << "row " << r;
    EXPECT_TRUE(s.regValid[r]) << "row " << r;
  }
  EXPECT_GT(s.loadsOf(3), 1);
}

TEST(Ll128RecvWait, NoConfirmWindowSpansValidRowsBetweenLateOnes)
{
  // Rows 1 and 6 late: rows 2-5 are valid but inside the window, so they are reloaded with them.
  FakeSlice<8> s;
  s.validAt[1] = s.validAt[6] = 30;
  s.wait<false>();
  EXPECT_EQ(s.loadsOf(0), 1);
  EXPECT_EQ(s.loadsOf(7), 1);
  for (int r = 2; r < 6; r++) EXPECT_GT(s.loadsOf(r), 1) << "row " << r;
  for (int r = 0; r < 8; r++) EXPECT_TRUE(s.regValid[r]) << "row " << r;
}

TEST(Ll128RecvWait, ConfirmLoadsEveryRowAfterAVoteSawItValid)
{
  FakeSlice<8> s;
  s.validAt = {0, 25, 3, 60, 0, 12, 45, 9};
  s.wait<true>();
  for (int r = 0; r < 8; r++) {
    ASSERT_GE(s.seenValidAt[r], 0) << "row " << r;
    EXPECT_GE(s.regLoadedAt[r], s.seenValidAt[r]) << "row " << r;
    EXPECT_TRUE(s.regValid[r]) << "row " << r;
  }
}

template <int Rows, bool Confirm>
void checkRandomSchedules()
{
  std::mt19937 rng(1234 + Rows + 7 * Confirm);
  std::uniform_int_distribution<int> late(0, 4 * Rows);
  for (int trial = 0; trial < 500; trial++) {
    FakeSlice<Rows> s;
    for (int r = 0; r < Rows; r++) s.validAt[r] = trial % 5 == 0 ? 0 : late(rng);
    s.template wait<Confirm>();
    for (int r = 0; r < Rows; r++) {
      ASSERT_TRUE(s.regValid[r]) << "trial " << trial << " row " << r;
      if (Confirm) ASSERT_GE(s.regLoadedAt[r], s.seenValidAt[r]) << "trial " << trial << " row " << r;
    }
    // Without confirm, windows only shrink, so rows outside the first one are loaded once.
    // The first pass loads row r at tick r.
    if (!Confirm) {
      int lo = Rows, hi = -1;
      for (int r = 0; r < Rows; r++) {
        if (s.validAt[r] > r) {
          lo = std::min(lo, r);
          hi = std::max(hi, r);
        }
      }
      for (int r = 0; r < Rows; r++) {
        if (r < lo || r > hi) ASSERT_EQ(s.loadsOf(r), 1) << "trial " << trial << " row " << r;
      }
    }
  }
}

TEST(Ll128RecvWait, RandomSchedulesEndValidWithoutConfirm)
{
  checkRandomSchedules<4, false>();
  checkRandomSchedules<8, false>();
  checkRandomSchedules<32, false>();
}

TEST(Ll128RecvWait, RandomSchedulesEndValidWithConfirm)
{
  checkRandomSchedules<4, true>();
  checkRandomSchedules<8, true>();
  checkRandomSchedules<32, true>();
}

TEST(Ll128RecvWait, KeepWaitingFalseEndsTheWait)
{
  for (bool confirm : {false, true}) {
    FakeSlice<8> s;
    s.validAt[5] = 1 << 30;  // never arrives
    if (confirm) {
      s.wait<true>(10);
    } else {
      s.wait<false>(10);
    }
    EXPECT_EQ(s.spins, 10) << "confirm " << confirm;
    EXPECT_FALSE(s.regValid[5]) << "confirm " << confirm;
  }
}

}  // namespace RcclUnitTesting
