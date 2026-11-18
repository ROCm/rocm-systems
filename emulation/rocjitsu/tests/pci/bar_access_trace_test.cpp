// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "rocjitsu/vm/amdgpu/pci/bar_access_trace.h"
#include "rocjitsu/vm/amdgpu/pci/register_symbols.h"

#include <gtest/gtest.h>

namespace {

constexpr int kRegisterBar = 5;

// Byte offsets of registers amdgpu reads before IP discovery, derived from the
// dword indices the driver defines in amdgpu_discovery.c.
constexpr uint64_t kRccConfigMemsize = 0xde3 * 4;
constexpr uint64_t kMp0SmnC2pmsg33 = 0x16061 * 4;
constexpr uint64_t kDriverScratch0 = 0x94 * 4;

rocjitsu::RegisterSymbols pre_discovery_symbols() {
  rocjitsu::RegisterSymbols symbols;
  rocjitsu::add_pre_discovery_symbols(symbols, kRegisterBar);
  return symbols;
}

TEST(RegisterSymbols, NamesPreDiscoveryRegistersByByteOffset) {
  const rocjitsu::RegisterSymbols symbols = pre_discovery_symbols();

  EXPECT_EQ(symbols.lookup(kRegisterBar, kRccConfigMemsize), "RCC_CONFIG_MEMSIZE");
  EXPECT_EQ(symbols.lookup(kRegisterBar, kMp0SmnC2pmsg33), "MP0_SMN_C2PMSG_33");
  EXPECT_EQ(symbols.lookup(kRegisterBar, kDriverScratch0), "DRIVER_SCRATCH_0");
}

TEST(RegisterSymbols, DoesNotNameAnUnknownOffsetOrTheWrongBar) {
  const rocjitsu::RegisterSymbols symbols = pre_discovery_symbols();

  EXPECT_TRUE(symbols.lookup(kRegisterBar, 0x12340).empty());
  EXPECT_TRUE(symbols.lookup(0, kRccConfigMemsize).empty());
}

TEST(BarAccessTrace, ReportsNothingWhenEveryAccessIsModeled) {
  const rocjitsu::RegisterSymbols symbols = pre_discovery_symbols();
  rocjitsu::BarAccessTrace trace(symbols);

  trace.record(kRegisterBar, kRccConfigMemsize, 4, /*write=*/false, /*modeled=*/true);

  EXPECT_TRUE(trace.unmodeled_report().empty());
}

TEST(BarAccessTrace, RanksUnmodeledRegistersByUseAndNamesThem) {
  const rocjitsu::RegisterSymbols symbols = pre_discovery_symbols();
  rocjitsu::BarAccessTrace trace(symbols);

  constexpr uint64_t kUnnamedOffset = 0x28a04;
  trace.record(kRegisterBar, kUnnamedOffset, 4, /*write=*/false, /*modeled=*/false);
  for (int i = 0; i < 3; ++i) {
    trace.record(kRegisterBar, kDriverScratch0, 4, /*write=*/false, /*modeled=*/false);
  }

  const std::string report = trace.unmodeled_report();
  const std::size_t scratch_pos = report.find("DRIVER_SCRATCH_0");
  const std::size_t unnamed_pos = report.find("0x00028a04");

  ASSERT_NE(scratch_pos, std::string::npos);
  ASSERT_NE(unnamed_pos, std::string::npos);
  EXPECT_LT(scratch_pos, unnamed_pos) << "the most used register must come first:\n" << report;
  EXPECT_NE(report.find("reads=3"), std::string::npos) << report;
}

TEST(BarAccessTrace, WarnsOnceWhenTheGuestSpinsOnOneRegister) {
  const rocjitsu::RegisterSymbols symbols = pre_discovery_symbols();
  rocjitsu::BarAccessTrace trace(symbols, {.spin_threshold = 8});

  for (int i = 0; i < 64; ++i) {
    trace.record(kRegisterBar, kMp0SmnC2pmsg33, 4, /*write=*/false, /*modeled=*/true);
  }

  EXPECT_EQ(trace.spin_warnings(), 1u);
}

TEST(BarAccessTrace, DoesNotWarnWhenReadsAlternateBetweenRegisters) {
  const rocjitsu::RegisterSymbols symbols = pre_discovery_symbols();
  rocjitsu::BarAccessTrace trace(symbols, {.spin_threshold = 8});

  for (int i = 0; i < 64; ++i) {
    trace.record(kRegisterBar, kMp0SmnC2pmsg33, 4, /*write=*/false, /*modeled=*/true);
    trace.record(kRegisterBar, kRccConfigMemsize, 4, /*write=*/false, /*modeled=*/true);
  }

  EXPECT_EQ(trace.spin_warnings(), 0u);
}

TEST(BarAccessTrace, TreatsAWriteAsProgressAndStartsTheRunOver) {
  const rocjitsu::RegisterSymbols symbols = pre_discovery_symbols();
  rocjitsu::BarAccessTrace trace(symbols, {.spin_threshold = 8});

  for (int round = 0; round < 8; ++round) {
    for (int i = 0; i < 7; ++i) {
      trace.record(kRegisterBar, kMp0SmnC2pmsg33, 4, /*write=*/false, /*modeled=*/true);
    }
    trace.record(kRegisterBar, kDriverScratch0, 4, /*write=*/true, /*modeled=*/true);
  }

  EXPECT_EQ(trace.spin_warnings(), 0u);
}

TEST(BarAccessTrace, WarnsAgainForANewSpinEpisodeAfterProgress) {
  const rocjitsu::RegisterSymbols symbols = pre_discovery_symbols();
  rocjitsu::BarAccessTrace trace(symbols, {.spin_threshold = 4});

  for (int i = 0; i < 16; ++i) {
    trace.record(kRegisterBar, kMp0SmnC2pmsg33, 4, /*write=*/false, /*modeled=*/true);
  }
  trace.record(kRegisterBar, kDriverScratch0, 4, /*write=*/true, /*modeled=*/true);
  for (int i = 0; i < 16; ++i) {
    trace.record(kRegisterBar, kMp0SmnC2pmsg33, 4, /*write=*/false, /*modeled=*/true);
  }

  EXPECT_EQ(trace.spin_warnings(), 2u);
}

} // namespace
