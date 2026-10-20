/*************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * Host-only tests for src/misc/ionicdvsymbols.cc.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#include <gtest/gtest.h>

#include <cstring>
#include <string>
#include <vector>

#include "fakes/dl_fakes.h"
#include "misc/DlSymbolTable.h"

#include "fakes/dl_seam.h"
#include IONICDVSYMBOLS_CC_PATH
#include "fakes/dl_seam_undef.h"

namespace {

#define IONIC_SYM(field) \
  DlSym { "ionic_dv_" #field, "IONIC_1.0", offsetof(ncclIonicdvSymbols, ionicdv_internal_##field) }
const DlSym kUdmaMask = IONIC_SYM(pd_set_udma_mask);
const std::vector<DlSym> kOptional = {IONIC_SYM(qp_set_gda), IONIC_SYM(qp_set_puec_plane_route)};
#undef IONIC_SYM

std::vector<DlSym> AllSyms() {
  std::vector<DlSym> all = kOptional;
  all.push_back(kUdmaMask);
  return all;
}

class IonicdvsymbolsBuildMicrotest : public ::testing::Test {
 protected:
  void SetUp() override { std::memset(&syms_, 0xff, sizeof(syms_)); }

  ScriptedDl dl_;
  ncclIonicdvSymbols syms_;
};

TEST_F(IonicdvsymbolsBuildMicrotest, AllSymbolsExported_FillsEverySlot) {
  AddLibraryExporting(dl_, "libionic.so", AllSyms());

  ASSERT_EQ(buildIonicdvSymbols(&syms_), ncclSuccess);
  for (const DlSym& sym : AllSyms()) EXPECT_EQ(Slot(syms_, sym), AddrOf(sym)) << sym.name;
  EXPECT_TRUE(dl_.closes.empty());
}

TEST_F(IonicdvsymbolsBuildMicrotest, OptionalSymbolsMissing_LeavesThemNullAndSucceeds) {
  AddLibraryExporting(dl_, "libionic.so", {kUdmaMask});

  ASSERT_EQ(buildIonicdvSymbols(&syms_), ncclSuccess);
  EXPECT_EQ(Slot(syms_, kUdmaMask), AddrOf(kUdmaMask));
  ExpectAllSlotsNull(syms_, kOptional);
}

// libionic is also an ibverbs provider, so it must never be unmapped.
TEST_F(IonicdvsymbolsBuildMicrotest, LibraryPresent_OpensItWithNodelete) {
  AddLibraryExporting(dl_, "libionic.so", {kUdmaMask});

  ASSERT_EQ(buildIonicdvSymbols(&syms_), ncclSuccess);
  ASSERT_EQ(dl_.opens.size(), 1u);
  EXPECT_TRUE(dl_.opens[0].mode & RTLD_NODELETE);
}

TEST_F(IonicdvsymbolsBuildMicrotest, OnlySo1Present_LoadsFromSo1WithNodelete) {
  AddLibraryExporting(dl_, "libionic.so.1", {kUdmaMask});

  EXPECT_EQ(buildIonicdvSymbols(&syms_), ncclSuccess);
  ASSERT_EQ(OpenedFiles(dl_), (std::vector<std::string>{"libionic.so", "libionic.so.1"}));
  EXPECT_TRUE(dl_.opens[1].mode & RTLD_NODELETE);
}

TEST_F(IonicdvsymbolsBuildMicrotest, NoLibraryOpens_FailsWithNullSlots) {
  EXPECT_EQ(buildIonicdvSymbols(&syms_), ncclSystemError);
  EXPECT_EQ(OpenedFiles(dl_), (std::vector<std::string>{"libionic.so", "libionic.so.1"}));
  ExpectAllSlotsNull(syms_, AllSyms());
  EXPECT_TRUE(dl_.closes.empty());
}

TEST_F(IonicdvsymbolsBuildMicrotest, UdmaMaskMissing_FailsClosesLibraryAndNullsEverySlot) {
  void* handle = AddLibraryExporting(dl_, "libionic.so", kOptional);

  EXPECT_EQ(buildIonicdvSymbols(&syms_), ncclSystemError);
  ExpectAllSlotsNull(syms_, AllSyms());
  EXPECT_EQ(dl_.closes, (std::vector<void*>{handle}));
}

TEST_F(IonicdvsymbolsBuildMicrotest, SoOpensButLacksUdmaMask_FailsWithoutTryingSo1) {
  AddLibraryExporting(dl_, "libionic.so", kOptional);
  AddLibraryExporting(dl_, "libionic.so.1", {kUdmaMask});

  EXPECT_EQ(buildIonicdvSymbols(&syms_), ncclSystemError);
  EXPECT_EQ(OpenedFiles(dl_), (std::vector<std::string>{"libionic.so"}));
}

TEST_F(IonicdvsymbolsBuildMicrotest, UdmaMaskAtWrongVersion_Fails) {
  DlSym wrong = kUdmaMask;
  wrong.version = "IONIC_0.9";
  AddLibraryExporting(dl_, "libionic.so", {wrong});

  EXPECT_EQ(buildIonicdvSymbols(&syms_), ncclSystemError);
}

}  // namespace
