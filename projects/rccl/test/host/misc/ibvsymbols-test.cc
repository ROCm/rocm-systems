/*************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * Host-only tests for src/misc/ibvsymbols.cc (dynamic-loading mode).
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#include <gtest/gtest.h>

#include <cstring>
#include <string>
#include <vector>

#include "fakes/dl_fakes.h"
#include "fakes/env_fakes.h"
#include "misc/DlSymbolTable.h"

#include "fakes/dl_seam.h"
#include IBVSYMBOLS_CC_PATH
#include "fakes/dl_seam_undef.h"

namespace {

#define IBV_SYM(field, version) \
  DlSym { "ibv_" #field, version, offsetof(ncclIbvSymbols, ibv_internal_##field) }
#define IBV_SYM_1_1(field) IBV_SYM(field, "IBVERBS_1.1")

const std::vector<DlSym> kRequired = {
    IBV_SYM_1_1(get_device_list), IBV_SYM_1_1(free_device_list), IBV_SYM_1_1(get_device_name),
    IBV_SYM_1_1(open_device),     IBV_SYM_1_1(close_device),     IBV_SYM_1_1(get_async_event),
    IBV_SYM_1_1(ack_async_event), IBV_SYM_1_1(query_device),     IBV_SYM_1_1(query_port),
    IBV_SYM_1_1(query_gid),       IBV_SYM_1_1(query_pkey),       IBV_SYM_1_1(query_qp),
    IBV_SYM_1_1(alloc_pd),        IBV_SYM_1_1(dealloc_pd),       IBV_SYM_1_1(reg_mr),
    IBV_SYM_1_1(dereg_mr),        IBV_SYM_1_1(create_cq),        IBV_SYM_1_1(destroy_cq),
    IBV_SYM_1_1(create_qp),       IBV_SYM_1_1(modify_qp),        IBV_SYM_1_1(destroy_qp),
    IBV_SYM_1_1(fork_init),       IBV_SYM_1_1(event_type_str),
};

// Newer rdma-core APIs, each pinned to the version that introduced it.
const std::vector<DlSym> kOptional = {
    IBV_SYM(reg_mr_iova2, "IBVERBS_1.8"), IBV_SYM(reg_dmabuf_mr, "IBVERBS_1.12"),
    IBV_SYM(query_ece, "IBVERBS_1.10"),   IBV_SYM(set_ece, "IBVERBS_1.10"),
    IBV_SYM(query_port_speed, "IBVERBS_1.16"),
};

#undef IBV_SYM_1_1
#undef IBV_SYM

std::vector<DlSym> AllSyms() {
  std::vector<DlSym> all = kRequired;
  all.insert(all.end(), kOptional.begin(), kOptional.end());
  return all;
}

class IbvsymbolsBuildMicrotest : public ::testing::Test {
 protected:
  void SetUp() override {
    SetMicroEnvAbsent("NCCL_IBVERBS_LIB");
    SetMicroEnvAbsent("NCCL_LIBIBVERBS_SO");
    std::memset(&syms_, 0xff, sizeof(syms_));  // so a slot the builder never writes shows up
  }
  void TearDown() override { ResetEnvFakes(); }

  ScriptedDl dl_;
  ncclIbvSymbols syms_;
};

TEST_F(IbvsymbolsBuildMicrotest, AllSymbolsAtTheirVersions_FillsEverySlot) {
  AddLibraryExporting(dl_, "libibverbs.so", AllSyms());

  ASSERT_EQ(buildIbvSymbols(&syms_), ncclSuccess);
  for (const DlSym& sym : AllSyms()) EXPECT_EQ(Slot(syms_, sym), AddrOf(sym)) << sym.name;
  EXPECT_TRUE(dl_.closes.empty());
}

TEST_F(IbvsymbolsBuildMicrotest, OptionalSymbolsMissing_LeavesThemNullAndSucceeds) {
  AddLibraryExporting(dl_, "libibverbs.so", kRequired);

  ASSERT_EQ(buildIbvSymbols(&syms_), ncclSuccess);
  for (const DlSym& sym : kRequired) EXPECT_EQ(Slot(syms_, sym), AddrOf(sym)) << sym.name;
  ExpectAllSlotsNull(syms_, kOptional);
}

TEST_F(IbvsymbolsBuildMicrotest, NoLibraryOpens_TriesBothSonamesAndFailsWithNullSlots) {
  EXPECT_EQ(buildIbvSymbols(&syms_), ncclSystemError);
  EXPECT_EQ(OpenedFiles(dl_), (std::vector<std::string>{"libibverbs.so", "libibverbs.so.1"}));
  ExpectAllSlotsNull(syms_, AllSyms());
  EXPECT_TRUE(dl_.closes.empty());
}

TEST_F(IbvsymbolsBuildMicrotest, OnlySo1Present_LoadsFromSo1) {
  AddLibraryExporting(dl_, "libibverbs.so.1", kRequired);

  EXPECT_EQ(buildIbvSymbols(&syms_), ncclSuccess);
  EXPECT_EQ(OpenedFiles(dl_), (std::vector<std::string>{"libibverbs.so", "libibverbs.so.1"}));
}

TEST_F(IbvsymbolsBuildMicrotest, IbverbsLibEnvSet_OpensItFirst) {
  SetMicroEnv("NCCL_IBVERBS_LIB", "/opt/custom/libibverbs.so");
  AddLibraryExporting(dl_, "/opt/custom/libibverbs.so", kRequired);
  AddLibraryExporting(dl_, "libibverbs.so", kRequired);

  EXPECT_EQ(buildIbvSymbols(&syms_), ncclSuccess);
  EXPECT_EQ(OpenedFiles(dl_), (std::vector<std::string>{"/opt/custom/libibverbs.so"}));
}

TEST_F(IbvsymbolsBuildMicrotest, EnvLibDoesNotOpen_FallsBackToSystemLib) {
  SetMicroEnv("NCCL_IBVERBS_LIB", "/missing/libibverbs.so");
  AddLibraryExporting(dl_, "libibverbs.so", kRequired);

  EXPECT_EQ(buildIbvSymbols(&syms_), ncclSuccess);
  EXPECT_EQ(OpenedFiles(dl_), (std::vector<std::string>{"/missing/libibverbs.so", "libibverbs.so"}));
}

TEST_F(IbvsymbolsBuildMicrotest, OnlyLegacyEnvSet_OpensIt) {
  SetMicroEnv("NCCL_LIBIBVERBS_SO", "/legacy/libibverbs.so");
  AddLibraryExporting(dl_, "/legacy/libibverbs.so", kRequired);

  EXPECT_EQ(buildIbvSymbols(&syms_), ncclSuccess);
  EXPECT_EQ(OpenedFiles(dl_), (std::vector<std::string>{"/legacy/libibverbs.so"}));
}

TEST_F(IbvsymbolsBuildMicrotest, BothEnvNamesSet_IbverbsLibWins) {
  SetMicroEnv("NCCL_IBVERBS_LIB", "/new/libibverbs.so");
  SetMicroEnv("NCCL_LIBIBVERBS_SO", "/legacy/libibverbs.so");
  AddLibraryExporting(dl_, "/new/libibverbs.so", kRequired);
  AddLibraryExporting(dl_, "/legacy/libibverbs.so", kRequired);

  EXPECT_EQ(buildIbvSymbols(&syms_), ncclSuccess);
  EXPECT_EQ(OpenedFiles(dl_), (std::vector<std::string>{"/new/libibverbs.so"}));
}

TEST_F(IbvsymbolsBuildMicrotest, EnvLibOpensButLacksSymbol_FailsWithoutTryingSystemLib) {
  SetMicroEnv("NCCL_IBVERBS_LIB", "/opt/custom/libibverbs.so");
  AddLibraryExporting(dl_, "/opt/custom/libibverbs.so", {kRequired[0]});
  AddLibraryExporting(dl_, "libibverbs.so", kRequired);

  EXPECT_EQ(buildIbvSymbols(&syms_), ncclSystemError);
  EXPECT_EQ(OpenedFiles(dl_), (std::vector<std::string>{"/opt/custom/libibverbs.so"}));
}

class IbvsymbolsMissingRequiredMicrotest : public IbvsymbolsBuildMicrotest,
                                           public ::testing::WithParamInterface<size_t> {};

TEST_P(IbvsymbolsMissingRequiredMicrotest, RequiredSymbolMissing_FailsClosesLibraryAndNullsEverySlot) {
  std::vector<DlSym> exports = kOptional;
  for (size_t i = 0; i < kRequired.size(); ++i) {
    if (i != GetParam()) exports.push_back(kRequired[i]);
  }
  void* handle = AddLibraryExporting(dl_, "libibverbs.so", exports);

  EXPECT_EQ(buildIbvSymbols(&syms_), ncclSystemError);
  ExpectAllSlotsNull(syms_, AllSyms());
  EXPECT_EQ(dl_.closes, (std::vector<void*>{handle}));
}

TEST_P(IbvsymbolsMissingRequiredMicrotest, RequiredSymbolAtWrongVersion_Fails) {
  std::vector<DlSym> exports = kRequired;
  exports[GetParam()].version = "IBVERBS_1.0";
  AddLibraryExporting(dl_, "libibverbs.so", exports);

  EXPECT_EQ(buildIbvSymbols(&syms_), ncclSystemError);
}

INSTANTIATE_TEST_SUITE_P(Ibvsymbols, IbvsymbolsMissingRequiredMicrotest,
                         ::testing::Range<size_t>(0, kRequired.size()),
                         [](const ::testing::TestParamInfo<size_t>& info) {
                           return std::string(kRequired[info.param].name);
                         });

}  // namespace
