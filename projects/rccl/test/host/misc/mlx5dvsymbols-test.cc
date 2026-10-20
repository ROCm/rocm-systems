/*************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * Host-only tests for src/misc/mlx5dvsymbols.cc (dynamic-loading mode).
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
#include MLX5DVSYMBOLS_CC_PATH
#include "fakes/dl_seam_undef.h"

namespace {

#define MLX5_SYM(field, version) \
  DlSym { "mlx5dv_" #field, version, offsetof(ncclMlx5dvSymbols, mlx5dv_internal_##field) }
const DlSym kIsSupported = MLX5_SYM(is_supported, "MLX5_1.8");
const std::vector<DlSym> kOptional = {
    MLX5_SYM(get_data_direct_sysfs_path, "MLX5_1.25"),
    MLX5_SYM(reg_dmabuf_mr, "MLX5_1.25"),
    MLX5_SYM(query_device, "MLX5_1.0"),
    MLX5_SYM(create_qp, "MLX5_1.3"),
};
#undef MLX5_SYM

std::vector<DlSym> AllSyms() {
  std::vector<DlSym> all = kOptional;
  all.push_back(kIsSupported);
  return all;
}

class Mlx5dvsymbolsBuildMicrotest : public ::testing::Test {
 protected:
  void SetUp() override { std::memset(&syms_, 0xff, sizeof(syms_)); }

  ScriptedDl dl_;
  ncclMlx5dvSymbols syms_;
};

TEST_F(Mlx5dvsymbolsBuildMicrotest, AllSymbolsAtTheirVersions_FillsEverySlot) {
  AddLibraryExporting(dl_, "libmlx5.so", AllSyms());

  ASSERT_EQ(buildMlx5dvSymbols(&syms_), ncclSuccess);
  for (const DlSym& sym : AllSyms()) EXPECT_EQ(Slot(syms_, sym), AddrOf(sym)) << sym.name;
  EXPECT_TRUE(dl_.closes.empty());
}

TEST_F(Mlx5dvsymbolsBuildMicrotest, OptionalSymbolsMissing_LeavesThemNullAndSucceeds) {
  AddLibraryExporting(dl_, "libmlx5.so", {kIsSupported});

  ASSERT_EQ(buildMlx5dvSymbols(&syms_), ncclSuccess);
  EXPECT_EQ(Slot(syms_, kIsSupported), AddrOf(kIsSupported));
  ExpectAllSlotsNull(syms_, kOptional);
}

// libmlx5 is also an ibverbs provider, so it must never be unmapped.
TEST_F(Mlx5dvsymbolsBuildMicrotest, LibraryPresent_OpensItWithNodelete) {
  AddLibraryExporting(dl_, "libmlx5.so", {kIsSupported});

  ASSERT_EQ(buildMlx5dvSymbols(&syms_), ncclSuccess);
  ASSERT_EQ(dl_.opens.size(), 1u);
  EXPECT_TRUE(dl_.opens[0].mode & RTLD_NODELETE);
}

TEST_F(Mlx5dvsymbolsBuildMicrotest, OnlySo1Present_LoadsFromSo1WithNodelete) {
  AddLibraryExporting(dl_, "libmlx5.so.1", {kIsSupported});

  EXPECT_EQ(buildMlx5dvSymbols(&syms_), ncclSuccess);
  EXPECT_EQ(OpenedFiles(dl_), (std::vector<std::string>{"libmlx5.so", "libmlx5.so.1"}));
  EXPECT_TRUE(dl_.opens[1].mode & RTLD_NODELETE);
}

TEST_F(Mlx5dvsymbolsBuildMicrotest, NoLibraryOpens_FailsWithNullSlots) {
  EXPECT_EQ(buildMlx5dvSymbols(&syms_), ncclSystemError);
  EXPECT_EQ(OpenedFiles(dl_), (std::vector<std::string>{"libmlx5.so", "libmlx5.so.1"}));
  ExpectAllSlotsNull(syms_, AllSyms());
  EXPECT_TRUE(dl_.closes.empty());
}

TEST_F(Mlx5dvsymbolsBuildMicrotest, IsSupportedMissing_FailsClosesLibraryAndNullsEverySlot) {
  void* handle = AddLibraryExporting(dl_, "libmlx5.so", kOptional);

  EXPECT_EQ(buildMlx5dvSymbols(&syms_), ncclSystemError);
  ExpectAllSlotsNull(syms_, AllSyms());
  EXPECT_EQ(dl_.closes, (std::vector<void*>{handle}));
}

TEST_F(Mlx5dvsymbolsBuildMicrotest, SoOpensButLacksIsSupported_FailsWithoutTryingSo1) {
  AddLibraryExporting(dl_, "libmlx5.so", kOptional);
  AddLibraryExporting(dl_, "libmlx5.so.1", {kIsSupported});

  EXPECT_EQ(buildMlx5dvSymbols(&syms_), ncclSystemError);
  EXPECT_EQ(OpenedFiles(dl_), (std::vector<std::string>{"libmlx5.so"}));
}

TEST_F(Mlx5dvsymbolsBuildMicrotest, IsSupportedAtWrongVersion_Fails) {
  DlSym wrong = kIsSupported;
  wrong.version = "MLX5_1.0";
  AddLibraryExporting(dl_, "libmlx5.so", {wrong});

  EXPECT_EQ(buildMlx5dvSymbols(&syms_), ncclSystemError);
}

}  // namespace
