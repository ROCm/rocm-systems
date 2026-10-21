/*************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * Host-only tests for src/misc/mlx5dvsymbols.cc (dynamic-loading mode) and
 * src/misc/ionicdvsymbols.cc: one loader contract, run once per provider.
 * dv_symbols_test.cc (rccl-UnitTestsMicro) checks the same builders against real provider objects.
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
#undef LOAD_SYM
#undef LOAD_SYM_VERSION
#include IONICDVSYMBOLS_CC_PATH
#undef LOAD_SYM
#undef LOAD_SYM_VERSION
#include "fakes/dl_seam_undef.h"

namespace {

struct Mlx5Provider {
  using Table = ncclMlx5dvSymbols;
  static ncclResult_t Build(Table* table) { return buildMlx5dvSymbols(table); }
  static constexpr const char* kLib = "libmlx5.so";
  static constexpr const char* kWrongVersion = "MLX5_1.0";
#define MLX5_SYM(field, version) \
  DlSym { "mlx5dv_" #field, version, offsetof(ncclMlx5dvSymbols, mlx5dv_internal_##field) }
  static DlSym Required() { return MLX5_SYM(is_supported, "MLX5_1.8"); }
  static std::vector<DlSym> Optional() {
    return {MLX5_SYM(get_data_direct_sysfs_path, "MLX5_1.25"), MLX5_SYM(reg_dmabuf_mr, "MLX5_1.25"),
            MLX5_SYM(query_device, "MLX5_1.0"), MLX5_SYM(create_qp, "MLX5_1.3")};
  }
#undef MLX5_SYM
};

struct IonicProvider {
  using Table = ncclIonicdvSymbols;
  static ncclResult_t Build(Table* table) { return buildIonicdvSymbols(table); }
  static constexpr const char* kLib = "libionic.so";
  static constexpr const char* kWrongVersion = "IONIC_0.9";
#define IONIC_SYM(field) \
  DlSym { "ionic_dv_" #field, "IONIC_1.0", offsetof(ncclIonicdvSymbols, ionicdv_internal_##field) }
  static DlSym Required() { return IONIC_SYM(pd_set_udma_mask); }
  static std::vector<DlSym> Optional() { return {IONIC_SYM(qp_set_gda), IONIC_SYM(qp_set_puec_plane_route)}; }
#undef IONIC_SYM
};

template <typename Provider>
class DvsymbolsBuildMicrotest : public ::testing::Test {
 protected:
  void SetUp() override { std::memset(&syms_, 0xff, sizeof(syms_)); }

  ncclResult_t Build() { return Provider::Build(&syms_); }
  static std::string So() { return Provider::kLib; }
  static std::string So1() { return std::string(Provider::kLib) + ".1"; }
  static std::vector<DlSym> All() {
    std::vector<DlSym> all = Provider::Optional();
    all.push_back(Provider::Required());
    return all;
  }

  ScriptedDl dl_;
  typename Provider::Table syms_;
};

TYPED_TEST_SUITE_P(DvsymbolsBuildMicrotest);

TYPED_TEST_P(DvsymbolsBuildMicrotest, AllSymbolsAtTheirVersions_FillsEverySlot) {
  AddLibraryExporting(this->dl_, this->So(), this->All());

  ASSERT_EQ(this->Build(), ncclSuccess);
  for (const DlSym& sym : this->All()) EXPECT_EQ(Slot(this->syms_, sym), AddrOf(sym)) << sym.name;
  EXPECT_TRUE(this->dl_.closes.empty());
}

TYPED_TEST_P(DvsymbolsBuildMicrotest, OptionalSymbolsMissing_LeavesThemNullAndSucceeds) {
  AddLibraryExporting(this->dl_, this->So(), {TypeParam::Required()});

  ASSERT_EQ(this->Build(), ncclSuccess);
  EXPECT_EQ(Slot(this->syms_, TypeParam::Required()), AddrOf(TypeParam::Required()));
  ExpectAllSlotsNull(this->syms_, TypeParam::Optional());
}

// The dv libraries are also ibverbs providers, so they must never be unmapped.
TYPED_TEST_P(DvsymbolsBuildMicrotest, LibraryPresent_OpensItWithNodelete) {
  AddLibraryExporting(this->dl_, this->So(), {TypeParam::Required()});

  ASSERT_EQ(this->Build(), ncclSuccess);
  ASSERT_EQ(this->dl_.opens.size(), 1u);
  EXPECT_TRUE(this->dl_.opens[0].mode & RTLD_NODELETE);
}

TYPED_TEST_P(DvsymbolsBuildMicrotest, OnlySo1Present_LoadsFromSo1WithNodelete) {
  AddLibraryExporting(this->dl_, this->So1(), {TypeParam::Required()});

  EXPECT_EQ(this->Build(), ncclSuccess);
  ASSERT_EQ(OpenedFiles(this->dl_), (std::vector<std::string>{this->So(), this->So1()}));
  EXPECT_TRUE(this->dl_.opens[1].mode & RTLD_NODELETE);
}

TYPED_TEST_P(DvsymbolsBuildMicrotest, NoLibraryOpens_FailsWithNullSlots) {
  EXPECT_EQ(this->Build(), ncclSystemError);
  EXPECT_EQ(OpenedFiles(this->dl_), (std::vector<std::string>{this->So(), this->So1()}));
  ExpectAllSlotsNull(this->syms_, this->All());
  EXPECT_TRUE(this->dl_.closes.empty());
}

TYPED_TEST_P(DvsymbolsBuildMicrotest, RequiredMissing_FailsClosesLibraryAndNullsEverySlot) {
  void* handle = AddLibraryExporting(this->dl_, this->So(), TypeParam::Optional());

  EXPECT_EQ(this->Build(), ncclSystemError);
  ExpectAllSlotsNull(this->syms_, this->All());
  EXPECT_EQ(this->dl_.closes, (std::vector<void*>{handle}));
}

TYPED_TEST_P(DvsymbolsBuildMicrotest, SoOpensButLacksRequired_FailsWithoutTryingSo1) {
  AddLibraryExporting(this->dl_, this->So(), TypeParam::Optional());
  AddLibraryExporting(this->dl_, this->So1(), {TypeParam::Required()});

  EXPECT_EQ(this->Build(), ncclSystemError);
  EXPECT_EQ(OpenedFiles(this->dl_), (std::vector<std::string>{this->So()}));
}

TYPED_TEST_P(DvsymbolsBuildMicrotest, RequiredAtWrongVersion_Fails) {
  DlSym wrong = TypeParam::Required();
  wrong.version = TypeParam::kWrongVersion;
  AddLibraryExporting(this->dl_, this->So(), {wrong});

  EXPECT_EQ(this->Build(), ncclSystemError);
}

REGISTER_TYPED_TEST_SUITE_P(DvsymbolsBuildMicrotest, AllSymbolsAtTheirVersions_FillsEverySlot,
                            OptionalSymbolsMissing_LeavesThemNullAndSucceeds, LibraryPresent_OpensItWithNodelete,
                            OnlySo1Present_LoadsFromSo1WithNodelete, NoLibraryOpens_FailsWithNullSlots,
                            RequiredMissing_FailsClosesLibraryAndNullsEverySlot,
                            SoOpensButLacksRequired_FailsWithoutTryingSo1, RequiredAtWrongVersion_Fails);

INSTANTIATE_TYPED_TEST_SUITE_P(Mlx5dvsymbols, DvsymbolsBuildMicrotest, Mlx5Provider);
INSTANTIATE_TYPED_TEST_SUITE_P(Ionicdvsymbols, DvsymbolsBuildMicrotest, IonicProvider);

}  // namespace
