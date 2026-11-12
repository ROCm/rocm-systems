/*************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/
//
// Host-only microtests for buildIonicdvSymbols (src/misc/ionicdvsymbols.cc) and
// buildMlx5dvSymbols (src/misc/mlx5dvsymbols.cc).
//
// Contract: libionic and libmlx5 are also libibverbs providers, so libibverbs may hold
// pointers into them. When a required dv symbol is missing the builder returns
// ncclSystemError, but the provider must stay mapped, or the next ibv_get_device_list
// calls through a dangling driver-ops pointer. ionic_dv_qp_set_gda is optional: a
// libionic without it (older releases) still builds.
//
// Each test loads a fake provider (fakes/dv_provider_fake.cc, SONAME libionic.so or
// libmlx5.so) by path, so the builder's dlopen by name resolves to it. In the
// missing-symbol tests, once the test drops its own reference, only the builder's
// flags can keep it mapped.
//
// The builders open the fakes with RTLD_NODELETE, so a fake can never be unloaded
// from the process that ran the test. Every test therefore runs in its own process
// (RUN_ISOLATED_TEST), so no later test can bind to a fake by SONAME.

#include <dlfcn.h>

#include <cstring>
#include <filesystem>
#include <string>

#include <gtest/gtest.h>

#include "../common/ProcessIsolatedTestRunner.hpp"  // RUN_ISOLATED_TEST
#include "fakes/dv_provider_fake.h"

#include IONICDV_SYMBOLS_CC_PATH
#undef LOAD_SYM
#undef LOAD_SYM_VERSION
#include MLX5DV_SYMBOLS_CC_PATH
#undef LOAD_SYM
#undef LOAD_SYM_VERSION

namespace {

using ProviderOps = int (*)();

// The fakes sit beside the test binary, in the build tree and in an install.
std::string FakeProviderPath(const char* name) {
  std::error_code ec;
  auto path = std::filesystem::read_symlink("/proc/self/exe", ec).parent_path() / name;
  return !ec && std::filesystem::exists(path) ? path.string() : "";
}

// Loads the fake provider as the only reference, runs build, then checks the
// provider is still mapped and callable after the test drops its reference.
template <typename Build>
void ExpectProviderStaysMapped(const char* path, ncclResult_t expected, Build build) {
  void* provider = dlopen(path, RTLD_NOW);
  ASSERT_NE(provider, nullptr) << dlerror();
  auto ops = reinterpret_cast<ProviderOps>(dlsym(provider, "rcclDvProviderFakeOps"));
  ASSERT_NE(ops, nullptr) << dlerror();

  EXPECT_EQ(build(), expected);
  dlclose(provider);

  void* stillMapped = dlopen(path, RTLD_NOW | RTLD_NOLOAD);
  ASSERT_NE(stillMapped, nullptr) << path << " was unmapped after the build";
  EXPECT_EQ(ops(), kDvProviderFakeOpsResult);
  dlclose(stillMapped);
}

TEST(DvSymbolsTestIsolated, IonicProviderStaysMappedWhenDvSymbolMissing) {
  const std::string path = FakeProviderPath(DV_PROVIDER_FAKE_IONIC_NAME);
  if (path.empty()) GTEST_SKIP() << DV_PROVIDER_FAKE_IONIC_NAME << " not found beside the test binary";
  RUN_ISOLATED_TEST("DvSymbols_IonicProviderStaysMappedWhenDvSymbolMissing", [path]() {
    struct ncclIonicdvSymbols symbols;
    memset(&symbols, 0xff, sizeof(symbols));
    ExpectProviderStaysMapped(path.c_str(), ncclSystemError, [&] { return buildIonicdvSymbols(&symbols); });
    EXPECT_EQ(symbols.ionicdv_internal_qp_set_gda, nullptr);
    EXPECT_EQ(symbols.ionicdv_internal_pd_set_udma_mask, nullptr);
    EXPECT_EQ(symbols.ionicdv_internal_qp_set_puec_plane_route, nullptr);
  });
}

TEST(DvSymbolsTestIsolated, IonicWithoutQpSetGdaStillBuilds) {
  const std::string path = FakeProviderPath(DV_PROVIDER_FAKE_IONIC_NO_GDA_NAME);
  if (path.empty()) GTEST_SKIP() << DV_PROVIDER_FAKE_IONIC_NO_GDA_NAME << " not found beside the test binary";
  RUN_ISOLATED_TEST("DvSymbols_IonicWithoutQpSetGdaStillBuilds", [path]() {
    struct ncclIonicdvSymbols symbols;
    memset(&symbols, 0xff, sizeof(symbols));
    ExpectProviderStaysMapped(path.c_str(), ncclSuccess, [&] { return buildIonicdvSymbols(&symbols); });
    EXPECT_NE(symbols.ionicdv_internal_pd_set_udma_mask, nullptr);
    EXPECT_EQ(symbols.ionicdv_internal_qp_set_gda, nullptr);
    EXPECT_EQ(symbols.ionicdv_internal_qp_set_puec_plane_route, nullptr);
  });
}

TEST(DvSymbolsTestIsolated, Mlx5ProviderStaysMappedWhenDvSymbolMissing) {
  const std::string path = FakeProviderPath(DV_PROVIDER_FAKE_MLX5_NAME);
  if (path.empty()) GTEST_SKIP() << DV_PROVIDER_FAKE_MLX5_NAME << " not found beside the test binary";
  RUN_ISOLATED_TEST("DvSymbols_Mlx5ProviderStaysMappedWhenDvSymbolMissing", [path]() {
    struct ncclMlx5dvSymbols symbols;
    memset(&symbols, 0xff, sizeof(symbols));
    ExpectProviderStaysMapped(path.c_str(), ncclSystemError, [&] { return buildMlx5dvSymbols(&symbols); });
    EXPECT_EQ(symbols.mlx5dv_internal_is_supported, nullptr);
    EXPECT_EQ(symbols.mlx5dv_internal_get_data_direct_sysfs_path, nullptr);
    EXPECT_EQ(symbols.mlx5dv_internal_reg_dmabuf_mr, nullptr);
    EXPECT_EQ(symbols.mlx5dv_internal_query_device, nullptr);
    EXPECT_EQ(symbols.mlx5dv_internal_create_qp, nullptr);
  });
}

}  // namespace
