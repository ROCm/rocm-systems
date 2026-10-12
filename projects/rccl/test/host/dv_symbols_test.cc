/*************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/
//
// Host-only microtests for buildIonicdvSymbols (src/misc/ionicdvsymbols.cc),
// buildMlx5dvSymbols (src/misc/mlx5dvsymbols.cc) and gin_dv_dlopen
// (src/include/gin/gin_dv_dlopen.h, used by the GIN GDA factory).
//
// Contract: libionic and libmlx5 are also libibverbs providers, so libibverbs may hold
// pointers into them. When a required dv symbol is missing the builder returns
// ncclSystemError, but the provider must stay mapped, or the next ibv_get_device_list
// calls through a dangling driver-ops pointer. ionic_dv_qp_set_gda is optional: a
// libionic without it (older releases) still builds.
//
// Each test loads a fake provider (fakes/dv_provider_fake.cc, SONAME libionic.so or
// libmlx5.so) by path, so the builder's dlopen by name resolves to it. Once the test
// drops its own reference, only the builder's flags can keep it mapped.
//
// The builders open the fakes with RTLD_NODELETE, so a fake can never be unloaded
// from the process that ran the test. Every test therefore runs in its own process
// (RUN_ISOLATED_TEST), so no later test can bind to a fake by SONAME.

#include <dlfcn.h>
#include <sys/stat.h>

#include <cstring>

#include <gtest/gtest.h>

#include "../common/ProcessIsolatedTestRunner.hpp"  // RUN_ISOLATED_TEST
#include "fakes/dv_provider_fake.h"
#include "gin/gin_dv_dlopen.h"

#include IONICDV_SYMBOLS_CC_PATH
#undef LOAD_SYM
#undef LOAD_SYM_VERSION
#include MLX5DV_SYMBOLS_CC_PATH
#undef LOAD_SYM
#undef LOAD_SYM_VERSION

namespace {

using ProviderOps = int (*)();

bool FakeProviderExists(const char* path) {
  struct stat st;
  return stat(path, &st) == 0;
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
  if (!FakeProviderExists(DV_PROVIDER_FAKE_IONIC_PATH)) {
    GTEST_SKIP() << "fake provider not found at " << DV_PROVIDER_FAKE_IONIC_PATH;
  }
  RUN_ISOLATED_TEST("DvSymbols_IonicProviderStaysMappedWhenDvSymbolMissing", []() {
    struct ncclIonicdvSymbols symbols;
    memset(&symbols, 0xff, sizeof(symbols));
    ExpectProviderStaysMapped(DV_PROVIDER_FAKE_IONIC_PATH, ncclSystemError,
                              [&] { return buildIonicdvSymbols(&symbols); });
    EXPECT_EQ(symbols.ionicdv_internal_qp_set_gda, nullptr);
    EXPECT_EQ(symbols.ionicdv_internal_pd_set_udma_mask, nullptr);
    EXPECT_EQ(symbols.ionicdv_internal_qp_set_puec_plane_route, nullptr);
  });
}

TEST(DvSymbolsTestIsolated, IonicWithoutQpSetGdaStillBuilds) {
  if (!FakeProviderExists(DV_PROVIDER_FAKE_IONIC_NO_GDA_PATH)) {
    GTEST_SKIP() << "fake provider not found at " << DV_PROVIDER_FAKE_IONIC_NO_GDA_PATH;
  }
  RUN_ISOLATED_TEST("DvSymbols_IonicWithoutQpSetGdaStillBuilds", []() {
    struct ncclIonicdvSymbols symbols;
    memset(&symbols, 0xff, sizeof(symbols));
    ExpectProviderStaysMapped(DV_PROVIDER_FAKE_IONIC_NO_GDA_PATH, ncclSuccess,
                              [&] { return buildIonicdvSymbols(&symbols); });
    EXPECT_NE(symbols.ionicdv_internal_pd_set_udma_mask, nullptr);
    EXPECT_EQ(symbols.ionicdv_internal_qp_set_gda, nullptr);
    EXPECT_EQ(symbols.ionicdv_internal_qp_set_puec_plane_route, nullptr);
  });
}

TEST(DvSymbolsTestIsolated, Mlx5ProviderStaysMappedWhenDvSymbolMissing) {
  if (!FakeProviderExists(DV_PROVIDER_FAKE_MLX5_PATH)) {
    GTEST_SKIP() << "fake provider not found at " << DV_PROVIDER_FAKE_MLX5_PATH;
  }
  RUN_ISOLATED_TEST("DvSymbols_Mlx5ProviderStaysMappedWhenDvSymbolMissing", []() {
    struct ncclMlx5dvSymbols symbols;
    memset(&symbols, 0xff, sizeof(symbols));
    ExpectProviderStaysMapped(DV_PROVIDER_FAKE_MLX5_PATH, ncclSystemError,
                              [&] { return buildMlx5dvSymbols(&symbols); });
    EXPECT_EQ(symbols.mlx5dv_internal_is_supported, nullptr);
    EXPECT_EQ(symbols.mlx5dv_internal_create_qp, nullptr);
  });
}

TEST(DvSymbolsTestIsolated, GinDvLibraryStaysMappedAfterDlclose) {
  if (!FakeProviderExists(DV_PROVIDER_FAKE_IONIC_PATH)) {
    GTEST_SKIP() << "fake provider not found at " << DV_PROVIDER_FAKE_IONIC_PATH;
  }
  RUN_ISOLATED_TEST("DvSymbols_GinDvLibraryStaysMappedAfterDlclose", []() {
    // The GIN factory's DLSYM_HELPER dlcloses the handle when a symbol is missing.
    ExpectProviderStaysMapped(DV_PROVIDER_FAKE_IONIC_PATH, ncclSuccess, [] {
      void* handle = gin_dv_dlopen("libionic.so");
      if (handle == nullptr) return ncclSystemError;
      dlclose(handle);
      return ncclSuccess;
    });
  });
}

}  // namespace
