/*************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#include <gtest/gtest.h>

#include <dlfcn.h>
#include <cstdlib>

#include "ibvsymbols.h"

namespace RcclUnitTesting
{
  // Whitebox coverage for the NCCL_IBVERBS_LIB dynamic-loader override in
  // src/misc/ibvsymbols.cc (dlopen path). buildIbvSymbols() is not guarded by
  // std::call_once (that lives in wrap_ibv_symbols), and the internal env
  // plugin reads std::getenv() live, so each call re-reads the environment.
  // These tests exercise:
  //   - default load (env unset)               -> libibverbs.so[.1]
  //   - explicit override to a valid soname     -> env slot (index 0) is used
  //   - override to a nonexistent path          -> fallback loop to the defaults
  // The all-paths-fail branch (ncclSystemError) is not reachable here: the two
  // hardcoded fallbacks cannot be removed via env, so it requires a host with
  // rdma-core absent and is left as a documented ceiling.

  namespace
  {
    constexpr const char* kEnvVar = "NCCL_IBVERBS_LIB";

    // Restores NCCL_IBVERBS_LIB to its pre-test value on scope exit so cases
    // do not contaminate one another (tests share a process).
    class ScopedEnv
    {
    public:
      ScopedEnv() : had_(false)
      {
        const char* v = std::getenv(kEnvVar);
        if (v) { had_ = true; saved_ = v; }
      }
      void set(const char* value) { setenv(kEnvVar, value, 1); }
      void unset() { unsetenv(kEnvVar); }
      ~ScopedEnv()
      {
        if (had_) setenv(kEnvVar, saved_.c_str(), 1);
        else      unsetenv(kEnvVar);
      }
    private:
      bool        had_;
      std::string saved_;
    };

    // True when libibverbs is installed and loadable on this host. When false,
    // every case below is a hardware/environment SKIP rather than a failure.
    bool LibibverbsAvailable()
    {
      void* h = dlopen("libibverbs.so.1", RTLD_NOW | RTLD_LOCAL);
      if (!h) h = dlopen("libibverbs.so", RTLD_NOW | RTLD_LOCAL);
      if (!h) return false;
      dlclose(h);
      return true;
    }

    // A successful buildIbvSymbols() populates the whole function-pointer
    // table; spot-check representative entries that every provider exports.
    void ExpectSymbolsLoaded(const ncclIbvSymbols& s)
    {
      EXPECT_NE(s.ibv_internal_get_device_list, nullptr);
      EXPECT_NE(s.ibv_internal_open_device,     nullptr);
      EXPECT_NE(s.ibv_internal_create_qp,       nullptr);
      EXPECT_NE(s.ibv_internal_reg_mr,          nullptr);
    }
  }  // namespace

  // Env unset: loader falls through to the hardcoded libibverbs.so[.1] names.
  TEST(IbvSymbolsEnv, DefaultLoadsLibrary)
  {
    if (!LibibverbsAvailable())
      GTEST_SKIP() << "libibverbs not installed on this host";

    ScopedEnv env;
    env.unset();

    ncclIbvSymbols symbols = {};
    ASSERT_EQ(buildIbvSymbols(&symbols), ncclSuccess);
    ExpectSymbolsLoaded(symbols);
  }

  // Env set to a valid soname: the override slot (index 0) is taken first and
  // dlopen succeeds, so the hardcoded fallbacks are never reached.
  TEST(IbvSymbolsEnv, EnvOverrideValidSoname)
  {
    if (!LibibverbsAvailable())
      GTEST_SKIP() << "libibverbs not installed on this host";

    ScopedEnv env;
    env.set("libibverbs.so.1");

    ncclIbvSymbols symbols = {};
    ASSERT_EQ(buildIbvSymbols(&symbols), ncclSuccess);
    ExpectSymbolsLoaded(symbols);
  }

  // Env set to a nonexistent path: index 0 fails to dlopen, the loop continues
  // and recovers via the hardcoded libibverbs.so[.1] fallback. This is the
  // branch the override adds over the old hardcoded-only loader.
  TEST(IbvSymbolsEnv, BogusPathFallsBackToDefault)
  {
    if (!LibibverbsAvailable())
      GTEST_SKIP() << "libibverbs not installed on this host";

    ScopedEnv env;
    env.set("/nonexistent/path/libibverbs.so");

    ncclIbvSymbols symbols = {};
    ASSERT_EQ(buildIbvSymbols(&symbols), ncclSuccess);
    ExpectSymbolsLoaded(symbols);
  }
}  // namespace RcclUnitTesting
