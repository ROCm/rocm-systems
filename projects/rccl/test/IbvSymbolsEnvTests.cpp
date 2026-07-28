/*************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#ifndef _GNU_SOURCE
#define _GNU_SOURCE  // dlinfo()
#endif

#include <gtest/gtest.h>

#include <dlfcn.h>
#include <link.h>
#include <cstdlib>
#include <string>

#include "ibvsymbols.h"

// buildIbvSymbols() is compiled into this test target (it is hidden-visibility
// in librccl and therefore not linkable from here). It resolves the libibverbs
// path via ncclGetEnv(), which is likewise hidden in librccl. Provide the same
// behavior as RCCL's built-in default env plugin
// (src/plugin/env/env_v1.cc: ncclEnvGetEnv -> std::getenv) so the unit under
// test reads the process environment directly, without linking the env-plugin
// machinery. Keep it hidden so it only satisfies the locally-compiled
// ibvsymbols.cc reference and never interposes librccl's own ncclGetEnv in
// Debug builds (where librccl is compiled with -fvisibility=default).
__attribute__((visibility("hidden")))
const char* ncclGetEnv(const char* name) { return std::getenv(name); }

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

    // Resolves the absolute path of the loadable libibverbs on this host, or an
    // empty string if neither soname can be opened. Using the resolved absolute
    // path (rather than a bare soname) lets EnvOverrideValidSoname prove the env
    // slot was actually used: buildIbvSymbols' hardcoded fallbacks only try the
    // bare "libibverbs.so[.1]" names, so a load from an absolute path can only
    // have come through NCCL_IBVERBS_LIB.
    std::string LibibverbsPath()
    {
      for (const char* soname : {"libibverbs.so.1", "libibverbs.so"})
      {
        void* h = dlopen(soname, RTLD_NOW | RTLD_LOCAL);
        if (!h) continue;
        std::string path;
        struct link_map* lm = nullptr;
        if (dlinfo(h, RTLD_DI_LINKMAP, &lm) == 0 && lm && lm->l_name && lm->l_name[0])
          path = lm->l_name;
        dlclose(h);
        if (!path.empty()) return path;
      }
      return {};
    }

    // True when libibverbs is installed and loadable on this host. When false,
    // every case below is a hardware/environment SKIP rather than a failure.
    bool LibibverbsAvailable() { return !LibibverbsPath().empty(); }

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

  // Env set to the resolved absolute path: the override slot (index 0) is taken
  // first and dlopen succeeds, so the hardcoded fallbacks are never reached. An
  // absolute path can only load via the env slot (the fallbacks try bare
  // sonames only), so success here proves the override was honored.
  TEST(IbvSymbolsEnv, EnvOverrideValidSoname)
  {
    std::string libPath = LibibverbsPath();
    if (libPath.empty())
      GTEST_SKIP() << "libibverbs not installed on this host";

    ScopedEnv env;
    env.set(libPath.c_str());

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
