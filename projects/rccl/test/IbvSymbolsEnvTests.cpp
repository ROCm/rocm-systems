/*************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#ifndef _GNU_SOURCE
#define _GNU_SOURCE  // dlinfo(), dl_iterate_phdr()
#endif

#include <gtest/gtest.h>

#include <dlfcn.h>
#include <link.h>
#include <unistd.h>
#include <cstdlib>
#include <fstream>
#include <string>

#include "ibvsymbols.h"

// buildIbvSymbols() is compiled straight into this test target for non-Debug
// builds only (see CMakeLists.txt); Debug links librccl's own exported copy
// instead, so this ncclGetEnv() stub is unused there. Kept hidden so it can
// never interpose librccl's own ncclGetEnv in Debug builds.
__attribute__((visibility("hidden")))
const char* ncclGetEnv(const char* name) { return std::getenv(name); }

namespace RcclUnitTesting
{
  // Whitebox coverage for the NCCL_IBVERBS_LIB override (and its
  // NCCL_LIBIBVERBS_SO alias) in src/misc/ibvsymbols.cc. Covers: default load
  // (env unset), override to a uniquely-located copy, override to a
  // nonexistent path (fallback), the alias by itself, and precedence when
  // both are set. The all-paths-fail branch needs a host with no rdma-core
  // and is left as a documented ceiling.
  //
  // Note: buildIbvSymbols()'s loader handle is a function-local static that
  // is never dlclose()'d, so repeated calls in this process accumulate
  // dlopen refs rather than releasing prior ones -- harmless here.

  namespace
  {
    constexpr const char* kEnvVar      = "NCCL_IBVERBS_LIB";
    constexpr const char* kAliasEnvVar = "NCCL_LIBIBVERBS_SO";

    // Restores `name`'s pre-test value on scope exit so cases do not
    // contaminate one another (tests share a process).
    class ScopedEnv
    {
    public:
      explicit ScopedEnv(const char* name) : name_(name), had_(false)
      {
        const char* v = std::getenv(name_);
        if (v) { had_ = true; saved_ = v; }
      }
      void set(const char* value) { setenv(name_, value, 1); }
      void unset() { unsetenv(name_); }
      ~ScopedEnv()
      {
        if (had_) setenv(name_, saved_.c_str(), 1);
        else      unsetenv(name_);
      }
    private:
      const char* name_;
      bool        had_;
      std::string saved_;
    };

    // Resolves the absolute path of the loadable libibverbs on this host, or
    // an empty string if neither soname can be opened.
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

    bool LibibverbsAvailable() { return !LibibverbsPath().empty(); }

    // Byte-copies `src` to a uniquely-named temp file. NCCL_IBVERBS_LIB's
    // hardcoded fallbacks only ever try "libibverbs.so[.1]", so a load from
    // this path can only have come through the override, not a fallback.
    std::string MakeUniqueLibCopy(const std::string& src)
    {
      char tmpl[] = "/tmp/ibvsymbols_test_XXXXXX";
      int fd = mkstemp(tmpl);
      if (fd < 0) return {};
      close(fd);
      std::string dst = tmpl;

      std::ifstream in(src, std::ios::binary);
      std::ofstream out(dst, std::ios::binary | std::ios::trunc);
      if (!in || !out) { unlink(dst.c_str()); return {}; }
      out << in.rdbuf();
      if (!out) { unlink(dst.c_str()); return {}; }
      return dst;
    }

    // Proof that NCCL_IBVERBS_LIB was actually honored: something dlopen'd
    // this exact path. Success alone wouldn't rule out a silently-ignored
    // override that happened to succeed via a fallback instead.
    bool IsLoadedInProcess(const std::string& path)
    {
      struct Ctx { const std::string* path; bool found; } ctx{&path, false};
      dl_iterate_phdr(
        [](struct dl_phdr_info* info, size_t, void* data) -> int {
          auto* c = static_cast<Ctx*>(data);
          if (info->dlpi_name && *c->path == info->dlpi_name) { c->found = true; return 1; }
          return 0;
        },
        &ctx);
      return ctx.found;
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

    ScopedEnv env(kEnvVar);
    env.unset();

    ncclIbvSymbols symbols = {};
    ASSERT_EQ(buildIbvSymbols(&symbols), ncclSuccess);
    ExpectSymbolsLoaded(symbols);
  }

  // Env set to a uniquely-named copy of libibverbs, so a successful load can
  // only be explained by the override, not a fallback coincidentally
  // resolving to the same file. This is the one case that would fail
  // outright if the NCCL_IBVERBS_LIB code were removed.
  TEST(IbvSymbolsEnv, EnvOverrideValidSoname)
  {
    std::string libPath = LibibverbsPath();
    if (libPath.empty())
      GTEST_SKIP() << "libibverbs not installed on this host";

    std::string copyPath = MakeUniqueLibCopy(libPath);
    if (copyPath.empty())
      GTEST_SKIP() << "could not create a temp copy of libibverbs";

    ScopedEnv env(kEnvVar);
    env.set(copyPath.c_str());

    ncclIbvSymbols symbols = {};
    ncclResult_t result = buildIbvSymbols(&symbols);
    unlink(copyPath.c_str());

    ASSERT_EQ(result, ncclSuccess);
    ExpectSymbolsLoaded(symbols);
    EXPECT_TRUE(IsLoadedInProcess(copyPath))
        << "NCCL_IBVERBS_LIB override was not actually dlopen'd";
  }

  // Env set to a nonexistent path: index 0 fails to dlopen, the loop
  // continues and recovers via the hardcoded libibverbs.so[.1] fallback.
  TEST(IbvSymbolsEnv, BogusPathFallsBackToDefault)
  {
    if (!LibibverbsAvailable())
      GTEST_SKIP() << "libibverbs not installed on this host";

    ScopedEnv env(kEnvVar);
    env.set("/nonexistent/path/libibverbs.so");

    ncclIbvSymbols symbols = {};
    ASSERT_EQ(buildIbvSymbols(&symbols), ncclSuccess);
    ExpectSymbolsLoaded(symbols);
  }

  // NCCL_IBVERBS_LIB unset, alias set to a uniquely-named copy: the alias is
  // honored the same way, proven with the same dlopen check as above.
  TEST(IbvSymbolsEnv, AliasOverrideValidSoname)
  {
    std::string libPath = LibibverbsPath();
    if (libPath.empty())
      GTEST_SKIP() << "libibverbs not installed on this host";

    std::string copyPath = MakeUniqueLibCopy(libPath);
    if (copyPath.empty())
      GTEST_SKIP() << "could not create a temp copy of libibverbs";

    ScopedEnv primary(kEnvVar);
    primary.unset();
    ScopedEnv alias(kAliasEnvVar);
    alias.set(copyPath.c_str());

    ncclIbvSymbols symbols = {};
    ncclResult_t result = buildIbvSymbols(&symbols);
    unlink(copyPath.c_str());

    ASSERT_EQ(result, ncclSuccess);
    ExpectSymbolsLoaded(symbols);
    EXPECT_TRUE(IsLoadedInProcess(copyPath))
        << "NCCL_LIBIBVERBS_SO alias was not actually dlopen'd";
  }

  // Both set to distinct unique copies: NCCL_IBVERBS_LIB must win over the
  // NCCL_LIBIBVERBS_SO alias.
  TEST(IbvSymbolsEnv, PrimaryTakesPrecedenceOverAlias)
  {
    std::string libPath = LibibverbsPath();
    if (libPath.empty())
      GTEST_SKIP() << "libibverbs not installed on this host";

    std::string primaryCopy = MakeUniqueLibCopy(libPath);
    std::string aliasCopy   = MakeUniqueLibCopy(libPath);
    if (primaryCopy.empty() || aliasCopy.empty())
    {
      if (!primaryCopy.empty()) unlink(primaryCopy.c_str());
      if (!aliasCopy.empty())   unlink(aliasCopy.c_str());
      GTEST_SKIP() << "could not create temp copies of libibverbs";
    }

    ScopedEnv primary(kEnvVar);
    primary.set(primaryCopy.c_str());
    ScopedEnv alias(kAliasEnvVar);
    alias.set(aliasCopy.c_str());

    ncclIbvSymbols symbols = {};
    ncclResult_t result = buildIbvSymbols(&symbols);
    unlink(primaryCopy.c_str());
    unlink(aliasCopy.c_str());

    ASSERT_EQ(result, ncclSuccess);
    ExpectSymbolsLoaded(symbols);
    EXPECT_TRUE(IsLoadedInProcess(primaryCopy))
        << "NCCL_IBVERBS_LIB was not honored";
    EXPECT_FALSE(IsLoadedInProcess(aliasCopy))
        << "alias should not be used when NCCL_IBVERBS_LIB is set";
  }
}  // namespace RcclUnitTesting
