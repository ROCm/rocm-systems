// Regression test for symbolizing IPs inside an ELF whose PT_LOAD
// p_offset != p_vaddr (lld's layout under -z noseparate-code).
//
// The fixture library (fixtures/lld_mismatch_lib.cpp) is only built
// when ld.lld is available; see tests/sampling/CMakeLists.txt.

#include <gtest/gtest.h>

#include "cpu_snap/processing/symbols/symbolizer.hpp"

#if defined(CPU_SNAP_HAVE_LLD_MISMATCH_LIB)

#include <cstdint>
#include <dlfcn.h>
#include <string>

namespace
{
using cpu_snap::processing::symbols::symbolizer;

std::string resolve_symbol(const char* fn_name)
{
    void* handle = ::dlopen(CPU_SNAP_LLD_MISMATCH_LIB_PATH, RTLD_NOW | RTLD_LOCAL);
    if(handle == nullptr)
    {
        ADD_FAILURE() << "dlopen(" << CPU_SNAP_LLD_MISMATCH_LIB_PATH
                       << ") failed: " << ::dlerror();
        return {};
    }

    void* sym = ::dlsym(handle, fn_name);
    if(sym == nullptr)
    {
        ::dlclose(handle);
        ADD_FAILURE() << "dlsym(" << fn_name << ") failed: " << ::dlerror();
        return {};
    }

    symbolizer sym_resolver;
    sym_resolver.rescan_maps();
    const auto ip = reinterpret_cast<std::uint64_t>(sym);
    sym_resolver.preload({ ip });
    std::string resolved{ sym_resolver.resolve(ip) };

    ::dlclose(handle);
    return resolved;
}
}  // namespace

TEST(ProcMapsLldMismatch, ResolvesExportedFunctionAcrossOffsetVaddrMismatch)
{
    const std::string resolved = resolve_symbol("cpu_snap_lld_fixture_exported_fn");
    EXPECT_NE(resolved.find("cpu_snap_lld_fixture_exported_fn"), std::string::npos)
        << "resolved: " << resolved;
}

TEST(ProcMapsLldMismatch, ResolvesLocalFunctionAcrossOffsetVaddrMismatch)
{
    const std::string resolved = resolve_symbol("cpu_snap_lld_fixture_call_static");
    EXPECT_NE(resolved.find("cpu_snap_lld_fixture_call_static"), std::string::npos)
        << "resolved: " << resolved;
}

#else

TEST(ProcMapsLldMismatch, SkippedNoLldToolchain)
{
    GTEST_SKIP() << "ld.lld not found at configure time; cannot build the "
                    "PT_LOAD offset/vaddr mismatch fixture";
}

#endif
