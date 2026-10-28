// addr2line names an address after the nearest PRECEDING symbol with
// no size check. In a stripped shared library (only .dynsym, no
// .symtab/.debug_info) an address that falls outside every symbol's
// [st_value, st_value+st_size) range must not be reported as the
// neighbouring exported function.

#include <gtest/gtest.h>

#include <array>
#include <cstdint>
#include <cstdio>
#include <string>

#include <dlfcn.h>
#include <stdio.h>  // NOLINT(misc-include-cleaner,modernize-deprecated-headers): POSIX popen/pclose live in <stdio.h>, not <cstdio>

#include "cpu_snap/processing/symbols/symbolizer.hpp"

#ifdef CPU_SNAP_STRIPPED_SYMBOL_LIB_PATH

namespace
{
using cpu_snap::processing::symbols::symbolizer;

/** Runs @p cmd via popen and returns its combined output. */
std::string shell_capture(const std::string& cmd)
{
    // NOLINTNEXTLINE(cert-env33-c,concurrency-mt-unsafe,bugprone-command-processor): test-only helper
    FILE* pipe = ::popen(cmd.c_str(), "r");
    if(pipe == nullptr) return {};
    std::string out;
    std::array<char, 4096> buf{};
    while(std::fgets(buf.data(), static_cast<int>(buf.size()), pipe) != nullptr)
    {
        out += buf.data();
    }
    [[maybe_unused]] const auto pclose_rc = ::pclose(pipe);
    return out;
}

// NOLINTNEXTLINE(readability-identifier-naming) -- gtest fixture must be CamelCase to satisfy google-readability-avoid-underscore-in-googletest-name
class StrippedSymbolTest : public ::testing::Test
{
protected:
    void SetUp() override
    {
        m_handle = ::dlopen(CPU_SNAP_STRIPPED_SYMBOL_LIB_PATH, RTLD_NOW | RTLD_LOCAL);
        ASSERT_NE(m_handle, nullptr) << ::dlerror();
    }

    void TearDown() override
    {
        if(m_handle != nullptr) ::dlclose(m_handle);
    }

    void* sym(const char* name) const
    {
        void* addr = ::dlsym(m_handle, name);
        EXPECT_NE(addr, nullptr) << name << ": " << ::dlerror();
        return addr;
    }

    void* m_handle = nullptr;
};

}  // namespace

TEST(StrippedSymbolFixtureSanity, HasNoDebugInfoOrSymtab)
{
    const std::string sections =
        shell_capture("readelf -S '" + std::string{ CPU_SNAP_STRIPPED_SYMBOL_LIB_PATH } + "' 2>/dev/null");
    EXPECT_EQ(sections.find(".debug_info"), std::string::npos) << sections;
    EXPECT_EQ(sections.find(".symtab"), std::string::npos) << sections;
}

TEST(StrippedSymbolFixtureSanity, DynsymStillExportsFnAWithSize)
{
    const std::string dynsym =
        shell_capture("nm -D -S '" + std::string{ CPU_SNAP_STRIPPED_SYMBOL_LIB_PATH } + "' 2>/dev/null");
    EXPECT_NE(dynsym.find("stripped_fixture_fn_a"), std::string::npos) << dynsym;
}

TEST_F(StrippedSymbolTest, AddressInsideExportedFunctionResolvesToItsName)
{
    const void* const fn_a = sym("stripped_fixture_fn_a");

    symbolizer sym_resolver;
    sym_resolver.rescan_maps();
    const auto ip = reinterpret_cast<std::uint64_t>(fn_a);
    sym_resolver.preload({ ip });

    const std::string resolved{ sym_resolver.resolve(ip) };
    EXPECT_NE(resolved.find("stripped_fixture_fn_a"), std::string::npos) << resolved;
}

TEST_F(StrippedSymbolTest, AddressOutsideEveryDynsymSymbolIsNotNamedAfterNeighbour)
{
    using get_addr_fn = void* (*) ();
    auto* get_static_fn_addr =
        reinterpret_cast<get_addr_fn>(sym("stripped_fixture_get_static_fn_addr"));
    ASSERT_NE(get_static_fn_addr, nullptr);
    const void* const static_fn = get_static_fn_addr();

    symbolizer sym_resolver;
    sym_resolver.rescan_maps();
    const auto ip = reinterpret_cast<std::uint64_t>(static_fn);
    sym_resolver.preload({ ip });

    const std::string resolved{ sym_resolver.resolve(ip) };
    EXPECT_EQ(resolved.find("stripped_fixture_fn_a"), std::string::npos) << resolved;
    EXPECT_EQ(resolved.find("stripped_fixture_fn_b"), std::string::npos) << resolved;
    EXPECT_NE(resolved.find("libstripped_symbol_fixture.so+0x"), std::string::npos) << resolved;
}

#endif  // CPU_SNAP_STRIPPED_SYMBOL_LIB_PATH
