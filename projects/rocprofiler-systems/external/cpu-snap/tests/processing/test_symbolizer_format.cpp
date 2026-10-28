// Unit tests for symbolizer_config::include_binary_name via
// format_with_binary(). The ResolveInline* tests run the full addr2line -i
// pipeline on this binary and need -g and -O2 so always_inline helpers fold.

#include <array>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

#include <gtest/gtest.h>

#include <sys/auxv.h>
#include <sys/stat.h>
#include <unistd.h>

#include "cpu_snap/processing/symbols/symbolizer.hpp"

namespace
{

using cpu_snap::processing::symbols::symbolizer;
using cpu_snap::processing::symbols::symbolizer_config;

// Deliberately-inlined chain. At -O2 these collapse into the body of
// inline_outer_workload, so the IP of inline_outer_workload should
// resolve to a multi-entry inline chain.
[[gnu::always_inline]] inline double inline_chain_leaf(double x) noexcept
{
    return x * 1.0000001 + 1.0;
}
[[gnu::always_inline]] inline double inline_chain_mid(double x) noexcept
{
    return inline_chain_leaf(x) + 0.5;
}
[[gnu::noinline]] double inline_outer_workload(double x) noexcept
{
    double acc = x;
    for(int i = 0; i < 32; ++i)
    {
        acc = inline_chain_mid(acc);
    }
    return acc;
}

// Installs a fake `addr2line` script at the front of PATH for the object's lifetime.
class scoped_fake_addr2line
{
public:
    explicit scoped_fake_addr2line(const std::string& script_body)
    {
        std::array<char, 64> dir_template{};
        std::snprintf(dir_template.data(), dir_template.size(),
                      "/tmp/cpu_snap_fake_addr2line_XXXXXX");
        const char* dir = ::mkdtemp(dir_template.data());
        EXPECT_NE(dir, nullptr);
        m_dir = dir;

        m_script_path = m_dir + "/addr2line";
        FILE* file    = std::fopen(m_script_path.c_str(), "w");
        EXPECT_NE(file, nullptr);
        std::fputs(script_body.c_str(), file);
        std::fclose(file);
        ::chmod(m_script_path.c_str(), 0700);

        const char* old_path       = std::getenv("PATH");
        m_old_path                 = (old_path != nullptr) ? old_path : "";
        const std::string new_path = m_dir + ":" + m_old_path;
        ::setenv("PATH", new_path.c_str(), 1);
    }

    scoped_fake_addr2line(const scoped_fake_addr2line&)            = delete;
    scoped_fake_addr2line& operator=(const scoped_fake_addr2line&) = delete;

    ~scoped_fake_addr2line()
    {
        ::setenv("PATH", m_old_path.c_str(), 1);
        ::unlink(m_script_path.c_str());
        ::rmdir(m_dir.c_str());
    }

private:
    std::string m_dir;
    std::string m_script_path;
    std::string m_old_path;
};

}  // namespace

TEST(SymbolizerFormat, DefaultFormatIsFunctionSpaceBracketBinary)
{
    const symbolizer sym;
    EXPECT_EQ(sym.format_with_binary("do_work(int)", "demo"),
              "do_work(int) [demo]");
}

TEST(SymbolizerFormat, IncludeBinaryFalseDropsSuffix)
{
    const symbolizer sym(symbolizer_config{ .include_binary_name = false });
    EXPECT_EQ(sym.format_with_binary("do_work(int)", "demo"), "do_work(int)");
}

TEST(SymbolizerFormat, HexFallbackHonorsIncludeBinaryDefault)
{
    // The unresolved-IP path in resolve_batch funnels "0x<hex>" through
    // the same formatter; the result with default config is
    // "0x<hex> [binary]".
    const symbolizer sym;
    EXPECT_EQ(sym.format_with_binary("0xdeadbeef", "libc.so.6"),
              "0xdeadbeef [libc.so.6]");
}

TEST(SymbolizerFormat, HexFallbackWithBinaryFalseIsBareHex)
{
    const symbolizer sym(symbolizer_config{ .include_binary_name = false });
    EXPECT_EQ(sym.format_with_binary("0xdeadbeef", "libc.so.6"), "0xdeadbeef");
}

TEST(SymbolizerFormat, EmptyBinaryStillRendersBracketsWhenEnabled)
{
    // Defensive: pathological caller passes an empty base. The formatter
    // does not invent content; it just emits " []" so the toggle's
    // boolean contract holds (suffix present iff include_binary_name == true).
    const symbolizer sym;
    EXPECT_EQ(sym.format_with_binary("main", ""), "main []");
}

TEST(SymbolizerResolveInline, EmptyChainWhenIpNotPreloaded)
{
    const symbolizer sym;
    EXPECT_TRUE(sym.resolve_inline(0xdeadbeefULL).empty());
}

TEST(SymbolizerResolveInline, ResolveReturnsBackOfChain)
{
    // Anchor on a noinline function: its first instruction IP must
    // resolve, and resolve()/resolve_inline().back() must agree.
    symbolizer sym(symbolizer_config{ .include_binary_name = false,
                                      .inline_frames       = true });

    // Force the address of inline_outer_workload to escape so the
    // linker does not eliminate it.
    const auto ip = reinterpret_cast<std::uint64_t>(&inline_outer_workload);
    sym.preload({ ip });

    const auto chain = sym.resolve_inline(ip);
    ASSERT_FALSE(chain.empty()) << "expected at least one resolved frame";
    EXPECT_EQ(sym.resolve(ip), chain.back());
}

TEST(SymbolizerResolveInline, ChainSurfacesInlinedFramesAtO2)
{
    // The body of inline_outer_workload calls inline_chain_mid which
    // calls inline_chain_leaf; both are always_inline so at -O2 the
    // outer's body contains DWARF inline info for all three. An IP
    // inside the outer function should therefore resolve to a chain
    // of length >= 2 (the outermost entry is the outer function).
    //
    // To get an IP that lands in the inlined body we offset from the
    // start of the outer; +16 bytes is enough to clear the prologue
    // on x86_64 -O2 but stays inside the function.
    symbolizer sym(symbolizer_config{ .include_binary_name = false,
                                      .inline_frames       = true });
    const auto fn_start = reinterpret_cast<std::uint64_t>(&inline_outer_workload);
    const auto ip       = fn_start + 16;
    sym.preload({ fn_start, ip });

    const auto chain_inside = sym.resolve_inline(ip);
    ASSERT_FALSE(chain_inside.empty());
    // Outermost frame is the noinline outer function name.
    EXPECT_NE(chain_inside.back().find("inline_outer_workload"),
              std::string::npos)
        << "outermost frame should name the noinline caller; got: "
        << chain_inside.back();
    // If addr2line surfaces inline info (-g, -O2, .debug_info present),
    // we expect at least one of the inline helpers to appear earlier
    // in the chain.
    if(chain_inside.size() >= 2)
    {
        bool saw_inlined = false;
        for(std::size_t i = 0; i + 1 < chain_inside.size(); ++i)
        {
            const auto& f = chain_inside[i];
            if(f.find("inline_chain_leaf") != std::string::npos
               || f.find("inline_chain_mid") != std::string::npos)
            {
                saw_inlined = true;
                break;
            }
        }
        EXPECT_TRUE(saw_inlined)
            << "chain has >1 entry but no inline helper surfaced";
    }
}

TEST(SymbolizerResolveInline, ResolveInlineRespectsConfigOff)
{
    // With inline_frames=false (the default), resolve_inline must collapse
    // the chain to a single outer-frame entry even for an IP with a known
    // multi-frame inline chain. Compare against a sibling symbolizer with
    // inline_frames=true to confirm the multi-frame chain is the natural
    // shape that gets filtered.
    const auto fn_start = reinterpret_cast<std::uint64_t>(&inline_outer_workload);
    const auto ip       = fn_start + 16;

    symbolizer sym_off(symbolizer_config{ .include_binary_name = false,
                                          .inline_frames       = false });
    sym_off.preload({ fn_start, ip });
    const auto chain_off = sym_off.resolve_inline(ip);
    ASSERT_EQ(chain_off.size(), 1U)
        << "inline_frames=false must collapse to a single outer frame";
    EXPECT_EQ(chain_off.back(), sym_off.resolve(ip));
}

// An unresolved IP in the middle of a batch must not shift later IPs' names.
TEST(SymbolizerResolveInline, UnresolvedIpInMiddleOfBatchDoesNotShiftLaterNames)
{
    const std::string     script = "#!/bin/sh\n"
                                   "printf 'outer_name_one\\nfile_one.cpp:10\\n'\n"
                                   "printf '??\\n??:0\\n'\n"
                                   "printf '??\\n??:0\\n'\n"
                                   "printf '??\\n??:0\\n'\n"
                                   "printf 'outer_name_two\\nfile_two.cpp:20\\n'\n";
    scoped_fake_addr2line fake_addr2line(script);

    const auto base          = reinterpret_cast<std::uint64_t>(&inline_outer_workload);
    const auto ip_a          = base;
    const auto ip_unresolved = base + 1;
    const auto ip_b          = base + 2;

    symbolizer sym(symbolizer_config{ .include_binary_name = false });
    sym.preload({ ip_a, ip_unresolved, ip_b });

    EXPECT_EQ(sym.resolve(ip_a), "outer_name_one");
    EXPECT_EQ(sym.resolve(ip_b), "outer_name_two")
        << "later IP's name was shifted by the unresolved IP in between";
}

TEST(SymbolizerVdso, LiveVdsoIpResolvesToVdsoPlusOffset)
{
    const auto vdso_base = static_cast<std::uint64_t>(getauxval(AT_SYSINFO_EHDR));
    if(vdso_base == 0) GTEST_SKIP() << "no vDSO mapped";

    constexpr std::uint64_t k_offset = 0x10;
    symbolizer              sym;
    sym.preload({ vdso_base + k_offset });

    EXPECT_EQ(std::string{ sym.resolve(vdso_base + k_offset) }, "[vdso]+0x10");
    EXPECT_FALSE(sym.symbol_identity(vdso_base + k_offset).has_value());
}

TEST(SymbolizerVdso, IpOutsideEveryMappingStaysRawHex)
{
    symbolizer sym;
    sym.preload({ 0x10 });

    EXPECT_EQ(std::string{ sym.resolve(0x10) }, "0x10");
}
