#include <cstddef>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include <gtest/gtest.h>

#include "cpu_snap/processing/symbols/addr2line_client.hpp"

namespace
{

using cpu_snap::processing::symbols::addr2line_client;

auto line_source(std::vector<std::string> lines)
{
    return [lines = std::move(lines), idx = std::size_t{ 0 }]() mutable
           -> std::optional<std::string> {
        if(idx >= lines.size()) return std::nullopt;
        return lines[idx++];
    };
}

TEST(Addr2lineClient, ShellSingleQuotePlain)
{
    EXPECT_EQ(addr2line_client::shell_single_quote("/bin/ls"), "'/bin/ls'");
}

TEST(Addr2lineClient, ShellSingleQuoteEscapesEmbeddedQuote)
{
    EXPECT_EQ(addr2line_client::shell_single_quote("a'b"), "'a'\\''b'");
}

TEST(Addr2lineClient, BuildCommandSeparatesAddressesWithSentinel)
{
    const std::vector<addr2line_client::address_pair> pairs{ { 1, 0x10 }, { 2, 0x20 } };
    EXPECT_EQ(addr2line_client::build_command("/lib/x.so", pairs),
              "addr2line -ifCe '/lib/x.so' 2>/dev/null 0x10 0xffffffffffffffff 0x20");
}

TEST(Addr2lineClient, BuildCommandSingleAddressHasNoSentinel)
{
    const std::vector<addr2line_client::address_pair> pairs{ { 1, 0x10 } };
    EXPECT_EQ(addr2line_client::build_command("/b", pairs),
              "addr2line -ifCe '/b' 2>/dev/null 0x10");
}

TEST(Addr2lineClient, NormalizeFileLine)
{
    EXPECT_EQ(addr2line_client::normalize_file_line("??:0"), "");
    EXPECT_EQ(addr2line_client::normalize_file_line("??:?"), "");
    EXPECT_EQ(addr2line_client::normalize_file_line(""), "");
    EXPECT_EQ(addr2line_client::normalize_file_line("a.cpp:7"), "a.cpp:7");
}

TEST(Addr2lineClient, SentinelDetection)
{
    EXPECT_TRUE(addr2line_client::is_sentinel_function("??"));
    EXPECT_FALSE(addr2line_client::is_sentinel_function("main"));
    EXPECT_FALSE(addr2line_client::is_sentinel_function("???"));
}

TEST(Addr2lineClient, ReadChainParsesInlineChainAndStopsAtSentinel)
{
    auto src = line_source({ "leaf", "a.cpp:1", "outer", "b.cpp:2", "??", "??:0", "next", "c.cpp:3" });
    const auto chain = addr2line_client::read_chain(src);
    ASSERT_EQ(chain.functions.size(), 2U);
    EXPECT_EQ(chain.functions[0], "leaf");
    EXPECT_EQ(chain.functions[1], "outer");
    EXPECT_EQ(chain.file_line, "a.cpp:1");
    const auto following = addr2line_client::read_chain(src);
    ASSERT_EQ(following.functions.size(), 1U);
    EXPECT_EQ(following.functions[0], "next");
}

TEST(Addr2lineClient, ReadChainUnresolvedConsumesSeparatorPair)
{
    auto src = line_source({ "??", "??:0", "??", "??:0", "next", "c.cpp:3" });
    const auto unresolved = addr2line_client::read_chain(src);
    EXPECT_TRUE(unresolved.functions.empty());
    EXPECT_TRUE(unresolved.file_line.empty());
    const auto following = addr2line_client::read_chain(src);
    ASSERT_EQ(following.functions.size(), 1U);
    EXPECT_EQ(following.functions[0], "next");
}

TEST(Addr2lineClient, ReadChainFileLineComesFromFirstFrameOnly)
{
    auto src = line_source({ "leaf", "??:0", "outer", "b.cpp:2" });
    const auto chain = addr2line_client::read_chain(src);
    ASSERT_EQ(chain.functions.size(), 2U);
    EXPECT_TRUE(chain.file_line.empty());
}

TEST(Addr2lineClient, ReadChainEmptyOnEndOfInput)
{
    auto src = line_source({});
    EXPECT_TRUE(addr2line_client::read_chain(src).functions.empty());
}

}  // namespace
