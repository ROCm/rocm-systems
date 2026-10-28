// Unit tests for cpu_snap::processing::symbols::detail::find_symbol.

#include <gtest/gtest.h>

#include <array>
#include <optional>
#include <span>
#include <string>

#include "cpu_snap/processing/symbols/elf_symbols.hpp"

namespace
{

using cpu_snap::processing::symbols::detail::elf_symbol;
using cpu_snap::processing::symbols::detail::find_symbol;
using cpu_snap::processing::symbols::detail::open_elf64;

constexpr std::array<elf_symbol, 3> k_symbols{ {
    { .st_value = 0x1000, .st_size = 0x10, .name = "fn_a" },
    { .st_value = 0x1020, .st_size = 0x20, .name = "fn_b" },  // gap: [0x1010, 0x1020)
    { .st_value = 0x1040, .st_size = 0x10, .name = "fn_c" },
} };  // NOLINT(readability-trailing-comma) -- outer brace is the array init, not a list

static_assert(find_symbol(0x1000, k_symbols)->name == "fn_a");
static_assert(find_symbol(0x100f, k_symbols)->name == "fn_a");
static_assert(!find_symbol(0x1010, k_symbols).has_value());  // in the gap
static_assert(!find_symbol(0x101f, k_symbols).has_value());  // in the gap
static_assert(find_symbol(0x1020, k_symbols)->name == "fn_b");
static_assert(find_symbol(0x103f, k_symbols)->name == "fn_b");
static_assert(!find_symbol(0x0fff, k_symbols).has_value());  // before first symbol
static_assert(find_symbol(0x104f, k_symbols)->name == "fn_c");
static_assert(!find_symbol(0x1050, k_symbols).has_value());  // past last symbol's range

}  // namespace

TEST(FindSymbol, InsideFirstSymbolResolves)
{
    const auto found = find_symbol(0x1005, k_symbols);
    ASSERT_TRUE(found.has_value());
    EXPECT_EQ(found->name, "fn_a");
}

TEST(FindSymbol, InGapBetweenSymbolsReturnsNullopt)
{
    EXPECT_FALSE(find_symbol(0x1015, k_symbols).has_value());
}

TEST(FindSymbol, PastLastSymbolReturnsNullopt)
{
    EXPECT_FALSE(find_symbol(0x1050, k_symbols).has_value());
}

TEST(FindSymbol, BeforeFirstSymbolReturnsNullopt)
{
    EXPECT_FALSE(find_symbol(0x0ff0, k_symbols).has_value());
}

TEST(FindSymbol, AtExactUpperBoundaryIsExcluded)
{
    // st_value + st_size is exclusive.
    EXPECT_FALSE(find_symbol(0x1010, k_symbols).has_value());
}

TEST(FindSymbol, EmptySymbolListReturnsNullopt)
{
    EXPECT_FALSE(find_symbol(0x1000, std::span<const elf_symbol>{}).has_value());
}

TEST(ReadElfSymbolTable, MissingFileReturnsEmptyTable)
{
    const auto table = cpu_snap::processing::symbols::detail::read_elf_symbol_table("/nonexistent/path/xyz");
    EXPECT_TRUE(table.symbols.empty());
    EXPECT_FALSE(table.has_debug_info);
}

TEST(OpenElf64, SelfExecutableOpensWithValidHeader)
{
    const auto elf = open_elf64("/proc/self/exe");
    ASSERT_TRUE(elf.has_value());
    EXPECT_NE(elf->file, nullptr);
    EXPECT_EQ(elf->ehdr.e_ident[EI_CLASS], ELFCLASS64);
    EXPECT_GT(elf->ehdr.e_phnum, 0);
}

TEST(OpenElf64, NonElfFileIsRejected)
{
    EXPECT_FALSE(open_elf64("/etc/passwd").has_value());
}

TEST(OpenElf64, MissingFileIsRejected)
{
    EXPECT_FALSE(open_elf64("/nonexistent/cpu_snap_no_such_file").has_value());
}
