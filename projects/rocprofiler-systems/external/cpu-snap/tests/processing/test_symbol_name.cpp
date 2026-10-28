#include <string>

#include <gtest/gtest.h>

#include "cpu_snap/processing/symbols/symbol_name.hpp"

namespace
{

using cpu_snap::processing::symbols::symbol_name;

TEST(SymbolName, CleanStripsCloneTag)
{
    std::string fn = "foo(int) [clone .cold]";
    symbol_name::clean(fn);
    EXPECT_EQ(fn, "foo(int)");
}

TEST(SymbolName, CleanStripsAnonymousNamespaceEverywhere)
{
    std::string fn = "(anonymous namespace)::a((anonymous namespace)::T)";
    symbol_name::clean(fn);
    EXPECT_EQ(fn, "a(T)");
}

TEST(SymbolName, CleanLeavesPlainNameAlone)
{
    std::string fn = "ns::f(int)";
    symbol_name::clean(fn);
    EXPECT_EQ(fn, "ns::f(int)");
}

TEST(SymbolName, DemangleMangledAndFallback)
{
    EXPECT_EQ(symbol_name::demangle("_ZN2ns1fEi"), "ns::f(int)");
    EXPECT_EQ(symbol_name::demangle("not_mangled"), "not_mangled");
}

}  // namespace
