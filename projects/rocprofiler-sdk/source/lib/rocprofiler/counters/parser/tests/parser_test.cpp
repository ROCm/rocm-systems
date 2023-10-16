#include <map>
#include <string>

#include <gtest/gtest.h>

#include "lib/rocprofiler/counters/metrics.hpp"
#include "lib/rocprofiler/counters/parser/reader.hpp"

TEST(parser, base_ops)
{
    std::map<std::string, std::string> expressionToExpected = {
        {"AB + BA",
         "{\"Type\":\"ADDITION_NODE\", \"Operation\":\"NONE\",\"ReferenceSet\":[], "
         "\"CounterSet\":[{\"Type\":\"REFERENCE_NODE\", \"Operation\":\"NONE\", "
         "\"Value\":\"AB\",\"ReferenceSet\":[], \"CounterSet\":[]},{\"Type\":\"REFERENCE_NODE\", "
         "\"Operation\":\"NONE\", \"Value\":\"BA\",\"ReferenceSet\":[], \"CounterSet\":[]}]}"},
        {"CD - ZX",
         "{\"Type\":\"SUBTRACTION_NODE\", \"Operation\":\"NONE\",\"ReferenceSet\":[], "
         "\"CounterSet\":[{\"Type\":\"REFERENCE_NODE\", \"Operation\":\"NONE\", "
         "\"Value\":\"CD\",\"ReferenceSet\":[], \"CounterSet\":[]},{\"Type\":\"REFERENCE_NODE\", "
         "\"Operation\":\"NONE\", \"Value\":\"ZX\",\"ReferenceSet\":[], \"CounterSet\":[]}]}"},
        {"NM / DB",
         "{\"Type\":\"DIVIDE_NODE\", \"Operation\":\"NONE\",\"ReferenceSet\":[], "
         "\"CounterSet\":[{\"Type\":\"REFERENCE_NODE\", \"Operation\":\"NONE\", "
         "\"Value\":\"NM\",\"ReferenceSet\":[], \"CounterSet\":[]},{\"Type\":\"REFERENCE_NODE\", "
         "\"Operation\":\"NONE\", \"Value\":\"DB\",\"ReferenceSet\":[], \"CounterSet\":[]}]}"},
        {"AB * BA",
         "{\"Type\":\"MULTIPLY_NODE\", \"Operation\":\"NONE\",\"ReferenceSet\":[], "
         "\"CounterSet\":[{\"Type\":\"REFERENCE_NODE\", \"Operation\":\"NONE\", "
         "\"Value\":\"AB\",\"ReferenceSet\":[], \"CounterSet\":[]},{\"Type\":\"REFERENCE_NODE\", "
         "\"Operation\":\"NONE\", \"Value\":\"BA\",\"ReferenceSet\":[], \"CounterSet\":[]}]}"}};

    for(auto [op, expected] : expressionToExpected)
    {
        RawAST* ast = nullptr;
        auto*   buf = yy_scan_string(op.c_str());
        yyparse(&ast);
        ASSERT_TRUE(ast);
        EXPECT_EQ(fmt::format("{}", *ast), expected);
        yy_delete_buffer(buf);
        delete ast;
    }
}

TEST(parser, order_of_ops)
{
    std::map<std::string, std::string> expressionToExpected = {
        {"(AB + BA) / CD",
         "{\"Type\":\"DIVIDE_NODE\", \"Operation\":\"NONE\",\"ReferenceSet\":[], "
         "\"CounterSet\":[{\"Type\":\"ADDITION_NODE\", \"Operation\":\"NONE\",\"ReferenceSet\":[], "
         "\"CounterSet\":[{\"Type\":\"REFERENCE_NODE\", \"Operation\":\"NONE\", "
         "\"Value\":\"AB\",\"ReferenceSet\":[], \"CounterSet\":[]},{\"Type\":\"REFERENCE_NODE\", "
         "\"Operation\":\"NONE\", \"Value\":\"BA\",\"ReferenceSet\":[], "
         "\"CounterSet\":[]}]},{\"Type\":\"REFERENCE_NODE\", \"Operation\":\"NONE\", "
         "\"Value\":\"CD\",\"ReferenceSet\":[], \"CounterSet\":[]}]}"},
        {"AD / (CD - ZX)",
         "{\"Type\":\"DIVIDE_NODE\", \"Operation\":\"NONE\",\"ReferenceSet\":[], "
         "\"CounterSet\":[{\"Type\":\"REFERENCE_NODE\", \"Operation\":\"NONE\", "
         "\"Value\":\"AD\",\"ReferenceSet\":[], \"CounterSet\":[]},{\"Type\":\"SUBTRACTION_NODE\", "
         "\"Operation\":\"NONE\",\"ReferenceSet\":[], "
         "\"CounterSet\":[{\"Type\":\"REFERENCE_NODE\", \"Operation\":\"NONE\", "
         "\"Value\":\"CD\",\"ReferenceSet\":[], \"CounterSet\":[]},{\"Type\":\"REFERENCE_NODE\", "
         "\"Operation\":\"NONE\", \"Value\":\"ZX\",\"ReferenceSet\":[], \"CounterSet\":[]}]}]}"},
        {"MN * (NM / DB)",
         "{\"Type\":\"MULTIPLY_NODE\", \"Operation\":\"NONE\",\"ReferenceSet\":[], "
         "\"CounterSet\":[{\"Type\":\"REFERENCE_NODE\", \"Operation\":\"NONE\", "
         "\"Value\":\"MN\",\"ReferenceSet\":[], \"CounterSet\":[]},{\"Type\":\"DIVIDE_NODE\", "
         "\"Operation\":\"NONE\",\"ReferenceSet\":[], "
         "\"CounterSet\":[{\"Type\":\"REFERENCE_NODE\", \"Operation\":\"NONE\", "
         "\"Value\":\"NM\",\"ReferenceSet\":[], \"CounterSet\":[]},{\"Type\":\"REFERENCE_NODE\", "
         "\"Operation\":\"NONE\", \"Value\":\"DB\",\"ReferenceSet\":[], \"CounterSet\":[]}]}]}"},
        {"(AB / BA) - BN",
         "{\"Type\":\"SUBTRACTION_NODE\", \"Operation\":\"NONE\",\"ReferenceSet\":[], "
         "\"CounterSet\":[{\"Type\":\"DIVIDE_NODE\", \"Operation\":\"NONE\",\"ReferenceSet\":[], "
         "\"CounterSet\":[{\"Type\":\"REFERENCE_NODE\", \"Operation\":\"NONE\", "
         "\"Value\":\"AB\",\"ReferenceSet\":[], \"CounterSet\":[]},{\"Type\":\"REFERENCE_NODE\", "
         "\"Operation\":\"NONE\", \"Value\":\"BA\",\"ReferenceSet\":[], "
         "\"CounterSet\":[]}]},{\"Type\":\"REFERENCE_NODE\", \"Operation\":\"NONE\", "
         "\"Value\":\"BN\",\"ReferenceSet\":[], \"CounterSet\":[]}]}"}};

    for(auto [op, expected] : expressionToExpected)
    {
        RawAST* ast = nullptr;
        auto*   buf = yy_scan_string(op.c_str());
        yyparse(&ast);
        ASSERT_TRUE(ast);
        EXPECT_EQ(fmt::format("{}", *ast), expected);
        yy_delete_buffer(buf);
        delete ast;
    }
}

TEST(parser, reduction)
{
    std::map<std::string, std::string> expressionToExpected = {
        {"reduce(AB, SUM)",
         "{\"Type\":\"REDUCE_NODE\", \"Operation\":\"NONE\", "
         "\"Value\":\"SUM\",\"ReferenceSet\":[], \"CounterSet\":[{\"Type\":\"REFERENCE_NODE\", "
         "\"Operation\":\"NONE\", \"Value\":\"AB\",\"ReferenceSet\":[], \"CounterSet\":[]}]}"},
        {"reduce(AB+CD, SUM)",
         "{\"Type\":\"REDUCE_NODE\", \"Operation\":\"NONE\", "
         "\"Value\":\"SUM\",\"ReferenceSet\":[], \"CounterSet\":[{\"Type\":\"ADDITION_NODE\", "
         "\"Operation\":\"NONE\",\"ReferenceSet\":[], "
         "\"CounterSet\":[{\"Type\":\"REFERENCE_NODE\", \"Operation\":\"NONE\", "
         "\"Value\":\"AB\",\"ReferenceSet\":[], \"CounterSet\":[]},{\"Type\":\"REFERENCE_NODE\", "
         "\"Operation\":\"NONE\", \"Value\":\"CD\",\"ReferenceSet\":[], \"CounterSet\":[]}]}]}"},
        {"reduce(AB,DIV)+reduce(DC,SUM)",
         "{\"Type\":\"ADDITION_NODE\", \"Operation\":\"NONE\",\"ReferenceSet\":[], "
         "\"CounterSet\":[{\"Type\":\"REDUCE_NODE\", \"Operation\":\"NONE\", "
         "\"Value\":\"DIV\",\"ReferenceSet\":[], \"CounterSet\":[{\"Type\":\"REFERENCE_NODE\", "
         "\"Operation\":\"NONE\", \"Value\":\"AB\",\"ReferenceSet\":[], "
         "\"CounterSet\":[]}]},{\"Type\":\"REDUCE_NODE\", \"Operation\":\"NONE\", "
         "\"Value\":\"SUM\",\"ReferenceSet\":[], \"CounterSet\":[{\"Type\":\"REFERENCE_NODE\", "
         "\"Operation\":\"NONE\", \"Value\":\"DC\",\"ReferenceSet\":[], \"CounterSet\":[]}]}]}"},
        {"reduce(AB, SUM, shader)",
         "{\"Type\":\"REDUCE_NODE\", \"Operation\":\"NONE\", "
         "\"Value\":\"SUM\",\"ReferenceSet\":[{\"Type\":\"REFERENCE_NODE\", "
         "\"Operation\":\"NONE\", \"Value\":\"shader\",\"ReferenceSet\":[], \"CounterSet\":[]}], "
         "\"CounterSet\":[{\"Type\":\"REFERENCE_NODE\", \"Operation\":\"NONE\", "
         "\"Value\":\"AB\",\"ReferenceSet\":[], \"CounterSet\":[]}]}"}};

    for(auto [op, expected] : expressionToExpected)
    {
        RawAST* ast = nullptr;
        auto*   buf = yy_scan_string(op.c_str());
        yyparse(&ast);
        ASSERT_TRUE(ast);
        EXPECT_EQ(fmt::format("{}", *ast), expected);
        yy_delete_buffer(buf);
        delete ast;
    }
}

TEST(parser, selection)
{
    std::map<std::string, std::string> expressionToExpected = {
        {"select(AB, SUM)",
         "{\"Type\":\"SELECT_NODE\", \"Operation\":\"NONE\", "
         "\"Value\":\"SUM\",\"ReferenceSet\":[], \"CounterSet\":[{\"Type\":\"REFERENCE_NODE\", "
         "\"Operation\":\"NONE\", \"Value\":\"AB\",\"ReferenceSet\":[], \"CounterSet\":[]}]}"},
        {"select(AB+CD, SUM)",
         "{\"Type\":\"SELECT_NODE\", \"Operation\":\"NONE\", "
         "\"Value\":\"SUM\",\"ReferenceSet\":[], \"CounterSet\":[{\"Type\":\"ADDITION_NODE\", "
         "\"Operation\":\"NONE\",\"ReferenceSet\":[], "
         "\"CounterSet\":[{\"Type\":\"REFERENCE_NODE\", \"Operation\":\"NONE\", "
         "\"Value\":\"AB\",\"ReferenceSet\":[], \"CounterSet\":[]},{\"Type\":\"REFERENCE_NODE\", "
         "\"Operation\":\"NONE\", \"Value\":\"CD\",\"ReferenceSet\":[], \"CounterSet\":[]}]}]}"},
        {"select(AB,DIV)+select(DC,SUM)",
         "{\"Type\":\"ADDITION_NODE\", \"Operation\":\"NONE\",\"ReferenceSet\":[], "
         "\"CounterSet\":[{\"Type\":\"SELECT_NODE\", \"Operation\":\"NONE\", "
         "\"Value\":\"DIV\",\"ReferenceSet\":[], \"CounterSet\":[{\"Type\":\"REFERENCE_NODE\", "
         "\"Operation\":\"NONE\", \"Value\":\"AB\",\"ReferenceSet\":[], "
         "\"CounterSet\":[]}]},{\"Type\":\"SELECT_NODE\", \"Operation\":\"NONE\", "
         "\"Value\":\"SUM\",\"ReferenceSet\":[], \"CounterSet\":[{\"Type\":\"REFERENCE_NODE\", "
         "\"Operation\":\"NONE\", \"Value\":\"DC\",\"ReferenceSet\":[], \"CounterSet\":[]}]}]}"},
        {"select(AB, SUM, shader)",
         "{\"Type\":\"SELECT_NODE\", \"Operation\":\"NONE\", "
         "\"Value\":\"SUM\",\"ReferenceSet\":[{\"Type\":\"REFERENCE_NODE\", "
         "\"Operation\":\"NONE\", \"Value\":\"shader\",\"ReferenceSet\":[], \"CounterSet\":[]}], "
         "\"CounterSet\":[{\"Type\":\"REFERENCE_NODE\", \"Operation\":\"NONE\", "
         "\"Value\":\"AB\",\"ReferenceSet\":[], \"CounterSet\":[]}]}"}};

    for(auto [op, expected] : expressionToExpected)
    {
        RawAST* ast = nullptr;
        auto*   buf = yy_scan_string(op.c_str());
        yyparse(&ast);
        ASSERT_TRUE(ast);
        EXPECT_EQ(fmt::format("{}", *ast), expected);
        yy_delete_buffer(buf);
        delete ast;
    }
}

TEST(parser, parse_derived_counters)
{
    // Checks that ASTs are properly formed from derived counters defined in XML
    // Does not check accuracy, only parseability
    auto derived_counters = rocprofiler::counters::getDerivedHardwareMetrics();
    for(auto& [gfx, counter_list] : derived_counters)
    {
        for(const auto& v : counter_list)
        {
            RawAST* ast = nullptr;
            auto*   buf = yy_scan_string(v.expression().c_str());
            yyparse(&ast);
            ASSERT_TRUE(ast);
            yy_delete_buffer(buf);
            delete ast;
        }
    }
}
