// MIT License
//
// Copyright (c) 2023 Advanced Micro Devices, Inc. All rights reserved.
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.

#include <map>
#include <string>
#include <tuple>

#include <gtest/gtest.h>

#include "lib/rocprofiler-sdk/counters/metrics.hpp"
#include "lib/rocprofiler-sdk/counters/parser/reader.hpp"

TEST(parser, base_ops)
{
    std::map<std::string, std::string> expressionToExpected = {
        {"AB * BA",
         "{\"Type\":\"MULTIPLY_NODE\", \"REDUCE_OP\":\"\", "
         "\"Counter_Set\":[{\"Type\":\"REFERENCE_NODE\", \"REDUCE_OP\":\"\", \"Value\":\"AB\", "
         "\"Counter_Set\":[], \"Reduce_Dimension_Set\":[], "
         "\"Select_Dimension_Set\":[]},{\"Type\":\"REFERENCE_NODE\", \"REDUCE_OP\":\"\", "
         "\"Value\":\"BA\", \"Counter_Set\":[], \"Reduce_Dimension_Set\":[], "
         "\"Select_Dimension_Set\":[]}], \"Reduce_Dimension_Set\":[], "
         "\"Select_Dimension_Set\":[]}"},
        {"AB + BA",
         "{\"Type\":\"ADDITION_NODE\", \"REDUCE_OP\":\"\", "
         "\"Counter_Set\":[{\"Type\":\"REFERENCE_NODE\", \"REDUCE_OP\":\"\", \"Value\":\"AB\", "
         "\"Counter_Set\":[], \"Reduce_Dimension_Set\":[], "
         "\"Select_Dimension_Set\":[]},{\"Type\":\"REFERENCE_NODE\", \"REDUCE_OP\":\"\", "
         "\"Value\":\"BA\", \"Counter_Set\":[], \"Reduce_Dimension_Set\":[], "
         "\"Select_Dimension_Set\":[]}], \"Reduce_Dimension_Set\":[], "
         "\"Select_Dimension_Set\":[]}"},
        {"CD - ZX",
         "{\"Type\":\"SUBTRACTION_NODE\", \"REDUCE_OP\":\"\", "
         "\"Counter_Set\":[{\"Type\":\"REFERENCE_NODE\", \"REDUCE_OP\":\"\", \"Value\":\"CD\", "
         "\"Counter_Set\":[], \"Reduce_Dimension_Set\":[], "
         "\"Select_Dimension_Set\":[]},{\"Type\":\"REFERENCE_NODE\", \"REDUCE_OP\":\"\", "
         "\"Value\":\"ZX\", \"Counter_Set\":[], \"Reduce_Dimension_Set\":[], "
         "\"Select_Dimension_Set\":[]}], \"Reduce_Dimension_Set\":[], "
         "\"Select_Dimension_Set\":[]}"},
        {"NM / DB",
         "{\"Type\":\"DIVIDE_NODE\", \"REDUCE_OP\":\"\", "
         "\"Counter_Set\":[{\"Type\":\"REFERENCE_NODE\", \"REDUCE_OP\":\"\", \"Value\":\"NM\", "
         "\"Counter_Set\":[], \"Reduce_Dimension_Set\":[], "
         "\"Select_Dimension_Set\":[]},{\"Type\":\"REFERENCE_NODE\", \"REDUCE_OP\":\"\", "
         "\"Value\":\"DB\", \"Counter_Set\":[], \"Reduce_Dimension_Set\":[], "
         "\"Select_Dimension_Set\":[]}], \"Reduce_Dimension_Set\":[], "
         "\"Select_Dimension_Set\":[]}"}};

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
         "{\"Type\":\"DIVIDE_NODE\", \"REDUCE_OP\":\"\", "
         "\"Counter_Set\":[{\"Type\":\"ADDITION_NODE\", \"REDUCE_OP\":\"\", "
         "\"Counter_Set\":[{\"Type\":\"REFERENCE_NODE\", \"REDUCE_OP\":\"\", \"Value\":\"AB\", "
         "\"Counter_Set\":[], \"Reduce_Dimension_Set\":[], "
         "\"Select_Dimension_Set\":[]},{\"Type\":\"REFERENCE_NODE\", \"REDUCE_OP\":\"\", "
         "\"Value\":\"BA\", \"Counter_Set\":[], \"Reduce_Dimension_Set\":[], "
         "\"Select_Dimension_Set\":[]}], \"Reduce_Dimension_Set\":[], "
         "\"Select_Dimension_Set\":[]},{\"Type\":\"REFERENCE_NODE\", \"REDUCE_OP\":\"\", "
         "\"Value\":\"CD\", \"Counter_Set\":[], \"Reduce_Dimension_Set\":[], "
         "\"Select_Dimension_Set\":[]}], \"Reduce_Dimension_Set\":[], "
         "\"Select_Dimension_Set\":[]}"},
        {"(AB / BA) - BN",
         "{\"Type\":\"SUBTRACTION_NODE\", \"REDUCE_OP\":\"\", "
         "\"Counter_Set\":[{\"Type\":\"DIVIDE_NODE\", \"REDUCE_OP\":\"\", "
         "\"Counter_Set\":[{\"Type\":\"REFERENCE_NODE\", \"REDUCE_OP\":\"\", \"Value\":\"AB\", "
         "\"Counter_Set\":[], \"Reduce_Dimension_Set\":[], "
         "\"Select_Dimension_Set\":[]},{\"Type\":\"REFERENCE_NODE\", \"REDUCE_OP\":\"\", "
         "\"Value\":\"BA\", \"Counter_Set\":[], \"Reduce_Dimension_Set\":[], "
         "\"Select_Dimension_Set\":[]}], \"Reduce_Dimension_Set\":[], "
         "\"Select_Dimension_Set\":[]},{\"Type\":\"REFERENCE_NODE\", \"REDUCE_OP\":\"\", "
         "\"Value\":\"BN\", \"Counter_Set\":[], \"Reduce_Dimension_Set\":[], "
         "\"Select_Dimension_Set\":[]}], \"Reduce_Dimension_Set\":[], "
         "\"Select_Dimension_Set\":[]}"},
        {"AD / (CD - ZX)",
         "{\"Type\":\"DIVIDE_NODE\", \"REDUCE_OP\":\"\", "
         "\"Counter_Set\":[{\"Type\":\"REFERENCE_NODE\", \"REDUCE_OP\":\"\", \"Value\":\"AD\", "
         "\"Counter_Set\":[], \"Reduce_Dimension_Set\":[], "
         "\"Select_Dimension_Set\":[]},{\"Type\":\"SUBTRACTION_NODE\", \"REDUCE_OP\":\"\", "
         "\"Counter_Set\":[{\"Type\":\"REFERENCE_NODE\", \"REDUCE_OP\":\"\", \"Value\":\"CD\", "
         "\"Counter_Set\":[], \"Reduce_Dimension_Set\":[], "
         "\"Select_Dimension_Set\":[]},{\"Type\":\"REFERENCE_NODE\", \"REDUCE_OP\":\"\", "
         "\"Value\":\"ZX\", \"Counter_Set\":[], \"Reduce_Dimension_Set\":[], "
         "\"Select_Dimension_Set\":[]}], \"Reduce_Dimension_Set\":[], "
         "\"Select_Dimension_Set\":[]}], \"Reduce_Dimension_Set\":[], "
         "\"Select_Dimension_Set\":[]}"},
        {"MN * (NM / DB)",
         "{\"Type\":\"MULTIPLY_NODE\", \"REDUCE_OP\":\"\", "
         "\"Counter_Set\":[{\"Type\":\"REFERENCE_NODE\", \"REDUCE_OP\":\"\", \"Value\":\"MN\", "
         "\"Counter_Set\":[], \"Reduce_Dimension_Set\":[], "
         "\"Select_Dimension_Set\":[]},{\"Type\":\"DIVIDE_NODE\", \"REDUCE_OP\":\"\", "
         "\"Counter_Set\":[{\"Type\":\"REFERENCE_NODE\", \"REDUCE_OP\":\"\", \"Value\":\"NM\", "
         "\"Counter_Set\":[], \"Reduce_Dimension_Set\":[], "
         "\"Select_Dimension_Set\":[]},{\"Type\":\"REFERENCE_NODE\", \"REDUCE_OP\":\"\", "
         "\"Value\":\"DB\", \"Counter_Set\":[], \"Reduce_Dimension_Set\":[], "
         "\"Select_Dimension_Set\":[]}], \"Reduce_Dimension_Set\":[], "
         "\"Select_Dimension_Set\":[]}], \"Reduce_Dimension_Set\":[], "
         "\"Select_Dimension_Set\":[]}"}};

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
    std::vector<std::tuple<std::string, std::string>> expressionToExpected = {
        {"reduce(AB, SUM, [DIMENSION_XCC,DIMENSION_SHADER_ENGINE])",
         "{\"Type\":\"REDUCE_NODE\", \"REDUCE_OP\":\"SUM\", "
         "\"Counter_Set\":[{\"Type\":\"REFERENCE_NODE\", \"REDUCE_OP\":\"\", \"Value\":\"AB\", "
         "\"Counter_Set\":[], \"Reduce_Dimension_Set\":[], \"Select_Dimension_Set\":[]}], "
         "\"Reduce_Dimension_Set\":[\"2\",\"1\"], \"Select_Dimension_Set\":[]}"},
        {"reduce(AB+CD, SUM, [DIMENSION_XCC,DIMENSION_SHADER_ENGINE])",
         "{\"Type\":\"REDUCE_NODE\", \"REDUCE_OP\":\"SUM\", "
         "\"Counter_Set\":[{\"Type\":\"ADDITION_NODE\", \"REDUCE_OP\":\"\", "
         "\"Counter_Set\":[{\"Type\":\"REFERENCE_NODE\", \"REDUCE_OP\":\"\", \"Value\":\"AB\", "
         "\"Counter_Set\":[], \"Reduce_Dimension_Set\":[], "
         "\"Select_Dimension_Set\":[]},{\"Type\":\"REFERENCE_NODE\", \"REDUCE_OP\":\"\", "
         "\"Value\":\"CD\", \"Counter_Set\":[], \"Reduce_Dimension_Set\":[], "
         "\"Select_Dimension_Set\":[]}], \"Reduce_Dimension_Set\":[], "
         "\"Select_Dimension_Set\":[]}], \"Reduce_Dimension_Set\":[\"2\",\"1\"], "
         "\"Select_Dimension_Set\":[]}"},
        {"reduce(AB,DIV, [DIMENSION_XCC,DIMENSION_SHADER_ENGINE])+reduce(DC,SUM, "
         "[DIMENSION_XCC,DIMENSION_SHADER_ENGINE])",
         "{\"Type\":\"ADDITION_NODE\", \"REDUCE_OP\":\"\", "
         "\"Counter_Set\":[{\"Type\":\"REDUCE_NODE\", \"REDUCE_OP\":\"DIV\", "
         "\"Counter_Set\":[{\"Type\":\"REFERENCE_NODE\", \"REDUCE_OP\":\"\", \"Value\":\"AB\", "
         "\"Counter_Set\":[], \"Reduce_Dimension_Set\":[], \"Select_Dimension_Set\":[]}], "
         "\"Reduce_Dimension_Set\":[\"2\",\"1\"], "
         "\"Select_Dimension_Set\":[]},{\"Type\":\"REDUCE_NODE\", \"REDUCE_OP\":\"SUM\", "
         "\"Counter_Set\":[{\"Type\":\"REFERENCE_NODE\", \"REDUCE_OP\":\"\", \"Value\":\"DC\", "
         "\"Counter_Set\":[], \"Reduce_Dimension_Set\":[], \"Select_Dimension_Set\":[]}], "
         "\"Reduce_Dimension_Set\":[\"2\",\"1\"], \"Select_Dimension_Set\":[]}], "
         "\"Reduce_Dimension_Set\":[], \"Select_Dimension_Set\":[]}"}};

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

TEST(parser, DISABLED_selection)
{
    std::map<std::string, std::string> expressionToExpected = {
        {"select(AB, [SE=1,XCC=0])+select(DC,[SE=2])",
         "{\"Type\":\"ADDITION_NODE\", \"REDUCE_OP\":\"\", "
         "\"Counter_Set\":[{\"Type\":\"SELECT_NODE\", \"REDUCE_OP\":\"\", "
         "\"Counter_Set\":[{\"Type\":\"REFERENCE_NODE\", \"REDUCE_OP\":\"\", \"Value\":\"AB\", "
         "\"Counter_Set\":[], \"Reduce_Dimension_Set\":[], \"Select_Dimension_Set\":[]}], "
         "\"Reduce_Dimension_Set\":[], \"Select_Dimension_Set\":[\"(\"XCC\", 0)\",\"(\"SE\", "
         "1)\"]},{\"Type\":\"SELECT_NODE\", \"REDUCE_OP\":\"\", "
         "\"Counter_Set\":[{\"Type\":\"REFERENCE_NODE\", \"REDUCE_OP\":\"\", \"Value\":\"DC\", "
         "\"Counter_Set\":[], \"Reduce_Dimension_Set\":[], \"Select_Dimension_Set\":[]}], "
         "\"Reduce_Dimension_Set\":[], \"Select_Dimension_Set\":[\"(\"SE\", 2)\"]}], "
         "\"Reduce_Dimension_Set\":[], \"Select_Dimension_Set\":[]}"},
        {"select(AB, [SE=2,XCC=1,WGP=3])",
         "{\"Type\":\"SELECT_NODE\", \"REDUCE_OP\":\"\", "
         "\"Counter_Set\":[{\"Type\":\"REFERENCE_NODE\", \"REDUCE_OP\":\"\", \"Value\":\"AB\", "
         "\"Counter_Set\":[], \"Reduce_Dimension_Set\":[], \"Select_Dimension_Set\":[]}], "
         "\"Reduce_Dimension_Set\":[], \"Select_Dimension_Set\":[\"(\"WGP\", 3)\",\"(\"XCC\", "
         "1)\",\"(\"SE\", 2)\"]}"},
        {"select(AB, [XCC=0])",
         "{\"Type\":\"SELECT_NODE\", \"REDUCE_OP\":\"\", "
         "\"Counter_Set\":[{\"Type\":\"REFERENCE_NODE\", \"REDUCE_OP\":\"\", \"Value\":\"AB\", "
         "\"Counter_Set\":[], \"Reduce_Dimension_Set\":[], \"Select_Dimension_Set\":[]}], "
         "\"Reduce_Dimension_Set\":[], \"Select_Dimension_Set\":[\"(\"XCC\", 0)\"]}"}};

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

// TEST(parser, parse_complex_counters)
// {
//     std::map<std::string, std::string> expressionToExpected = {
//         {"(TCC_EA_WRREQ_sum-TCC_EA_WRREQ_64B_sum)+(TCC_EA1_WRREQ_sum-TCC_EA1_WRREQ_64B_sum)+(TCC_EA_WRREQ_64B_sum+TCC_EA1_WRREQ_64B_sum)*2",""}
//     };

//     for(auto [op, expected] : expressionToExpected)
//     {
//         RawAST* ast = nullptr;
//         auto*   buf = yy_scan_string(op.c_str());
//         yyparse(&ast);
//         ASSERT_TRUE(ast);
//         EXPECT_EQ(fmt::format("{}", *ast), expected);
//         yy_delete_buffer(buf);
//         delete ast;
//     }
// }
