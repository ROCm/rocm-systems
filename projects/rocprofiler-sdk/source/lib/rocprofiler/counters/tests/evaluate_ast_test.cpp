#include <tuple>

#include <gtest/gtest.h>

#include "lib/rocprofiler/counters/evaluate_ast.hpp"
#include "lib/rocprofiler/counters/parser/reader.hpp"

namespace
{
bool
isIdentical(const EvaluateAST& eval_ast, const RawAST& raw_ast)
{
    if(raw_ast.counter_set.size() != eval_ast.children().size() ||
       raw_ast.type != eval_ast.type() || raw_ast.operation != eval_ast.op())
    {
        return false;
    }

    for(size_t i = 0; i < raw_ast.counter_set.size(); i++)
    {
        if(!isIdentical(eval_ast.children()[i], *raw_ast.counter_set[i]))
        {
            return false;
        }
    }
    return true;
}
}  // namespace

TEST(evaluate_ast, basic_copy)
{
    std::unordered_map<std::string, Metric> metrics = {
        {"SQ_WAVES", Metric("a", "a", "a", "a", "a", "", 0)},
        {"TCC_HIT", Metric("b", "b", "b", "b", "b", "", 1)}};

    RawAST* ast = nullptr;
    auto*   buf = yy_scan_string("SQ_WAVES + TCC_HIT");
    yyparse(&ast);
    ASSERT_TRUE(ast);

    auto eval_ast = EvaluateAST(metrics, *ast);

    EXPECT_TRUE(isIdentical(eval_ast, *ast));
    yy_delete_buffer(buf);
    delete ast;
}

TEST(evaluate_ast, counter_expansion)
{
    std::unordered_map<std::string, Metric> metrics = {
        {"SQ_WAVES", Metric("SQ_WAVES", "a", "a", "a", "", "", 0)},
        {"TCC_HIT", Metric("TCC_HIT", "b", "b", "b", "", "", 1)},
        {"TEST_DERRIVED", Metric("TEST_DERRIVED", "C", "C", "C", "SQ_WAVES+TCC_HIT", "", 2)}};

    std::unordered_map<std::string, EvaluateAST> asts;
    for(auto [val, metric] : metrics)
    {
        RawAST* ast = nullptr;
        auto    buf = yy_scan_string(metric.expression().empty() ? metric.name().c_str()
                                                                 : metric.expression().c_str());
        yyparse(&ast);
        ASSERT_TRUE(ast);
        asts.emplace(val, std::move(EvaluateAST(metrics, *ast)));
        yy_delete_buffer(buf);
        delete ast;
    }

    std::set<Metric> required_counters;
    asts.at("TEST_DERRIVED").get_required_counters(asts, required_counters);
    EXPECT_EQ(required_counters.size(), 2);
    auto expected = std::set<Metric>{{Metric("TCC_HIT", "b", "b", "b", "", "", 1),
                                      Metric("SQ_WAVES", "a", "a", "a", "", "", 0)}};

    for(auto& counter_found : required_counters)
    {
        EXPECT_NE(expected.find(counter_found), expected.end());
    }
}

TEST(evaluate_ast, counter_expansion_multi_derived)
{
    std::unordered_map<std::string, Metric> metrics = {
        {"SQ_WAVES", Metric("SQ_WAVES", "a", "a", "a", "", "", 0)},
        {"TCC_HIT", Metric("TCC_HIT", "b", "b", "b", "", "", 1)},
        {"TEST_DERRIVED", Metric("TEST_DERRIVED", "C", "C", "C", "SQ_WAVES+TCC_HIT", "", 2)},
        {"TEST_DERRIVED3",
         Metric("TEST_DERRIVED3", "C", "C", "C", "TEST_DERRIVED+SQ_WAVES+TCC_HIT", "", 3)}};

    std::unordered_map<std::string, EvaluateAST> asts;
    for(auto [val, metric] : metrics)
    {
        RawAST* ast = nullptr;
        auto    buf = yy_scan_string(metric.expression().empty() ? metric.name().c_str()
                                                                 : metric.expression().c_str());
        yyparse(&ast);
        ASSERT_TRUE(ast);
        asts.emplace(val, std::move(EvaluateAST(metrics, *ast)));
        yy_delete_buffer(buf);
        delete ast;
    }

    std::set<Metric> required_counters;
    asts.at("TEST_DERRIVED3").get_required_counters(asts, required_counters);
    EXPECT_EQ(required_counters.size(), 2);
    auto expected = std::set<Metric>{{Metric("TCC_HIT", "b", "b", "b", "", "", 1),
                                      Metric("SQ_WAVES", "a", "a", "a", "", "", 0)}};

    for(auto& counter_found : required_counters)
    {
        EXPECT_NE(expected.find(counter_found), expected.end());
    }
}

TEST(evaluate_ast, counter_expansion_order)
{
    std::unordered_map<std::string, Metric> metrics = {
        {"SQ_WAVES", Metric("SQ_WAVES", "a", "a", "a", "", "", 0)},
        {"TCC_HIT", Metric("TCC_HIT", "b", "b", "b", "", "", 1)},
        {"VLL", Metric("VLL", "b", "b", "b", "", "", 4)},
        {"TEST_DERRIVED", Metric("TEST_DERRIVED", "C", "C", "C", "SQ_WAVES+VLL", "", 2)},
        {"TEST_DERRIVED3",
         Metric("TEST_DERRIVED3", "C", "C", "C", "TEST_DERRIVED+SQ_WAVES+TCC_HIT", "", 3)}};

    std::unordered_map<std::string, EvaluateAST> asts;
    for(auto [val, metric] : metrics)
    {
        RawAST* ast = nullptr;
        auto    buf = yy_scan_string(metric.expression().empty() ? metric.name().c_str()
                                                                 : metric.expression().c_str());
        yyparse(&ast);
        ASSERT_TRUE(ast);
        asts.emplace(val, std::move(EvaluateAST(metrics, *ast)));
        yy_delete_buffer(buf);
        delete ast;
    }

    std::set<Metric> required_counters;
    asts.at("TEST_DERRIVED3").get_required_counters(asts, required_counters);
    EXPECT_EQ(required_counters.size(), 3);
    auto expected = std::set<Metric>{{Metric("VLL", "b", "b", "b", "", "", 4),
                                      Metric("TCC_HIT", "b", "b", "b", "", "", 1),
                                      Metric("SQ_WAVES", "a", "a", "a", "", "", 0)}};

    for(auto& counter_found : required_counters)
    {
        EXPECT_NE(expected.find(counter_found), expected.end());
    }
}

TEST(evaluate_ast, counter_expansion_function)
{
    std::unordered_map<std::string, Metric> metrics = {
        {"SQ_WAVES", Metric("SQ_WAVES", "a", "a", "a", "", "", 0)},
        {"TCC_HIT", Metric("TCC_HIT", "b", "b", "b", "", "", 1)},
        {"VLL", Metric("VLL", "b", "b", "b", "", "", 4)},
        {"TEST_DERRIVED", Metric("TEST_DERRIVED", "C", "C", "C", "SQ_WAVES+VLL", "", 2)},
        {"TEST_DERRIVED3",
         Metric("TEST_DERRIVED3",
                "C",
                "C",
                "C",
                "reduce(TEST_DERRIVED,max)+SQ_WAVES+TCC_HIT",
                "",
                3)}};

    std::unordered_map<std::string, EvaluateAST> asts;
    for(auto [val, metric] : metrics)
    {
        RawAST* ast = nullptr;
        auto    buf = yy_scan_string(metric.expression().empty() ? metric.name().c_str()
                                                                 : metric.expression().c_str());
        yyparse(&ast);
        ASSERT_TRUE(ast);
        asts.emplace(val, std::move(EvaluateAST(metrics, *ast)));
        yy_delete_buffer(buf);
        delete ast;
    }

    std::set<Metric> required_counters;
    asts.at("TEST_DERRIVED3").get_required_counters(asts, required_counters);
    EXPECT_EQ(required_counters.size(), 3);
    auto expected = std::set<Metric>{{Metric("VLL", "b", "b", "b", "", "", 4),
                                      Metric("TCC_HIT", "b", "b", "b", "", "", 1),
                                      Metric("SQ_WAVES", "a", "a", "a", "", "", 0)}};

    for(auto& counter_found : required_counters)
    {
        EXPECT_NE(expected.find(counter_found), expected.end());
    }
}
