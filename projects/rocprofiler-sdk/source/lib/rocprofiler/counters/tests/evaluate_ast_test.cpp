#include <cstdint>
#include <tuple>

#include <fmt/core.h>
#include <gtest/gtest.h>

#include "evaluate_ast_test.hpp"
#include "lib/rocprofiler/counters/parser/reader.hpp"

namespace
{
ReduceOperation
get_reduce_op_type_from_string(const std::string& op)
{
    static const std::unordered_map<std::string, ReduceOperation> reduce_op_string_to_type = {
        {"min", REDUCE_MIN}, {"max", REDUCE_MAX}, {"sum", REDUCE_SUM}, {"avr", REDUCE_AVG}};

    ReduceOperation type           = REDUCE_NONE;
    const auto*     reduce_op_type = rocprofiler::common::get_val(reduce_op_string_to_type, op);
    if(reduce_op_type) type = *reduce_op_type;
    return type;
}

bool
isIdentical(const EvaluateAST& eval_ast, const RawAST& raw_ast)
{
    if(raw_ast.counter_set.size() != eval_ast.children().size() ||
       raw_ast.type != eval_ast.type() ||
       get_reduce_op_type_from_string(raw_ast.reduce_op) != eval_ast.reduce_op())
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
        {"SQ_WAVES", Metric("gfx9", "a", "a", "a", "a", "a", "", 0)},
        {"TCC_HIT", Metric("gfx9", "b", "b", "b", "b", "b", "", 1)}};

    RawAST* ast = nullptr;
    auto*   buf = yy_scan_string("SQ_WAVES + TCC_HIT");
    yyparse(&ast);
    ASSERT_TRUE(ast);

    auto eval_ast = EvaluateAST(metrics, *ast, "gfx9");

    EXPECT_TRUE(isIdentical(eval_ast, *ast));
    yy_delete_buffer(buf);
    delete ast;
}

TEST(evaluate_ast, counter_expansion)
{
    std::unordered_map<std::string, Metric> metrics = {
        {"SQ_WAVES", Metric("gfx9", "SQ_WAVES", "a", "a", "a", "", "", 0)},
        {"TCC_HIT", Metric("gfx9", "TCC_HIT", "b", "b", "b", "", "", 1)},
        {"TEST_DERRIVED",
         Metric("gfx9", "TEST_DERRIVED", "C", "C", "C", "SQ_WAVES+TCC_HIT", "", 2)}};

    std::unordered_map<std::string, EvaluateAST> asts;
    for(auto [val, metric] : metrics)
    {
        RawAST* ast = nullptr;
        auto    buf = yy_scan_string(metric.expression().empty() ? metric.name().c_str()
                                                                 : metric.expression().c_str());
        yyparse(&ast);
        ASSERT_TRUE(ast);
        asts.emplace(val, std::move(EvaluateAST(metrics, *ast, "gfx9")));
        yy_delete_buffer(buf);
        delete ast;
    }

    std::set<Metric> required_counters;
    asts.at("TEST_DERRIVED").get_required_counters(asts, required_counters);
    EXPECT_EQ(required_counters.size(), 2);
    auto expected = std::set<Metric>{{Metric("gfx9", "TCC_HIT", "b", "b", "b", "", "", 1),
                                      Metric("gfx9", "SQ_WAVES", "a", "a", "a", "", "", 0)}};

    for(auto& counter_found : required_counters)
    {
        EXPECT_NE(expected.find(counter_found), expected.end());
    }
}

TEST(evaluate_ast, counter_expansion_multi_derived)
{
    std::unordered_map<std::string, Metric> metrics = {
        {"SQ_WAVES", Metric("gfx9", "SQ_WAVES", "a", "a", "a", "", "", 0)},
        {"TCC_HIT", Metric("gfx9", "TCC_HIT", "b", "b", "b", "", "", 1)},
        {"TEST_DERRIVED",
         Metric("gfx9", "TEST_DERRIVED", "C", "C", "C", "SQ_WAVES+TCC_HIT", "", 2)},
        {"TEST_DERRIVED3",
         Metric("gfx9", "TEST_DERRIVED3", "C", "C", "C", "TEST_DERRIVED+SQ_WAVES+TCC_HIT", "", 3)}};

    std::unordered_map<std::string, EvaluateAST> asts;
    for(auto [val, metric] : metrics)
    {
        RawAST* ast = nullptr;
        auto    buf = yy_scan_string(metric.expression().empty() ? metric.name().c_str()
                                                                 : metric.expression().c_str());
        yyparse(&ast);
        ASSERT_TRUE(ast);
        asts.emplace(val, std::move(EvaluateAST(metrics, *ast, "gfx9")));
        yy_delete_buffer(buf);
        delete ast;
    }

    std::set<Metric> required_counters;
    asts.at("TEST_DERRIVED3").get_required_counters(asts, required_counters);
    EXPECT_EQ(required_counters.size(), 2);
    auto expected = std::set<Metric>{{Metric("gfx9", "TCC_HIT", "b", "b", "b", "", "", 1),
                                      Metric("gfx9", "SQ_WAVES", "a", "a", "a", "", "", 0)}};

    for(auto& counter_found : required_counters)
    {
        EXPECT_NE(expected.find(counter_found), expected.end());
    }
}

TEST(evaluate_ast, counter_expansion_order)
{
    std::unordered_map<std::string, Metric> metrics = {
        {"SQ_WAVES", Metric("gfx9", "SQ_WAVES", "a", "a", "a", "", "", 0)},
        {"TCC_HIT", Metric("gfx9", "TCC_HIT", "b", "b", "b", "", "", 1)},
        {"VLL", Metric("gfx9", "VLL", "b", "b", "b", "", "", 4)},
        {"TEST_DERRIVED", Metric("gfx9", "TEST_DERRIVED", "C", "C", "C", "SQ_WAVES+VLL", "", 2)},
        {"TEST_DERRIVED3",
         Metric("gfx9", "TEST_DERRIVED3", "C", "C", "C", "TEST_DERRIVED+SQ_WAVES+TCC_HIT", "", 3)}};

    std::unordered_map<std::string, EvaluateAST> asts;
    for(auto [val, metric] : metrics)
    {
        RawAST* ast = nullptr;
        auto    buf = yy_scan_string(metric.expression().empty() ? metric.name().c_str()
                                                                 : metric.expression().c_str());
        yyparse(&ast);
        ASSERT_TRUE(ast);
        asts.emplace(val, std::move(EvaluateAST(metrics, *ast, "gfx9")));
        yy_delete_buffer(buf);
        delete ast;
    }

    std::set<Metric> required_counters;
    asts.at("TEST_DERRIVED3").get_required_counters(asts, required_counters);
    EXPECT_EQ(required_counters.size(), 3);
    auto expected = std::set<Metric>{{Metric("gfx9", "VLL", "b", "b", "b", "", "", 4),
                                      Metric("gfx9", "TCC_HIT", "b", "b", "b", "", "", 1),
                                      Metric("gfx9", "SQ_WAVES", "a", "a", "a", "", "", 0)}};

    for(auto& counter_found : required_counters)
    {
        EXPECT_NE(expected.find(counter_found), expected.end());
    }
}

// TEST(evaluate_ast, counter_expansion_function)
// {
//     std::unordered_map<std::string, Metric> metrics = {
//         {"SQ_WAVES", Metric("gfx9", "SQ_WAVES", "a", "a", "a", "", "", 0)},
//         {"TCC_HIT", Metric("gfx9", "TCC_HIT", "b", "b", "b", "", "", 1)},
//         {"VLL", Metric("gfx9", "VLL", "b", "b", "b", "", "", 4)},
//         {"TEST_DERRIVED", Metric("gfx9", "TEST_DERRIVED", "C", "C", "C", "SQ_WAVES+VLL", "",
//         2)}};

//     std::unordered_map<std::string, EvaluateAST> asts;
//     for(auto [val, metric] : metrics)
//     {
//         RawAST* ast = nullptr;
//         auto    buf = yy_scan_string(metric.expression().empty() ? metric.name().c_str()
//                                                                  : metric.expression().c_str());
//         yyparse(&ast);
//         ASSERT_TRUE(ast) << metric.expression() <<  " " << metric.name();
//         asts.emplace(val, std::move(EvaluateAST(metrics, *ast, "gfx9")));
//         yy_delete_buffer(buf);
//         delete ast;
//     }
// }

// TEST(evaluate_ast, evaluate_simple_math)
// {
//     std::unordered_map<std::string, Metric>           metrics;
//     std::unordered_map<uint64_t, metric_result>       results_map;
//     std::unordered_map<uint64_t, std::vector<double>> expected_values;

//     uint64_t id = 0;
//     for(auto& data : test_data_evaluate_simple_math)
//     {
//         metrics.emplace(
//         data.name, Metric("gfx9", data.name, "Block", "0", "", data.expr, "", id));
//         metric_result res = {id, data.sample_values};
//         results_map.emplace(id, res);
//         expected_values.emplace(id, data.expected_values);
//         ++id;
//     }

//     std::unordered_map<std::string, EvaluateAST> asts;
//     for(auto [val, metric] : metrics)
//     {
//         RawAST* ast = nullptr;
//         auto    buf = yy_scan_string(metric.expression().empty() ? metric.name().c_str()
//                                                                  : metric.expression().c_str());
//         yyparse(&ast);
//         ASSERT_TRUE(ast);
//         asts.emplace(val, std::move(EvaluateAST(metrics, *ast, "gfx9")));
//         yy_delete_buffer(buf);
//         delete ast;
//     }

//     for(auto [metric_name, ast] : asts)
//     {
//         double   value     = ast.evaluate(results_map);
//         uint64_t metric_id = metrics.at(metric_name).id();
//         EXPECT_EQ(value, expected_values.at(metric_id)[0]);
//     }
// }

// TEST(evaluate_ast, evaluate_evaluate_simple_reduce)
// {
//     std::unordered_map<std::string, Metric>     metrics;
//     std::unordered_map<uint64_t, metric_result> results_map;
//     std::unordered_map<uint64_t, std::vector<double>> expected_values;

//     uint64_t id = 0;
//     for(auto& data : test_data_evaluate_simple_reduce)
//     {
//         metrics.emplace(
//             data.name, Metric("gfx9", data.name, "Block", "0", "", data.expr, "", id));
//         metric_result res = {id, data.sample_values};
//         results_map.emplace(id, res);
//         expected_values.emplace(id, data.expected_values);
//         ++id;
//     }

//     std::unordered_map<std::string, EvaluateAST> asts;
//     for(auto [val, metric] : metrics)
//     {
//         RawAST* ast = nullptr;
//         auto    buf = yy_scan_string(metric.expression().empty() ? metric.name().c_str()
//                                                               : metric.expression().c_str());
//         yyparse(&ast);
//         ASSERT_TRUE(ast);
//         asts.emplace(val, EvaluateAST(metrics, *ast, "gfx9"));
//         yy_delete_buffer(buf);
//         delete ast;
//     }

//     for(auto [metric_name, ast]: asts){
//         double value = ast.evaluate(results_map);
//         uint64_t metric_id = metrics.at(metric_name).id();
//         EXPECT_EQ(value, expected_values.at(metric_id)[0]);
//     }
// }
