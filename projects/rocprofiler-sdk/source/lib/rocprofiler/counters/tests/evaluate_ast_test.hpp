#pragma once

#include <string>
#include <unordered_map>
#include <vector>
#include "lib/rocprofiler/counters/evaluate_ast.hpp"

struct test_data
{
    std::string         name;
    std::string         expr;
    std::vector<int>    sample_values;
    std::vector<double> expected_values;
};

static const std::vector<test_data> test_data_evaluate_simple_math = {
    {"SQ_WAVES", "", {10, 20, 30}, {60}},
    {"TCC_HIT", "", {1, 2, 3, 4, 5}, {15}},
    {"SQ_INSTS_VALU", "", {2, 4, 6, 8}, {20}},

    /* Add/Subtract/Multiply/Divide */
    {"Metric_1", "SQ_WAVES+TCC_HIT", {}, {75}},
    {"Metric_2", "SQ_WAVES-TCC_HIT", {}, {45}},
    {"Metric_3", "SQ_WAVES*TCC_HIT", {}, {900}},
    {"Metric_4", "SQ_WAVES/TCC_HIT", {}, {4}},

    /* Order of Ops */
    {"Metric_5", "(SQ_WAVES+TCC_HIT)/SQ_INSTS_VALU", {}, {3.75}},
    {"Metric_6", "(SQ_WAVES/TCC_HIT)-SQ_INSTS_VALU", {}, {-16}},
    {"Metric_7", "SQ_WAVES/(TCC_HIT-SQ_INSTS_VALU)", {}, {-12}},
    {"Metric_8", "SQ_WAVES*(TCC_HIT/SQ_INSTS_VALU)", {}, {45}}};

static const std::vector<test_data> test_data_evaluate_simple_reduce = {
    {"SQ_WAVES", "", {10, 20, 30}, {60}},
    {"TCC_HIT", "", {1, 2, 3}, {6}},

    /* Simple reduce operations */
    {"Metric_1", "reduce(SQ_WAVES, sum)", {}, {60}},
    {"Metric_2", "reduce(SQ_WAVES, sum, [DIMENSION_XCC])", {}, {60}},
    {"Metric_3", "reduce(SQ_WAVES, min, [DIMENSION_XCC])", {}, {10}},
    {"Metric_4", "reduce(SQ_WAVES, max, [DIMENSION_XCC])", {}, {30}},
    {"Metric_5", "reduce(SQ_WAVES, avr, [DIMENSION_XCC])", {}, {20}},
    {"Metric_6", "reduce(SQ_WAVES, sum) + reduce(SQ_WAVES, avr)", {}, {80}},
    {"Metric_7", "reduce(SQ_WAVES, max) - reduce(TCC_HIT, min)", {}, {29}}};
