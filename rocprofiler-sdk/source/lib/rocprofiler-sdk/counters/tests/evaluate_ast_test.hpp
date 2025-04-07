// MIT License
//
// Copyright (c) 2023-2025 Advanced Micro Devices, Inc. All rights reserved.
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

#pragma once

#include <string>
#include <vector>

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
