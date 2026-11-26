// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "library/pmc/collectors/gpu_perf_counter/counter_definitions.hpp"
#include "library/pmc/collectors/gpu_perf_counter/derived_expression.hpp"
#include "library/pmc/collectors/gpu_perf_counter/expression_tree.hpp"
#include "library/pmc/collectors/gpu_perf_counter/types.hpp"

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <array>
#include <initializer_list>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace rocprofsys::pmc::collectors::gpu_perf_counter::testing
{
namespace
{
// Definitions copied from rocprofiler-sdk share/rocprofiler-sdk/config.yaml (gfx942).
constexpr std::string_view k_mfma_util =
    "reduce(SQ_VALU_MFMA_BUSY_CYCLES,sum)/(reduce(GRBM_GUI_ACTIVE,max)*SIMD_NUM)*100";
constexpr std::string_view k_fetch_size =
    "(TCC_EA_RDREQ_32B_sum*32+(TCC_EA_RDREQ_sum-TCC_EA_RDREQ_32B_sum)*64)/1024";

counter_metadata
raw(std::string name)
{
    auto meta = counter_metadata{};
    meta.name = std::move(name);
    return meta;
}

counter_metadata
constant(std::string name)
{
    auto meta        = raw(std::move(name));
    meta.is_constant = true;
    return meta;
}

counter_metadata
derived(std::string name, std::string expression)
{
    auto meta       = raw(std::move(name));
    meta.expression = std::move(expression);
    meta.is_derived = true;
    return meta;
}

counter_definitions
sdk_definitions()
{
    auto const counters = std::vector<counter_metadata>{
        raw("SQ_VALU_MFMA_BUSY_CYCLES"),
        raw("GRBM_GUI_ACTIVE"),
        raw("GRBM_COUNT"),
        raw("TCC_EA_RDREQ"),
        raw("TCC_EA_RDREQ_32B"),
        raw("SQ_INSTS_VALU_MFMA_MOPS_F16"),
        raw("SQ_LEVEL_WAVES"),
        raw("SQ_BUSY_CU_CYCLES"),
        constant("simd_count"),
        constant("simd_per_cu"),
        derived("SIMD_NUM", "simd_count"),
        derived("CU_NUM", "simd_count/simd_per_cu"),
        derived("TCC_EA_RDREQ_sum", "reduce(TCC_EA_RDREQ,sum)"),
        derived("TCC_EA_RDREQ_32B_sum", "reduce(TCC_EA_RDREQ_32B,sum)"),
        derived("LOOP_A", "LOOP_B+1"),
        derived("LOOP_B", "LOOP_A+1"),
    };
    return counter_definitions{ counters };
}

derived_expression
parse(std::string_view expression)
{
    auto slots = input_slots{};
    return expression_parser::parse(expression, sdk_definitions(), slots);
}

input_reduction
reduction_of(std::initializer_list<double> values)
{
    auto result = input_reduction{};
    for(const double value : values)
    {
        result.add(value);
    }
    return result;
}
}  // namespace

TEST(DerivedExpressionTest, InputReductionTracksSumMaxMinAndCount)
{
    auto const reduction = reduction_of({ 3.0, -1.0, 7.0 });

    EXPECT_DOUBLE_EQ(reduction.sum, 9.0);
    EXPECT_DOUBLE_EQ(reduction.max, 7.0);
    EXPECT_DOUBLE_EQ(reduction.min, -1.0);
    EXPECT_EQ(reduction.count, 3U);
}

TEST(DerivedExpressionTest, MfmaUtilIsEvaluatedOnItsInputs)
{
    constexpr double k_busy_xcc0   = 2000.0;
    constexpr double k_busy_xcc1   = 1480.0;
    constexpr double k_active_xcc0 = 1000.0;
    constexpr double k_active_xcc1 = 990.0;
    constexpr double k_simd_count  = 4.0;
    constexpr double k_expected    = 87.0;

    auto       slots = input_slots{};
    auto const expression =
        expression_parser::parse(k_mfma_util, sdk_definitions(), slots);

    ASSERT_EQ(expression.evaluation, derived_evaluation::interval);
    ASSERT_THAT(slots, ::testing::UnorderedElementsAre(
                           ::testing::Pair("SQ_VALU_MFMA_BUSY_CYCLES", 0),
                           ::testing::Pair("GRBM_GUI_ACTIVE", 1),
                           ::testing::Pair("simd_count", 2)));
    EXPECT_THAT(expression.referenced_names,
                ::testing::UnorderedElementsAre("SQ_VALU_MFMA_BUSY_CYCLES",
                                                "GRBM_GUI_ACTIVE", "SIMD_NUM",
                                                "simd_count"));

    auto const inputs = std::array{ reduction_of({ k_busy_xcc0, k_busy_xcc1 }),
                                    reduction_of({ k_active_xcc0, k_active_xcc1 }),
                                    reduction_of({ k_simd_count }) };
    EXPECT_DOUBLE_EQ(expression.evaluate(inputs), k_expected);
}

TEST(DerivedExpressionTest, OperatorsFollowPrecedenceAndReduceOperations)
{
    auto const expression = parse(
        "reduce(GRBM_COUNT,max) - reduce(GRBM_COUNT,min) * 2 + reduce(GRBM_COUNT,avr)");

    auto const inputs = std::array{ reduction_of({ 10.0, 4.0, 1.0 }) };
    EXPECT_DOUBLE_EQ(expression.evaluate(inputs), 10.0 - (1.0 * 2.0) + 5.0);
}

TEST(DerivedExpressionTest, DivisionByZeroIntervalIsZero)
{
    auto const expression = parse(k_mfma_util);

    auto const inputs =
        std::array{ reduction_of({ 0.0 }), reduction_of({ 0.0 }), reduction_of({ 4.0 }) };
    EXPECT_DOUBLE_EQ(expression.evaluate(inputs), 0.0);
}

TEST(DerivedExpressionTest, ClassifiesHowASampleIsObtained)
{
    struct evaluation_case
    {
        std::string_view   expression;
        derived_evaluation expected;
    };

    constexpr auto k_cases = std::array{
        evaluation_case{ .expression = "simd_count",
                         .expected   = derived_evaluation::constant },
        evaluation_case{ .expression = "simd_count/simd_per_cu",
                         .expected   = derived_evaluation::constant },
        evaluation_case{ .expression = "reduce(TCC_EA_RDREQ,sum)",
                         .expected   = derived_evaluation::linear },
        evaluation_case{ .expression = "reduce(TCC_EA_RDREQ,avr)",
                         .expected   = derived_evaluation::linear },
        evaluation_case{ .expression = k_fetch_size,
                         .expected   = derived_evaluation::linear },
        evaluation_case{ .expression = "SQ_INSTS_VALU_MFMA_MOPS_F16*512",
                         .expected   = derived_evaluation::linear },
        evaluation_case{ .expression = "100*reduce(GRBM_COUNT,sum)/CU_NUM",
                         .expected   = derived_evaluation::linear },
        evaluation_case{ .expression = k_mfma_util,
                         .expected   = derived_evaluation::interval },
        evaluation_case{ .expression = "reduce(GRBM_GUI_ACTIVE,max)",
                         .expected   = derived_evaluation::interval },
        evaluation_case{ .expression = "reduce(GRBM_GUI_ACTIVE,min)",
                         .expected   = derived_evaluation::interval },
        evaluation_case{ .expression = "reduce(GRBM_COUNT,sum)+1",
                         .expected   = derived_evaluation::interval },
        evaluation_case{ .expression = "TCC_EA_RDREQ_sum*TCC_EA_RDREQ_32B_sum",
                         .expected   = derived_evaluation::interval },
    };

    for(const auto& test_case : k_cases)
    {
        SCOPED_TRACE(test_case.expression);
        auto const expression = parse(test_case.expression);
        EXPECT_EQ(expression.evaluation, test_case.expected);
    }
}

TEST(DerivedExpressionTest, RejectsUnsupportedExpressionsNamingTheReason)
{
    struct rejected_case
    {
        std::string_view expression;
        std::string_view reason;
    };

    constexpr auto k_cases = std::array{
        rejected_case{ .expression = "reduce(accumulate(SQ_LEVEL_WAVES, HIGH_RES),sum)/"
                                     "reduce(SQ_BUSY_CU_CYCLES,sum)",
                       .reason     = "accumulate() is not supported" },
        rejected_case{ .expression = "SQ_INSTS_VALU_MFMA_MOPS_F16/reduce(GRBM_COUNT,max)",
                       .reason     = "reads a counter outside reduce()" },
        rejected_case{ .expression = "reduce(NO_SUCH_COUNTER,sum)",
                       .reason     = "unknown counter 'NO_SUCH_COUNTER'" },
        rejected_case{ .expression = "LOOP_A",
                       .reason     = "cyclic definition of 'LOOP_A'" },
        rejected_case{ .expression = "reduce(TCC_EA_RDREQ_sum,sum)",
                       .reason     = "reduce() of derived counter 'TCC_EA_RDREQ_sum'" },
        rejected_case{ .expression = "reduce(GRBM_COUNT,median)",
                       .reason     = "unknown reduce operation 'median'" },
        rejected_case{ .expression = "(reduce(GRBM_COUNT,sum)",
                       .reason     = "expected ')'" },
        rejected_case{ .expression = "reduce(GRBM_COUNT,sum) GRBM_COUNT",
                       .reason     = "unexpected 'GRBM_COUNT'" },
        rejected_case{ .expression = "1..2", .reason = "invalid number '1..2'" },
    };

    for(const auto& test_case : k_cases)
    {
        SCOPED_TRACE(test_case.expression);
        auto const parse_case = [&test_case] {
            [[maybe_unused]] auto const expression = parse(test_case.expression);
        };
        EXPECT_THAT(parse_case,
                    ::testing::ThrowsMessage<std::invalid_argument>(
                        ::testing::HasSubstr(std::string{ test_case.reason })));
    }
}

}  // namespace rocprofsys::pmc::collectors::gpu_perf_counter::testing
