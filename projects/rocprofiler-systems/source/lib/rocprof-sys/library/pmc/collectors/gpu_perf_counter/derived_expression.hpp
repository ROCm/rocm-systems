// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

#include "library/pmc/collectors/gpu_perf_counter/counter_definitions.hpp"
#include "library/pmc/collectors/gpu_perf_counter/expression_parser.hpp"
#include "library/pmc/collectors/gpu_perf_counter/expression_tree.hpp"

#include <span>
#include <string>
#include <string_view>
#include <vector>

namespace rocprofsys::pmc::collectors::gpu_perf_counter
{

/**
 * @brief A parsed rocprofiler-sdk derived counter expression.
 *
 * Grammar: numbers, counter names, `+ - * /`, parentheses and
 * `reduce(<counter>, sum|max|min|avr)`. A name of another derived counter is expanded
 * to its expression.
 */
class derived_expression
{
public:
    /**
     * @brief Parses @p expression and expands derived names through @p definitions.
     * @throws std::invalid_argument on a syntax error, an unknown name, a cyclic
     *         definition, `accumulate(...)`, or an interval expression that reads a
     *         counter outside `reduce()`.
     */
    derived_expression(std::string_view           expression,
                       const counter_definitions& definitions)
    : m_parsed{ detail::expression_parser{ definitions, expression }.parse() }
    {}

    /// @brief How a sample value of this expression is obtained.
    [[nodiscard]] derived_evaluation evaluation() const noexcept
    {
        return m_parsed.evaluation;
    }

    /// @brief Raw and constant counters the expression reads, in input index order.
    [[nodiscard]] const std::vector<std::string>& input_names() const noexcept
    {
        return m_parsed.input_names;
    }

    /// @brief Every counter name the expression refers to, including expanded ones.
    [[nodiscard]] const std::vector<std::string>& referenced_names() const noexcept
    {
        return m_parsed.referenced_names;
    }

    /**
     * @brief Evaluates the expression.
     * @param inputs One reduction per input_names() entry, in the same order.
     * @return The value. A division by zero yields 0.
     */
    [[nodiscard]] double evaluate(std::span<const input_reduction> inputs) const
    {
        return m_parsed.tree.evaluate(inputs);
    }

private:
    parsed_expression m_parsed;
};

}  // namespace rocprofsys::pmc::collectors::gpu_perf_counter
