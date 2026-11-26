// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

#include <fmt/format.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <span>
#include <stdexcept>
#include <vector>

namespace rocprofsys::pmc::collectors::gpu_perf_counter
{

/**
 * @brief How a sample value of a derived counter is obtained from the SDK device
 *        counting service, whose values are totals since start_context.
 */
enum class derived_evaluation : std::uint8_t
{
    constant,  ///< Reads no counter: the SDK value is reported as is.
    linear,    ///< Homogeneous linear in its counters: the delta of the SDK value is
               ///< the expression evaluated on the interval.
    interval,  ///< Non-linear: evaluated by rocprof-sys on its inputs' interval values.
};

/**
 * @brief One sample's values of one input counter, reduced over its instances.
 */
struct input_reduction
{
    double sum{ 0.0 };
    double max{ 0.0 };
    double min{ 0.0 };
    size_t count{ 0 };

    /// @brief Adds the value of one instance.
    void add(double value) noexcept
    {
        max = count == 0 ? value : std::max(max, value);
        min = count == 0 ? value : std::min(min, value);
        sum += value;
        ++count;
    }
};

namespace detail
{

/// @brief How an input node reduces the instances of its counter.
enum class input_read : std::uint8_t
{
    sum,
    max,
    min,
    avr,
    value,  // a constant outside reduce(): every instance has the value
};

enum class node_kind : std::uint8_t
{
    number,
    input,
    add,
    subtract,
    multiply,
    divide,
};

/// @brief One node of an expression tree. Children are indices into the tree.
struct expression_node
{
    node_kind  kind{ node_kind::number };
    double     number{ 0.0 };
    size_t     input{ 0 };
    input_read read{ input_read::value };
    size_t     lhs{ 0 };
    size_t     rhs{ 0 };
};

/**
 * @brief A parsed arithmetic expression over input counters, as a flat node array.
 *
 * Knows nothing about syntax or counter names: it is built node by node and evaluated
 * against one input_reduction per input slot.
 */
class expression_tree
{
public:
    /// @brief Appends @p node and returns its index.
    size_t add(expression_node node)
    {
        m_nodes.push_back(node);
        return m_nodes.size() - 1;
    }

    /// @brief Makes the node at @p index the one evaluate() computes.
    void set_root(size_t index) noexcept { m_root = index; }

    /**
     * @brief Evaluates the tree.
     * @param inputs One reduction per input slot.
     * @return The value. A division by zero yields 0.
     */
    [[nodiscard]] double evaluate(std::span<const input_reduction> inputs) const
    {
        return evaluate_node(m_root, inputs);
    }

private:
    [[nodiscard]] static double read_input(const input_reduction& input, input_read read)
    {
        switch(read)
        {
            case input_read::sum: return input.sum;
            case input_read::max: return input.max;
            case input_read::min: return input.min;
            case input_read::avr:
                return input.count == 0 ? 0.0
                                        : input.sum / static_cast<double>(input.count);
            case input_read::value: return input.max;
            default:
                throw std::logic_error(
                    fmt::format("input_read {} not handled", static_cast<int>(read)));
        }
    }

    [[nodiscard]] double evaluate_node(size_t                           index,
                                       std::span<const input_reduction> inputs) const
    {
        const auto& current = m_nodes.at(index);
        switch(current.kind)
        {
            case node_kind::number: return current.number;
            case node_kind::input: return read_input(inputs[current.input], current.read);
            case node_kind::add:
                return evaluate_node(current.lhs, inputs) +
                       evaluate_node(current.rhs, inputs);
            case node_kind::subtract:
                return evaluate_node(current.lhs, inputs) -
                       evaluate_node(current.rhs, inputs);
            case node_kind::multiply:
                return evaluate_node(current.lhs, inputs) *
                       evaluate_node(current.rhs, inputs);
            case node_kind::divide:
            {
                // ponytail: an idle interval (e.g. no GRBM_GUI_ACTIVE cycles) reports 0
                // instead of inf/NaN; revisit if users need to tell idle from 0 %.
                auto const denominator = evaluate_node(current.rhs, inputs);
                return denominator == 0.0
                           ? 0.0
                           : evaluate_node(current.lhs, inputs) / denominator;
            }
            default:
                throw std::logic_error(fmt::format("node_kind {} not handled",
                                                   static_cast<int>(current.kind)));
        }
    }

    std::vector<expression_node> m_nodes;
    size_t                       m_root{ 0 };
};

}  // namespace detail
}  // namespace rocprofsys::pmc::collectors::gpu_perf_counter
