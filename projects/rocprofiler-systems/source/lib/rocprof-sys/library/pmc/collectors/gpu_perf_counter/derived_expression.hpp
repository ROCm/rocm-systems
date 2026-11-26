// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

#include "common/string_utility.hpp"
#include "library/pmc/collectors/gpu_perf_counter/counter_definitions.hpp"
#include "library/pmc/collectors/gpu_perf_counter/expression_scanner.hpp"
#include "library/pmc/collectors/gpu_perf_counter/expression_tree.hpp"
#include "library/pmc/collectors/gpu_perf_counter/types.hpp"

#include <fmt/format.h>

#include <algorithm>
#include <array>
#include <concepts>
#include <cstddef>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <unordered_map>
#include <utility>
#include <vector>

namespace rocprofsys::pmc::collectors::gpu_perf_counter
{

/// @brief Index of an input counter's reduction, by counter name.
using input_slots = std::unordered_map<std::string, size_t>;

/**
 * @brief A parsed rocprofiler-sdk derived counter expression.
 *
 * Grammar: numbers, counter names, `+ - * /`, parentheses and
 * `reduce(<counter>, sum|max|min|avr)`. A name of another derived counter is expanded
 * to its expression. Build it with expression_parser::parse().
 */
struct derived_expression
{
    detail::expression_tree  tree;
    derived_evaluation       evaluation{ derived_evaluation::constant };
    std::vector<std::string> referenced_names;  ///< Every counter name, incl. expanded.

    /**
     * @brief Evaluates the expression.
     * @param inputs One reduction per slot of the input_slots it was parsed with.
     * @return The value. A division by zero yields 0.
     */
    [[nodiscard]] double evaluate(std::span<const input_reduction> inputs) const
    {
        return tree.evaluate(inputs);
    }
};

/**
 * @brief Recursive descent parser: expression text to an expression_tree.
 *
 * A name of a derived counter is expanded to its expression. While parsing it
 * classifies how the value can be obtained (derived_evaluation).
 */
class expression_parser
{
public:
    /**
     * @brief Parses @p expression and expands derived names through @p definitions.
     *
     * Every counter the expression reads gets a slot in @p slots, added after the ones
     * already there, so expressions parsed with one map share their input slots.
     * @throws std::invalid_argument on a syntax error, an unknown name, a cyclic
     *         definition, `accumulate(...)`, or an interval expression that reads a
     *         counter outside `reduce()`.
     */
    [[nodiscard]] static derived_expression parse(std::string_view           expression,
                                                  const counter_definitions& definitions,
                                                  input_slots&               slots)
    {
        return expression_parser{ definitions, expression, slots }.parse_root();
    }

private:
    using expression_scanner = detail::expression_scanner;
    using node_kind          = detail::node_kind;
    using input_read         = detail::input_read;

    expression_parser(const counter_definitions& definitions, std::string_view expression,
                      input_slots& slots)
    : m_definitions{ definitions }
    , m_expression{ expression }
    , m_slots{ slots }
    {}

    [[nodiscard]] derived_expression parse_root()
    {
        auto       scanner = expression_scanner{ m_expression, m_expression };
        auto const root    = parse_complete(scanner);
        if(root.shape == derived_evaluation::interval && root.per_instance)
        {
            throw std::invalid_argument(fmt::format(
                "derived counter expression '{}': non-linear expression reads a "
                "counter outside reduce()",
                m_expression));
        }
        m_result.tree.set_root(root.index);
        m_result.evaluation = root.shape;
        return std::move(m_result);
    }

    // A parsed sub-expression: its node, how its sample value behaves, and whether it
    // reads a counter outside reduce() (one value per counter instance).
    struct subexpression
    {
        size_t             index{ 0 };
        derived_evaluation shape{ derived_evaluation::constant };
        bool               per_instance{ false };
    };

    struct binary_operator
    {
        char      symbol;
        node_kind kind;
    };

    static constexpr auto k_sum_operators =
        std::array{ binary_operator{ .symbol = '+', .kind = node_kind::add },
                    binary_operator{ .symbol = '-', .kind = node_kind::subtract } };
    static constexpr auto k_product_operators =
        std::array{ binary_operator{ .symbol = '*', .kind = node_kind::multiply },
                    binary_operator{ .symbol = '/', .kind = node_kind::divide } };

    [[nodiscard]] subexpression parse_complete(expression_scanner& scanner)
    {
        auto const result = parse_sum(scanner);
        scanner.expect_end();
        return result;
    }

    [[nodiscard]] subexpression parse_sum(expression_scanner& scanner)
    {
        return parse_chain(scanner, k_sum_operators,
                           [&] { return parse_product(scanner); });
    }

    [[nodiscard]] subexpression parse_product(expression_scanner& scanner)
    {
        return parse_chain(scanner, k_product_operators,
                           [&] { return parse_factor(scanner); });
    }

    // operand (operator operand)*, combined left to right.
    [[nodiscard]] subexpression parse_chain(expression_scanner&              scanner,
                                            std::span<const binary_operator> operators,
                                            std::invocable auto parse_operand)
    {
        auto result = parse_operand();
        while(true)
        {
            auto const next =
                std::ranges::find_if(operators, [&](binary_operator binary_op) {
                    return scanner.consume(binary_op.symbol);
                });
            if(next == operators.end())
            {
                return result;
            }
            result = add_binary(next->kind, result, parse_operand());
        }
    }

    [[nodiscard]] subexpression parse_factor(expression_scanner& scanner)
    {
        if(scanner.consume('('))
        {
            auto const inner = parse_sum(scanner);
            scanner.expect(')');
            return inner;
        }
        if(scanner.at_number())
        {
            return parse_number(scanner);
        }
        return parse_name(scanner);
    }

    [[nodiscard]] subexpression parse_number(expression_scanner& scanner)
    {
        auto const token  = scanner.read_number();
        auto const number = utility::string::safe_stod(token);
        if(number.is_error())
        {
            throw std::invalid_argument(
                fmt::format("derived counter expression '{}': invalid number '{}'",
                            m_expression, token));
        }
        return { .index = m_result.tree.add(
                     { .kind = node_kind::number, .number = number.value() }),
                 .shape = derived_evaluation::constant };
    }

    [[nodiscard]] subexpression parse_name(expression_scanner& scanner)
    {
        auto const name = scanner.read_identifier();
        reject_accumulate(name);
        if(name == "reduce")
        {
            return parse_reduce(scanner);
        }
        const auto& meta = lookup(name);
        if(meta.is_derived)
        {
            return expand_derived(meta);
        }
        return add_input(meta, input_read::value);
    }

    [[nodiscard]] subexpression parse_reduce(expression_scanner& scanner)
    {
        scanner.expect('(');
        auto const name = scanner.read_identifier();
        reject_accumulate(name);
        const auto& meta = lookup(name);
        if(meta.is_derived)
        {
            throw std::invalid_argument(fmt::format(
                "derived counter expression '{}': reduce() of derived counter '{}'",
                m_expression, name));
        }
        scanner.expect(',');
        auto const read = to_input_read(scanner.read_identifier());
        scanner.expect(')');
        return add_input(meta, read);
    }

    [[nodiscard]] subexpression expand_derived(const counter_metadata& meta)
    {
        if(std::ranges::find(m_expanding, meta.name) != m_expanding.end())
        {
            throw std::invalid_argument(
                fmt::format("derived counter expression '{}': cyclic definition of '{}'",
                            m_expression, meta.name));
        }
        m_expanding.push_back(meta.name);
        auto       scanner = expression_scanner{ meta.expression, m_expression };
        auto const result  = parse_complete(scanner);
        m_expanding.pop_back();
        return result;
    }

    [[nodiscard]] subexpression add_input(const counter_metadata& meta, input_read read)
    {
        auto const index = m_result.tree.add(
            { .kind = node_kind::input, .input = input_slot(meta.name), .read = read });
        if(meta.is_constant)
        {
            return { .index = index, .shape = derived_evaluation::constant };
        }
        auto const is_extremum = read == input_read::max || read == input_read::min;
        return { .index        = index,
                 .shape        = is_extremum ? derived_evaluation::interval
                                             : derived_evaluation::linear,
                 .per_instance = read == input_read::value };
    }

    [[nodiscard]] subexpression add_binary(node_kind kind, subexpression lhs,
                                           subexpression rhs)
    {
        return { .index = m_result.tree.add(
                     { .kind = kind, .lhs = lhs.index, .rhs = rhs.index }),
                 .shape        = combine_shapes(kind, lhs.shape, rhs.shape),
                 .per_instance = lhs.per_instance || rhs.per_instance };
    }

    // The delta of a sample equals the expression evaluated on the interval only when
    // the expression is homogeneous linear in its counters.
    [[nodiscard]] static derived_evaluation combine_shapes(node_kind          kind,
                                                           derived_evaluation lhs,
                                                           derived_evaluation rhs)
    {
        using enum derived_evaluation;
        if(lhs == interval || rhs == interval)
        {
            return interval;
        }
        switch(kind)
        {
            case node_kind::add:
            case node_kind::subtract: return lhs == rhs ? lhs : interval;
            case node_kind::multiply:
                if(lhs == constant)
                {
                    return rhs;
                }
                return rhs == constant ? lhs : interval;
            case node_kind::divide: return rhs == constant ? lhs : interval;
            default:
                throw std::logic_error(
                    fmt::format("node_kind {} not handled", static_cast<int>(kind)));
        }
    }

    [[nodiscard]] input_read to_input_read(std::string_view operation) const
    {
        using read_name               = std::pair<std::string_view, input_read>;
        static constexpr auto k_reads = std::array{ read_name{ "sum", input_read::sum },
                                                    read_name{ "max", input_read::max },
                                                    read_name{ "min", input_read::min },
                                                    read_name{ "avr", input_read::avr } };

        auto const itr = std::ranges::find(k_reads, operation, &read_name::first);
        if(itr == k_reads.end())
        {
            throw std::invalid_argument(fmt::format(
                "derived counter expression '{}': unknown reduce operation '{}'",
                m_expression, operation));
        }
        return itr->second;
    }

    // accumulate() selects an SQ counter collection mode that the SDK applies only
    // inside its own derived counter; it cannot be reproduced from the plain counter.
    void reject_accumulate(std::string_view name) const
    {
        if(name == "accumulate")
        {
            throw std::invalid_argument(
                fmt::format("derived counter expression '{}': accumulate() is not "
                            "supported: it is an SQ counter collection mode "
                            "the SDK applies only inside its own derived counter",
                            m_expression));
        }
    }

    [[nodiscard]] const counter_metadata& lookup(std::string_view name)
    {
        const auto* meta = m_definitions.find(name);
        if(meta == nullptr)
        {
            throw std::invalid_argument(
                fmt::format("derived counter expression '{}': unknown counter '{}'",
                            m_expression, name));
        }
        auto& referenced = m_result.referenced_names;
        if(std::ranges::find(referenced, meta->name) == referenced.end())
        {
            referenced.push_back(meta->name);
        }
        return *meta;
    }

    [[nodiscard]] size_t input_slot(const std::string& name)
    {
        return m_slots.try_emplace(name, m_slots.size()).first->second;
    }

    const counter_definitions& m_definitions;
    std::string_view           m_expression;
    input_slots&               m_slots;
    derived_expression         m_result;
    std::vector<std::string>   m_expanding;
};

}  // namespace rocprofsys::pmc::collectors::gpu_perf_counter
