// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

#include "common/string_utility.hpp"
#include "library/pmc/collectors/gpu_perf_counter/expression_scanner.hpp"
#include "library/pmc/collectors/gpu_perf_counter/expression_tree.hpp"
#include "library/pmc/collectors/gpu_perf_counter/types.hpp"

#include <fmt/format.h>

#include <algorithm>
#include <cstddef>
#include <iterator>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <unordered_map>
#include <utility>
#include <vector>

namespace rocprofsys::pmc::collectors::gpu_perf_counter
{

/// @brief Counter metadata by counter name; resolves the names used in expressions.
using counter_definitions = std::unordered_map<std::string, counter_metadata>;

/// @brief Indexes @p counters by name. Instances of one counter share a definition.
[[nodiscard]] inline counter_definitions
make_counter_definitions(std::span<const counter_metadata> counters)
{
    auto definitions = counter_definitions{};
    for(const auto& meta : counters)
    {
        definitions.try_emplace(meta.name, meta);
    }
    return definitions;
}

/// @brief What parsing an expression yields.
struct parsed_expression
{
    detail::expression_tree  tree;
    derived_evaluation       evaluation{ derived_evaluation::constant };
    std::vector<std::string> input_names;       ///< Counters the tree reads, by slot.
    std::vector<std::string> referenced_names;  ///< Every counter name, incl. expanded.
};

namespace detail
{

/**
 * @brief Recursive descent parser: expression text to an expression_tree.
 *
 * A name of a derived counter is expanded to its expression. While parsing it
 * classifies how the value can be obtained (derived_evaluation).
 */
class expression_parser
{
public:
    expression_parser(const counter_definitions& definitions, std::string_view expression)
    : m_definitions{ definitions }
    , m_expression{ expression }
    {}

    /// @brief Parses the expression; call once.
    [[nodiscard]] parsed_expression parse()
    {
        auto       scanner = expression_scanner{ m_expression, m_expression };
        auto const root    = parse_complete(scanner);
        if(root.shape == derived_evaluation::interval && root.per_instance)
        {
            fail_expression(m_expression,
                            "non-linear expression reads a counter outside reduce()");
        }
        m_result.tree.set_root(root.index);
        m_result.evaluation = root.shape;
        return std::move(m_result);
    }

private:
    // A parsed sub-expression: its node, how its sample value behaves, and whether it
    // reads a counter outside reduce() (one value per counter instance).
    struct subexpression
    {
        size_t             index{ 0 };
        derived_evaluation shape{ derived_evaluation::constant };
        bool               per_instance{ false };
    };

    [[nodiscard]] subexpression parse_complete(expression_scanner& scanner)
    {
        auto const result = parse_sum(scanner);
        scanner.expect_end();
        return result;
    }

    [[nodiscard]] subexpression parse_sum(expression_scanner& scanner)
    {
        auto result = parse_product(scanner);
        while(true)
        {
            if(scanner.consume('+'))
            {
                result = add_binary(node_kind::add, result, parse_product(scanner));
            }
            else if(scanner.consume('-'))
            {
                result = add_binary(node_kind::subtract, result, parse_product(scanner));
            }
            else
            {
                return result;
            }
        }
    }

    [[nodiscard]] subexpression parse_product(expression_scanner& scanner)
    {
        auto result = parse_factor(scanner);
        while(true)
        {
            if(scanner.consume('*'))
            {
                result = add_binary(node_kind::multiply, result, parse_factor(scanner));
            }
            else if(scanner.consume('/'))
            {
                result = add_binary(node_kind::divide, result, parse_factor(scanner));
            }
            else
            {
                return result;
            }
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
            scanner.fail(fmt::format("invalid number '{}'", token));
        }
        return { .index = m_result.tree.add(
                     { .kind = node_kind::number, .number = number.value() }),
                 .shape = derived_evaluation::constant };
    }

    [[nodiscard]] subexpression parse_name(expression_scanner& scanner)
    {
        auto const name = scanner.read_identifier();
        reject_accumulate(name, scanner);
        if(name == "reduce")
        {
            return parse_reduce(scanner);
        }
        const auto& meta = lookup(name, scanner);
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
        reject_accumulate(name, scanner);
        const auto& meta = lookup(name, scanner);
        if(meta.is_derived)
        {
            scanner.fail(fmt::format("reduce() of derived counter '{}'", name));
        }
        scanner.expect(',');
        auto const read = to_input_read(scanner.read_identifier(), scanner);
        scanner.expect(')');
        return add_input(meta, read);
    }

    [[nodiscard]] subexpression expand_derived(const counter_metadata& meta)
    {
        if(std::ranges::find(m_expanding, meta.name) != m_expanding.end())
        {
            fail_expression(m_expression,
                            fmt::format("cyclic definition of '{}'", meta.name));
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

    [[nodiscard]] static input_read to_input_read(std::string_view    operation,
                                                  expression_scanner& scanner)
    {
        if(operation == "sum")
        {
            return input_read::sum;
        }
        if(operation == "max")
        {
            return input_read::max;
        }
        if(operation == "min")
        {
            return input_read::min;
        }
        if(operation == "avr")
        {
            return input_read::avr;
        }
        scanner.fail(fmt::format("unknown reduce operation '{}'", operation));
    }

    // accumulate() selects an SQ counter collection mode that the SDK applies only
    // inside its own derived counter; it cannot be reproduced from the plain counter.
    static void reject_accumulate(std::string_view name, expression_scanner& scanner)
    {
        if(name == "accumulate")
        {
            scanner.fail("accumulate() is not supported: it is an SQ counter collection "
                         "mode the SDK applies only inside its own derived counter");
        }
    }

    [[nodiscard]] const counter_metadata& lookup(std::string_view    name,
                                                 expression_scanner& scanner)
    {
        auto const itr = m_definitions.find(std::string{ name });
        if(itr == m_definitions.end())
        {
            scanner.fail(fmt::format("unknown counter '{}'", name));
        }
        auto& referenced = m_result.referenced_names;
        if(std::ranges::find(referenced, itr->second.name) == referenced.end())
        {
            referenced.push_back(itr->second.name);
        }
        return itr->second;
    }

    [[nodiscard]] size_t input_slot(const std::string& name)
    {
        auto& inputs = m_result.input_names;
        auto  itr    = std::ranges::find(inputs, name);
        if(itr == inputs.end())
        {
            inputs.push_back(name);
            return inputs.size() - 1;
        }
        return static_cast<size_t>(std::distance(inputs.begin(), itr));
    }

    const counter_definitions& m_definitions;
    std::string_view           m_expression;
    parsed_expression          m_result;
    std::vector<std::string>   m_expanding;
};

}  // namespace detail
}  // namespace rocprofsys::pmc::collectors::gpu_perf_counter
