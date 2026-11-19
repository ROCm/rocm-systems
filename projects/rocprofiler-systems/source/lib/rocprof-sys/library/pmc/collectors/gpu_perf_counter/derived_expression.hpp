// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

#include "common/string_utility.hpp"
#include "library/pmc/collectors/gpu_perf_counter/types.hpp"

#include <fmt/format.h>

#include <algorithm>
#include <cctype>
#include <cstddef>
#include <cstdint>
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

namespace detail
{
class expression_parser;
}  // namespace detail

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
    void add(double value) noexcept;
};

/// @brief Counter metadata by counter name; resolves the names used in expressions.
using counter_definitions = std::unordered_map<std::string, counter_metadata>;

/// @brief Indexes @p counters by name. Instances of one counter share a definition.
[[nodiscard]] counter_definitions
make_counter_definitions(std::span<const counter_metadata> counters);

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
                       const counter_definitions& definitions);

    /// @brief How a sample value of this expression is obtained.
    [[nodiscard]] derived_evaluation evaluation() const noexcept { return m_evaluation; }

    /// @brief Raw and constant counters the expression reads, in input index order.
    [[nodiscard]] const std::vector<std::string>& input_names() const noexcept
    {
        return m_input_names;
    }

    /// @brief Every counter name the expression refers to, including expanded ones.
    [[nodiscard]] const std::vector<std::string>& referenced_names() const noexcept
    {
        return m_referenced_names;
    }

    /**
     * @brief Evaluates the expression.
     * @param inputs One reduction per input_names() entry, in the same order.
     * @return The value. A division by zero yields 0.
     */
    [[nodiscard]] double evaluate(std::span<const input_reduction> inputs) const;

private:
    friend class detail::expression_parser;

    // How an input node reads its input counter.
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

    // Children are indices into m_nodes.
    struct node
    {
        node_kind  kind{ node_kind::number };
        double     number{ 0.0 };
        size_t     input{ 0 };
        input_read read{ input_read::value };
        size_t     lhs{ 0 };
        size_t     rhs{ 0 };
    };

    [[nodiscard]] static double read_input(const input_reduction& input, input_read read);
    [[nodiscard]] double        evaluate_node(size_t                           index,
                                              std::span<const input_reduction> inputs) const;

    std::vector<node>        m_nodes;
    size_t                   m_root{ 0 };
    derived_evaluation       m_evaluation{ derived_evaluation::constant };
    std::vector<std::string> m_input_names;
    std::vector<std::string> m_referenced_names;
};

inline void
input_reduction::add(double value) noexcept
{
    max = count == 0 ? value : std::max(max, value);
    min = count == 0 ? value : std::min(min, value);
    sum += value;
    ++count;
}

inline counter_definitions
make_counter_definitions(std::span<const counter_metadata> counters)
{
    auto definitions = counter_definitions{};
    for(const auto& meta : counters)
    {
        definitions.try_emplace(meta.name, meta);
    }
    return definitions;
}

namespace detail
{

class expression_parser
{
    using node       = derived_expression::node;
    using node_kind  = derived_expression::node_kind;
    using input_read = derived_expression::input_read;

public:
    // A parsed sub-expression: its node, how its sample value behaves, and whether it
    // reads a counter outside reduce() (one value per counter instance).
    struct parsed
    {
        size_t             index{ 0 };
        derived_evaluation shape{ derived_evaluation::constant };
        bool               per_instance{ false };
    };

    expression_parser(derived_expression& target, const counter_definitions& definitions,
                      std::string_view expression)
    : m_target{ target }
    , m_definitions{ definitions }
    , m_expression{ expression }
    {}

    [[nodiscard]] parsed parse(std::string_view text)
    {
        auto const outer_text     = std::exchange(m_text, text);
        auto const outer_position = std::exchange(m_position, 0);

        auto const result = parse_sum();
        skip_spaces();
        if(m_position != m_text.size())
        {
            fail(fmt::format("unexpected '{}'", m_text.substr(m_position)));
        }

        m_text     = outer_text;
        m_position = outer_position;
        return result;
    }

    [[noreturn]] void fail(std::string_view reason) const
    {
        throw std::invalid_argument(
            fmt::format("derived counter expression '{}': {}", m_expression, reason));
    }

private:
    [[nodiscard]] parsed parse_sum()
    {
        auto result = parse_product();
        while(true)
        {
            if(consume('+'))
            {
                result = add_binary(node_kind::add, result, parse_product());
            }
            else if(consume('-'))
            {
                result = add_binary(node_kind::subtract, result, parse_product());
            }
            else
            {
                return result;
            }
        }
    }

    [[nodiscard]] parsed parse_product()
    {
        auto result = parse_factor();
        while(true)
        {
            if(consume('*'))
            {
                result = add_binary(node_kind::multiply, result, parse_factor());
            }
            else if(consume('/'))
            {
                result = add_binary(node_kind::divide, result, parse_factor());
            }
            else
            {
                return result;
            }
        }
    }

    [[nodiscard]] parsed parse_factor()
    {
        skip_spaces();
        if(consume('('))
        {
            auto const inner = parse_sum();
            expect(')');
            return inner;
        }
        if(m_position < m_text.size() && is_number_char(m_text[m_position]))
        {
            return parse_number();
        }
        return parse_name();
    }

    [[nodiscard]] parsed parse_number()
    {
        auto const start = m_position;
        while(m_position < m_text.size() && is_number_char(m_text[m_position]))
        {
            ++m_position;
        }
        auto const token  = m_text.substr(start, m_position - start);
        auto const number = utility::string::safe_stod(token);
        if(number.is_error())
        {
            fail(fmt::format("invalid number '{}'", token));
        }
        return { .index =
                     add_node({ .kind = node_kind::number, .number = number.value() }),
                 .shape = derived_evaluation::constant };
    }

    [[nodiscard]] parsed parse_name()
    {
        auto const name = read_identifier();
        if(name == "accumulate")
        {
            fail("accumulate() is not supported");
        }
        if(name == "reduce")
        {
            return parse_reduce();
        }
        const auto& meta = lookup(name);
        if(meta.is_derived)
        {
            return expand_derived(meta);
        }
        return add_input(meta, input_read::value);
    }

    [[nodiscard]] parsed parse_reduce()
    {
        expect('(');
        auto const name = read_identifier();
        if(name == "accumulate")
        {
            fail("accumulate() is not supported");
        }
        const auto& meta = lookup(name);
        if(meta.is_derived)
        {
            fail(fmt::format("reduce() of derived counter '{}'", name));
        }
        expect(',');
        auto const read = to_input_read(read_identifier());
        expect(')');
        return add_input(meta, read);
    }

    [[nodiscard]] parsed expand_derived(const counter_metadata& meta)
    {
        if(std::ranges::find(m_expanding, meta.name) != m_expanding.end())
        {
            fail(fmt::format("cyclic definition of '{}'", meta.name));
        }
        m_expanding.push_back(meta.name);
        auto const result = parse(meta.expression);
        m_expanding.pop_back();
        return result;
    }

    [[nodiscard]] parsed add_input(const counter_metadata& meta, input_read read)
    {
        auto const index = add_node(
            { .kind = node_kind::input, .input = input_index(meta.name), .read = read });
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

    [[nodiscard]] parsed add_binary(node_kind kind, parsed lhs, parsed rhs)
    {
        return { .index = add_node({ .kind = kind, .lhs = lhs.index, .rhs = rhs.index }),
                 .shape = combine_shapes(kind, lhs.shape, rhs.shape),
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
        fail(fmt::format("unknown reduce operation '{}'", operation));
    }

    [[nodiscard]] const counter_metadata& lookup(std::string_view name)
    {
        auto const itr = m_definitions.find(std::string{ name });
        if(itr == m_definitions.end())
        {
            fail(fmt::format("unknown counter '{}'", name));
        }
        auto& referenced = m_target.m_referenced_names;
        if(std::ranges::find(referenced, itr->second.name) == referenced.end())
        {
            referenced.push_back(itr->second.name);
        }
        return itr->second;
    }

    [[nodiscard]] size_t input_index(const std::string& name)
    {
        auto& inputs = m_target.m_input_names;
        auto  itr    = std::ranges::find(inputs, name);
        if(itr == inputs.end())
        {
            inputs.push_back(name);
            return inputs.size() - 1;
        }
        return static_cast<size_t>(std::distance(inputs.begin(), itr));
    }

    [[nodiscard]] size_t add_node(node value)
    {
        m_target.m_nodes.push_back(value);
        return m_target.m_nodes.size() - 1;
    }

    [[nodiscard]] std::string_view read_identifier()
    {
        skip_spaces();
        auto const start = m_position;
        while(m_position < m_text.size() && is_identifier_char(m_text[m_position]))
        {
            ++m_position;
        }
        if(start == m_position)
        {
            fail(fmt::format("expected a name at '{}'", m_text.substr(start)));
        }
        return m_text.substr(start, m_position - start);
    }

    void expect(char token)
    {
        if(!consume(token))
        {
            fail(fmt::format("expected '{}' at '{}'", token, m_text.substr(m_position)));
        }
    }

    [[nodiscard]] bool consume(char token)
    {
        skip_spaces();
        if(m_position < m_text.size() && m_text[m_position] == token)
        {
            ++m_position;
            return true;
        }
        return false;
    }

    void skip_spaces()
    {
        while(m_position < m_text.size() &&
              std::isspace(static_cast<unsigned char>(m_text[m_position])) != 0)
        {
            ++m_position;
        }
    }

    [[nodiscard]] static bool is_number_char(char character)
    {
        return std::isdigit(static_cast<unsigned char>(character)) != 0 ||
               character == '.';
    }

    [[nodiscard]] static bool is_identifier_char(char character)
    {
        return std::isalnum(static_cast<unsigned char>(character)) != 0 ||
               character == '_';
    }

    derived_expression&        m_target;
    const counter_definitions& m_definitions;
    std::string_view           m_expression;
    std::string_view           m_text;
    size_t                     m_position{ 0 };
    std::vector<std::string>   m_expanding;
};

}  // namespace detail

inline derived_expression::derived_expression(std::string_view           expression,
                                              const counter_definitions& definitions)
{
    auto       parser = detail::expression_parser{ *this, definitions, expression };
    auto const root   = parser.parse(expression);
    if(root.shape == derived_evaluation::interval && root.per_instance)
    {
        parser.fail("non-linear expression reads a counter outside reduce()");
    }
    m_root       = root.index;
    m_evaluation = root.shape;
}

inline double
derived_expression::evaluate(std::span<const input_reduction> inputs) const
{
    return evaluate_node(m_root, inputs);
}

inline double
derived_expression::read_input(const input_reduction& input, input_read read)
{
    switch(read)
    {
        case input_read::sum: return input.sum;
        case input_read::max: return input.max;
        case input_read::min: return input.min;
        case input_read::avr:
            return input.count == 0 ? 0.0 : input.sum / static_cast<double>(input.count);
        case input_read::value: return input.max;
        default:
            throw std::logic_error(
                fmt::format("input_read {} not handled", static_cast<int>(read)));
    }
}

inline double
derived_expression::evaluate_node(size_t                           index,
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
            return denominator == 0.0 ? 0.0
                                      : evaluate_node(current.lhs, inputs) / denominator;
        }
        default:
            throw std::logic_error(
                fmt::format("node_kind {} not handled", static_cast<int>(current.kind)));
    }
}

}  // namespace rocprofsys::pmc::collectors::gpu_perf_counter
