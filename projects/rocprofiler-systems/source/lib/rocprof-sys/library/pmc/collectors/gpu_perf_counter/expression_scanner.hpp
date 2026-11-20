// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

#include <fmt/format.h>

#include <cctype>
#include <cstddef>
#include <stdexcept>
#include <string_view>

namespace rocprofsys::pmc::collectors::gpu_perf_counter::detail
{

/**
 * @brief A cursor over expression text that skips spaces and reads tokens.
 *
 * Errors name @p expression, the text the user wrote, even when the scanner reads
 * a derived counter's expansion of it.
 */
class expression_scanner
{
public:
    expression_scanner(std::string_view text, std::string_view expression)
    : m_text{ text }
    , m_expression{ expression }
    {}

    /// @brief Reads @p token if it is next.
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

    /// @brief Reads @p token or fails.
    void expect(char token)
    {
        if(!consume(token))
        {
            throw std::invalid_argument(
                fmt::format("derived counter expression '{}': expected '{}' at '{}'",
                            m_expression, token, m_text.substr(m_position)));
        }
    }

    /// @brief Fails unless only spaces are left.
    void expect_end()
    {
        skip_spaces();
        if(m_position != m_text.size())
        {
            throw std::invalid_argument(
                fmt::format("derived counter expression '{}': unexpected '{}'",
                            m_expression, m_text.substr(m_position)));
        }
    }

    /// @brief Whether a number starts at the cursor.
    [[nodiscard]] bool at_number()
    {
        skip_spaces();
        return m_position < m_text.size() && is_number_char(m_text[m_position]);
    }

    /// @brief Reads a run of digits and dots; the caller checks it is a number.
    [[nodiscard]] std::string_view read_number() { return read_while(is_number_char); }

    /// @brief Reads a name of letters, digits and underscores, or fails.
    [[nodiscard]] std::string_view read_identifier()
    {
        skip_spaces();
        auto const name = read_while(is_identifier_char);
        if(name.empty())
        {
            throw std::invalid_argument(
                fmt::format("derived counter expression '{}': expected a name at '{}'",
                            m_expression, m_text.substr(m_position)));
        }
        return name;
    }

private:
    template <typename Predicate>
    [[nodiscard]] std::string_view read_while(Predicate is_token_char)
    {
        auto const start = m_position;
        while(m_position < m_text.size() && is_token_char(m_text[m_position]))
        {
            ++m_position;
        }
        return m_text.substr(start, m_position - start);
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

    std::string_view m_text;
    std::string_view m_expression;
    size_t           m_position{ 0 };
};

}  // namespace rocprofsys::pmc::collectors::gpu_perf_counter::detail
