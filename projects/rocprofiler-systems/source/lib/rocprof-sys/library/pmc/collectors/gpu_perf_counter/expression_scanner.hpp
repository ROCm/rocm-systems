// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

#include <fmt/format.h>

#include <algorithm>
#include <cctype>
#include <concepts>
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
    : m_rest{ text }
    , m_expression{ expression }
    {}

    /// @brief Reads @p token if it is next.
    [[nodiscard]] bool consume(char token)
    {
        skip_spaces();
        if(!m_rest.starts_with(token))
        {
            return false;
        }
        m_rest.remove_prefix(1);
        return true;
    }

    /// @brief Reads @p token or fails.
    void expect(char token)
    {
        if(!consume(token))
        {
            throw std::invalid_argument(
                fmt::format("derived counter expression '{}': expected '{}' at '{}'",
                            m_expression, token, m_rest));
        }
    }

    /// @brief Fails unless only spaces are left.
    void expect_end()
    {
        skip_spaces();
        if(!m_rest.empty())
        {
            throw std::invalid_argument(
                fmt::format("derived counter expression '{}': unexpected '{}'",
                            m_expression, m_rest));
        }
    }

    /// @brief Whether a number starts at the cursor.
    [[nodiscard]] bool at_number()
    {
        skip_spaces();
        return !m_rest.empty() && is_number_char(m_rest.front());
    }

    /// @brief Reads a run of digits and dots; the caller checks it is a number.
    [[nodiscard]] std::string_view read_number() { return take_while(is_number_char); }

    /// @brief Reads a name of letters, digits and underscores, or fails.
    [[nodiscard]] std::string_view read_identifier()
    {
        skip_spaces();
        auto const name = take_while(is_identifier_char);
        if(name.empty())
        {
            throw std::invalid_argument(
                fmt::format("derived counter expression '{}': expected a name at '{}'",
                            m_expression, m_rest));
        }
        return name;
    }

private:
    [[nodiscard]] std::string_view take_while(std::predicate<char> auto is_token_char)
    {
        auto const end    = std::ranges::find_if_not(m_rest, is_token_char);
        auto const length = static_cast<size_t>(end - m_rest.begin());
        auto const token  = m_rest.substr(0, length);
        m_rest.remove_prefix(length);
        return token;
    }

    void skip_spaces() { (void) take_while(is_space_char); }

    [[nodiscard]] static bool is_space_char(char character)
    {
        return std::isspace(static_cast<unsigned char>(character)) != 0;
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

    std::string_view m_rest;  // the text not read yet
    std::string_view m_expression;
};

}  // namespace rocprofsys::pmc::collectors::gpu_perf_counter::detail
