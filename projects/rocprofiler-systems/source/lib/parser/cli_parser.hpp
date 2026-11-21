// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

#include "parser/parse_result.hpp"
#include "parser/schema.hpp"

#include <cstdint>
#include <string>
#include <vector>

namespace rocprofsys::parser
{
/// What to do with a token that is not one of the schema's argument names.
enum class unknown_option : std::uint8_t
{
    /// Record an error and keep parsing.
    report = 0,
    /// Keep that token, and the plain arguments that follow it, such as its value, in
    /// unparsed, in order, and keep parsing. A plain argument that comes first is kept
    /// as well. A wrapper passes unparsed to the tool it runs.
    forward,
};

/**
 * Reads argv tokens against each parameter's command-line spelling.
 *
 * A value option accepts both "--name value" and "--name=value". "--name=value" is
 * exactly one token. In the space form, arg_min_count and arg_max_count select how
 * many following tokens belong to the option; unset counts mean one value, or none
 * for a boolean. An enumeration list with no arg_max_count reads every token up to the
 * next option or "--", and its tokens are one list value: a token boundary separates
 * two names, and a separator already at the boundary is not doubled, so "hip," "mpi"
 * and "hip" "," "mpi" both give hip,mpi. That value is read like the same text from
 * any other source, so "," alone is refused; the error names the element at fault, and
 * the option then stores nothing. A repeated enumeration list adds to the elements
 * given earlier. The tokens of any other option are joined with the list separator,
 * and a repeat keeps the last value. A following token that starts with '-' is the
 * next option, not a value. A boolean option stores "true" and does not consume the
 * next token; "--name=false" still sets the value. An integer or number may be
 * negative ("--num -5"). A single-letter bundle such as "-qv" is accepted when every
 * letter is a boolean option. "--" ends option parsing, and only the tokens after it
 * are the command.
 *
 * Plain tokens right after an option and its values are unexpected arguments, as in
 * "--traces=hip mpi"; after the = form of a list, the error suggests the comma form,
 * "--traces=hip,mpi". Any other plain token before "--" is an unexpected argument too,
 * unless unknown_option::forward keeps it in unparsed.
 */
class cli_parser
{
public:
    cli_parser(std::vector<std::string> args, unknown_option unknown);

    [[nodiscard]] parse_result parse(const schema& config_schema) const;

private:
    std::vector<std::string> m_args;
    unknown_option           m_unknown;
};

}  // namespace rocprofsys::parser
