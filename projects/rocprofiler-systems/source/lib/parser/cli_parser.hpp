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
    /// Stop and return that token plus everything after it as unparsed.
    stop,
};

/**
 * Reads argv tokens against each parameter's arg spec.
 *
 * A value option accepts both "--name value" and "--name=value"; arg.assign_style
 * records the preferred form and does not reject the other. A flag with no value
 * stores "true", and "--flag=false" stores the explicit value. A single-letter
 * flag bundle such as "-qv" is accepted when every letter is a flag argument.
 * "--" ends option parsing; the tokens after it are the command and are not parsed.
 */
class cli_parser
{
public:
    cli_parser(std::vector<std::string> args, unknown_option unknown);

    /// Copies argv[1..argc). argv[0], the program name, is not an option.
    [[nodiscard]] static std::vector<std::string> arguments_from(int argc, char** argv);

    [[nodiscard]] parse_result parse(const schema& config_schema) const;

private:
    std::vector<std::string> m_args;
    unknown_option           m_unknown;
};

}  // namespace rocprofsys::parser
