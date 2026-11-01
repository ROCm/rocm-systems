// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// Playground for the parser library as it works today. It does not add
// rocsys options. It does not launch the application.
//
// Precedence, lowest to highest: default, config file, environment, preset,
// command line. Defaults are resolved and omitted from both exports.
// The process environment is read. The only preset this binary supplies is
// "demo", so the preset source can be tried without the product registry.
//
//   parser-demo -- ./my_app
//   ROCPROFSYS_VERBOSE=1 parser-demo -- ./my_app
//   ROCPROFSYS_VERBOSE=1 parser-demo --preset demo --verbose 4 -- ./my_app
//   parser-demo --config demo.json --output /tmp/out --trace -- ./my_app
//
// demo.json is grouped by category id:
//   { "configuration": { "output_path": "from-file" },
//     "logging": { "verbose": 1 },
//     "tracing": { "trace": true } }
//
// --debug, --trace, and --profile are flags. A bare flag stores true and does
// not consume the next token. --debug=false stores false. --debug= is an error.

#include "parser/configuration_parser.hpp"
#include "parser/default_schema.hpp"
#include "parser/preset_parser.hpp"
#include "parser/schema.hpp"

#include <cstdlib>
#include <exception>
#include <iostream>
#include <string>
#include <string_view>
#include <vector>

namespace
{
using rocprofsys::parser::configuration_parser;
using rocprofsys::parser::preset_parser;

void
print_usage()
{
    std::cout << "Usage: parser-demo [options] -- <application>\n"
              << "Options: --output PATH  --config FILE  --preset demo\n"
              << "         --verbose N (-1..4)  --debug  --trace  --profile\n"
              << "Precedence: default < file < environment < preset < command line.\n"
              << "This program does not launch the application.\n";
}

preset_parser::settings_t
demo_preset(std::string_view name)
{
    if(name != "demo")
    {
        return {};
    }
    return preset_parser::settings_t{
        { "ROCPROFSYS_OUTPUT_PATH", "from-preset" },
        { "ROCPROFSYS_TRACE", "true" },
        { "ROCPROFSYS_VERBOSE", "2" },
    };
}

void
print_lines(std::string_view title, const std::vector<std::string>& lines)
{
    std::cout << title << '\n';
    if(lines.empty())
    {
        std::cout << "  (none)\n";
        return;
    }
    for(const auto& line : lines)
    {
        std::cout << "  " << line << '\n';
    }
}

void
print_environment(const rocprofsys::parser::value_map_t& environment)
{
    std::cout << "export_as_environment():\n";
    if(environment.empty())
    {
        std::cout << "  (none)\n";
        return;
    }
    for(const auto& [name, value] : environment)
    {
        std::cout << "  " << name << '=' << value << '\n';
    }
}

void
print_report(const configuration_parser&            parser,
             const rocprofsys::parser::value_map_t& values)
{
    std::cout << "Resolved values (defaults included):\n";
    for(const auto& [param_id, value] : values)
    {
        std::cout << "  " << param_id << '=' << value << '\n';
    }
    print_lines("Errors:", parser.errors());
    print_lines("Command (tokens after --):", parser.command());
    print_lines("Unparsed:", parser.unparsed());
    print_environment(parser.export_as_environment());
    std::cout << "export_as_json():\n" << parser.export_as_json() << '\n';
}
}  // namespace

int
main(int argc, char** argv)
{
    configuration_parser parser;
    parser.set_preset_lookup(demo_preset);
    if(!parser.initialize(rocprofsys::parser::get_default_schema()))
    {
        std::cerr << parser.failure() << '\n';
        return EXIT_FAILURE;
    }
    if(argc < 2)
    {
        print_usage();
        std::cout << parser.render_help() << '\n';
        return EXIT_SUCCESS;
    }

    rocprofsys::parser::value_map_t values;
    try
    {
        values = parser.parse(argc, argv);
    } catch(const std::exception& err)
    {
        std::cerr << err.what() << '\n';
        return EXIT_FAILURE;
    }
    print_report(parser, values);
    return parser.errors().empty() ? EXIT_SUCCESS : EXIT_FAILURE;
}
