/* Copyright (c) Advanced Micro Devices, Inc. All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 */

#pragma once

#include "args.h"

#include <cstdlib>
#include <cstdint>
#include <iostream>
#include <string>

inline void
parseCli(args::ArgumentParser &parser, int argc, char **argv)
{
    try {
        parser.ParseCLI(argc, argv);
    }
    catch (const args::Help &) {
        std::cout << parser;
        std::exit(EXIT_SUCCESS);
    }
    catch (const args::Error &error) {
        std::cerr << error.what() << '\n' << parser;
        std::exit(EXIT_FAILURE);
    }
}

struct SystemTestOptions {
    std::string ais_capable_dir;
    uint32_t    sleep_seconds;
    void        parseTestOptions(int argc, char **argv)
    {
        args::ArgumentParser            parser{"System test options"};
        [[maybe_unused]] args::HelpFlag help_arg(parser, "help", "Show this help message",
                                                 args::Matcher{'h', "help"});
        args::ValueFlag<std::string>    ais_capable_dir_arg(parser, "path", "Path to AIS capable directory",
                                                            args::Matcher{"ais-capable-dir"}, "/tmp",
                                                            args::Options::Single);
        args::ValueFlag<uint32_t>       sleep_on_exit_arg(parser, "seconds", "Sleep before returning to main",
                                                          args::Matcher{"sleep-on-exit"}, 0U,
                                                          args::Options::Single);

        parseCli(parser, argc, argv);

        ais_capable_dir = args::get(ais_capable_dir_arg);
        sleep_seconds   = args::get(sleep_on_exit_arg);
    }
};

struct UnitTestOptions {
    uint32_t sleep_seconds;
    void     parseTestOptions(int argc, char **argv)
    {
        args::ArgumentParser            parser{"Unit test options"};
        [[maybe_unused]] args::HelpFlag help_arg(parser, "help", "Show this help message",
                                                 args::Matcher{'h', "help"});
        args::ValueFlag<uint32_t>       sleep_on_exit_arg(parser, "seconds", "Sleep before returning to main",
                                                          args::Matcher{"sleep-on-exit"}, 0U,
                                                          args::Options::Single);

        parseCli(parser, argc, argv);

        sleep_seconds = args::get(sleep_on_exit_arg);
    }
};
