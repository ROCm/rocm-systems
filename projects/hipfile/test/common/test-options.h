/* Copyright (c) Advanced Micro Devices, Inc. All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 */

#pragma once

#include <boost/program_options.hpp>
#include <boost/program_options/detail/parsers.hpp>
#include <boost/program_options/options_description.hpp>
#include <boost/program_options/value_semantic.hpp>
#include <boost/program_options/variables_map.hpp>
#include <cstdint>

struct SystemTestOptions {
    std::string ais_capable_dir;
    uint32_t    sleep_seconds;
    void        parseTestOptions(int argc, char **argv)
    {
        namespace po = boost::program_options;
        po::options_description desc("System test options");
        desc.add_options()("ais-capable-dir", po::value<std::string>()->default_value("/tmp"),
                           "Path to AIS capable directory");
        desc.add_options()("sleep-on-exit", po::value<uint32_t>()->default_value(0),
                           "Sleep before returning to main");
        po::variables_map vm;
        po::store(po::parse_command_line(argc, argv, desc), vm);

        ais_capable_dir = vm["ais-capable-dir"].as<std::string>();
        sleep_seconds   = vm["sleep-on-exit"].as<uint32_t>();
    }
};

struct UnitTestOptions {
    uint32_t sleep_seconds;
    void     parseTestOptions(int argc, char **argv)
    {
        namespace po = boost::program_options;
        po::options_description desc("System test options");
        desc.add_options()("sleep-on-exit", po::value<uint32_t>()->default_value(0),
                           "Sleep before returning to main");
        po::variables_map vm;
        po::store(po::parse_command_line(argc, argv, desc), vm);

        sleep_seconds = vm["sleep-on-exit"].as<uint32_t>();
    }
};
