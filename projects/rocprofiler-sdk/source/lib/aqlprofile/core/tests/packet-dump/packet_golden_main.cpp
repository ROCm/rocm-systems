// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "packet_golden_options.hpp"

#include <gtest/gtest.h>

#include <getopt.h>

#include <algorithm>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>

namespace
{
uint32_t
parse_event_id(const std::string& value)
{
    size_t              parsed = 0;
    const unsigned long event  = std::stoul(value, &parsed, 0);
    if(parsed != value.size() || event > std::numeric_limits<uint32_t>::max())
        throw std::invalid_argument{"invalid event ID: " + value};
    return static_cast<uint32_t>(event);
}

void
append_events(std::vector<packet_test_event>& events, const std::string& csv)
{
    size_t begin = 0;
    while(begin <= csv.size())
    {
        const size_t end   = csv.find(',', begin);
        const auto   token = csv.substr(begin, end - begin);
        if(token.empty()) throw std::invalid_argument{"empty event ID in: " + csv};

        const size_t separator = token.rfind(':');
        if(separator == std::string::npos || separator == 0 || separator + 1 == token.size())
            throw std::invalid_argument{"event must use name:id format: " + token};

        const packet_test_event event{
            token.substr(0, separator),
            parse_event_id(token.substr(separator + 1)),
        };
        const auto duplicate =
            std::find_if(events.begin(), events.end(), [&event](const auto& candidate) {
                return candidate.name == event.name && candidate.id == event.id;
            });
        if(duplicate == events.end()) events.emplace_back(event);

        if(end == std::string::npos) break;
        begin = end + 1;
    }
}

}  // namespace

packet_test_options&
get_packet_test_options()
{
    static packet_test_options options{};
    return options;
}

packet_test_options
parse_packet_test_options(int argc, char** argv)
{
    packet_test_options       options{};
    packet_test_architecture* current_architecture{};
    packet_test_case*         current_case{};

    static const option long_options[] = {
        {"gfx", required_argument, nullptr, 'g'},
        {"block", required_argument, nullptr, 'b'},
        {"events", required_argument, nullptr, 'e'},
        {"skipped-events", required_argument, nullptr, 's'},
        {nullptr, 0, nullptr, 0},
    };

    opterr = 0;
    optind = 1;
    while(true)
    {
        const int parsed_option = getopt_long(argc, argv, ":g:b:e:s:", long_options, nullptr);
        if(parsed_option == -1) break;

        switch(parsed_option)
        {
            case 'g':
                options.architectures.push_back({optarg, {}});
                current_architecture = &options.architectures.back();
                current_case         = nullptr;
                break;
            case 'b':
                if(!current_architecture)
                    throw std::invalid_argument{"--block must follow a --gfx option"};
                current_architecture->cases.push_back({optarg, {}});
                current_case = &current_architecture->cases.back();
                break;
            case 'e':
                if(!current_case)
                    throw std::invalid_argument{"--events must follow a --block option"};
                append_events(current_case->events, optarg);
                break;
            case 's':
                if(!current_case)
                    throw std::invalid_argument{"--skipped-events must follow a --block option"};
                append_events(current_case->skipped_events, optarg);
                break;
            case ':':
                throw std::invalid_argument{"missing value for option " +
                                            std::string{argv[optind - 1]}};
            case '?':
            default:
                throw std::invalid_argument{"unknown packet-test option: " +
                                            std::string{argv[optind - 1]}};
        }
    }
    if(optind < argc)
        throw std::invalid_argument{"unexpected positional argument: " + std::string{argv[optind]}};

    for(const auto& architecture : options.architectures)
    {
        if(architecture.gfx.empty()) throw std::invalid_argument{"gfx name cannot be empty"};
        if(architecture.cases.empty())
            throw std::invalid_argument{"no blocks specified for " + architecture.gfx};

        for(const auto& test_case : architecture.cases)
        {
            if(test_case.block.empty()) throw std::invalid_argument{"block name cannot be empty"};
            if(test_case.events.empty())
                throw std::invalid_argument{"no events specified for block " + test_case.block};
        }
    }
    return options;
}

int
main(int argc, char** argv)
{
    ::testing::InitGoogleTest(&argc, argv);
    try
    {
        get_packet_test_options() = parse_packet_test_options(argc, argv);
    } catch(const std::exception& error)
    {
        std::cerr << "packet test argument error: " << error.what() << '\n';
        return 2;
    }
    return RUN_ALL_TESTS();
}
