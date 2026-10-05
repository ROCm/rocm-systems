// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <cstdint>
#include <string>
#include <vector>

struct packet_test_case
{
    std::string           block{};
    std::vector<uint32_t> events{};
};

struct packet_test_architecture
{
    std::string                   gfx{};
    std::vector<packet_test_case> cases{};
};

struct packet_test_options
{
    std::vector<packet_test_architecture> architectures{};
};

packet_test_options&
get_packet_test_options();

packet_test_options
parse_packet_test_options(int argc, char** argv);
