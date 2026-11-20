// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "rocprof-sys/library/components/backtrace_metrics.hpp"

#include <gtest/gtest.h>
#include <string>
#include <timemory/backends/papi.hpp>

namespace
{
std::string
net_rx_byte_symbol()
{
    std::string symbol;
    for(const auto& event : tim::papi::available_events_info())
    {
        const auto& name = event.symbol();
        if(name.find("net:::") != std::string::npos &&
           name.find("rx:byte") != std::string::npos)
        {
            symbol = name;
            break;
        }
    }
    if(symbol.empty() || !tim::papi::working())
    {
        return {};
    }
    return symbol;
}
}  // namespace

TEST(backtrace_metrics, net_counter_track_uses_short_description)
{
    const auto symbol = net_rx_byte_symbol();
    if(symbol.empty())
    {
        GTEST_SKIP() << "PAPI net receive-byte event is not available";
    }

    // available_events_info() initializes PAPI without filling the component map.
    // init() must fill it, or the track title stays the raw symbol.
    ASSERT_TRUE(tim::papi::init());

    const auto track = rocprofsys::component::hardware_counter_track_name(symbol, 0);
    EXPECT_NE(track.find("receive byte"), std::string::npos);
    EXPECT_EQ(track.find("rx:byte"), std::string::npos);
    EXPECT_NE(track.find("Thread "), std::string::npos);
    EXPECT_NE(track.find(" [0] (S)"), std::string::npos);
}
