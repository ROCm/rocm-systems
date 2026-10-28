// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "rocprof-sys/library/components/backtrace_metrics.hpp"

#include <gtest/gtest.h>
#include <string>
#include <timemory/backends/papi.hpp>

TEST(backtrace_metrics, net_counter_track_uses_short_description)
{
    std::string _symbol;
    for(const auto& _event : tim::papi::available_events_info())
    {
        const auto& _name = _event.symbol();
        if(_name.find("net:::") != std::string::npos &&
           _name.find("rx:byte") != std::string::npos)
        {
            _symbol = _name;
            break;
        }
    }
    if(_symbol.empty() || !tim::papi::working())
    {
        GTEST_SKIP() << "PAPI net receive-byte event is not available";
    }

    // available_events_info() initializes PAPI without filling the component map.
    // init() must fill it, or the track title stays the raw symbol.
    ASSERT_TRUE(tim::papi::init());

    const auto _track = rocprofsys::component::hardware_counter_track_name(_symbol, 0);
    EXPECT_NE(_track.find("receive byte"), std::string::npos);
    EXPECT_EQ(_track.find("rx:byte"), std::string::npos);
    EXPECT_NE(_track.find("Thread "), std::string::npos);
    EXPECT_NE(_track.find(" [0] (S)"), std::string::npos);
}
