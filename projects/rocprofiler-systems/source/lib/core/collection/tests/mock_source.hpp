// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

#include "core/collection/sink_interface.hpp"
#include "core/collection/source_interface.hpp"

#include <gmock/gmock.h>

#include <cstdint>
#include <string_view>

namespace rocprofsys::collection::testing
{
class mock_source : public source_interface
{
public:
    MOCK_METHOD(std::string_view, name, (), (const, noexcept, override));
    MOCK_METHOD(bool, describe, (sink_interface & out), (override));
    MOCK_METHOD(void, poll, (std::uint64_t timeline_ns, sink_interface& out), (override));
    MOCK_METHOD(void, release, (), (override));
};
}  // namespace rocprofsys::collection::testing
