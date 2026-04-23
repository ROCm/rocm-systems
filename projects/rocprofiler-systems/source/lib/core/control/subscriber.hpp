// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

#include <functional>
#include <string>

namespace rocprofsys::control
{
struct subscriber
{
    std::function<void()> on_pause;
    std::function<void()> on_resume;
    std::string           name;
};
}  // namespace rocprofsys::control
