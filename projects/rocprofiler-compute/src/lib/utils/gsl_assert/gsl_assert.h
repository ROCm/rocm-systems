// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier:  MIT
#pragma once
#include <sstream>
#include <string>

#define Expects(cond)                                   \
    if (!(cond))                                        \
    {                                                   \
        std::stringstream ss;                           \
        ss << "[" __FILE__ << ": " << __LINE__ << "] "; \
        ss << "Precondition failed: " #cond;            \
        throw std::runtime_error(ss.str());             \
    }
