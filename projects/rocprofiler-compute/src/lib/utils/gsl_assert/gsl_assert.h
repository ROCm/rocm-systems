// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier:  MIT
#pragma once

#define Expects(cond) \
    if (!(cond))      \
        throw std::runtime_error("Precondition failed: " #cond)
