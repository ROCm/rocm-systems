/* Copyright (c) Advanced Micro Devices, Inc. All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 */

#pragma once

#include "hipfile-warnings.h"

#include <array>
#include <string>

enum class IoTestBackend {
    Fastpath,
    Fallback,
};

struct IoTestParam {
    IoTestBackend backend;
    std::string   name;
};

HIPFILE_WARN_NO_EXIT_DTOR_OFF
inline const std::array<IoTestParam, 2> io_test_params{
    {{IoTestBackend::Fastpath, "Fastpath"}, {IoTestBackend::Fallback, "Fallback"}}};
HIPFILE_WARN_NO_EXIT_DTOR_ON
