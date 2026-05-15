// Copyright (c) Advanced Micro Devices, Inc.
//  SPDX-License-Identifier:  MIT

#pragma once

#include "common/defines.h"

#define ROCPROFSYS_METADATA(...) ::tim::manager::add_metadata(__VA_ARGS__)

#if !defined(ROCPROFSYS_DEFAULT_OBJECT)
#    define ROCPROFSYS_DEFAULT_OBJECT(NAME)                                              \
        NAME()                           = default;                                      \
        NAME(const NAME&)                = default;                                      \
        NAME(NAME&&) noexcept            = default;                                      \
        NAME& operator=(const NAME&)     = default;                                      \
        NAME& operator=(NAME&&) noexcept = default;
#endif

#if !defined(ROCPROFSYS_DEFAULT_COPY_MOVE)
#    define ROCPROFSYS_DEFAULT_COPY_MOVE(NAME)                                           \
        NAME(const NAME&)                = default;                                      \
        NAME(NAME&&) noexcept            = default;                                      \
        NAME& operator=(const NAME&)     = default;                                      \
        NAME& operator=(NAME&&) noexcept = default;
#endif
