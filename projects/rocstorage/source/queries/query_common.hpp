// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier:  MIT

#pragma once

namespace rocstorage::queries
{

enum class sort_order
{
    ascending,
    descending
};

enum class join_type
{
    inner,
    left,
    right
};

}  // namespace rocstorage::queries
