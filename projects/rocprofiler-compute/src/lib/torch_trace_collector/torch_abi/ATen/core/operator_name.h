// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier:  MIT

#pragma once

#include <torch_abi.h>

#include <cstddef>
#include <string>
#include <type_traits>

#if !defined(_GLIBCXX_USE_CXX11_ABI) || _GLIBCXX_USE_CXX11_ABI != 1
#    error "The PyTorch ABI shim requires the C++11 libstdc++ ABI"
#endif

namespace c10
{

struct OperatorName
{
    std::string name;
    std::string overload_name;
};

static_assert(std::is_standard_layout_v<OperatorName>);
static_assert(sizeof(OperatorName) == torch_abi::kOperatorNameSize);
static_assert(alignof(OperatorName) == torch_abi::kOperatorNameAlignment);
static_assert(offsetof(OperatorName, name) == torch_abi::kOperatorNameNameOff);
static_assert(offsetof(OperatorName, overload_name) == torch_abi::kOperatorNameOverloadNameOff);

}  // namespace c10
