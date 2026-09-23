// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier:  MIT
//
// Minimal declarations for the c10 thread-local debug-info API used to carry
// the Python autograd launcher thread ID into RecordFunction worker callbacks.

#pragma once

#include <torch_abi.h>

#include <memory>
#include <string_view>
#include <type_traits>

namespace c10
{

class DebugInfoKind
{
public:
    explicit constexpr DebugInfoKind(const std::string_view* value)
        : value_{value}
    {
    }

private:
    [[maybe_unused]] const std::string_view* value_ = nullptr;
};

static_assert(sizeof(DebugInfoKind) == torch_abi::kDebugInfoKindSize);
static_assert(alignof(DebugInfoKind) == torch_abi::kDebugInfoKindAlignment);
static_assert(std::is_trivially_copyable_v<DebugInfoKind>);
static_assert(std::is_standard_layout_v<DebugInfoKind>);

class DebugInfoBase
{
public:
    DebugInfoBase()          = default;
    virtual ~DebugInfoBase() = default;
};

static_assert(sizeof(DebugInfoBase) == torch_abi::kDebugInfoBaseSize);
static_assert(alignof(DebugInfoBase) == torch_abi::kDebugInfoBaseAlignment);
static_assert(sizeof(std::shared_ptr<DebugInfoBase>) == torch_abi::kSharedDebugInfoSize);
static_assert(alignof(std::shared_ptr<DebugInfoBase>) == torch_abi::kSharedDebugInfoAlignment);

class ThreadLocalDebugInfo
{
public:
    static DebugInfoBase* get(DebugInfoKind kind);
    static void           _push(DebugInfoKind kind, std::shared_ptr<DebugInfoBase> info);
    static std::shared_ptr<DebugInfoBase> _pop(DebugInfoKind kind);
};

}  // namespace c10
