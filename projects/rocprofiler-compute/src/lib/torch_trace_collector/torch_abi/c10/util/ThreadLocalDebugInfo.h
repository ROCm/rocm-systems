// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier:  MIT
//
// Minimal declarations for the c10 ThreadLocalDebugInfo ABI used to publish
// Python user-scope stacks to autograd workers.

#pragma once

#include <torch_abi.h>

#include <memory>
#include <string_view>
#include <type_traits>

namespace c10
{

struct DebugInfoKind
{
    explicit constexpr DebugInfoKind(const std::string_view* value) noexcept
        : value_{value}
    {
    }

    const std::string_view* value_ = nullptr;
};

// These assertions validate the replacement declaration, not PyTorch itself.
// test_torch_trace_collector.cpp checks the constants against the real headers.
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
};

class DebugInfoGuard
{
public:
    DebugInfoGuard(DebugInfoKind kind, std::shared_ptr<DebugInfoBase> info);
    ~DebugInfoGuard();

    DebugInfoGuard(const DebugInfoGuard&)            = delete;
    DebugInfoGuard(DebugInfoGuard&&)                 = delete;
    DebugInfoGuard& operator=(const DebugInfoGuard&) = delete;
    DebugInfoGuard& operator=(DebugInfoGuard&&)      = delete;

private:
    bool                                  active_    = false;
    std::shared_ptr<ThreadLocalDebugInfo> prev_info_ = nullptr;
};

static_assert(sizeof(DebugInfoGuard) == torch_abi::kDebugInfoGuardSize);
static_assert(alignof(DebugInfoGuard) == torch_abi::kDebugInfoGuardAlignment);

}  // namespace c10
