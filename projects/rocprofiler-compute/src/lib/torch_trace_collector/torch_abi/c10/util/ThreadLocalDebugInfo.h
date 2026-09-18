// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier:  MIT
//
// Stub of the c10 thread-local debug info API. ThreadLocalDebugInfo::get and
// the DebugInfoGuard constructor and destructor are exported by libc10 and
// bind at load time; the collector allocates the guard, so it needs the size.

#pragma once

#include <torch_abi.h>

#include <cstdint>
#include <memory>

namespace c10
{

// Opaque on purpose. PyTorch 2.12 makes this a one-byte enum of fixed slots,
// 2.13 a pointer wrapper that lets callers own a slot. Both arrive in one
// integer register, so an 8-byte value serves either. user_scope_kind() picks
// the payload at runtime.
class DebugInfoKind
{
public:
    constexpr DebugInfoKind() = default;

    explicit constexpr DebugInfoKind(std::uint64_t value)
        : value_(value)
    {
    }

    constexpr bool operator==(const DebugInfoKind other) const { return value_ == other.value_; }

    constexpr bool operator!=(const DebugInfoKind other) const { return value_ != other.value_; }

private:
    std::uint64_t value_ = 0;
};

static_assert(sizeof(DebugInfoKind) == torch_abi::kDebugInfoKindSize);

// The collector derives from this and recovers its own type with dynamic_cast.
class DebugInfoBase
{
public:
    DebugInfoBase()          = default;
    virtual ~DebugInfoBase() = default;
};

class ThreadLocalDebugInfo;

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

}  // namespace c10
