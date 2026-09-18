// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier:  MIT
//
// Stub of the ATen RecordFunction observer API, so the collector builds with
// no PyTorch install. name(), currentThreadId(), addGlobalCallback and
// removeCallback are exported by libtorch_cpu and bind at load time.
//
// Layout is only needed where data crosses the boundary: RecordFunctionCallback
// is passed by value, and seqNr() and forwardThreadId() read fields. Padding
// stands in for everything else, which matches the shipped libtorch whatever
// standard library we build with.

#pragma once

#include "torch_abi.h"

#include <array>
#include <cstddef>
#include <cstdint>
#include <initializer_list>
#include <memory>

namespace at
{

enum class RecordScope : std::uint8_t
{
    FUNCTION = 0,
    BACKWARD_FUNCTION,
    TORCHSCRIPT_FUNCTION,
    KERNEL_FUNCTION_DTYPE,
    CUSTOM_CLASS,
    BUILD_FEATURE,
    LITE_INTERPRETER,
    USER_SCOPE,
    STATIC_RUNTIME_OP,
    STATIC_RUNTIME_MODEL,
    NUM_SCOPES,
};

static_assert(static_cast<std::size_t>(RecordScope::NUM_SCOPES) == torch_abi::kScopeCount);

using CallbackHandle = std::uint64_t;

inline constexpr CallbackHandle INVALID_CALLBACK_HANDLE{0};

// libtorch deletes this through the base pointer.
struct ObserverContext
{
    virtual ~ObserverContext() = default;

protected:
    ObserverContext() = default;
};

struct RecordFunction
{
    virtual ~RecordFunction();

    RecordFunction(const RecordFunction&)            = delete;
    RecordFunction& operator=(const RecordFunction&) = delete;

    const char* name() const;

    std::int64_t seqNr() const { return sequence_nr_; }

    std::uint64_t forwardThreadId() const { return fwd_thread_id_; }

    static std::uint64_t currentThreadId();

    std::byte    abi_head[torch_abi::kRecordFunctionSeqNrOff - sizeof(void*)];
    std::int64_t sequence_nr_;
    std::byte abi_mid[torch_abi::kRecordFunctionFwdThreadOff - torch_abi::kRecordFunctionSeqNrOff -
                      sizeof(std::int64_t)];
    std::uint64_t fwd_thread_id_;
    std::byte     abi_tail[torch_abi::kRecordFunctionSize - torch_abi::kRecordFunctionFwdThreadOff -
                           sizeof(std::uint64_t)];
};

static_assert(sizeof(RecordFunction) == torch_abi::kRecordFunctionSize);
static_assert(offsetof(RecordFunction, sequence_nr_) == torch_abi::kRecordFunctionSeqNrOff);
static_assert(offsetof(RecordFunction, fwd_thread_id_) == torch_abi::kRecordFunctionFwdThreadOff);

class RecordFunctionCallback
{
public:
    using StartCallback = std::unique_ptr<ObserverContext> (*)(const RecordFunction&);
    using EndCallback   = void (*)(const RecordFunction&, ObserverContext*);

    explicit RecordFunctionCallback(StartCallback start, EndCallback end = nullptr)
        : start_(start)
        , end_(end)
    {
        scopes_.fill(true);
    }

    // An empty list means every scope, matching PyTorch.
    RecordFunctionCallback& scopes(std::initializer_list<RecordScope> scopes)
    {
        scopes_.fill(scopes.size() == 0);
        for (RecordScope scope : scopes)
        {
            scopes_[static_cast<std::size_t>(scope)] = true;
        }
        return *this;
    }

    StartCallback start_;
    EndCallback   end_;
    double        sampling_prob_ = 1.0;

    std::array<bool, torch_abi::kScopeCount> scopes_ = {};

    bool needs_inputs_  = false;
    bool needs_outputs_ = false;
    bool needs_ids_     = false;
};

static_assert(sizeof(RecordFunctionCallback) == torch_abi::kCallbackSize);

CallbackHandle addGlobalCallback(RecordFunctionCallback cb);
void           removeCallback(CallbackHandle handle);

}  // namespace at
