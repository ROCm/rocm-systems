// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier:  MIT
//
// Stub of the ATen RecordFunction observer API, so the collector builds with
// no PyTorch install. name(), operator_name(), currentThreadId(), and
// addGlobalCallback() are exported by libtorch_cpu and bind at load time.
//
// Layout is only needed where data crosses the boundary: RecordFunctionCallback
// is passed by value, while scope(), seqNr(), inputs, and forwardThreadId()
// require measured field offsets. Padding stands in for everything else.

#pragma once

#include <ATen/core/operator_name.h>
#include <torch_abi.h>

#include <array>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <memory>
#include <optional>
#include <type_traits>

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

static_assert(sizeof(ObserverContext) == torch_abi::kObserverContextSize);
static_assert(alignof(ObserverContext) == torch_abi::kObserverContextAlignment);

struct RecordFunction
{
    virtual ~RecordFunction();

    RecordFunction(const RecordFunction&)            = delete;
    RecordFunction& operator=(const RecordFunction&) = delete;

    const char* name() const;

    std::optional<c10::OperatorName> operator_name() const;

    [[nodiscard]] RecordScope scope() const noexcept
    {
        return read_abi_value<RecordScope>(torch_abi::kRecordFunctionScopeOff);
    }

    [[nodiscard]] std::int64_t seqNr() const noexcept
    {
        return read_abi_value<std::int64_t>(torch_abi::kRecordFunctionSeqNrOff);
    }

    [[nodiscard]] std::uint64_t forwardThreadId() const noexcept
    {
        return read_abi_value<std::uint64_t>(torch_abi::kRecordFunctionFwdThreadOff);
    }

    static std::uint64_t currentThreadId();

private:
    template<typename Tp>
    [[nodiscard]] Tp read_abi_value(std::size_t offset) const noexcept
    {
        Tp value{};
        std::memcpy(&value, reinterpret_cast<const std::byte*>(this) + offset, sizeof(value));
        return value;
    }

    std::byte abi_storage_[torch_abi::kRecordFunctionSize - sizeof(void*)];
};

static_assert(sizeof(RecordFunction) == torch_abi::kRecordFunctionSize);
static_assert(alignof(RecordFunction) == torch_abi::kRecordFunctionAlignment);
static_assert(sizeof(std::optional<c10::OperatorName>) == torch_abi::kOptionalOperatorNameSize);
static_assert(alignof(std::optional<c10::OperatorName>) == torch_abi::kOptionalOperatorNameAlignment);

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

    RecordFunctionCallback& needsInputs(bool needs_inputs)
    {
        needs_inputs_ = needs_inputs;
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
static_assert(alignof(RecordFunctionCallback) == torch_abi::kCallbackAlignment);
static_assert(std::is_standard_layout_v<RecordFunctionCallback>);
static_assert(offsetof(RecordFunctionCallback, start_) == torch_abi::kCallbackStartOff);
static_assert(offsetof(RecordFunctionCallback, end_) == torch_abi::kCallbackEndOff);
static_assert(offsetof(RecordFunctionCallback, sampling_prob_) == torch_abi::kCallbackProbabilityOff);
static_assert(offsetof(RecordFunctionCallback, scopes_) == torch_abi::kCallbackScopesOff);
static_assert(offsetof(RecordFunctionCallback, needs_inputs_) == torch_abi::kCallbackNeedsInputsOff);

CallbackHandle addGlobalCallback(RecordFunctionCallback cb);

}  // namespace at
