// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier:  MIT
//
// Layout of the PyTorch types the collector touches, for the stub headers
// beside this file. Verified identical in the measured CPU PyTorch 2.13/2.14
// and ROCm PyTorch 2.14 artifacts. PyTorch does not guarantee this ABI.
//
// Keep this list short. Every entry is a version-specific fact.

#pragma once

#include <cstddef>
#include <cstdint>

namespace torch_abi
{

// at::RecordFunction. Offset 0 is the vptr.
inline constexpr std::size_t kRecordFunctionSize         = 344;
inline constexpr std::size_t kRecordFunctionAlignment    = 8;
inline constexpr std::size_t kRecordFunctionScopeOff     = 96;
inline constexpr std::size_t kRecordFunctionSeqNrOff     = 200;
inline constexpr std::size_t kRecordFunctionInputsOff    = 208;
inline constexpr std::size_t kRecordFunctionFwdThreadOff = 304;

// at::RecordScope::NUM_SCOPES, the length of RecordFunctionCallback::scopes_.
inline constexpr std::size_t kScopeCount = 10;

// at::RecordFunctionCallback is passed by value to at::addGlobalCallback().
inline constexpr std::size_t kCallbackSize           = 40;
inline constexpr std::size_t kCallbackAlignment      = 8;
inline constexpr std::size_t kCallbackStartOff       = 0;
inline constexpr std::size_t kCallbackEndOff         = 8;
inline constexpr std::size_t kCallbackProbabilityOff = 16;
inline constexpr std::size_t kCallbackScopesOff      = 24;
inline constexpr std::size_t kCallbackNeedsInputsOff = 34;

inline constexpr std::size_t kObserverContextSize      = 8;
inline constexpr std::size_t kObserverContextAlignment = 8;

inline constexpr std::size_t kDebugInfoKindSize        = 8;
inline constexpr std::size_t kDebugInfoKindAlignment   = 8;
inline constexpr std::size_t kDebugInfoBaseSize        = 8;
inline constexpr std::size_t kDebugInfoBaseAlignment   = 8;
inline constexpr std::size_t kSharedDebugInfoSize      = 16;
inline constexpr std::size_t kSharedDebugInfoAlignment = 8;

// c10::IValue stores an at::Tensor directly in its payload for the Tensor tag.
inline constexpr std::size_t   kIValueSize          = 16;
inline constexpr std::size_t   kIValueAlignment     = 8;
inline constexpr std::size_t   kIValuePayloadOff    = 0;
inline constexpr std::size_t   kIValueTagOff        = 8;
inline constexpr std::uint32_t kIValueTensorTag     = 1;
inline constexpr std::size_t   kIValueKnownTagCount = 28;
inline constexpr std::size_t   kTensorSize          = 8;

// c10::detail::ListImpl's std::vector<IValue> begin/end pointers (libstdc++).
inline constexpr std::size_t kListImplElementsBeginOff = 16;
inline constexpr std::size_t kListImplElementsEndOff   = 24;

inline constexpr std::size_t kOperatorNameSize              = 64;
inline constexpr std::size_t kOperatorNameAlignment         = 8;
inline constexpr std::size_t kOperatorNameNameOff           = 0;
inline constexpr std::size_t kOperatorNameOverloadNameOff   = 32;
inline constexpr std::size_t kOptionalOperatorNameSize      = 72;
inline constexpr std::size_t kOptionalOperatorNameAlignment = 8;

inline constexpr std::size_t kOperatorHandleSize          = 16;
inline constexpr std::size_t kOperatorHandleAlignment     = 8;
inline constexpr std::size_t kOperatorHandleDefinitionOff = 0;
inline constexpr std::size_t kOperatorHandleIteratorOff   = 8;

// Dispatcher registration listener vptr and returned std::function owner.
inline constexpr std::size_t kOpRegistrationListenerSize      = 8;
inline constexpr std::size_t kOpRegistrationListenerAlignment = 8;
inline constexpr std::size_t kRegistrationHandleSize          = 32;
inline constexpr std::size_t kRegistrationHandleAlignment     = 8;

inline constexpr std::size_t kInputArrayViewSize      = 16;
inline constexpr std::size_t kInputArrayViewAlignment = 8;

inline constexpr std::size_t  kKnownScalarTypeCount = 47;
inline constexpr std::int32_t kBComplex32ScalarType = 46;

// AOTI dtype availability differs between the supported PyTorch minors.
inline constexpr std::uint64_t kTorch214AotiAbi = 0x020e000000000000ULL;

}  // namespace torch_abi
