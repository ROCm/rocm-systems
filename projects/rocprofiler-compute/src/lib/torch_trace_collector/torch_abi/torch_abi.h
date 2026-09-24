// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier:  MIT

#pragma once

#include <cstddef>

namespace torch_abi
{

// Byte offsets and sizes of the PyTorch types the collector reads. The
// collector builds without PyTorch headers, so inline accessors such as
// RecordFunction::scope() have no symbol to call and are read positionally
// instead. These cover PyTorch 2.13 and 2.14; test_torch_trace_collector.cpp
// checks every value against the real headers.
inline constexpr std::size_t kRecordFunctionSize         = 344;
inline constexpr std::size_t kRecordFunctionAlignment    = 8;
inline constexpr std::size_t kRecordFunctionScopeOff     = 96;
inline constexpr std::size_t kRecordFunctionSeqNrOff     = 200;
inline constexpr std::size_t kRecordFunctionFwdThreadOff = 304;

inline constexpr std::size_t kScopeCount = 10;

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
inline constexpr std::size_t kDebugInfoGuardSize       = 24;
inline constexpr std::size_t kDebugInfoGuardAlignment  = 8;
inline constexpr std::size_t kSharedDebugInfoSize      = 16;
inline constexpr std::size_t kSharedDebugInfoAlignment = 8;

}  // namespace torch_abi
