// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier:  MIT
//
// Layout of the PyTorch types the collector touches, for the stub headers
// beside this file. Verified identical in PyTorch 2.12, 2.13 and 2.14, though
// only 2.13 and 2.14 are advertised below.
//
// Keep this list short. Every entry is a version-specific fact; anything that
// can be resolved at runtime belongs in user_scope_kind.h instead.

#pragma once

#include <cstddef>
#include <cstdint>

namespace torch_abi
{

inline constexpr const char* kSupportedTorchVersions[] = {"2.13", "2.14"};

// at::RecordFunction. Offset 0 is the vptr.
inline constexpr std::size_t kRecordFunctionSize         = 344;
inline constexpr std::size_t kRecordFunctionSeqNrOff     = 200;
inline constexpr std::size_t kRecordFunctionFwdThreadOff = 304;

// at::RecordScope::NUM_SCOPES, the length of RecordFunctionCallback::scopes_.
inline constexpr std::size_t kScopeCount = 10;

// Types the collector passes or allocates whole.
inline constexpr std::size_t kCallbackSize       = 40;
inline constexpr std::size_t kDebugInfoGuardSize = 24;
inline constexpr std::size_t kDebugInfoKindSize  = 8;

// c10::DebugInfoKind::TEST_INFO_2. Only sent to a PyTorch 2.12 libc10, where
// slots are a fixed enum and none is free. A collision costs overlay frames.
inline constexpr std::uint64_t kLegacyUserScopeKind = 6;

}  // namespace torch_abi
