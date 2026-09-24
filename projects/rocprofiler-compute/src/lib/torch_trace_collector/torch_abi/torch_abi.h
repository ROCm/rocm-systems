// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier:  MIT
//
// Measured layouts for the exact PyTorch runtime artifacts accepted by
// runtime.cpp. PyTorch does not guarantee this C++ ABI.

#pragma once

#include <array>
#include <cstddef>

namespace torch_abi
{

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

inline constexpr std::size_t kGnuBuildIdSize = 20;
using GnuBuildId                             = std::array<unsigned char, kGnuBuildIdSize>;

struct SupportedRuntime
{
    GnuBuildId torch_cpu_build_id;
    GnuBuildId c10_build_id;
};

inline constexpr std::array<SupportedRuntime, 2> kSupportedRuntimes = {{
    // Official CPU PyTorch 2.13 wheel.
    {{{0x8e, 0xc0, 0x8e, 0xc8, 0xf7, 0x1d, 0xe0, 0x4e, 0xe2, 0xba,
       0xa4, 0x6c, 0x0d, 0xbe, 0x26, 0x28, 0x58, 0xb1, 0xe2, 0x7c}},
     {{0x53, 0xf5, 0xa9, 0x54, 0xf0, 0x2f, 0xb0, 0xf8, 0x58, 0x5f,
       0x56, 0xce, 0xf6, 0x97, 0xd4, 0x80, 0x4f, 0xa7, 0x9b, 0xb9}}},
    // AMD ROCm PyTorch 2.14 nightly wheel from 2026-09-17.
    {{{0x7d, 0x0a, 0x1c, 0x66, 0xc6, 0x43, 0xd0, 0x33, 0x44, 0x9d,
       0x60, 0xa2, 0xd0, 0xbf, 0xb4, 0xbb, 0x86, 0x9a, 0xb1, 0x2d}},
     {{0x6e, 0xba, 0x1a, 0xdc, 0xdf, 0xcb, 0xbb, 0xb9, 0x3b, 0x69,
       0x96, 0x68, 0x90, 0xd0, 0x36, 0x5f, 0x5e, 0x2c, 0x4b, 0xf4}}},
}};

}  // namespace torch_abi
