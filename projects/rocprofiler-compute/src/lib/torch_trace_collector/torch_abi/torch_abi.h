// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier:  MIT
//
// Layout of the PyTorch types the collector touches, for the stub headers
// beside this file. Verified identical in the measured CPU PyTorch 2.13/2.14
// and ROCm PyTorch 2.14 artifacts. PyTorch does not guarantee this ABI.
//
// Keep this list short. Every entry is a version-specific fact.

#pragma once

#include <array>
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

inline constexpr std::size_t kInputArrayViewSize      = 16;
inline constexpr std::size_t kInputArrayViewAlignment = 8;

inline constexpr std::size_t  kKnownScalarTypeCount = 47;
inline constexpr std::int32_t kBComplex32ScalarType = 46;

// This is the authoritative native layout gate. It includes standalone CPU
// libtorch archives used by direct C++ callers and validation tests. The
// Python loader's narrower allowlist contains only complete wheel identities
// validated through that loader, so the two lists are intentionally not
// one-to-one.
inline constexpr std::uint64_t kTorch213AotiAbi = 0x020d000000000000ULL;
inline constexpr std::uint64_t kTorch214AotiAbi = 0x020e000000000000ULL;

inline constexpr std::size_t kGnuBuildIdSize = 20;
using GnuBuildId                             = std::array<unsigned char, kGnuBuildIdSize>;

struct SupportedRuntime
{
    GnuBuildId    torch_cpu_build_id;
    GnuBuildId    c10_build_id;
    std::uint64_t aoti_abi;
};

inline constexpr std::array<SupportedRuntime, 4> kSupportedRuntimes = {{
    // Official CPU PyTorch 2.13 archive.
    {{{0x82, 0x4f, 0x5c, 0xb3, 0xca, 0x50, 0x3d, 0x2e, 0x46, 0xd0,
       0x3b, 0x04, 0xe3, 0x7b, 0x97, 0xf9, 0x7b, 0x82, 0x04, 0x6f}},
     {{0x53, 0xf5, 0xa9, 0x54, 0xf0, 0x2f, 0xb0, 0xf8, 0x58, 0x5f,
       0x56, 0xce, 0xf6, 0x97, 0xd4, 0x80, 0x4f, 0xa7, 0x9b, 0xb9}},
     kTorch213AotiAbi},
    // Official CPU PyTorch 2.13 wheel.
    {{{0x8e, 0xc0, 0x8e, 0xc8, 0xf7, 0x1d, 0xe0, 0x4e, 0xe2, 0xba,
       0xa4, 0x6c, 0x0d, 0xbe, 0x26, 0x28, 0x58, 0xb1, 0xe2, 0x7c}},
     {{0x53, 0xf5, 0xa9, 0x54, 0xf0, 0x2f, 0xb0, 0xf8, 0x58, 0x5f,
       0x56, 0xce, 0xf6, 0x97, 0xd4, 0x80, 0x4f, 0xa7, 0x9b, 0xb9}},
     kTorch213AotiAbi},
    // Official CPU PyTorch 2.14 archive.
    {{{0xb6, 0x0a, 0x0f, 0xc5, 0x53, 0x59, 0xe5, 0x0b, 0x75, 0xdc,
       0x96, 0x06, 0xf6, 0x22, 0xeb, 0x18, 0x61, 0xc9, 0x04, 0xc3}},
     {{0x49, 0xfb, 0xcd, 0xba, 0x07, 0x46, 0xa9, 0xf6, 0x7c, 0x13,
       0xa1, 0x69, 0x3b, 0xa2, 0xe0, 0xe8, 0xd6, 0x57, 0xe5, 0x6a}},
     kTorch214AotiAbi},
    // AMD ROCm PyTorch 2.14 nightly wheel from 2026-09-17.
    {{{0x7d, 0x0a, 0x1c, 0x66, 0xc6, 0x43, 0xd0, 0x33, 0x44, 0x9d,
       0x60, 0xa2, 0xd0, 0xbf, 0xb4, 0xbb, 0x86, 0x9a, 0xb1, 0x2d}},
     {{0x6e, 0xba, 0x1a, 0xdc, 0xdf, 0xcb, 0xbb, 0xb9, 0x3b, 0x69,
       0x96, 0x68, 0x90, 0xd0, 0x36, 0x5f, 0x5e, 0x2c, 0x4b, 0xf4}},
     kTorch214AotiAbi},
}};

}  // namespace torch_abi
