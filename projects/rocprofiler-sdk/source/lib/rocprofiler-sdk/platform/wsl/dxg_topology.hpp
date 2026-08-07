// MIT License
//
// Copyright (c) 2023-2026 Advanced Micro Devices, Inc. All rights reserved.
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
// THE SOFTWARE.

#pragma once

#include <rocprofiler-sdk/agent.h>

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace rocprofiler
{
namespace platform
{
namespace wsl
{
// === librocdxg KMT topology ABI ===
//
// On WSL the KMT topology lives behind librocdxg (the DXG thunk), the same
// component the HSA runtime loads for /dev/dxg systems. rocprofiler-sdk reads
// it directly so that agent records are complete before they are published,
// instead of being refined once the HSA runtime happens to come up.
//
// The declarations below mirror the ABI published by
// rocr-runtime/libhsakmt/include/hsakmt/hsakmt_dxg.h. They are duplicated
// rather than included because rocprofiler-sdk builds standalone against an
// installed ROCm and cannot require an hsakmt header revision that may be
// older than the thunk it dlopens at runtime. Duplication is safe here only
// because this ABI is small, append-only, and handshaked at run time
// (DxgNodeTopology::StructSize / AbiVersion); the full HsaNodeProperties
// layout is deliberately NOT duplicated.

// HSAKMT_STATUS values this file cares about.
inline constexpr int32_t kHsaKmtStatusSuccess             = 0;
inline constexpr int32_t kHsaKmtStatusKernelAlreadyOpened = 22;

inline constexpr uint32_t kDxgNodeTopologyAbiVersion = 1;
inline constexpr uint32_t kDxgNodeTopologyMinSize    = 8;

// Mirror of HsaStructureSizes. DxgAbiCheck only reads it - and validates its
// size before reading anything else - so this cannot be overrun by the thunk.
struct DxgStructureSizes
{
    uint16_t StructureSizes;
    uint16_t SizeOfHsaNodeProperties;
    uint16_t SizeOfHsaExternalHandleDesc;
    uint16_t Reserved[5];
};

static_assert(sizeof(DxgStructureSizes) == 16, "HsaStructureSizes ABI mismatch");

// Mirror of HsaDxgNodeTopology.
struct DxgNodeTopology
{
    uint32_t StructSize;
    uint32_t AbiVersion;

    uint32_t NumCPUCores;
    uint32_t NumFComputeCores;
    uint32_t NumSIMDPerCU;
    uint32_t NumShaderBanks;
    uint32_t NumArrays;
    uint32_t NumCUPerArray;
    uint32_t WaveFrontSize;
    uint32_t MaxWavesPerSIMD;
    uint32_t MaxSlotsScratchCU;
    uint32_t LDSSizeInKB;
    uint32_t GDSSizeInKB;
    uint32_t NumGws;
    uint32_t NumXcc;
    uint32_t NumSdmaEngines;
    uint32_t NumSdmaXgmiEngines;
    uint32_t NumSdmaQueuesPerEngine;
    uint32_t NumCpQueues;
    uint32_t MaxEngineClockMhzFCompute;
    uint32_t MaxEngineClockMhzCCompute;

    uint32_t EngineIdMajor;
    uint32_t EngineIdMinor;
    uint32_t EngineIdStepping;
    uint32_t EngineIdUCode;
    uint32_t OverrideEngineIdMajor;
    uint32_t OverrideEngineIdMinor;
    uint32_t OverrideEngineIdStepping;
    uint32_t SdmaUCode;

    uint32_t Capability;
    uint32_t FamilyID;
    uint32_t Domain;
    uint32_t LocationId;
    uint32_t VendorId;
    uint32_t DeviceId;
    uint32_t KFDGpuID;
    uint32_t LuidLowPart;
    uint32_t LuidHighPart;
    uint32_t Integrated;
    uint32_t Reserved32;

    uint64_t UniqueID;
    uint64_t HiveID;
    uint64_t LocalMemSize;
    uint64_t WallClockKHz;
};

static_assert(sizeof(DxgNodeTopology) == 192, "HsaDxgNodeTopology ABI mismatch");
static_assert(offsetof(DxgNodeTopology, NumCPUCores) == kDxgNodeTopologyMinSize,
              "HsaDxgNodeTopology ABI mismatch");
static_assert(offsetof(DxgNodeTopology, Reserved32) == 156, "HsaDxgNodeTopology ABI mismatch");
static_assert(offsetof(DxgNodeTopology, UniqueID) == 160, "HsaDxgNodeTopology ABI mismatch");
static_assert(offsetof(DxgNodeTopology, WallClockKHz) == 184, "HsaDxgNodeTopology ABI mismatch");

// Every GPU node the thunk reported, in KMT node order. CPU-only nodes are
// dropped: the agent enumerator pairs these against DXCore adapters.
std::vector<DxgNodeTopology>
read_dxg_gpu_topology();

// Find the KMT node describing a DXCore adapter.
//
// LUID is the multi-GPU-safe key: the thunk reports the same Windows adapter
// LUID that D3DKMTQueryAdapterInfo returns. Only when one of the two sides does
// not carry a LUID does this fall back to the PCI device id, which is ambiguous
// across identical GPUs and therefore never used when a LUID pair is available.
const DxgNodeTopology*
match_node_to_adapter(const std::vector<DxgNodeTopology>& nodes,
                      uint32_t                            luid_low,
                      int32_t                             luid_high,
                      uint32_t                            device_id);

// Resolve the gfx target name for a node, or an empty string if the node does
// not report one.
//
// The thunk reports HSA_OVERRIDE_GFX_VERSION through OverrideEngineId and
// leaves EngineId's version fields zero in that case, which is the precedence
// the HSA runtime applies too (see amd_gpu_agent.cpp). An explicit, valid
// ROCPROFILER_FORCE_GFX wins over both, so a user can select the counter
// definitions for a target rocprofiler-sdk does not know about.
std::string
resolve_gfx_name(const DxgNodeTopology& node);

// Copy a node's topology and identity into an agent record. Returns false, and
// leaves the record untouched, when the node cannot describe a publishable GPU:
// counter collection divides by these values, so a partially described GPU is
// omitted rather than published with invented ones.
bool
apply_node_topology(const DxgNodeTopology& node, rocprofiler_agent_t& info);
}  // namespace wsl
}  // namespace platform
}  // namespace rocprofiler
