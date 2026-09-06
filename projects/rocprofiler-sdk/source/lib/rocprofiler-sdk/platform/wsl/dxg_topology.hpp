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

#include <hsakmt/hsakmttypes.h>

#include <cstdint>
#include <set>
#include <string>
#include <vector>

namespace rocprofiler
{
namespace platform
{
namespace wsl
{
// === Reading the KMT topology on WSL ===
//
// On WSL the KMT topology lives behind librocdxg (the DXG thunk), the same
// component the HSA runtime loads for /dev/dxg systems. rocprofiler-sdk reads
// it directly so that agent records are complete before they are published,
// instead of being refined once the HSA runtime happens to come up.
//
// The read goes through the thunk's ordinary KMT entry points -
// hsaKmtGetNodeProperties() and the snapshot pair around it - and the records
// are the driver's own HsaNodeProperties. librocdxg is resolved with dlopen
// rather than linked, because on WSL it is the object the HSA runtime has
// already loaded and there is nothing to link against before that happens.
//
// HsaNodeProperties has no stable layout across ROCm packages: it has grown
// (364 bytes to 396) and has also been rearranged at an unchanged size, and
// hsaKmtGetNodeProperties() takes no size argument, so the callee writes
// sizeof() bytes as *it* knows the type. Reading it through a dynamically
// resolved thunk therefore assumes the thunk and this build come from
// compatible ROCr/hsakmt package layouts - which is what shipping librocdxg in
// the ROCr package makes true, since the hsakmt headers this build compiles
// against come from that same package.
//
// What is checked at run time is that the object loads, that it exports every
// entry point the read needs, and - through DxgAbiCheck, the hsakmt
// structure-size handshake the HSA runtime also performs - that the thunk
// agrees on sizeof(HsaNodeProperties). A thunk that disagrees is refused rather
// than read. A thunk too old to export the handshake is still read, and a
// rearrangement that left sizeof() alone still cannot be detected from here, so
// the read is bounded by a destination larger than the record to keep the
// undetectable cases from becoming memory corruption.

// One KMT node as this process read it.
//
// Internal bookkeeping, not an interface: nothing outside this library sees it
// and nothing serializes it. The node id is carried alongside the properties
// because hsaKmtGetNodeProperties() does not echo back the node it described,
// and the enumerator publishes the driver's node id rather than inventing an
// ordinal for it.
struct DxgNode
{
    uint32_t          node_id = 0;
    HsaNodeProperties props   = {};
};

// True only while the SDK is being initialized through rocprof-attach.
// rocprofiler-register sets this explicit marker before invoking tool
// registration; it is not inferred from any librocdxg state.
bool
is_late_attach_mode();

// Every GPU node the thunk reported, in KMT node order. CPU-only nodes are
// dropped: the agent enumerator pairs these against DXCore adapters.
std::vector<DxgNode>
read_dxg_gpu_topology();

// Outcome of pairing one DXCore adapter with a KMT topology node.
//
// `ambiguous` distinguishes "this adapter has no node" from "several nodes
// could be this adapter and nothing can tell them apart". The two need
// different diagnostics, and the second must never be resolved by picking one:
// on a machine with two identical GPUs that would attribute one GPU's counters
// to the other.
struct NodeMatch
{
    const DxgNode* node      = nullptr;
    bool           ambiguous = false;
};

// Find the KMT node describing a DXCore adapter.
//
// LUID is the multi-GPU-safe key: the thunk reports the same Windows adapter
// LUID that D3DKMTQueryAdapterInfo returns, so when both sides carry one the
// pairing is exact. The PCI device id is only a fallback for the nodes a LUID
// cannot speak for, and it has to identify exactly one of them.
//
// `consumed_node_ids` holds the node id of every node already claimed by an
// earlier adapter. Matching is otherwise stateless, so without it two adapters
// would happily claim the same node.
NodeMatch
match_node_to_adapter(const std::vector<DxgNode>& nodes,
                      const std::set<uint32_t>&   consumed_node_ids,
                      uint32_t                    luid_low,
                      int32_t                     luid_high,
                      uint32_t                    device_id);

// Resolve the gfx target name for a node, or an empty string if the node does
// not report one.
//
// The thunk reports HSA_OVERRIDE_GFX_VERSION through OverrideEngineId and
// leaves EngineId's version fields zero in that case, which is the precedence
// the HSA runtime applies too (see amd_gpu_agent.cpp). An explicit, valid
// ROCPROFILER_FORCE_GFX wins over both, so a user can select the counter
// definitions for a target rocprofiler-sdk does not know about.
std::string
resolve_gfx_name(const HsaNodeProperties& props);

// Copy a node's topology and identity into an agent record. Returns false, and
// leaves the record untouched, when the node cannot describe a publishable GPU:
// counter collection divides by these values, so a partially described GPU is
// omitted rather than published with invented ones.
bool
apply_node_topology(const DxgNode& node, rocprofiler_agent_t& info);
}  // namespace wsl
}  // namespace platform
}  // namespace rocprofiler
