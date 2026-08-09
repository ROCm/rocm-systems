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

// Internal seam between "how librocdxg is brought into the process" and "what
// rocprofiler-sdk asks it for". Neither this header nor dxg_topology.hpp is
// installed; both are lib-private. Splitting them lets the topology read be
// driven by a fake function table in a unit test, with no dlopen, no thunk, no
// HSA runtime and no GPU. read_dxg_gpu_topology() in dxg_topology.hpp remains
// the only entry point the rest of the SDK uses.

#include "lib/rocprofiler-sdk/platform/wsl/dxg_topology.hpp"

#include <array>
#include <cstdint>
#include <functional>
#include <vector>

namespace rocprofiler
{
namespace platform
{
namespace wsl
{
using PFN_DxgGetNodeTopology         = int32_t (*)(uint32_t, uint32_t, DxgNodeTopology*);
using PFN_DxgAcquireTopologySnapshot = int32_t (*)(uint32_t*);
using PFN_DxgReleaseTopologySnapshot = int32_t (*)();
using PFN_hsaKmtOpenKFD              = int32_t (*)();
using PFN_hsaKmtCloseKFD             = int32_t (*)();

// The unversioned soname, matching ThunkLoader::whoami() in the HSA runtime.
// Never a versioned name (librocdxg.so.1 / .so.7): hard-coding a soversion
// would silently stop resolving the very object the HSA runtime has loaded,
// and the soversion is not what decides compatibility anyway - every
// DxgGetNodeTopology reply carries its own size and ABI version.
inline constexpr const char* kLibRocdxgSoname = "librocdxg.so";

// Every entry point the topology read needs. The librocdxg that ships today
// exports hsaKmtOpenKFD and hsaKmtCloseKFD but none of the Dxg* topology
// calls, so all five are required before anything is called.
struct DxgThunk
{
    PFN_DxgGetNodeTopology         get_node         = nullptr;
    PFN_DxgAcquireTopologySnapshot acquire_snapshot = nullptr;
    PFN_DxgReleaseTopologySnapshot release_snapshot = nullptr;
    PFN_hsaKmtOpenKFD              open_kfd         = nullptr;
    PFN_hsaKmtCloseKFD             close_kfd        = nullptr;

    bool complete() const
    {
        return get_node != nullptr && acquire_snapshot != nullptr && release_snapshot != nullptr &&
               open_kfd != nullptr && close_kfd != nullptr;
    }
};

// The dynamic-loader calls RocdxgHandle makes, as data. Defaults to dlopen and
// friends; a test substitutes its own to observe the open/close balance.
struct DxgLoaderOps
{
    std::function<void*(const char*, int)>   open;
    std::function<void*(void*, const char*)> sym;
    std::function<int(void*)>                close;
    std::function<const char*()>             error;
};

const DxgLoaderOps&
default_loader_ops();

// Resolve every entry point out of an already-open handle, warning about each
// one the object does not export.
DxgThunk
resolve_dxg_thunk(void* handle, const DxgLoaderOps& ops);

// Read the GPU nodes through an already-resolved thunk: open, acquire
// snapshot, per-node read, then release and close on the way out of every path
// including the early ones. Every record is checked against the ABI version
// and size this build was compiled for before it is trusted. Returns an empty
// vector for anything it refuses to trust; nothing here is fatal.
std::vector<DxgNodeTopology>
read_dxg_gpu_topology(const DxgThunk& thunk);

// The same, preceded by loading and resolving librocdxg through `ops`. The
// handle is closed on the way out of every path. read_dxg_gpu_topology() is
// this with default_loader_ops().
std::vector<DxgNodeTopology>
read_dxg_gpu_topology(const DxgLoaderOps& ops);
}  // namespace wsl
}  // namespace platform
}  // namespace rocprofiler
