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

#include <hsakmt/hsakmttypes.h>
#if defined(__has_include)
#    if __has_include(<hsakmt/rocdxg_version.h>)
#        include <hsakmt/rocdxg_version.h>
#    endif
#endif

#include <cstdint>
#include <functional>
#include <string>
#include <vector>

#define ROCPROFILER_ROCDXG_STR2(v) #v
#define ROCPROFILER_ROCDXG_STR(v)  ROCPROFILER_ROCDXG_STR2(v)

namespace rocprofiler
{
namespace platform
{
namespace wsl
{
// Spelled the way ROCr's own thunk_loader.h spells them, so the two describe the same
// exports in the same terms.
using PFN_hsaKmtOpenKFD                 = HSAKMT_STATUS (*)();
using PFN_hsaKmtAcquireSystemProperties = HSAKMT_STATUS (*)(HsaSystemProperties*);
using PFN_hsaKmtGetNodeProperties       = HSAKMT_STATUS (*)(HSAuint32, HsaNodeProperties*);
using PFN_hsaKmtReleaseSystemProperties = HSAKMT_STATUS (*)();
using PFN_hsaKmtCloseKFD                = HSAKMT_STATUS (*)();
using PFN_DxgAbiCheck                   = HSAKMT_STATUS (*)(HsaStructureSizes*);

// The names the HSA runtime's ThunkLoader tries, in the order it tries them
// (thunk_loader_linux.cpp): the versioned soname first, then the unversioned
// development name. The major comes from the hsakmt rocdxg_version.h that the
// thunk's own soname is derived from, never from a literal, so the two cannot
// drift apart. Headers that predate rocdxg_version.h leave the versioned name
// empty and only the unversioned name is tried.
#if defined(ROCDXG_LIB_VERSION_MAJOR)
inline constexpr const char* kLibRocdxgVersionedName =
    "librocdxg.so." ROCPROFILER_ROCDXG_STR(ROCDXG_LIB_VERSION_MAJOR);
#else
inline constexpr const char* kLibRocdxgVersionedName = "";
#endif
inline constexpr const char* kLibRocdxgUnversionedName = "librocdxg.so";

// Every entry point the topology read needs, all of them long-standing KMT
// exports. Requiring all five before any of them is called is what keeps the
// read from having to unwind a half-open thunk.
//
// Released external librocdxg packages keep one global snapshot and no count
// of who holds it, so a release drops it for every consumer. The prerequisite
// in-tree runtime refcounts that snapshot, but no entry point in this table
// reports which ownership model a loaded thunk implements. What makes this read
// safe with an older external package is ordering rather than sharing - it runs
// from a library constructor, before hsa_init(). Attaching to an already-running
// process has no such ordering, so the explicit rocprofiler-register attach
// marker disables WSL enumeration before this table is resolved or called.
// PR #10034 is required to enable WSL late attach, but it is not a prerequisite
// for constructor-ordered enumeration in PR #7016; see dxg_topology.cpp.
//
// abi_check is the sixth and is deliberately outside complete(): it is the
// structure-size handshake, and a thunk built before the handshake existed does
// not export it. Requiring it would refuse thunks that work. It is not a
// capability bit for anything else either - the released packages export it and
// the in-tree build does not, which says which librocdxg is installed and
// nothing about how it manages the snapshot.
struct DxgThunk
{
    PFN_hsaKmtOpenKFD                 open_kfd         = nullptr;
    PFN_hsaKmtAcquireSystemProperties acquire_snapshot = nullptr;
    PFN_hsaKmtGetNodeProperties       get_node         = nullptr;
    PFN_hsaKmtReleaseSystemProperties release_snapshot = nullptr;
    PFN_hsaKmtCloseKFD                close_kfd        = nullptr;
    PFN_DxgAbiCheck                   abi_check        = nullptr;

    bool complete() const
    {
        return open_kfd != nullptr && acquire_snapshot != nullptr && get_node != nullptr &&
               release_snapshot != nullptr && close_kfd != nullptr;
    }
};

// The dynamic-loader calls RocdxgHandle makes, as data. Defaults to dlopen and
// friends; a test substitutes its own to observe the open order and the
// open/close balance.
//
// hsa_runtime_dir answers the directory, with a trailing '/', of the
// libhsa-runtime64 already mapped into this process, or an empty string if
// none is. It is where the HSA runtime itself falls back to looking for the
// thunk, so a versioned-only package layout that the global search cannot see
// still resolves to the copy the HSA runtime will use.
struct DxgLoaderOps
{
    std::function<void*(const char*, int)>   open;
    std::function<void*(void*, const char*)> sym;
    std::function<int(void*)>                close;
    std::function<const char*()>             error;
    std::function<std::string()>             hsa_runtime_dir;
};

const DxgLoaderOps&
default_loader_ops();

// Resolve every entry point out of an already-open handle, warning about each
// required one the object does not export. A missing abi_check is not warned
// about: it means an older thunk, not a broken one.
DxgThunk
resolve_dxg_thunk(void* handle, const DxgLoaderOps& ops);

// Read the GPU nodes through an already-resolved thunk: open, acquire
// snapshot, per-node read, then release and close on the way out of every path
// including the early ones. Returns an empty vector for anything it cannot
// read; nothing here is fatal.
std::vector<DxgNode>
read_dxg_gpu_topology(const DxgThunk& thunk);

// The same, preceded by loading and resolving librocdxg through `ops`. A copy
// already resident under either name is reused before anything is loaded; a
// cold process then loads the first of the versioned name, the versioned name
// beside libhsa-runtime64, the unversioned name and the unversioned name beside
// libhsa-runtime64 - the HSA runtime's own order, so the copy loaded here is
// the one it will later find by soname instead of a second one. The handle is
// closed on the way out of every path. read_dxg_gpu_topology() is this with
// default_loader_ops().
std::vector<DxgNode>
read_dxg_gpu_topology(const DxgLoaderOps& ops);
}  // namespace wsl
}  // namespace platform
}  // namespace rocprofiler
