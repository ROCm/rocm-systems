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

// Pluggable D3DKMT loader. This header deliberately exposes ONLY plain integer
// structs and free functions so the rest of the native-Windows platform tree
// (agent.cpp in particular) never has to include the vendored WDK headers or
// <windows.h>. The .cpp is the single translation unit that pulls in
// d3dkmthk.h and resolves the __stdcall/NTAPI D3DKMT entry points.
//
// Design mirrors the WSL DxcoreHandle: a loader object resolves the module +
// symbols at construction and reports readiness via ready(); the difference is
// that the D3DKMT structs are consumed inside the .cpp and the results are
// handed back to callers as the neutral structs declared below.

#include <cstdint>
#include <vector>

namespace rocprofiler
{
namespace platform
{
namespace windows
{
// Neutral copy of a single enumerated adapter (subset of D3DKMT_ADAPTERINFO).
// hadapter is the D3DKMT_HANDLE the caller passes back into the query/close
// helpers below.
struct d3dkmt_adapter
{
    uint32_t hadapter  = 0;  // D3DKMT_HANDLE
    uint32_t luid_low  = 0;  // LUID.LowPart
    int32_t  luid_high = 0;  // LUID.HighPart
};

// Neutral copy of D3DKMT_DEVICE_IDS (KMTQAITYPE_PHYSICALADAPTERDEVICEIDS).
struct d3dkmt_device_ids
{
    uint32_t vendor_id    = 0;
    uint32_t device_id    = 0;
    uint32_t subvendor_id = 0;
    uint32_t subsystem_id = 0;
    uint32_t revision_id  = 0;
    uint32_t bus_type     = 0;
};

// Neutral copy of the PCI bus address (KMTQAITYPE_ADAPTERADDRESS).
struct d3dkmt_adapter_address
{
    uint32_t bus_number      = 0;
    uint32_t device_number   = 0;
    uint32_t function_number = 0;
};

// Neutral copy of the segment-size query (KMTQAITYPE_GETSEGMENTSIZE).
struct d3dkmt_segment_size
{
    uint64_t dedicated_video_memory  = 0;
    uint64_t dedicated_system_memory = 0;
    uint64_t shared_system_memory    = 0;
};

// Owning handle for the loaded module (gdi32.dll or an override) plus the
// resolved D3DKMT* symbols. Mirrors WSL DxcoreHandle. The destructor only
// FreeLibrary()s the module when this loader actually LoadLibraryW'd it; a
// module that was already resident (resolved via GetModuleHandleW) is left
// untouched, per the "never FreeLibrary a module you didn't load" rule.
class d3dkmt_loader
{
public:
    d3dkmt_loader();
    ~d3dkmt_loader();

    d3dkmt_loader(const d3dkmt_loader&) = delete;
    d3dkmt_loader& operator=(const d3dkmt_loader&) = delete;
    d3dkmt_loader(d3dkmt_loader&&)                 = delete;
    d3dkmt_loader& operator=(d3dkmt_loader&&) = delete;

    // True iff the module loaded and all three required D3DKMT symbols
    // resolved. Mirrors WSL DxcoreHandle::ready().
    bool ready() const;

    // Two-pass D3DKMTEnumAdapters3 (count then fill) with
    // Filter.IncludeComputeOnly=1. Returns the enumerated adapters; an empty
    // vector means either zero adapters or a failed call (logged internally).
    std::vector<d3dkmt_adapter> enumerate_adapters() const;

    // Per-adapter queries. Each returns true on STATUS_SUCCESS and fills out;
    // false (with an internal INFO log) otherwise. The caller owns closing the
    // adapter via close_adapter() on every exit path.
    bool query_device_ids(uint32_t hadapter, d3dkmt_device_ids& out) const;
    bool query_adapter_address(uint32_t hadapter, d3dkmt_adapter_address& out) const;
    bool query_segment_size(uint32_t hadapter, d3dkmt_segment_size& out) const;

    // Closes a previously enumerated adapter handle. Safe to call on any
    // handle returned by enumerate_adapters().
    void close_adapter(uint32_t hadapter) const;

private:
    struct impl_t;
    impl_t* m_impl = nullptr;
};
}  // namespace windows
}  // namespace platform
}  // namespace rocprofiler
