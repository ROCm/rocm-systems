// MIT License
//
// Copyright (c) 2023-2025 Advanced Micro Devices, Inc. All rights reserved.
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

#include "lib/rocprofiler-sdk/topology/dxg_provider.hpp"

#include "lib/common/logging.hpp"

#include <dlfcn.h>

namespace rocprofiler
{
namespace topology
{
struct DxgProvider::Impl
{
    void* dxcore_handle  = nullptr;
    void* create_factory = nullptr;  // CreateDXCoreAdapterFactory fn ptr
    // ... cached IDXCoreAdapterFactory* once we call it
};

namespace
{
bool
probe_libdxcore()
{
    static const bool _v = []() {
        void* h = ::dlopen("libdxcore.so", RTLD_NOW | RTLD_LOCAL);
        if(!h)
        {
            ROCP_INFO << "DxgProvider: dlopen(libdxcore.so) failed: " << ::dlerror();
            return false;
        }
        void* sym = ::dlsym(h, "CreateDXCoreAdapterFactory");
        if(!sym)
        {
            ROCP_INFO << "DxgProvider: dlsym(CreateDXCoreAdapterFactory) failed";
            ::dlclose(h);
            return false;
        }
        ::dlclose(h);  // re-open in DxgProvider() ctor when actually used
        return true;
    }();
    return _v;
}
}  // namespace

bool
DxgProvider::is_available()
{
    return probe_libdxcore();
}

DxgProvider::DxgProvider()
: m_impl(std::make_unique<Impl>())
{
    m_impl->dxcore_handle = ::dlopen("libdxcore.so", RTLD_NOW | RTLD_LOCAL);
    if(m_impl->dxcore_handle)
        m_impl->create_factory = ::dlsym(m_impl->dxcore_handle, "CreateDXCoreAdapterFactory");
}

DxgProvider::~DxgProvider()
{
    if(m_impl && m_impl->dxcore_handle) ::dlclose(m_impl->dxcore_handle);
}

std::vector<unique_agent_t>
DxgProvider::enumerate()
{
    std::vector<unique_agent_t> out;
    if(!m_impl || !m_impl->create_factory)
    {
        ROCP_WARNING << "DxgProvider: libdxcore.so not available; returning empty topology";
        return out;
    }

    // TODO(wsl-dxcore): full DXCore enumeration.
    //   1. CreateDXCoreAdapterFactory(IID_IDXCoreAdapterFactory, &factory)
    //   2. factory->CreateAdapterList(1, &DXCORE_ADAPTER_ATTRIBUTE_D3D12_GRAPHICS, &list)
    //      (also enumerate compute-only adapters)
    //   3. for each adapter in list:
    //        adapter->GetProperty(HardwareID,           ...)  // vendor_id/device_id/...
    //        adapter->GetProperty(DedicatedAdapterMemory, ...)
    //        adapter->GetProperty(SharedAdapterMemory,    ...)
    //        adapter->GetProperty(DriverDescription,      ...)
    //        adapter->GetProperty(InstanceLuid,           ...)
    //      Translate into rocprofiler_agent_t (see Linux path for field semantics).
    //   4. Call update_agent_runtime_visibility() on each.
    //
    // This is the same call sequence the wsl-temp branch contemplated; see
    // commit 9d9820c5 for the HSA-derived shim it actually used (which we are
    // NOT salvaging).
    ROCP_WARNING << "DxgProvider::enumerate(): not yet implemented (TODO)";
    return out;
}

}  // namespace topology
}  // namespace rocprofiler
