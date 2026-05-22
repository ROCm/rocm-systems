////////////////////////////////////////////////////////////////////////////////
//
// The University of Illinois/NCSA
// Open Source License (NCSA)
//
// Copyright (c) 2020, Advanced Micro Devices, Inc. All rights reserved.
//
// Developed by:
//
//                 AMD Research and AMD HSA Software Development
//
//                 Advanced Micro Devices, Inc.
//
//                 www.amd.com
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to
// deal with the Software without restriction, including without limitation
// the rights to use, copy, modify, merge, publish, distribute, sublicense,
// and/or sell copies of the Software, and to permit persons to whom the
// Software is furnished to do so, subject to the following conditions:
//
//  - Redistributions of source code must retain the above copyright notice,
//    this list of conditions and the following disclaimers.
//  - Redistributions in binary form must reproduce the above copyright
//    notice, this list of conditions and the following disclaimers in
//    the documentation and/or other materials provided with the distribution.
//  - Neither the names of Advanced Micro Devices, Inc,
//    nor the names of its contributors may be used to endorse or promote
//    products derived from this Software without specific prior written
//    permission.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
// THE CONTRIBUTORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR
// OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE,
// ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
// DEALINGS WITH THE SOFTWARE.
//
////////////////////////////////////////////////////////////////////////////////

#include "shared/include/constants.h"
#include "shared/include/thunks.h"
#include "shared/include/platform.h"
#include "shared/include/device.h"
#include "shared/include/gpu_info.h"
#include "shared/include/utils.h"
#include "shared/include/thunk_proxy/thunk_proxy.h"
#include <memory>
#include <vector>

namespace wsl {
namespace thunk {

using namespace std;

__attribute__((visibility("hidden")))
Platform& Platform::instance() {
  static Platform *platform_ = new Platform();
  return *platform_;
}

ErrorCode Platform::Init() {
  return ReEnumerateDevices();
}

void Platform::Destroy() {
  TearDownDevices();

  for (auto i = 0u; i < lda_chain_count_; i++) {
    delete lda_chain_list_[i];
    lda_chain_list_[i] = nullptr;
  }

  lda_chain_count_ = 0;
}

void Platform::TearDownDevices() {
  for (auto dev : devices_)
    delete dev;
  devices_.clear();
}

ErrorCode Platform::ReEnumerateDevices() {
  Destroy();
  auto code = reQueryDevices();
  if (code != ErrorCode::Success)
    Destroy();
  return code;
}

ErrorCode Platform::EnumerateDevices(vector<Device *> &devices) {
  auto code = ReEnumerateDevices();
  if (code == ErrorCode::Success)
    devices = devices_;
  return code;
}

ErrorCode Platform::reQueryDevices() {
  auto code = ErrorCode::Success;
  devices_.reserve(MaxDevices);

  D3DKMT_ENUMADAPTERS2 args{};
  if (code = d3dthunk::EnumAdapters(&args); code != ErrorCode::Success)
    return code;
  vector<D3DKMT_ADAPTERINFO> infos(args.NumAdapters);

  args.pAdapters = infos.data();
  if (code = d3dthunk::EnumAdapters(&args); code != ErrorCode::Success)
    return code;

  if (args.NumAdapters > 0)
    queryWddmVersion(infos[0].hAdapter);

  for (u32 i = 0; i < args.NumAdapters; ++i)
    code = queryLinkedDevicesInLdaChain(args.pAdapters[i]);

  if (lda_chain_count_ > 0)
    code = ErrorCode::Success;

  return code;
}

ErrorCode Platform::queryLinkedDevicesInLdaChain(
                      const D3DKMT_ADAPTERINFO &adapterInfo) {

  auto code = ErrorCode::Success;

  constexpr u32 intelVendorId = 0x8086;
  D3DKMT_PHYSICAL_ADAPTER_COUNT countInfo{};
  d3dthunk::QueryAdapterInfoArgs queryInfo{};

  queryInfo.hAdapter = adapterInfo.hAdapter;
  queryInfo.Type = KMTQAITYPE_PHYSICALADAPTERCOUNT;
  queryInfo.pPrivateDriverData = &countInfo;
  queryInfo.PrivateDriverDataSize = sizeof(countInfo);

  code = d3dthunk::QueryAdapterInfo(&queryInfo);
  for (u32 i = 0; (i < countInfo.Count) && (code == ErrorCode::Success); i++) {
    D3DKMT_QUERY_DEVICE_IDS curPhysDev{};
    curPhysDev.PhysicalAdapterIndex = i;

    queryInfo.hAdapter = adapterInfo.hAdapter;
    queryInfo.Type = KMTQAITYPE_PHYSICALADAPTERDEVICEIDS;
    queryInfo.pPrivateDriverData = &curPhysDev;
    queryInfo.PrivateDriverDataSize = sizeof(curPhysDev);

    code = d3dthunk::QueryAdapterInfo(&queryInfo);

    if (code != ErrorCode::Success)
      break;

    const u32 vendorId = curPhysDev.DeviceIds.VendorID;
    if (vendorId != AMD_VENDOR_ID && vendorId != ATI_VENDOR_ID &&
        vendorId != intelVendorId) {
      code = ErrorCode::IncompatibleDevice;
      break;
    }

    if ((vendorId == AMD_VENDOR_ID || vendorId == ATI_VENDOR_ID) &&
        !QueryAdapterSupported(curPhysDev.DeviceIds.DeviceID)) {
      code = ErrorCode::IncompatibleDevice;
      break;
    }
  }

  if (code != ErrorCode::Success)
    return code;

  if (lda_chain_count_ >= MaxDevices)
    return ErrorCode::InitializationFailed;

  auto ctx = std::unique_ptr<thunk_proxy::ChainContext>(
      thunk_proxy::ChainContext::Create(
          static_cast<uint32_t>(adapterInfo.hAdapter),
          adapterInfo.AdapterLuid));

  if (!ctx)
    return ErrorCode::InitializationFailed;

  auto lda_chain = 
    std::make_unique<LdaChain>(this, std::move(ctx),
                               adapterInfo.AdapterLuid, adapterInfo.hAdapter);
  code = lda_chain->QueryLinkedGpusInChain(devices_, false);
  if (code == ErrorCode::Success) {
    lda_chain_list_[lda_chain_count_] = lda_chain.release();
    ++lda_chain_count_;
  }

  return code;
}

void Platform::queryWddmVersion(D3DKMT_HANDLE adapter) {
  D3DKMT_DRIVERVERSION version = static_cast<D3DKMT_DRIVERVERSION>(0);
  d3dthunk::QueryAdapterInfoArgs queryInfo{};

  queryInfo.hAdapter = adapter;
  queryInfo.Type = KMTQAITYPE_DRIVERVERSION;
  queryInfo.pPrivateDriverData = &version;
  queryInfo.PrivateDriverDataSize = sizeof(version);

  if (d3dthunk::QueryAdapterInfo(&queryInfo) == ErrorCode::Success)
    wddm_version_ = version;
}

} // namespace thunk
} // namespace wsl
