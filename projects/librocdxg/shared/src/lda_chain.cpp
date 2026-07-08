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

#include "shared/include/device.h"
#include "shared/include/thunks.h"
#include "shared/include/lda_chain.h"
#include "shared/include/thunk_proxy/thunk_proxy.h"
#include "shared/include/constants.h"
#include <algorithm>
#include <memory>

using namespace std;

namespace wsl {
namespace thunk {

LdaChain::LdaChain(Platform *platform,
                   std::unique_ptr<thunk_proxy::ChainContext> chain_ctx,
                   LUID             adapter_luid,
                   WinAdapterHandle adapter_handle)
    : platform_(platform),
      chain_ctx_(std::move(chain_ctx)),
      adapter_luid_(adapter_luid),
      adapter_handle_(adapter_handle) {
  std::fill(std::begin(chained_devices_), std::end(chained_devices_), nullptr);
}

LdaChain::~LdaChain() {
  d3dthunk::DestroyDevice(*this);
  platform_    = nullptr;
  adapter_handle_ = 0;
  device_handle_  = 0;
}

ErrorCode LdaChain::QueryLinkedGpusInChain(vector<Device *> &devices,
                                            bool disableGpuTimeout) {
  // Open a D3DKMT device for QAI escapes.
  d3dthunk::CreateDeviceArgs args{};
  args.hAdapter = AdapterHandle();
  args.Flags.DisableGpuTimeout = disableGpuTimeout;
  if (auto code = d3dthunk::CreateDevice(*this, &args);
      code != ErrorCode::Success)
    return code;
  device_handle_ = args.hDevice;

  // QueryAdapterInfo issues all KMD escapes and updates chain-side caches.
  const auto qaiCode = chain_ctx_->QueryAdapterInfo();
  if (qaiCode != ErrorCode::Success)
    return qaiCode;

  auto code = ErrorCode::Success;
  for (u32 chain = 0; chain < ChainedDeviceCount() && code == ErrorCode::Success;
       ++chain) {
    if (devices.size() >= MaxDevices) {
      code = ErrorCode::InitializationFailed;
      break;
    }

    switch (VendorId(chain)) {
    case ATI_VENDOR_ID: {
      code = Device::Create(GetPlatform(), this,
                            static_cast<u32>(devices.size()),
                            chain, &chained_devices_[chain]);
      if (code == ErrorCode::Success && ChainedDevice(chain)) {
        devices.push_back(ChainedDevice(chain));
        devices.back()->Init();  // pre-fetch sensor limits; ignore failure
      }
      break;
    }
    default:
      code = ErrorCode::UnSupported;
      break;
    }
  }
  return code;
}

} // namespace thunk
} // namespace wsl
