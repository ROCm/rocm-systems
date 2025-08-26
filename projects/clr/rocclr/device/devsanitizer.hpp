/* Copyright (c) 2021 - 2021 Advanced Micro Devices, Inc.

 Permission is hereby granted, free of charge, to any person obtaining a copy
 of this software and associated documentation files (the "Software"), to deal
 in the Software without restriction, including without limitation the rights
 to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 copies of the Software, and to permit persons to whom the Software is
 furnished to do so, subject to the following conditions:

 The above copyright notice and this permission notice shall be included in
 all copies or substantial portions of the Software.

 THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
 THE SOFTWARE. */

#pragma once
#include "device/devhostcall.hpp"
#include "device/device.hpp"
#include "device/devurilocator.hpp"
#include "utils/debug.hpp"
#include "platform/memory.hpp"

#include <inttypes.h>  //to exp
#include <string>
#include <vector>
#include <tuple>
#include <algorithm>

// Address sanitizer runtime entry-function to report the invalid device memory access
// this will be defined in llvm-project/compiler-rt/lib/asan, and will have effect only
// when compiler-rt is build for AMDGPU.
// Note: This API is runtime interface of asan library and only defined for linux os.
extern "C" void __asan_report_nonself_error(uint64_t* callstack, uint32_t n_callstack,
                                            uint64_t* addr, uint32_t naddr, uint64_t* entity_ids,
                                            uint32_t n_entities, bool is_write,
                                            uint32_t access_size, bool is_abort, const char* name,
                                            int64_t vma_adjust, int fd, uint64_t file_extent_size,
                                            uint64_t file_extent_start = 0);

namespace amd {
void handleSanitizerService(Payload* packt_payload, uint64_t activemask,
                            const amd::Device* gpu_device, device::UriLocator* uri_locator) {
  // An address results in invalid access in each active lane
  uint64_t device_failing_addresses[64];
  // An array of identifications of entities requesting a report.
  // index 0       - contains device id
  // index 1,2,3   - contains wg_idx, wg_idy, wg_idz respectively.
  // index 4 to 67 - contains reporting wave ids in a wave-front.
  uint64_t entity_id[68], callstack[1];
  uint32_t n_activelanes = __builtin_popcountl(activemask);
  uint64_t access_info = 0, access_size = 0;
  bool is_abort = true;
  entity_id[0] = gpu_device->index();

  assert(packt_payload != nullptr && "packet payload is null?");

  int indx = 0, en_idx = 1;
  bool first_workitem = false;
  while (activemask) {
    auto wi = amd::leastBitSet(activemask);
    activemask ^= static_cast<decltype(activemask)>(1) << wi;
    auto data_slot = packt_payload->slots[wi];
    // encoding of packet payload arguments is
    // defined in device-libs/asanrtl/src/report.cl
    if (!first_workitem) {
      device_failing_addresses[indx] = data_slot[0];
      callstack[0] = data_slot[1];
      entity_id[en_idx] = data_slot[2];
      entity_id[++en_idx] = data_slot[3];
      entity_id[++en_idx] = data_slot[4];
      entity_id[++en_idx] = data_slot[5];
      access_info = data_slot[6];
      access_size = data_slot[7];
      first_workitem = true;
    } else {
      device_failing_addresses[indx] = data_slot[0];
      entity_id[en_idx] = data_slot[5];
    }
    indx++;
    en_idx++;
  }

  bool is_write = false;
  if (access_info & 0xFFFFFFFF00000000) is_abort = false;
  if (access_info & 1) is_write = true;

  std::string fileuri;
  uint64_t size = 0, offset = 0;
  int64_t loadAddrAdjust = 0;
  auto uri_fd = amd::Os::FDescInit();
  if (uri_locator) {
    device::UriLocator::UriInfo uri_info = uri_locator->lookUpUri(callstack[0]);
    std::tie(offset, size) = uri_locator->decodeUriAndGetFd(uri_info, &uri_fd);
    loadAddrAdjust = uri_info.loadAddressDiff;
  }

#if defined(__linux__)
  __asan_report_nonself_error(callstack, 1, device_failing_addresses, n_activelanes, entity_id,
                              n_activelanes + 4, is_write, access_size, is_abort,
                              /*thread key*/ "amdgpu", loadAddrAdjust, uri_fd, size, offset);
#endif
}
}  // namespace amd
