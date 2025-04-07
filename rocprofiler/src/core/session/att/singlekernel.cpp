/* Copyright (c) 2022 Advanced Micro Devices, Inc.

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

#include "att.h"
#include <cassert>
#include <atomic>
#include "src/core/hsa/packets/packets_generator.h"
#include "src/api/rocprofiler_singleton.h"
#include "src/core/isa_capture/code_object_track.hpp"

#define __NR_gettid 186

namespace rocprofiler {

namespace att {

std::pair<std::vector<bool>, bool>
AttTracer::GetAllowedProfilesList(const void* packets, int pkt_count) {

  std::vector<bool> can_profile_packet;
  bool b_can_profile_anypacket = false;
  can_profile_packet.reserve(pkt_count);

  rocprofiler::HSASupport_Singleton& hsasupport_singleton =
      rocprofiler::HSASupport_Singleton::GetInstance();

  std::lock_guard<std::mutex> lock(hsasupport_singleton.ksymbol_map_lock);
  assert(hsasupport_singleton.ksymbols);

  uint32_t current_writer_id = WRITER_ID.load(std::memory_order_relaxed);

  for (int i = 0; i < pkt_count; ++i) {
    auto& original_packet = static_cast<const hsa_barrier_and_packet_t*>(packets)[i];
    bool b_profile_this_object = false;

    // Skip packets other than kernel dispatch packets.
    if (Packet::IsDispatchPacket(original_packet)) {
      auto& kdispatch = static_cast<const hsa_kernel_dispatch_packet_s*>(packets)[i];
      try {
        // Can throw
        const std::string& kernel_name = hsasupport_singleton.ksymbols->at(kdispatch.kernel_object);

        // If no filters specified, auto profile this kernel
        if (kernel_profile_names.size() == 0 &&
            kernel_name.find("__amd_rocclr_") == std::string::npos
        ) {
          b_profile_this_object = true;
        } else {
          // Try to match the mangled kernel name with given matches in input.txt
          // We want to initiate att profiling if a match exists
          for (const std::string& kernel_matches : kernel_profile_names)
            if (kernel_name.find(kernel_matches) != std::string::npos) b_profile_this_object = true;
        }
      } catch (...) {
        rocprofiler::warning("Warning: Unknown name for object %lu\n", kdispatch.kernel_object);
      }
      current_writer_id += 1;
    }
    b_can_profile_anypacket |= b_profile_this_object;
    can_profile_packet.push_back(b_profile_this_object);
  }
  // If we're going to skip all packets, need to update writer ID
  if (!b_can_profile_anypacket)
    WRITER_ID.store(current_writer_id, std::memory_order_release);

  return {can_profile_packet, b_can_profile_anypacket};
}

bool AttTracer::ATTSingleWriteInterceptor(
  const void* packets,
  uint64_t pkt_count,
  uint64_t user_pkt_index,
  queue::Queue& queue_info,
  hsa_amd_queue_intercept_packet_writer writer,
  rocprofiler_buffer_id_t buffer_id
) {
  static int KernelInterceptCount = 0;
  static const char* env_MAX_ATT_PROFILES = getenv("ROCPROFILER_MAX_ATT_PROFILES");
  static int MAX_ATT_PROFILES = env_MAX_ATT_PROFILES ? atoi(env_MAX_ATT_PROFILES) : 1;

  if (KernelInterceptCount >= MAX_ATT_PROFILES) return false;

  const packet_t* packets_arr = reinterpret_cast<const packet_t*>(packets);
  std::vector<packet_t> transformed_packets;

  // att start
  // Getting Queue Data and Information
  rocprofiler::HSAAgentInfo& agentInfo =
      rocprofiler::HSASupport_Singleton::GetInstance().GetHSAAgentInfo(
          queue_info.GetGPUAgent().handle);

  bool can_profile_anypacket = false;
  std::vector<bool> can_profile_packet;
  std::tie(can_profile_packet, can_profile_anypacket) = GetAllowedProfilesList(packets, pkt_count);

  if (!can_profile_anypacket) return false;

  // Preparing att Packets
  packet_t start_packet{};
  packet_t stop_packet{};
  hsa_ven_amd_aqlprofile_profile_t* profile = nullptr;
  rocprofiler_codeobj_capture_mode_t capturem = ROCPROFILER_CAPTURE_SYMBOLS_ONLY;

  std::tie(profile, capturem) = ProcessATTParams(start_packet, stop_packet, queue_info, agentInfo);

  // Searching across all the packets given during this write
  for (size_t i = 0; i < pkt_count; ++i) {
    auto& original_packet = static_cast<const hsa_barrier_and_packet_t*>(packets)[i];
    uint32_t writer_id = 0;

    // Skip all packets marked with !can_profile
    if (i >= can_profile_packet.size() || can_profile_packet[i] == false) {
      transformed_packets.emplace_back(packets_arr[i]);

      // increment writer ID for every packet
      if (Packet::IsDispatchPacket(original_packet))
        writer_id = WRITER_ID.fetch_add(1, std::memory_order_release);

      continue;
    }
    KernelInterceptCount += 1;
    writer_id = WRITER_ID.fetch_add(1, std::memory_order_release);

    if (profile) {
      // Adding start packet and its barrier with a dummy signal
      hsa_signal_t dummy_signal{};
      dummy_signal.handle = 0;
      start_packet.header = HSA_PACKET_TYPE_VENDOR_SPECIFIC << HSA_PACKET_HEADER_TYPE;
      Packet::AddVendorSpecificPacket(&start_packet, &transformed_packets, dummy_signal);
      Packet::CreateBarrierPacket(&transformed_packets, &start_packet.completion_signal, nullptr) ;
    }

    auto& packet = transformed_packets.emplace_back(packets_arr[i]);
    auto& dispatch_packet = reinterpret_cast<hsa_kernel_dispatch_packet_t&>(packet);

    CreateSignal(HSA_AMD_SIGNAL_AMD_GPU_ONLY, &packet.completion_signal);

    // Adding the dispatch packet newly created signal to the pending signals
    // list to be processed by the signal interrupt
    uint64_t record_id = rocprofiler::ROCProfiler_Singleton::GetInstance().GetUniqueRecordId();
    AddKernelNameWithDispatchID(GetKernelNameFromKsymbols(dispatch_packet.kernel_object), record_id);

    this->AddPendingSignals(
      writer_id,
      record_id,
      original_packet.completion_signal,
      packet.completion_signal,
      session_id_,
      buffer_id,
      profile,
      {0},
      (uint32_t)syscall(__NR_gettid),
      user_pkt_index
    );

    uint64_t IsGFX9 = HSASupport_Singleton::GetInstance()
                        .GetHSAAgentInfo(queue_info.GetGPUAgent().handle)
                        .GetDeviceInfo()
                        .getName()
                        .find("gfx9") != std::string::npos;
    codeobj_record::make_capture(rocprofiler_record_id_t{record_id}, capturem, IsGFX9);
    codeobj_record::start_capture(rocprofiler_record_id_t{record_id});
    codeobj_record::stop_capture(rocprofiler_record_id_t{record_id});

    // Make a copy of the original packet, adding its signal to a barrier packet
    if (original_packet.completion_signal.handle != 0U) {
      hsa_barrier_and_packet_t barrier{};
      barrier.header = HSA_PACKET_TYPE_BARRIER_AND << HSA_PACKET_HEADER_TYPE;
      barrier.dep_signal[0] = packet.completion_signal;
      packet_t* __attribute__((__may_alias__)) pkt =
          (reinterpret_cast<packet_t*>(&barrier));
      transformed_packets.emplace_back(*pkt).completion_signal =
          original_packet.completion_signal;
    }

    // Adding a barrier packet with the original packet's completion signal.
    hsa_signal_t interrupt_signal;
    CreateSignal(0, &interrupt_signal);

    // Adding Stop PM4 Packets
    if (profile) {
      stop_packet.header = HSA_PACKET_TYPE_VENDOR_SPECIFIC << HSA_PACKET_HEADER_TYPE;
      Packet::AddVendorSpecificPacket(&stop_packet, &transformed_packets, interrupt_signal);

      // Added Interrupt Signal with barrier and provided handler for it
      Packet::CreateBarrierPacket(&transformed_packets, &interrupt_signal, nullptr);
    } else {
      hsa_barrier_and_packet_t barrier{};
      barrier.header = HSA_PACKET_TYPE_BARRIER_AND << HSA_PACKET_HEADER_TYPE;
      barrier.completion_signal = interrupt_signal;
      packet_t* __attribute__((__may_alias__)) pkt =
          (reinterpret_cast<packet_t*>(&barrier));
      transformed_packets.emplace_back(*pkt);
    }

    // Creating Async Handler to be called every time the interrupt signal is
    // marked complete
    signalAsyncHandlerATT(interrupt_signal, new queue::queue_info_session_t{
      queue_info.GetGPUAgent(),
      session_id_,
      queue_info.GetQueueID(),
      writer_id,
      interrupt_signal,
      HSASupport_Singleton::GetInstance()
                        .GetHSAAgentInfo(queue_info.GetGPUAgent().handle)
                        .GetDeviceInfo()
                        .getNumaNode()
    });
  }
  /* Write the transformed packets to the hardware queue.  */
  writer(&transformed_packets[0], transformed_packets.size());
  return true;
}




}  // namespace att

}  // namespace rocprofiler
