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

void AttTracer::InsertMarker(
  std::vector<packet_t>& transformed_packets,
  hsa_agent_t agent,
  uint32_t data
) {
  packet_t marker_packet{};
  auto desc = Packet::GenerateATTMarkerPackets(agent, marker_packet, data);
  if (desc.ptr && desc.size)
    Packet::AddVendorSpecificPacket(&marker_packet, &transformed_packets, hsa_signal_t{.handle = 0});
}


std::optional<std::pair<size_t, size_t>> AttTracer::RequiresStartPacket(size_t rstart, size_t size)
{
  for (auto& r : kernel_profile_dispatch_ids)
    if (rstart <= r.first && rstart+size > r.first)
      return r;
  return {};
}

bool AttTracer::ATTContiguousWriteInterceptor(
  const void* packets,
  uint64_t pkt_count,
  queue::Queue& queue_info,
  hsa_amd_queue_intercept_packet_writer writer,
  rocprofiler_buffer_id_t buffer_id
) {
  const packet_t* packets_arr = reinterpret_cast<const packet_t*>(packets);
  std::vector<packet_t> transformed_packets;

  // att start
  // Getting Queue Data and Information
  rocprofiler::HSAAgentInfo& agentInfo = rocprofiler::HSASupport_Singleton::GetInstance()
                                        .GetHSAAgentInfo(queue_info.GetGPUAgent().handle);

  auto dispatchPackets = Packet::ExtractDispatchPackets(packets, pkt_count);
  if (dispatchPackets.size() == 0) return false;

  size_t writer_id = WRITER_ID.fetch_add(dispatchPackets.size(), std::memory_order_relaxed);
  uint32_t new_load_cnt = codeobj_capture_instance::GetLoadCount();

  auto bInsertStart = RequiresStartPacket(writer_id, dispatchPackets.size());
  {
    std::lock_guard<std::mutex> lk(att_enable_disable_mutex);
    // If att_start already exists, don't start again
    auto agent_pending_packets = pending_stop_packets.find(queue_info.GetGPUAgent().handle);
    if (agent_pending_packets != pending_stop_packets.end())
      bInsertStart = {};

    // If nothing will be added or removed, return
    if (!bInsertStart && codeobj_load_cnt == new_load_cnt)
    {
      if (
        agent_pending_packets == pending_stop_packets.end() ||
        agent_pending_packets->second.last_kernel_exec > writer_id + dispatchPackets.size()
      )
        return false;
    }
  }

  if (bInsertStart)
  {
    // Preparing att Packets
    packet_t start_packet{};
    packet_t stop_packet{};
    hsa_ven_amd_aqlprofile_profile_t* profile = nullptr;
    rocprofiler_codeobj_capture_mode_t capturem = ROCPROFILER_CAPTURE_SYMBOLS_ONLY;
    std::tie(profile, capturem) = ProcessATTParams(start_packet, stop_packet, queue_info, agentInfo);

    if (!profile)
    {
      rocprofiler::warning("Failed to create profile from queue!");
      return false;
    }

    uint64_t IsGFX9 = HSASupport_Singleton::GetInstance()
                      .GetHSAAgentInfo(queue_info.GetGPUAgent().handle)
                      .GetDeviceInfo()
                      .getName()
                      .find("gfx9") != std::string::npos;

    hsa_signal_t dummy_signal{};
    dummy_signal.handle = 0;
    start_packet.header = HSA_PACKET_TYPE_VENDOR_SPECIFIC << HSA_PACKET_HEADER_TYPE;
    Packet::AddVendorSpecificPacket(&start_packet, &transformed_packets, dummy_signal);
    Packet::CreateBarrierPacket(&transformed_packets, &start_packet.completion_signal, nullptr);

    uint64_t record_id = rocprofiler::ROCProfiler_Singleton::GetInstance().GetUniqueRecordId();
    AddKernelNameWithDispatchID("ATT_Contiguous", record_id);

    this->AddPendingSignals(
      writer_id,
      record_id,
      start_packet.completion_signal,
      start_packet.completion_signal,
      session_id_,
      buffer_id,
      profile,
      {0},
      (uint32_t)syscall(__NR_gettid),
      0
    );

    codeobj_record::make_capture(rocprofiler_record_id_t{record_id}, capturem, IsGFX9);
    codeobj_record::start_capture(rocprofiler_record_id_t{record_id});

    stop_packet.header = HSA_PACKET_TYPE_VENDOR_SPECIFIC << HSA_PACKET_HEADER_TYPE;
    std::lock_guard<std::mutex> lk(att_enable_disable_mutex);
    pending_stop_packets[queue_info.GetGPUAgent().handle]
                = {record_id, writer_id, bInsertStart->second, session_id_, stop_packet};
  }

  if (codeobj_load_cnt != new_load_cnt)
  {
    codeobj_load_cnt = new_load_cnt;
    InsertMarker(transformed_packets, queue_info.GetGPUAgent(), new_load_cnt);
  }

  // Searching across all the packets given during this write
  for (size_t i = 0; i < pkt_count; ++i)
    transformed_packets.emplace_back(packets_arr[i]);

  {
    std::lock_guard<std::mutex> lk(att_enable_disable_mutex);
    auto agent_pending_packets = pending_stop_packets.find(queue_info.GetGPUAgent().handle);

    if (agent_pending_packets != pending_stop_packets.end() &&
        agent_pending_packets->second.last_kernel_exec <= writer_id + dispatchPackets.size()
    ) {
      const ATTRecordSignal& rsignal = agent_pending_packets->second;
      // Adding a barrier packet with the original packet's completion signal.
      hsa_signal_t interrupt_signal;
      CreateSignal(0, &interrupt_signal);

      // Adding Stop PM4 Packets
      Packet::AddVendorSpecificPacket(&rsignal.stop_packet, &transformed_packets, interrupt_signal);

      // Added Interrupt Signal with barrier and provided handler for it
      Packet::CreateBarrierPacket(&transformed_packets, &interrupt_signal, nullptr);

      // Creating Async Handler to be called every time the interrupt signal is marked complete
      signalAsyncHandlerATT(interrupt_signal, new queue::queue_info_session_t{
          queue_info.GetGPUAgent(),
          rsignal.session_id_snapshot,
          queue_info.GetQueueID(),
          rsignal.writer_id,
          interrupt_signal
      });

      codeobj_record::stop_capture(rocprofiler_record_id_t{rsignal.record_id});
      pending_stop_packets.erase(queue_info.GetGPUAgent().handle);
    }
  }

  /* Write the transformed packets to the hardware queue.  */
  writer(&transformed_packets[0], transformed_packets.size());
  return true;
}

}  // namespace att

}  // namespace rocprofiler
