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

#define ATT_MARKER_HEADER_CHANNEL HSA_VEN_AMD_AQLPROFILE_ATT_CHANNEL_0
#define ATT_MARKER_SIZE_CHANNEL HSA_VEN_AMD_AQLPROFILE_ATT_CHANNEL_1
#define ATT_MARKER_LO_CHANNEL HSA_VEN_AMD_AQLPROFILE_ATT_CHANNEL_2
#define ATT_MARKER_HI_CHANNEL HSA_VEN_AMD_AQLPROFILE_ATT_CHANNEL_3

enum rocprofiler_att_marker_type_t {
  ROCPROFILER_ATT_MARKER_LOAD = 0,
  ROCPROFILER_ATT_MARKER_UNLOAD = 1
};

union att_header_marker_t
{
  uint32_t raw;
  struct {
    uint32_t type : 1;
    uint32_t bFromStart : 1;
    uint32_t id : 30;
  };
};

namespace rocprofiler {

namespace att {

void AttTracer::InsertUnloadMarker(
  std::vector<packet_t>& transformed_packets,
  hsa_agent_t agent,
  uint32_t data
) {
  att_header_marker_t header{.raw = 0};
  header.type = ROCPROFILER_ATT_MARKER_UNLOAD;
  header.id = data;
  hsa_ven_amd_aqlprofile_att_marker_channel_t channel = ATT_MARKER_HEADER_CHANNEL;

  this->InsertMarker(transformed_packets, agent, header.raw, channel);
}

void AttTracer::InsertLoadMarker(
  std::vector<packet_t>& transformed_packets,
  hsa_agent_t agent,
  rocprofiler_intercepted_codeobj_t codeobj,
  bool bFromStart
) {
  this->InsertMarker(transformed_packets, agent, codeobj.mem_size, ATT_MARKER_SIZE_CHANNEL);
  auto sizehi = static_cast<hsa_ven_amd_aqlprofile_att_marker_channel_t>(4);
  this->InsertMarker(transformed_packets, agent, codeobj.mem_size>>32, sizehi);

  uint64_t addr = codeobj.base_address;
  this->InsertMarker(transformed_packets, agent, addr & ((1ul << 32)-1), ATT_MARKER_LO_CHANNEL);
  this->InsertMarker(transformed_packets, agent, addr >> 32, ATT_MARKER_HI_CHANNEL);

  att_header_marker_t header{.raw = 0};
  header.type = ROCPROFILER_ATT_MARKER_LOAD;
  header.bFromStart = bFromStart;
  header.id = codeobj.att_marker_id;
  this->InsertMarker(transformed_packets, agent, header.raw, ATT_MARKER_HEADER_CHANNEL);
}

void AttTracer::InsertMarker(
  std::vector<packet_t>& transformed_packets,
  hsa_agent_t agent,
  uint32_t data,
  hsa_ven_amd_aqlprofile_att_marker_channel_t channel
) {
  packet_t marker_packet{};
  auto desc = Packet::GenerateATTMarkerPackets(agent, marker_packet, data, channel);
  if (desc.ptr && desc.size)
    Packet::AddVendorSpecificPacket(&marker_packet, &transformed_packets, hsa_signal_t{.handle = 0});
  else
    rocprofiler::warning("Could not add ATT Marker");
}


std::optional<std::pair<size_t, size_t>> AttTracer::RequiresStartPacket(size_t rstart, size_t size)
{
  for (auto& r : kernel_profile_dispatch_ids)
    if (rstart <= r.first && rstart+size > r.first)
      return r;
  return {};
}

bool AttTracer::InsertPacketStart(
  std::vector<packet_t>& transformed_packets,
  queue::Queue& queue,
  size_t writer_id,
  rocprofiler_buffer_id_t buffer_id,
  size_t stop_id,
  const std::string& kernel_name
) {
  // Preparing att Packets
    packet_t start_packet{};
    packet_t stop_packet{};
    hsa_ven_amd_aqlprofile_profile_t* profile = nullptr;
    rocprofiler_codeobj_capture_mode_t capturem = ROCPROFILER_CAPTURE_SYMBOLS_ONLY;

    auto agent_handle = queue.GetGPUAgent().handle;
    rocprofiler::HSAAgentInfo& agentInfo = rocprofiler::HSASupport_Singleton::GetInstance()
                                           .GetHSAAgentInfo(agent_handle);

    std::tie(profile, capturem) = ProcessATTParams(start_packet, stop_packet, queue, agentInfo);

    if (!profile)
    {
      rocprofiler::warning("Failed to create profile from queue!");
      return false;
    }

    uint64_t IsGFX9 = agentInfo.GetDeviceInfo()
                               .getName()
                               .find("gfx9") != std::string::npos;

    hsa_signal_t dummy_signal{};
    dummy_signal.handle = 0;
    start_packet.header = HSA_PACKET_TYPE_VENDOR_SPECIFIC << HSA_PACKET_HEADER_TYPE;
    Packet::AddVendorSpecificPacket(&start_packet, &transformed_packets, dummy_signal);
    Packet::CreateBarrierPacket(&transformed_packets, &start_packet.completion_signal, nullptr);

    uint64_t record_id = rocprofiler::ROCProfiler_Singleton::GetInstance().GetUniqueRecordId();
    AddKernelNameWithDispatchID(kernel_name, record_id);

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

    this->capture_id = rocprofiler_record_id_t{record_id};
    codeobj_record::make_capture(this->capture_id, capturem, IsGFX9);
    codeobj_record::start_capture(this->capture_id);

    stop_packet.header = HSA_PACKET_TYPE_VENDOR_SPECIFIC << HSA_PACKET_HEADER_TYPE;
    pending_stop_packets[agent_handle] = {record_id, writer_id, stop_id, session_id_, stop_packet};

    return true;
}

void AttTracer::InsertPacketStop(
  std::vector<packet_t>& transformed,
  const ATTRecordSignal& rsignal,
  queue::Queue& queue,
  uint64_t agent_handle
) {
  // Adding a barrier packet with the original packet's completion signal.
  hsa_signal_t interrupt_signal;
  CreateSignal(0, &interrupt_signal);

  // Adding Stop PM4 Packets
  Packet::AddVendorSpecificPacket(&rsignal.stop_packet, &transformed, interrupt_signal);

  // Added Interrupt Signal with barrier and provided handler for it
  Packet::CreateBarrierPacket(&transformed, &interrupt_signal, nullptr);

  // Creating Async Handler to be called every time the interrupt signal is marked complete
  signalAsyncHandlerATT(interrupt_signal, new queue::queue_info_session_t{
      queue.GetGPUAgent(),
      rsignal.session_id_snapshot,
      queue.GetQueueID(),
      rsignal.writer_id,
      interrupt_signal,
      HSASupport_Singleton::GetInstance()
                        .GetHSAAgentInfo(agent_handle)
                        .GetDeviceInfo()
                        .getNumaNode()
  });

  //codeobj_record::stop_capture(rocprofiler_record_id_t{rsignal.record_id});
  codeobj_record::stop_capture(this->capture_id);
  active_capture_event_ids.clear();
  pending_stop_packets.erase(agent_handle);
}

bool AttTracer::ATTContiguousWriteInterceptor(
  const void* packets,
  uint64_t pkt_count,
  queue::Queue& queue_info,
  hsa_amd_queue_intercept_packet_writer writer,
  rocprofiler_buffer_id_t buffer_id
) {
  const packet_t* packets_arr = reinterpret_cast<const packet_t*>(packets);
  std::vector<packet_t> transformed;

  // att start
  // Getting Queue Data and Information
  const auto agent_handle = queue_info.GetGPUAgent().handle;

  auto dispatchPackets = Packet::ExtractDispatchPackets(packets, pkt_count);
  if (dispatchPackets.size() == 0) return false;

  const size_t writer_id = WRITER_ID.fetch_add(dispatchPackets.size(), std::memory_order_relaxed);
  const size_t writer_end_id = writer_id + dispatchPackets.size();
  const uint32_t new_load_cnt = codeobj_capture_instance::GetEventCount();

  auto insertStart = RequiresStartPacket(writer_id, dispatchPackets.size());
  {
    std::lock_guard<std::mutex> lk(att_enable_disable_mutex);
    // If att_start already exists, don't start again
    bool bIsActive = this->HasActiveTracerATT(agent_handle);
    if (bIsActive)
      insertStart = {};

    // If nothing will be added or removed, return
    if (!insertStart && codeobj_event_cnt == new_load_cnt)
    {
      if (!bIsActive || pending_stop_packets.at(agent_handle).last_kernel_exec > writer_end_id)
        return false;
    }
  }

  if (insertStart)
  {
    size_t end_writer = insertStart->second;
    std::string name = GetKernelNameFromKsymbols(dispatchPackets.at(0)->kernel_object);
    if (insertStart->first != end_writer)
      name = "ATT_Start_" + name;

    std::lock_guard<std::mutex> lk(att_enable_disable_mutex);
    bool bSuc = InsertPacketStart(transformed, queue_info, writer_id, buffer_id, end_writer, name);
    if (!bSuc) return false;
  }

  bool bHasPending = false;
  {
    std::lock_guard<std::mutex> lk(att_enable_disable_mutex);
    bHasPending = pending_stop_packets.find(agent_handle) != pending_stop_packets.end();
  }

  if (bHasPending && (insertStart || codeobj_event_cnt != new_load_cnt))
  {
    codeobj_event_cnt = new_load_cnt;

    auto symbols = codeobj_record::get_capture(this->capture_id);
    std::unordered_set<size_t> current_ids;

    for (size_t s=0; s<symbols.count; s++)
      current_ids.insert(symbols.symbols[s].att_marker_id);

    for (size_t prev_id : active_capture_event_ids)
      if (current_ids.find(prev_id) == current_ids.end())
        InsertUnloadMarker(transformed, queue_info.GetGPUAgent(), prev_id);

    for (size_t s=0; s<symbols.count; s++)
    {
      auto& symbol = symbols.symbols[s];
      if (active_capture_event_ids.find(symbol.att_marker_id) == active_capture_event_ids.end())
        InsertLoadMarker(transformed, queue_info.GetGPUAgent(), symbol, bool(insertStart));
    }

    active_capture_event_ids = std::move(current_ids);
  }

  // Searching across all the packets given during this write
  for (size_t i = 0; i < pkt_count; ++i)
    transformed.emplace_back(packets_arr[i]);

  if (bHasPending)
  {
    std::lock_guard<std::mutex> lk(att_enable_disable_mutex);
    auto agent_pending_packets = pending_stop_packets.at(agent_handle);

    if (agent_pending_packets.last_kernel_exec <= writer_end_id)
    {
      InsertPacketStop(transformed, agent_pending_packets, queue_info, agent_handle);
      active_capture_event_ids = {};
    }
  }

  /* Write the transformed packets to the hardware queue.  */
  writer(transformed.data(), transformed.size());
  return true;
}

}  // namespace att

}  // namespace rocprofiler
