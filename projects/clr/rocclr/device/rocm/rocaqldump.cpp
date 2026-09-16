/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

#include "device/rocm/rocaqldump.hpp"
#include "device/rocm/rocdevice.hpp"
#include "device/rocm/rockernel.hpp"
#include "device/rocm/rocvirtual.hpp"
#include "platform/kernel.hpp"
#include "os/os.hpp"
#include "utils/debug.hpp"

#include <algorithm>
#include <iterator>
#include <chrono>
#include <cinttypes>
#include <cstdio>
#include <filesystem>
#include <string>
#include <thread>

namespace amd::roc {

namespace {

unsigned HeaderBits(uint16_t v, unsigned pos, unsigned width) {
  return (v >> pos) & ((1u << width) - 1);
}

std::string Hex(uint64_t v) {
  char buf[24];
  snprintf(buf, sizeof(buf), "\"0x%" PRIx64 "\"", v);
  return buf;
}

std::string BlobHex(const uint8_t* data, size_t size) {
  static constexpr char kDigits[] = "0123456789abcdef";
  std::string s(2 * size + 2, '"');
  for (size_t i = 0; i < size; ++i) {
    s[1 + 2 * i] = kDigits[data[i] >> 4];
    s[2 + 2 * i] = kDigits[data[i] & 0xF];
  }
  return s;
}

std::string JsonString(const std::string& in) {
  std::string s = "\"";
  for (char c : in) {
    if (c == '"' || c == '\\') {
      s += '\\';
      s += c;
    } else if (static_cast<unsigned char>(c) < 0x20) {
      char buf[8];
      snprintf(buf, sizeof(buf), "\\u%04x", static_cast<unsigned char>(c));
      s += buf;
    } else {
      s += c;
    }
  }
  s += '"';
  return s;
}

}  // namespace

// ================================================================================================
AqlDispatchDumper::AqlDispatchDumper() {
  std::error_code ec;
  std::filesystem::create_directories(GPU_DUMP_AQL_DISPATCH, ec);
  const std::string path = std::string(GPU_DUMP_AQL_DISPATCH) + "/aql_dispatch_" +
                           std::to_string(amd::Os::getProcessId()) + ".jsonl";
  out_.open(path, std::ios::app);
  if (!out_.is_open()) {
    LogPrintfError("AQL dispatch dump disabled, cannot open %s", path.c_str());
    return;
  }
  poller_ = std::thread(&AqlDispatchDumper::PollLoop, this);
}

// ================================================================================================
AqlDispatchDumper& AqlDispatchDumper::Instance() {
  // Never destroyed: teardown blit kernels can dispatch after static destructors have run.
  static AqlDispatchDumper* dumper = new AqlDispatchDumper();
  return *dumper;
}

// ================================================================================================
void AqlDispatchDumper::Record(const Device& dev, const hsa_queue_t* queue, uint64_t index,
                               const void* packet, bool ext_packet, uint16_t header, uint16_t rest,
                               Origin origin, ProfilingSignal* signal) {
  AqlDispatchDumper& self = Instance();
  std::lock_guard<std::mutex> lock(self.mutex_);
  if (!self.out_.is_open()) {
    return;
  }

  const auto* pkt = reinterpret_cast<const hsa_kernel_dispatch_packet_t*>(packet);
  // On an ext packet the low byte of `rest` is amd_format and setup travels in the high byte.
  const uint16_t setup = ext_packet ? static_cast<uint16_t>(rest >> 8) : rest;

  std::string s;
  s.reserve(4096);
  s += "{\"event\":\"submit\",\"ts_ns\":" + std::to_string(amd::Os::timeNanos());
  s += ",\"pid\":" + std::to_string(amd::Os::getProcessId());
  s += ",\"tid\":" + Hex(std::hash<std::thread::id>{}(std::this_thread::get_id()));
  s += ",\"origin\":";
  s += (origin == Origin::kGraph) ? "\"graph\"" : "\"eager\"";
  s += ",\"device\":" + std::to_string(dev.index());
  s += ",\"queue_id\":" + std::to_string(queue->id);
  s += ",\"queue_base\":" + Hex(reinterpret_cast<uint64_t>(queue->base_address));
  s += ",\"packet_index\":" + std::to_string(index);
  s += ",\"ext_packet\":";
  s += ext_packet ? "true" : "false";
  s += ",\"header_raw\":" + Hex(header);
  s += ",\"header_type\":" +
       std::to_string(HeaderBits(header, HSA_PACKET_HEADER_TYPE, HSA_PACKET_HEADER_WIDTH_TYPE));
  s += ",\"header_barrier\":" + std::to_string(HeaderBits(header, HSA_PACKET_HEADER_BARRIER,
                                                          HSA_PACKET_HEADER_WIDTH_BARRIER));
  s += ",\"header_acquire\":" +
       std::to_string(HeaderBits(header, HSA_PACKET_HEADER_SCACQUIRE_FENCE_SCOPE,
                                 HSA_PACKET_HEADER_WIDTH_SCACQUIRE_FENCE_SCOPE));
  s += ",\"header_release\":" +
       std::to_string(HeaderBits(header, HSA_PACKET_HEADER_SCRELEASE_FENCE_SCOPE,
                                 HSA_PACKET_HEADER_WIDTH_SCRELEASE_FENCE_SCOPE));
  s += ",\"setup_raw\":" + Hex(setup);
  s += ",\"dims\":" + std::to_string(HeaderBits(setup, HSA_KERNEL_DISPATCH_PACKET_SETUP_DIMENSIONS,
                                                HSA_KERNEL_DISPATCH_PACKET_SETUP_WIDTH_DIMENSIONS));

  if (ext_packet) {
    const auto* ext = reinterpret_cast<const hsa_amd_ext_kernel_dispatch_packet_t*>(packet);
    s += ",\"cluster_count\":[" + std::to_string(ext->cluster_count_x) + "," +
         std::to_string(ext->cluster_count_y) + "," + std::to_string(ext->cluster_count_z) + "]";
    s += ",\"cluster_size\":[" + std::to_string(ext->cluster_size_x) + "," +
         std::to_string(ext->cluster_size_y) + "," + std::to_string(ext->cluster_size_z) + "]";
  } else {
    s += ",\"grid\":[" + std::to_string(pkt->grid_size_x) + "," + std::to_string(pkt->grid_size_y) +
         "," + std::to_string(pkt->grid_size_z) + "]";
  }

  s += ",\"workgroup\":[" + std::to_string(pkt->workgroup_size_x) + "," +
       std::to_string(pkt->workgroup_size_y) + "," + std::to_string(pkt->workgroup_size_z) + "]";
  s += ",\"group_segment_size\":" + std::to_string(pkt->group_segment_size);
  s += ",\"private_segment_size\":" + std::to_string(pkt->private_segment_size);
  s += ",\"kernel_object\":" + Hex(pkt->kernel_object);
  s += ",\"kernarg_address\":" + Hex(reinterpret_cast<uint64_t>(pkt->kernarg_address));
  if (origin == Origin::kEager) {
    s += ",\"completion_signal\":" + Hex(pkt->completion_signal.handle);
  }

  std::string demangled = "<unknown>";
  auto kit = dev.KernelMap().find(pkt->kernel_object);
  if (kit == dev.KernelMap().end()) {
    s += ",\"kernel_name\":\"<unknown>\"";
  } else {
    Kernel& kernel = kit->second;
    const device::Kernel::WorkGroupInfo& wgi = *kernel.workGroupInfo();
    demangled = kernel.getDemangledName();
    s += ",\"kernel_name\":" + JsonString(kernel.name());
    s += ",\"kernel_demangled\":" + JsonString(demangled);
    s += ",\"kernel_internal\":";
    s += kernel.isInternalKernel() ? "true" : "false";
    s += ",\"kernarg_segment_size\":" + std::to_string(kernel.KernargSegmentByteSize());
    s += ",\"static_group_segment_size\":" + std::to_string(kernel.WorkgroupGroupSegmentByteSize());
    s += ",\"wavefront_size\":" + std::to_string(wgi.wavefrontSize_);
    s += ",\"used_vgprs\":" + std::to_string(wgi.usedVGPRs_);
    s += ",\"used_sgprs\":" + std::to_string(wgi.usedSGPRs_);
    s += ",\"used_lds\":" + std::to_string(wgi.usedLDSSize_);
    s += ",\"max_occupancy_per_cu\":" + std::to_string(wgi.maxOccupancyPerCu_);

    if (pkt->kernarg_address != nullptr && kernel.KernargSegmentByteSize() != 0) {
      const amd::KernelSignature& signature = kernel.signature();
      const auto* blob = reinterpret_cast<const uint8_t*>(pkt->kernarg_address);
      const size_t blobSize =
          std::min<size_t>(kernel.KernargSegmentByteSize(), signature.paramsSize());
      s += ",\"args\":[";
      for (uint32_t i = 0; i < signature.numParametersAll(); ++i) {
        const amd::KernelParameterDescriptor& desc = signature.at(i);
        const size_t begin = std::min(blobSize, desc.offset_);
        const size_t len = std::min(desc.size_, blobSize - begin);
        if (i != 0) {
          s += ',';
        }
        s += "{\"index\":" + std::to_string(i);
        s += ",\"name\":" + JsonString(desc.name_);
        s += ",\"type_name\":" + JsonString(desc.typeName_);
        s += ",\"offset\":" + std::to_string(desc.offset_);
        s += ",\"size\":" + std::to_string(desc.size_);
        s += ",\"hidden\":";
        s += desc.info_.hidden_ ? "true" : "false";
        s += ",\"kind\":" + std::to_string(desc.info_.oclObject_);
        s += ",\"value_hex\":" + BlobHex(blob + begin, len);
        s += '}';
      }
      s += "],\"kernarg_blob_hex\":" + BlobHex(blob, blobSize);
    }
  }

  s += "}\n";
  self.out_ << s;
  self.out_.flush();

  if (signal != nullptr && !self.stop_) {
    signal->retain();
    self.inflight_.push_back(
        {signal, dev.getBackendDevice(), queue->id, index, origin, std::move(demangled)});
    self.cv_.notify_one();
  }
}

// ================================================================================================
void AqlDispatchDumper::PollLoop() {
  std::unique_lock<std::mutex> lock(mutex_);
  while (!stop_) {
    if (inflight_.empty()) {
      cv_.wait(lock);
      continue;
    }
    Drain();
    cv_.wait_for(lock, std::chrono::microseconds(50));
  }
}

// ================================================================================================
void AqlDispatchDumper::Drain() {
  auto pending = std::partition(inflight_.begin(), inflight_.end(), [](const InFlight& e) {
    return Hsa::signal_load_relaxed(e.signal->signal_) != 0;
  });
  std::vector<InFlight> done(std::make_move_iterator(pending),
                             std::make_move_iterator(inflight_.end()));
  inflight_.erase(pending, inflight_.end());
  std::sort(done.begin(), done.end(),
            [](const InFlight& a, const InFlight& b) { return a.packet_index < b.packet_index; });

  for (size_t k = 0; k < done.size(); ++k) {
    const InFlight& e = done[k];
    uint64_t beginTicks = 0;
    uint64_t endTicks = 0;
    fetchSignalTime(e.signal->signal_, e.agent, &beginTicks, &endTicks);
    const double ticksToTime = Timestamp::getGpuTicksToTime();
    const uint64_t beginNs = static_cast<uint64_t>(beginTicks * ticksToTime);
    const uint64_t endNs = static_cast<uint64_t>(endTicks * ticksToTime);

    uint64_t nextIndex = 0;
    bool hasNext = false;
    auto consider = [&](const InFlight& other) {
      if (other.queue_id == e.queue_id && (!hasNext || other.packet_index < nextIndex)) {
        nextIndex = other.packet_index;
        hasNext = true;
      }
    };
    for (const InFlight& other : inflight_) {
      consider(other);
    }
    for (size_t j = k + 1; j < done.size(); ++j) {
      consider(done[j]);
    }

    std::string s = "{\"event\":\"complete\",\"ts_ns\":" + std::to_string(amd::Os::timeNanos());
    s += ",\"queue_id\":" + std::to_string(e.queue_id);
    s += ",\"packet_index\":" + std::to_string(e.packet_index);
    s += ",\"origin\":";
    s += (e.origin == Origin::kGraph) ? "\"graph\"" : "\"eager\"";
    s += ",\"kernel_demangled\":" + JsonString(e.kernel);
    s += ",\"gpu_begin_ns\":" + std::to_string(beginNs);
    s += ",\"gpu_end_ns\":" + std::to_string(endNs);
    s += ",\"gpu_duration_ns\":" + std::to_string(endNs >= beginNs ? endNs - beginNs : 0);
    s += ",\"next_packet_index\":";
    s += hasNext ? std::to_string(nextIndex) : "null";
    s += "}\n";
    out_ << s;
    out_.flush();
    e.signal->release();
  }
}

// ================================================================================================
void AqlDispatchDumper::Shutdown() {
  if (!Enabled()) {
    return;
  }
  AqlDispatchDumper& self = Instance();
  {
    std::lock_guard<std::mutex> lock(self.mutex_);
    self.stop_ = true;
  }
  self.cv_.notify_one();
  if (self.poller_.joinable()) {
    self.poller_.join();
  }
  std::lock_guard<std::mutex> lock(self.mutex_);
  self.Drain();
  for (InFlight& e : self.inflight_) {
    Hsa::signal_silent_store_relaxed(e.signal->signal_, 0);
    e.signal->release();
  }
  self.inflight_.clear();
}

}  // namespace amd::roc
