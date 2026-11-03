/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

#include "rocaqlcapture.hpp"

#include "os/os.hpp"
#include "utils/flags.hpp"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cinttypes>
#include <cstdio>
#include <fstream>
#include <mutex>
#include <string>

#if !defined(_WIN32)
#include <sys/syscall.h>
#include <unistd.h>
#endif

namespace amd::roc {

namespace {

//! OS thread id, cached per thread: the syscall is far more expensive than the
//! rest of the record builder.
uint64_t CurrentThreadId() {
#if defined(_WIN32)
  static thread_local uint64_t cached_tid = static_cast<uint64_t>(GetCurrentThreadId());
#else
  static thread_local uint64_t cached_tid = static_cast<uint64_t>(syscall(SYS_gettid));
#endif
  return cached_tid;
}

//! Hex helpers. snprintf into a small stack buffer keeps the record builder free of
//! stream manipulators, which would otherwise have to be reset after every field.
std::string Hex(uint64_t value) {
  char buf[32];
  snprintf(buf, sizeof(buf), "0x%" PRIx64, value);
  return buf;
}

//! Minimal JSON string escaping. Demangled C++ names carry '<', '>', ',' and spaces,
//! none of which need escaping, but a name can reach here from a corrupted code object.
std::string JsonString(const char* text) {
  std::string out("\"");
  if (text != nullptr) {
    for (const char* p = text; *p != '\0'; ++p) {
      const unsigned char c = static_cast<unsigned char>(*p);
      switch (c) {
        case '"':  out += "\\\""; break;
        case '\\': out += "\\\\"; break;
        case '\n': out += "\\n";  break;
        case '\r': out += "\\r";  break;
        case '\t': out += "\\t";  break;
        default:
          if (c < 0x20) {
            char esc[8];
            snprintf(esc, sizeof(esc), "\\u%04x", c);
            out += esc;
          } else {
            out += static_cast<char>(c);
          }
      }
    }
  }
  out += '"';
  return out;
}

const char* PrefetchModeName(uint32_t mode) {
  switch (mode) {
    case 0:  return "DISABLED";
    case 1:  return "ABSOLUTE_VA";
    case 2:  return "KERNARG_OFS";
    default: return "RESERVED";
  }
}

const char* PrefetchTemporalName(uint32_t temporal) {
  switch (temporal) {
    case 0:  return "REGULAR";
    case 1:  return "NON_TEMPORAL";
    case 2:  return "HIGH_TEMPORAL";
    default: return "LAST_USE";
  }
}

const char* PrefetchScopeName(uint32_t scope) {
  switch (scope) {
    case 0:  return "CU";
    case 1:  return "SE";
    case 2:  return "DEVICE";
    default: return "SYSTEM";
  }
}

//! Owns the output stream and the sequence counter. Constructed on first use from
//! ROC_AQL_CAPTURE_OUTPUT; a failed open leaves the capture permanently disabled
//! rather than retrying (and re-logging) on every dispatch.
class CaptureState {
 public:
  static CaptureState& Instance() {
    static CaptureState state;
    return state;
  }

  bool Active() const { return active_; }

  //! Reserves the next sequence number, or returns false when the record budget
  //! (ROC_AQL_CAPTURE_MAX_RECORDS) is spent. Separate from Write() so a dispatch
  //! that will not be recorded skips building the record string entirely.
  bool TryReserve(uint64_t* seq) {
    if (!active_) {
      return false;
    }
    const uint64_t limit = ROC_AQL_CAPTURE_MAX_RECORDS;
    const uint64_t next = sequence_.fetch_add(1, std::memory_order_relaxed);
    if (limit != 0 && next >= limit) {
      return false;
    }
    *seq = next;
    return true;
  }

  void Write(const std::string& line) {
    std::lock_guard<std::mutex> lock(write_lock_);
    out_ << line << '\n';
    // Flush per record: the usual reason to run this capture is a dispatch that
    // never completes, and an unflushed tail would drop the record of interest.
    out_.flush();
  }

 private:
  CaptureState() {
    if (flagIsDefault(ROC_AQL_CAPTURE_OUTPUT)) {
      return;
    }
    std::string path = ROC_AQL_CAPTURE_OUTPUT;
    if (path.empty()) {
      return;
    }
    // Append the PID so concurrent processes do not interleave into one file.
    const size_t dot = path.find_last_of('.');
    const size_t slash = path.find_last_of("/\\");
    const std::string pid = std::to_string(amd::Os::getProcessId());
    if (dot != std::string::npos && (slash == std::string::npos || dot > slash)) {
      path = path.substr(0, dot) + "." + pid + path.substr(dot);
    } else {
      path += "." + pid;
    }

    out_.open(path, std::ios::out | std::ios::trunc);
    if (!out_.is_open()) {
      ClPrint(amd::LOG_ERROR, amd::LOG_AQL, "AQL capture: cannot open '%s' for writing",
              path.c_str());
      return;
    }
    active_ = true;
    ClPrint(amd::LOG_INFO, amd::LOG_AQL, "AQL capture: writing dispatch records to '%s'",
            path.c_str());
  }

  bool active_ = false;
  std::ofstream out_;
  std::mutex write_lock_;
  std::atomic<uint64_t> sequence_{0};
};

void AppendHeaderFields(std::string& rec, uint16_t header) {
  rec += "\"header\":{\"raw\":" + JsonString(Hex(header).c_str());
  rec += ",\"type\":" +
      std::to_string(
          (header >> HSA_PACKET_HEADER_TYPE) & ((1 << HSA_PACKET_HEADER_WIDTH_TYPE) - 1));
  rec += ",\"barrier\":" +
      std::to_string(
          (header >> HSA_PACKET_HEADER_BARRIER) & ((1 << HSA_PACKET_HEADER_WIDTH_BARRIER) - 1));
  rec += ",\"scacquire_scope\":" +
      std::to_string((header >> HSA_PACKET_HEADER_SCACQUIRE_FENCE_SCOPE) &
                     ((1 << HSA_PACKET_HEADER_WIDTH_SCACQUIRE_FENCE_SCOPE) - 1));
  rec += ",\"screlease_scope\":" +
      std::to_string((header >> HSA_PACKET_HEADER_SCRELEASE_FENCE_SCOPE) &
                     ((1 << HSA_PACKET_HEADER_WIDTH_SCRELEASE_FENCE_SCOPE) - 1));
  rec += "}";
}

void AppendAqlPacket(std::string& rec, const AqlDispatchCaptureInput& in) {
  rec += "\"aql\":{";
  AppendHeaderFields(rec, in.header);
  rec += ",\"ext\":";
  rec += in.ext_dispatch ? "true" : "false";

  if (in.ext_dispatch) {
    const auto* pkt = static_cast<const hsa_amd_ext_kernel_dispatch_packet_t*>(in.packet);
    // On the ext packet byte+2 is amd_format and the dimensions live in byte+3, both
    // of which travel in |rest| rather than in the packet struct the caller holds.
    rec += ",\"amd_format\":" + std::to_string(in.rest & 0xFF);
    rec += ",\"setup\":" + std::to_string((in.rest >> 8) & 0xFF);
    rec += ",\"cluster_count\":[" + std::to_string(pkt->cluster_count_x) + "," +
        std::to_string(pkt->cluster_count_y) + "," + std::to_string(pkt->cluster_count_z) + "]";
    rec += ",\"cluster_size\":[" + std::to_string(pkt->cluster_size_x) + "," +
        std::to_string(pkt->cluster_size_y) + "," + std::to_string(pkt->cluster_size_z) + "]";
    rec += ",\"workgroup_size\":[" + std::to_string(pkt->workgroup_size_x) + "," +
        std::to_string(pkt->workgroup_size_y) + "," + std::to_string(pkt->workgroup_size_z) + "]";
    rec += ",\"private_segment_size\":" + std::to_string(pkt->private_segment_size);
    rec += ",\"group_segment_size\":" + std::to_string(pkt->group_segment_size);
    rec += ",\"group_mem_carveout\":" +
        std::to_string(static_cast<unsigned>(pkt->perf_hint.group_mem_carveout));
    rec += ",\"kernel_object\":" + JsonString(Hex(pkt->kernel_object).c_str());
    rec += ",\"kernarg_address\":" +
        JsonString(Hex(reinterpret_cast<uint64_t>(pkt->kernarg_address)).c_str());
    rec += ",\"dep_signal\":" + JsonString(Hex(pkt->dep_signal.handle).c_str());
    rec += ",\"completion_signal\":" + JsonString(Hex(pkt->completion_signal.handle).c_str());
  } else {
    const auto* pkt = static_cast<const hsa_kernel_dispatch_packet_t*>(in.packet);
    rec += ",\"setup\":" + std::to_string(in.rest);
    rec += ",\"grid_size\":[" + std::to_string(pkt->grid_size_x) + "," +
        std::to_string(pkt->grid_size_y) + "," + std::to_string(pkt->grid_size_z) + "]";
    rec += ",\"workgroup_size\":[" + std::to_string(pkt->workgroup_size_x) + "," +
        std::to_string(pkt->workgroup_size_y) + "," + std::to_string(pkt->workgroup_size_z) + "]";
    rec += ",\"private_segment_size\":" + std::to_string(pkt->private_segment_size);
    rec += ",\"group_segment_size\":" + std::to_string(pkt->group_segment_size);
    rec += ",\"kernel_object\":" + JsonString(Hex(pkt->kernel_object).c_str());
    rec += ",\"kernarg_address\":" +
        JsonString(Hex(reinterpret_cast<uint64_t>(pkt->kernarg_address)).c_str());
    rec += ",\"completion_signal\":" + JsonString(Hex(pkt->completion_signal.handle).c_str());
  }
  rec += "}";
}

void AppendPrefetchRegion(std::string& rec, const amd_data_prefetch_t& pf, uint32_t slot) {
  rec += "{\"slot\":" + std::to_string(slot);
  rec += ",\"mode\":" + std::to_string(pf.mode);
  rec += ",\"mode_name\":" + JsonString(PrefetchModeName(pf.mode));
  if (pf.mode == 0) {
    // DISABLED: the remaining fields are don't-care, and decoding them invites
    // mistaking a skipped slot for a configured one.
    rec += "}";
    return;
  }
  if (pf.mode == 1) {
    // ABSOLUTE_VA: addr_lo is VA[31:8], addr_hi is VA[56:32].
    const uint64_t va =
        (static_cast<uint64_t>(pf.addr_lo) << 8) | (static_cast<uint64_t>(pf.addr_hi) << 32);
    rec += ",\"address\":" + JsonString(Hex(va).c_str());
  } else {
    // KERNARG_OFS: addr_lo[11:0] is a byte offset into the kernarg buffer.
    rec += ",\"kernarg_offset\":" + std::to_string(pf.addr_lo & 0xFFFu);
  }
  rec += ",\"addr_lo\":" + JsonString(Hex(pf.addr_lo).c_str());
  rec += ",\"addr_hi\":" + JsonString(Hex(pf.addr_hi).c_str());
  // burst_size and num_burst are (N - 1) encoded; report both the raw field and
  // the decoded byte/element count so a capture can be diffed against the ISA doc.
  rec += ",\"burst_size_raw\":" + std::to_string(pf.burst_size);
  rec += ",\"burst_size_bytes\":" + std::to_string((static_cast<uint64_t>(pf.burst_size) + 1) * 256);
  rec += ",\"num_burst_raw\":" + std::to_string(pf.num_burst);
  rec += ",\"num_burst\":" + std::to_string(static_cast<uint64_t>(pf.num_burst) + 1);
  rec += ",\"stride_raw\":" + std::to_string(pf.stride);
  rec += ",\"stride_bytes\":" + std::to_string(static_cast<uint64_t>(pf.stride) * 256);
  rec += ",\"chunk_size\":" + std::to_string(pf.chunk_size);
  rec += ",\"translation_permissions\":" + std::to_string(pf.translation_permissions);
  rec += ",\"cooperative\":" + std::to_string(pf.cooperative);
  rec += ",\"temporal\":" + std::to_string(pf.temporal);
  rec += ",\"temporal_name\":" + JsonString(PrefetchTemporalName(pf.temporal));
  rec += ",\"scope\":" + std::to_string(pf.scope);
  rec += ",\"scope_name\":" + JsonString(PrefetchScopeName(pf.scope));
  rec += "}";
}

void AppendLaunchDescriptor(std::string& rec, const amd_launch_descriptor_t& ld) {
  rec += "\"launch_descriptor\":{";

  // Raw dwords first: a version the decoder below does not know about still leaves
  // the capture usable.
  constexpr size_t kDwords = sizeof(amd_launch_descriptor_t) / sizeof(uint32_t);
  const auto* dwords = reinterpret_cast<const uint32_t*>(&ld);
  rec += "\"raw\":[";
  for (size_t i = 0; i < kDwords; ++i) {
    if (i != 0) rec += ",";
    rec += JsonString(Hex(dwords[i]).c_str());
  }
  rec += "]";

  rec += ",\"version\":" + std::to_string(ld.version);
  rec += ",\"priority\":" + std::to_string(ld.priority);
  rec += ",\"pm_hint\":" + std::to_string(ld.pm_hint);
  rec += ",\"cu_enable\":{\"raw\":" + JsonString(Hex(ld.cu_enable).c_str());
  rec += ",\"cu_start\":" + std::to_string(ld.cu_start);
  rec += ",\"cu_count\":" + std::to_string(ld.cu_count);
  rec += ",\"se_en\":" + std::to_string(ld.se_en) + "}";
  rec += ",\"dispatch_granularity_limiter\":" + std::to_string(ld.dispatch_granularity_limiter);
  rec += ",\"tg_chunk_size\":" + std::to_string(ld.tg_chunk_size);

  rec += ",\"prefetch\":[";
  for (uint32_t i = 0; i < AMD_LAUNCH_DESCRIPTOR_MAX_PREFETCH_REGIONS; ++i) {
    if (i != 0) rec += ",";
    AppendPrefetchRegion(rec, ld.prefetch[i], i);
  }
  rec += "]}";
}

//! The preloaded kernargs are split across three arrays separated by the interleaved
//! header dwords, so map a logical dword index onto the right section.
uint32_t PreloadDword(const hsa_amd_metadata_kernel_dispatch_packet_t& meta, uint16_t dword_idx) {
  constexpr uint16_t kSection0 = sizeof(meta.kernarg_preload_0_14) / sizeof(uint32_t);
  constexpr uint16_t kSection1 = sizeof(meta.kernarg_preload_15_29) / sizeof(uint32_t);
  if (dword_idx < kSection0) {
    return meta.kernarg_preload_0_14[dword_idx];
  }
  if (dword_idx < kSection0 + kSection1) {
    return meta.kernarg_preload_15_29[dword_idx - kSection0];
  }
  return meta.kernarg_preload_30_31[dword_idx - kSection0 - kSection1];
}

void AppendMetadata(std::string& rec, const hsa_amd_metadata_kernel_dispatch_packet_t& meta) {
  const auto& desc = meta.kernel_descriptor;

  rec += "\"metadata\":{";
  rec += "\"headers\":[" + JsonString(Hex(meta.header0).c_str()) + "," +
      JsonString(Hex(meta.header1).c_str()) + "," + JsonString(Hex(meta.header2).c_str()) + "," +
      JsonString(Hex(meta.header3).c_str()) + "]";
  rec += ",\"event_id\":" + JsonString(Hex(meta.event_id).c_str());

  rec += ",\"kernel_descriptor\":{";
  rec += "\"kernel_code_entry_byte_offset\":" +
      JsonString(Hex(static_cast<uint64_t>(desc.kernel_code_entry_byte_offset)).c_str());
  rec += ",\"compute_pgm_rsrc1\":" + JsonString(Hex(desc.compute_pgm_rsrc1).c_str());
  rec += ",\"compute_pgm_rsrc2\":" + JsonString(Hex(desc.compute_pgm_rsrc2).c_str());
  rec += ",\"compute_pgm_rsrc3\":" + JsonString(Hex(desc.compute_pgm_rsrc3).c_str());
  rec += ",\"kernel_code_properties\":" + JsonString(Hex(desc.kernel_code_properties).c_str());
  rec += ",\"kernarg_preload\":{\"length\":" + std::to_string(desc.kernarg_preload.length);
  rec += ",\"offset\":" + std::to_string(desc.kernarg_preload.offset) + "}}";

  // Clamp before indexing: hang analysis can reach here with a corrupted descriptor.
  constexpr uint16_t kMaxPreloadDwords =
      (sizeof(meta.kernarg_preload_0_14) + sizeof(meta.kernarg_preload_15_29) +
       sizeof(meta.kernarg_preload_30_31)) / sizeof(uint32_t);
  const uint16_t preload_dwords =
      std::min<uint16_t>(desc.kernarg_preload.length, kMaxPreloadDwords);
  rec += ",\"kernarg_preload_dwords\":[";
  for (uint16_t i = 0; i < preload_dwords; ++i) {
    if (i != 0) rec += ",";
    rec += JsonString(Hex(PreloadDword(meta, i)).c_str());
  }
  rec += "],";

  AppendLaunchDescriptor(rec, meta.launch_descriptor);
  rec += "}";
}

void AppendKernargs(std::string& rec, const AqlDispatchCaptureInput& in) {
  const void* kernarg_address = in.ext_dispatch
      ? static_cast<const hsa_amd_ext_kernel_dispatch_packet_t*>(in.packet)->kernarg_address
      : static_cast<const hsa_kernel_dispatch_packet_t*>(in.packet)->kernarg_address;

  rec += "\"kernargs\":{\"segment_size\":" + std::to_string(in.kernarg_segment_size);

  const uint32_t budget = ROC_AQL_CAPTURE_KERNARG_BYTES;
  const uint32_t captured = std::min(budget, in.kernarg_segment_size);
  if (kernarg_address == nullptr || captured == 0) {
    rec += ",\"captured\":0}";
    return;
  }

  // The kernarg buffer is host-written and host-visible, and the GPU never writes it,
  // so reading it back right after the packet is published is safe. The slot can be
  // recycled by a later dispatch, which is why this runs inline rather than deferred.
  const auto* bytes = static_cast<const uint8_t*>(kernarg_address);
  std::string blob;
  blob.reserve(static_cast<size_t>(captured) * 3);
  char byte_text[4];
  for (uint32_t i = 0; i < captured; ++i) {
    snprintf(byte_text, sizeof(byte_text), "%02x", bytes[i]);
    if (i != 0) blob += ' ';
    blob += byte_text;
  }
  rec += ",\"captured\":" + std::to_string(captured);
  rec += ",\"bytes\":" + JsonString(blob.c_str()) + "}";
}

}  // namespace

// ================================================================================================
bool AqlCapture::Enabled() {
  static const bool enabled = CaptureState::Instance().Active();
  return enabled;
}

// ================================================================================================
void AqlCapture::Record(const AqlDispatchCaptureInput& in) {
  uint64_t seq = 0;
  if (!CaptureState::Instance().TryReserve(&seq)) {
    return;
  }

  const auto now = std::chrono::steady_clock::now().time_since_epoch();
  const auto now_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(now).count();

  std::string rec;
  rec.reserve(2048);
  rec += "{\"seq\":" + std::to_string(seq);
  rec += ",\"t_ns\":" + std::to_string(now_ns);
  rec += ",\"pid\":" + std::to_string(amd::Os::getProcessId());
  rec += ",\"tid\":" + std::to_string(CurrentThreadId());
  rec += ",\"device\":" + std::to_string(in.device_index);
  rec += ",\"queue\":" + JsonString(Hex(reinterpret_cast<uint64_t>(in.queue)).c_str());
  rec += ",\"queue_id\":" + std::to_string(in.queue != nullptr ? in.queue->id : 0);
  rec += ",\"priority\":" + std::to_string(in.priority);
  rec += ",\"wptr\":" + std::to_string(in.slot_index);
  rec += ",\"kernel\":" + JsonString(in.kernel_name != nullptr ? in.kernel_name : "<unknown>");
  rec += ",";
  AppendAqlPacket(rec, in);
  rec += ",";
  if (in.metadata != nullptr) {
    AppendMetadata(rec, *in.metadata);
  } else {
    // No metadata ring on this queue (pre-gfx12.5, or DEBUG_CLR_ENABLE_KDQ=0).
    rec += "\"metadata\":null";
  }
  rec += ",";
  AppendKernargs(rec, in);
  rec += "}";

  CaptureState::Instance().Write(rec);
}

}  // namespace amd::roc
