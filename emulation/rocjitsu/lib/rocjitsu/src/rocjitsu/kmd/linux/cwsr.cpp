// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

/// @file cwsr.cpp
/// @brief CWSR (context-save-restore) area serialization for rocm-dbgapi.

#include "rocjitsu/kmd/linux/cwsr.h"

#include <algorithm>
#include <cstring>
#include <limits>

namespace rocjitsu {
namespace kmd {

namespace {

// Layout constants matching gfx9_4/mi cwsr_record_t in rocdbgapi.
constexpr uint32_t kHwregCount = 32; // hwreg_count()
constexpr uint32_t kTtmpCount = 16;  // ttmps saved at the top of the hwreg block
constexpr uint32_t kVgprLaneBytes = 64 * sizeof(uint32_t); // one VGPR = 64 lanes * 4 bytes
constexpr uint32_t kMaxSgprs = 106;
constexpr uint32_t kMaxVgprs = 256;
// gfx9.4: scalar_register_count() (102) + scalar_alias_count() (6). Determines
// where VCC/FLAT_SCRATCH alias into the saved SGPR block.
constexpr uint32_t kArchScalars = 108;
constexpr uint32_t kCwsrHeaderBytes = 10 * sizeof(uint32_t);
constexpr uint32_t kControlStackOffset = 0x100u;

// COMPUTE_RELAUNCH classification bits (control_stack_iterate): a word with bit
// 30 set is an event (skipped) and a word with bit 31 set is a state word; a
// wave word has both clear.
constexpr uint32_t kRelaunchStateBit = 1u << 31;

constexpr uint32_t round_up(uint32_t v, uint32_t m) { return (v / m + (v % m != 0)) * m; }

// Encode the COMPUTE_RELAUNCH "state" word so rocm-dbgapi decodes exactly
// @vgpr_count / @sgpr_count with zero accumulation (ACC) VGPRs and no LDS:
//   vgpr_count      = (accum_offset[24:29] + 1) * 4
//   acc_vgpr_count  = (vgprs[0:5]      + 1) * 8 - vgpr_count   (== 0 here)
//   sgpr_count      = (sgprs[6:8]      + 1) * 16 - 16
//   lds_size        = lds[9:16] * ...   (== 0 here)
uint32_t encode_state_word(uint32_t vgpr_count, uint32_t sgpr_count) {
  uint32_t vgprs_field = (vgpr_count / 8) - 1;  // acc == 0  =>  vgpr_count = (vgprs+1)*8
  uint32_t accum_offset = (vgpr_count / 4) - 1; // vgpr_count = (accum_offset+1)*4
  uint32_t sgprs_field = sgpr_count / 16;       // sgpr_count = sgprs_field*16
  uint32_t w = 0;
  w |= (vgprs_field & 0x3Fu);
  w |= (sgprs_field & 0x7u) << 6;
  // lds field [9:16] left 0.
  w |= (accum_offset & 0x3Fu) << 24;
  w |= kRelaunchStateBit;
  return w;
}

// Encode the COMPUTE_RELAUNCH "wave" word (bits 30/31 clear so it is neither an
// event nor a state word).
uint32_t encode_wave_word(bool first_wave, bool last_wave, uint32_t scratch_scoreboard_id) {
  uint32_t w = 0;
  // scratch_scoreboard_id[0:8] locates the wave's private memory; se_id[9:11]=0,
  // scratch_en[15]=0.
  w |= (scratch_scoreboard_id & 0x1FFu);
  if (last_wave)
    w |= 1u << 16;
  if (first_wave)
    w |= 1u << 17;
  return w;
}

uint32_t encode_ttmp6(const CwsrWaveState &w) {
  uint32_t v = 0;
  if (w.wave_stopped)
    v |= 1u << 30;
  if (w.saved_status_halt)
    v |= 1u << 29;
  v |= (w.trap_id & 0xFu) << 25;
  // bit 31 (spi_ttmps_setup_disabled) reflects whether the SPI initialized the
  // dispatch bookkeeping TTMPs (group ids in ttmp8-10, packet id in ttmp11).
  // When the process runtime-enabled without ttmp-save (kfd_runtime_info
  // ttmp_setup == 0), those registers are not meaningful, so mark the wave
  // accordingly. rocm-dbgapi (>= r_debug v10) then skips packet/workgroup
  // correlation and uses its dummy dispatch instead of validating a packet id
  // against the queue's read/write dispatch ids (rocdbgapi architecture.cpp
  // spi_ttmps_setup_enabled, queue.cpp get_os_queue_packet_id).
  if (!w.spi_ttmps_setup)
    v |= 1u << 31;
  return v;
}

uint32_t encode_ttmp11(const CwsrWaveState &w) {
  uint32_t v = 0;
  v |= (w.wave_in_group & 0x3Fu);
  v |= (w.queue_packet_id & 0x1FFFFFFu) << 6; // [6:30]
  v |= 1u << 31;                              // trap_handler_ttmps_setup
  return v;
}

struct CwsrGeometry {
  CwsrLayout layout{};
  uint32_t vgpr_count = 0;
  uint32_t sgpr_count = 112;
  uint32_t vcc_lo_slot = 0;
  uint32_t hwreg_bytes = 0;
  uint32_t sgpr_bytes = 0;
  uint32_t vgpr_bytes = 0;
  uint64_t per_wave = 0;
};

CwsrGeometry compute_geometry(uint32_t area_size, const std::vector<CwsrWaveState> &waves) {
  CwsrGeometry geometry{};
  if (waves.empty() || waves.size() > std::numeric_limits<uint32_t>::max())
    return geometry;

  uint32_t max_vgprs = 0;
  for (const auto &wave : waves) {
    if (wave.num_vgprs > kMaxVgprs || wave.num_sgprs > kMaxSgprs || wave.trap_id > 0xFu ||
        wave.wave_in_group > 0x3Fu || wave.queue_packet_id > 0x1FFFFFFu)
      return geometry;
    max_vgprs = std::max(max_vgprs, wave.num_vgprs);
  }

  geometry.vgpr_count = std::max<uint32_t>(round_up(max_vgprs, 8), 8);
  geometry.vcc_lo_slot = std::min(kArchScalars, geometry.sgpr_count) - 2;
  geometry.hwreg_bytes = kHwregCount * sizeof(uint32_t);
  geometry.sgpr_bytes = geometry.sgpr_count * sizeof(uint32_t);
  geometry.vgpr_bytes = geometry.vgpr_count * kVgprLaneBytes;
  geometry.per_wave = 64u + geometry.hwreg_bytes + geometry.sgpr_bytes + geometry.vgpr_bytes;

  const uint64_t num_waves = waves.size();
  const uint64_t wave_state_size = geometry.per_wave * num_waves;
  const uint64_t control_stack_size = (3u + num_waves) * sizeof(uint32_t);
  const uint64_t wave_area_begin = kControlStackOffset + control_stack_size;
  const uint64_t wave_state_offset = wave_area_begin + wave_state_size;
  if (control_stack_size > std::numeric_limits<uint32_t>::max() ||
      wave_state_size > std::numeric_limits<uint32_t>::max() ||
      wave_state_offset > std::numeric_limits<uint32_t>::max() || wave_state_offset > area_size)
    return geometry;

  constexpr uint32_t kDebuggerBytesPerWave = 32;
  constexpr uint32_t kDebuggerBytesAlign = 64;
  constexpr uint32_t kDebuggerReserveChunks = 8;
  const uint64_t debug_size_64 =
      ((num_waves + kDebuggerReserveChunks) * kDebuggerBytesPerWave + kDebuggerBytesAlign - 1) /
      kDebuggerBytesAlign * kDebuggerBytesAlign;
  const uint64_t debug_offset_64 =
      (wave_state_offset + kDebuggerBytesAlign - 1) / kDebuggerBytesAlign * kDebuggerBytesAlign;
  const bool debug_fits = debug_offset_64 <= std::numeric_limits<uint32_t>::max() &&
                          debug_size_64 <= std::numeric_limits<uint32_t>::max() &&
                          debug_offset_64 + debug_size_64 <= area_size;

  geometry.layout.control_stack_offset = kControlStackOffset;
  geometry.layout.control_stack_size = static_cast<uint32_t>(control_stack_size);
  geometry.layout.wave_state_offset = static_cast<uint32_t>(wave_state_offset);
  geometry.layout.wave_state_size = static_cast<uint32_t>(wave_state_size);
  geometry.layout.debug_offset = debug_fits ? static_cast<uint32_t>(debug_offset_64) : 0;
  geometry.layout.debug_size = debug_fits ? static_cast<uint32_t>(debug_size_64) : 0;
  geometry.layout.ok = true;
  return geometry;
}

uint32_t image_word(std::span<const uint8_t> image, uint32_t offset) {
  uint32_t value = 0;
  std::memcpy(&value, image.data() + offset, sizeof(value));
  return value;
}

} // namespace

CwsrLayout serialize_queue_cwsr(uint64_t ctx_base, uint32_t area_size,
                                const std::vector<CwsrWaveState> &waves,
                                const std::function<void(uint64_t, uint32_t)> &write32) {
  CwsrLayout layout{};
  if (waves.empty() || !write32 || (ctx_base & (alignof(uint32_t) - 1)) != 0)
    return layout;

  // Uniform per-dispatch register geometry. ACC-VGPRs and LDS are not modeled
  // in the save area (state word encodes zero of each), which keeps the layout
  // to [TTMP|HWREG]/[SGPR]/[VGPR] per wave.
  const CwsrGeometry geometry = compute_geometry(area_size, waves);
  if (!geometry.layout.ok)
    return layout;
  layout = geometry.layout;
  const uint32_t vgpr_count = geometry.vgpr_count;
  const uint32_t sgpr_count = geometry.sgpr_count;
  const uint32_t vcc_lo_slot = geometry.vcc_lo_slot;
  const uint32_t hwreg_bytes = geometry.hwreg_bytes;
  const uint32_t sgpr_bytes = geometry.sgpr_bytes;
  const uint32_t vgpr_bytes = geometry.vgpr_bytes;
  const uint64_t num_waves = waves.size();
  const uint64_t wave_state_offset = layout.wave_state_offset;
  if (ctx_base > std::numeric_limits<uint64_t>::max() - wave_state_offset)
    return CwsrLayout{};

  // Write the payload first so a newly allocated area is not advertised before
  // all bytes referenced by its header are initialized.
  const uint64_t cs_base = ctx_base + layout.control_stack_offset;
  write32(cs_base + 0, 0); // PM4 (skipped)
  write32(cs_base + 4, 0); // PM4 (skipped)
  write32(cs_base + 8, encode_state_word(vgpr_count, sgpr_count));
  for (size_t i = 0; i < num_waves; ++i) {
    // first/last mark workgroup boundaries in the control stack, not the whole
    // queue: a wave is "first" if it opens a new workgroup (group leader) and
    // "last" if it closes one. Callers order waves so each workgroup's waves are
    // contiguous and set these flags per workgroup.
    write32(cs_base + 12 + i * 4,
            encode_wave_word(waves[i].is_first_in_group, waves[i].is_last_in_group,
                             waves[i].scratch_scoreboard_id));
  }

  // --- Per-wave register blocks, laid out high-to-low from wave_state_offset,
  // reproducing gfx9_architecture_t::cwsr_record_t::register_address. ---
  uint64_t last_wave_area = ctx_base + wave_state_offset;
  for (size_t i = 0; i < num_waves; ++i) {
    const CwsrWaveState &w = waves[i];
    const uint64_t save_area_addr = last_wave_area - 64;
    const uint64_t hwregs_addr = save_area_addr - hwreg_bytes;
    const uint64_t ttmps_addr = save_area_addr - kTtmpCount * sizeof(uint32_t);
    const uint64_t sgprs_addr = hwregs_addr - sgpr_bytes;
    const uint64_t vgprs_addr = sgprs_addr - vgpr_bytes;

    for (uint32_t byte = 0; byte < 64; byte += sizeof(uint32_t))
      write32(save_area_addr + byte, 0);

    // HWREG block (32 dwords). The top 16 dwords are the TTMPs.
    write32(hwregs_addr + 0 * 4, w.m0);
    write32(hwregs_addr + 1 * 4, static_cast<uint32_t>(w.pc & 0xFFFFFFFF));
    write32(hwregs_addr + 2 * 4, static_cast<uint32_t>(w.pc >> 32));
    write32(hwregs_addr + 3 * 4, static_cast<uint32_t>(w.exec & 0xFFFFFFFF));
    write32(hwregs_addr + 4 * 4, static_cast<uint32_t>(w.exec >> 32));
    write32(hwregs_addr + 5 * 4, w.status);
    write32(hwregs_addr + 6 * 4, w.trapsts);
    write32(hwregs_addr + 7 * 4, 0); // xnack_mask_lo
    write32(hwregs_addr + 8 * 4, 0); // xnack_mask_hi
    write32(hwregs_addr + 9 * 4, w.mode);
    for (uint32_t h = 10; h < 16; ++h)
      write32(hwregs_addr + h * 4, 0);

    // TTMP0-15 occupy hwreg[16..31] (== ttmps_addr).
    uint32_t ttmp[kTtmpCount] = {};
    ttmp[4] = static_cast<uint32_t>(w.wave_id & 0xFFFFFFFF);
    ttmp[5] = static_cast<uint32_t>(w.wave_id >> 32);
    ttmp[6] = encode_ttmp6(w);
    ttmp[8] = w.group_ids[0];
    ttmp[9] = w.group_ids[1];
    ttmp[10] = w.group_ids[2];
    ttmp[11] = encode_ttmp11(w);
    for (uint32_t t = 0; t < kTtmpCount; ++t)
      write32(ttmps_addr + t * 4, ttmp[t]);

    // SGPR block. Fill meaningful scalars, then place VCC at its aliased slot.
    for (uint32_t s = 0; s < sgpr_count; ++s) {
      uint32_t val = (s < w.num_sgprs && s < w.sgprs.size()) ? w.sgprs[s] : 0u;
      write32(sgprs_addr + s * 4, val);
    }
    write32(sgprs_addr + vcc_lo_slot * 4, static_cast<uint32_t>(w.vcc & 0xFFFFFFFF));
    write32(sgprs_addr + (vcc_lo_slot + 1) * 4, static_cast<uint32_t>(w.vcc >> 32));
    // FLAT_SCRATCH aliases the two scalar slots below VCC (gfx9_4:
    // aliased_sgpr_end - 6/-5; rocdbgapi architecture.cpp register_address).
    // rocm-dbgapi checks its computed per-wave scratch base against this
    // register, so it must hold the wave's scratch base.
    const uint32_t flat_scratch_lo_slot = std::min(kArchScalars, sgpr_count) - 6;
    write32(sgprs_addr + flat_scratch_lo_slot * 4,
            static_cast<uint32_t>(w.flat_scratch & 0xFFFFFFFF));
    write32(sgprs_addr + (flat_scratch_lo_slot + 1) * 4,
            static_cast<uint32_t>(w.flat_scratch >> 32));

    // VGPR block: each VGPR is 64 lanes * 4 bytes; lane l of vgpr r at +r*256+l*4.
    for (uint32_t r = 0; r < vgpr_count; ++r) {
      for (uint32_t lane = 0; lane < 64; ++lane) {
        uint32_t idx = r * 64 + lane;
        uint32_t val = (r < w.num_vgprs && idx < w.vgprs.size()) ? w.vgprs[idx] : 0u;
        write32(vgprs_addr + r * kVgprLaneBytes + lane * 4, val);
      }
    }

    last_wave_area = vgprs_addr; // == register_address(v0_64)
  }

  // Publish the ABI header after the complete payload. A caller that permits a
  // debugger to read concurrently must still provide the required exclusion.
  write32(ctx_base + 0, layout.control_stack_offset);
  write32(ctx_base + 4, layout.control_stack_size);
  write32(ctx_base + 8, layout.wave_state_offset);
  write32(ctx_base + 12, layout.wave_state_size);
  write32(ctx_base + 16, layout.debug_offset);
  write32(ctx_base + 20, layout.debug_size);
  write32(ctx_base + 24, 0); // err_payload_addr lo
  write32(ctx_base + 28, 0); // err_payload_addr hi
  write32(ctx_base + 32, 0); // err_event_id
  write32(ctx_base + 36, 0); // reserved1

  return layout;
}

CwsrLayout serialize_queue_cwsr_bulk(
    uint64_t ctx_base, uint32_t area_size, const std::vector<CwsrWaveState> &waves,
    const std::function<void(uint64_t, std::span<const uint8_t>)> &write_block) {
  if (!write_block)
    return {};

  std::vector<uint8_t> image;
  CwsrLayout layout =
      serialize_queue_cwsr(ctx_base, area_size, waves, [&](uint64_t address, uint32_t value) {
        const uint64_t offset = address - ctx_base;
        if (offset + sizeof(value) > image.size())
          image.resize(static_cast<size_t>(offset + sizeof(value)));
        std::memcpy(image.data() + offset, &value, sizeof(value));
      });
  if (!layout.ok)
    return layout;

  image.resize(layout.wave_state_offset);
  const std::span<const uint8_t> bytes{image};
  write_block(ctx_base + layout.control_stack_offset,
              bytes.subspan(layout.control_stack_offset,
                            layout.wave_state_offset - layout.control_stack_offset));
  write_block(ctx_base, bytes.first(kCwsrHeaderBytes));
  return layout;
}

bool deserialize_queue_cwsr(uint64_t ctx_base, uint32_t area_size,
                            std::vector<CwsrWaveState> &waves,
                            const std::function<uint32_t(uint64_t)> &read32) {
  if (waves.empty() || !read32 || (ctx_base & (alignof(uint32_t) - 1)) != 0)
    return false;

  // Reproduce the exact geometry serialize_queue_cwsr chose (see that function).
  // The wave count and per-wave sgpr/vgpr counts must match, so the register
  // block addresses computed below land on the same dwords that were written.
  const CwsrGeometry geometry = compute_geometry(area_size, waves);
  if (!geometry.layout.ok ||
      ctx_base > std::numeric_limits<uint64_t>::max() - geometry.layout.wave_state_offset)
    return false;
  const uint32_t vcc_lo_slot = geometry.vcc_lo_slot;
  const uint32_t hwreg_bytes = geometry.hwreg_bytes;
  const uint32_t sgpr_bytes = geometry.sgpr_bytes;
  const uint32_t vgpr_bytes = geometry.vgpr_bytes;
  const uint32_t num_waves = static_cast<uint32_t>(waves.size());

  uint64_t last_wave_area = ctx_base + geometry.layout.wave_state_offset;
  for (uint32_t i = 0; i < num_waves; ++i) {
    CwsrWaveState &w = waves[i];
    const uint64_t save_area_addr = last_wave_area - 64;
    const uint64_t hwregs_addr = save_area_addr - hwreg_bytes;
    const uint64_t ttmps_addr = save_area_addr - kTtmpCount * sizeof(uint32_t);
    const uint64_t sgprs_addr = hwregs_addr - sgpr_bytes;
    const uint64_t vgprs_addr = sgprs_addr - vgpr_bytes;

    w.m0 = read32(hwregs_addr + 0 * 4);
    w.pc = static_cast<uint64_t>(read32(hwregs_addr + 1 * 4)) |
           (static_cast<uint64_t>(read32(hwregs_addr + 2 * 4)) << 32);
    w.exec = static_cast<uint64_t>(read32(hwregs_addr + 3 * 4)) |
             (static_cast<uint64_t>(read32(hwregs_addr + 4 * 4)) << 32);
    w.status = read32(hwregs_addr + 5 * 4);
    w.trapsts = read32(hwregs_addr + 6 * 4);
    w.mode = read32(hwregs_addr + 9 * 4);

    const uint32_t ttmp4 = read32(ttmps_addr + 4 * 4);
    const uint32_t ttmp5 = read32(ttmps_addr + 5 * 4);
    w.wave_id = static_cast<uint64_t>(ttmp4) | (static_cast<uint64_t>(ttmp5) << 32);
    const uint32_t ttmp6 = read32(ttmps_addr + 6 * 4);
    w.wave_stopped = (ttmp6 & (1u << 30)) != 0;
    w.saved_status_halt = (ttmp6 & (1u << 29)) != 0;
    w.spi_ttmps_setup = (ttmp6 & (1u << 31)) == 0;
    w.group_ids[0] = read32(ttmps_addr + 8 * 4);
    w.group_ids[1] = read32(ttmps_addr + 9 * 4);
    w.group_ids[2] = read32(ttmps_addr + 10 * 4);
    const uint32_t ttmp11 = read32(ttmps_addr + 11 * 4);
    w.wave_in_group = ttmp11 & 0x3Fu;
    w.queue_packet_id = (ttmp11 >> 6) & 0x1FFFFFFu;

    w.sgprs.resize(w.num_sgprs);
    for (uint32_t s = 0; s < w.num_sgprs; ++s)
      w.sgprs[s] = read32(sgprs_addr + s * 4);
    const uint32_t vcc_lo = read32(sgprs_addr + vcc_lo_slot * 4);
    const uint32_t vcc_hi = read32(sgprs_addr + (vcc_lo_slot + 1) * 4);
    w.vcc = static_cast<uint64_t>(vcc_lo) | (static_cast<uint64_t>(vcc_hi) << 32);

    w.vgprs.resize(static_cast<size_t>(w.num_vgprs) * 64);
    for (uint32_t r = 0; r < w.num_vgprs; ++r)
      for (uint32_t lane = 0; lane < 64; ++lane)
        w.vgprs[r * 64 + lane] = read32(vgprs_addr + r * kVgprLaneBytes + lane * 4);

    last_wave_area = vgprs_addr;
  }
  return true;
}

bool deserialize_queue_cwsr_bulk(
    uint64_t ctx_base, uint32_t area_size, std::vector<CwsrWaveState> &waves,
    const std::function<void(uint64_t, std::span<uint8_t>)> &read_block) {
  if (!read_block || (ctx_base & (alignof(uint32_t) - 1)) != 0)
    return false;
  const CwsrGeometry geometry = compute_geometry(area_size, waves);
  if (!geometry.layout.ok ||
      ctx_base > std::numeric_limits<uint64_t>::max() - geometry.layout.wave_state_offset)
    return false;

  std::array<uint8_t, kCwsrHeaderBytes> header{};
  read_block(ctx_base, header);
  const std::span<const uint8_t> header_bytes{header};
  if (image_word(header_bytes, 0) != geometry.layout.control_stack_offset ||
      image_word(header_bytes, 4) != geometry.layout.control_stack_size ||
      image_word(header_bytes, 8) != geometry.layout.wave_state_offset ||
      image_word(header_bytes, 12) != geometry.layout.wave_state_size ||
      image_word(header_bytes, 16) != geometry.layout.debug_offset ||
      image_word(header_bytes, 20) != geometry.layout.debug_size)
    return false;

  std::vector<uint8_t> image(geometry.layout.wave_state_offset);
  std::memcpy(image.data(), header.data(), header.size());
  std::span<uint8_t> payload{image.data() + geometry.layout.control_stack_offset,
                             geometry.layout.wave_state_offset -
                                 geometry.layout.control_stack_offset};
  read_block(ctx_base + geometry.layout.control_stack_offset, payload);

  bool in_bounds = true;
  const bool decoded = deserialize_queue_cwsr(ctx_base, area_size, waves, [&](uint64_t address) {
    const uint64_t offset = address - ctx_base;
    if (offset + sizeof(uint32_t) > image.size()) {
      in_bounds = false;
      return uint32_t{0};
    }
    return image_word(image, static_cast<uint32_t>(offset));
  });
  return decoded && in_bounds;
}

} // namespace kmd
} // namespace rocjitsu
