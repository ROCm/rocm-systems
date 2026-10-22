// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

/// @file cwsr.cpp
/// @brief CWSR (context-save-restore) area serialization for rocm-dbgapi.

#include "rocjitsu/kmd/linux/cwsr.h"
#include "rocjitsu/isa/arch/amdgpu/generated/shared/isa_properties.h"

#include <algorithm>
#include <cstring>
#include <limits>

namespace rocjitsu {
namespace kmd {

namespace {

// Layout constants matching the gfx9.4 and gfx12.5 cwsr_record_t definitions
// in rocdbgapi.
constexpr uint32_t kGfx94HwregCount = 32;
constexpr uint32_t kGfx1250HwregCount = 128;
constexpr uint32_t kGfx94RecordPrefixBytes = 64;
constexpr uint32_t kGfx94StateWords = 1;
constexpr uint32_t kGfx1250StateWords = 2;
constexpr uint32_t kDebuggerBytesPerWave = 32;
constexpr uint32_t kDebuggerBytesAlign = 64;
constexpr uint32_t kDebuggerReserveChunks = 8;
constexpr uint32_t kTtmpCount = 16; // ttmps saved at the top of the hwreg block
// The saved block holds kCwsrSavedSgprSlots scalars, so that is the ceiling a
// wave can carry. It is deliberately not the architected count (102): s104 and
// s105 are ordinary registers above the FLAT_SCRATCH alias, and CDNA3/CDNA4
// default to 112 SGPRs per wave (config_loader default_sgprs_per_wf), so a
// lower bound here would reject an ordinary dispatch and fail the queue's whole
// CWSR publish. Slots that alias travel in their own fields either way.
constexpr uint32_t kCwsrHeaderBytes = 10 * sizeof(uint32_t);
constexpr uint32_t kControlStackOffset = 0x100u;

// COMPUTE_RELAUNCH classification bits (control_stack_iterate): a word with bit
// 30 set is an event (skipped) and a word with bit 31 set is a state word; a
// wave word has both clear.
constexpr uint32_t kRelaunchStateBit = 1u << 31;

constexpr uint32_t round_up(uint32_t v, uint32_t m) { return (v / m + (v % m != 0)) * m; }

// Encode the COMPUTE_RELAUNCH "state" word so rocm-dbgapi decodes exactly
// @vgpr_count / @sgpr_count with zero accumulation (ACC) VGPRs:
//   vgpr_count      = (accum_offset[24:29] + 1) * 4
//   acc_vgpr_count  = (vgprs[0:5]      + 1) * 8 - vgpr_count   (== 0 here)
//   sgpr_count      = (sgprs[6:8]      + 1) * 16 - 16
//   lds_size        = lds[9:16] * 512 bytes on gfx942, * 1280 on gfx950
uint32_t encode_state_word(uint32_t vgpr_count, uint32_t sgpr_count, uint32_t lds_size,
                           uint32_t lds_granule) {
  uint32_t vgprs_field = (vgpr_count / kCwsrGfx94VgprGranule) - 1;
  uint32_t accum_offset = (vgpr_count / 4) - 1; // vgpr_count = (accum_offset+1)*4
  uint32_t sgprs_field = sgpr_count / 16;       // sgpr_count = sgprs_field*16
  uint32_t w = 0;
  w |= (vgprs_field & 0x3Fu);
  w |= (sgprs_field & 0x7u) << 6;
  w |= ((lds_size / lds_granule) & 0xFFu) << 9;
  w |= (accum_offset & 0x3Fu) << 24;
  w |= kRelaunchStateBit;
  return w;
}

uint32_t encode_gfx1250_state_word(uint32_t vgpr_count, uint32_t lds_size) {
  uint32_t w = ((vgpr_count / kCwsrGfx1250VgprGranule) - 1) & 0x3Fu;
  w |= ((lds_size / kCwsrGfx1250LdsGranuleBytes) & 0x1FFu) << 10;
  w |= 1u << 24; // gfx1250 supports wave32 only.
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

uint32_t encode_gfx1250_wave_word(const CwsrWaveState &wave) {
  uint32_t w = wave.scratch_scoreboard_id & 0x3FFu;
  if (wave.flat_scratch != 0)
    w |= 1u << 11;
  if (wave.is_first_in_group)
    w |= 1u << 12;
  if (wave.is_last_in_group)
    w |= 1u << 13;
  w |= (wave.shader_engine_id & 0x3u) << 24;
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

uint32_t encode_gfx1250_ttmp8(const CwsrWaveState &w) {
  uint32_t value = w.queue_packet_id & 0x1FFFFFFu;
  value |= (w.wave_in_group & 0x1Fu) << 25;
  value |= 1u << 30; // group Y/Z are valid in TTMP7.
  value |= 1u << 31; // trap-handler TTMPs initialized; makes the wave id valid.
  return value;
}

struct CwsrGeometry {
  CwsrLayout layout{};
  uint32_t vgpr_count = 0;
  uint32_t sgpr_count = kCwsrSavedSgprSlots;
  uint32_t vcc_lo_slot = kCwsrVccLoSlot;
  uint32_t flat_scratch_lo_slot = kCwsrFlatScratchLoSlot;
  uint32_t hwreg_bytes = 0;
  uint32_t sgpr_bytes = 0;
  uint32_t vgpr_bytes = 0;
  std::vector<uint32_t> lds_bytes_by_wave;
  uint32_t vgpr_lane_bytes = 64 * sizeof(uint32_t);
  uint32_t lane_count = 64;
  uint32_t record_prefix_bytes = 64;
  uint32_t state_words = 1;
  uint64_t per_wave = 0;
};

CwsrGeometry compute_geometry(uint32_t area_size, const std::vector<CwsrWaveState> &waves,
                              rj_code_arch_t arch) {
  CwsrGeometry geometry{};
  if (!cwsr_layout_modelled(arch) || waves.empty() ||
      waves.size() > std::numeric_limits<uint32_t>::max())
    return geometry;

  const bool gfx12_5 = cwsr_layout_kind(arch) == CwsrLayoutKind::Gfx12_5;
  const auto properties = isa_properties(arch);
  geometry.sgpr_count = gfx12_5 ? kCwsrGfx1250SavedSgprSlots : kCwsrSavedSgprSlots;
  geometry.hwreg_bytes = (gfx12_5 ? kGfx1250HwregCount : kGfx94HwregCount) * sizeof(uint32_t);
  geometry.lane_count = properties.wave_size_max;
  geometry.vgpr_lane_bytes = geometry.lane_count * sizeof(uint32_t);
  geometry.record_prefix_bytes = gfx12_5 ? 0 : kGfx94RecordPrefixBytes;
  geometry.state_words = gfx12_5 ? kGfx1250StateWords : kGfx94StateWords;

  uint32_t max_vgprs = 0;
  uint32_t active_lds = 0;
  uint64_t lds_total = 0;
  uint64_t state_changes = 0;
  const uint32_t lds_granule = cwsr_lds_granule_bytes(arch);
  const uint32_t max_encoded_lds = cwsr_max_encoded_lds_bytes(arch);
  geometry.lds_bytes_by_wave.resize(waves.size());
  for (size_t i = 0; i < waves.size(); ++i) {
    const auto &wave = waves[i];
    const uint32_t max_saved_vgprs = properties.max_addressable_vgprs_per_wf;
    if (wave.num_vgprs > max_saved_vgprs || wave.num_sgprs > geometry.sgpr_count ||
        wave.trap_id > 0xFu || wave.wave_in_group > (gfx12_5 ? 0x1Fu : 0x3Fu) ||
        wave.queue_packet_id > 0x1FFFFFFu ||
        (gfx12_5 && (wave.group_ids[1] > 0xFFFFu || wave.group_ids[2] > 0xFFFFu)))
      return geometry;
    max_vgprs = std::max(max_vgprs, wave.num_vgprs);
    if (wave.is_first_in_group) {
      if (wave.lds.size() > max_encoded_lds)
        return geometry;
      const uint32_t group_lds = round_up(static_cast<uint32_t>(wave.lds.size()), lds_granule);
      geometry.lds_bytes_by_wave[i] = group_lds;
      lds_total += group_lds;
      if (i == 0 || group_lds != active_lds)
        ++state_changes;
      active_lds = group_lds;
    } else if (i == 0) {
      ++state_changes;
    }
  }

  const uint32_t vgpr_granule = gfx12_5 ? kCwsrGfx1250VgprGranule : kCwsrGfx94VgprGranule;
  geometry.vgpr_count = std::max<uint32_t>(round_up(max_vgprs, vgpr_granule), vgpr_granule);
  // Alias slots come from cwsr.h so the codec, the wave writeback and the tests
  // share one definition. FLAT_SCRATCH being written but read back from a
  // separately open-coded offset is exactly how it came to be dropped.
  geometry.sgpr_bytes = geometry.sgpr_count * sizeof(uint32_t);
  geometry.vgpr_bytes = geometry.vgpr_count * geometry.vgpr_lane_bytes;
  geometry.per_wave = geometry.record_prefix_bytes + geometry.hwreg_bytes + geometry.sgpr_bytes +
                      geometry.vgpr_bytes;

  const uint64_t num_waves = waves.size();
  const uint64_t wave_state_size = geometry.per_wave * num_waves + lds_total;
  const uint64_t control_stack_size =
      (2u + geometry.state_words * state_changes + num_waves) * sizeof(uint32_t);
  const uint64_t wave_area_begin = kControlStackOffset + control_stack_size;
  const uint64_t wave_state_offset = wave_area_begin + wave_state_size;
  if (control_stack_size > std::numeric_limits<uint32_t>::max() ||
      wave_state_size > std::numeric_limits<uint32_t>::max() ||
      wave_state_offset > std::numeric_limits<uint32_t>::max() || wave_state_offset > area_size)
    return geometry;

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

CwsrCapacity cwsr_capacity_for_device(const config::KfdDeviceConfig &device, uint32_t num_xcc) {
  const rj_code_arch_t arch =
      cwsr_arch_for_gc_ip_version(gc_ip_version_for_gfx_target_version(device.gfx_target_version));
  if (num_xcc == 0 || !cwsr_capacity_eligible_device(device, arch))
    return {};
  if (!cwsr_layout_modelled(arch)) {
    // Queue consumers require nonzero storage even when this codec does not
    // implement the target's save record. Keep that compatibility reserve
    // separate from the CDNA serializer-derived capacity below.
    constexpr uint32_t kUnsupportedControlReserveBytes = 4096;
    constexpr uint32_t kUnsupportedContextReserveBytes = 64 * 1024 * 1024;
    return {kUnsupportedControlReserveBytes, kUnsupportedContextReserveBytes};
  }
  const bool gfx12_5 = cwsr_layout_kind(arch) == CwsrLayoutKind::Gfx12_5;
  const auto properties = isa_properties(arch);
  constexpr uint64_t kPageSize = 4096;
  constexpr uint64_t kMaxAlignedBytes = std::numeric_limits<uint32_t>::max() - (kPageSize - 1);
  const uint64_t kfd_cus =
      ((uint64_t{device.simd_count} + device.simd_per_cu - 1) / device.simd_per_cu + num_xcc - 1) /
      num_xcc;
  // KFD may advertise only enabled SIMD units, while the simulator can stop
  // waves in every configured CU slot. Bound the producer's complete XCC.
  const uint64_t arrays = std::max(1u, device.num_shader_arrays_per_engine);
  if (device.num_shader_engines != 0 && arrays > kMaxAlignedBytes / device.num_shader_engines)
    return {};
  const uint64_t shader_arrays = uint64_t{device.num_shader_engines} * arrays;
  if (shader_arrays != 0 && device.num_cu_per_sh > kMaxAlignedBytes / shader_arrays)
    return {};
  const uint64_t simulated_cus = shader_arrays * device.num_cu_per_sh;
  const uint64_t active_cus = std::max(kfd_cus, simulated_cus);
  const uint64_t slots_per_cu = std::max(uint64_t{device.simd_per_cu} * device.max_waves_per_simd,
                                         uint64_t{device.max_slots_scratch_cu});
  if (active_cus == 0 || slots_per_cu == 0 || active_cus > kMaxAlignedBytes / slots_per_cu)
    return {};
  const uint64_t waves = active_cus * slots_per_cu;
  if (waves > kMaxAlignedBytes / sizeof(uint32_t))
    return {};

  // compute_geometry pads every record to the largest VGPR count in the
  // queue. One high-VGPR wave therefore gives all resident waves this stride.
  const uint64_t max_vgprs = round_up(properties.max_addressable_vgprs_per_wf,
                                      gfx12_5 ? kCwsrGfx1250VgprGranule : kCwsrGfx94VgprGranule);
  const uint64_t record_bytes =
      (gfx12_5 ? 0 : kGfx94RecordPrefixBytes) +
      (gfx12_5 ? kGfx1250HwregCount : kGfx94HwregCount) * sizeof(uint32_t) +
      (gfx12_5 ? kCwsrGfx1250SavedSgprSlots : kCwsrSavedSgprSlots) * sizeof(uint32_t) +
      max_vgprs * properties.wave_size_max * sizeof(uint32_t);
  // A new state may be needed before every one-wave workgroup as its LDS
  // allocation changes. rocm-dbgapi accepts state words between wave words.
  const uint64_t state_words = gfx12_5 ? kGfx1250StateWords : kGfx94StateWords;
  const uint64_t control_words = 2 + waves * (1 + state_words);
  const uint64_t control_raw = kControlStackOffset + control_words * sizeof(uint32_t);
  if (control_raw > kMaxAlignedBytes || record_bytes == 0 ||
      waves > kMaxAlignedBytes / record_bytes)
    return {};
  const uint64_t control = (control_raw + kPageSize - 1) & ~(kPageSize - 1);
  const uint64_t record_total = waves * record_bytes;
  if (control > kMaxAlignedBytes - record_total || device.lds_size_kb > kMaxAlignedBytes / 1024)
    return {};

  // The CU allocator admits at most lds_size_kb of live workgroups per CU.
  // The codec saves each group's LDS separately, rounded to its record
  // granule. Allow one rounding remainder for every possible one-wave group.
  const uint64_t lds_granule = cwsr_lds_granule_bytes(arch);
  const uint64_t lds_per_cu = uint64_t{device.lds_size_kb} * 1024;
  if (lds_per_cu > cwsr_max_encoded_lds_bytes(arch))
    return {};
  const uint64_t remaining = kMaxAlignedBytes - control - record_total;
  if (waves > remaining / (lds_granule - 1))
    return {};
  const uint64_t lds_rounding = waves * (lds_granule - 1);
  if (lds_per_cu != 0 && active_cus > (remaining - lds_rounding) / lds_per_cu)
    return {};
  const uint64_t lds_total = active_cus * lds_per_cu + lds_rounding;
  const uint64_t debug_bytes =
      ((waves + kDebuggerReserveChunks) * kDebuggerBytesPerWave + kDebuggerBytesAlign - 1) /
      kDebuggerBytesAlign * kDebuggerBytesAlign;
  if (debug_bytes > remaining - lds_total)
    return {};
  const uint64_t context =
      (control + record_total + lds_total + debug_bytes + kPageSize - 1) & ~(kPageSize - 1);
  return {static_cast<uint32_t>(control), static_cast<uint32_t>(context)};
}

CwsrLayout serialize_queue_cwsr(uint64_t ctx_base, uint32_t area_size,
                                const std::vector<CwsrWaveState> &waves,
                                const std::function<void(uint64_t, uint32_t)> &write32,
                                rj_code_arch_t arch) {
  CwsrLayout layout{};
  if (!cwsr_layout_modelled(arch) || waves.empty() || !write32 ||
      (ctx_base & (alignof(uint32_t) - 1)) != 0)
    return layout;

  // Registers retain one queue-wide stride; ACC-VGPRs are not modeled.
  // A state change before a workgroup leader selects that group's LDS size.
  const CwsrGeometry geometry = compute_geometry(area_size, waves, arch);
  if (!geometry.layout.ok)
    return layout;
  layout = geometry.layout;
  const uint32_t vgpr_count = geometry.vgpr_count;
  const uint32_t sgpr_count = geometry.sgpr_count;
  const uint32_t vcc_lo_slot = geometry.vcc_lo_slot;
  const uint32_t hwreg_bytes = geometry.hwreg_bytes;
  const uint32_t sgpr_bytes = geometry.sgpr_bytes;
  const uint32_t vgpr_bytes = geometry.vgpr_bytes;
  const auto &lds_bytes_by_wave = geometry.lds_bytes_by_wave;
  const uint32_t lane_count = geometry.lane_count;
  const uint32_t vgpr_lane_bytes = geometry.vgpr_lane_bytes;
  const bool gfx12_5 = cwsr_layout_kind(arch) == CwsrLayoutKind::Gfx12_5;
  const uint64_t num_waves = waves.size();
  const uint64_t wave_state_offset = layout.wave_state_offset;
  if (ctx_base > std::numeric_limits<uint64_t>::max() - wave_state_offset)
    return CwsrLayout{};

  // Write the payload first so a newly allocated area is not advertised before
  // all bytes referenced by its header are initialized.
  const uint64_t cs_base = ctx_base + layout.control_stack_offset;
  write32(cs_base + 0, 0); // PM4 (skipped)
  write32(cs_base + 4, 0); // PM4 (skipped)
  uint64_t control_word = 2;
  uint32_t active_lds = 0;
  for (size_t i = 0; i < num_waves; ++i) {
    const uint32_t group_lds = lds_bytes_by_wave[i];
    if (i == 0 || (waves[i].is_first_in_group && group_lds != active_lds)) {
      write32(cs_base + control_word++ * sizeof(uint32_t),
              gfx12_5 ? encode_gfx1250_state_word(vgpr_count, group_lds)
                      : encode_state_word(vgpr_count, sgpr_count, group_lds,
                                          cwsr_lds_granule_bytes(arch)));
      if (gfx12_5)
        write32(cs_base + control_word++ * sizeof(uint32_t), 0); // COMPUTE_RELAUNCH2
      active_lds = group_lds;
    }
    // first/last delimit a workgroup in the control stack, not the queue.
    write32(cs_base + control_word++ * sizeof(uint32_t),
            gfx12_5 ? encode_gfx1250_wave_word(waves[i])
                    : encode_wave_word(waves[i].is_first_in_group, waves[i].is_last_in_group,
                                       waves[i].scratch_scoreboard_id));
  }

  // --- Per-wave register blocks, laid out high-to-low from wave_state_offset,
  // reproducing gfx9_architecture_t::cwsr_record_t::register_address. ---
  uint64_t last_wave_area = ctx_base + wave_state_offset;
  for (size_t i = 0; i < num_waves; ++i) {
    const CwsrWaveState &w = waves[i];
    const uint64_t save_area_addr = last_wave_area - geometry.record_prefix_bytes;
    uint64_t register_area_end = save_area_addr;
    if (!gfx12_5)
      for (uint32_t byte = 0; byte < geometry.record_prefix_bytes; byte += sizeof(uint32_t))
        write32(save_area_addr + byte, 0);

    if (w.is_first_in_group) {
      const uint32_t lds_bytes = lds_bytes_by_wave[i];
      register_area_end -= lds_bytes;
      for (uint32_t byte = 0; byte < lds_bytes; byte += sizeof(uint32_t)) {
        uint32_t value = 0;
        if (byte < w.lds.size())
          std::memcpy(&value, w.lds.data() + byte,
                      std::min<size_t>(sizeof(value), w.lds.size() - byte));
        write32(register_area_end + byte, value);
      }
    }
    const uint64_t hwregs_addr = register_area_end - hwreg_bytes;
    const uint64_t sgprs_addr = hwregs_addr - sgpr_bytes;
    const uint64_t vgprs_addr = sgprs_addr - vgpr_bytes;
    const uint64_t ttmps_addr = gfx12_5 ? sgprs_addr + (sgpr_count - kTtmpCount) * sizeof(uint32_t)
                                        : register_area_end - kTtmpCount * sizeof(uint32_t);

    // HWREG block. gfx9.4 stores TTMPs in its top 16 dwords; gfx12.5 keeps
    // them in the final 16 slots of the SGPR block.
    for (uint32_t h = 0; h < hwreg_bytes / sizeof(uint32_t); ++h)
      write32(hwregs_addr + h * 4, 0);
    write32(hwregs_addr + 0 * 4, w.m0);
    write32(hwregs_addr + 1 * 4, static_cast<uint32_t>(w.pc & 0xFFFFFFFF));
    write32(hwregs_addr + 2 * 4, static_cast<uint32_t>(w.pc >> 32));
    write32(hwregs_addr + 3 * 4, static_cast<uint32_t>(w.exec & 0xFFFFFFFF));
    write32(hwregs_addr + 4 * 4, static_cast<uint32_t>(w.exec >> 32));
    if (gfx12_5) {
      write32(hwregs_addr + 5 * 4, w.state_priv);
      write32(hwregs_addr + 6 * 4, w.excp_flag_priv);
      write32(hwregs_addr + 7 * 4, w.xnack_mask);
      write32(hwregs_addr + 8 * 4, w.mode);
      write32(hwregs_addr + 9 * 4, static_cast<uint32_t>(w.flat_scratch));
      write32(hwregs_addr + 10 * 4, static_cast<uint32_t>(w.flat_scratch >> 32));
      write32(hwregs_addr + 11 * 4, w.excp_flag_user);
      write32(hwregs_addr + 12 * 4, w.trap_ctrl);
      write32(hwregs_addr + 13 * 4, w.status);
    } else {
      write32(hwregs_addr + 5 * 4, w.status);
      write32(hwregs_addr + 6 * 4, w.trapsts);
      write32(hwregs_addr + 9 * 4, w.mode);
    }

    // TTMP0-15 occupy hwreg[16..31] (== ttmps_addr).
    uint32_t ttmp[kTtmpCount] = {};
    ttmp[4] = static_cast<uint32_t>(w.wave_id & 0xFFFFFFFF);
    ttmp[5] = static_cast<uint32_t>(w.wave_id >> 32);
    ttmp[6] = encode_ttmp6(w) & (gfx12_5 ? ~((1u << 31) | (0xFu << 25)) : UINT32_MAX);
    if (gfx12_5) {
      ttmp[7] = (w.group_ids[1] & 0xFFFFu) | ((w.group_ids[2] & 0xFFFFu) << 16);
      ttmp[8] = encode_gfx1250_ttmp8(w);
      ttmp[9] = w.group_ids[0];
      ttmp[11] = (w.xnack_state_priv & 0x0FFFFFFFu) | ((w.trap_id & 0xFu) << 28);
    } else {
      ttmp[8] = w.group_ids[0];
      ttmp[9] = w.group_ids[1];
      ttmp[10] = w.group_ids[2];
      ttmp[11] = encode_ttmp11(w);
    }
    if (!gfx12_5)
      for (uint32_t t = 0; t < kTtmpCount; ++t)
        write32(ttmps_addr + t * 4, ttmp[t]);

    // SGPR block. Fill meaningful scalars, then place VCC at its aliased slot.
    // num_sgprs may reach the full block, so without this bound the aliases
    // would be filled from sgprs[] first and then overwritten, and deserialize
    // would read the alias bytes straight back as SGPRs. Skip only the slots
    // the aliases below actually occupy, not everything above the gfx9.4
    // architected count: s104/s105 sit between FLAT_SCRATCH and VCC and are
    // real registers. Filling an aliased slot from sgprs[] first would be
    // harmless (it is overwritten), but deserialize and the wave writeback have
    // to skip exactly the same set, and deriving all three from one predicate
    // is what keeps them in step.
    for (uint32_t s = 0; s < sgpr_count; ++s) {
      const bool aliased = cwsr_sgpr_slot_is_aliased(s, arch);
      uint32_t val = (!aliased && s < w.num_sgprs && s < w.sgprs.size()) ? w.sgprs[s] : 0u;
      write32(sgprs_addr + s * 4, val);
    }
    write32(sgprs_addr + vcc_lo_slot * 4, static_cast<uint32_t>(w.vcc & 0xFFFFFFFF));
    write32(sgprs_addr + (vcc_lo_slot + 1) * 4, static_cast<uint32_t>(w.vcc >> 32));
    // FLAT_SCRATCH aliases the two scalar slots below VCC (gfx9_4:
    // aliased_sgpr_end - 6/-5; rocdbgapi architecture.cpp register_address).
    // rocm-dbgapi checks its computed per-wave scratch base against this
    // register, so it must hold the wave's scratch base.
    if (!gfx12_5) {
      const uint32_t flat_scratch_lo_slot = geometry.flat_scratch_lo_slot;
      write32(sgprs_addr + flat_scratch_lo_slot * 4,
              static_cast<uint32_t>(w.flat_scratch & 0xFFFFFFFF));
      write32(sgprs_addr + (flat_scratch_lo_slot + 1) * 4,
              static_cast<uint32_t>(w.flat_scratch >> 32));
    }
    if (gfx12_5)
      for (uint32_t t = 0; t < kTtmpCount; ++t)
        write32(ttmps_addr + t * 4, ttmp[t]);

    // The input vector keeps a uniform wave64-shaped stride; gfx12.5 packs the
    // live 32 lanes into a 128-byte register in the CWSR image.
    for (uint32_t r = 0; r < vgpr_count; ++r) {
      for (uint32_t lane = 0; lane < lane_count; ++lane) {
        uint32_t idx = r * 64 + lane;
        uint32_t val = (r < w.num_vgprs && idx < w.vgprs.size()) ? w.vgprs[idx] : 0u;
        write32(vgprs_addr + r * vgpr_lane_bytes + lane * 4, val);
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
    const std::function<void(uint64_t, std::span<const uint8_t>)> &write_block,
    rj_code_arch_t arch) {
  if (!write_block)
    return {};

  std::vector<uint8_t> image;
  CwsrLayout layout = serialize_queue_cwsr(
      ctx_base, area_size, waves,
      [&](uint64_t address, uint32_t value) {
        const uint64_t offset = address - ctx_base;
        if (offset + sizeof(value) > image.size())
          image.resize(static_cast<size_t>(offset + sizeof(value)));
        std::memcpy(image.data() + offset, &value, sizeof(value));
      },
      arch);
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
                            const std::function<uint32_t(uint64_t)> &read32, rj_code_arch_t arch) {
  if (!cwsr_layout_modelled(arch) || waves.empty() || !read32 ||
      (ctx_base & (alignof(uint32_t) - 1)) != 0)
    return false;

  // Reproduce the exact geometry serialize_queue_cwsr chose (see that function).
  // The wave count, register counts, group leaders, and live LDS lengths must
  // match, so the addresses below land on the same dwords that were written.
  const CwsrGeometry geometry = compute_geometry(area_size, waves, arch);
  if (!geometry.layout.ok ||
      ctx_base > std::numeric_limits<uint64_t>::max() - geometry.layout.wave_state_offset)
    return false;
  const uint32_t vcc_lo_slot = geometry.vcc_lo_slot;
  const uint32_t hwreg_bytes = geometry.hwreg_bytes;
  const uint32_t sgpr_bytes = geometry.sgpr_bytes;
  const uint32_t vgpr_bytes = geometry.vgpr_bytes;
  const auto &lds_bytes_by_wave = geometry.lds_bytes_by_wave;
  const uint32_t lane_count = geometry.lane_count;
  const uint32_t vgpr_lane_bytes = geometry.vgpr_lane_bytes;
  const bool gfx12_5 = cwsr_layout_kind(arch) == CwsrLayoutKind::Gfx12_5;
  const uint32_t num_waves = static_cast<uint32_t>(waves.size());

  uint64_t last_wave_area = ctx_base + geometry.layout.wave_state_offset;
  for (uint32_t i = 0; i < num_waves; ++i) {
    CwsrWaveState &w = waves[i];
    const uint64_t save_area_addr = last_wave_area - geometry.record_prefix_bytes;
    uint64_t register_area_end = save_area_addr;
    if (w.is_first_in_group) {
      const uint32_t lds_bytes = lds_bytes_by_wave[i];
      register_area_end -= lds_bytes;
      const size_t live_lds_bytes = w.lds.size();
      for (uint32_t byte = 0; byte < live_lds_bytes; byte += sizeof(uint32_t)) {
        const uint32_t value = read32(register_area_end + byte);
        std::memcpy(w.lds.data() + byte, &value,
                    std::min<size_t>(sizeof(value), live_lds_bytes - byte));
      }
    } else {
      w.lds.clear();
    }
    const uint64_t hwregs_addr = register_area_end - hwreg_bytes;
    const uint64_t sgprs_addr = hwregs_addr - sgpr_bytes;
    const uint64_t vgprs_addr = sgprs_addr - vgpr_bytes;
    const uint64_t ttmps_addr =
        gfx12_5 ? sgprs_addr + (geometry.sgpr_count - kTtmpCount) * sizeof(uint32_t)
                : register_area_end - kTtmpCount * sizeof(uint32_t);

    w.m0 = read32(hwregs_addr + 0 * 4);
    w.pc = static_cast<uint64_t>(read32(hwregs_addr + 1 * 4)) |
           (static_cast<uint64_t>(read32(hwregs_addr + 2 * 4)) << 32);
    w.exec = static_cast<uint64_t>(read32(hwregs_addr + 3 * 4)) |
             (static_cast<uint64_t>(read32(hwregs_addr + 4 * 4)) << 32);
    if (gfx12_5) {
      w.state_priv = read32(hwregs_addr + 5 * 4);
      w.excp_flag_priv = read32(hwregs_addr + 6 * 4);
      w.xnack_mask = read32(hwregs_addr + 7 * 4);
      w.mode = read32(hwregs_addr + 8 * 4);
      const uint32_t scratch_lo = read32(hwregs_addr + 9 * 4);
      const uint32_t scratch_hi = read32(hwregs_addr + 10 * 4);
      w.flat_scratch =
          static_cast<uint64_t>(scratch_lo) | (static_cast<uint64_t>(scratch_hi) << 32);
      w.excp_flag_user = read32(hwregs_addr + 11 * 4);
      w.trap_ctrl = read32(hwregs_addr + 12 * 4);
      w.status = read32(hwregs_addr + 13 * 4);
    } else {
      w.status = read32(hwregs_addr + 5 * 4);
      w.trapsts = read32(hwregs_addr + 6 * 4);
      w.mode = read32(hwregs_addr + 9 * 4);
    }

    const uint32_t ttmp4 = read32(ttmps_addr + 4 * 4);
    const uint32_t ttmp5 = read32(ttmps_addr + 5 * 4);
    w.wave_id = static_cast<uint64_t>(ttmp4) | (static_cast<uint64_t>(ttmp5) << 32);
    const uint32_t ttmp6 = read32(ttmps_addr + 6 * 4);
    w.wave_stopped = (ttmp6 & (1u << 30)) != 0;
    w.saved_status_halt = (ttmp6 & (1u << 29)) != 0;
    if (gfx12_5) {
      const uint32_t ttmp7 = read32(ttmps_addr + 7 * 4);
      const uint32_t ttmp8 = read32(ttmps_addr + 8 * 4);
      w.spi_ttmps_setup = true;
      w.group_ids[0] = read32(ttmps_addr + 9 * 4);
      w.group_ids[1] = ttmp7 & 0xFFFFu;
      w.group_ids[2] = ttmp7 >> 16;
      w.wave_in_group = (ttmp8 >> 25) & 0x1Fu;
      w.queue_packet_id = ttmp8 & 0x1FFFFFFu;
      const uint32_t ttmp11 = read32(ttmps_addr + 11 * 4);
      w.xnack_state_priv = ttmp11 & 0x0FFFFFFFu;
      w.trap_id = ttmp11 >> 28;
    } else {
      w.spi_ttmps_setup = (ttmp6 & (1u << 31)) == 0;
      w.group_ids[0] = read32(ttmps_addr + 8 * 4);
      w.group_ids[1] = read32(ttmps_addr + 9 * 4);
      w.group_ids[2] = read32(ttmps_addr + 10 * 4);
      const uint32_t ttmp11 = read32(ttmps_addr + 11 * 4);
      w.wave_in_group = ttmp11 & 0x3Fu;
      w.queue_packet_id = (ttmp11 >> 6) & 0x1FFFFFFu;
    }

    // Read back only the slots that hold real registers. The aliased ones are
    // decoded as the registers they alias, below; treating them as SGPRs put
    // FLAT_SCRATCH's bytes into s102/s103 on a round trip.
    w.sgprs.assign(w.num_sgprs, 0u);
    for (uint32_t s = 0; s < w.num_sgprs; ++s)
      if (!cwsr_sgpr_slot_is_aliased(s, arch))
        w.sgprs[s] = read32(sgprs_addr + s * 4);
    const uint32_t vcc_lo = read32(sgprs_addr + vcc_lo_slot * 4);
    const uint32_t vcc_hi = read32(sgprs_addr + (vcc_lo_slot + 1) * 4);
    w.vcc = static_cast<uint64_t>(vcc_lo) | (static_cast<uint64_t>(vcc_hi) << 32);
    // FLAT_SCRATCH is serialized into its alias slots but was never read back,
    // so a serialize/deserialize round trip silently dropped it.
    if (!gfx12_5) {
      const uint32_t flat_scratch_lo_slot = geometry.flat_scratch_lo_slot;
      const uint32_t fs_lo = read32(sgprs_addr + flat_scratch_lo_slot * 4);
      const uint32_t fs_hi = read32(sgprs_addr + (flat_scratch_lo_slot + 1) * 4);
      w.flat_scratch = static_cast<uint64_t>(fs_lo) | (static_cast<uint64_t>(fs_hi) << 32);
    }

    w.vgprs.resize(static_cast<size_t>(w.num_vgprs) * 64);
    for (uint32_t r = 0; r < w.num_vgprs; ++r)
      for (uint32_t lane = 0; lane < lane_count; ++lane)
        w.vgprs[r * 64 + lane] = read32(vgprs_addr + r * vgpr_lane_bytes + lane * 4);

    last_wave_area = vgprs_addr;
  }
  return true;
}

bool deserialize_queue_cwsr_bulk(
    uint64_t ctx_base, uint32_t area_size, std::vector<CwsrWaveState> &waves,
    const std::function<void(uint64_t, std::span<uint8_t>)> &read_block, rj_code_arch_t arch) {
  if (!read_block || (ctx_base & (alignof(uint32_t) - 1)) != 0)
    return false;
  const CwsrGeometry geometry = compute_geometry(area_size, waves, arch);
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
  const bool decoded = deserialize_queue_cwsr(
      ctx_base, area_size, waves,
      [&](uint64_t address) {
        const uint64_t offset = address - ctx_base;
        if (offset + sizeof(uint32_t) > image.size()) {
          in_bounds = false;
          return uint32_t{0};
        }
        return image_word(image, static_cast<uint32_t>(offset));
      },
      arch);
  return decoded && in_bounds;
}

} // namespace kmd
} // namespace rocjitsu
