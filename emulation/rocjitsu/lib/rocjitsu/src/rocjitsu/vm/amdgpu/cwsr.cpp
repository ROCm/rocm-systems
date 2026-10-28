// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

/// @file cwsr.cpp
/// @brief CWSR (context-save-restore) area serialization for rocm-dbgapi.

#include "rocjitsu/vm/amdgpu/cwsr.h"
#include "rocjitsu/isa/arch/amdgpu/generated/shared/isa_properties.h"
// Wavefront::lds() is defined inline in compute_unit.h.
#include "rocjitsu/vm/amdgpu/compute_unit.h"
#include "rocjitsu/vm/amdgpu/hwreg.h"
#include "rocjitsu/vm/amdgpu/wavefront.h"

#include <algorithm>
#include <cstring>
#include <limits>

namespace rocjitsu::amdgpu {

namespace {

constexpr uint32_t kCwsrPrivateContextMagic = 0x50434a52u; // "RJCP"

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
//   lds_size        = lds[9:16] * lds_granule bytes
uint32_t encode_state_word(uint32_t vgpr_count, uint32_t sgpr_count, uint32_t lds_size,
                           uint32_t lds_granule) {
  uint32_t vgprs_field = (vgpr_count / kCwsrGfx94VgprGranule) - 1;
  uint32_t accum_offset = (vgpr_count / 4) - 1; // vgpr_count = (accum_offset+1)*4
  uint32_t sgprs_field = sgpr_count / 16;       // sgpr_count = sgprs_field*16
  uint32_t w = 0;
  w |= (vgprs_field & 0x3Fu);
  w |= (sgprs_field & 0x7u) << 6;
  w |= ((lds_size / lds_granule) & kCwsrGfx94LdsFieldMax) << 9;
  w |= (accum_offset & 0x3Fu) << 24;
  w |= kRelaunchStateBit;
  return w;
}

uint32_t encode_gfx1250_state_word(uint32_t vgpr_count, uint32_t lds_size) {
  uint32_t w = ((vgpr_count / kCwsrGfx1250VgprGranule) - 1) & 0x3Fu;
  w |= ((lds_size / kCwsrGfx1250LdsGranuleBytes) & kCwsrGfx1250LdsFieldMax) << 10;
  w |= 1u << 24; // gfx1250 supports wave32 only.
  w |= kRelaunchStateBit;
  return w;
}

// Encode the COMPUTE_RELAUNCH "wave" word (bits 30/31 clear so it is neither an
// event nor a state word).
uint32_t encode_wave_word(const CwsrWaveState &wave) {
  // gfx9.4 wave payload: scratch slot [0:8], shader engine [9:10].
  uint32_t w = wave.scratch_scoreboard_id & 0x1FFu;
  w |= (wave.shader_engine_id & 0x3u) << 9;
  if (wave.is_last_in_group)
    w |= 1u << 16;
  if (wave.is_first_in_group)
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
  // The saved block holds kCwsrSavedSgprSlots scalars, so that is the ceiling a
  // wave can carry. It is deliberately not the architected count (102): s104 and
  // s105 are ordinary registers above the FLAT_SCRATCH alias, and CDNA3/CDNA4
  // default to 112 SGPRs per wave (config_loader default_sgprs_per_wf), so a
  // lower bound here would reject an ordinary dispatch and fail the queue's whole
  // CWSR publish. Slots that alias travel in their own fields either way.
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
                              rj_code_arch_t arch, uint32_t control_stack_capacity,
                              std::optional<CwsrDebugRegion> runtime_debug_region = {},
                              size_t context_bytes = 0) {
  CwsrGeometry geometry{};
  const CwsrAbiProperties abi = cwsr_abi_properties(arch);
  if (abi.layout_kind == CwsrLayoutKind::Unsupported || abi.lds_granule_bytes == 0 ||
      (waves.empty() && context_bytes == 0) || context_bytes > UINT32_MAX ||
      waves.size() > std::numeric_limits<uint32_t>::max())
    return geometry;

  const bool gfx12_5 = abi.layout_kind == CwsrLayoutKind::Gfx12_5;
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
  if (control_stack_capacity != 0 &&
      (control_stack_capacity < cwsr_header_size_bytes() + control_stack_size ||
       (control_stack_capacity % sizeof(uint32_t)) != 0))
    return geometry;
  const uint64_t wave_area_begin = control_stack_capacity != 0
                                       ? control_stack_capacity
                                       : kControlStackOffset + control_stack_size;
  const uint64_t control_stack_offset = wave_area_begin - control_stack_size;
  const uint64_t wave_state_offset = wave_area_begin + wave_state_size;
  if (control_stack_size > std::numeric_limits<uint32_t>::max() ||
      wave_state_size > std::numeric_limits<uint32_t>::max() ||
      wave_state_offset > std::numeric_limits<uint32_t>::max() || wave_state_offset > area_size)
    return geometry;

  const uint64_t image_end =
      wave_state_offset + (context_bytes == 0 ? 0 : 8 + ((uint64_t{context_bytes} + 3) / 4) * 4);
  if (image_end > area_size)
    return geometry;

  const uint64_t debug_size_64 =
      ((num_waves + kDebuggerReserveChunks) * kDebuggerBytesPerWave + kDebuggerBytesAlign - 1) /
      kDebuggerBytesAlign * kDebuggerBytesAlign;
  const uint64_t debug_offset_64 =
      (image_end + kDebuggerBytesAlign - 1) / kDebuggerBytesAlign * kDebuggerBytesAlign;
  const bool debug_fits = debug_offset_64 <= std::numeric_limits<uint32_t>::max() &&
                          debug_size_64 <= std::numeric_limits<uint32_t>::max() &&
                          debug_offset_64 + debug_size_64 <= area_size;

  geometry.layout.control_stack_offset = static_cast<uint32_t>(control_stack_offset);
  geometry.layout.control_stack_size = static_cast<uint32_t>(control_stack_size);
  geometry.layout.wave_state_offset = static_cast<uint32_t>(wave_state_offset);
  geometry.layout.wave_state_size = static_cast<uint32_t>(wave_state_size);
  if (runtime_debug_region) {
    // libhsakmt may place this region beyond all per-XCC save areas. When it
    // lies inside this area, the wave payload must not overwrite it.
    if (runtime_debug_region->size != 0 && runtime_debug_region->offset < image_end)
      return geometry;
    geometry.layout.debug_offset = runtime_debug_region->offset;
    geometry.layout.debug_size = runtime_debug_region->size;
  } else {
    geometry.layout.debug_offset = debug_fits ? static_cast<uint32_t>(debug_offset_64) : 0;
    geometry.layout.debug_size = debug_fits ? static_cast<uint32_t>(debug_size_64) : 0;
  }
  geometry.layout.ok = true;
  return geometry;
}

uint32_t image_word(std::span<const uint8_t> image, uint32_t offset) {
  uint32_t value = 0;
  std::memcpy(&value, image.data() + offset, sizeof(value));
  return value;
}

std::optional<CwsrLayout> decode_cwsr_header(const std::array<uint32_t, 4> &header,
                                             uint32_t area_size, uint32_t control_stack_capacity) {
  CwsrLayout layout{.control_stack_offset = header[0],
                    .control_stack_size = header[1],
                    .wave_state_offset = header[2],
                    .wave_state_size = header[3]};
  const uint64_t control_end =
      static_cast<uint64_t>(layout.control_stack_offset) + layout.control_stack_size;
  if (((header[0] | header[1] | header[2] | header[3]) % sizeof(uint32_t)) != 0 ||
      layout.control_stack_offset < cwsr_header_size_bytes() ||
      layout.control_stack_size < 2 * sizeof(uint32_t) || layout.wave_state_offset > area_size ||
      layout.wave_state_size > layout.wave_state_offset ||
      control_end != layout.wave_state_offset - layout.wave_state_size ||
      (control_stack_capacity != 0 && control_end != control_stack_capacity))
    return std::nullopt;
  layout.ok = true;
  return layout;
}

} // namespace

CwsrCapacity cwsr_capacity_for_producer(rj_code_arch_t arch, CwsrProducerLimits producer,
                                        uint32_t lds_size_kb) {
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
  if (producer.cus_per_xcc == 0)
    return {};
  const uint64_t active_cus = producer.cus_per_xcc;
  const uint64_t waves = producer.waves_per_xcc;
  if (waves > kCwsrMaxAlignedBytes / sizeof(uint32_t))
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
  if (control_raw > kCwsrMaxAlignedBytes || record_bytes == 0 ||
      waves > kCwsrMaxAlignedBytes / record_bytes)
    return {};
  const uint64_t control = (control_raw + kCwsrPageSize - 1) & ~(kCwsrPageSize - 1);
  const uint64_t record_total = waves * record_bytes;
  if (control > kCwsrMaxAlignedBytes - record_total || lds_size_kb > kCwsrMaxAlignedBytes / 1024)
    return {};

  // The CU allocator admits at most lds_size_kb of live workgroups per CU.
  // The codec saves each group's LDS separately, rounded to its record
  // granule. Allow one rounding remainder for every possible one-wave group.
  const uint64_t lds_granule = cwsr_lds_granule_bytes(arch);
  const uint64_t lds_per_cu = uint64_t{lds_size_kb} * 1024;
  if (lds_per_cu > cwsr_max_encoded_lds_bytes(arch))
    return {};
  const uint64_t remaining = kCwsrMaxAlignedBytes - control - record_total;
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
      (control + record_total + lds_total + debug_bytes + kCwsrPageSize - 1) & ~(kCwsrPageSize - 1);
  return {static_cast<uint32_t>(control), static_cast<uint32_t>(context)};
}

CwsrLayout serialize_queue_cwsr(uint64_t ctx_base, uint32_t area_size,
                                const std::vector<CwsrWaveState> &waves,
                                const std::function<void(uint64_t, uint32_t)> &write32,
                                rj_code_arch_t arch, uint32_t control_stack_capacity,
                                std::optional<CwsrDebugRegion> runtime_debug_region,
                                std::span<const uint8_t> queue_context) {
  CwsrLayout layout{};
  if (!cwsr_layout_modelled(arch) || (waves.empty() && queue_context.empty()) || !write32 ||
      (ctx_base & (alignof(uint32_t) - 1)) != 0)
    return layout;

  // Registers retain one queue-wide stride; ACC-VGPRs are not modeled.
  // A state change before a workgroup leader selects that group's LDS size.
  const CwsrGeometry geometry = compute_geometry(area_size, waves, arch, control_stack_capacity,
                                                 runtime_debug_region, queue_context.size());
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
  if (ctx_base > std::numeric_limits<uint64_t>::max() - area_size)
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
            gfx12_5 ? encode_gfx1250_wave_word(waves[i]) : encode_wave_word(waves[i]));
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
    auto ttmp = w.ttmps;
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

  // MODEL ASSUMPTION: the private CP checkpoint occupies unused space after
  // the public wave records, bounded above by the runtime debugger region.
  // This placement/encoding is rocjitsu-owned, not an AMD firmware format.
  if (!queue_context.empty()) {
    const uint64_t base = ctx_base + wave_state_offset;
    write32(base, kCwsrPrivateContextMagic);
    write32(base + 4, static_cast<uint32_t>(queue_context.size()));
    for (size_t offset = 0; offset < queue_context.size(); offset += sizeof(uint32_t)) {
      uint32_t word = 0;
      std::memcpy(&word, queue_context.data() + offset,
                  std::min(sizeof(word), queue_context.size() - offset));
      write32(base + 8 + offset, word);
    }
  }

  // Publish the ABI header after the complete payload. A caller that permits a
  // debugger to read concurrently must still provide the required exclusion.
  write32(ctx_base + 0, layout.control_stack_offset);
  write32(ctx_base + 4, layout.control_stack_size);
  write32(ctx_base + 8, layout.wave_state_offset);
  write32(ctx_base + 12, layout.wave_state_size);
  if (!runtime_debug_region) {
    write32(ctx_base + cwsr_debug_fields_offset(), layout.debug_offset);
    write32(ctx_base + cwsr_debug_fields_offset() + sizeof(uint32_t), layout.debug_size);
    write32(ctx_base + 24, 0); // err_payload_addr lo
    write32(ctx_base + 28, 0); // err_payload_addr hi
    write32(ctx_base + 32, 0); // err_event_id
    write32(ctx_base + 36, 0); // reserved1
  }

  return layout;
}

CwsrLayout serialize_queue_cwsr_bulk(
    uint64_t ctx_base, uint32_t area_size, const std::vector<CwsrWaveState> &waves,
    const std::function<void(uint64_t, std::span<const uint8_t>)> &write_block, rj_code_arch_t arch,
    uint32_t control_stack_capacity, std::optional<CwsrDebugRegion> runtime_debug_region,
    std::span<const uint8_t> queue_context) {
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
      arch, control_stack_capacity, runtime_debug_region, queue_context);
  if (!layout.ok)
    return layout;

  if (image.size() < layout.wave_state_offset)
    image.resize(layout.wave_state_offset);
  const std::span<const uint8_t> bytes{image};
  write_block(ctx_base + layout.control_stack_offset, bytes.subspan(layout.control_stack_offset));
  write_block(ctx_base, bytes.first(runtime_debug_region ? cwsr_debug_fields_offset()
                                                         : cwsr_header_size_bytes()));
  return layout;
}

bool deserialize_queue_cwsr_context(
    uint64_t ctx_base, uint32_t area_size, std::vector<uint8_t> &context,
    const std::function<void(uint64_t, std::span<uint8_t>)> &read_block,
    uint32_t control_stack_capacity) {
  if (!read_block || area_size < cwsr_header_size_bytes() || (ctx_base & 3u) != 0 ||
      ctx_base > UINT64_MAX - area_size)
    return false;
  constexpr size_t debug_fields_word = cwsr_debug_fields_offset() / sizeof(uint32_t);
  std::array<uint32_t, debug_fields_word + 2> header{};
  read_block(ctx_base, {reinterpret_cast<uint8_t *>(header.data()), sizeof(header)});
  const auto layout = decode_cwsr_header({header[0], header[1], header[2], header[3]}, area_size,
                                         control_stack_capacity);
  if (!layout)
    return false;
  const uint32_t debug_offset = header[debug_fields_word];
  const uint32_t debug_size = header[debug_fields_word + 1];
  const uint64_t limit = debug_size != 0 ? std::min(area_size, debug_offset) : area_size;
  const uint64_t offset = layout->wave_state_offset;
  if (offset + 8 > limit)
    return false;
  std::array<uint32_t, 2> prefix{};
  read_block(ctx_base + offset, {reinterpret_cast<uint8_t *>(prefix.data()), sizeof(prefix)});
  if (prefix[0] != kCwsrPrivateContextMagic)
    return false;
  const uint64_t size = prefix[1];
  if (size == 0 || offset + 8 + ((size + 3) / 4) * 4 > limit)
    return false;
  std::vector<uint8_t> decoded(size);
  read_block(ctx_base + offset + 8, decoded);
  context = std::move(decoded);
  return true;
}

bool deserialize_queue_cwsr(uint64_t ctx_base, uint32_t area_size,
                            std::vector<CwsrWaveState> &waves,
                            const std::function<uint32_t(uint64_t)> &read32, rj_code_arch_t arch,
                            uint32_t control_stack_capacity) {
  if (!cwsr_layout_modelled(arch) || !read32 || area_size < cwsr_header_size_bytes() ||
      (ctx_base & (alignof(uint32_t) - 1)) != 0 ||
      ctx_base > std::numeric_limits<uint64_t>::max() - area_size)
    return false;
  std::array<uint32_t, 4> header{};
  for (size_t i = 0; i < header.size(); ++i)
    header[i] = read32(ctx_base + i * sizeof(uint32_t));
  const auto layout = decode_cwsr_header(header, area_size, control_stack_capacity);
  if (!layout)
    return false;

  const auto abi = cwsr_abi_properties(arch);
  const bool gfx12_5 = abi.layout_kind == CwsrLayoutKind::Gfx12_5;
  const uint32_t sgpr_count = gfx12_5 ? kCwsrGfx1250SavedSgprSlots : kCwsrSavedSgprSlots;
  const uint32_t hwreg_bytes = (gfx12_5 ? kGfx1250HwregCount : kGfx94HwregCount) * sizeof(uint32_t);
  const uint32_t sgpr_bytes = sgpr_count * sizeof(uint32_t);
  const uint32_t lane_count = gfx12_5 ? 32 : 64;
  const uint32_t vgpr_lane_bytes = lane_count * sizeof(uint32_t);
  const uint32_t prefix_bytes = gfx12_5 ? 0 : 64;
  const uint64_t control_base = ctx_base + layout->control_stack_offset;
  const uint32_t control_words = layout->control_stack_size / sizeof(uint32_t);
  const uint64_t wave_area_begin = ctx_base + layout->wave_state_offset - layout->wave_state_size;
  uint64_t last_wave_area = ctx_base + layout->wave_state_offset;
  uint32_t vgpr_count = 0;
  uint32_t lds_bytes = 0;
  std::vector<CwsrWaveState> decoded;
  // Like ROCdbgapi's control_stack_iterate(), skip the two PM4 prefix words.
  for (uint32_t i = 2; i < control_words; ++i) {
    const uint32_t relaunch = read32(control_base + i * sizeof(uint32_t));
    if (relaunch & (1u << 30))
      continue; // event
    if (relaunch & kRelaunchStateBit) {
      if (gfx12_5) {
        if (i + 1 >= control_words || (relaunch & (1u << 24)) == 0)
          return false;
        ++i; // COMPUTE_RELAUNCH2; no shared VGPRs in the gfx12.5 layout.
        vgpr_count = ((relaunch & 0x3Fu) + 1) * 16;
        lds_bytes = ((relaunch >> 10) & 0x1FFu) * abi.lds_granule_bytes;
      } else {
        vgpr_count = (((relaunch >> 24) & 0x3Fu) + 1) * 4;
        // This codec models the fixed SGPR block and no separate ACC payload.
        if (((relaunch >> 6) & 0x7u) * 16 != sgpr_count ||
            ((relaunch & 0x3Fu) + 1) * 8 != vgpr_count)
          return false;
        lds_bytes = ((relaunch >> 9) & 0xFFu) * abi.lds_granule_bytes;
      }
      if (vgpr_count > isa_properties(arch).max_addressable_vgprs_per_wf)
        return false;
      continue;
    }
    if (vgpr_count == 0 || (gfx12_5 && (relaunch & ((1u << 27) | (1u << 28)))))
      return false; // No state, or a cluster record not modelled by this codec.
    CwsrWaveState w;
    w.is_first_in_group = (relaunch & (1u << (gfx12_5 ? 12 : 17))) != 0;
    w.is_last_in_group = (relaunch & (1u << (gfx12_5 ? 13 : 16))) != 0;
    w.scratch_scoreboard_id = relaunch & (gfx12_5 ? 0x3FFu : 0x1FFu);
    w.shader_engine_id = (relaunch >> (gfx12_5 ? 24 : 9)) & 0x3u;
    w.num_sgprs = sgpr_count;
    w.num_vgprs = vgpr_count;
    // Existing resume callers request only the live allocation's register
    // subset. Those counts never determine record addresses or wave ordering.
    if (!waves.empty()) {
      if (decoded.size() >= waves.size())
        return false;
      const auto &requested = waves[decoded.size()];
      if (requested.num_sgprs > sgpr_count || requested.num_vgprs > vgpr_count)
        return false;
      w.num_sgprs = requested.num_sgprs;
      w.num_vgprs = requested.num_vgprs;
    }
    const uint32_t vgpr_bytes = vgpr_count * vgpr_lane_bytes;
    const uint32_t saved_lds_bytes = w.is_first_in_group ? lds_bytes : 0;
    const uint64_t record_bytes = static_cast<uint64_t>(prefix_bytes) + hwreg_bytes + sgpr_bytes +
                                  vgpr_bytes + saved_lds_bytes;
    if (record_bytes > last_wave_area - wave_area_begin)
      return false;
    const uint64_t register_area_end = last_wave_area - prefix_bytes - saved_lds_bytes;
    const size_t live_lds_bytes =
        waves.empty() ? saved_lds_bytes
                      : (w.is_first_in_group ? waves[decoded.size()].lds.size() : 0);
    if (live_lds_bytes > saved_lds_bytes)
      return false;
    w.lds.resize(live_lds_bytes);
    for (uint32_t byte = 0; byte < live_lds_bytes; byte += sizeof(uint32_t)) {
      const uint32_t value = read32(register_area_end + byte);
      std::memcpy(w.lds.data() + byte, &value,
                  std::min<size_t>(sizeof(value), live_lds_bytes - byte));
    }
    const uint64_t hwregs_addr = register_area_end - hwreg_bytes;
    const uint64_t sgprs_addr = hwregs_addr - sgpr_bytes;
    const uint64_t vgprs_addr = sgprs_addr - vgpr_bytes;
    const uint64_t ttmps_addr = gfx12_5 ? sgprs_addr + (sgpr_count - kTtmpCount) * sizeof(uint32_t)
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

    for (size_t t = 0; t < w.ttmps.size(); ++t)
      w.ttmps[t] = read32(ttmps_addr + t * sizeof(uint32_t));
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
      w.trap_id = (ttmp6 >> 25) & 0xFu;
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
    const uint32_t vcc_lo = read32(sgprs_addr + kCwsrVccLoSlot * 4);
    const uint32_t vcc_hi = read32(sgprs_addr + (kCwsrVccLoSlot + 1) * 4);
    w.vcc = static_cast<uint64_t>(vcc_lo) | (static_cast<uint64_t>(vcc_hi) << 32);
    // FLAT_SCRATCH is serialized into its alias slots but was never read back,
    // so a serialize/deserialize round trip silently dropped it.
    if (!gfx12_5) {
      const uint32_t flat_scratch_lo_slot = kCwsrFlatScratchLoSlot;
      const uint32_t fs_lo = read32(sgprs_addr + flat_scratch_lo_slot * 4);
      const uint32_t fs_hi = read32(sgprs_addr + (flat_scratch_lo_slot + 1) * 4);
      w.flat_scratch = static_cast<uint64_t>(fs_lo) | (static_cast<uint64_t>(fs_hi) << 32);
    }

    w.vgprs.resize(static_cast<size_t>(w.num_vgprs) * 64);
    for (uint32_t r = 0; r < w.num_vgprs; ++r)
      for (uint32_t lane = 0; lane < lane_count; ++lane)
        w.vgprs[r * 64 + lane] = read32(vgprs_addr + r * vgpr_lane_bytes + lane * 4);

    last_wave_area = vgprs_addr;
    decoded.push_back(std::move(w));
  }
  if (last_wave_area != wave_area_begin || (!waves.empty() && decoded.size() != waves.size()))
    return false;
  waves = std::move(decoded);
  return true;
}

bool deserialize_queue_cwsr_bulk(
    uint64_t ctx_base, uint32_t area_size, std::vector<CwsrWaveState> &waves,
    const std::function<void(uint64_t, std::span<uint8_t>)> &read_block, rj_code_arch_t arch,
    uint32_t control_stack_capacity) {
  if (!cwsr_layout_modelled(arch) || !read_block || area_size < cwsr_header_size_bytes() ||
      (ctx_base & (alignof(uint32_t) - 1)) != 0 ||
      ctx_base > std::numeric_limits<uint64_t>::max() - area_size)
    return false;
  std::array<uint32_t, 4> header{};
  read_block(ctx_base, {reinterpret_cast<uint8_t *>(header.data()), sizeof(header)});
  const auto layout = decode_cwsr_header(header, area_size, control_stack_capacity);
  if (!layout)
    return false;

  // The runtime-owned debugger region does not determine the wave payload.
  std::vector<uint8_t> image(layout->wave_state_offset);
  std::memcpy(image.data(), header.data(), sizeof(header));
  std::span<uint8_t> payload{image.data() + layout->control_stack_offset,
                             layout->wave_state_offset - layout->control_stack_offset};
  read_block(ctx_base + layout->control_stack_offset, payload);
  return deserialize_queue_cwsr(
      ctx_base, area_size, waves,
      [&](uint64_t address) {
        return image_word(image, static_cast<uint32_t>(address - ctx_base));
      },
      arch, control_stack_capacity);
}

namespace {

uint32_t gfx1250_status(uint32_t status) {
  // SCC, priorities and HALT moved to STATE_PRIV. The remaining fields used by
  // dbgapi retain their gfx9 positions.
  return status & ~((0x1Fu << 0) | (1u << 7) | (1u << 13) | (1u << 19));
}

uint32_t internal_status_from_gfx1250(const CwsrWaveState &state) {
  uint32_t status = update_status_from_gfx12_state_priv(state.status, state.state_priv);
  status &= ~Wavefront::kStatusHaltMask;
  if (state.saved_status_halt)
    status |= Wavefront::kStatusHaltMask;
  return status;
}

uint32_t internal_trapsts_from_gfx1250(const CwsrWaveState &state) {
  uint32_t trapsts = update_trapsts_from_gfx12_excp_flag_priv(0, state.excp_flag_priv);
  return update_trapsts_from_gfx12_excp_flag_user(trapsts, state.excp_flag_user);
}

} // namespace

CwsrWaveState build_cwsr_wave_state(Wavefront &wf, rj_code_arch_t arch, bool runtime_save) {
  CwsrWaveState state;
  const bool gfx12_5 = cwsr_layout_kind(arch) == CwsrLayoutKind::Gfx12_5;
  const uint32_t raw_status = wf.status_raw();
  state.pc = wf.pc;
  if (runtime_save && wf.in_trap_handler() && wf.fatal_exception_pending()) {
    const uint32_t pc_hi_mask = gfx12_5 ? 0x01FFFFFFu : 0x0000FFFFu;
    state.pc = wf.ttmp(0) | (static_cast<uint64_t>(wf.ttmp(1) & pc_hi_mask) << 32);
  }
  // A wave frozen inside the trap handler has the HANDLER's EXEC live, not the
  // application's: the handler runs with its own mask and parks 0x80000000 (and
  // then a doorbell id) in EXEC_LO around MSG_GET_DOORBELL. Publishing that
  // makes the debugger report every lane active, which inverts `lane apply
  // -active/-inactive` (gdb.rocm/lane-info.exp). trap_saved_exec_ holds the
  // interrupted value for the whole handler, so un-shadow it here, as the
  // STATUS field below does with trap_saved_status().
  //
  // in_trap_handler() is the predicate for both, and it opens at handler entry
  // rather than at MSG_INTERRUPT because the doorbell park happens on the way
  // there. It closes at s_rfe, which puts the interrupted values back, so from
  // then on the live registers are the application's again.
  //
  // The record is not write-only -- apply_cwsr_to_wave() feeds it back on
  // resume -- so un-shadowing here is only safe because that path shadows in
  // the other direction under the same predicate: a debugger edit to EXEC is
  // written to trap_saved_exec_, which the handler's s_rfe installs, instead of
  // overwriting the mask a mid-flight handler is still running under.
  state.exec = wf.in_trap_handler() ? wf.trap_saved_exec() : wf.exec();
  state.vcc = wf.vcc();
  state.flat_scratch = wf.scratch_base();
  state.m0 = wf.m0();
  state.mode = wf.mode_raw();
  state.trapsts = wf.trapsts();
  // Whether the application was already halted when it trapped outlives the
  // handler: trap_saved_status_ still describes the interrupted wave after
  // s_rfe, and trap_interrupt_sent() is the flag that says a handler produced
  // this stop. in_trap_handler() has to be part of the predicate too, or the
  // two halves of the published STATUS come from different registers between
  // trap entry and MSG_INTERRUPT -- the body from trap_saved_status_ and the
  // HALT bit from the handler's live one. apply_cwsr_to_wave() recombines them
  // into a word that never existed on the wave, so an identity round-trip in
  // that window rewrites the application's saved HALT with the handler's.
  const bool handler_context = wf.in_trap_handler() || wf.trap_interrupt_sent();
  state.saved_status_halt = ((handler_context ? wf.trap_saved_status() : raw_status) >> 13) & 1u;
  state.wave_stopped = wf.debug_halted() || (runtime_save && wf.fatal_exception_pending());
  // The published STATUS shadows on the narrower predicate, the same one EXEC
  // uses: only while the handler is actually running is the live register the
  // handler's rather than the application's. Once s_rfe has run, the live value
  // is the application's again and is what the debugger should see.
  const uint32_t application_status = wf.in_trap_handler() ? wf.trap_saved_status() : raw_status;
  state.status =
      state.wave_stopped ? application_status | (1u << 13) : application_status & ~(1u << 13);
  if (gfx12_5) {
    uint32_t debug_status = application_status;
    if (state.wave_stopped)
      debug_status |= Wavefront::kStatusHaltMask;
    else
      debug_status &= ~Wavefront::kStatusHaltMask;
    state.state_priv = gfx12_state_priv_from_status(debug_status, state.flat_scratch != 0);
    state.status = gfx1250_status(application_status);
    state.excp_flag_priv = gfx12_excp_flag_priv_from_trapsts(state.trapsts);
    state.excp_flag_user = (state.trapsts & 0x7Fu) | wf.gfx12_excp_flag_user_extra_raw();
    state.trap_ctrl = wf.gfx12_trap_ctrl_raw();
    state.xnack_state_priv = wf.gfx1250_xnack_state_priv_raw();
    state.xnack_mask = wf.gfx1250_xnack_mask_raw();
  }
  for (uint32_t t = 0; t < state.ttmps.size(); ++t)
    state.ttmps[t] = wf.ttmp(t);
  state.trap_id = wf.trap_id();
  state.wave_id = wf.debug_wave_id();
  state.group_ids = wf.wg_coord();
  state.wave_in_group = wf.wave_in_group();
  state.queue_packet_id = wf.queue_packet_id() & 0x1FFFFFFu;
  state.scratch_scoreboard_id = wf.scratch_scoreboard_id();
  state.shader_engine_id = wf.shader_engine_id();
  state.spi_ttmps_setup = true;
  state.num_sgprs = wf.num_sgprs();
  state.num_vgprs = wf.num_vgprs();

  state.sgprs.resize(state.num_sgprs);
  for (uint32_t s = 0; s < state.num_sgprs; ++s)
    state.sgprs[s] = wf.debug_read_sgpr(s);
  // The CWSR record's VGPR stride is wave64-shaped (cwsr.cpp kVgprLaneBytes)
  // whatever the wave size is, but the physical register only has wf_size()
  // lanes -- a wave32 VectorReg<32> is exactly 32 wide, so reading lanes 32-63
  // runs off it into the neighbouring register. Read the live lanes and leave
  // the rest of the wave64-shaped slot at the zero resize() already wrote; the
  // restore path is bounded by wf_size() too and never looks at them.
  state.vgprs.resize(static_cast<size_t>(state.num_vgprs) * 64);
  const uint32_t live_lanes = wf.wf_size();
  for (uint32_t r = 0; r < state.num_vgprs; ++r)
    for (uint32_t lane = 0; lane < live_lanes; ++lane)
      state.vgprs[static_cast<size_t>(r) * 64 + lane] = wf.debug_read_vgpr(r, lane);
  state.lds.resize(wf.lds_size());
  if (!state.lds.empty())
    static_cast<const Lds &>(wf.lds()).read(wf.lds_base(), state.lds.data(),
                                            static_cast<uint32_t>(state.lds.size()));
  return state;
}

void prepare_cwsr_wave_states(std::vector<CwsrWaveState> &waves) {
  std::ranges::sort(waves, [](const auto &lhs, const auto &rhs) {
    if (lhs.queue_packet_id != rhs.queue_packet_id)
      return lhs.queue_packet_id < rhs.queue_packet_id;
    if (lhs.group_ids != rhs.group_ids)
      return lhs.group_ids < rhs.group_ids;
    return lhs.wave_in_group < rhs.wave_in_group;
  });
  auto same_group = [](const auto &lhs, const auto &rhs) {
    return lhs.queue_packet_id == rhs.queue_packet_id && lhs.group_ids == rhs.group_ids;
  };
  for (size_t index = 0; index < waves.size(); ++index) {
    waves[index].is_first_in_group = index == 0 || !same_group(waves[index], waves[index - 1]);
    waves[index].is_last_in_group =
        index + 1 == waves.size() || !same_group(waves[index], waves[index + 1]);
    if (!waves[index].is_first_in_group)
      waves[index].lds.clear();
  }
}

std::optional<std::vector<size_t>> match_cwsr_wave_states(std::span<const CwsrWaveState> states,
                                                          std::span<Wavefront *const> waves) {
  if (states.size() != waves.size())
    return std::nullopt;
  std::vector<size_t> matches;
  matches.reserve(states.size());
  std::vector<bool> matched(waves.size());
  for (const auto &state : states) {
    auto find_match = [&](bool by_wave_id) {
      for (size_t index = 0; index < waves.size(); ++index) {
        if (matched[index])
          continue;
        const auto *candidate = waves[index];
        if (by_wave_id ? state.wave_id != 0 && candidate->debug_wave_id() == state.wave_id
                       : candidate->debug_wave_id() == 0 &&
                             candidate->queue_packet_id() == state.queue_packet_id &&
                             candidate->wg_coord() == state.group_ids &&
                             candidate->wave_in_group() == state.wave_in_group)
          return index;
      }
      return waves.size();
    };
    size_t index = find_match(true);
    if (index == waves.size())
      index = find_match(false);
    if (index == waves.size())
      return std::nullopt;
    matched[index] = true;
    matches.push_back(index);
  }
  return matches;
}

void apply_cwsr_to_wave(Wavefront &wave, const CwsrWaveState &state, rj_code_arch_t arch,
                        CwsrRestorePolicy policy) {
  const bool restore_saved_registers = policy == CwsrRestorePolicy::RestoreSavedRegisters;
  if (restore_saved_registers)
    for (uint32_t t = 0; t < state.ttmps.size(); ++t)
      wave.set_ttmp(t, state.ttmps[t]);
  if (!state.lds.empty())
    wave.lds().write(wave.lds_base(), state.lds.data(), wave.lds_size());
  constexpr uint32_t kModeDebugEnMask = 1u << 11;
  constexpr uint32_t kStatusHaltMask = Wavefront::kStatusHaltMask;
  const bool gfx12_5 = cwsr_layout_kind(arch) == CwsrLayoutKind::Gfx12_5;
  wave.pc = state.pc;
  // Mirror of the un-shadowing in build_cwsr_wave_state(): while a handler is
  // mid-flight the published EXEC is the interrupted application mask, so the
  // value coming back belongs to the shadow, not to the live register the
  // handler is executing under. s_rfe installs it when the handler returns.
  if (wave.in_trap_handler())
    wave.set_trap_saved_exec(state.exec);
  else
    wave.set_exec(state.exec);
  wave.set_vcc_raw(state.vcc);
  wave.set_m0(state.m0);
  // STATUS shadows the same way EXEC does, and the cost of getting it wrong is
  // higher: the ROCr handler raises STATUS.HALT and only then returns, so a
  // record applied in that window used to clear the HALT the handler had just
  // set. s_rfe then saw a running wave, resumed it, and the breakpoint was
  // silently lost -- one inferior in gdb.rocm/multi-inferior-stress.exp running
  // its kernel to completion and exiting without ever stopping. Route the
  // record's HALT to the interrupted state the handler restores and leave the
  // live register to the handler.
  const uint32_t restored_status = gfx12_5 ? internal_status_from_gfx1250(state)
                                           : (state.status & ~kStatusHaltMask) |
                                                 (state.saved_status_halt ? kStatusHaltMask : 0u);
  if (wave.in_trap_handler())
    wave.set_trap_saved_status(restored_status);
  else
    wave.set_status_raw(restored_status);
  // The live STATUS.HALT stays as it is for a handler that raised it with
  // s_setreg: there the bit is the handler's own request to keep the wave
  // stopped, and clearing it on a resume loses the breakpoint --
  // DbgTrapCwsrShadowsTrapHandlerRegistersAndRoutesDebuggerEdits pins that.
  // A wave halted at s_sendmsghalt sits in the same window and wants the
  // opposite: it has already reported and is waiting to be let go, so leaving
  // the bit set strands it at the handler return. The record cannot tell the
  // two apart, which is why the wave records who raised the bit.
  if (!state.wave_stopped && wave.self_halted()) {
    wave.set_status_halt(false);
    wave.set_self_halted(false);
  }
  wave.set_mode_raw(state.mode);
  wave.set_trapsts(gfx12_5 ? internal_trapsts_from_gfx1250(state) : state.trapsts);
  if (gfx12_5) {
    wave.set_gfx12_excp_flag_user_extra_raw(state.excp_flag_user & (0x3u << 30));
    wave.set_gfx12_trap_ctrl_raw(state.trap_ctrl);
    wave.set_gfx1250_xnack_state_priv_raw(state.xnack_state_priv);
    wave.set_gfx1250_xnack_mask_raw(state.xnack_mask);
  }
  wave.set_debug_wave_id(state.wave_id);
  wave.set_queue_packet_id(state.queue_packet_id);
  // rocm-dbgapi may submit a lightweight control-state update with no register
  // payload after displaced stepping. Preserve the SGPR/VGPR values produced
  // by the displaced instruction in that case rather than restoring zeros.
  //
  // Skip exactly the slots the codec treats as aliases, not everything above
  // the architected count: s104/s105 sit between the FLAT_SCRATCH and VCC
  // aliases and are ordinary registers the debugger may edit. Sharing
  // cwsr_sgpr_slot_is_aliased() with the codec is what keeps the two ends of
  // the round trip from disagreeing about which slots carry register state.
  if (!state.sgprs.empty())
    for (uint32_t s = 0; s < state.num_sgprs && s < state.sgprs.size(); ++s)
      if (!cwsr_sgpr_slot_is_aliased(s, arch) && (restore_saved_registers || (s != 32 && s != 33)))
        wave.debug_write_sgpr(s, state.sgprs[s]);
  if (!state.vgprs.empty())
    for (uint32_t r = 0; r < state.num_vgprs; ++r)
      for (uint32_t lane = 0; lane < wave.wf_size(); ++lane) {
        const size_t index = static_cast<size_t>(r) * 64 + lane;
        if (index < state.vgprs.size())
          wave.debug_write_vgpr(r, lane, state.vgprs[index]);
      }
  // FLAT_SCRATCH travels in its own field because it aliases two SGPR slots the
  // loop above skips. This used to save and re-install wave.scratch_base()
  // around the loop, which the loop cannot reach anyway (scratch_base_ is a
  // standalone member), so the only effect was discarding the debugger's edit.
  wave.set_scratch_base(state.flat_scratch);
  const bool single_step = !state.wave_stopped && (gfx12_5 ? (state.trap_ctrl & (1u << 9)) != 0
                                                           : (state.mode & kModeDebugEnMask) != 0);
  wave.set_debug_single_step(single_step);
  wave.set_debug_halted(state.wave_stopped);
  // Keep debug_suspended set while restoring individual waves. KFD and MES
  // callers clear suspension only after every wave and LDS image in the queue
  // is restored, so no wave runs against a partially restored queue.
}

} // namespace rocjitsu::amdgpu
