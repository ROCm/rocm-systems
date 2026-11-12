// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

#include "checkpoint_generated.h"
#include "rocjitsu/vm/amdgpu/compute_unit.h"
#include "util/diagnostic.h"

namespace rocjitsu::config {

// The frontend programs the launch apertures and S_SETREG can separately
// change the shadow belonging to an individual running wave.
class Gfx10CheckpointResources {
public:
  static bool supported(const amdgpu::ComputeUnitCore &cu) {
    return cu.arch() == ROCJITSU_CODE_ARCH_RDNA1 || cu.arch() == ROCJITSU_CODE_ARCH_RDNA2;
  }

  static util::FailureOr<flatbuffers::Offset<fb::Gfx10CuResources>>
  save_cu(flatbuffers::FlatBufferBuilder &builder, const amdgpu::ComputeUnitCore &cu,
          util::DiagnosticEmitter emit_error) {
    // Restore builds a standalone machine. The default KFD handlers resolve by
    // process ID, and save_wave verifies ownership of resident waves. Custom controls,
    // exports and trace sinks can affect standalone waves and have no saved owner.
    if (cu.external_trap_resolver_ || cu.external_trap_updater_ || cu.external_sendmsg_handler_)
      return emit_error.emit() << "Cannot checkpoint GFX10 frontend callbacks on " << cu.name();
    if (cu.lds_allocation_pinned())
      return emit_error.emit() << "Cannot checkpoint shared LDS reservations on " << cu.name();
    const fb::Gfx10Apertures aperture(cu.shared_aperture_base_, cu.shared_aperture_limit_,
                                      cu.private_aperture_base_, cu.private_aperture_limit_,
                                      cu.flat_pointer32_);
    const auto size = static_cast<uint32_t>(cu.lds().materialized_size_bytes());
    flatbuffers::Offset<flatbuffers::Vector<uint8_t>> lds;
    if (size) {
      uint8_t *bytes = nullptr;
      lds = builder.CreateUninitializedVector<uint8_t>(size, &bytes);
      cu.lds().read(uint32_t{0}, bytes, size);
    }
    return fb::CreateGfx10CuResources(builder, &aperture, cu.next_lds_alloc_, lds);
  }

  static util::FailureOr<flatbuffers::Offset<fb::Gfx10WaveResources>>
  save_wave(flatbuffers::FlatBufferBuilder &builder, const amdgpu::ComputeUnitCore &cu,
            const amdgpu::Wavefront &wave, bool internal_address_space,
            util::DiagnosticEmitter emit_error) {
    if (!wave.wait_counters().empty() ||
        (wave.memory_wait_scoreboard() && !wave.memory_wait_scoreboard()->empty()) ||
        (wave.state() != amdgpu::WfState::RUNNING && wave.state() != amdgpu::WfState::BARRIER))
      return emit_error.emit() << "Cannot checkpoint pending GFX10 memory or wait state on "
                               << cu.name() << " wf" << wave.wf_id();
    if ((!internal_address_space &&
         (wave.address_space() || wave.vm_access() || wave.process_id())) ||
        wave.pm4_failure_ || wave.scratch_lease_ || (wave.lds_size() && &wave.lds() != &cu.lds()) ||
        (internal_address_space && wave.scratch_lane_size()))
      return emit_error.emit() << "Cannot checkpoint an externally owned GFX10 wave resource";
    // Trap entry captures private handler state and a debugger stop retains
    // its external owner. Preserve TRAPSTS for ordinary waves after return.
    const bool halt_requested = (wave.status_raw() & ((1u << 13) | (1u << 23))) != 0;
    if ((wave.status_raw() & (1u << 5)) || (wave.self_halted() && !halt_requested) ||
        wave.debug_halted() != wave.self_halted() || wave.fatal_exception_pending() ||
        wave.debug_single_step())
      return emit_error.emit() << "Cannot checkpoint GFX10 privileged or halted wave state";
    const fb::Gfx10Apertures aperture(wave.shared_aperture_base(), wave.shared_aperture_limit(),
                                      wave.private_aperture_base(), wave.private_aperture_limit(),
                                      wave.flat_pointer32());
    const auto &target = wave.wait_target();
    const fb::Gfx10WaitLimits limits(target.vmcnt, target.lgkmcnt, target.expcnt, target.vscnt);
    return fb::CreateGfx10WaveResources(
        builder, &aperture, wave.aperture_override_, &limits, wave.trapsts(), wave.scratch_base(),
        wave.scratch_lane_size(), wave.lds_base(), wave.lds_size(), wave.dispatch_id(),
        wave.wave_in_group(), wave.queue_id(), wave.queue_packet_id(), wave.num_vgprs(),
        wave.self_halted());
  }

  static util::Result restore_cu(const fb::Gfx10CuResources &s, amdgpu::ComputeUnitCore &cu,
                                 util::DiagnosticEmitter emit_error) {
    const auto &a = *s.launch_apertures();
    if ((a.ptr32() && cu.arch() != ROCJITSU_CODE_ARCH_RDNA1) ||
        (s.lds() && s.lds()->size() > cu.lds().size_bytes()) ||
        (s.next_lds_alloc() && (!s.lds() || s.next_lds_alloc() > s.lds()->size())) ||
        !cu.restore_lds_cursor(s.next_lds_alloc()))
      return emit_error.emit() << "Invalid GFX10 CU resource state";
    cu.set_apertures(a.shared_base(), a.shared_limit(), a.private_base(), a.private_limit(),
                     a.ptr32());
    if (s.lds())
      cu.lds().write(uint32_t{0}, s.lds()->data(), s.lds()->size());
    return util::Result::success();
  }

  static util::Result restore_wave(const fb::Gfx10WaveResources &s, amdgpu::ComputeUnitCore &cu,
                                   amdgpu::Wavefront &wave, util::DiagnosticEmitter emit_error) {
    const auto &a = *s.apertures();
    const auto &limits = *s.wait_limits();
    if ((a.ptr32() && cu.arch() != ROCJITSU_CODE_ARCH_RDNA1) ||
        uint64_t{s.lds_base()} + s.lds_size() > cu.lds().size_bytes() ||
        (wave.status_raw() & (1u << 5)) ||
        (s.self_halted() && !(wave.status_raw() & ((1u << 13) | (1u << 23)))) || limits.vm() > 63 ||
        limits.lgkm() > 63 || limits.exp() > 7 || limits.vs() > 63)
      return emit_error.emit() << "Invalid GFX10 wave resource state";
    wave.flat_pointer32_ = a.ptr32();
    wave.set_apertures(a.shared_base(), a.shared_limit(), a.private_base(), a.private_limit(),
                       s.override_launch());
    wave.set_wait_target(limits.vm(), limits.lgkm(), limits.exp());
    wave.set_wait_target_vscnt(limits.vs());
    wave.set_trapsts(s.trapsts());
    if (s.self_halted()) {
      wave.set_debug_halted(true);
      wave.set_self_halted(true);
    }
    wave.set_scratch_base(s.scratch_base());
    wave.set_scratch_lane_size(s.scratch_lane_size());
    wave.set_lds_base(s.lds_base());
    wave.set_lds_size(s.lds_size());
    wave.set_dispatch_id(s.dispatch_id());
    wave.set_wave_in_group(s.wave_in_group());
    wave.set_queue_id(s.queue_id());
    wave.set_queue_packet_id(s.queue_packet_id());
    return util::Result::success();
  }
};

} // namespace rocjitsu::config
