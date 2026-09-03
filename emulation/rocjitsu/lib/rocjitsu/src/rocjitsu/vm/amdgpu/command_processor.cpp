// Copyright (c) 2025-2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "rocjitsu/vm/amdgpu/command_processor.h"
#include "rocjitsu/code/kernel_descriptor_scan.h"
#include "rocjitsu/code/kernel_symbol.h"
#include "rocjitsu/isa/arch/amdgpu/generated/shared/isa_properties.h"
#include "rocjitsu/vm/amdgpu/gpu_vm.h"
#include "rocjitsu/vm/amdgpu/hsa_clock.h"
#include "rocjitsu/vm/amdgpu/mem_state.h"

#include "rocjitsu/base/rj_compiler.h"
RJ_DIAGNOSTIC_PUSH
RJ_DIAGNOSTIC_IGNORE_PEDANTIC
#include "hsa/amd_ext_aql_packet.h"
#include "hsa/amd_hsa_queue.h"
RJ_DIAGNOSTIC_POP

#include "simdojo/sim/message.h"
#include "simdojo/sim/simulation.h"
#include "util/bit.h"
#include "util/log.h"

#include <algorithm>
#include <atomic>
#include <bit>
#include <cassert>
#include <chrono>
#include <cstring>
#include <elf.h>
#include <format>
#include <limits>
#include <optional>
#include <stdexcept>
#include <string>
#include <sys/mman.h>
#include <thread>

namespace rocjitsu {
namespace amdgpu {

CommandProcessor::CommandProcessor(std::string name) : simdojo::Component(std::move(name)) {
  // Bind the doorbell handler at construction, not in startup(): register_queue()
  // may start the doorbell poll thread (which fires doorbell_event_ via
  // schedule_event_now) as soon as a host-accessible queue is registered, which can
  // happen before startup() runs. Binding here removes that ordering hazard — a
  // handlerless doorbell_event_ would be silently dropped by the engine.
  doorbell_event_.set_handler(
      [this](simdojo::Tick ts, simdojo::Message *) { handle_doorbell(ts); });
}

CommandProcessor::~CommandProcessor() { stop_doorbell_monitor(); }

void CommandProcessor::configure_for_arch(rj_code_arch_t arch) {
  // Matches LLVM's FeaturePackedTID: gfx90a and later CDNA targets, plus
  // GFX11 and later RDNA targets, receive work-item IDs packed in v0.
  packed_tid_ = arch == ROCJITSU_CODE_ARCH_CDNA2 || arch == ROCJITSU_CODE_ARCH_CDNA3 ||
                arch == ROCJITSU_CODE_ARCH_CDNA4 || arch == ROCJITSU_CODE_ARCH_RDNA3 ||
                arch == ROCJITSU_CODE_ARCH_RDNA3_5 || arch == ROCJITSU_CODE_ARCH_RDNA4 ||
                arch == ROCJITSU_CODE_ARCH_CDNA5;

  sdma_packet_dialect_ = SdmaPacketDialect::Legacy;
  if (arch == ROCJITSU_CODE_ARCH_CDNA5)
    sdma_packet_dialect_ = SdmaPacketDialect::Gfx1250;
  else if (arch == ROCJITSU_CODE_ARCH_RDNA3 || arch == ROCJITSU_CODE_ARCH_RDNA3_5 ||
           arch == ROCJITSU_CODE_ARCH_RDNA4)
    sdma_packet_dialect_ = SdmaPacketDialect::Gfx11Plus;
}

namespace {

// The supported cluster size must fit the M0 multicast mask captured at issue time.
constexpr uint32_t kMaxClusterWorkgroups = kClusterMulticastMaskBits;
static_assert(kMaxClusterWorkgroups <= kClusterMulticastMaskBits);
static_assert(kMaxClusterWorkgroups <= 16,
              "TTMP6 cluster max and max-flat-ID fields are 4 bits wide");

// GFX12 launch-state TTMP indices used by compiler-generated workgroup and
// cluster identity sequences. These are indices into the wave's trap-temporary
// file (Wavefront::ttmp()), not SGPR numbers: the shader reaches them through
// the TTMP operand encodings (scalar selectors 108..123), which the ISA decoder
// routes to that file rather than to the SGPR allocation.
constexpr uint32_t kGfx12Ttmp6 = 6;
constexpr uint32_t kGfx12Ttmp7 = 7;
constexpr uint32_t kGfx12Ttmp8 = 8;
constexpr uint32_t kGfx12Ttmp9 = 9;

// LLVM's gfx1250 architected-SGPR ABI maps TTMP6 as seven 4-bit fields:
// cluster-local XYZ, cluster-max XYZ, and max-flat-ID from low to high bits.
// TTMP7 holds 16-bit cluster-grid Y/Z IDs. TTMP8 holds queue-packet ID
// [24:0], wave-in-workgroup [29:25], grid-Y/Z-valid [30], and debug-mark
// [31]. TTMP9 holds cluster-grid X.
constexpr uint32_t kGfx12Ttmp6ClusterLocalXShift = 0;
constexpr uint32_t kGfx12Ttmp6ClusterLocalYShift = 4;
constexpr uint32_t kGfx12Ttmp6ClusterLocalZShift = 8;
constexpr uint32_t kGfx12Ttmp6ClusterMaxXShift = 12;
constexpr uint32_t kGfx12Ttmp6ClusterMaxYShift = 16;
constexpr uint32_t kGfx12Ttmp6ClusterMaxZShift = 20;
constexpr uint32_t kGfx12Ttmp6ClusterMaxFlatIdShift = 24;
constexpr uint32_t kGfx12Ttmp7ClusterGridDimensionMask = 0xFFFFu;
constexpr uint32_t kGfx12Ttmp8QueuePacketIdMask = 0x1FFFFFFu;
constexpr uint32_t kGfx12Ttmp8WaveIdInGroupShift = 25;
constexpr uint32_t kGfx12Ttmp8GridYzValidShift = 30;

struct PlannedWorkgroup {
  uint32_t local_wg_id = 0;
  uint32_t global_wg_id = 0;
  ComputeUnitCore *cu = nullptr;
};

uint32_t nonzero_or_one(uint32_t v) { return v == 0 ? 1 : v; }

uint32_t checked_ext_dispatch_grid_size(uint32_t cluster_count, uint32_t cluster_size,
                                        uint32_t workgroup_size, const char *axis) {
  if (cluster_count == 0 || cluster_size == 0 || workgroup_size == 0) {
    throw std::runtime_error(
        std::format("AMD extended dispatch {} fields must be nonzero: cluster_count={} "
                    "cluster_size={} workgroup_size={}",
                    axis, cluster_count, cluster_size, workgroup_size));
  }
  uint64_t grid_size =
      static_cast<uint64_t>(cluster_count) * cluster_size * static_cast<uint64_t>(workgroup_size);
  if (grid_size > std::numeric_limits<uint32_t>::max()) {
    throw std::runtime_error(std::format(
        "AMD extended dispatch grid_size_{} overflows 32 bits: cluster_count={} cluster_size={} "
        "workgroup_size={}",
        axis, cluster_count, cluster_size, workgroup_size));
  }
  return static_cast<uint32_t>(grid_size);
}

void validate_cluster_shape(const DispatchEntry &dp) {
  if (!dp.has_workgroup_clusters())
    return;
  auto cluster_size =
      static_cast<uint64_t>(dp.cluster_size_x) * dp.cluster_size_y * dp.cluster_size_z;
  // This also keeps every TTMP6 cluster dimension/max field within 4 bits.
  if (cluster_size == 0 || cluster_size > kMaxClusterWorkgroups) {
    throw std::runtime_error(
        std::format("unsupported workgroup cluster size {}x{}x{} ({} workgroups)",
                    dp.cluster_size_x, dp.cluster_size_y, dp.cluster_size_z, cluster_size));
  }
  if (!dp.cluster_grid_is_complete()) {
    throw std::runtime_error(std::format(
        "workgroup cluster shape {}x{}x{} count {}x{}x{} does not cover grid {}x{}x{} exactly",
        dp.cluster_size_x, dp.cluster_size_y, dp.cluster_size_z, dp.cluster_count_x,
        dp.cluster_count_y, dp.cluster_count_z, dp.grid_wgs_x, dp.grid_wgs_y, dp.grid_wgs_z));
  }
  const uint64_t rank_period = dp.cluster_rank_period();
  if (dp.workgroup_id_offset % rank_period != 0) {
    throw std::runtime_error(std::format(
        "clustered workgroup ID offset {} does not preserve cluster-local ranks; expected a "
        "multiple of {}",
        dp.workgroup_id_offset, rank_period));
  }
}

uint32_t read_memory_u32(GpuMemory *memory, uint64_t addr, uint32_t vmid = 0) {
  uint32_t value = 0;
  for (uint32_t i = 0; i < sizeof(value); ++i)
    value |= static_cast<uint32_t>(memory->read8(addr + i, vmid)) << (i * 8);
  return value;
}

uint32_t aligned_lds_bytes_per_workgroup(const DispatchEntry &entry) {
  // Match ComputeUnitCore::allocate_lds()/can_accept_workgroup() granularity for all dispatches.
  return util::align_up(entry.group_segment_fixed_size, 256u);
}

bool any_active_wavefronts(const std::vector<ComputeUnitCore *> &cus) {
  return std::any_of(cus.begin(), cus.end(), [](const auto *cu) { return cu->has_active_wfs(); });
}

bool plan_cluster_workgroups(const DispatchEntry &entry, uint32_t cluster_base_local_wg_id,
                             size_t next_cu, const std::vector<ComputeUnitCore *> &cus,
                             std::vector<PlannedWorkgroup> &plan, size_t &planned_next_cu) {
  plan.clear();
  uint32_t cluster_size = entry.cluster_size();
  const uint32_t lds_bytes_per_wg = aligned_lds_bytes_per_workgroup(entry);
  constexpr auto kU32Max = std::numeric_limits<uint32_t>::max();
  std::vector<uint32_t> planned_per_cu(cus.size(), 0);
  size_t last_cu_idx = next_cu;

  for (uint32_t rank = 0; rank < cluster_size; ++rank) {
    bool assigned = false;
    uint32_t local_wg_id = entry.cluster_peer_local_wg_id(cluster_base_local_wg_id, rank);
    for (size_t attempt = 0; attempt < cus.size(); ++attempt) {
      size_t cu_idx = (next_cu + rank + attempt) % cus.size();
      auto *cu = cus[cu_idx];

      uint32_t reserved_wgs = planned_per_cu[cu_idx] + 1;
      uint64_t reserved_wfs = static_cast<uint64_t>(entry.wfs_per_workgroup) * reserved_wgs;
      uint64_t reserved_lds = static_cast<uint64_t>(lds_bytes_per_wg) * reserved_wgs;
      if (reserved_wfs > kU32Max || reserved_lds > kU32Max)
        continue;
      if (!cu->can_accept_workgroup(static_cast<uint32_t>(reserved_wfs),
                                    static_cast<uint32_t>(reserved_lds)))
        continue;

      plan.push_back({local_wg_id, local_wg_id + entry.workgroup_id_offset, cu});
      ++planned_per_cu[cu_idx];
      last_cu_idx = cu_idx;
      assigned = true;
      break;
    }
    if (!assigned) {
      plan.clear();
      return false;
    }
  }

  planned_next_cu = (last_cu_idx + 1) % cus.size();
  return true;
}

bool sgpr_count_is_descriptor_encoded(rj_code_arch_t arch, uint32_t sgpr_gran) {
  if (sgpr_gran != 0)
    return true;
  return isa_properties(arch).descriptor_sgpr_count_encoded;
}

bool compute_pgm_rsrc1_mode_preserves_dx10_ieee(rj_code_arch_t arch) {
  /*
   * New ISA families should classify descriptor-to-MODE field initialization
   * for the architecture's MODE layout.
   */
  switch (arch) {
  case ROCJITSU_CODE_ARCH_CDNA1:
  case ROCJITSU_CODE_ARCH_CDNA2:
  case ROCJITSU_CODE_ARCH_CDNA3:
  case ROCJITSU_CODE_ARCH_CDNA4:
  case ROCJITSU_CODE_ARCH_RDNA1:
  case ROCJITSU_CODE_ARCH_RDNA2:
  case ROCJITSU_CODE_ARCH_RDNA3:
  case ROCJITSU_CODE_ARCH_RDNA3_5:
    return true;
  case ROCJITSU_CODE_ARCH_RDNA4:
  case ROCJITSU_CODE_ARCH_CDNA5:
  case ROCJITSU_CODE_ARCH_RV32I:
  case ROCJITSU_CODE_ARCH_RV64I:
  case ROCJITSU_CODE_ARCH_NUM_ARCHS:
    return false;
  }
  // Handle out-of-range values without a default, so -Wswitch catches new architectures.
  return false;
}

bool compute_pgm_rsrc1_mode_has_debug_field(rj_code_arch_t arch) {
  switch (arch) {
  case ROCJITSU_CODE_ARCH_CDNA1:
  case ROCJITSU_CODE_ARCH_CDNA2:
  case ROCJITSU_CODE_ARCH_CDNA3:
  case ROCJITSU_CODE_ARCH_CDNA4:
  case ROCJITSU_CODE_ARCH_RDNA1:
  case ROCJITSU_CODE_ARCH_RDNA2:
  case ROCJITSU_CODE_ARCH_RDNA3:
  case ROCJITSU_CODE_ARCH_RDNA3_5:
    return true;
  case ROCJITSU_CODE_ARCH_RDNA4:
  case ROCJITSU_CODE_ARCH_CDNA5:
  case ROCJITSU_CODE_ARCH_RV32I:
  case ROCJITSU_CODE_ARCH_RV64I:
  case ROCJITSU_CODE_ARCH_NUM_ARCHS:
    return false;
  }
  // Handle out-of-range values without a default, so -Wswitch catches new architectures.
  return false;
}

uint32_t initial_mode_from_compute_pgm_rsrc1(uint32_t rsrc1, rj_code_arch_t arch) {
  using namespace rocr::llvm::amdhsa;

  uint32_t mode = 0;
  mode |= AMDHSA_BITS_GET(rsrc1, COMPUTE_PGM_RSRC1_FLOAT_ROUND_MODE_32) << 0;
  mode |= AMDHSA_BITS_GET(rsrc1, COMPUTE_PGM_RSRC1_FLOAT_ROUND_MODE_16_64) << 2;
  mode |= AMDHSA_BITS_GET(rsrc1, COMPUTE_PGM_RSRC1_FLOAT_DENORM_MODE_32) << 4;
  mode |= AMDHSA_BITS_GET(rsrc1, COMPUTE_PGM_RSRC1_FLOAT_DENORM_MODE_16_64) << 6;
  if (compute_pgm_rsrc1_mode_preserves_dx10_ieee(arch)) {
    mode |= AMDHSA_BITS_GET(rsrc1, COMPUTE_PGM_RSRC1_ENABLE_DX10_CLAMP) << 8;
    mode |= AMDHSA_BITS_GET(rsrc1, COMPUTE_PGM_RSRC1_ENABLE_IEEE_MODE) << 9;
  }
  if (compute_pgm_rsrc1_mode_has_debug_field(arch))
    mode |= AMDHSA_BITS_GET(rsrc1, COMPUTE_PGM_RSRC1_DEBUG_MODE) << 11;
  if (AMDHSA_BITS_GET(rsrc1, COMPUTE_PGM_RSRC1_FP16_OVFL))
    mode |= Wavefront::FP16_OVFL_BIT;
  return mode;
}

} // namespace

void CommandProcessor::init_wavefront_regs(ComputeUnitCore *cu, Wavefront *wf,
                                           const DispatchEntry &pkt, uint32_t global_wg_id,
                                           uint32_t wf_index_in_wg) {
  using namespace rocr::llvm::amdhsa;
  uint32_t sbase = wf->sgpr_alloc().base;
  uint32_t kcp = pkt.kernel_code_properties;

  // User SGPRs per AMDHSA ABI: placed sequentially based on enable bits.
  // Order: private_segment_buffer(4), dispatch_ptr(2), queue_ptr(2),
  //        kernarg_segment_ptr(2), dispatch_id(2), flat_scratch_init(2),
  //        private_segment_size(1).
  // When kernel_code_properties is 0 (internal test dispatches), fall back to
  // the legacy layout: kernarg at s[0:1].
  int flat_scratch_init_sgpr = -1;
  if (kcp != 0) {
    const auto launch_metadata = dispatch_launch_metadata_.find(pkt.dispatch_id);
    if (pkt.queue_ptr != 0 && launch_metadata == dispatch_launch_metadata_.end())
      throw std::runtime_error("dispatch is missing captured launch metadata");
    uint32_t idx = 0;
    if (AMDHSA_BITS_GET(kcp, KERNEL_CODE_PROPERTY_ENABLE_SGPR_PRIVATE_SEGMENT_BUFFER)) {
      if (pkt.queue_ptr != 0) {
        for (uint32_t word = 0; word < 4; ++word)
          cu->write_sgpr(sbase + idx + word,
                         launch_metadata->second.scratch_resource_descriptor[word]);
      }
      idx += 4;
    }
    if (AMDHSA_BITS_GET(kcp, KERNEL_CODE_PROPERTY_ENABLE_SGPR_DISPATCH_PTR)) {
      cu->write_sgpr(sbase + idx, static_cast<uint32_t>(pkt.dispatch_ptr));
      cu->write_sgpr(sbase + idx + 1, static_cast<uint32_t>(pkt.dispatch_ptr >> 32));
      idx += 2;
    }
    if (AMDHSA_BITS_GET(kcp, KERNEL_CODE_PROPERTY_ENABLE_SGPR_QUEUE_PTR)) {
      cu->write_sgpr(sbase + idx, static_cast<uint32_t>(pkt.queue_ptr));
      cu->write_sgpr(sbase + idx + 1, static_cast<uint32_t>(pkt.queue_ptr >> 32));
      idx += 2;
    }
    if (AMDHSA_BITS_GET(kcp, KERNEL_CODE_PROPERTY_ENABLE_SGPR_KERNARG_SEGMENT_PTR)) {
      cu->write_sgpr(sbase + idx, static_cast<uint32_t>(pkt.kernarg_addr));
      cu->write_sgpr(sbase + idx + 1, static_cast<uint32_t>(pkt.kernarg_addr >> 32));
      util::Logger::vm("CP: init_wf kernarg s[", idx, ":", idx + 1, "] = 0x", std::hex,
                       pkt.kernarg_addr, std::dec, " sbase=", sbase);
      idx += 2;
    }
    if (AMDHSA_BITS_GET(kcp, KERNEL_CODE_PROPERTY_ENABLE_SGPR_DISPATCH_ID)) {
      const uint64_t dispatch_id =
          pkt.queue_ptr == 0 ? 0 : launch_metadata->second.write_dispatch_id;
      cu->write_sgpr(sbase + idx, static_cast<uint32_t>(dispatch_id));
      cu->write_sgpr(sbase + idx + 1, static_cast<uint32_t>(dispatch_id >> 32));
      idx += 2;
    }
    if (AMDHSA_BITS_GET(kcp, KERNEL_CODE_PROPERTY_ENABLE_SGPR_FLAT_SCRATCH_INIT)) {
      flat_scratch_init_sgpr = static_cast<int>(idx);
      idx += 2;
    }
    if (AMDHSA_BITS_GET(kcp, KERNEL_CODE_PROPERTY_ENABLE_SGPR_PRIVATE_SEGMENT_SIZE)) {
      cu->write_sgpr(sbase + idx, pkt.private_segment_fixed_size);
      idx += 1;
    }

    uint32_t preload_length = AMDHSA_BITS_GET(pkt.kernarg_preload, KERNARG_PRELOAD_SPEC_LENGTH);
    uint32_t preload_offset = AMDHSA_BITS_GET(pkt.kernarg_preload, KERNARG_PRELOAD_SPEC_OFFSET);
    if (preload_length != 0) {
      if (pkt.kernarg_addr == 0 || memory_ == nullptr)
        throw std::runtime_error("AMDHSA kernarg preload requires a mapped kernarg segment");
      if (idx + preload_length > pkt.num_user_sgprs)
        throw std::runtime_error("AMDHSA kernarg preload exceeds declared user SGPR count");
      uint32_t preload_end = preload_offset + preload_length;
      // Some assembly code objects leave descriptor kernarg_size at zero while
      // carrying the real size in metadata; treat zero as unknown.
      if (pkt.kernarg_size != 0 && preload_end > pkt.kernarg_size / sizeof(uint32_t))
        throw std::runtime_error("AMDHSA kernarg preload exceeds kernarg segment size");

      uint64_t preload_addr = pkt.kernarg_addr + static_cast<uint64_t>(preload_offset) * 4;
      for (uint32_t i = 0; i < preload_length; ++i)
        cu->write_sgpr(sbase + idx + i,
                       read_memory_u32(memory_, preload_addr + i * 4, pkt.process_id));
      util::Logger::vm("CP: init_wf kernarg preload s[", idx, ":", idx + preload_length - 1,
                       "] length=", preload_length, " offset=", preload_offset, " sbase=", sbase);
      idx += preload_length;
    }
  } else {
    // Legacy: kernarg at s[0:1].
    if (pkt.kernarg_addr != 0) {
      cu->write_sgpr(sbase + 0, static_cast<uint32_t>(pkt.kernarg_addr));
      cu->write_sgpr(sbase + 1, static_cast<uint32_t>(pkt.kernarg_addr >> 32));
    }
  }

  uint32_t gx = pkt.grid_wgs_x > 0 ? pkt.grid_wgs_x : 1;
  uint32_t gy = pkt.grid_wgs_y > 0 ? pkt.grid_wgs_y : 1;
  uint32_t grid_wg_id_x = global_wg_id % gx;
  uint32_t wg_id_y = (global_wg_id / gx) % gy;
  uint32_t wg_id_z = global_wg_id / (gx * gy);
  uint32_t wg_id_x = (pkt.enable_wg_id_y || pkt.enable_wg_id_z) ? grid_wg_id_x : global_wg_id;

  // System SGPRs: workgroup_id_{x,y,z} placed sequentially after user SGPRs.
  // Only the IDs whose enable bits are set in compute_pgm_rsrc2 are written.
  // When kernel_code_properties is 0 (internal test dispatches), always write
  // workgroup_id_x as a fallback since internal kernels expect it.
  uint32_t sys_idx = pkt.num_user_sgprs;
  {
    bool kcp_zero = (pkt.kernel_code_properties == 0);
    if (pkt.enable_wg_id_x || kcp_zero)
      cu->write_sgpr(sbase + sys_idx++, wg_id_x);
    if (pkt.enable_wg_id_y)
      cu->write_sgpr(sbase + sys_idx++, wg_id_y);
    if (pkt.enable_wg_id_z)
      cu->write_sgpr(sbase + sys_idx++, wg_id_z);
  }
  const auto properties = isa_properties(cu->arch());
  if (properties.uses_ttmp_workgroup_ids) {
    // The ordinary TTMP ABI uses grid coordinates. Targets advertising the
    // clustered extension reinterpret these fields below.
    uint32_t ttmp6 = 0;
    uint32_t ttmp7 = ((wg_id_z & kGfx12Ttmp7ClusterGridDimensionMask) << 16) |
                     (wg_id_y & kGfx12Ttmp7ClusterGridDimensionMask);
    uint32_t ttmp8 = pkt.queue_packet_id & kGfx12Ttmp8QueuePacketIdMask;
    ttmp8 |= wf_index_in_wg << kGfx12Ttmp8WaveIdInGroupShift;
    if (pkt.grid_yz_valid)
      ttmp8 |= 1u << kGfx12Ttmp8GridYzValidShift;
    uint32_t ttmp9 = grid_wg_id_x;
    if (properties.uses_cluster_ttmp_workgroup_ids) {
      const uint32_t cluster_size_x = nonzero_or_one(pkt.cluster_size_x);
      const uint32_t cluster_size_y = nonzero_or_one(pkt.cluster_size_y);
      const uint32_t cluster_size_z = nonzero_or_one(pkt.cluster_size_z);
      const WorkgroupCoord cluster_local = pkt.cluster_local_wg_coord_for_flat_wg_id(global_wg_id);
      const uint32_t cluster_max_x = cluster_size_x - 1;
      const uint32_t cluster_max_y = cluster_size_y - 1;
      const uint32_t cluster_max_z = cluster_size_z - 1;
      const uint32_t cluster_max_flat_id = cluster_size_x * cluster_size_y * cluster_size_z - 1;

      ttmp6 = (cluster_local.x << kGfx12Ttmp6ClusterLocalXShift) |
              (cluster_local.y << kGfx12Ttmp6ClusterLocalYShift) |
              (cluster_local.z << kGfx12Ttmp6ClusterLocalZShift) |
              (cluster_max_x << kGfx12Ttmp6ClusterMaxXShift) |
              (cluster_max_y << kGfx12Ttmp6ClusterMaxYShift) |
              (cluster_max_z << kGfx12Ttmp6ClusterMaxZShift) |
              (cluster_max_flat_id << kGfx12Ttmp6ClusterMaxFlatIdShift);
      const uint32_t cluster_grid_y =
          (wg_id_y / cluster_size_y) & kGfx12Ttmp7ClusterGridDimensionMask;
      const uint32_t cluster_grid_z =
          (wg_id_z / cluster_size_z) & kGfx12Ttmp7ClusterGridDimensionMask;
      ttmp7 = (cluster_grid_z << 16) | cluster_grid_y;
      ttmp9 = grid_wg_id_x / cluster_size_x;
    }
    wf->set_ttmp(kGfx12Ttmp6, ttmp6);
    wf->set_ttmp(kGfx12Ttmp7, ttmp7);
    wf->set_ttmp(kGfx12Ttmp8, ttmp8);
    wf->set_ttmp(kGfx12Ttmp9, ttmp9);
  }

  // Workitem IDs per AMDHSA ABI. The SPI decomposes the flat thread index
  // into (x, y, z) using the AQL packet's workgroup dimensions.
  // enable_vgpr_workitem_id (TIDIG_COMP_CNT from compute_pgm_rsrc2):
  //   0 = v0 only (workitem_id_x)
  //   1 = v0 + v1 (workitem_id_x, workitem_id_y)
  //   2 = v0 + v1 + v2 (workitem_id_x, workitem_id_y, workitem_id_z)
  // On packed-TID targets (CDNA3/4 and GFX11+): v0[9:0]=X, v0[19:10]=Y,
  // v0[29:20]=Z. TIDIG_COMP_CNT controls which components the SPI supplies;
  // unused packed components are zero.
  uint32_t vbase = wf->vgpr_alloc().base;
  for (uint32_t lane = 0; lane < wf->wf_size(); ++lane) {
    const WorkitemCoord id = workitem_local_coord(pkt, wf_index_in_wg, lane, wf->wf_size());
    if (packed_tid_) {
      cu->write_vgpr(vbase, lane, pack_workitem_id(id, pkt.enable_vgpr_workitem_id));
    } else {
      cu->write_vgpr(vbase, lane, id.x);
      if (pkt.enable_vgpr_workitem_id >= 1)
        cu->write_vgpr(vbase + 1, lane, id.y);
      if (pkt.enable_vgpr_workitem_id >= 2)
        cu->write_vgpr(vbase + 2, lane, id.z);
    }
  }

  // Scratch (private segment) setup.
  // Each wavefront gets a unique slice of scratch memory. The per-lane
  // private size is private_segment_fixed_size; the per-wave region is
  // that multiplied by wf_size. The global wave index is derived from
  // (global_wg_id, wf_index_in_wg) to ensure non-overlapping scratch
  // across all CUs and workgroups in the dispatch.
  if (pkt.private_segment_fixed_size > 0) {
    uint64_t scratch_pool = pkt.scratch_backing_addr;
    if (scratch_pool == 0)
      scratch_pool = 0x1'0000'0000ULL;
    // Round the per-wave region to the 1 KB COMPUTE_TMPRING_SIZE.WAVESIZE granule
    // so that each wave's base equals scratch_pool + scoreboard_id * wavesize,
    // which is exactly what rocm-dbgapi computes to locate a wave's private
    // memory (rocdbgapi architecture.cpp scratch_memory_region).
    uint64_t raw_per_wave = static_cast<uint64_t>(pkt.private_segment_fixed_size) * wf->wf_size();
    uint64_t per_wave_size = ((raw_per_wave + 1023) / 1024) * 1024;
    uint32_t wg_total_size = static_cast<uint32_t>(pkt.workgroup_size_x) *
                             std::max<uint16_t>(1, pkt.workgroup_size_y) *
                             std::max<uint16_t>(1, pkt.workgroup_size_z);
    uint32_t waves_per_wg = (wg_total_size + wf->wf_size() - 1) / wf->wf_size();
    uint64_t global_wave_idx = static_cast<uint64_t>(global_wg_id) * waves_per_wg + wf_index_in_wg;
    uint64_t scratch_slot = global_wave_idx;
    if (cu->arch() == ROCJITSU_CODE_ARCH_CDNA5) {
      const uint32_t shader_engine_count =
          std::max(scratch_wave_divisor_, scratch_shader_engine_count_);
      const uint32_t shader_engine_id = wf->shader_engine_id();
      const uint32_t scoreboard_id = wf->scratch_scoreboard_id();
      assert(shader_engine_id < shader_engine_count);
      assert(scoreboard_id < scratch_waves_per_se_);
      scratch_slot =
          (static_cast<uint64_t>(scratch_xcc_id_) * shader_engine_count + shader_engine_id) *
              scratch_waves_per_se_ +
          scoreboard_id;
    } else {
      // Legacy CWSR records use the dispatch-wide logical scratch slot.
      wf->set_scratch_scoreboard_id(static_cast<uint32_t>(global_wave_idx));
    }
    uint64_t wave_scratch = scratch_pool + scratch_slot * per_wave_size;

    if (memory_ && memory_->resolve_host_ptr(wave_scratch, pkt.process_id) == nullptr &&
        scratch_allocator_) {
      // Size against the whole grid, not this XCD's share: every XCD of a
      // fanned-out dispatch shares the allocation. CDNA5 uses the complete
      // physical XCC/SE/scoreboard address space instead of logical grid slots.
      uint64_t scratch_slots = static_cast<uint64_t>(pkt.grid_total_wgs()) * waves_per_wg;
      if (cu->arch() == ROCJITSU_CODE_ARCH_CDNA5) {
        const uint32_t shader_engine_count =
            std::max(scratch_wave_divisor_, scratch_shader_engine_count_);
        scratch_slots =
            static_cast<uint64_t>(scratch_xcc_count_) * shader_engine_count * scratch_waves_per_se_;
      }
      uint64_t total_scratch = per_wave_size * scratch_slots;
      scratch_allocator_(pkt.process_id, scratch_pool, static_cast<size_t>(total_scratch));
    }

    wf->set_scratch_base(wave_scratch);
    wf->set_scratch_lane_size(pkt.private_segment_fixed_size);
    // CDNA compiler-generated functions use s32 as the private stack pointer
    // and s33 as its current frame value for explicit scratch SADDR operands.
    // The pointer is an offset within the per-wave scratch slice, not the SRD
    // base address supplied in the user SGPR block.
    if (cu->config().arch == ROCJITSU_CODE_ARCH_CDNA3 ||
        cu->config().arch == ROCJITSU_CODE_ARCH_CDNA4) {
      cu->write_sgpr(sbase + 32, 32);
      cu->write_sgpr(sbase + 33, 0);
    }
    util::Logger::cp([&](auto &os) {
      os << std::format(
          "SCRATCH wf{} pool={:#x} wave_scratch={:#x} per_wave={} priv_size={} "
          "backing_addr={:#x} mapped={}",
          wf->wf_id(), scratch_pool, wave_scratch, per_wave_size, pkt.private_segment_fixed_size,
          pkt.scratch_backing_addr,
          memory_ ? (memory_->resolve_host_ptr(wave_scratch, pkt.process_id) != nullptr) : false);
    });

    if (flat_scratch_init_sgpr >= 0) {
      cu->write_sgpr(sbase + flat_scratch_init_sgpr, static_cast<uint32_t>(wave_scratch));
      cu->write_sgpr(sbase + flat_scratch_init_sgpr + 1, static_cast<uint32_t>(wave_scratch >> 32));
    }
  }
}

void CommandProcessor::startup() {
  // INVARIANT: this CP and every CU it dispatches to share one partition (engine
  // thread). on_cu_idle() dispatches inline and calls ComputeUnitCore::schedule_work()
  // on those CUs, which mutates their non-atomic executing_/tick_event_ and pushes to
  // the partition event queue without synchronization — safe only same-partition. The
  // generic balanced partitioner could in principle split a CP from a CU under
  // num_threads > 1; assert here (after partitioning, before the run loop) so any
  // such split fails loudly rather than silently racing.
  for ([[maybe_unused]] const auto *cu : cus_)
    assert(cu->partition_id() == partition_id() &&
           "CommandProcessor and its compute units must share one partition");
  // doorbell_event_'s handler is bound in the constructor (see there) so it is live
  // before register_queue() can start the poll thread; nothing to (re)bind here.
  completion_ = std::make_unique<CompletionTracker>(memory_, gpu_vm_, cus_);
  completion_->set_plugin_group(plugin_group_);
  completion_->set_dispatch_retired_callback([this](const DispatchEntry &entry) {
    erase_cluster_workgroups(entry.dispatch_id);
    dispatch_launch_metadata_.erase(entry.dispatch_id);
  });
  completion_->set_grid_retired_callback([this](const DispatchEntry &) { wake_all_xcds(); });
}

void CommandProcessor::shutdown() {
  stop_doorbell_monitor();
  if (is_primary_ && engine()) {
    engine()->primary_release();
    is_primary_ = false;
  }
  completion_.reset();
}

void CommandProcessor::set_xcd_topology(uint32_t rank, std::vector<CommandProcessor *> peers) {
  assert(rank < peers.size() && "XCD rank must index its own SoC's CP list");
  assert(peers[rank] == this && "XCD rank must be this CP's own position");
  xcd_rank_ = rank;
  xcd_peers_ = std::move(peers);
  // Carve this XCD its own dispatch-id space; see allocate_dispatch_id().
  dispatch_id_stride_ = static_cast<uint32_t>(xcd_peers_.size());
  dispatch_id_base_ = 1 + rank;
  next_dispatch_id_ = dispatch_id_base_;
}

HwQueueState *CommandProcessor::find_queue_state(uint32_t queue_id, uint32_t process_id) {
  for (size_t i = 0; i < hw_queues_.size(); ++i) {
    if (hw_queues_[i].queue_id == queue_id && hw_queues_[i].process_id == process_id)
      return &new_queue_states_[i];
  }
  return nullptr;
}

void CommandProcessor::accept_fanout_shard(DispatchEntry shard) {
  accept_fanout_shard(std::move(shard), {});
}

void CommandProcessor::accept_fanout_shard(DispatchEntry shard,
                                           DispatchLaunchMetadata launch_metadata) {
  {
    util::Logger::cp([&](auto &os) {
      os << std::format("{}: FANOUT_SHARD d={} rank={}/{} wgs={}", name(), shard.dispatch_id,
                        shard.shard.rank(), shard.shard.stride(), shard.total_wgs);
    });
    // Deliberately NOT hw_queue_mutex_. The caller runs under its own CP's
    // hw_queue_mutex_ (fan-out happens inside handle_doorbell), so taking a peer's
    // hw_queue_mutex_ here would let two CPs fanning out concurrently acquire each
    // other's locks in opposite orders. Nothing that can lead back to another CP's
    // hw_queue_mutex_ is acquired while holding this one, and it is held only for
    // the push so a peer's engine thread never blocks on it for long.
    std::lock_guard<std::mutex> lock(fanout_inbox_mutex_);
    if (!shard.is_non_kernel())
      fanout_launch_metadata_inbox_.insert_or_assign(shard.dispatch_id, std::move(launch_metadata));
    fanout_inbox_.push_back(std::move(shard));
  }
  // Cross-thread and cross-partition safe: the engine buffers the event and drains
  // it into this CP's partition at its next safe point. Dispatching inline here
  // would reach into another partition's compute units.
  if (engine())
    engine()->schedule_event_now(doorbell_event());
}

void CommandProcessor::accept_dispatch_fault(DispatchFaultNotification fault) {
  {
    std::lock_guard<std::mutex> lock(dispatch_fault_inbox_mutex_);
    dispatch_fault_inbox_.push_back(fault);
  }
  if (engine())
    engine()->schedule_event_now(doorbell_event());
}

void CommandProcessor::drain_dispatch_fault_inbox() {
  std::vector<DispatchFaultNotification> faults;
  {
    std::lock_guard<std::mutex> lock(dispatch_fault_inbox_mutex_);
    faults.swap(dispatch_fault_inbox_);
  }
  for (const DispatchFaultNotification &fault : faults) {
    (void)fault_dispatch_local(fault.queue_id, fault.process_id, fault.dispatch_id, fault.outcome);
  }
}

void CommandProcessor::drain_fanout_inbox() {
  std::vector<DispatchEntry> inbox;
  std::unordered_map<uint32_t, DispatchLaunchMetadata> launch_metadata;
  {
    std::lock_guard<std::mutex> lock(fanout_inbox_mutex_);
    inbox.swap(fanout_inbox_);
    launch_metadata.swap(fanout_launch_metadata_inbox_);
  }
  // Real work arrived: the next wait this CP takes starts from a tight re-check.
  // Reset here rather than in accept_fanout_shard(), which runs on the OWNER's
  // thread -- writing this CP's backoff from there races the reads and writes its
  // own partition thread makes in arm_stall_recheck(). The shard is not visible to
  // this CP until it is drained anyway, and the drain runs before the re-arm in the
  // same handler pass, so resetting here is both correct and correctly ordered.
  if (inbox.empty())
    return;
  stall_recheck_backoff_ = 1;

  std::lock_guard<std::recursive_mutex> lock(hw_queue_mutex_);
  for (auto &shard : inbox) {
    const auto queue = std::ranges::find_if(hw_queues_, [&](const HwQueue &candidate) {
      return candidate.queue_id == shard.queue_id && candidate.process_id == shard.process_id;
    });
    if (queue == hw_queues_.end()) {
      // The replica was destroyed between the owner handing this shard over and
      // this drain. Drop it, exactly as unregister_queue drops a share it never
      // published: these workgroups have not run and this XCD's caches have not
      // been written back, so crediting the share would let the owner retire the
      // grid and fire the completion signal for work that never executed.
      //
      // Nothing is stranded by dropping it, for the reason unregister_queue
      // already relies on: a fan-out queue is destroyed on every XCD at once, so
      // the teardown that removed this replica removes the owner too. KFD
      // teardown is what reaches this window -- for_each_cp removes replicas in
      // XCD order while a later owner is still registered, and an
      // already-scheduled peer doorbell can drain concurrently.
      continue;
    }
    // A terminal VM fault closes execution admission for the queue. A peer may
    // already have emitted another shard before it observed the shared fault;
    // dropping it here prevents future work from appearing behind the fault.
    if (queue->faulted)
      continue;
    auto &qs = new_queue_states_[static_cast<size_t>(queue - hw_queues_.begin())];
    // Honour the packet's acquire fence on this XCD too. The owner invalidated
    // only its own CUs; this runs on our partition's thread, so ours are safe
    // to touch here and the peer ends up with the same view the owner has.
    if (shard.acquire_invalidate)
      flush_gpu_caches();
    if (!shard.is_non_kernel()) {
      const auto metadata = launch_metadata.find(shard.dispatch_id);
      if (metadata == launch_metadata.end()) {
        queue->faulted = true;
        continue;
      }
      dispatch_launch_metadata_.insert_or_assign(shard.dispatch_id, std::move(metadata->second));
    }
    qs.push_entry(std::move(shard));
  }
}

void CommandProcessor::wake_all_xcds() {
  // Cross-partition safe: the engine buffers each event into the target's own
  // partition and never re-enters the component, so this is callable while
  // holding hw_queue_mutex_.
  for (auto *peer : xcd_peers_) {
    if (peer && peer->engine())
      peer->engine()->schedule_event_now(peer->doorbell_event());
  }
}

void CommandProcessor::fan_out_dispatch(DispatchEntry &dp,
                                        const DispatchLaunchMetadata &launch_metadata) {
  const auto num_xcds = static_cast<uint32_t>(xcd_peers_.size());
  if (num_xcds <= 1)
    return;

  const uint32_t grid_wgs = dp.total_wgs;
  auto grid = std::make_shared<GridCompletion>();
  grid->grid_wgs = grid_wgs;

  for (uint32_t rank = 0; rank < num_xcds; ++rank) {
    if (rank == xcd_rank_)
      continue;
    DispatchEntry shard = dp;
    shard.grid_completion = grid;
    shard.fanout_peer = true;
    // The peer must not fire the dispatch's completion signal; the owning XCD
    // does that once the grid counter shows every share retired.
    shard.completion_signal = 0;
    shard.apply_shard(XcdShard(rank, num_xcds));
    xcd_peers_[rank]->accept_fanout_shard(std::move(shard), launch_metadata);
  }

  dp.grid_completion = std::move(grid);
  dp.apply_shard(XcdShard(xcd_rank_, num_xcds));
}

void CommandProcessor::replicate_non_kernel_entry(const DispatchEntry &dp) {
  const auto num_xcds = static_cast<uint32_t>(xcd_peers_.size());
  if (num_xcds <= 1)
    return;

  assert(dp.is_non_kernel() && "only packets that run no shader are replicated whole");
  for (uint32_t rank = 0; rank < num_xcds; ++rank) {
    if (rank == xcd_rank_)
      continue;
    DispatchEntry copy = dp;
    copy.fanout_peer = true;
    copy.completion_signal = 0;
    xcd_peers_[rank]->accept_fanout_shard(std::move(copy));
  }
}

uint64_t CommandProcessor::register_queue(HwQueue queue) {
  util::Logger::cp([&](auto &os) {
    os << std::format("{}: REGISTER_QUEUE id={} pid={} ring={:#x} size={} rptr={:#x} wptr={:#x} "
                      "doorbell_off={} is_sdma={} db_base={}",
                      name(), queue.queue_id, queue.process_id, queue.ring_base_va, queue.ring_size,
                      queue.read_ptr_va, queue.write_ptr_va, queue.doorbell_offset, queue.is_sdma,
                      reinterpret_cast<uintptr_t>(queue.doorbell_base));
  });
  // A replica exists only to receive dispatch shards from the XCD that owns the
  // queue. It must never read the ring or poll the doorbell, or the same packets
  // would be dispatched once per XCD.
  bool start_poll = queue.host_accessible && !queue.fanout_replica;
  // Replicate onto the peer XCDs without holding this CP's lock:
  // accept_fanout_shard() and the peers' register_queue() take their own locks.
  // A shard can arrive only after this function returns, so the owner may be
  // committed last without exposing an incomplete topology.
  {
    // Checked before replicating, so a rejection cannot leave replicas behind on
    // the peers. Shard routing keys on (queue_id, process_id), so a duplicate would
    // silently deliver every shard to whichever slot matched first -- a wrong-answer
    // bug rather than a crash, which is precisely what an assert compiled out of a
    // release build would let through. Enforced in every build for that reason.
    std::lock_guard<std::recursive_mutex> lock(hw_queue_mutex_);
    if (find_queue_state(queue.queue_id, queue.process_id) != nullptr) {
      throw std::runtime_error(
          std::format("duplicate queue registration on {}: queue_id={} process_id={} is already "
                      "registered, and fan-out routes dispatch shards by that key",
                      name(), queue.queue_id, queue.process_id));
    }
    queue.registration_id = next_queue_registration_id_++;
    if (next_queue_registration_id_ == 0)
      next_queue_registration_id_ = 1;
  }
  const auto num_xcds = static_cast<uint32_t>(xcd_peers_.size());
  bool replicate = queue.xcd_fanout && num_xcds > 1;
  std::vector<uint32_t> registered_peer_ranks;
  if (replicate) {
    registered_peer_ranks.reserve(num_xcds - 1);
    try {
      for (uint32_t rank = 0; rank < num_xcds; ++rank) {
        if (rank == xcd_rank_)
          continue;
        HwQueue replica = queue;
        replica.registration_id = 0;
        replica.xcd_fanout = false;
        replica.fanout_replica = true;
        xcd_peers_[rank]->register_queue(std::move(replica));
        registered_peer_ranks.push_back(rank);
      }
    } catch (...) {
      for (const uint32_t rank : registered_peer_ranks)
        xcd_peers_[rank]->unregister_queue(queue.queue_id, queue.process_id);
      throw;
    }
  }
  try {
    std::lock_guard<std::recursive_mutex> lock(hw_queue_mutex_);
    if (find_queue_state(queue.queue_id, queue.process_id) != nullptr) {
      throw std::runtime_error(
          std::format("duplicate queue registration on {}: queue_id={} process_id={} is already "
                      "registered, and fan-out routes dispatch shards by that key",
                      name(), queue.queue_id, queue.process_id));
    }
    HwQueueState qs{};
    qs.queue_desc_va = queue.queue_desc_va;
    qs.address_space = queue.address_space;
    qs.interrupt_sink = queue.interrupt_sink;
    qs.fanout_replica = queue.fanout_replica;
    hw_queues_.push_back(queue);
    try {
      new_queue_states_.push_back(std::move(qs));
      if (queue.is_sdma) {
        if (gpu_vm_ == nullptr)
          throw std::runtime_error("cannot register an SDMA queue without a GPU VM");
        SdmaExecutorCallbacks callbacks{
            .poll_register =
                [](uint32_t, uint32_t, uint32_t, uint32_t) {
                  // The legacy/interposer CP has no MMIO register surface. Its
                  // historical functional model treats register polls as already
                  // satisfied; PCI supplies a real register-backed callback.
                  return VmAccessOutcome::Complete;
                },
            .write_register = {},
            .deliver_interrupt =
                [interrupt_sink = queue.interrupt_sink,
                 process_id = queue.process_id](uint32_t event_id) {
                  interrupt_sink.deliver(process_id, event_id);
                  return VmAccessOutcome::Complete;
                },
            .maintain_caches =
                [this](SdmaCacheOperation operation) {
                  if (operation == SdmaCacheOperation::Invalidate)
                    invalidate_gpu_caches();
                  else
                    flush_gpu_caches();
                },
            .timestamp =
                [] {
                  const auto now = std::chrono::steady_clock::now().time_since_epoch();
                  return static_cast<uint64_t>(
                      std::chrono::duration_cast<std::chrono::nanoseconds>(now).count());
                },
        };
        sdma_queue_runtimes_.emplace(
            queue.registration_id,
            SdmaQueueRuntime{.runner = std::make_unique<SdmaQueueRunner>(
                                 *gpu_vm_,
                                 SdmaQueueRunnerConfig{
                                     .address_space = queue.address_space,
                                     .ring_base = queue.ring_base_va,
                                     .ring_bytes = queue.ring_size,
                                     .read_pointer_address = queue.read_ptr_va,
                                     .initial_cursor = queue.initial_read_pointer,
                                 },
                                 sdma_packet_dialect_, std::move(callbacks))});
      }
    } catch (...) {
      sdma_queue_runtimes_.erase(queue.registration_id);
      if (new_queue_states_.size() == hw_queues_.size())
        new_queue_states_.pop_back();
      hw_queues_.pop_back();
      throw;
    }
    // KFD queues rely on the VM-level primary (rj_vm.cpp); only internal test
    // queues (no host-accessible queue anywhere on this CP) need the CP to own the
    // primary lifecycle. Gate on the same aggregate predicate as the teardown
    // release (!has_kfd_queues()) — checked AFTER the push_back so it reflects the
    // new queue — so a CP can never register a primary it will never release.
    if (!is_primary_ && engine() && !has_kfd_queues()) {
      engine()->register_as_primary();
      is_primary_ = true;
    } else if (is_primary_ && engine() && has_kfd_queues()) {
      // A KFD queue joined a CP that had registered a test-owned primary; the
      // VM-level primary now anchors this CP's lifecycle, so drop the CP-owned
      // primary to keep register/release symmetric (the teardown path only
      // releases when !has_kfd_queues()).
      engine()->primary_release();
      is_primary_ = false;
    }
  } catch (...) {
    for (const uint32_t rank : registered_peer_ranks)
      xcd_peers_[rank]->unregister_queue(queue.queue_id, queue.process_id);
    throw;
  }
  // Start (or restart) the doorbell poll thread for KFD (host-accessible) queues
  // AFTER releasing hw_queue_mutex_. ensure_doorbell_monitor() serializes on its
  // own doorbell_thread_mutex_. Keeping that lock order consistent with the stop
  // path avoids joining a monitor while holding the queue mutex it needs to finish
  // a scan. Internal test queues inject doorbell events directly via
  // schedule_event_now() and need no monitor.
  if (start_poll)
    ensure_doorbell_monitor();
  return queue.registration_id;
}

void CommandProcessor::notify_queue_doorbell(uint64_t registration_id, uint64_t value) {
  if (registration_id == 0)
    return;
  {
    // Transport callbacks never wait behind queue execution.  The owner thread
    // validates the stable registration id when it drains this leaf inbox.
    std::lock_guard lock(doorbell_inbox_mutex_);
    doorbell_inbox_.push_back({.registration_id = registration_id, .value = value});
  }
  if (engine())
    engine()->schedule_event_now(&doorbell_event_);
}

void CommandProcessor::drain_doorbell_inbox() {
  std::vector<DoorbellNotification> notifications;
  {
    std::lock_guard lock(doorbell_inbox_mutex_);
    notifications.swap(doorbell_inbox_);
  }
  if (notifications.empty())
    return;

  std::lock_guard<std::recursive_mutex> lock(hw_queue_mutex_);
  for (const DoorbellNotification &notification : notifications) {
    const auto queue =
        std::ranges::find(hw_queues_, notification.registration_id, &HwQueue::registration_id);
    if (queue != hw_queues_.end())
      queue->last_doorbell = notification.value;
  }
}

bool CommandProcessor::signal_queue_exception(uint32_t queue_id, uint32_t process_id,
                                              uint64_t status) {
  uint64_t exception_status_va = 0;
  uint32_t exception_event_id = 0;
  InterruptSink interrupt_sink;
  {
    std::lock_guard<std::recursive_mutex> lk(hw_queue_mutex_);
    auto queue = std::find_if(hw_queues_.begin(), hw_queues_.end(), [&](const HwQueue &candidate) {
      return candidate.queue_id == queue_id && candidate.process_id == process_id;
    });
    if (queue == hw_queues_.end() || queue->exception_status_va == 0)
      return false;
    exception_status_va = queue->exception_status_va;
    exception_event_id = queue->exception_event_id;
    interrupt_sink = queue->interrupt_sink;
  }

  memory_->write64(exception_status_va, status, process_id);
  interrupt_sink.deliver(process_id, exception_event_id);
  const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(1);
  while (memory_->read64(exception_status_va, process_id) == status &&
         std::chrono::steady_clock::now() < deadline)
    std::this_thread::yield();

  for (auto *cu : cus_) {
    cu->with_wave_state_locked([&] {
      for (uint32_t slot = 0; slot < cu->num_wf_slots(); ++slot) {
        auto *wave = cu->wf(slot);
        if (!wave->is_halted() && wave->process_id() == process_id && wave->queue_id() == queue_id)
          wave->set_debug_suspended(true);
      }
    });
  }
  return true;
}

void CommandProcessor::unregister_queue(uint32_t queue_id, uint32_t process_id) {
  bool drop_replicas = false;
  {
    // Holds hw_queue_mutex_ across with_wave_state_locked(), which is the order
    // the dispatch path uses too (handle_doorbell -> dispatch_workgroups ->
    // dispatch_wf). Nothing takes them the other way any more: a wave reaching
    // s_endpgm under the wave-state lock queues its completion instead of sending
    // it, and WaveStateGuard delivers it after that lock is dropped. Keep it that
    // way -- a CU-side call back into the CP while the wave-state lock is held
    // would deadlock a DESTROY_QUEUE against the engine worker.
    std::lock_guard<std::recursive_mutex> lock(hw_queue_mutex_);
    for (auto *cu : cus_) {
      cu->with_wave_state_locked([&] {
        for (uint32_t slot = 0; slot < cu->num_wf_slots(); ++slot) {
          auto *wave = cu->wf(slot);
          if (!wave->is_halted() && wave->process_id() == process_id &&
              wave->queue_id() == queue_id)
            wave->halt();
        }
      });
    }
    for (size_t i = 0; i < hw_queues_.size(); ++i) {
      if (hw_queues_[i].queue_id == queue_id && hw_queues_[i].process_id == process_id) {
        drop_replicas = hw_queues_[i].xcd_fanout;
        sdma_queue_runtimes_.erase(hw_queues_[i].registration_id);
        for (const DispatchEntry &entry : new_queue_states_[i].entries)
          dispatch_launch_metadata_.erase(entry.dispatch_id);
        // Any shares still unpublished here are simply dropped. They cannot be
        // credited to the grid from this thread: publish_share is the release edge
        // that must follow this XCD's cache write-back, and flushing walks cus_,
        // which belong to the engine partition rather than to the caller. Crediting
        // without the flush would let the owner fire the completion signal with this
        // XCD's results still cached.
        //
        // Dropping them is safe because a fan-out queue is only ever destroyed on
        // every XCD at once: the KFD paths sweep all command processors, and an
        // owner cascades to its replicas below. No XCD is left holding a grid that
        // can no longer retire. Unregistering a lone replica is not supported.
        hw_queues_.erase(hw_queues_.begin() + static_cast<ptrdiff_t>(i));
        new_queue_states_.erase(new_queue_states_.begin() + static_cast<ptrdiff_t>(i));
        break;
      }
    }
  }
  // Tear the replicas down outside our own lock: a peer's unregister_queue takes
  // that peer's lock, and holding both would fix no order between two CPs whose
  // queues are being destroyed concurrently.
  if (drop_replicas) {
    const auto num_xcds = static_cast<uint32_t>(xcd_peers_.size());
    for (uint32_t rank = 0; rank < num_xcds; ++rank) {
      if (rank != xcd_rank_)
        xcd_peers_[rank]->unregister_queue(queue_id, process_id);
    }
  }

  // Reap the monitor when the last host queue is removed. This runs after
  // releasing hw_queue_mutex_: the poller needs that mutex to finish its current
  // scan. The poll loop may also be in the engine event-queue path or the
  // interrupt/event-state callback, but neither path enters a KFD ioctl or acquires
  // KfdProcess::op_mutex_, which the production callers hold here. Preserve that
  // invariant: no poll-loop callback may wait for a lock held by an
  // unregister_queue() caller. The synchronous join makes queue-destroy latency
  // include at most the current poll iteration and its bounded callbacks. The
  // helper rechecks the queue set while holding the lifecycle mutex, so a concurrent
  // registration either keeps this monitor alive or starts a new one after the join.
  stop_doorbell_monitor_if_idle();
}

void CommandProcessor::update_queue(uint32_t queue_id, uint32_t process_id, uint64_t ring_base_va,
                                    uint32_t ring_size, uint32_t queue_percentage) {
  const bool suspended = queue_percentage == 0;
  bool changed = false;
  bool wake_command_processor = false;
  bool update_replicas = false;
  {
    std::lock_guard<std::recursive_mutex> lock(hw_queue_mutex_);
    for (auto &q : hw_queues_) {
      if (q.queue_id == queue_id && q.process_id == process_id) {
        const bool ring_changed = q.ring_base_va != ring_base_va || q.ring_size != ring_size;
        q.ring_base_va = ring_base_va;
        q.ring_size = ring_size;
        if (ring_changed && q.is_sdma) {
          if (const auto runtime = sdma_queue_runtimes_.find(q.registration_id);
              runtime != sdma_queue_runtimes_.end()) {
            // Reprogramming the ring cannot discard a partially committed
            // packet and then allow it to be fetched again from an unchanged
            // external read pointer. Fail closed if teardown did not first
            // quiesce the queue; a clean idle rebind starts from the newly
            // programmed read pointer on its next service pass.
            SdmaQueueRunnerConfig config = runtime->second.runner->config();
            config.ring_base = ring_base_va;
            config.ring_bytes = ring_size;
            config.initial_cursor.reset();
            q.faulted = !runtime->second.runner->reconfigure(std::move(config));
          }
        }
        update_replicas = q.xcd_fanout;
        changed = q.runtime_suspended != suspended;
        q.runtime_suspended = suspended;
        // Only consume the deferral once *no* reason still gates the queue.
        // debug_work_deferred is shared by both suspend reasons, so clearing it
        // here while the debugger still holds the gate would leave the later
        // debugger resume with nothing to release, and the already-fetched
        // packets would sit until an unrelated doorbell arrived.
        if (changed && !suspended && !q.debug_suspended)
          wake_command_processor = std::exchange(q.debug_work_deferred, false);
        break;
      }
    }
  }
  if (update_replicas) {
    const uint32_t num_xcds = static_cast<uint32_t>(xcd_peers_.size());
    for (uint32_t rank = 0; rank < num_xcds; ++rank) {
      if (rank != xcd_rank_)
        xcd_peers_[rank]->update_queue(queue_id, process_id, ring_base_va, ring_size,
                                       queue_percentage);
    }
  }
  if (!changed)
    return;
  for (auto *cu : cus_) {
    cu->with_wave_state_locked([&] {
      for (uint32_t slot = 0; slot < cu->num_wf_slots(); ++slot) {
        auto *wave = cu->wf(slot);
        if (!wave->is_halted() && wave->process_id() == process_id && wave->queue_id() == queue_id)
          // The runtime's own pause reason. Writing the debugger's bit here let
          // a runtime resume clear a debugger pause, and a debugger or CWSR
          // resume clear an active runtime pause.
          wave->set_runtime_suspended(suspended);
      }
    });
    if (!suspended)
      cu->schedule_work_async();
  }
  // Runtime resume has to release deferred queue work the same way a debugger
  // resume does, or already-fetched work sits until the next doorbell.
  if (wake_command_processor && engine())
    engine()->schedule_event_now(&doorbell_event_);
}

void CommandProcessor::set_queue_debug_suspended(uint32_t queue_id, uint32_t process_id,
                                                 bool suspended) {
  bool wake_command_processor = false;
  {
    std::lock_guard<std::recursive_mutex> lock(hw_queue_mutex_);
    for (size_t index = 0; index < hw_queues_.size(); ++index) {
      auto &q = hw_queues_[index];
      if (q.queue_id == queue_id && q.process_id == process_id) {
        if (q.debug_suspended == suspended)
          continue;
        q.debug_suspended = suspended;
        if (suspended) {
          // Existing queue work needs a resume pass only when the gate, rather
          // than an earlier incomplete dispatch, is what prevents it from
          // running. Resident waves are reactivated directly by KFD resume.
          auto &state = new_queue_states_[index];
          // Accumulate: the flag is shared with the runtime's suspend reason,
          // and fetch_from_queue() may already have recorded a deferral for a
          // queue the runtime had gated. Assigning would discard it, leaving
          // neither resume path with anything to release.
          if (state.next_dispatch_idx < state.entries.size()) {
            const auto &entry = state.entries[state.next_dispatch_idx];
            const bool barrier_ready =
                !entry.barrier_bit || barrier_satisfied(state, state.next_dispatch_idx);
            q.debug_work_deferred |=
                barrier_ready && (entry.is_non_kernel() || !entry.fully_dispatched());
          }
        } else if (!q.runtime_suspended) {
          wake_command_processor |= std::exchange(q.debug_work_deferred, false);
        }
      }
    }
  }
  if (wake_command_processor && engine())
    engine()->schedule_event_now(&doorbell_event_);
}

void CommandProcessor::set_doorbell_base(uint32_t process_id, void *base) {
  std::lock_guard<std::recursive_mutex> lock(hw_queue_mutex_);
  for (auto &q : hw_queues_) {
    if (q.process_id == process_id)
      q.doorbell_base = base;
  }
}

void CommandProcessor::ensure_doorbell_monitor() {
  std::lock_guard<std::mutex> lock(doorbell_thread_mutex_);
  // A monitor is already servicing this CP's queues — nothing to do. (The
  // register→ring window is fine: the live monitor scans every registered queue
  // each pass, so it will pick up the queue this call just added.)
  if (doorbell_running_)
    return;
  util::Logger::cp([&](auto &os) { os << std::format("{}: STARTING doorbell thread", name()); });
  // Construct the thread BEFORE setting doorbell_running_: if the jthread
  // constructor throws (std::system_error on thread-creation failure) the flag
  // must stay false so a later ensure_doorbell_monitor() retries instead of
  // no-oping forever. We still hold doorbell_thread_mutex_, so teardown cannot
  // observe the new handle until both it and the running flag are published.
  assert(!doorbell_thread_.joinable());
  doorbell_thread_ = std::jthread([this](std::stop_token stop) { doorbell_poll_loop(stop); });
  doorbell_running_ = true;
}

void CommandProcessor::stop_doorbell_monitor() {
  std::lock_guard<std::mutex> lock(doorbell_thread_mutex_);
  if (doorbell_thread_.joinable()) {
    doorbell_thread_.request_stop();
    doorbell_thread_.join();
  }
  doorbell_running_ = false;
}

void CommandProcessor::stop_doorbell_monitor_if_idle() {
  std::lock_guard<std::mutex> thread_lock(doorbell_thread_mutex_);
  {
    std::lock_guard<std::recursive_mutex> queue_lock(hw_queue_mutex_);
    // polls_kfd_queues(), not has_kfd_queues(): a fan-out replica is
    // host-accessible but is never polled, so keying this on presence would
    // strand a monitor on a CP whose own queue was destroyed while a replica of
    // some other queue happened to remain.
    if (polls_kfd_queues())
      return;
  }
  if (doorbell_thread_.joinable()) {
    doorbell_thread_.request_stop();
    doorbell_thread_.join();
  }
  doorbell_running_ = false;
}

AtomicLoadResult CommandProcessor::read_gpu_u64(AddressSpaceHandle address_space, uint64_t va,
                                                uint32_t vmid) const {
  uint64_t val = 0;
  if (address_space) {
    if (gpu_vm_ == nullptr)
      return {.outcome = VmAccessOutcome::Faulted, .value = 0};
    if ((va & (alignof(uint64_t) - 1)) != 0) {
      const VmAccessOutcome outcome = read_gpu_block(address_space, va, &val, sizeof(val), vmid);
      return {.outcome = outcome, .value = val};
    }
    const AtomicLoadResult atomic = gpu_vm_->atomic_load(address_space, va, sizeof(val));
    return atomic;
  }
  if (memory_ == nullptr)
    return {.outcome = VmAccessOutcome::Unavailable, .value = 0};
  // Ring pointers and signal values are 64-bit locations the host publishes with a
  // single atomic store, so read them with a single atomic load where the mapping
  // allows it. The byte walk below can observe a half-updated value, and it also
  // leaves this CP with no ordering edge to the writer -- which is why the packets a
  // new write index publishes were being read unsynchronized as well.
  if (memory_->try_read_u64_atomic(va, &val, vmid))
    return {.outcome = VmAccessOutcome::Complete, .value = val};
  // Fall back for a split, unaligned or page-crossing range: those cannot be read
  // atomically, and the byte walk resolves each byte's mapping independently.
  auto *dst = reinterpret_cast<uint8_t *>(&val);
  for (uint32_t i = 0; i < sizeof(val); ++i)
    dst[i] = memory_->read8(va + i, vmid);
  return {.outcome = VmAccessOutcome::Complete, .value = val};
}

AtomicLoadResult CommandProcessor::read_gpu_u32(AddressSpaceHandle address_space, uint64_t va,
                                                uint32_t vmid) const {
  uint32_t val = 0;
  if (address_space) {
    if (gpu_vm_ == nullptr)
      return {.outcome = VmAccessOutcome::Faulted, .value = 0};
    const VmAccessOutcome outcome = read_gpu_block(address_space, va, &val, sizeof(val), vmid);
    return {.outcome = outcome, .value = val};
  }
  if (memory_ == nullptr)
    return {.outcome = VmAccessOutcome::Unavailable, .value = 0};
  auto *dst = reinterpret_cast<uint8_t *>(&val);
  for (uint32_t i = 0; i < sizeof(val); ++i)
    dst[i] = memory_->read8(va + i, vmid);
  return {.outcome = VmAccessOutcome::Complete, .value = val};
}

VmAccessOutcome CommandProcessor::read_gpu_block(AddressSpaceHandle address_space, uint64_t va,
                                                 void *dst, size_t size, uint32_t vmid) const {
  if (address_space) {
    if (gpu_vm_ == nullptr)
      return VmAccessOutcome::Faulted;
    return gpu_vm_->read(address_space, va,
                         std::span<std::byte>(static_cast<std::byte *>(dst), size));
  }
  if (memory_ == nullptr)
    return VmAccessOutcome::Unavailable;
  auto *p = static_cast<uint8_t *>(dst);
  for (size_t i = 0; i < size; ++i)
    p[i] = memory_->read8(va + i, vmid);
  return VmAccessOutcome::Complete;
}

VmAccessOutcome CommandProcessor::write_gpu_block(AddressSpaceHandle address_space, uint64_t va,
                                                  const void *src, size_t size, uint32_t vmid) {
  if (address_space) {
    if (gpu_vm_ == nullptr)
      return VmAccessOutcome::Faulted;
    return gpu_vm_->write(address_space, va,
                          std::span<const std::byte>(static_cast<const std::byte *>(src), size));
  }
  if (memory_ == nullptr)
    return VmAccessOutcome::Unavailable;
  return memory_->write_block(va, std::span<const uint8_t>(static_cast<const uint8_t *>(src), size),
                              vmid) == AccessOutcome::Complete
             ? VmAccessOutcome::Complete
             : VmAccessOutcome::Faulted;
}

/// @brief Scan all HW queues for doorbell changes; return true if any changed.
/// Caller must NOT hold hw_queue_mutex_.
bool CommandProcessor::scan_doorbells() {
  bool found = false;
  std::lock_guard<std::recursive_mutex> lock(hw_queue_mutex_);
  for (auto &q : hw_queues_) {
    // A replica shares the owner's ring and doorbell. Only the owning XCD may
    // consume them, or every XCD would dispatch the whole grid.
    if (q.fanout_replica)
      continue;
    uint64_t val;
    if (q.host_accessible) {
      if (!q.doorbell_base)
        continue;
      val = std::atomic_ref<uint64_t>(*reinterpret_cast<uint64_t *>(
                                          static_cast<char *>(q.doorbell_base) + q.doorbell_offset))
                .load(std::memory_order_acquire);
    } else {
      if (q.doorbell_va == 0)
        continue;
      const AtomicLoadResult loaded = read_gpu_u64(q.address_space, q.doorbell_va, q.process_id);
      if (loaded.outcome == VmAccessOutcome::Unavailable)
        continue;
      if (loaded.outcome != VmAccessOutcome::Complete) {
        q.faulted = true;
        continue;
      }
      val = loaded.value;
    }
    if (val != q.last_doorbell) {
      util::Logger::cp([&](auto &os) {
        os << std::format("{}: DOORBELL_CHANGE pid={} qid={} sdma={} old={:#x} new={:#x} "
                          "db_base={} db_off={}",
                          name(), q.process_id, q.queue_id, q.is_sdma, q.last_doorbell, val,
                          reinterpret_cast<uintptr_t>(q.doorbell_base), q.doorbell_offset);
      });
      q.last_doorbell = val;
      found = true;
    }
  }
  return found;
}

void CommandProcessor::doorbell_poll_loop(std::stop_token stop) {
  using namespace std::chrono_literals;
  uint64_t poll_count = 0;
  // INVARIANT: this loop must re-read invalid_pending_/stall_pending_ (below) on
  // EVERY iteration, unconditionally. Those flags are level-triggered — the engine
  // clears them at handle_doorbell entry and re-sets them if a stall is still
  // unsatisfied — so this unconditional 100us heartbeat re-check is what guarantees a
  // pending stall is eventually retried. A future change that lets the loop skip the
  // flag re-read on some iterations (an early continue before the retry check) would
  // reintroduce a lost-wakeup.
  while (!stop.stop_requested()) {
    bool doorbell_changed = scan_doorbells();
    // Retry on a pending INVALID packet (the runtime has not finished writing it
    // yet) OR a pending barrier/dependency stall (waiting on a signal a peer rank
    // or another queue will write) even when no doorbell value changed. Pace those
    // retries with the same 100us idle wait rather than spinning: a real doorbell
    // change fires the event immediately (latency-sensitive), but a pending retry
    // only needs to poll until the awaited state changes, so it must not burn a
    // core. Rescheduling these on the main event queue (now+1) instead would spin
    // simulated time millions of ticks per collective while wall-clock RPC latency
    // elapses — the RCCL slowdown this replaces.
    bool retry = !doorbell_changed && (invalid_pending_.load(std::memory_order_acquire) ||
                                       stall_pending_.load(std::memory_order_acquire));
    if (doorbell_changed)
      engine()->schedule_event_now(&doorbell_event_);
    else if (retry) {
      std::this_thread::sleep_for(100us);
      engine()->schedule_event_now(&doorbell_event_);
    } else
      std::this_thread::sleep_for(100us);
    ++poll_count;

    // HQD idle monitoring: periodically fire HQD_IDLE for queues that are
    // currently empty. On real hardware the CP continuously monitors queue
    // activity and fires the idle interrupt whenever the queue is inactive.
    // Our drain_completions fires on the non-empty→empty transition, but a
    // process may create a new event AFTER that transition and miss the
    // signal. Re-broadcasting every ~10ms ensures late-created events see
    // the idle state within a bounded window.
    if (poll_count % 100 == 0) {
      // Snapshot the idle queues' process ids under the lock, then deliver interrupts
      // OUTSIDE it. Subscribers are external frontend callbacks whose internal
      // locking is opaque to the CP; invoking it while holding hw_queue_mutex_ risks a
      // lock-order inversion if that callback ever takes a lock held elsewhere while
      // acquiring hw_queue_mutex_.
      std::vector<std::pair<InterruptSink, uint32_t>> idle_queues;
      {
        std::lock_guard<std::recursive_mutex> lock(hw_queue_mutex_);
        for (size_t i = 0; i < hw_queues_.size(); ++i) {
          // A replica does not own the queue, so it must not report it idle: its
          // shards drain before the owner's and the same KFD queue would otherwise
          // raise this from several CPs at once.
          if (hw_queues_[i].fanout_replica)
            continue;
          if (new_queue_states_[i].entries.empty() && hw_queues_[i].process_id != 0)
            idle_queues.emplace_back(hw_queues_[i].interrupt_sink, hw_queues_[i].process_id);
        }
      }
      for (const auto &[interrupt_sink, process_id] : idle_queues)
        interrupt_sink.deliver(process_id, 0);
    }

    if (poll_count % 5000 == 1) {
      std::lock_guard<std::recursive_mutex> lock(hw_queue_mutex_);
      for (auto &q : hw_queues_) {
        uint64_t current = q.last_doorbell;
        if (q.host_accessible && q.doorbell_base) {
          current = std::atomic_ref<uint64_t>(
                        *reinterpret_cast<uint64_t *>(static_cast<char *>(q.doorbell_base) +
                                                      q.doorbell_offset))
                        .load(std::memory_order_acquire);
        } else if (!q.host_accessible && q.doorbell_va != 0) {
          const AtomicLoadResult loaded =
              read_gpu_u64(q.address_space, q.doorbell_va, q.process_id);
          if (loaded.outcome == VmAccessOutcome::Complete)
            current = loaded.value;
          else if (loaded.outcome != VmAccessOutcome::Unavailable)
            q.faulted = true;
        }
        util::Logger::cp([&](auto &os) {
          os << std::format("{}: DOORBELL_POLL pid={} qid={} current={:#x} last={:#x} "
                            "monitor_base={} db_off={} polls={}",
                            name(), q.process_id, q.queue_id, current, q.last_doorbell,
                            reinterpret_cast<uintptr_t>(q.doorbell_base), q.doorbell_offset,
                            poll_count);
        });
      }
    }
  }
}

HwQueueState *CommandProcessor::schedule_next_queue() {
  if (new_queue_states_.empty())
    return nullptr;
  size_t start = next_queue_idx_;
  for (size_t i = 0; i < new_queue_states_.size(); ++i) {
    size_t idx = (start + i) % new_queue_states_.size();
    auto &qs = new_queue_states_[idx];
    if (hw_queues_[idx].faulted || hw_queues_[idx].is_sdma || hw_queues_[idx].debug_suspended ||
        hw_queues_[idx].runtime_suspended)
      continue;
    if (qs.next_dispatch_idx < qs.entries.size()) {
      next_queue_idx_ = (idx + 1) % new_queue_states_.size();
      return &qs;
    }
  }
  return nullptr;
}

bool CommandProcessor::barrier_satisfied(const HwQueueState &qs, size_t idx) const {
  if (idx == 0 && !qs.implicit_barrier_next)
    return true;

  // Barrier bit: all prior entries must be fully completed, device-wide. A prior
  // entry that is one XCD's share of a fanned-out dispatch is not done just
  // because this XCD finished it, so gate on the whole grid or this XCD would run
  // the next packet while a peer is still executing the previous one.
  for (size_t i = 0; i < idx; ++i) {
    if (!qs.entries[i].grid_fully_completed())
      return false;
  }
  return true;
}

void CommandProcessor::register_cluster_workgroup(const DispatchEntry &entry, uint32_t local_wg_id,
                                                  uint32_t global_wg_id, ComputeUnitCore *cu,
                                                  uint32_t lds_base) {
  if (!entry.has_workgroup_clusters())
    return;
  std::lock_guard<std::recursive_mutex> lock(cluster_placements_mutex_);
  uint32_t cluster_base_wg_id =
      entry.cluster_base_local_wg_id(local_wg_id) + entry.workgroup_id_offset;
  uint64_t cluster_key = wg_key(entry.dispatch_id, cluster_base_wg_id);
  cu->pin_lds_until_cluster_retired(cluster_key);
  ClusterWorkgroupPlacement placement{};
  placement.cu = cu;
  placement.lds_base = lds_base;
  placement.cluster_key = cluster_key;
  placement.cluster_rank = entry.cluster_rank_for_flat_wg_id(global_wg_id);
  placement.cluster_size = entry.cluster_size();
  placement.peer_wg_ids.reserve(placement.cluster_size);
  for (uint32_t rank = 0; rank < placement.cluster_size; ++rank) {
    uint32_t peer_local_wg_id = entry.cluster_peer_local_wg_id(local_wg_id, rank);
    placement.peer_wg_ids.push_back(peer_local_wg_id + entry.workgroup_id_offset);
  }
  cluster_wg_placements_[wg_key(entry.dispatch_id, global_wg_id)] = std::move(placement);
  auto &barriers = cluster_barriers_[cluster_key];
  if (barriers.expected_member_count == 0) {
    barriers.expected_member_count = entry.cluster_size();
    barriers.member_count = entry.cluster_size();
  }
  barriers.registered_workgroups.insert(global_wg_id);
}

bool CommandProcessor::find_valid_cluster_barrier_locked(const Wavefront &wf, int32_t barrier_id,
                                                         ClusterWorkgroupPlacement *&placement,
                                                         ClusterBarrierState *&barriers) {
  if (barrier_id != kClusterBarrierId && barrier_id != kClusterTrapBarrierId)
    return false;
  auto placement_it = cluster_wg_placements_.find(wg_key(wf.dispatch_id(), wf.wg_id()));
  if (placement_it == cluster_wg_placements_.end())
    return false;
  auto barriers_it = cluster_barriers_.find(placement_it->second.cluster_key);
  if (barriers_it == cluster_barriers_.end() || barriers_it->second.member_count == 0 ||
      barriers_it->second.registered_workgroups.size() != barriers_it->second.expected_member_count)
    return false;
  placement = &placement_it->second;
  barriers = &barriers_it->second;
  return true;
}

bool CommandProcessor::find_valid_cluster_barrier_locked(
    const Wavefront &wf, int32_t barrier_id, const ClusterWorkgroupPlacement *&placement,
    const ClusterBarrierState *&barriers) const {
  if (barrier_id != kClusterBarrierId && barrier_id != kClusterTrapBarrierId)
    return false;
  auto placement_it = cluster_wg_placements_.find(wg_key(wf.dispatch_id(), wf.wg_id()));
  if (placement_it == cluster_wg_placements_.end())
    return false;
  auto barriers_it = cluster_barriers_.find(placement_it->second.cluster_key);
  if (barriers_it == cluster_barriers_.end() || barriers_it->second.member_count == 0 ||
      barriers_it->second.registered_workgroups.size() != barriers_it->second.expected_member_count)
    return false;
  placement = &placement_it->second;
  barriers = &barriers_it->second;
  return true;
}

bool CommandProcessor::cluster_barrier_valid(const Wavefront &wf, int32_t barrier_id) const {
  std::lock_guard<std::recursive_mutex> lock(cluster_placements_mutex_);
  const ClusterWorkgroupPlacement *placement = nullptr;
  const ClusterBarrierState *barriers = nullptr;
  return find_valid_cluster_barrier_locked(wf, barrier_id, placement, barriers);
}

uint32_t CommandProcessor::cluster_barrier_state(const Wavefront &wf, int32_t barrier_id,
                                                 uint32_t allocation_blocks) const {
  std::lock_guard<std::recursive_mutex> lock(cluster_placements_mutex_);
  const ClusterWorkgroupPlacement *placement = nullptr;
  const ClusterBarrierState *barriers = nullptr;
  if (!find_valid_cluster_barrier_locked(wf, barrier_id, placement, barriers))
    return 0;
  const uint32_t index = static_cast<uint32_t>(-barrier_id - kClusterBarrierBit);
  return 1u | ((barriers->member_count & 0x7fu) << 4) |
         ((barriers->signaled_workgroups[index].size() & 0x7fu) << 16) |
         ((allocation_blocks & 0x7u) << 24);
}

bool CommandProcessor::cluster_barrier_signal(Wavefront &wf, int32_t barrier_id) {
  bool is_first = false;
  std::vector<std::pair<ComputeUnitCore *, uint32_t>> peers;
  const uint8_t completion_bit = static_cast<uint8_t>(-barrier_id);
  {
    std::lock_guard<std::recursive_mutex> lock(cluster_placements_mutex_);
    ClusterWorkgroupPlacement *placement = nullptr;
    ClusterBarrierState *barriers = nullptr;
    if (!find_valid_cluster_barrier_locked(wf, barrier_id, placement, barriers))
      return false;

    const uint32_t index = static_cast<uint32_t>(completion_bit - kClusterBarrierBit);
    auto [_, inserted] = barriers->signaled_workgroups[index].insert(wf.wg_id());
    if (!inserted)
      return false;
    is_first = barriers->signaled_workgroups[index].size() == 1;
    if (barriers->signaled_workgroups[index].size() < barriers->member_count)
      return is_first;

    barriers->signaled_workgroups[index].clear();
    peers.reserve(placement->peer_wg_ids.size());
    for (uint32_t peer_wg_id : placement->peer_wg_ids) {
      auto peer = cluster_wg_placements_.find(wg_key(wf.dispatch_id(), peer_wg_id));
      if (peer != cluster_wg_placements_.end() && peer->second.cu)
        peers.emplace_back(peer->second.cu, peer_wg_id);
    }
  }

  std::vector<Wavefront *> members;
  for (auto [cu, peer_wg_id] : peers) {
    auto peer_members = cu->complete_barrier(wf.dispatch_id(), peer_wg_id, completion_bit);
    members.insert(members.end(), peer_members.begin(), peer_members.end());
  }
  if (!members.empty())
    plugin_group_->onAmdgpuBarrierResolved(std::span<Wavefront *>(members));
  return is_first;
}

void CommandProcessor::mark_cluster_workgroup_complete(uint32_t dispatch_id, uint32_t wg_id) {
  // Cluster barriers resolve here, but complete_barrier() and the LDS reclaim
  // both reach into a CU. Collect them under the lock and act after it is
  // dropped -- see cluster_placements_mutex_.
  std::array<std::vector<std::pair<ComputeUnitCore *, uint32_t>>, 2> resolved_peers;
  std::vector<std::pair<ComputeUnitCore *, uint64_t>> unpin;
  {
    std::lock_guard<std::recursive_mutex> lock(cluster_placements_mutex_);
    auto it = cluster_wg_placements_.find(wg_key(dispatch_id, wg_id));
    if (it == cluster_wg_placements_.end() || it->second.completed)
      return;

    it->second.completed = true;
    const uint64_t cluster_key = it->second.cluster_key;
    const auto peer_wg_ids = it->second.peer_wg_ids;
    auto barrier_it = cluster_barriers_.find(cluster_key);
    if (barrier_it != cluster_barriers_.end() && barrier_it->second.member_count != 0) {
      auto &barriers = barrier_it->second;
      --barriers.member_count;
      for (uint32_t index = 0; index < barriers.signaled_workgroups.size(); ++index) {
        barriers.signaled_workgroups[index].erase(wg_id);
        if (barriers.member_count == 0 ||
            barriers.signaled_workgroups[index].size() < barriers.member_count)
          continue;
        barriers.signaled_workgroups[index].clear();
        for (uint32_t peer_wg_id : peer_wg_ids) {
          auto peer = cluster_wg_placements_.find(wg_key(dispatch_id, peer_wg_id));
          if (peer != cluster_wg_placements_.end() && !peer->second.completed && peer->second.cu)
            resolved_peers[index].emplace_back(peer->second.cu, peer_wg_id);
        }
      }
    }

    const bool all_completed = std::ranges::all_of(peer_wg_ids, [&](uint32_t peer_wg_id) {
      auto peer = cluster_wg_placements_.find(wg_key(dispatch_id, peer_wg_id));
      return peer != cluster_wg_placements_.end() && peer->second.completed;
    });
    if (all_completed) {
      for (uint32_t peer_wg_id : peer_wg_ids) {
        auto peer = cluster_wg_placements_.find(wg_key(dispatch_id, peer_wg_id));
        if (peer != cluster_wg_placements_.end() && peer->second.cu)
          unpin.emplace_back(peer->second.cu, cluster_key);
        cluster_wg_placements_.erase(wg_key(dispatch_id, peer_wg_id));
      }
      cluster_barriers_.erase(cluster_key);
    }
  }

  for (uint32_t index = 0; index < resolved_peers.size(); ++index) {
    std::vector<Wavefront *> members;
    const uint8_t completion_bit = static_cast<uint8_t>(kClusterBarrierBit + index);
    for (auto [cu, peer_wg_id] : resolved_peers[index]) {
      auto peer_members = cu->complete_barrier(dispatch_id, peer_wg_id, completion_bit);
      members.insert(members.end(), peer_members.begin(), peer_members.end());
    }
    if (!members.empty())
      plugin_group_->onAmdgpuBarrierResolved(std::span<Wavefront *>(members));
  }

  release_cluster_lds_pins(unpin);
}

// The waves halted (and freed) before their pin was released, so reclaim each
// peer CU's LDS once the whole cluster is done. Runs with
// cluster_placements_mutex_ released: maybe_reset_lds_alloc() reaches the CU's
// wave-state lock, which is ordered ahead of it.
void CommandProcessor::release_cluster_lds_pins(
    const std::vector<std::pair<ComputeUnitCore *, uint64_t>> &unpin) {
  for (const auto &[cu, cluster_key] : unpin) {
    cu->unpin_lds_for_cluster(cluster_key);
    cu->maybe_reset_lds_alloc();
  }
}

void CommandProcessor::erase_cluster_workgroup(uint32_t dispatch_id, uint32_t wg_id) {
  // maybe_reset_lds_alloc() takes the CU's wave-state lock, so it runs after the
  // placements lock is dropped -- see cluster_placements_mutex_.
  std::vector<std::pair<ComputeUnitCore *, uint64_t>> unpin;
  {
    std::lock_guard<std::recursive_mutex> lock(cluster_placements_mutex_);
    auto it = cluster_wg_placements_.find(wg_key(dispatch_id, wg_id));
    if (it == cluster_wg_placements_.end())
      return;
    if (it->second.cu)
      unpin.emplace_back(it->second.cu, it->second.cluster_key);
    cluster_barriers_.erase(it->second.cluster_key);
    cluster_wg_placements_.erase(it);
  }
  release_cluster_lds_pins(unpin);
}

void CommandProcessor::erase_cluster_workgroups(uint32_t dispatch_id) {
  std::vector<std::pair<ComputeUnitCore *, uint64_t>> unpin;
  {
    std::lock_guard<std::recursive_mutex> lock(cluster_placements_mutex_);
    for (auto it = cluster_wg_placements_.begin(); it != cluster_wg_placements_.end();) {
      if ((it->first >> 32) == dispatch_id) {
        cluster_barriers_.erase(it->second.cluster_key);
        if (it->second.cu)
          unpin.emplace_back(it->second.cu, it->second.cluster_key);
        it = cluster_wg_placements_.erase(it);
      } else {
        ++it;
      }
    }
  }
  release_cluster_lds_pins(unpin);
}

std::vector<ClusterLdsTarget>
CommandProcessor::cluster_lds_targets(uint32_t dispatch_id, uint32_t wg_id, uint32_t mcast_mask) {
  std::lock_guard<std::recursive_mutex> lock(cluster_placements_mutex_);
  std::vector<ClusterLdsTarget> targets;
  auto src_it = cluster_wg_placements_.find(wg_key(dispatch_id, wg_id));
  if (src_it == cluster_wg_placements_.end())
    return targets;

  const auto &src = src_it->second;
  const uint32_t self_mask = cluster_multicast_rank_mask(src.cluster_rank);
  // Defensive for direct helper callers; the issue path handles mask 0 locally.
  if (mcast_mask == 0 || (src.cluster_size <= 1 && (mcast_mask & self_mask) != 0)) {
    targets.push_back({src.cu, wg_id, src.lds_base, src.cluster_rank});
    return targets;
  }
  if (src.cluster_size <= 1)
    return targets;

  for (uint32_t rank = 0; rank < src.cluster_size && rank < kClusterMulticastMaskBits; ++rank) {
    if ((mcast_mask & (1u << rank)) == 0)
      continue;
    uint32_t peer_wg_id = src.peer_wg_ids[rank];
    auto peer_it = cluster_wg_placements_.find(wg_key(dispatch_id, peer_wg_id));
    if (peer_it == cluster_wg_placements_.end()) {
      throw std::runtime_error(std::format(
          "cluster multicast target is not resident: dispatch={} source_wg={} peer_wg={} rank={}",
          dispatch_id, wg_id, peer_wg_id, rank));
    }
    const auto &peer = peer_it->second;
    targets.push_back({peer.cu, peer_wg_id, peer.lds_base, peer.cluster_rank});
  }

  if (targets.empty() && (mcast_mask & self_mask) != 0)
    targets.push_back({src.cu, wg_id, src.lds_base, src.cluster_rank});
  return targets;
}

uint32_t CommandProcessor::dispatch_workgroups(DispatchEntry &entry) {
  assert(!cus_.empty() && "command processor has no compute units");

  // A peer can publish the shared terminal-fault latch before this CP drains
  // its fault inbox. Stop placement as soon as that publication is visible;
  // the inbox drain will remove the entry and abort any waves already resident.
  if (entry.grid_faulted())
    return 0;

  // All waves in one workgroup currently land on one physical CU so the
  // existing barrier implementation remains local. WGP mode additionally
  // reserves that CU's sibling and binds the waves to their shared LDS pool.
  // Query the complete placement before dispatching for all-or-nothing setup.
  uint32_t dispatched = 0;
  // Places one workgroup's waves on the chosen CU. Returns false only on an
  // internal invariant violation: dispatch_wf() returning null after placement was
  // gated on can_accept_workgroup(). The caller turns that into a hard error rather
  // than silently dereferencing a null wave in a release build (where the callee's
  // assert is compiled out).
  auto dispatch_to_placement =
      [&](uint32_t local_wg_id, uint32_t global_wg_id,
          const ShaderProcessorInput::WorkgroupPlacement &placement) -> bool {
    // Fire the dispatch-execution-begin hook exactly once, on the first workgroup
    // actually placed on a CU, guarded by the per-dispatch flag. One begin per
    // dispatch, not per XCD. This cannot be pinned to the XCD that read the
    // packet: when the grid is smaller than the XCD count that XCD's share may be
    // empty, so it never places anything. Let whichever XCD places the grid's
    // first workgroup claim the report.
    if (!entry.execution_begun) {
      entry.execution_begun = true;
      if (!entry.grid_completion || entry.grid_completion->claim_execution_begin())
        plugin_group_->onAmdgpuDispatchExecutionBegin(entry.dispatch_id);
    }
    ComputeUnitCore *cu = placement.cu;
    uint32_t lds_base = placement.lds_base;
    // Reserve AND fully initialize all waves BEFORE committing WG-completion
    // bookkeeping. begin_workgroup() installs the WG refcount and
    // register_cluster_workgroup() installs the LDS pin; both are released only via
    // release_wf() when the waves halt. Committing them first and then failing
    // mid-workgroup would orphan the refcount and pin, permanently blocking
    // maybe_reset_lds_alloc() on this CU. Two failure modes are covered by doing all
    // fallible work up front: a dispatch_wf() null (placement gating makes this
    // unreachable, but the assert is compiled out in release), and a throw from
    // init_wavefront_regs() (malformed kernarg-preload / undersized TTMP SGPR block).
    // On either, release the reserved-but-uncommitted waves. Use
    // free_wavefront_resources() rather than halt(): these waves never executed, so
    // firing halt()'s onAmdgpuWavefrontHalted hook would feed observers a spurious
    // "completed" wave. free_wavefront_resources() frees the SGPR/VGPR blocks and
    // resets the slot without the hook or a CP completion notify — and since
    // begin_workgroup() has not run, there is no active_wgs_ entry / cluster pin to
    // unwind either. Also reclaim the placement's LDS/WGP reservation symmetrically
    // with how it was reserved:
    //   - CU / cluster mode: the SPI (or the direct CU path) advanced the CU's
    //     next_lds_alloc_ via allocate_lds(); maybe_reset_lds_alloc() rolls it back
    //     iff the CU is now idle (freeing these waves left no active waves) and
    //     unpinned. It correctly no-ops when peers of the same dispatch are resident.
    //   - WGP mode: allocate_workgroup() reserved SPI-side state (wgp.next_lds_alloc,
    //     wgp.active_workgroups, resident_wgp_workgroups_) that is NOT CU-local, so
    //     maybe_reset_lds_alloc() cannot reach it; release_wgp_workgroup() is the
    //     matching release (the same call notify_wg_complete uses on the normal path).
    // Without the WGP release a failed WGP dispatch would permanently pin that WGP.
    std::vector<Wavefront *> wg_wavefronts;
    wg_wavefronts.reserve(entry.wfs_per_workgroup);
    const auto free_reserved = [&]() {
      for (auto *claimed : wg_wavefronts)
        cu->free_wavefront_resources(*claimed);
      if (entry.wgp_mode) {
        for (auto *spi : spis_)
          if (spi->release_wgp_workgroup(entry.dispatch_id, global_wg_id))
            break;
      }
      cu->maybe_reset_lds_alloc();
    };
    for (uint32_t w = 0; w < entry.wfs_per_workgroup; ++w) {
      Wavefront *wf = cu->dispatch_wf(global_wg_id, entry.kernel_entry_pc, entry.sgprs_per_wf,
                                      entry.vgprs_per_wf, entry.kernel_wave_size);
      if (!wf) {
        assert(false && "dispatch_wf failed after placement was reserved");
        free_reserved();
        return false;
      }
      wg_wavefronts.push_back(wf);
    }
    for (uint32_t w = 0; w < entry.wfs_per_workgroup; ++w) {
      Wavefront *wf = wg_wavefronts[w];
      wf->set_lds_base(lds_base);
      wf->set_lds_size(aligned_lds_bytes_per_workgroup(entry));
      wf->set_lds(placement.lds);
      wf->set_dispatch_id(entry.dispatch_id);
      wf->set_aql_packet_id(entry.aql_packet_id);
      wf->set_code_load_bias(entry.code_load_bias);
      wf->set_wave_in_group(w);
      wf->set_address_space(entry.address_space);
      wf->set_process_id(entry.process_id);
      wf->set_mode_raw(entry.initial_mode_raw);
      wf->set_queue_id(entry.queue_id);
      wf->set_exec(initial_exec_mask_for_wave(entry, global_wg_id, w, wf->wf_size()));
      const uint32_t relative_wg_id = global_wg_id - entry.workgroup_id_offset;
      const WorkgroupCoord coord = entry.local_wg_coord(relative_wg_id);
      wf->set_wg_coord(coord.x, coord.y, coord.z);
      wf->set_cluster_info(entry.cluster_rank_for_flat_wg_id(global_wg_id), entry.cluster_size());
      try {
        init_wavefront_regs(cu, wf, entry, global_wg_id, w);
      } catch (...) {
        free_reserved();
        throw;
      }
    }

    // All fallible per-wave work succeeded: commit WG bookkeeping and the cluster pin.
    cu->begin_workgroup(entry.dispatch_id, global_wg_id, entry.wfs_per_workgroup,
                        entry.num_named_barriers);
    register_cluster_workgroup(entry, local_wg_id, global_wg_id, cu, lds_base);

    plugin_group_->onAmdgpuWorkgroupDispatched(
        entry.dispatch_id, global_wg_id, cu->vgpr_allocation_block_size(),
        cu->sgpr_allocation_block_size(), std::span<Wavefront *>(wg_wavefronts));
    for (auto *wf : wg_wavefronts)
      plugin_group_->onAmdgpuWavefrontDispatched(*wf);

    ++entry.dispatched_wgs;
    ++dispatched;
    dispatched_workgroups_.fetch_add(1, std::memory_order_relaxed);
    return true;
  };

  while (entry.dispatched_wgs < entry.total_wgs) {
    if (entry.grid_faulted())
      break;
    if (entry.has_workgroup_clusters()) {
      assert(!entry.wgp_mode && "workgroup clusters are gfx1250-only and use CU mode");
      // The SPI interface chooses one WG at a time and cannot reserve all peers
      // in a cluster atomically. Plan clusters directly across the CP-visible CU
      // list until SPI grows an all-or-nothing cluster placement API.
      uint32_t cluster_size = entry.cluster_size();
      assert(entry.dispatched_wgs % cluster_size == 0 &&
             "clustered dispatch advances by whole clusters");
      assert(entry.total_wgs - entry.dispatched_wgs >= cluster_size &&
             "validate_cluster_shape guarantees a complete trailing cluster");
      // dispatched_wgs counts workgroups; the chunk here is a whole cluster, so
      // convert to a cluster index before asking the shard for its ordinal.
      uint32_t cluster_ordinal = entry.chunk_ordinal_for(entry.dispatched_wgs / cluster_size);
      uint32_t local_wg_id = entry.cluster_base_local_wg_id_for_ordinal(cluster_ordinal);
      std::vector<PlannedWorkgroup> plan;
      size_t planned_next_cu = next_cu_;
      if (!plan_cluster_workgroups(entry, local_wg_id, next_cu_, cus_, plan, planned_next_cu)) {
        if (!any_active_wavefronts(cus_)) {
          throw std::runtime_error(
              std::format("workgroup cluster {}x{}x{} cannot fit in available CU resources",
                          entry.cluster_size_x, entry.cluster_size_y, entry.cluster_size_z));
        }
        break;
      }
      next_cu_ = planned_next_cu;
      // A cluster is all-or-nothing: dispatch_to_placement() commits each peer's WG
      // bookkeeping (begin_workgroup) and LDS cluster pin (register_cluster_workgroup)
      // as it succeeds. If a later peer fails (dispatch_wf null or an init_wavefront_regs
      // throw), the already-committed peers would otherwise keep their refcount and pin
      // forever, permanently blocking maybe_reset_lds_alloc() on those CUs. Track the
      // committed peers and roll them back on any failure before propagating the error.
      std::vector<std::pair<ComputeUnitCore *, uint32_t>> committed_peers;
      committed_peers.reserve(plan.size());
      try {
        for (const auto &wg : plan) {
          if (entry.grid_faulted()) {
            for (const auto &[cu, gwg] : committed_peers) {
              erase_cluster_workgroup(entry.dispatch_id, gwg);
              cu->abort_workgroup(entry.dispatch_id, gwg);
            }
            const uint32_t rolled_back = static_cast<uint32_t>(committed_peers.size());
            entry.dispatched_wgs -= rolled_back;
            dispatched -= rolled_back;
            return dispatched;
          }
          ShaderProcessorInput::WorkgroupPlacement placement{
              wg.cu, &wg.cu->lds(), wg.cu->allocate_lds(entry.group_segment_fixed_size)};
          if (!dispatch_to_placement(wg.local_wg_id, wg.global_wg_id, placement))
            throw std::runtime_error("dispatch_wf failed after cluster placement was reserved");
          committed_peers.emplace_back(wg.cu, wg.global_wg_id);
        }
      } catch (...) {
        // Roll back the peers committed before the failure (dispatch_to_placement
        // already unwound its own reserved-but-uncommitted waves). erase_cluster_workgroup
        // unpins that peer's cluster LDS and drops its placement; abort_workgroup frees
        // its resident waves and clears the WG refcount without a completion notify.
        for (const auto &[cu, gwg] : committed_peers) {
          erase_cluster_workgroup(entry.dispatch_id, gwg);
          cu->abort_workgroup(entry.dispatch_id, gwg);
        }
        throw;
      }
      continue;
    }

    // Unclustered: the chunk is a single workgroup, so dispatched_wgs indexes
    // the shard's chunks directly and the shard maps that to a grid-wide id.
    // An unsharded entry maps the ordinal to itself.
    uint32_t local_wg_id = entry.chunk_ordinal_for(entry.dispatched_wgs);
    uint32_t global_wg_id = local_wg_id + entry.workgroup_id_offset;

    // SPI selects the CU or sibling-CU WGP based on descriptor mode and
    // resource availability.
    std::optional<ShaderProcessorInput::WorkgroupPlacement> placement;
    if (!spis_.empty()) {
      for (auto *spi : spis_) {
        placement = spi->allocate_workgroup(entry, global_wg_id);
        if (placement)
          break;
      }
    } else if (!entry.wgp_mode) {
      for (size_t attempt = 0; attempt < cus_.size(); ++attempt) {
        size_t cu_idx = (next_cu_ + attempt) % cus_.size();
        if (cus_[cu_idx]->can_accept_workgroup(entry.wfs_per_workgroup,
                                               entry.group_segment_fixed_size)) {
          auto *cu = cus_[cu_idx];
          placement = ShaderProcessorInput::WorkgroupPlacement{
              cu, &cu->lds(), cu->allocate_lds(entry.group_segment_fixed_size)};
          next_cu_ = (cu_idx + 1) % cus_.size();
          break;
        }
      }
    }

    if (!placement) {
      if (!any_active_wavefronts(cus_)) {
        throw std::runtime_error(
            std::format("workgroup cannot fit in available {} resources: waves={} LDS={} bytes",
                        entry.wgp_mode ? "WGP" : "CU", entry.wfs_per_workgroup,
                        entry.group_segment_fixed_size));
      }
      break;
    }

    if (!dispatch_to_placement(local_wg_id, global_wg_id, *placement))
      throw std::runtime_error("dispatch_wf failed after workgroup placement was reserved");
  }
  return dispatched;
}

// ---------------------------------------------------------------------------
// Completion notification from CU
// ---------------------------------------------------------------------------

void CommandProcessor::notify_wg_complete(uint32_t dispatch_id, uint32_t wg_id) {
  util::Logger::cp(
      [&](auto &os) { os << std::format("WG_COMPLETE d={} wg={}", dispatch_id, wg_id); });
  {
    std::lock_guard<std::recursive_mutex> lock(hw_queue_mutex_);
    for (auto *spi : spis_)
      if (spi->release_wgp_workgroup(dispatch_id, wg_id))
        break;
  }
  mark_cluster_workgroup_complete(dispatch_id, wg_id);
  {
    std::lock_guard<std::recursive_mutex> lock(hw_queue_mutex_);
    if (completion_) {
      completion_->notify_wg_complete(dispatch_id, wg_id, new_queue_states_);
      (void)drain_completions();
    }
  }
}

bool CommandProcessor::drain_completions() {
  if (!completion_)
    return true;

  bool stop_processing = false;
  bool retry_pending = false;
  for (;;) {
    const CompletionDrainResult result = completion_->drain_completions(new_queue_states_);
    retry_pending |= result.retry_pending;
    if (!result.terminal_fault)
      break;

    stop_processing = true;
    const CompletionDrainFault &fault = *result.terminal_fault;
    if (!fault.queue_idle) {
      notify_dispatch_vm_fault(fault.queue_id, fault.process_id, fault.dispatch_id, fault.outcome);
      continue;
    }

    const auto queue = std::ranges::find_if(hw_queues_, [&](const HwQueue &candidate) {
      return candidate.queue_id == fault.queue_id && candidate.process_id == fault.process_id;
    });
    if (queue != hw_queues_.end())
      queue->faulted = true;
  }

  if (retry_pending) {
    arm_stall_recheck(engine()->context(partition_id()).current_tick());
  }
  return !stop_processing;
}

bool CommandProcessor::fault_dispatch_local(uint32_t queue_id, uint32_t process_id,
                                            uint64_t dispatch_id, VmAccessOutcome outcome) {
  bool found = false;
  {
    // A shard may still be in transit from its owner. Remove it under the leaf
    // inbox lock without taking the queue lock, preserving the existing
    // cross-CP lock ordering.
    std::lock_guard<std::mutex> lock(fanout_inbox_mutex_);
    const auto old_size = fanout_inbox_.size();
    std::erase_if(fanout_inbox_, [&](DispatchEntry &entry) {
      if (entry.queue_id != queue_id || entry.process_id != process_id ||
          entry.dispatch_id != dispatch_id)
        return false;
      entry.terminal_faulted = true;
      if (entry.grid_completion)
        entry.grid_completion->mark_faulted();
      fanout_launch_metadata_inbox_.erase(entry.dispatch_id);
      return true;
    });
    found = fanout_inbox_.size() != old_size;
  }

  std::lock_guard<std::recursive_mutex> lock(hw_queue_mutex_);
  const auto queue = std::ranges::find_if(hw_queues_, [&](const HwQueue &candidate) {
    return candidate.queue_id == queue_id && candidate.process_id == process_id;
  });
  if (queue == hw_queues_.end())
    return false;
  const size_t queue_index = static_cast<size_t>(queue - hw_queues_.begin());
  HwQueueState &state = new_queue_states_[queue_index];

  size_t index = 0;
  for (auto entry = state.entries.begin(); entry != state.entries.end();) {
    if (entry->dispatch_id != dispatch_id) {
      ++entry;
      ++index;
      continue;
    }
    found = true;
    entry->terminal_faulted = true;
    if (entry->grid_completion)
      entry->grid_completion->mark_faulted();
    if (index < state.next_dispatch_idx)
      --state.next_dispatch_idx;
    entry = state.entries.erase(entry);
  }
  if (!found)
    return false;

  dispatch_launch_metadata_.erase(static_cast<uint32_t>(dispatch_id));
  queue->faulted = true;
  util::Logger::cp([&](auto &os) {
    os << std::format("{}: terminal VM fault pid={} qid={} dispatch={} outcome={}", name(),
                      process_id, queue_id, dispatch_id, static_cast<unsigned>(outcome));
  });
  erase_cluster_workgroups(static_cast<uint32_t>(dispatch_id));
  for (auto *cu : cus_)
    cu->abort_dispatch(static_cast<uint32_t>(dispatch_id));
  return true;
}

void CommandProcessor::notify_dispatch_vm_fault(uint32_t queue_id, uint32_t process_id,
                                                uint64_t dispatch_id, VmAccessOutcome outcome) {
  if (outcome == VmAccessOutcome::Complete || outcome == VmAccessOutcome::Unavailable ||
      dispatch_id > std::numeric_limits<uint32_t>::max())
    return;

  if (!fault_dispatch_local(queue_id, process_id, dispatch_id, outcome))
    return;
  // A fault can originate on any shard. Cross-XCD delivery uses a leaf inbox
  // and the peer's event thread, exactly like dispatch fan-out, so this path
  // never acquires another CP's queue mutex.
  for (CommandProcessor *peer : xcd_peers_) {
    if (peer != nullptr && peer != this) {
      peer->accept_dispatch_fault({.queue_id = queue_id,
                                   .process_id = process_id,
                                   .dispatch_id = dispatch_id,
                                   .outcome = outcome});
    }
  }
}

// INVARIANT: on_cu_idle() runs on the owning partition's engine thread — it is
// invoked from CU::execute_quantum() (the CU's own per-partition tick event), so
// the CU, this CP, and the CUs it dispatches to all share one partition. Dispatch
// therefore happens inline (same-partition, non-thread-safe path); a
// schedule_event_now() here would collapse ticks and break causal ordering.
void CommandProcessor::on_cu_idle() {
  if (cus_.empty())
    return;

  std::lock_guard<std::recursive_mutex> lock(hw_queue_mutex_);

  if (!drain_completions())
    return;

  // Retire any non-kernel entries (barrier-kind packets) that are now at
  // the head, then drain again so a dependent kernel behind them can proceed.
  for (size_t qi = 0; qi < new_queue_states_.size(); ++qi) {
    if (hw_queues_[qi].faulted || hw_queues_[qi].debug_suspended ||
        hw_queues_[qi].runtime_suspended || new_queue_states_[qi].publication_retry_pending)
      continue;
    auto &qs = new_queue_states_[qi];
    while (qs.next_dispatch_idx < qs.entries.size()) {
      auto &e = qs.entries[qs.next_dispatch_idx];
      if (e.barrier_bit && !barrier_satisfied(qs, qs.next_dispatch_idx))
        break;
      if (!e.is_non_kernel())
        break;
      e.completed_wgs = e.total_wgs;
      ++qs.next_dispatch_idx;
    }
  }
  if (!drain_completions())
    return;

  // Continue dispatching pending workgroups onto the just-freed CU in this same
  // tick rather than deferring to a now+1 doorbell event. Deferring routed every
  // dispatch continuation through the doorbell path; across the many tiny kernels
  // of an RCCL collective the engine would idle a full tick between steps, adding
  // latency the host socket layer then paid for. dispatch_workgroups() schedules
  // the CU's own tick event in the ordinary case. Record quiesced CUs because a
  // CU containing only debug-halted waves needs an explicit activation when a
  // newly dispatched wave makes it runnable again.
  std::vector<bool> was_idle(cus_.size());
  for (size_t i = 0; i < cus_.size(); ++i)
    was_idle[i] = cus_[i]->is_idle();
  for (size_t qi = 0; qi < new_queue_states_.size(); ++qi) {
    if (hw_queues_[qi].faulted || hw_queues_[qi].debug_suspended ||
        hw_queues_[qi].runtime_suspended || new_queue_states_[qi].publication_retry_pending)
      continue;
    auto &qs = new_queue_states_[qi];
    if (qs.next_dispatch_idx < qs.entries.size()) {
      auto &entry = qs.entries[qs.next_dispatch_idx];
      if (entry.barrier_bit && !barrier_satisfied(qs, qs.next_dispatch_idx))
        continue;
      if (!entry.is_non_kernel() && !entry.fully_dispatched()) {
        uint32_t sent = dispatch_workgroups(entry);
        if (sent > 0 && entry.fully_dispatched())
          ++qs.next_dispatch_idx;
      }
    }
  }
  for (size_t i = 0; i < cus_.size(); ++i) {
    if (was_idle[i] && !cus_[i]->is_idle())
      cus_[i]->schedule_work();
  }

  // The last workgroup of this XCD's share retires here, not in handle_doorbell,
  // so this is where a fanned-out shard parks to wait for its peers.
  arm_grid_wait_recheck();
}

bool CommandProcessor::step() {
  // Process dispatches across all queues.
  process_queues();
  return pending_entries() > 0;
}

void CommandProcessor::process_queues() {
  for (size_t qi = 0; qi < new_queue_states_.size(); ++qi) {
    if (hw_queues_[qi].faulted || hw_queues_[qi].is_sdma || hw_queues_[qi].debug_suspended ||
        hw_queues_[qi].runtime_suspended || new_queue_states_[qi].publication_retry_pending)
      continue;
    auto &qs = new_queue_states_[qi];
    while (qs.next_dispatch_idx < qs.entries.size()) {
      auto &entry = qs.entries[qs.next_dispatch_idx];

      if (entry.barrier_bit && !barrier_satisfied(qs, qs.next_dispatch_idx))
        break; // Stalled on barrier bit.

      if (entry.is_non_kernel()) {
        entry.completed_wgs = entry.total_wgs; // 0 == 0, immediately complete.
        ++qs.next_dispatch_idx;
        continue;
      }

      uint32_t sent = dispatch_workgroups(entry);
      if (entry.fully_dispatched())
        ++qs.next_dispatch_idx;
      if (sent == 0)
        break; // CU backpressure.
    }
  }
}

CommandProcessor::KernelDescriptorReadResult
CommandProcessor::read_kernel_descriptor(AddressSpaceHandle address_space, uint64_t kernel_object,
                                         uint32_t vmid) const {
  using namespace rocr::llvm::amdhsa;
  kernel_descriptor_t kd{};
  const VmAccessOutcome outcome =
      read_gpu_block(address_space, kernel_object, &kd, sizeof(kd), vmid);
  return {.outcome = outcome, .descriptor = kd};
}

/// Scan backward from ptr to find the ELF header (\x7fELF) at a page boundary.
/// Both ptr and limit must be readable host memory.
static const uint8_t *find_elf_base(const uint8_t *ptr, const uint8_t *limit) {
  auto *page = reinterpret_cast<const uint8_t *>(reinterpret_cast<uintptr_t>(ptr) & ~0xFFFULL);
  for (; page >= limit; page -= 0x1000) {
    if (page[0] == 0x7f && page[1] == 'E' && page[2] == 'L' && page[3] == 'F')
      return page;
  }
  return nullptr;
}

VmAccessOutcome CommandProcessor::process_aql_packet(const hsa_kernel_dispatch_packet_t &pkt,
                                                     const HwQueue &queue, uint64_t pkt_addr,
                                                     uint32_t queue_packet_id, HwQueueState &qs,
                                                     uint64_t aql_packet_id,
                                                     ClusterDispatchShape cluster_shape) {
  bool host_accessible = queue.host_accessible;
  using namespace rocr::llvm::amdhsa;
  const KernelDescriptorReadResult descriptor =
      read_kernel_descriptor(queue.address_space, pkt.kernel_object, queue.process_id);
  if (descriptor.outcome != VmAccessOutcome::Complete)
    return descriptor.outcome;
  const kernel_descriptor_t &kd = descriptor.descriptor;
  uint32_t vgpr_gran =
      AMDHSA_BITS_GET(kd.compute_pgm_rsrc1, COMPUTE_PGM_RSRC1_GRANULATED_WORKITEM_VGPR_COUNT);
  uint32_t sgpr_gran =
      AMDHSA_BITS_GET(kd.compute_pgm_rsrc1, COMPUTE_PGM_RSRC1_GRANULATED_WAVEFRONT_SGPR_COUNT);
  rj_code_arch_t arch = cus_.empty() ? ROCJITSU_CODE_ARCH_CDNA1 : cus_[0]->config().arch;
  const uint32_t wave_size = kernel_wavefront_size(arch, kd);
  const auto vgpr_granularity = descriptor_vgpr_count_granule_for_wavefront(arch, wave_size);
  if (!vgpr_granularity)
    throw std::runtime_error("unsupported kernel wave size for VGPR descriptor decoding");
  uint32_t vgprs = (vgpr_gran + 1) * *vgpr_granularity;
  uint32_t sgprs = sgpr_count_is_descriptor_encoded(arch, sgpr_gran) ? (sgpr_gran + 1) * 8 : 0;
  uint32_t user_sgprs = kernel_descriptor_user_sgpr_count(arch, kd);
  uint64_t entry_pc = pkt.kernel_object + static_cast<uint64_t>(kd.kernel_code_entry_byte_offset);
  uint64_t code_load_bias = 0;

  uint32_t wg_size =
      static_cast<uint32_t>(pkt.workgroup_size_x) * pkt.workgroup_size_y * pkt.workgroup_size_z;
  uint32_t wfs_per_wg = (wg_size + wave_size - 1) / wave_size;

  uint32_t num_dims = pkt.setup & 0x3;
  uint32_t grid_wgs_x =
      util::ceil_div_or_one(pkt.grid_size_x, static_cast<uint32_t>(pkt.workgroup_size_x));
  uint32_t grid_wgs_y =
      util::ceil_div_or_one(pkt.grid_size_y, static_cast<uint32_t>(pkt.workgroup_size_y));
  uint32_t grid_wgs_z =
      util::ceil_div_or_one(pkt.grid_size_z, static_cast<uint32_t>(pkt.workgroup_size_z));
  uint32_t total_wgs = grid_wgs_x * grid_wgs_y * grid_wgs_z;

  DispatchLaunchMetadata launch_metadata{};
  uint64_t queue_ptr = 0;
  uint64_t scratch_backing_addr = 0;
  if (host_accessible) {
    queue_ptr = queue.read_ptr_va - offsetof(amd_queue_t, read_dispatch_id);
    if (AMDHSA_BITS_GET(kd.kernel_code_properties,
                        KERNEL_CODE_PROPERTY_ENABLE_SGPR_PRIVATE_SEGMENT_BUFFER)) {
      const uint64_t descriptor_va = queue_ptr + offsetof(amd_queue_t, scratch_resource_descriptor);
      const VmAccessOutcome outcome = read_gpu_block(
          queue.address_space, descriptor_va, launch_metadata.scratch_resource_descriptor.data(),
          sizeof(launch_metadata.scratch_resource_descriptor), queue.process_id);
      if (outcome != VmAccessOutcome::Complete)
        return outcome;
    }
    if (AMDHSA_BITS_GET(kd.kernel_code_properties, KERNEL_CODE_PROPERTY_ENABLE_SGPR_DISPATCH_ID)) {
      const uint64_t dispatch_id_va = queue_ptr + offsetof(amd_queue_t, write_dispatch_id);
      const AtomicLoadResult loaded =
          read_gpu_u64(queue.address_space, dispatch_id_va, queue.process_id);
      if (loaded.outcome != VmAccessOutcome::Complete)
        return loaded.outcome;
      launch_metadata.write_dispatch_id = loaded.value;
    }

    const uint32_t private_segment_fixed_size =
        std::max(kd.private_segment_fixed_size, pkt.private_segment_size);
    if (private_segment_fixed_size > 0) {
      const uint64_t scratch_loc_va =
          queue_ptr + offsetof(amd_queue_t, scratch_backing_memory_location);
      const AtomicLoadResult loaded =
          read_gpu_u64(queue.address_space, scratch_loc_va, queue.process_id);
      if (loaded.outcome != VmAccessOutcome::Complete)
        return loaded.outcome;
      scratch_backing_addr = loaded.value;
    }
  }

  DispatchEntry dp{};
  dp.dispatch_id = allocate_dispatch_id();
  dp.profiling_start_timestamp = hsa_system_timestamp();
  dp.queue_id = queue.queue_id;
  dp.queue_packet_id = queue_packet_id;
  dp.address_space = queue.address_space;
  dp.interrupt_sink = queue.interrupt_sink;
  dp.process_id = queue.process_id;
  dp.aql_packet_id = static_cast<uint32_t>(aql_packet_id);
  dp.kernel_entry_pc = entry_pc;
  dp.total_wgs = total_wgs;
  dp.kind = DispatchPacketKind::Kernel;
  dp.dispatched_wgs = 0;
  dp.completed_wgs = 0;
  dp.wfs_per_workgroup = wfs_per_wg;
  uint32_t sgpr_limit = cus_.empty() ? 112 : cus_[0]->config().sgprs_per_wf;
  uint32_t vgpr_limit = cus_.empty() ? 256 : cus_[0]->vgpr_allocation_block_size();
  uint32_t required_sgprs = sgprs > 0 ? sgprs : sgpr_limit;
  if (arch == ROCJITSU_CODE_ARCH_CDNA3 || arch == ROCJITSU_CODE_ARCH_CDNA4)
    required_sgprs = std::max(required_sgprs, 34u); // s32 stack pointer, s33 frame pointer
  dp.sgprs_per_wf = std::min(required_sgprs, sgpr_limit);
  dp.vgprs_per_wf = std::min(vgprs > 0 ? vgprs : vgpr_limit, vgpr_limit);
  dp.kernarg_addr = reinterpret_cast<uint64_t>(pkt.kernarg_address);
  dp.kernarg_size = kd.kernarg_size;
  dp.num_user_sgprs = user_sgprs;
  dp.kernel_code_properties = kd.kernel_code_properties;
  if (arch == ROCJITSU_CODE_ARCH_CDNA5) {
    const uint32_t named_barrier_blocks =
        AMDHSA_BITS_GET(kd.compute_pgm_rsrc3, COMPUTE_PGM_RSRC3_GFX125_NAMED_BAR_CNT);
    dp.num_named_barriers = std::min(named_barrier_blocks * 4u, ComputeUnitCore::kMaxNamedBarriers);
  }
  dp.kernel_wave_size = wave_size;
  dp.kernarg_preload = kd.kernarg_preload;
  dp.initial_mode_raw = initial_mode_from_compute_pgm_rsrc1(kd.compute_pgm_rsrc1, arch);
  dp.private_segment_fixed_size = std::max(kd.private_segment_fixed_size, pkt.private_segment_size);
  dp.group_segment_fixed_size = std::max(kd.group_segment_fixed_size, pkt.group_segment_size);
  dp.wgp_mode = isa_properties(arch).supports_wgp_mode &&
                AMDHSA_BITS_GET(kd.compute_pgm_rsrc1, COMPUTE_PGM_RSRC1_WGP_MODE) != 0;

  uint64_t lds_capacity = 0;
  if (dp.wgp_mode) {
    for (const auto *spi : spis_)
      lds_capacity = std::max<uint64_t>(lds_capacity, spi->max_wgp_lds_bytes());
  } else {
    for (const auto *cu : cus_)
      lds_capacity =
          std::max<uint64_t>(lds_capacity, static_cast<uint64_t>(cu->config().lds_size_kb) * 1024u);
  }
  const uint64_t aligned_lds =
      (static_cast<uint64_t>(dp.group_segment_fixed_size) + 255u) & ~uint64_t{255u};
  if (dp.wgp_mode && lds_capacity == 0) {
    throw std::runtime_error(
        "WGP-mode kernel dispatch requires a sibling-CU pair, but none is configured");
  }
  if (aligned_lds > lds_capacity) {
    throw std::runtime_error(std::format(
        "kernel dispatch requests {} bytes of LDS ({} bytes after alignment) in {} mode, but "
        "the simulator topology provides at most {} bytes",
        dp.group_segment_fixed_size, aligned_lds, dp.wgp_mode ? "WGP" : "CU", lds_capacity));
  }

  // For KFD dispatches, provide pointers the kernel may need via user SGPRs.
  // The queue_ptr and dispatch_ptr are GPU VAs that the kernel reads via SMEM.
  if (host_accessible) {
    dp.dispatch_ptr = pkt_addr;
    dp.queue_ptr = queue_ptr;
    if (dp.private_segment_fixed_size > 0) {
      uint64_t scratch_loc_va =
          dp.queue_ptr + offsetof(amd_queue_t, scratch_backing_memory_location);
      dp.scratch_backing_addr = scratch_backing_addr;
      if (dp.scratch_backing_addr == 0 && scratch_resolver_)
        dp.scratch_backing_addr = scratch_resolver_(queue.process_id);

      // Publish the scratch backing location and COMPUTE_TMPRING_SIZE into the
      // ABI-stable part of amd_queue_t so rocm-dbgapi can compute each wave's
      // private (scratch) memory region, letting ROCgdb read scratch-resident
      // variables. Real CP firmware populates these when it assigns scratch to
      // the queue; the emulator's ROCr instead sets the backing via
      // SET_SCRATCH_BACKING_VA and leaves these fields zero, so the CP fills
      // them here. Field layout per rocdbgapi architecture.cpp
      // scratch-memory region. The WAVES field is common, while ISA properties
      // describe the generation-specific WAVESIZE unit and field width.
      if (dp.scratch_backing_addr != 0 && !cus_.empty()) {
        uint64_t per_wave_bytes =
            static_cast<uint64_t>(dp.private_segment_fixed_size) * cus_[0]->wf_size();
        // setup_wavefront() allocates scratch slots at a 1 KiB boundary. Encode
        // that actual stride, rather than merely rounding to the register's
        // unit, so flat_scratch agrees with rocm-dbgapi for every scoreboard
        // slot after slot zero.
        const uint64_t per_wave_stride = ((per_wave_bytes + 1023) / 1024) * 1024;
        const auto properties = isa_properties(arch);
        const uint32_t wavesize_unit = properties.compute_tmpring_wavesize_granule;
        assert(wavesize_unit != 0 && properties.compute_tmpring_wavesize_bits != 0);
        const uint32_t wavesize_field = static_cast<uint32_t>(per_wave_stride / wavesize_unit);
        uint32_t waves_field = 0;
        if (arch == ROCJITSU_CODE_ARCH_CDNA5) {
          // gfx12 interprets WAVES as the number of physical scratch slots per
          // shader engine. The CWSR wave word supplies the SE plus its per-SE
          // scoreboard slot, so publish the same capacity used by allocation.
          waves_field = scratch_waves_per_se_;
        } else {
          // Older debugger layouts interpret WAVES as a device-wide count and
          // require it to be divisible by the shader-engine count.
          uint32_t se = std::max(1u, scratch_wave_divisor_);
          uint64_t total_waves = static_cast<uint64_t>(total_wgs) * wfs_per_wg;
          waves_field =
              static_cast<uint32_t>(((std::max<uint64_t>(1, total_waves) + se - 1) / se) * se);
        }
        const uint32_t wavesize_mask =
            util::mask<uint32_t>(properties.compute_tmpring_wavesize_bits);
        uint32_t tmpring = (waves_field & 0xFFFu) | ((wavesize_field & wavesize_mask) << 12);
        (void)write_gpu_block(queue.address_space, scratch_loc_va, &dp.scratch_backing_addr,
                              sizeof(dp.scratch_backing_addr), queue.process_id);
        (void)write_gpu_block(queue.address_space,
                              dp.queue_ptr + offsetof(amd_queue_t, compute_tmpring_size), &tmpring,
                              sizeof(tmpring), queue.process_id);
      }
    }
  }

  dp.workgroup_id_offset = workgroup_id_offset_;
  dp.grid_size_x = pkt.grid_size_x;
  dp.grid_size_y = (num_dims >= 2) ? pkt.grid_size_y : 1;
  dp.grid_size_z = (num_dims >= 3) ? pkt.grid_size_z : 1;
  // For WG ID decomposition, use the dispatch dimensionality (setup field).
  // A 1D dispatch flattens the entire grid into workgroup_id_x.
  dp.grid_wgs_x = (num_dims <= 1) ? total_wgs : grid_wgs_x;
  dp.grid_wgs_y = (num_dims >= 2) ? grid_wgs_y : 1;
  dp.grid_wgs_z = (num_dims >= 3) ? grid_wgs_z : 1;
  dp.grid_yz_valid = num_dims >= 2;
  dp.cluster_size_x = nonzero_or_one(cluster_shape.size_x);
  dp.cluster_size_y = nonzero_or_one(cluster_shape.size_y);
  dp.cluster_size_z = nonzero_or_one(cluster_shape.size_z);
  dp.cluster_count_x = cluster_shape.count_x == 0
                           ? (dp.grid_wgs_x + dp.cluster_size_x - 1) / dp.cluster_size_x
                           : cluster_shape.count_x;
  dp.cluster_count_y = cluster_shape.count_y == 0
                           ? (dp.grid_wgs_y + dp.cluster_size_y - 1) / dp.cluster_size_y
                           : cluster_shape.count_y;
  dp.cluster_count_z = cluster_shape.count_z == 0
                           ? (dp.grid_wgs_z + dp.cluster_size_z - 1) / dp.cluster_size_z
                           : cluster_shape.count_z;
  validate_cluster_shape(dp);
  dp.enable_wg_id_x =
      AMDHSA_BITS_GET(kd.compute_pgm_rsrc2, COMPUTE_PGM_RSRC2_ENABLE_SGPR_WORKGROUP_ID_X);
  dp.enable_wg_id_y =
      AMDHSA_BITS_GET(kd.compute_pgm_rsrc2, COMPUTE_PGM_RSRC2_ENABLE_SGPR_WORKGROUP_ID_Y);
  dp.enable_wg_id_z =
      AMDHSA_BITS_GET(kd.compute_pgm_rsrc2, COMPUTE_PGM_RSRC2_ENABLE_SGPR_WORKGROUP_ID_Z);
  dp.enable_vgpr_workitem_id = static_cast<uint8_t>(
      AMDHSA_BITS_GET(kd.compute_pgm_rsrc2, COMPUTE_PGM_RSRC2_ENABLE_VGPR_WORKITEM_ID));
  dp.workgroup_size_x = pkt.workgroup_size_x;
  dp.workgroup_size_y = pkt.workgroup_size_y;
  dp.workgroup_size_z = pkt.workgroup_size_z;
  dp.completion_signal = pkt.completion_signal.handle;
  dp.host_signal = false;
  dp.barrier_bit = (pkt.header >> HSA_PACKET_HEADER_BARRIER) & 1;

  // Process AQL acquire fence: invalidate caches so the kernel sees the
  // latest host/agent writes (kernarg data, input buffers, etc.).
  // On real hardware the CP issues GL1_INV + GL2_INV for SYSTEM/AGENT scope.
  // Only this XCD's CUs are reachable from here; a peer XCD's caches belong to
  // another partition and must not be touched from this thread. The shard carries
  // the fence instead, and each peer performs the same invalidate on its own thread
  // when it takes delivery -- see drain_fanout_inbox().
  uint32_t acquire_scope = (pkt.header >> HSA_PACKET_HEADER_SCACQUIRE_FENCE_SCOPE) & 0x3;
  dp.acquire_invalidate = acquire_scope >= HSA_FENCE_SCOPE_AGENT;
  if (dp.acquire_invalidate && !cus_.empty()) {
    // Deliberately the per-CU walk, not the deduplicated flush_gpu_caches(). The
    // two are equivalent per invocation, but collapsing the repeated sweeps on the
    // release path caused peer ranks to hang on flags left unpublished in L2, and
    // that mechanism is still not understood. Until it is, this path -- which
    // predates fan-out -- keeps exactly the cache behaviour it had, and only the
    // new peer-side fence in drain_fanout_inbox() uses the collapsed form.
    for (auto *cu : cus_)
      cu->flush_all(queue.process_id);
  }

  std::string kernel_symbol;
  if (host_accessible && memory_) {
    auto [host_range_base, host_range_size] =
        memory_->find_host_range(pkt.kernel_object, queue.process_id);
    auto *kernel_object_host_ptr = memory_->resolve_host_ptr(pkt.kernel_object, queue.process_id);
    if (host_range_base != 0 && kernel_object_host_ptr) {
      auto *host_range_begin = reinterpret_cast<const uint8_t *>(host_range_base);
      auto *elf_base = find_elf_base(kernel_object_host_ptr, host_range_begin);
      if (elf_base) {
        const uint64_t kernel_offset = static_cast<uint64_t>(kernel_object_host_ptr - elf_base);
        if (pkt.kernel_object >= kernel_offset)
          code_load_bias = pkt.kernel_object - kernel_offset;
        uint64_t elf_accessible =
            host_range_size - static_cast<uint64_t>(elf_base - host_range_begin);
        kernel_symbol = find_kernel_symbol(kernel_object_host_ptr, elf_base, elf_accessible);
      }
    }
  }
  dp.code_load_bias = code_load_bias;
  std::string kernel_name = kernel_display_name(kernel_symbol);
  ++total_dispatched_;

  KernelDispatchInfo dispatch_info{};
  dispatch_info.dispatch_id = dp.dispatch_id;
  dispatch_info.kernel_object = pkt.kernel_object;
  dispatch_info.entry_pc = entry_pc;
  dispatch_info.kernel_symbol = kernel_symbol;
  dispatch_info.kernel_name = kernel_name;
  dispatch_info.grid_size_x = pkt.grid_size_x;
  dispatch_info.grid_size_y = pkt.grid_size_y;
  dispatch_info.grid_size_z = pkt.grid_size_z;
  dispatch_info.workgroup_size_x = pkt.workgroup_size_x;
  dispatch_info.workgroup_size_y = pkt.workgroup_size_y;
  dispatch_info.workgroup_size_z = pkt.workgroup_size_z;
  dispatch_info.workgroup_count = total_wgs;
  dispatch_info.wfs_per_workgroup = wfs_per_wg;
  dispatch_info.sgprs_per_wf = dp.sgprs_per_wf;
  dispatch_info.vgprs_per_wf = dp.vgprs_per_wf;
  plugin_group_->onAmdgpuDispatchPacketProcessed(dispatch_info);

  util::Logger::vm([&](auto &os) {
    os << std::format("dispatch #{} d={} \"{}\" symbol=\"{}\" grid=[{},{},{}] wg=[{},{},{}] wgs={} "
                      "lds={} mode={} sgpr={} vgpr={} sig={:#x}",
                      total_dispatched_, dp.dispatch_id, dispatch_info.kernelNameOrUnknown(),
                      dispatch_info.kernelSymbolOrUnknown(), pkt.grid_size_x, pkt.grid_size_y,
                      pkt.grid_size_z, pkt.workgroup_size_x, pkt.workgroup_size_y,
                      pkt.workgroup_size_z, total_wgs, kd.group_segment_fixed_size,
                      dp.wgp_mode ? "WGP" : "CU", dp.sgprs_per_wf, dp.vgprs_per_wf,
                      dp.completion_signal);
  });
  util::Logger::cp([&](auto &os) {
    os << std::format("DISPATCH #{} d={} \"{}\" symbol=\"{}\" wgs={} wfs/wg={} sig={:#x} pid={} "
                      "ko={:#x} pc={:#x} kernarg={:#x} user_sgprs={}",
                      total_dispatched_, dp.dispatch_id, dispatch_info.kernelNameOrUnknown(),
                      dispatch_info.kernelSymbolOrUnknown(), total_wgs, wfs_per_wg,
                      dp.completion_signal, dp.process_id, pkt.kernel_object, entry_pc,
                      dp.kernarg_addr, dp.num_user_sgprs);
    if (memory_) {
      auto *ko_ptr = memory_->translate_debug(pkt.kernel_object, queue.process_id);
      auto *pc_ptr = memory_->translate_debug(entry_pc, queue.process_id, sizeof(uint32_t));
      os << std::format(" ko_mapped={} pc_mapped={} mem={:#x}", ko_ptr != nullptr,
                        pc_ptr != nullptr, reinterpret_cast<uintptr_t>(memory_));
      if (pc_ptr) {
        uint32_t first_word;
        std::memcpy(&first_word, pc_ptr, sizeof(first_word));
        os << std::format(" first_inst={:#010x}", first_word);
      }
    }
  });

  dispatch_launch_metadata_.insert_or_assign(dp.dispatch_id, launch_metadata);
  if (queue.xcd_fanout)
    fan_out_dispatch(dp, launch_metadata);

  qs.push_entry(std::move(dp));
  return VmAccessOutcome::Complete;
}

void CommandProcessor::arm_grid_wait_recheck() {
  // A shard whose own share is done but whose grid is still running on another
  // XCD is a stall like any other, and has to be re-armed as one.
  //
  // The XCD that retires the grid does call wake_all_xcds(), but that wake
  // travels the engine's cross-thread async queue, which neither contributes to
  // LBTS nor counts as outstanding work when the engine tests for termination.
  // With one partition per XCD, every partition can publish TICK_MAX in the same
  // epoch the wake is deposited, and the run ends on that before the next epoch
  // drains it -- so the XCD holding the completion signal never re-drains and
  // never writes it. Keeping a re-check on this CP's own event queue holds its
  // partition's next-event time finite for exactly as long as it is waiting,
  // which leaves the wake an optimization rather than the only thing standing
  // between the grid retiring and the signal firing.
  //
  // Caller must hold hw_queue_mutex_ and must be on this CP's own partition
  // thread, which is where the re-check is enqueued.
  for (const auto &qs : new_queue_states_) {
    if (qs.entries.empty())
      continue;
    const auto &head = qs.entries.front();
    if (head.fully_completed() && !head.grid_fully_completed()) {
      arm_stall_recheck(engine()->context(partition_id()).current_tick());
      return;
    }
  }
}

void CommandProcessor::arm_stall_recheck(simdojo::Tick now) {
  // A doorbell poll thread runs only for queues this CP polls; it re-checks
  // stall_pending_ at its 100us cadence, so the engine can idle instead of spinning.
  // Internal test queues have no poll thread — they are driven by engine->run()/
  // step() — and neither do fan-out replicas, so in both cases the re-check must be
  // kept alive on the main event queue instead.
  if (polls_kfd_queues()) {
    stall_pending_.store(true, std::memory_order_release);
    return;
  }
  // Back the re-check off rather than re-arming on the very next tick. What
  // actually ends one of these waits is an external event -- a peer's shard, or
  // the wake the retiring XCD sends -- and each of those resets the backoff, so
  // the wait still ends promptly. This event only has to keep the partition's
  // next-event time finite so the engine cannot decide the run is over while a
  // cross-thread wake is still undelivered (see arm_grid_wait_recheck).
  //
  // At one tick it is a spin, and a costly one: with fan-out every peer XCD waits
  // on the owner's grid, and a peer re-entered this handler once per simulated
  // tick -- 17M times on a corpus case that needs 24 doorbells without fan-out,
  // which is where a 12x slowdown came from. Backing off is nearly free in
  // simulated time: with no other event pending the engine jumps straight to the
  // re-check, so a longer interval skips idle ticks rather than adding latency.
  schedule_event(&doorbell_event_, now + stall_recheck_backoff_);
  stall_recheck_backoff_ = std::min(stall_recheck_backoff_ * 2, kMaxStallRecheckBackoff);
}

void CommandProcessor::fetch_from_queue(HwQueue &queue, HwQueueState &qs, simdojo::Tick now) {
  if (!memory_)
    return;
  if (queue.faulted)
    return;
  // A replica's work arrives as dispatch shards, not from the ring. Reading the
  // ring here would also advance a read pointer the owning XCD owns, and its
  // suspension flags are the owner's copy rather than state this CP maintains.
  if (queue.fanout_replica)
    return;
  if (!queue.is_sdma && queue.read_pointer_publication_pending) {
    if (!queue.read_pointer_publication_access) {
      queue.faulted = true;
      return;
    }
    const VmAccessOutcome outcome = queue.read_pointer_publication_access->atomic_store(
        queue.read_ptr_va, sizeof(queue.pending_read_pointer), queue.pending_read_pointer);
    if (outcome == VmAccessOutcome::Unavailable) {
      arm_stall_recheck(now);
      return;
    }
    queue.read_pointer_publication_pending = false;
    queue.read_pointer_publication_access.reset();
    if (outcome != VmAccessOutcome::Complete) {
      queue.faulted = true;
      return;
    }
  }
  auto load_queue_pointer = [&](uint64_t address, uint64_t &value) {
    const AtomicLoadResult loaded = read_gpu_u64(queue.address_space, address, queue.process_id);
    if (loaded.outcome == VmAccessOutcome::Complete) {
      value = loaded.value;
      return true;
    }
    if (loaded.outcome == VmAccessOutcome::Unavailable)
      arm_stall_recheck(now);
    else
      queue.faulted = true;
    return false;
  };
  SdmaQueueRuntime *sdma_runtime = nullptr;
  if (queue.is_sdma) {
    const auto runtime = sdma_queue_runtimes_.find(queue.registration_id);
    if (runtime == sdma_queue_runtimes_.end()) {
      queue.faulted = true;
      return;
    }
    sdma_runtime = &runtime->second;
  }
  auto load_read_pointer = [&](uint64_t &value) {
    if (sdma_runtime != nullptr) {
      value = sdma_runtime->runner->cursor();
      return true;
    }
    return load_queue_pointer(queue.read_ptr_va, value);
  };
  if (queue.debug_suspended || queue.runtime_suspended) {
    // A command-processor event can race a debugger suspension even when this
    // queue has no new packets. Do not turn that stale event into an endless
    // resume/event chain: request a resume pass only when packet fetch really
    // was deferred. Compute queues count packets; SDMA queues count bytes.
    uint64_t write_idx = 0;
    uint64_t read_idx = 0;
    if (!load_read_pointer(read_idx))
      return;
    if (queue.is_sdma && !queue.host_accessible) {
      // A VFIO SDMA doorbell carries the producer byte offset. Its MQD write-pointer
      // address may itself live in the device GART aperture rather than the process
      // address space used by the ring, so it is neither necessary nor correct to
      // translate that address through the process page tables.
      write_idx = queue.last_doorbell;
    } else if (!load_queue_pointer(queue.write_ptr_va, write_idx)) {
      return;
    }
    const uint64_t fetch_idx = queue.is_sdma ? read_idx : std::max(read_idx, queue.fetch_cursor);
    queue.debug_work_deferred |= fetch_idx < write_idx;
    return;
  }
  if (queue.host_accessible ? (queue.doorbell_base == nullptr) : (queue.doorbell_va == 0))
    return;

  // Read write and read indices. For KFD queues, pointers are in host memory
  // and can be read directly. For internal test queues, they're in GpuMemory.
  uint64_t write_idx = 0;
  uint64_t read_idx = 0;
  if (!load_read_pointer(read_idx))
    return;
  if (queue.is_sdma && !queue.host_accessible) {
    // The MMIO doorbell is the authoritative producer pointer for a VFIO SDMA
    // queue. The MQD write-pointer address can be a GART address that is not
    // mapped in the process VM, so reading it through queue.address_space would
    // manufacture a queue fault before the doorbell value can be consumed.
    write_idx = queue.last_doorbell;
  } else if (!load_queue_pointer(queue.write_ptr_va, write_idx)) {
    return;
  }
  util::Logger::vm([&](auto &os) {
    static uint64_t fetch_count = 0;
    if (write_idx != read_idx && ++fetch_count <= 50)
      os << std::format("FETCH q={} w={} r={} delta={} sdma={}", queue.queue_id, write_idx,
                        read_idx, write_idx - read_idx, queue.is_sdma);
  });

  // SDMA queues use byte-granularity pointers and have their own doorbell
  // semantics — skip the AQL doorbell clamping that assumes packet indices.
  if (queue.is_sdma) {
    if (queue.doorbell_base) {
      uint64_t db_val = std::atomic_ref<uint64_t>(
                            *reinterpret_cast<uint64_t *>(static_cast<char *>(queue.doorbell_base) +
                                                          queue.doorbell_offset))
                            .load(std::memory_order_acquire);
      if (db_val != std::numeric_limits<uint64_t>::max() && db_val > write_idx)
        write_idx = db_val;
    }
    util::Logger::cp([&](auto &os) {
      os << std::format("{}: SDMA_FETCH pid={} qid={} read={} write={} delta={}", name(),
                        queue.process_id, queue.queue_id, read_idx, write_idx,
                        write_idx - read_idx);
    });
    if (read_idx >= write_idx && !sdma_runtime->runner->in_flight())
      return;
    switch (sdma_runtime->runner->service(write_idx)) {
    case SdmaQueueServiceOutcome::Drained:
      break;
    case SdmaQueueServiceOutcome::Unavailable:
      arm_stall_recheck(now);
      break;
    case SdmaQueueServiceOutcome::Faulted:
    case SdmaQueueServiceOutcome::Malformed:
      queue.faulted = true;
      break;
    }
    return;
  }

  // AQL doorbell clamping (compute queues only).
  // Use the CP-private fetch cursor as the authoritative next-packet index. It
  // normally equals read_ptr_va (the CP is the sole writer of a compute queue's
  // read pointer), but while the debugger holds read_ptr_va at a trapped
  // dispatch, the cursor stays ahead so already-dispatched packets are not
  // re-fetched.
  read_idx = std::max(read_idx, queue.fetch_cursor);
  uint64_t process_limit = write_idx;
  if (queue.host_accessible) {
    const uint64_t doorbell = queue.last_doorbell;
    if (doorbell != std::numeric_limits<uint64_t>::max()) {
      uint64_t doorbell_limit = doorbell + 1;
      if (doorbell_limit < process_limit)
        process_limit = doorbell_limit;
    }
    if (read_idx >= process_limit)
      return;
  } else if (read_idx >= process_limit) {
    return;
  }

  constexpr uint32_t AQL_PACKET_SIZE = 64;
  uint32_t num_slots = queue.ring_size / AQL_PACKET_SIZE;

  while (read_idx < process_limit) {
    uint32_t slot = static_cast<uint32_t>(read_idx % num_slots);
    uint64_t pkt_addr = queue.ring_base_va + slot * AQL_PACKET_SIZE;

    hsa_kernel_dispatch_packet_t pkt{};
    const VmAccessOutcome packet_read =
        read_gpu_block(queue.address_space, pkt_addr, &pkt, AQL_PACKET_SIZE, queue.process_id);
    if (packet_read != VmAccessOutcome::Complete) {
      process_limit = read_idx;
      if (packet_read == VmAccessOutcome::Unavailable)
        arm_stall_recheck(now);
      else
        queue.faulted = true;
      break;
    }

    uint8_t pkt_type = pkt.header & 0xFF;
    util::Logger::cp([&](auto &os) {
      auto type_name = [](uint8_t t) -> const char * {
        switch (t) {
        case 0:
          return "VENDOR_SPECIFIC";
        case 1:
          return "INVALID";
        case 2:
          return "KERNEL_DISPATCH";
        case 3:
          return "BARRIER_AND";
        case 5:
          return "BARRIER_OR";
        default:
          return "UNKNOWN";
        }
      };
      os << std::format("PKT q={} slot={} type={}({}) header={:#x} barrier_bit={} read_idx={}",
                        queue.queue_id, slot, pkt_type, type_name(pkt_type), pkt.header,
                        (pkt.header >> HSA_PACKET_HEADER_BARRIER) & 1, read_idx);
    });

    if (pkt_type == HSA_PACKET_TYPE_INVALID) {
      util::Logger::cp([&](auto &os) {
        os << std::format("{}: INVALID_RETRY q={} slot={} read_idx={} va={:#x}", name(),
                          queue.queue_id, slot, read_idx, pkt_addr);
      });
      process_limit = read_idx;
      // The runtime has not finished writing this packet's header. invalid_pending_
      // mirrors the barrier/dependency stalls' stall_pending_: the KFD poll thread
      // re-checks at its 100us cadence (both flags gate the same poll-loop nudge).
      // Unlike those stalls this has no internal-test-queue fallback because a test
      // writes the whole packet before ringing the doorbell, so a test queue never
      // observes an in-flight INVALID header (the only writer that leaves one is the
      // real runtime racing the doorbell, which always has a poll thread).
      invalid_pending_.store(true, std::memory_order_release);
      break;
    }

    if (pkt_type == HSA_PACKET_TYPE_KERNEL_DISPATCH) {
      const VmAccessOutcome outcome = process_aql_packet(pkt, queue, pkt_addr, slot, qs, read_idx);
      if (outcome != VmAccessOutcome::Complete) {
        process_limit = read_idx;
        if (outcome == VmAccessOutcome::Unavailable)
          arm_stall_recheck(now);
        else
          queue.faulted = true;
        break;
      }
    } else if (pkt_type == HSA_PACKET_TYPE_BARRIER_AND || pkt_type == HSA_PACKET_TYPE_BARRIER_OR) {
      constexpr uint32_t SIG_VAL_OFF = 8;
      bool is_and = (pkt_type == HSA_PACKET_TYPE_BARRIER_AND);
      hsa_barrier_and_packet_t barrier{};
      std::memcpy(&barrier, &pkt, sizeof(barrier));

      // Non-blocking dependency check: if any dependency is unsatisfied,
      // stop fetching from this queue. The next doorbell event will retry.
      // This prevents blocking the CP from processing other queues (SDMA)
      // that may be responsible for satisfying these dependencies.
      bool deps_satisfied =
          is_and; // AND: assume true until a dep fails; OR: assume false until one passes
      bool has_deps = false;
      bool dependency_access_failed = false;
      for (int dep = 0; dep < 5; ++dep) {
        const uint64_t dep_sig = barrier.dep_signal[dep].handle;
        if (dep_sig == 0)
          continue;
        has_deps = true;
        const AtomicLoadResult loaded =
            read_gpu_u64(queue.address_space, dep_sig + SIG_VAL_OFF, queue.process_id);
        if (loaded.outcome != VmAccessOutcome::Complete) {
          process_limit = read_idx;
          if (loaded.outcome == VmAccessOutcome::Unavailable)
            arm_stall_recheck(now);
          else
            queue.faulted = true;
          dependency_access_failed = true;
          break;
        }
        const auto v = static_cast<int64_t>(loaded.value);
        if (v > 0) {
          if (is_and) {
            deps_satisfied = false;
            break;
          }
        } else {
          if (!is_and) {
            deps_satisfied = true;
            break;
          }
        }
      }
      if (dependency_access_failed)
        break;
      if (!has_deps)
        deps_satisfied = true;
      util::Logger::cp([&](auto &os) {
        os << std::format("BARRIER_DEP q={} type={} deps_ok={} has_deps={}", queue.queue_id,
                          is_and ? "AND" : "OR", deps_satisfied, has_deps);
      });
      if (!deps_satisfied) {
        // Stall this queue until the dependency signal is satisfied (by an SDMA
        // queue on this engine, or a peer rank's completion arriving via the
        // daemon). arm_stall_recheck() re-arms without spinning simulated time.
        process_limit = read_idx;
        arm_stall_recheck(now);
        break;
      }

      DispatchEntry dp{
          .dispatch_id = allocate_dispatch_id(),
          .queue_id = queue.queue_id,
          .address_space = queue.address_space,
          .interrupt_sink = queue.interrupt_sink,
          .process_id = queue.process_id,
          .completion_signal = barrier.completion_signal.handle,
          .kind = DispatchPacketKind::NonKernel,
          .barrier_bit = ((pkt.header >> HSA_PACKET_HEADER_BARRIER) & 1) != 0,
      };

      if (queue.xcd_fanout)
        replicate_non_kernel_entry(dp);
      qs.push_entry(std::move(dp));
    } else if (pkt_type == HSA_PACKET_TYPE_VENDOR_SPECIFIC) {
      AmdExtKernelDispatchPacket ext{};
      std::memcpy(&ext, &pkt, sizeof(ext));
      util::Logger::cp([&](auto &os) {
        os << std::format("VENDOR_PKT q={} amd_format={} dep_sig={:#x}", queue.queue_id,
                          ext.amd_format, ext.dep_signal.handle);
      });
      if (ext.amd_format == kHsaAmdPacketTypeBarrierValue) {
        AmdBarrierValuePacket barrier{};
        std::memcpy(&barrier, &pkt, sizeof(barrier));

        bool condition_satisfied = true;
        if (barrier.signal.handle != 0) {
          constexpr uint32_t SIG_VAL_OFF = 8;
          const AtomicLoadResult loaded = read_gpu_u64(
              queue.address_space, barrier.signal.handle + SIG_VAL_OFF, queue.process_id);
          if (loaded.outcome != VmAccessOutcome::Complete) {
            process_limit = read_idx;
            if (loaded.outcome == VmAccessOutcome::Unavailable)
              arm_stall_recheck(now);
            else
              queue.faulted = true;
            break;
          }
          const auto signal_value = std::bit_cast<int64_t>(loaded.value);
          const auto masked_value = signal_value & barrier.mask;
          switch (barrier.condition) {
          case HSA_SIGNAL_CONDITION_EQ:
            condition_satisfied = masked_value == barrier.value;
            break;
          case HSA_SIGNAL_CONDITION_NE:
            condition_satisfied = masked_value != barrier.value;
            break;
          case HSA_SIGNAL_CONDITION_LT:
            condition_satisfied = masked_value < barrier.value;
            break;
          case HSA_SIGNAL_CONDITION_GTE:
            condition_satisfied = masked_value >= barrier.value;
            break;
          default:
            throw std::runtime_error("Unsupported AMD barrier-value condition: " +
                                     std::to_string(barrier.condition));
          }
        }

        if (!condition_satisfied) {
          // The barrier-value packet stalls this queue until the awaited signal
          // reaches its target value (written by another queue on this engine or a
          // peer rank via the daemon). arm_stall_recheck() re-arms the re-check
          // without spinning simulated time.
          process_limit = read_idx;
          arm_stall_recheck(now);
          break;
        }

        DispatchEntry dp{
            .dispatch_id = allocate_dispatch_id(),
            .queue_id = queue.queue_id,
            .address_space = queue.address_space,
            .interrupt_sink = queue.interrupt_sink,
            .process_id = queue.process_id,
            .completion_signal = barrier.completion_signal.handle,
            .kind = DispatchPacketKind::NonKernel,
            .barrier_bit = ((barrier.header >> HSA_PACKET_HEADER_BARRIER) & 1) != 0,
        };

        if (queue.xcd_fanout)
          replicate_non_kernel_entry(dp);
        qs.push_entry(std::move(dp));
      } else if (ext.amd_format == kHsaAmdPacketTypeExtKernelDispatch) {
        if (ext.dep_signal.handle != 0) {
          constexpr uint32_t SIG_VAL_OFF = 8;
          const AtomicLoadResult loaded = read_gpu_u64(
              queue.address_space, ext.dep_signal.handle + SIG_VAL_OFF, queue.process_id);
          if (loaded.outcome != VmAccessOutcome::Complete) {
            process_limit = read_idx;
            if (loaded.outcome == VmAccessOutcome::Unavailable)
              arm_stall_recheck(now);
            else
              queue.faulted = true;
            break;
          }
          const auto v = static_cast<int64_t>(loaded.value);
          util::Logger::cp([&](auto &os) {
            os << std::format("VENDOR_DEP_CHECK q={} dep_sig={:#x} val={}", queue.queue_id,
                              ext.dep_signal.handle, v);
          });
          if (v != 0) {
            // Vendor kernel-dispatch dependency not yet satisfied: re-arm the
            // re-check via arm_stall_recheck() without spinning simulated time.
            process_limit = read_idx;
            arm_stall_recheck(now);
            break;
          }
        }

        uint32_t grid_size_x = checked_ext_dispatch_grid_size(
            ext.cluster_count_x, ext.cluster_size_x, ext.workgroup_size_x, "x");
        uint32_t grid_size_y = checked_ext_dispatch_grid_size(
            ext.cluster_count_y, ext.cluster_size_y, ext.workgroup_size_y, "y");
        uint32_t grid_size_z = checked_ext_dispatch_grid_size(
            ext.cluster_count_z, ext.cluster_size_z, ext.workgroup_size_z, "z");

        hsa_kernel_dispatch_packet_t dispatch{};
        dispatch.header = ext.header;
        dispatch.setup = ext.setup;
        dispatch.workgroup_size_x = ext.workgroup_size_x;
        dispatch.workgroup_size_y = ext.workgroup_size_y;
        dispatch.workgroup_size_z = ext.workgroup_size_z;
        dispatch.grid_size_x = grid_size_x;
        dispatch.grid_size_y = grid_size_y;
        dispatch.grid_size_z = grid_size_z;
        dispatch.private_segment_size = ext.private_segment_size;
        dispatch.group_segment_size = ext.group_segment_size;
        dispatch.kernel_object = ext.kernel_object;
        dispatch.kernarg_address = ext.kernarg_address;
        dispatch.completion_signal = ext.completion_signal;
        ClusterDispatchShape cluster_shape{};
        cluster_shape.count_x = ext.cluster_count_x;
        cluster_shape.count_y = ext.cluster_count_y;
        cluster_shape.count_z = ext.cluster_count_z;
        cluster_shape.size_x = ext.cluster_size_x;
        cluster_shape.size_y = ext.cluster_size_y;
        cluster_shape.size_z = ext.cluster_size_z;
        const VmAccessOutcome outcome =
            process_aql_packet(dispatch, queue, pkt_addr, slot, qs, read_idx, cluster_shape);
        if (outcome != VmAccessOutcome::Complete) {
          process_limit = read_idx;
          if (outcome == VmAccessOutcome::Unavailable)
            arm_stall_recheck(now);
          else
            queue.faulted = true;
          break;
        }
      } else if (ext.amd_format == kAmdAqlFormatPm4Ib) {
        DispatchEntry dp{
            .dispatch_id = allocate_dispatch_id(),
            .queue_id = queue.queue_id,
            .address_space = queue.address_space,
            .interrupt_sink = queue.interrupt_sink,
            .process_id = queue.process_id,
            .completion_signal = ext.completion_signal.handle,
            .kind = DispatchPacketKind::NonKernel,
            .barrier_bit = ((pkt.header >> HSA_PACKET_HEADER_BARRIER) & 1) != 0,
        };

        if (queue.xcd_fanout)
          replicate_non_kernel_entry(dp);
        qs.push_entry(std::move(dp));
      } else {
        throw std::runtime_error("Unsupported AMD vendor-specific AQL packet format: " +
                                 std::to_string(ext.amd_format));
      }
    }

    ++read_idx;
  }

  // Advance the CP-private cursor to match. read_ptr_va may subsequently be
  // lowered by the debugger to hold a trapped dispatch; the cursor is not, so
  // the next fetch resumes here rather than re-fetching held packets.
  queue.fetch_cursor = process_limit;

  VmAccessOutcome publication = VmAccessOutcome::Faulted;
  if (queue.address_space) {
    if (gpu_vm_ == nullptr) {
      queue.faulted = true;
      return;
    }
    std::optional<GpuVmAccess> access = gpu_vm_->snapshot(queue.address_space);
    if (!access) {
      queue.faulted = true;
      return;
    }
    queue.read_pointer_publication_access = std::make_shared<GpuVmAccess>(*access);
    queue.pending_read_pointer = process_limit;
    publication = queue.read_pointer_publication_access->atomic_store(
        queue.read_ptr_va, sizeof(queue.pending_read_pointer), queue.pending_read_pointer);
  } else if (memory_ != nullptr) {
    publication = memory_->atomic_store(queue.read_ptr_va, sizeof(process_limit), process_limit,
                                        queue.process_id) == AccessOutcome::Complete
                      ? VmAccessOutcome::Complete
                      : VmAccessOutcome::Faulted;
  }

  if (publication == VmAccessOutcome::Unavailable) {
    queue.read_pointer_publication_pending = true;
    arm_stall_recheck(now);
    return;
  }
  queue.read_pointer_publication_access.reset();
  if (publication != VmAccessOutcome::Complete)
    queue.faulted = true;
}

void CommandProcessor::handle_doorbell(simdojo::Tick now) {
  doorbell_handle_count_.fetch_add(1, std::memory_order_relaxed);
  // Release so the doorbell poll thread's acquire-load cannot observe a stale
  // "pending" after this handler has re-fetched; pairs with the release-stores at
  // the INVALID-packet and barrier/dependency stall sites. A site that is still
  // unsatisfied on this pass re-sets its flag below, re-arming the paced re-check.
  invalid_pending_.store(false, std::memory_order_release);
  stall_pending_.store(false, std::memory_order_release);
  // Apply terminal faults before accepting any later shards. Both drains keep
  // their inbox mutexes leaf locks and acquire hw_queue_mutex_ only after the
  // corresponding inbox has been released.
  drain_dispatch_fault_inbox();
  drain_fanout_inbox();
  drain_doorbell_inbox();

  std::unique_lock<std::recursive_mutex> lock(hw_queue_mutex_);
  util::Logger::cp(
      [&](auto &os) { os << std::format("{}: DOORBELL queues={}", name(), hw_queues_.size()); });

  // Finish the prior empty transition before admitting a new queue generation.
  // In particular, a transient mailbox write must not turn into a stale idle
  // interrupt after packets from a later doorbell have already been accepted.
  if (!drain_completions())
    return;

  size_t entries_before = 0;
  for (auto &qs : new_queue_states_)
    entries_before += qs.entries.size();

  // Fetch packets (uses last_doorbell values set by the poll thread).
  for (size_t i = 0; i < hw_queues_.size(); ++i) {
    if (!new_queue_states_[i].publication_retry_pending)
      fetch_from_queue(hw_queues_[i], new_queue_states_[i], now);
  }

  // Ensure interrupt callback is set on completion tracker.

  size_t entries_after = 0;
  for (auto &qs : new_queue_states_)
    entries_after += qs.entries.size();
  // Any new work restarts the stall re-check at a tight interval. Shard delivery
  // resets it too, but not every wait here ends with a shard -- a barrier or a
  // cross-rank dependency is satisfied by a value another partition (or the
  // daemon) writes, and that arrives as fetched entries rather than as an inbox
  // hand-off. Without this, a CP that had backed off would stay backed off.
  if (entries_after != entries_before)
    stall_recheck_backoff_ = 1;
  util::Logger::cp([&](auto &os) {
    os << std::format("{}: FETCHED {} new entries (total={})", name(),
                      entries_after - entries_before, entries_after);
  });

  // Phase 1: Dispatch-Execute-Complete loop (functional mode).
  bool progress = true;
  while (progress) {
    progress = false;

    for (size_t qi = 0; qi < hw_queues_.size(); ++qi) {
      if (hw_queues_[qi].faulted || hw_queues_[qi].is_sdma || hw_queues_[qi].debug_suspended ||
          hw_queues_[qi].runtime_suspended || new_queue_states_[qi].publication_retry_pending)
        continue;
      auto &qs = new_queue_states_[qi];

      while (qs.next_dispatch_idx < qs.entries.size()) {
        auto &entry = qs.entries[qs.next_dispatch_idx];

        if (entry.barrier_bit && !barrier_satisfied(qs, qs.next_dispatch_idx))
          break;

        if (entry.is_non_kernel()) {
          entry.completed_wgs = entry.total_wgs;
          ++qs.next_dispatch_idx;
          if (!drain_completions())
            break;
          if (qs.publication_retry_pending)
            break;
          progress = true;
          continue;
        }

        // Dispatch-execute-retire loop: keep dispatching WGs, activating CUs,
        // and retiring WFs until the entry is fully dispatched and completed,
        // or we hit genuine backpressure (no CU can accept any WG).
        // NOTE: drain_completions may pop entries, so we must re-check indices
        // after each drain and not hold stale references.
        uint32_t dispatch_id = entry.dispatch_id;
        bool backpressure = false;
        for (;;) {
          if (qs.next_dispatch_idx >= qs.entries.size())
            break;
          auto &cur = qs.entries[qs.next_dispatch_idx];
          if (cur.dispatch_id != dispatch_id)
            break;

          uint32_t sent = dispatch_workgroups(cur);
          if (sent > 0)
            progress = true;

          if (!drain_completions()) {
            backpressure = true;
            break;
          }
          if (qs.publication_retry_pending) {
            backpressure = true;
            break;
          }

          if (qs.next_dispatch_idx >= qs.entries.size())
            break;
          auto &post = qs.entries[qs.next_dispatch_idx];
          if (post.dispatch_id != dispatch_id)
            break;

          if (post.fully_dispatched()) {
            ++qs.next_dispatch_idx;
            break;
          }
          if (sent == 0) {
            backpressure = true;
            break;
          }
        }
        if (backpressure)
          break;
      }
    }
  }

  // Final drain: catch any entries that became fully_completed during the
  // last iteration but weren't drained by the re-entrant path.
  if (!drain_completions())
    return;

  util::Logger::cp([&](auto &os) {
    size_t remaining = 0;
    for (auto &qs : new_queue_states_)
      remaining += qs.entries.size();
    uint32_t active_cus = 0;
    for (auto *cu : cus_)
      if (cu->has_active_wfs())
        ++active_cus;
    os << std::format("{}: PHASE1_DONE remaining={} active_cus={}/{}", name(), remaining,
                      active_cus, cus_.size());
    for (size_t qi = 0; qi < new_queue_states_.size(); ++qi) {
      auto &qs = new_queue_states_[qi];
      if (qs.entries.empty())
        continue;
      os << std::format("\n  queue[{}] entries={} next_disp={} implicit_barrier={}", qi,
                        qs.entries.size(), qs.next_dispatch_idx, qs.implicit_barrier_next);
      for (size_t ei = 0; ei < qs.entries.size(); ++ei) {
        auto &e = qs.entries[ei];
        os << std::format(
            "\n    [{}] d={} qid={} total_wgs={} disp={} comp={} barrier={} sig={:#x} non_kern={}",
            ei, e.dispatch_id, e.queue_id, e.total_wgs, e.dispatched_wgs, e.completed_wgs,
            e.barrier_bit, e.completion_signal, e.is_non_kernel());
      }
    }
  });

  // Re-fetch: pick up any packets the host submitted while we were executing
  // (e.g., barrier packets queued after a kernel dispatch). Process them
  // immediately so host signal waits see completed barriers before returning.
  for (size_t i = 0; i < hw_queues_.size(); ++i) {
    if (!new_queue_states_[i].publication_retry_pending)
      fetch_from_queue(hw_queues_[i], new_queue_states_[i], now);
  }
  // Process any new non-kernel entries (barrier-kind packets).
  for (size_t qi = 0; qi < hw_queues_.size(); ++qi) {
    if (hw_queues_[qi].faulted || hw_queues_[qi].debug_suspended ||
        hw_queues_[qi].runtime_suspended || new_queue_states_[qi].publication_retry_pending)
      continue;
    auto &qs = new_queue_states_[qi];
    while (qs.next_dispatch_idx < qs.entries.size()) {
      auto &entry = qs.entries[qs.next_dispatch_idx];
      if (entry.barrier_bit && !barrier_satisfied(qs, qs.next_dispatch_idx))
        break;
      if (!entry.is_non_kernel())
        break;
      entry.completed_wgs = entry.total_wgs;
      ++qs.next_dispatch_idx;
      if (!drain_completions())
        break;
      if (qs.publication_retry_pending)
        break;
    }
  }

  arm_grid_wait_recheck();

  // Register as primary on first dispatch (internal test queues only).
  // KFD queues rely on the VM-level primary registered at rj_vm.cpp.
  bool kfd = has_kfd_queues();
  if (!is_primary_ && pending_entries() > 0 && !kfd) {
    engine()->register_as_primary();
    is_primary_ = true;
  }

  for (size_t i = 0; i < cus_.size(); ++i) {
    if (!cus_[i]->is_idle()) {
      if (dispatch_ports_[i]->link())
        dispatch_ports_[i]->send(std::make_unique<simdojo::Message>(simdojo::MessageHeader{}));
      else
        cus_[i]->schedule_work();
    }
  }

  bool all_done = completion_ && completion_->all_complete(new_queue_states_);
  bool should_release = all_done && is_primary_ && !kfd;

  util::Logger::cp([&](auto &os) {
    os << std::format("{}: TEARDOWN_CHECK all_done={} kfd={} primary={} release={}", name(),
                      all_done, kfd, is_primary_.load(), should_release);
  });

  // CRITICAL: must unlock before stop_doorbell_monitor() — the doorbell poll
  // thread takes hw_queue_mutex_ in scan_doorbells(); joining while holding
  // the lock would deadlock.
  lock.unlock();

  if (should_release) {
    stop_doorbell_monitor();
    engine()->primary_release();
    is_primary_ = false;
  }
}

void CommandProcessor::flush_gpu_caches() {
  // Both L1 caches are write-through, so discard their clean snapshots around
  // direct backing writes. Flush dirty L2 data before the direct write so a
  // later L2 flush cannot overwrite it.
  for (auto *cu : cus_)
    cu->l1_scalar().invalidate_all();
  for (auto *l2 : l2_caches_)
    l2->flush_all();
  for (auto *cu : cus_) {
    cu->l1_vector().invalidate_all();
    // A direct backing write may land on code, and the I$ is not coherent with
    // data writes any more than the hardware one is.
    cu->instruction_cache().invalidate_all();
  }
}

void CommandProcessor::invalidate_gpu_caches() {
  for (auto *l2 : l2_caches_)
    l2->invalidate_all();
  for (auto *cu : cus_) {
    cu->l1_vector().invalidate_all();
    cu->instruction_cache().invalidate_all();
  }
}

} // namespace amdgpu
} // namespace rocjitsu
