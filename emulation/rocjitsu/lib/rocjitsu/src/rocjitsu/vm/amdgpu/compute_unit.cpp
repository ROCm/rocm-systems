// Copyright (c) 2025-2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "rocjitsu/vm/amdgpu/compute_unit.h"
#include "rocjitsu/isa/arch/amdgpu/shared/flat_address.h"
#include "rocjitsu/isa/arch/amdgpu/shared/register_modifiers.h"

#include "rocjitsu/vm/amdgpu/gpu_vm.h"

#include "rocjitsu/code/analysis/waitcheck/target.h"
#include "rocjitsu/vm/amdgpu/async_scoreboard.h"
#include "rocjitsu/vm/amdgpu/command_processor.h"
#include "rocjitsu/vm/amdgpu/mma_admission.h"

#include "rocjitsu/isa/arch/amdgpu/cdna1/isa.h"
#include "rocjitsu/isa/arch/amdgpu/cdna2/isa.h"
#include "rocjitsu/isa/arch/amdgpu/cdna3/isa.h"
#include "rocjitsu/isa/arch/amdgpu/cdna3/sgpr_layout.h"
#include "rocjitsu/isa/arch/amdgpu/cdna4/isa.h"
#include "rocjitsu/isa/arch/amdgpu/cdna5/isa.h"
#include "rocjitsu/isa/arch/amdgpu/generated/shared/isa_properties.h"
#include "rocjitsu/isa/arch/amdgpu/rdna1/isa.h"
#include "rocjitsu/isa/arch/amdgpu/rdna2/isa.h"
#include "rocjitsu/isa/arch/amdgpu/rdna3/isa.h"
#include "rocjitsu/isa/arch/amdgpu/rdna3_5/isa.h"
#include "rocjitsu/isa/arch/amdgpu/rdna4/isa.h"
#include "rocjitsu/isa/arch/amdgpu/shared/alu_exceptions.h"
#include "rocjitsu/isa/arch/amdgpu/shared/ds_transpose.h"
#include "rocjitsu/isa/arch/amdgpu/shared/register_modifiers.h"
#include "rocjitsu/isa/arch/amdgpu/shared/scalar_operand_read.h"
#include "rocjitsu/isa/arch/amdgpu/shared/tensor_dma.h"
#include "rocjitsu/isa/instruction.h"
#include "rocjitsu/isa/target_registry.h"
#include "rocjitsu/vm/amdgpu/hwreg.h"
#include "rocjitsu/vm/amdgpu/matrix_coexecution.h"
#include "rocjitsu/vm/amdgpu/mem_state.h"
#include "rocjitsu/vm/amdgpu/register_access.h"
#include "rocjitsu/vm/amdgpu/texel_byte_range.h"
#include "rocjitsu/vm/plugins/execution_plugin.h"
#include "rocjitsu/vm/plugins/memory_access_observation.h"
#include "util/bit.h"
#include "util/log.h"

#include <algorithm>
#include <array>
#include <bit>
#include <cassert>
#include <limits>
#include <memory>
#include <span>
#include <stdexcept>

namespace rocjitsu {
namespace amdgpu {

std::optional<uint32_t> ComputeUnitCore::xcc_id() const {
  if (!cp_)
    return std::nullopt;
  return cp_->xcc_id();
}

bool InstructionComputeUnitView::update_trap_address(bool memory, uint64_t mask, uint64_t value) {
  const auto &update = raw_cu().trap_address_updater_;
  return update && update(raw_wavefront(), memory, mask, value);
}

bool InstructionComputeUnitView::in_multiwave_workgroup() const {
  const auto &wave = raw_wavefront();
  return raw_cu().barrier_wgs_.contains(wg_key(wave.dispatch_id(), wave.wg_id()));
}

bool InstructionComputeUnitView::signal_queue_exception(
    uint32_t queue_id, uint32_t process_id, uint64_t status, bool clear_debug_stop_on_success,
    bool retain_failure_for_debugger, QueueExceptionDeliveryPolicy delivery_policy,
    bool retain_wave) {
  return raw_cu().defer_queue_exception(retain_wave ? &raw_wavefront() : nullptr, queue_id,
                                        process_id, status, clear_debug_stop_on_success,
                                        retain_failure_for_debugger, delivery_policy);
}

uint32_t InstructionComputeUnitView::fetch_instruction_word(uint64_t address,
                                                            uint32_t process_id) const {
  if (process_id == 0)
    return raw_cu().memory()->fetch32(address);
  if (!raw_cu().gpu_vm())
    return 0;
  const auto access = raw_cu().gpu_vm()->snapshot_vmid(process_id);
  uint32_t word = 0;
  const auto bytes = std::span<std::byte>(reinterpret_cast<std::byte *>(&word), sizeof(word));
  return access && access->read(address, bytes, VmAccessKind::Execute) == VmAccessOutcome::Complete
             ? word
             : 0;
}

// The logical register access has already been observed. Accessing the
// shared physical storage here must not synthesize a second notification.
uint64_t Wavefront::read_backed_sgpr_pair(const SgprPair &pair) const {
  const uint32_t lo = cu_.read_sgpr_storage(pair.physical_base);
  const uint64_t hi =
      pair.backed_words == 2 ? cu_.read_sgpr_storage(pair.physical_base + 1) : pair.local >> 32;
  return (hi << 32) | lo;
}

void Wavefront::write_backed_sgpr_pair(SgprPair &pair, uint64_t value) {
  cu_.sgpr_file_[pair.physical_base] = static_cast<uint32_t>(value);
  if (pair.backed_words == 2)
    cu_.sgpr_file_[pair.physical_base + 1] = static_cast<uint32_t>(value >> 32);
  else
    pair.local = value;
}

uint32_t Wavefront::debug_sgpr_capacity() const {
  return sgpr_alloc_.count ? cu_.sgpr_allocation_block_size() : 0;
}

uint32_t Wavefront::debug_read_sgpr(uint32_t reg) const {
  if (reg > UINT32_MAX - sgpr_alloc_.base)
    return 0;
  return cu_.read_sgpr_storage(sgpr_alloc_.base + reg);
}

uint32_t Wavefront::debug_read_vgpr(uint32_t reg, uint32_t lane) const {
  return cu_.read_vgpr_storage_bounded(vgpr_alloc_.base + reg, lane);
}

void Wavefront::debug_write_sgpr(uint32_t reg, uint32_t value) {
  if (reg <= UINT32_MAX - sgpr_alloc_.base)
    cu_.write_sgpr(sgpr_alloc_.base + reg, value);
}

void Wavefront::debug_write_vgpr(uint32_t reg, uint32_t lane, uint32_t value) {
  cu_.write_vgpr(vgpr_alloc_.base + reg, lane, value);
}

void ComputeUnitCore::observe_scalar_register_read(const Wavefront &wf, RegisterRef reg) const {
  plugin_group_->onAmdgpuReadScalarRegister(&wf, reg);
}

void ComputeUnitCore::observe_scalar_register_write(const Wavefront &wf, RegisterRef reg) const {
  plugin_group_->onAmdgpuWriteScalarRegister(&wf, reg);
}

void ComputeUnitCore::observe_vgpr_read(const Wavefront *wf, uint32_t reg_idx, uint64_t lane_mask,
                                        uint8_t byte_mask) const {
  plugin_group_->onAmdgpuReadVgprLanes(wf, reg_idx, lane_mask, byte_mask);
}

void ComputeUnitCore::observe_vgpr_write(const Wavefront *wf, uint32_t reg_idx, uint64_t lane_mask,
                                         uint8_t byte_mask) const {
  plugin_group_->onAmdgpuWriteVgprLanes(wf, reg_idx, lane_mask, byte_mask);
}

namespace {
constexpr uint32_t kPrivilegedStatusBit = 1u << 5;

bool has_setreg_vgpr_msb_fixup(const ComputeUnitCore::Config &config) {
  const IsaTargetRegistry &registry = default_isa_target_registry();
  const IsaGpuTargetDescription *target = nullptr;
  if (config.target != ROCJITSU_CODE_TARGET_INVALID) {
    target = registry.find_gpu_target(config.target);
  } else if (const IsaTargetDescriptor *descriptor = registry.find(config.arch)) {
    target = registry.find_default_gpu_target(*descriptor);
  }
  return target != nullptr && target->capabilities.setreg_vgpr_msb_fixup;
}

bool is_privileged(const Wavefront &wf) { return (wf.status_raw() & kPrivilegedStatusBit) != 0; }

std::string_view instruction_execution_error_name(InstructionExecutionError error) {
  switch (error) {
  case InstructionExecutionError::None:
    return "none";
  case InstructionExecutionError::InvalidInstructionEncoding:
    return "invalid instruction encoding";
  case InstructionExecutionError::UnsupportedOperandValue:
    return "unsupported operand value";
  case InstructionExecutionError::MissingExecutionBackend:
    return "operand execution backend is not linked";
  case InstructionExecutionError::InvalidRegisterAccess:
    return "invalid register access";
  case InstructionExecutionError::UnimplementedInstruction:
    return "unimplemented instruction";

  case InstructionExecutionError::DecodeFailure:
    return "instruction decode failure";
  case InstructionExecutionError::IllegalInstruction:
    return "illegal instruction";
  }
  return "unknown instruction execution error";
}

uint32_t pack_barrier_state(uint32_t member_count, uint32_t signal_count,
                            uint32_t allocation_blocks = 0) {
  return 1u | ((member_count & 0x7fu) << 4) | ((signal_count & 0x7fu) << 16) |
         ((allocation_blocks & 0x7u) << 24);
}
} // namespace

template <GpuIsa Isa>
util::Result validate_compute_unit_config(const ComputeUnitCore::Config &config,
                                          util::DiagnosticEmitter emit_error) {
  using Limits = IsaExecComputeUnit<simdojo::ExecMode::FUNCTIONAL, Isa>;

  if (config.num_wf_slots > Isa::MAX_WF_SLOTS) {
    return emit_error.emit() << "num_wf_slots exceeds the ISA maximum of " << Isa::MAX_WF_SLOTS;
  }

  const uint32_t vgprs_per_block = Limits::effective_vgpr_allocation_block_size(config);
  if (vgprs_per_block > Limits::MAX_VGPRS_PER_BLOCK) {
    return emit_error.emit() << "effective VGPRs per wavefront exceeds the ISA maximum of "
                             << Limits::MAX_VGPRS_PER_BLOCK;
  }

  const uint64_t vgpr_file_registers = static_cast<uint64_t>(config.num_wf_slots) * vgprs_per_block;
  if (vgpr_file_registers > std::numeric_limits<uint32_t>::max() ||
      vgpr_file_registers > Limits::MAX_VGPR_FILE_REGISTERS) {
    return emit_error.emit() << "configured VGPR file exceeds the ISA maximum capacity";
  }
  return util::Result::success();
}

std::unique_ptr<Decoder> ComputeUnitCore::create_decoder(const Config &config) {
  return config.target == ROCJITSU_CODE_TARGET_INVALID
             ? Decoder::create(config.arch)
             : Decoder::create(default_isa_target_registry(), config.target);
}

std::string ComputeUnitCore::disassemble_instruction(std::span<const uint32_t> encoding,
                                                     uint64_t pc) const {
  Instruction::ScopedHeapAllocation heap_allocation;
  util::StringDiagnostic error;
  auto decoded = decoder_->decode_window(encoding, pc, error.emitter());
  if (decoded.failed())
    throw std::logic_error(std::format("cannot disassemble captured instruction at PC {:#x}: {}",
                                       pc, error.message()));
  return decoded.value()->disassemble();
}

bool InstructionComputeUnitView::deliver_sendmsg(Wavefront &wf, uint32_t message) {
  if (raw_cu().handle_sendmsg(wf, message))
    return true;
  if (arch() != ROCJITSU_CODE_ARCH_RDNA1 && arch() != ROCJITSU_CODE_ARCH_RDNA2)
    return true;
  // GFX10 SENDMSG IDs from the SOPP message table. An installed handler may
  // implement these operations for either a graphics queue or a compute trap.
  enum class Gfx10Message {
    Gs = 2,
    GsDone = 3,
    SaveWave = 4,
    StallWaveGen = 5,
    HaltWaves = 6,
    OrderedPsDone = 7,
    GsAllocReq = 9
  };
  switch (static_cast<Gfx10Message>(message & 0xfu)) {
  case Gfx10Message::Gs:
  case Gfx10Message::GsDone:
  case Gfx10Message::OrderedPsDone:
  case Gfx10Message::GsAllocReq:
    wf.report_instruction_execution_error(InstructionExecutionError::MissingExecutionBackend,
                                          "graphics message requires a graphics backend");
    return false;
  case Gfx10Message::SaveWave:
  case Gfx10Message::StallWaveGen:
  case Gfx10Message::HaltWaves:
    wf.report_instruction_execution_error(InstructionExecutionError::MissingExecutionBackend,
                                          "wave-control message requires a compute backend");
    return false;
  default:
    return true;
  }
}

util::FailureOr<uint64_t> InstructionComputeUnitView::return_message(uint32_t message) {
  const auto &cu = raw_cu();
  auto &wf = raw_wavefront();
  switch (message) {
  case 0x80: // GET_DOORBELL
    return cu.doorbell_resolver_ ? cu.doorbell_resolver_(wf).value_or(0xbad) : 0xbad;
  case 0x81: // GET_DDID
    return wf.dispatch_id();
  case 0x82: // GET_TMA
  case 0x85: // GET_TBA
    return trap_message_address(message);
  case 0x83: // GET_REALTIME
    return wf.realtime_timestamp();
  case 0x84: // SAVE_WAVE
    if (wf.in_trap_handler()) {
      // SPI must reserve a distinct GPU save record before the handler can store
      // its registers. The model has no per-wave context-save allocator.
      wf.report_instruction_execution_error(InstructionExecutionError::MissingExecutionBackend,
                                            "privileged SAVE_WAVE requires a CWSR record");
      return util::Result::failure();
    }
    [[fallthrough]]; // An ordinary shader's SAVE_WAVE becomes MSG_ILLEGAL_RTN.
  case 0xff:         // MSG_ILLEGAL_RTN
    return 0;
  default:
    return 0;
  }
}

ComputeUnitCore::ComputeUnitCore(std::string name, const Config &config, GpuMemory *memory,
                                 L2Cache *l2, uint32_t wf_size, uint32_t vgpr_storage_lane_count,
                                 uint32_t vgpr_allocation_block_size)
    : simdojo::CompositeComponent(std::move(name)), config_(config), arithmetic_(config.arithmetic),
      memory_(memory), wf_size_(wf_size), vgpr_storage_lane_count_(vgpr_storage_lane_count),
      vgpr_allocation_block_size_(vgpr_allocation_block_size),
      scratch_slots_per_cu_(config.num_wf_slots),
      setreg_vgpr_msb_fixup_(has_setreg_vgpr_msb_fixup(config)), decoder_(create_decoder(config)),
      l2_(l2), l1_scalar_(l2),
      l1_vector_(l2, memory ? memory->transaction_policy() : MemoryTransactionPolicy{}),
      lds_(config.lds_size_kb), scalar_mem_pipeline_(&l1_scalar_),
      global_mem_pipeline_(&l1_vector_, l2), local_mem_pipeline_() {
  // Admitted built-in architectures always provide a decoder.
  assert(decoder_ && "validated ComputeUnit architecture must provide a decoder");

  inst_cache_.set_l2(l2_);

  // Decoded instructions use heap storage: cache entries and asynchronous
  // work can outlive an issue quantum and move between execution workers.
  // Do not install this decoder's pool on the constructing thread.

  wfs_.resize(config.num_wf_slots);
  sgpr_file_.init(config.num_wf_slots * config.sgprs_per_wf, config.sgprs_per_wf);
  sgpr_block_owners_.resize(config.num_wf_slots);
  if (std::has_single_bit(config.sgprs_per_wf))
    sgpr_block_shift_ = std::countr_zero(config.sgprs_per_wf);

  // Completer port: CP sends dispatch activation messages here.
  cpl_ = add_port(std::make_unique<simdojo::Port>("cpl", 0, this, simdojo::PortDirection::IN,
                                                  simdojo::PortProtocol::DISPATCH));
  cpl_->set_handler([this](simdojo::Tick, simdojo::Message *) { schedule_work(); });

  // Requester port: structural connection to shared L2 cache.
  req_ = add_port(std::make_unique<simdojo::Port>("req", 1, this, simdojo::PortDirection::OUT,
                                                  simdojo::PortProtocol::MEMORY));

  MemoryPipeline::FaultHandler vm_fault_handler =
      [this](Wavefront &wavefront, VmAccessOutcome outcome, uint64_t issue_pc) {
        handle_memory_access_failure(wavefront, outcome, issue_pc);
      };
  scalar_mem_pipeline_.set_fault_handler(vm_fault_handler);
  global_mem_pipeline_.set_fault_handler(vm_fault_handler);
  tensor_dma_pipeline_.set_fault_handler(std::move(vm_fault_handler));
}

ComputeUnitCore::~ComputeUnitCore() {
  for (const auto &wave : wfs_)
    if (wave && wave->state() == WfState::GWS_WAIT)
      gws_device_->notify_parked_wave_gone(*wave);
}

void ComputeUnitCore::report_unresolved_apertures(uint32_t identity, ProcessIdentityKind kind) {
  bool guest_banks_programmed = false;
  {
    std::lock_guard lock(vmid_aperture_mutex_);
    guest_banks_programmed = std::any_of(vmid_apertures_.begin(), vmid_apertures_.end(),
                                         [](const auto &bank) { return bank.has_value(); });
  }
  if (!guest_banks_programmed ||
      unresolved_aperture_reported_.exchange(true, std::memory_order_relaxed))
    return;
  util::Logger::warn(
      std::format("{}: no unambiguous guest shader aperture for {} {} (retaining CU defaults)",
                  full_path(), kind == ProcessIdentityKind::Vmid ? "VMID" : "PASID", identity));
}

template <GpuIsa Isa>
static std::unique_ptr<ComputeUnitCore> create_functional_cu(std::string name,
                                                             const ComputeUnitCore::Config &config,
                                                             GpuMemory *memory, L2Cache *l2) {
  using Base = IsaExecComputeUnit<simdojo::ExecMode::FUNCTIONAL, Isa>;
  if constexpr (HasAsyncMma<Isa>) {
    // Select the virtual step entry once at construction. Configurations without
    // helpers need no runtime async check even on an eligible ISA.
    struct Asynchronous final : Base {
      using Base::Base;
      MmaAdmissionCache admission;
      bool step() override {
        return this->template step_impl<true>(&admission, Isa::ASYNC_MMA_WAVE_SIZE,
                                              HasAccVgpr<Isa>);
      }
    };
    if (config.async_resources && config.async_resources->helpers() &&
        async_mma_policy::supported(config.arch))
      return std::make_unique<Asynchronous>(std::move(name), config, memory, l2);
  }
  return std::make_unique<Base>(std::move(name), config, memory, l2);
}

namespace {
template <typename F>
util::Result with_compute_unit_isa(rj_code_arch_t arch, F &&fn,
                                   util::DiagnosticEmitter emit_error) {
  switch (arch) {
    // \NPI new ISA family: add its architecture-to-ISA mapping here.
  case ROCJITSU_CODE_ARCH_CDNA1:
    return fn.template operator()<cdna1::Isa>();
  case ROCJITSU_CODE_ARCH_CDNA2:
    return fn.template operator()<cdna2::Isa>();
  case ROCJITSU_CODE_ARCH_CDNA3:
    return fn.template operator()<cdna3::Isa>();
  case ROCJITSU_CODE_ARCH_CDNA4:
    return fn.template operator()<cdna4::Isa>();
  case ROCJITSU_CODE_ARCH_RDNA1:
    return fn.template operator()<rdna1::Isa>();
  case ROCJITSU_CODE_ARCH_RDNA2:
    return fn.template operator()<rdna2::Isa>();
  case ROCJITSU_CODE_ARCH_RDNA3:
    return fn.template operator()<rdna3::Isa>();
  case ROCJITSU_CODE_ARCH_RDNA3_5:
    return fn.template operator()<rdna3_5::Isa>();
  case ROCJITSU_CODE_ARCH_RDNA4:
    return fn.template operator()<rdna4::Isa>();
  case ROCJITSU_CODE_ARCH_CDNA5:
    return fn.template operator()<cdna5::Isa>();
  default:
    return emit_error.emit() << "Unsupported architecture for ComputeUnit";
  }
}
} // namespace

util::Result ComputeUnitCore::validate(const Config &config, util::DiagnosticEmitter emit_error) {
  if (config.target != ROCJITSU_CODE_TARGET_INVALID) {
    const IsaTargetRegistry &registry = default_isa_target_registry();
    const IsaTargetDescriptor *target_descriptor = registry.find(config.target);
    const IsaGpuTargetDescription *target_binding = registry.find_gpu_target(config.target);
    if (target_descriptor == nullptr || target_binding == nullptr)
      return emit_error.emit() << "unsupported concrete GPU target";
    if (target_descriptor->architecture_id != config.arch)
      return emit_error.emit()
             << "concrete GPU target does not belong to the configured architecture";
    if (!target_descriptor->supports_execution ||
        !target_binding->capabilities.execution_implemented)
      return emit_error.emit() << "execution is not implemented for the concrete GPU target";
  }

  return with_compute_unit_isa(
      config.arch,
      [&]<GpuIsa Isa>() { return validate_compute_unit_config<Isa>(config, emit_error); },
      emit_error);
}

std::unique_ptr<ComputeUnitCore> ComputeUnitCore::create(std::string name, const Config &config,
                                                         GpuMemory *memory, L2Cache *l2,
                                                         simdojo::ExecMode exec_mode,
                                                         util::DiagnosticEmitter emit_error) {
  if (validate(config, emit_error).failed())
    return nullptr;
  std::unique_ptr<ComputeUnitCore> compute_unit;
  const util::Result result = with_compute_unit_isa(
      config.arch,
      [&]<GpuIsa Isa>() -> util::Result {
        switch (exec_mode) {
        case simdojo::ExecMode::FUNCTIONAL:
          compute_unit = create_functional_cu<Isa>(std::move(name), config, memory, l2);
          return util::Result::success();
        case simdojo::ExecMode::CLOCKED:
          compute_unit = std::make_unique<IsaExecComputeUnit<simdojo::ExecMode::CLOCKED, Isa>>(
              std::move(name), config, memory, l2);
          return util::Result::success();
        }
        return emit_error.emit() << "Unsupported execution mode for ComputeUnit";
      },
      emit_error);
  if (result.failed())
    return nullptr;
  return compute_unit;
}

Wavefront *ComputeUnitCore::dispatch_wf(uint32_t wg_id, uint64_t pc, uint32_t num_sgprs,
                                        uint32_t num_vgprs, uint32_t wave_size,
                                        uint32_t scratch_wave_limit_per_se) {
  std::lock_guard<std::recursive_mutex> wave_state_lock(wave_state_mutex_);
  assert(wfs_.size() == config_.num_wf_slots && "wavefront slots not properly initialized");
  // Halted wavefronts have already freed their SGPR/VGPR blocks at s_endpgm, so a
  // halted slot is immediately available, as is a slot never materialized.
  size_t slot = config_.num_wf_slots;
  const size_t slot_limit = scratch_wave_limit_per_se == UINT32_MAX
                                ? wfs_.size()
                                : std::min<size_t>(wfs_.size(), scratch_slots_per_cu_);
  for (size_t i = 0; i < slot_limit; ++i) {
    const uint64_t scratch_scoreboard_id = static_cast<uint64_t>(scratch_scoreboard_base_) + i;
    if (scratch_scoreboard_id < scratch_wave_limit_per_se && (!wfs_[i] || wfs_[i]->is_halted())) {
      slot = i;
      break;
    }
  }

  // No free slot: fail the dispatch (like the register-allocation failures below)
  // rather than indexing wfs_ out of bounds. The CP normally gates placement on
  // can_accept_workgroup(), but returning nullptr is part of this API's contract
  // and must hold even when a caller dispatches directly to a full CU.
  if (slot >= config_.num_wf_slots)
    return nullptr;

  return dispatch_wf_at(static_cast<uint32_t>(slot), wg_id, pc, num_sgprs, num_vgprs, wave_size);
}

Wavefront *ComputeUnitCore::dispatch_wf_at(uint32_t wf_id, uint32_t wg_id, uint64_t pc,
                                           uint32_t num_sgprs, uint32_t num_vgprs,
                                           uint32_t wave_size) {
  return dispatch_wf_at(wf_id, wg_id, pc, num_sgprs, num_vgprs, wave_size, std::nullopt);
}

Wavefront *ComputeUnitCore::dispatch_wf_at(uint32_t wf_id, uint32_t wg_id, uint64_t pc,
                                           uint32_t num_sgprs, uint32_t num_vgprs,
                                           uint32_t wave_size,
                                           std::optional<uint32_t> register_block) {
  std::lock_guard<std::recursive_mutex> wave_state_lock(wave_state_mutex_);
  assert(wfs_.size() == config_.num_wf_slots && "wavefront slots not properly initialized");
  if (wf_id >= config_.num_wf_slots || (wfs_[wf_id] && !wfs_[wf_id]->is_halted()))
    return nullptr;

  if (!wfs_[wf_id])
    wfs_[wf_id] = create_wavefront(wf_id);
  auto *wf = wfs_[wf_id].get();
  wf->activity_tracked_ = true;
  const uint32_t dispatched_wave_size = wave_size == 0 ? wf->default_wf_size_ : wave_size;
  if ((dispatched_wave_size != 32 && dispatched_wave_size != 64) ||
      dispatched_wave_size > wf->max_wf_size_)
    return nullptr;

  int32_t sgpr_base = sgpr_file_.allocate(num_sgprs, register_block);
  if (sgpr_base < 0)
    return nullptr;

  int32_t vgpr_base =
      register_block ? allocate_vgprs_at(num_vgprs, *register_block) : allocate_vgprs(num_vgprs);
  if (vgpr_base < 0) {
    sgpr_file_.free(static_cast<uint32_t>(sgpr_base));
    return nullptr;
  }

  // Invalidate the L1 scalar cache so this wavefront reads fresh kernel
  // arguments from L2/memory rather than stale lines from a prior kernel.
  // On real hardware, the driver issues s_dcache_inv at kernel launch.
  l1_scalar_.invalidate_all();

  wf->wf_size_ = dispatched_wave_size;
  wf->wg_id_ = wg_id;
  wf->pc = pc;
  wf->sgpr_alloc_ = {static_cast<uint32_t>(sgpr_base), num_sgprs};
  wf->vgpr_alloc_ = {static_cast<uint32_t>(vgpr_base), num_vgprs};
  wf->num_sgprs_ = num_sgprs;
  wf->num_vgprs_ = num_vgprs;
  wf->set_accum_vgpr_base(0);
  wf->exec_ = wf->lane_mask();
  wf->vgpr_write_mask_ = wf->lane_mask();
  wf->vcc_ = {};
  wf->scratch_base_ = {};
  if (arch() == ROCJITSU_CODE_ARCH_CDNA3) {
    const uint32_t end = cdna3::sgpr_alias_end(num_sgprs);
    auto bind_pair = [&](Wavefront::SgprPair &pair, uint32_t first) {
      if (first < config_.sgprs_per_wf) {
        pair.physical_base = wf->sgpr_alloc_.base + first;
        pair.backed_words = std::min(2u, config_.sgprs_per_wf - first);
      }
    };
    bind_pair(wf->vcc_, cdna3::vcc_base(end));
    bind_pair(wf->scratch_base_, cdna3::flat_scratch_base(end));
    wf->memory_wait_shadow_.set_sgpr_alias_end(end);
  }
  wf->m0_ = 0;
  wf->set_shader_engine_id(shader_engine_id_);
  wf->set_scratch_scoreboard_id(scratch_scoreboard_base_ + wf_id);
  wf->set_status_raw(0);
  wf->flat_pointer32_ = flat_pointer32_;
  wf->set_apertures(shared_aperture_base_, shared_aperture_limit_, private_aperture_base_,
                    private_aperture_limit_, /*override_launch=*/false);
  ++wf->dispatch_generation_;
  if (wf->dispatch_generation_ == 0)
    ++wf->dispatch_generation_;
  wf->set_state(WfState::RUNNING);
  wf->set_ready_cycle(cycle_counter_);
  wf->trace_inst_count_ = 0;

  sgpr_block_owners_[static_cast<uint32_t>(sgpr_base) / config_.sgprs_per_wf] = {
      wf, static_cast<uint32_t>(sgpr_base) + config_.sgprs_per_wf};
  set_vgpr_block_owner(static_cast<uint32_t>(vgpr_base), wf);

  util::Logger::cp([&](auto &os) {
    os << "DISPATCH_WF cu=" << full_path() << " wf=" << wf->wf_id() << " slot=" << wf_id << " pc=0x"
       << std::hex << pc << std::dec << " wg=" << wg_id << " pid=" << wf->process_id();
  });

  if (wave_reservation_handler_)
    wave_reservation_handler_(*wf);
  schedule_work();
  return wf;
}

size_t ComputeUnitCore::num_wfs() const {
  std::lock_guard<std::recursive_mutex> wave_state_lock(wave_state_mutex_);
  size_t count = 0;
  for (const auto &w : wfs_)
    if (w && !w->is_halted())
      ++count;
  return count;
}

void ComputeUnitCore::free_wavefront_resources(Wavefront &wf) {
  std::lock_guard<std::recursive_mutex> wave_state_lock(wave_state_mutex_);
  scalar_mem_pipeline_.cancel(wf);
  global_mem_pipeline_.cancel(wf);
  local_mem_pipeline_.cancel(wf);
  tensor_dma_pipeline_.cancel(wf);
  if (wf.sgpr_alloc().count > 0) {
    sgpr_block_owners_[wf.sgpr_alloc().base / config_.sgprs_per_wf] = {};
    sgpr_file_.free(wf.sgpr_alloc().base);
    free_vgprs(wf.vgpr_alloc().base);
  }
  // A wave halted or aborted while parked at a GWS rendezvous never went through
  // the store's wake path, so reconcile the device-global parked bookkeeping here
  // while the wave is still in GWS_WAIT -- reset() below clears its park state and
  // process id (see GwsDevice).
  if (wf.state() == WfState::GWS_WAIT)
    gws_device_->notify_parked_wave_gone(wf);
  wf.trace_inst_count_ = 0;
  wf.reset();
}

void ComputeUnitCore::flush_cp_notifications() {
  // Loops because a notification can retire more work and queue another
  // notification behind it; draining to empty keeps that from waiting for whatever
  // takes the wave-state lock next.
  for (;;) {
    bool capacity_changed = false;
    std::vector<PendingExecutionFailure> execution_failures;
    std::vector<PendingDispatchFailure> failures;
    std::vector<PendingQueueException> exceptions;
    std::vector<std::pair<uint32_t, uint32_t>> ready;
    {
      std::lock_guard<std::recursive_mutex> wave_state_lock(wave_state_mutex_);
      if (!has_pending_cp_notifications())
        return;
      failures.swap(pending_dispatch_failures_);
      execution_failures.swap(pending_execution_failures_);
      exceptions.swap(pending_queue_exceptions_);
      ready.swap(pending_wg_completions_);
      capacity_changed = std::exchange(pending_capacity_change_, false);
    }
    // The lock is released here, so taking hw_queue_mutex_ below cannot invert
    // against the CP's dispatch path.
    if (!cp_)
      return;
    for (const auto &failure : execution_failures)
      cp_->notify_execution_failure(failure.queue_id, failure.process_id, failure.reason);
    for (const PendingDispatchFailure &failure : failures) {
      if (failure.vm_outcome)
        cp_->notify_dispatch_vm_fault(failure.queue_id, failure.process_id, failure.dispatch_id,
                                      *failure.vm_outcome);
      else
        cp_->notify_dispatch_failure(failure.queue_id, failure.process_id, failure.dispatch_id);
    }
    for (const auto &exception : exceptions) {
      const bool delivered =
          queue_exception_handler_
              ? queue_exception_handler_(exception.queue_id, exception.process_id, exception.status,
                                         exception.retain_failure_for_debugger,
                                         exception.delivery_policy)
              : cp_->signal_queue_exception(exception.queue_id, exception.process_id,
                                            exception.status, /*publish_interrupt=*/false) &&
                    cp_->publish_queue_exception(
                        exception.queue_id, exception.process_id, exception.status,
                        exception.delivery_policy ==
                            QueueExceptionDeliveryPolicy::WaitForAcknowledgment);
      if (!delivered && exception.delivery_policy == QueueExceptionDeliveryPolicy::PublishOrFail) {
        if (auto *sim_engine = engine())
          sim_engine->request_exit(
              std::format("Failed to deliver queue exception {:#x} for process {} queue {}",
                          exception.status, exception.process_id, exception.queue_id),
              /*code=*/1);
      }
      if (delivered && exception.wave != nullptr) {
        std::lock_guard<std::recursive_mutex> wave_state_lock(wave_state_mutex_);
        // Queue teardown can reclaim this slot while the external handler is
        // waiting for acknowledgement. Only commit ownership to the wave that
        // originally queued the exception.
        if (exception.wave->process_id() == exception.process_id &&
            exception.wave->queue_id() == exception.queue_id) {
          exception.wave->add_trap_runtime_exception_status(exception.status);
          if (exception.clear_debug_stop_on_success) {
            exception.wave->set_debug_halted(false);
            exception.wave->set_status_halt(false);
          }
        }
      }
    }
    for (const auto &[dispatch_id, wg_id] : ready)
      cp_->notify_wg_complete(dispatch_id, wg_id);
    // A wave can free enough registers/slots for another workgroup before its
    // own workgroup retires. Retry only after dropping the wave-state lock.
    if (capacity_changed && cp_->dispatch_threads() <= 1 && engine())
      engine()->schedule_event_now(cp_->doorbell_event());
  }
}

bool ComputeUnitCore::enter_trap_handler(Wavefront &wave, uint32_t trap_id) {
  if (wave.in_trap_handler() || !trap_handler_resolver_)
    return false;
  const auto config = trap_handler_resolver_(wave);
  if (!config || !config->tba)
    return false;
  enter_trap_handler(wave, *config, trap_id, wave.pc);
  return true;
}

void ComputeUnitCore::deliver_memory_trap(Wavefront &wave) {
  assert(wave.state() == WfState::MEMORY_TRAP);
  wave.set_state(WfState::RUNNING);
  // Hardware saves the next PC and a zero software trap ID (RDNA2 §3.11).
  if (enter_trap_handler(wave, 0))
    return;
  wave.set_status_raw(wave.status_raw() | Wavefront::kStatusFatalHaltMask);
  if (wave.fail_pm4_submission()) {
    abort_dispatch(wave.dispatch_id());
    return;
  }
  defer_queue_exception(nullptr, wave.queue_id(), wave.process_id(), kAqlQueueApertureViolation,
                        /*clear_debug_stop_on_success=*/false,
                        /*retain_failure_for_debugger=*/false,
                        QueueExceptionDeliveryPolicy::PublishOrFail);
}

void ComputeUnitCore::enter_trap_handler(Wavefront &wave, const TrapHandlerConfig &config,
                                         uint32_t trap_id, uint64_t saved_pc) {
  if (arch() == ROCJITSU_CODE_ARCH_CDNA2 && config_.memory_wait_checks_enabled()) {
    if (auto *scoreboard = wave.memory_wait_scoreboard())
      scoreboard->clear();
  }
  const uint32_t saved_status = wave.status_raw();
  const auto properties = isa_properties(this->arch());
  const auto wave_state_layout = properties.wave_state_layout;
  const bool uses_split_wave_state = wave_state_layout != WaveStateLayout::Legacy;
  wave.set_ttmp(0, static_cast<uint32_t>(saved_pc));
  if (uses_split_wave_state) {
    // GFX12 trap entry carries the four-bit trap id in TTMP1[31:28].
    // GFX12.0 has a 48-bit PC and preserves SCHED_MODE in TTMP1[27:26];
    // GFX12.5 expands the PC to 57 bits and enters privileged scheduling.
    const uint32_t pc_hi_mask =
        wave_state_layout == WaveStateLayout::Gfx12_5 ? 0x01FFFFFFu : 0x0000FFFFu;
    const uint32_t sched_mode = wave_state_layout == WaveStateLayout::Gfx12
                                    ? (wave.wave_sched_mode_raw() & 0x3u) << 26
                                    : 0u;
    wave.set_ttmp(1, (static_cast<uint32_t>(saved_pc >> 32) & pc_hi_mask) | sched_mode |
                         ((trap_id & 0xFu) << 28));
    const uint32_t debug_enabled = config.debug_enabled ? (1u << 23) : 0u;
    wave.set_ttmp(11, (wave.ttmp(11) & ~(1u << 23)) | debug_enabled);
    constexpr uint16_t kWholeStatePriv = 4u | (31u << 11);
    uint32_t state_priv = 0;
    const auto state_result = read_hwreg_field(wave, kWholeStatePriv, state_priv);
    assert(state_result == HwregAccessResult::Success);
    (void)state_result;
    wave.set_ttmp(12, state_priv);
  } else {
    wave.set_ttmp(1, static_cast<uint32_t>(saved_pc >> 32) | (trap_id << 16));
  }
  // Dispatch identity. Which TTMPs carry it is architecture-specific and
  // this must not disagree with what CWSR publishes for the same wave, or
  // rocm-dbgapi correlates the stopped wave to the wrong workgroup.
  //
  // On the profiles where the SPI puts workgroup ids in TTMP6/7/9,
  // init_wavefront_regs() already seeded them at dispatch and trap entry
  // has nothing to add -- zeroing TTMP9 here destroyed one of them. The
  // rest use the gfx9 layout, TTMP8/9/10 = workgroup id x/y/z, which is
  // also what CWSR serializes (cwsr.cpp writes wg_coord into ttmp[8..10]).
  // Writing the flat wg_id() into TTMP8 disagreed with that too.
  if (!properties.uses_ttmp_workgroup_ids) {
    const auto &wg = wave.wg_coord();
    wave.set_ttmp(8, wg[0]);
    wave.set_ttmp(9, wg[1]);
    wave.set_ttmp(10, wg[2]);
  }
  if (!uses_split_wave_state) {
    wave.set_ttmp(11,
                  ((wave.queue_packet_id() & 0x1FFFFFFu) << 6) | (wave.wave_in_group() & 0x3Fu));
    wave.set_ttmp(12, saved_status);
    const uint32_t debug_enabled = config.debug_enabled ? (1u << 23) : 0u;
    wave.set_ttmp(13, (wave.ttmp(13) & ~(1u << 23)) | debug_enabled);
  }
  wave.set_ttmp(14, static_cast<uint32_t>(config.tma));
  wave.set_ttmp(15, static_cast<uint32_t>(config.tma >> 32));
  wave.set_trap_id(trap_id);
  wave.set_trap_saved_status(saved_status);
  wave.set_trap_saved_exec(wave.exec());
  wave.set_trap_interrupt_sent(false);
  wave.clear_trap_exception_status();
  // A fresh handler entry owns the halt state from here on; a marker left
  // over from a previous stop would attribute this entry's HALT to an
  // s_sendmsghalt that has already been resumed past.
  wave.set_self_halted(false);
  wave.set_in_trap_handler(true);
  if (wave.supports_sleep_wakeup_credit())
    wave.set_sleep_cycles(0);
  if (uses_split_wave_state)
    wave.set_status_raw(saved_status | kPrivilegedStatusBit);
  wave.pc = config.tba;
  if (plugin_group_)
    plugin_group_->onAmdgpuTrapEntered(wave);
}

void ComputeUnitCore::handle_memory_access_failure(Wavefront &wf, VmAccessOutcome outcome,
                                                   uint64_t issue_pc) {
  assert(outcome != VmAccessOutcome::Complete && outcome != VmAccessOutcome::Unavailable);
  // Consume the pipeline's marker now so a claimed fault cannot leave it behind.
  const char *unreported = wf.take_unreported_memory_failure();
  if (wf.instruction_execution_failed()) {
    fail_wave_instruction(
        wf, "memory instruction", issue_pc, wf.instruction_execution_error(),
        wf.instruction_execution_error_detail() ? wf.instruction_execution_error_detail() : "");
    return;
  }
  if (!unreported && wf.address_space() && !wf.snapshot_vm_access())
    unreported = "address space retired during the access";
  if (wf.fail_pm4_submission()) {
    abort_dispatch(wf.dispatch_id());
    return;
  }
  // Only architectural data faults belong to the runtime exception owner, and
  // only once a VM fault reporter saw them: KFD publishes them as a memory
  // violation, which ROCr pairs with that reporter's VM fault event.
  // Fetch faults are offered once at the fetch site, with their real address.
  if (outcome == VmAccessOutcome::Faulted && !unreported && memory_violation_handler_ &&
      memory_violation_handler_(wf, 0, false, issue_pc))
    return;
  handle_terminal_vm_fault(wf, outcome, issue_pc, unreported);
}

void ComputeUnitCore::handle_terminal_vm_fault(Wavefront &wf, VmAccessOutcome outcome,
                                               uint64_t issue_pc, const char *unreported) {
  assert(outcome != VmAccessOutcome::Complete && outcome != VmAccessOutcome::Unavailable);
  if (wf.fail_pm4_submission()) {
    abort_dispatch(wf.dispatch_id());
    return;
  }
  pending_dispatch_failures_.push_back({.queue_id = wf.queue_id(),
                                        .process_id = wf.process_id(),
                                        .dispatch_id = wf.dispatch_id(),
                                        .vm_outcome = outcome});
  // A revoked snapshot's reporter must not run, and VMID 0 physical memory has
  // none. Cancelling the dispatch alone, or publishing a memory-violation status
  // that waits for a VM fault event, would leave the completion waiter hanging.
  if (!unreported && outcome == VmAccessOutcome::Revoked)
    unreported = "address space revoked during the access";
  if (!unreported) {
    const std::optional<GpuVmAccess> access = wf.snapshot_vm_access();
    if (!access || !access->reports_faults())
      unreported = "memory fault without a VM fault reporter";
  }
  if (unreported) {
    util::Logger::warn("memory access failed: ", unreported, ": process=", wf.process_id(),
                       " queue=", wf.queue_id(), " dispatch=", wf.dispatch_id(), " cu=", name(),
                       " wave=", wf.wf_id(), " pc=", std::format("{:#x}", issue_pc));
    defer_queue_exception(nullptr, wf.queue_id(), wf.process_id(), kAqlQueueWaveAbort,
                          /*clear_debug_stop_on_success=*/false,
                          /*retain_failure_for_debugger=*/false);
  }
  abort_dispatch(wf.dispatch_id());
}

void ComputeUnitCore::fail_gws_wait(Wavefront &wf, std::string_view reason) {
  util::Logger::warn(reason, ": process=", wf.process_id(), " queue=", wf.queue_id(),
                     " dispatch=", wf.dispatch_id(), " cu=", name(), " wave=", wf.wf_id(),
                     " resource=", wf.gws_wait_rid_, " pc=", std::format("{:#x}", wf.pc),
                     " gws_wait_limit=", config_.gws_wait_limit);
  const uint32_t dispatch_id = wf.dispatch_id();
  const bool pm4_failed = wf.fail_pm4_submission();
  if (!pm4_failed && !(execution_failure_handler_ && execution_failure_handler_(wf))) {
    defer_queue_exception(nullptr, wf.queue_id(), wf.process_id(), kAqlQueueWaveAbort,
                          /*clear_debug_stop_on_success=*/false,
                          /*retain_failure_for_debugger=*/false,
                          QueueExceptionDeliveryPolicy::PublishAndCancel);
  }
  if (cp_ && engine())
    pending_execution_failures_.push_back({wf.queue_id(), wf.process_id(), std::string(reason)});
  abort_dispatch(dispatch_id);
  if (!cp_ || !engine()) {
    if (engine())
      engine()->request_exit(std::string(reason), 1);
    else if (!pm4_failed)
      throw std::runtime_error(std::string(reason));
  }
}

uint64_t InstructionComputeUnitView::trap_message_address(uint32_t message) const {
  const auto [tba, tma] = trap_addresses();
  if (message == 0x82)
    return tma >> 8; // GET_TMA supplies the address in 256-byte units.
  return tba ? tba | (uint64_t{1} << 63) : 0;
}

void ComputeUnitCore::maybe_reset_lds_alloc() {
  if (!has_active_wfs() && !lds_allocation_pinned())
    reset_lds_alloc();
}

void ComputeUnitCore::begin_workgroup(uint32_t dispatch_id, uint32_t wg_id, uint32_t wf_count,
                                      uint32_t num_named_barriers) {
  // The driver's s_icache_inv rides the launch packet, so it lands once per
  // dispatch, not once per wave: a kernel VA reused by a later dispatch still
  // sees fresh code, while the sibling waves of one dispatch keep filling a
  // shared I$ instead of cold-starting each other.
  if (inst_cache_dispatch_id_ != dispatch_id) {
    inst_cache_.invalidate_all();
    inst_cache_dispatch_id_ = dispatch_id;
  }

  const uint64_t key = wg_key(dispatch_id, wg_id);
  active_wgs_[key] = wf_count;
  if (wf_count <= 1) {
    // Single-wave workgroups are not allocated workgroup or named barriers.
    barrier_wgs_.erase(key);
    return;
  }

  auto &group = barrier_wgs_[key];
  group = {};
  group.allocated_count = std::min(num_named_barriers, kMaxNamedBarriers);
  for (auto &barrier : group.workgroup)
    barrier.member_count = wf_count;
}

std::vector<ComputeUnitCore::Rdna4CheckpointGroup>
ComputeUnitCore::rdna4_checkpoint_groups() const {
  std::vector<Rdna4CheckpointGroup> result;
  for (const auto &[key, members] : active_wgs_) {
    const auto it = barrier_wgs_.find(key);
    const bool multiwave = it != barrier_wgs_.end();
    result.push_back({wg_key_dispatch_id(key), static_cast<uint32_t>(key), members,
                      static_cast<uint8_t>(multiwave ? it->second.workgroup[0].signal_count : 0),
                      static_cast<uint8_t>(multiwave ? it->second.workgroup[1].signal_count : 0),
                      multiwave});
  }
  std::ranges::sort(result, {},
                    [](const Rdna4CheckpointGroup &g) { return wg_key(g.dispatch_id, g.wg_id); });
  return result;
}

void ComputeUnitCore::restore_rdna4_checkpoint_group(const Rdna4CheckpointGroup &group) {
  const auto key = wg_key(group.dispatch_id, group.wg_id);
  active_wgs_.emplace(key, group.live_members);
  if (group.multiwave) {
    auto &barriers = barrier_wgs_[key].workgroup;
    barriers[0] = {group.live_members, group.workgroup_signals};
    barriers[1] = {group.live_members, group.trap_signals};
  }
}

void ComputeUnitCore::named_barrier_init(Wavefront &wf, int32_t barrier_id, uint32_t member_count) {
  auto group = barrier_wgs_.find(wg_key(wf.dispatch_id(), wf.wg_id()));
  if (group == barrier_wgs_.end() || barrier_id <= 0 ||
      static_cast<uint32_t>(barrier_id) > group->second.allocated_count)
    return;

  auto &barrier = group->second.named[static_cast<uint32_t>(barrier_id)];
  if (member_count != 0)
    barrier.member_count = member_count & 0x7fu;
  barrier.signal_count = 0;
}

void ComputeUnitCore::named_barrier_join(Wavefront &wf, int32_t barrier_id) {
  auto group = barrier_wgs_.find(wg_key(wf.dispatch_id(), wf.wg_id()));
  if (group == barrier_wgs_.end())
    return;
  if (barrier_id == 0) {
    wf.named_barrier_id_ = 0;
    wf.barrier_complete_[kNamedBarrierBit] = false;
    if (wf.waiting_barrier_bit_ == kNamedBarrierBit)
      wf.waiting_barrier_bit_ = Wavefront::kNoBarrierWait;
    return;
  }
  if (barrier_id < 0 || static_cast<uint32_t>(barrier_id) > group->second.allocated_count)
    return;
  wf.named_barrier_id_ = static_cast<uint32_t>(barrier_id);
  wf.barrier_complete_[kNamedBarrierBit] = false;
  if (wf.waiting_barrier_bit_ == kNamedBarrierBit)
    wf.waiting_barrier_bit_ = Wavefront::kNoBarrierWait;
}

bool ComputeUnitCore::barrier_signal(Wavefront &wf, int32_t barrier_id, uint32_t member_count) {
  if (barrier_id == kClusterBarrierId || barrier_id == kClusterTrapBarrierId) {
    if (barrier_id == kClusterTrapBarrierId && !is_privileged(wf))
      return false;
    return cp_ ? cp_->cluster_barrier_signal(wf, barrier_id) : false;
  }

  auto group = barrier_wgs_.find(wg_key(wf.dispatch_id(), wf.wg_id()));
  if (group == barrier_wgs_.end() || barrier_id == 0 || barrier_id < kWorkgroupTrapBarrierId)
    return false;

  BarrierCounter *barrier = nullptr;
  uint8_t completion_bit = kNamedBarrierBit;
  uint32_t named_id = 0;
  if (barrier_id > 0) {
    named_id = static_cast<uint32_t>(barrier_id);
    if (named_id > group->second.allocated_count)
      return false;
    barrier = &group->second.named[named_id];
    if (member_count != 0)
      barrier->member_count = member_count & 0x7fu;
  } else {
    if (barrier_id == kWorkgroupTrapBarrierId && !is_privileged(wf))
      return false;
    completion_bit = static_cast<uint8_t>(-barrier_id);
    barrier = &group->second.workgroup[completion_bit - kWorkgroupBarrierBit];
  }

  if (barrier->member_count == 0)
    return false;
  const bool is_first = barrier->signal_count == 0;
  barrier->signal_count = std::min(barrier->signal_count + 1, 0x7fu);
  if (barrier->signal_count < barrier->member_count)
    return is_first;

  barrier->signal_count = 0;
  auto members = complete_barrier(wf.dispatch_id(), wf.wg_id(), completion_bit, named_id);
  notify_barrier_complete(members, completion_bit);
  return is_first;
}

std::vector<Wavefront *> ComputeUnitCore::complete_barrier(uint32_t dispatch_id, uint32_t wg_id,
                                                           uint8_t completion_bit,
                                                           uint32_t named_barrier_id) {
  std::vector<Wavefront *> members;
  for (const auto &candidate : wfs_) {
    if (!candidate || candidate->is_halted() || candidate->dispatch_id() != dispatch_id ||
        candidate->wg_id() != wg_id)
      continue;
    if (completion_bit == kNamedBarrierBit && candidate->named_barrier_id_ != named_barrier_id)
      continue;
    candidate->barrier_complete_[completion_bit] = true;
    members.push_back(candidate.get());
  }
  for (auto *member : members) {
    if (member->state() == WfState::BARRIER && member->waiting_barrier_bit_ == completion_bit) {
      member->barrier_complete_[completion_bit] = false;
      member->waiting_barrier_bit_ = Wavefront::kNoBarrierWait;
      member->set_state(WfState::RUNNING);
      member->set_ready_cycle(cycle_counter_);
    }
  }
  return members;
}

void ComputeUnitCore::notify_barrier_complete(std::span<Wavefront *> members,
                                              uint8_t completion_bit) {
  if (members.empty())
    return;
  BarrierKind kind;
  switch (completion_bit) {
  case kNamedBarrierBit:
    kind = BarrierKind::Named;
    break;
  case kWorkgroupBarrierBit:
  case kWorkgroupTrapBarrierBit:
    kind = BarrierKind::Workgroup;
    break;
  case kClusterBarrierBit:
  case kClusterTrapBarrierBit:
    kind = BarrierKind::Cluster;
    break;
  default:
    assert(false && "unknown barrier completion bit");
    return;
  }
  plugin_group_->onAmdgpuBarrierResolved(members, kind);
}

uint32_t ComputeUnitCore::barrier_state(const Wavefront &wf, int32_t barrier_id) const {
  auto group = barrier_wgs_.find(wg_key(wf.dispatch_id(), wf.wg_id()));
  const uint32_t allocation_blocks =
      group == barrier_wgs_.end() ? 0 : (group->second.allocated_count + 3) / 4;

  if (barrier_id == kClusterBarrierId || barrier_id == kClusterTrapBarrierId) {
    if (barrier_id == kClusterTrapBarrierId && !is_privileged(wf))
      return 0;
    return cp_ ? cp_->cluster_barrier_state(wf, barrier_id, allocation_blocks) : 0;
  }

  if (group == barrier_wgs_.end() || barrier_id == 0 || barrier_id < kWorkgroupTrapBarrierId)
    return 0;

  if (barrier_id < 0) {
    if (barrier_id == kWorkgroupTrapBarrierId && !is_privileged(wf))
      return 0;
    const auto &barrier = group->second.workgroup[static_cast<uint32_t>(-barrier_id - 1)];
    return pack_barrier_state(barrier.member_count, barrier.signal_count, allocation_blocks);
  }

  const uint32_t id = static_cast<uint32_t>(barrier_id);
  if (id > group->second.allocated_count)
    return 0;
  const auto &barrier = group->second.named[id];
  return pack_barrier_state(barrier.member_count, barrier.signal_count, allocation_blocks);
}

void ComputeUnitCore::barrier_wait(Wavefront &wf, int32_t barrier_id) {
  uint8_t completion_bit = kNamedBarrierBit;
  if (barrier_id >= 0) {
    auto group = barrier_wgs_.find(wg_key(wf.dispatch_id(), wf.wg_id()));
    if (group == barrier_wgs_.end() || wf.named_barrier_id_ == 0 ||
        wf.named_barrier_id_ > group->second.allocated_count)
      return;
  } else {
    if (barrier_id < kClusterTrapBarrierId ||
        ((barrier_id == kWorkgroupTrapBarrierId || barrier_id == kClusterTrapBarrierId) &&
         !is_privileged(wf)))
      return;
    completion_bit = static_cast<uint8_t>(-barrier_id);
    if (completion_bit <= kWorkgroupTrapBarrierBit) {
      if (!barrier_wgs_.contains(wg_key(wf.dispatch_id(), wf.wg_id())))
        return;
    } else if (!cp_ || !cp_->cluster_barrier_valid(wf, barrier_id)) {
      return;
    }
  }

  if (wf.barrier_complete_[completion_bit]) {
    wf.barrier_complete_[completion_bit] = false;
    return;
  }
  wf.waiting_barrier_bit_ = completion_bit;
  wf.set_state(WfState::BARRIER);
}

bool ComputeUnitCore::named_barrier_leave(Wavefront &wf) {
  auto group = barrier_wgs_.find(wg_key(wf.dispatch_id(), wf.wg_id()));
  const uint32_t id = wf.named_barrier_id_;
  if (group == barrier_wgs_.end())
    return false;
  if (id == 0)
    return true;
  if (id > group->second.allocated_count)
    return false;

  wf.named_barrier_id_ = 0;
  wf.barrier_complete_[kNamedBarrierBit] = false;
  if (wf.waiting_barrier_bit_ == kNamedBarrierBit)
    wf.waiting_barrier_bit_ = Wavefront::kNoBarrierWait;
  auto &barrier = group->second.named[id];
  if (barrier.member_count != 0)
    --barrier.member_count;
  if (barrier.signal_count >= barrier.member_count) {
    barrier.signal_count = 0;
    auto members = complete_barrier(wf.dispatch_id(), wf.wg_id(), kNamedBarrierBit, id);
    notify_barrier_complete(members, kNamedBarrierBit);
  }
  return barrier.member_count == 0;
}

// Global Wave Sync (GWS) is device-global and shared across the CUs that run a
// dispatch; all state and scheduling policy live in GwsDevice (see its class doc
// in gws_device.h for the authoritative description). These forward to the store
// this CU participates in.
void ComputeUnitCore::gws_init(Wavefront &wf, uint32_t rid, uint32_t count) {
  gws_device_->init(wf, rid, count);
}

void ComputeUnitCore::gws_barrier_arrive(Wavefront &wf, uint32_t rid, uint32_t count) {
  if (!gws_device_->barrier_arrive(*this, wf, rid, count)) {
    wf.gws_wait_rid_ = rid;
    fail_gws_wait(wf, "GWS barrier used before initialization");
  }
}

void ComputeUnitCore::gws_sema_v(Wavefront &wf, uint32_t rid) {
  gws_device_->sema_v(*this, wf, rid);
}

void ComputeUnitCore::gws_sema_p(Wavefront &wf, uint32_t rid) {
  gws_device_->sema_p(*this, wf, rid);
}

void ComputeUnitCore::gws_sema_br(Wavefront &wf, uint32_t rid, uint32_t count) {
  gws_device_->sema_br(*this, wf, rid, count);
}

void ComputeUnitCore::gws_sema_release_all(Wavefront &wf, uint32_t rid) {
  gws_device_->sema_release_all(*this, wf, rid);
}

void ComputeUnitCore::release_wf(uint32_t dispatch_id, uint32_t wg_id,
                                 Wavefront::CpCompletionNotice notice) {
  if (cp_ && notice == Wavefront::CpCompletionNotice::Send)
    pending_capacity_change_ = true;
  auto key = wg_key(dispatch_id, wg_id);
  auto group = barrier_wgs_.find(key);
  if (group != barrier_wgs_.end()) {
    const auto retire_member = [&](BarrierCounter &barrier, uint8_t completion_bit,
                                   uint32_t joined_id = 0) {
      if (barrier.member_count == 0)
        return;
      --barrier.member_count;
      if (barrier.member_count == 0 || barrier.signal_count < barrier.member_count)
        return;
      barrier.signal_count = 0;
      auto members = complete_barrier(dispatch_id, wg_id, completion_bit, joined_id);
      notify_barrier_complete(members, completion_bit);
    };

    retire_member(group->second.workgroup[0], kWorkgroupBarrierBit);
    retire_member(group->second.workgroup[1], kWorkgroupTrapBarrierBit);
  }

  auto it = active_wgs_.find(key);
  if (it != active_wgs_.end() && --it->second == 0) {
    active_wgs_.erase(it);
    barrier_wgs_.erase(key);
    // Queued rather than sent: notify_wg_complete() takes the CP's
    // hw_queue_mutex_, and this runs under the wave-state lock, which the CP
    // takes in the other order when it dispatches. WaveStateGuard delivers it
    // once the lock is dropped.
    if (cp_ && notice == Wavefront::CpCompletionNotice::Send)
      pending_wg_completions_.emplace_back(dispatch_id, wg_id);
  }
  // The whole workgroup's per-WG LDS region can be reclaimed once the CU has fully
  // drained and no cluster peer can still multicast into it.
  maybe_reset_lds_alloc();
}

void ComputeUnitCore::abort_workgroup(uint32_t dispatch_id, uint32_t wg_id) {
  std::lock_guard<std::recursive_mutex> wave_state_lock(wave_state_mutex_);
  // Roll back a workgroup that was committed via begin_workgroup() but whose peers
  // in the same clustered dispatch failed to fully dispatch. Unlike release_wf(),
  // this fires cancellation hooks but no CP completion notify. Free any
  // resident (not-yet-halted) waves belonging to this WG, drop the refcount entry,
  // and reclaim LDS if the CU is now idle and unpinned. The caller unpins the cluster
  // LDS separately (the pin is CP-side bookkeeping).
  for (const auto &wavefront : wfs_) {
    if (wavefront && !wavefront->is_halted() && wavefront->dispatch_id() == dispatch_id &&
        wavefront->wg_id() == wg_id) {
      plugin_group().onAmdgpuWavefrontCancelled(*wavefront);
      free_wavefront_resources(*wavefront);
    }
  }
  active_wgs_.erase(wg_key(dispatch_id, wg_id));
  barrier_wgs_.erase(wg_key(dispatch_id, wg_id));
  maybe_reset_lds_alloc();
}

void ComputeUnitCore::handle_instruction_failure(Wavefront &wf, const std::string &failure,
                                                 bool exception_claimed) {
  util::Logger::warn(failure);
  if (!wf.fail_pm4_submission()) {
    // Cancellation resets the wave before deferred exception publication. The
    // frontend may own the notification, but must not retain this incarnation.
    // A claimed illegal-instruction notification must not also publish queue abort.
    if (!exception_claimed && execution_failure_handler_)
      (void)execution_failure_handler_(wf);
    pending_dispatch_failures_.push_back({.queue_id = wf.queue_id(),
                                          .process_id = wf.process_id(),
                                          .dispatch_id = wf.dispatch_id(),
                                          .vm_outcome = std::nullopt});
    if (auto *sim_engine = engine())
      sim_engine->request_exit(failure, /*code=*/1);
  }
  // halt() follows the successful workgroup-completion path. A rejected
  // instruction instead cancels resident waves and their pending memory work.
  abort_dispatch(wf.dispatch_id());
}

void ComputeUnitCore::abort_dispatch(uint32_t dispatch_id) {
  std::lock_guard<std::recursive_mutex> wave_state_lock(wave_state_mutex_);
  for (const auto &wavefront : wfs_) {
    if (wavefront && !wavefront->is_halted() && wavefront->dispatch_id() == dispatch_id) {
      plugin_group().onAmdgpuWavefrontCancelled(*wavefront);
      free_wavefront_resources(*wavefront);
    }
  }

  std::erase_if(active_wgs_, [dispatch_id](const auto &entry) {
    return wg_key_dispatch_id(entry.first) == dispatch_id;
  });
  std::erase_if(barrier_wgs_, [dispatch_id](const auto &entry) {
    return wg_key_dispatch_id(entry.first) == dispatch_id;
  });
  std::erase_if(pending_wg_completions_,
                [dispatch_id](const auto &completion) { return completion.first == dispatch_id; });
  maybe_reset_lds_alloc();
}

bool ComputeUnitCore::can_accept_workgroup(uint32_t num_wfs, uint32_t lds_bytes,
                                           uint32_t scratch_wave_limit_per_se) const {
  // Reject exhausted register allocations before scanning the wave slots.
  // Check SGPR register blocks.
  uint32_t free_sgpr = sgpr_file_.free_block_count();
  if (free_sgpr < num_wfs) {
    util::Logger::vm("CU ", this->name(), " can_accept_wg: REJECT free_sgpr=", free_sgpr,
                     " < num_wfs=", num_wfs);
    return false;
  }

  // Check VGPR register blocks.
  uint32_t free_vgpr = free_vgpr_blocks();
  if (free_vgpr < num_wfs) {
    util::Logger::vm("CU ", this->name(), " can_accept_wg: REJECT free_vgpr=", free_vgpr,
                     " < num_wfs=", num_wfs);
    return false;
  }

  // Unrestricted placement can reuse the resident count. Paused waves still
  // occupy slots; scratch-limited placement must inspect the eligible prefix.
  uint32_t free_slots = 0;
  if (scratch_wave_limit_per_se == UINT32_MAX &&
      uint64_t{scratch_scoreboard_base_} + wfs_.size() <= UINT32_MAX) {
    const uint32_t active = static_cast<uint32_t>(wave_activity_.load(std::memory_order_acquire));
    assert(active <= wfs_.size());
    free_slots = static_cast<uint32_t>(wfs_.size()) - active;
  } else {
    const size_t slot_limit = scratch_wave_limit_per_se == UINT32_MAX
                                  ? wfs_.size()
                                  : std::min<size_t>(wfs_.size(), scratch_slots_per_cu_);
    for (size_t slot = 0; slot < slot_limit && free_slots < num_wfs; ++slot) {
      const uint64_t scratch_scoreboard_id = static_cast<uint64_t>(scratch_scoreboard_base_) + slot;
      if (scratch_scoreboard_id < scratch_wave_limit_per_se &&
          (!wfs_[slot] || wfs_[slot]->is_halted()))
        ++free_slots;
    }
  }
  if (free_slots < num_wfs) {
    util::Logger::vm("CU ", this->name(), " can_accept_wg: REJECT free_slots=", free_slots,
                     " < num_wfs=", num_wfs);
    return false;
  }

  if (lds_bytes > 0) {
    const uint64_t aligned = lds_allocation_bytes(arch(), lds_bytes);
    uint32_t lds_capacity_bytes = config_.lds_size_kb * 1024u;
    if (next_lds_alloc_ + aligned > lds_capacity_bytes) {
      return false;
    }
  }

  return true;
}

void ComputeUnitCore::retry_paused_queue_memory(uint32_t process_id, uint32_t queue_id,
                                                bool replay_loads) {
  WaveStateGuard guard(*this);
  auto pending = [&](const auto &wave) {
    return wave && wave->process_id() == process_id && wave->queue_id() == queue_id &&
           (wave->runtime_suspended() || wave->debug_stopped()) && !wave->wait_counters().empty();
  };
  if (std::ranges::any_of(wfs_, pending)) {
    scalar_mem_pipeline_.retry_for_queue(process_id, queue_id);
    global_mem_pipeline_.retry_for_queue(process_id, queue_id);
    local_mem_pipeline_.retry_for_queue(process_id, queue_id);
    tensor_dma_pipeline_.retry_for_queue(process_id, queue_id);
    // Publish the continuation/handler state before a paused queue is inspected for CWSR.
    for (const auto &wave : wfs_)
      if (wave && wave->process_id() == process_id && wave->queue_id() == queue_id &&
          wave->state() == WfState::MEMORY_TRAP)
        deliver_memory_trap(*wave);
  }
  if (replay_loads) {
    for (const auto &wave : wfs_) {
      if (!pending(wave))
        continue;
      // Unavailable loads have not committed register writeback or memory
      // side effects. Save their issuing PC and replay after debugger resume.
      // Side-effecting operations must finish before a context can be saved.
      (void)(scalar_mem_pipeline_.rewind_unavailable_load(*wave) ||
             global_mem_pipeline_.rewind_unavailable_load(*wave));
    }
  }
}

void ComputeUnitCore::tick_pipelines() {
  // Functional accesses normally complete at issue. Retry only pipelines
  // with retained work, avoiding four out-of-line calls on scalar-only steps.
  if (!scalar_mem_pipeline_.empty())
    scalar_mem_pipeline_.tick();
  if (!global_mem_pipeline_.empty())
    global_mem_pipeline_.tick();
  if (!local_mem_pipeline_.empty())
    local_mem_pipeline_.tick();
  if (!tensor_dma_pipeline_.empty())
    tensor_dma_pipeline_.tick();
}

// GFX11 ISA 3.3.3. Nullify before debugger probes, memory side effects, and writeback.
void ComputeUnitCore::qualify_gfx11_memory(Instruction &inst, const Wavefront &wf) {
  if (inst.data()->tag() != GLOBAL_MEM && inst.data()->tag() != LOCAL_MEM)
    return;
  auto &d = *inst.data_as<VectorMemState>();
  if (nullify_unallocated_memory_result(inst, wf))
    return;
  // Native DS addresses already include the workgroup base. Preserve active lanes
  // so a suppressed read writes zeros even for the second destination of a dual load.
  if (d.tag() == LOCAL_MEM && !d.lds_stack_inputs) {
    const uint32_t bytes = d.elem_size * d.num_elems;
    for (uint64_t lanes = d.lane_mask; lanes; lanes &= lanes - 1) {
      const unsigned lane = std::countr_zero(lanes);
      if (d.atomic_op != AtomicOp::NONE && d.elem_size != 0 &&
          ((d.per_lane_addr[lane] % d.elem_size) ||
           (d.ds2_active && d.ds2_per_lane_addr[lane] % d.elem_size)))
        d.ds_memviol = true;
      const bool primary = wf.contains_lds_range(d.per_lane_addr[lane], bytes);
      const bool secondary =
          !d.ds2_active || wf.contains_lds_range(d.ds2_per_lane_addr[lane], bytes);
      if (!primary || (d.is_load && !secondary))
        d.ds_oob_lane_mask |= uint64_t{1} << lane;
      if (d.ds2_active && (!secondary || (d.is_load && !primary)))
        d.ds2_oob_lane_mask |= uint64_t{1} << lane;
    }
  }
}

bool ComputeUnitCore::nullify_unallocated_memory_result(Instruction &inst, const Wavefront &wf) {
  const auto tag = inst.data()->tag();
  if (tag != GLOBAL_MEM && tag != LOCAL_MEM)
    return false;
  auto &d = *inst.data_as<VectorMemState>();
  // Block transfers check each enabled DWORD against the allocation independently.
  if (!d.is_load || d.lds_dst || d.is_block_transfer)
    return false;
  const auto &allocation = wf.vgpr_alloc();
  uint32_t words = d.destination_vgpr_count() + d.buffer_tfe;
  // On GFX12 the VBUFFER status register still participates in allocation
  // when the data fits exactly. LDS and FLAT have no such encoded status.
  if (wf.cu().arch() == ROCJITSU_CODE_ARCH_RDNA4 &&
      d.dst_reg_base + words == allocation.base + allocation.count) {
    RegisterModifiers modifiers;
    inst.amdgpu_register_modifiers(modifiers);
    words += modifiers.buffer_tfe;
  }
  if (allocation.contains(d.dst_reg_base, words) &&
      (!d.ds2_active || allocation.contains(d.ds2_dst_reg_base, d.ds2_destination_vgpr_count())))
    return false;
  d.exec_mask = d.lane_mask = 0;
  d.element_lane_masks.clear();
  if (d.image_sample)
    for (auto &tap : d.image_sample->taps)
      tap.lane_mask = 0;
  return true;
}

void ComputeUnitCore::qualify_cdna4_memory_destination(Instruction &inst, const Wavefront &wf) {
  if (inst.data()->tag() != GLOBAL_MEM && inst.data()->tag() != LOCAL_MEM)
    return;
  auto &d = *inst.data_as<VectorMemState>();
  if (!d.is_load || d.lds_dst)
    return;
  // CDNA4 section 3.6.1 nullifies the whole access, including a returning
  // atomic's memory update. Check before debugger probes and routing. AGPRs
  // use a separate simulator bank and have no per-wave extent here.
  const uint32_t base = d.dst_reg_base;
  // Both results of a CDNA4 dual DS operation occupy one consecutive tuple.
  const uint32_t count =
      d.destination_vgpr_count() + (d.ds2_active ? d.ds2_destination_vgpr_count() : 0);
  if (!owns_memory_result_vgprs(wf, base, count)) {
    d.exec_mask = d.lane_mask = 0;
    d.element_lane_masks.clear();
  }
}

VmAccessOutcome ComputeUnitCore::route_memory_inst(Instruction *inst, Wavefront &wf) {
  if (arch() == ROCJITSU_CODE_ARCH_CDNA4)
    qualify_cdna4_memory_destination(*inst, wf);
  if (arch() == ROCJITSU_CODE_ARCH_RDNA3 || arch() == ROCJITSU_CODE_ARCH_RDNA3_5)
    qualify_gfx11_memory(*inst, wf);
  if (arch() == ROCJITSU_CODE_ARCH_RDNA4)
    nullify_unallocated_memory_result(*inst, wf);
  return route_prepared_memory_inst(inst, wf);
}

// Only gfx1250 partitions the shared aperture into local and distributed LDS.
static bool flat_offset_is_dds(rj_code_arch_t arch, uint64_t offset) {
  return arch == ROCJITSU_CODE_ARCH_CDNA5 && (offset & (uint64_t{1} << 31));
}

// FLAT-to-LDS atomic alignment faults the whole wave before global translation.
// GFX1250's distributed shared segment has a distinct atomic path.
static bool flat_shared_atomic_misaligned(rj_code_arch_t arch, const VectorMemState &state,
                                          uint64_t shared_offset) {
  const bool checks_alignment =
      arch == ROCJITSU_CODE_ARCH_RDNA3 || arch == ROCJITSU_CODE_ARCH_RDNA3_5 ||
      arch == ROCJITSU_CODE_ARCH_RDNA4 ||
      (arch == ROCJITSU_CODE_ARCH_CDNA5 && !flat_offset_is_dds(arch, shared_offset));
  return checks_alignment && state.atomic_op != AtomicOp::NONE && state.elem_size &&
         shared_offset % state.elem_size != 0;
}

VmAccessOutcome ComputeUnitCore::route_prepared_memory_inst(Instruction *inst, Wavefront &wf) {
  if (observes_memory_instruction_routing_)
    plugin_group_->onAmdgpuRouteMemoryInstruction(*inst, wf);
  const bool observe_routed_access =
      observes_memory_routing_ && plugin_group_->observes_memory_routing(wf);
  const uint8_t decoded_route_tag = inst->data()->tag();
  bool normalized_to_local = false;
  uint64_t flat_local_lane_mask = 0;
  uint64_t flat_dds_lane_mask = 0;
  std::array<uint64_t, 64> pre_routing_address_storage;
  std::span<const uint64_t> pre_routing_addresses;
  const uint64_t shared_base =
      wf.use_wave_apertures() ? wf.shared_aperture_base() : shared_aperture_base_;
  const uint64_t shared_limit =
      wf.use_wave_apertures() ? wf.shared_aperture_limit() : shared_aperture_limit_;
  if (inst->data()->tag() == GLOBAL_MEM && inst->mnemonic().starts_with("flat_") &&
      shared_base != 0) {
    auto &d = *inst->data_as<VectorMemState>();
    // The wavefront is the authority on its own width. VectorMemState::wf_size
    // is set by whoever built the state, and the DS paths leave it at its
    // default until the local pipeline backfills it -- which is after this.
    const uint32_t wf_size = wf.wf_size();
    const uint64_t scratch_lanes = d.scratch_swizzle ? d.scratch_lane_mask : 0;
    for (uint32_t lane = 0; lane < wf_size; ++lane) {
      const uint64_t lane_bit = uint64_t{1} << lane;
      if ((d.lane_mask & ~scratch_lanes & lane_bit) != 0) {
        const uint64_t address = d.per_lane_addr[lane];
        const uint64_t aperture_pointer =
            address - (flat_selects_segment_before_offset(arch()) ? d.flat_aperture_offset : 0);
        if (flat_in_shared_aperture(aperture_pointer, shared_base, shared_limit)) {
          if (flat_offset_is_dds(arch(), address - shared_base))
            flat_dds_lane_mask |= lane_bit;
          else
            flat_local_lane_mask |= lane_bit;
        }
      }
    }
    // Normalize shared addresses once for both routes. The global operation
    // excludes those lanes, and retries reuse this state without rerouting.
    const uint64_t request_lanes = transpose_request_lane_mask(d, wf_size);
    const uint64_t flat_shared_lane_mask = flat_local_lane_mask | flat_dds_lane_mask;
    d.flat_local_lane_mask = flat_local_lane_mask & request_lanes;
    d.flat_dds_lane_mask = flat_dds_lane_mask & request_lanes;
    d.flat_local_oob_lane_mask = 0;
    d.lds_allocation.reset();
    if (d.flat_local_lane_mask) {
      d.lds_allocation = LdsAllocationRange::from_size(wf.lds_base(), wf.lds_size());
      d.flat_local_allocation_end = d.lds_allocation->end;
    }
    if (flat_shared_lane_mask != 0) {
      if (observe_routed_access) {
        std::ranges::copy_n(d.per_lane_addr.begin(), wf_size, pre_routing_address_storage.begin());
        pre_routing_addresses = {pre_routing_address_storage.data(), wf_size};
      }
      for (uint32_t lane = 0; lane < wf_size; ++lane) {
        if (flat_shared_lane_mask & (uint64_t{1} << lane)) {
          uint64_t offset = d.per_lane_addr[lane] - shared_base;
          // RDNA4 FLAT uses unsigned 17-bit LDS address math. Keep the
          // workgroup LDS base outside that wrapping calculation.
          if (arch() == ROCJITSU_CODE_ARCH_RDNA4)
            offset &= 0x1FFFFu;
          if (d.flat_local_allocation_end &&
              (offset >= wf.lds_size() || d.elem_size > wf.lds_size() - offset))
            d.flat_local_oob_lane_mask |= (uint64_t{1} << lane) & request_lanes;
          if (arch() == ROCJITSU_CODE_ARCH_RDNA1 || arch() == ROCJITSU_CODE_ARCH_RDNA2) {
            const uint64_t bytes = uint64_t{d.elem_size} * d.num_elems;
            if (!util::is_range_in_bounds(offset, bytes, wf.lds_size()))
              d.flat_local_oob_lane_mask |= (uint64_t{1} << lane) & request_lanes;
          }
          d.per_lane_addr[lane] = offset + wf.lds_base();
          const bool gfx11 =
              arch() == ROCJITSU_CODE_ARCH_RDNA3 || arch() == ROCJITSU_CODE_ARCH_RDNA3_5;
          if ((d.flat_local_lane_mask & (uint64_t{1} << lane)) &&
              flat_shared_atomic_misaligned(arch(), d, offset))
            d.ds_memviol = true;
          if (gfx11) {
            if (!wf.contains_lds_range(d.per_lane_addr[lane], d.elem_size * d.num_elems))
              d.flat_local_oob_lane_mask |= (uint64_t{1} << lane) & request_lanes;
          }
          if (d.lds_allocation && d.elem_size && (d.flat_local_lane_mask & (uint64_t{1} << lane))) {
            const uint32_t bytes = d.num_elems * d.elem_size;
            const auto retained =
                d.lds_allocation->retained_bytes(d.per_lane_addr[lane], bytes, d.elem_size);
            if (retained == 0)
              d.flat_local_oob_lane_mask |= uint64_t{1} << lane;
            if (arch() == ROCJITSU_CODE_ARCH_CDNA2 && retained < bytes)
              d.flat_memory_violation = true;
          }
        }
      }
    }
    if (request_lanes != 0 && (request_lanes & ~flat_shared_lane_mask) == 0) {
      inst->data()->set_tag(LOCAL_MEM);
      d.wait_counter_type = WaitCounterType::LGKMCNT;
      const auto *issue = inst->amdgpu_memory_issue_info();
      if (issue) {
        for (const auto obligation : issue->counter_obligations()) {
          if (obligation.completion_class() == MemoryCompletionClass::LDS) {
            d.wait_counter_type = obligation.wait_counter_type();
            break;
          }
        }
      }
      normalized_to_local = true;
    }
  }

  if (decoded_route_tag == GLOBAL_MEM && inst->mnemonic().starts_with("flat_")) {
    auto &d = *inst->data_as<VectorMemState>();
    if (const auto *issue = inst->amdgpu_memory_issue_info()) {
      const uint64_t requests = transpose_request_lane_mask(d, wf.wf_size());
      const uint64_t shared = (flat_local_lane_mask | flat_dds_lane_mask) & requests;
      d.routed_issue_info = issue->for_flat_memory_domains((requests & ~shared) != 0, shared != 0);
    }
  }

  const uint8_t route_tag = inst->data()->tag();
  // After the aperture rewrite, before the pipeline takes the instruction:
  // this is the one point at which the space, the counter, and the addresses
  // are all the ones the memory system is about to use. Gated on a plugin that
  // wants it rather than on any plugin at all, because building the
  // observation is real work on the per-instruction path and most plugins have
  // no use for it.
  if (route_tag == LOCAL_MEM && !normalized_to_local) {
    auto &data = *inst->data_as<VectorMemState>();
    if (native_ds_atomic_misaligned(data, arch(), transpose_request_lane_mask(data, wf.wf_size())))
      data.ds_memviol = true;
  }
  if (observe_routed_access)
    report_routed_access(*inst, wf, route_tag, decoded_route_tag, normalized_to_local,
                         pre_routing_addresses, flat_local_lane_mask, flat_dds_lane_mask);

  switch (route_tag) {
  case SCALAR_MEM:
    return scalar_mem_pipeline_.issue_deferred(inst, wf);
  case LOCAL_MEM:
    return local_mem_pipeline_.issue_deferred(inst, wf);
  case GLOBAL_MEM:
    return global_mem_pipeline_.issue_deferred(inst, wf);
  default:
    delete inst;
    return VmAccessOutcome::Malformed;
  }
}

void Wavefront::report_undefined_behavior(std::string_view reason) const {
  cu_.report_undefined_behavior(*this, reason);
}

void ComputeUnitCore::report_static_isa_diagnostics(const Instruction &inst, const Wavefront &wf) {
  const uint64_t flags = inst.flags();
  if (flags & INVALID_MFMA_BROADCAST)
    report_undefined_behavior(wf, "invalid MFMA CBSZ/ABID broadcast controls");
  // RDNA3/3.5 skip VOPD in wave64; only diagnose the qualified wave32 execution.
  if ((flags & INVALID_VOPD_OPERANDS) && wf.wf_size() == 32 && wf.exec())
    report_undefined_behavior(wf, "invalid VOPD operand combination");
  if ((flags & INVALID_IU_MODIFIERS) && wf.exec())
    report_undefined_behavior(wf, "undefined IU DOT/WMMA modifiers");
  if (flags & MISALIGNED_SCALAR_DATA)
    report_undefined_behavior(wf, "misaligned scalar data tuple");
}

void ComputeUnitCore::report_undefined_behavior(const Wavefront &wf, std::string_view reason) {
  const uint64_t count = isa_diagnostic_count_.fetch_add(1, std::memory_order_relaxed) + 1;
  if (count > kMaxIsaDiagnostics)
    return;
  util::Logger::warn(std::format("isa-undefined: {} wg={} wave={} pc={:#x}: {}.", full_path(),
                                 wf.wg_id(), wf.wf_id(), wf.pc, reason));
  if (count == kMaxIsaDiagnostics)
    util::Logger::warn("isa-undefined: further diagnostics on this CU are suppressed");
}

void ComputeUnitCore::report_memory_wait(void *context,
                                         const MemoryWaitScoreboard::Hazard &hazard) {
  auto &wf = *static_cast<Wavefront *>(context);
  auto &cu = wf.raw_cu();
  const bool replay = hazard.producer.counter == WaitCounterKind::X;
  const auto count = replay ? ++cu.xcnt_diagnostic_count_ : ++cu.memory_wait_diagnostic_count_;
  if (count > kMaxMemoryWaitDiagnostics)
    return;
  auto counter_name = wait_counter_name(hazard.producer.counter);
  const auto model = waitcnt_model(cu.arch());
  if (model.succeeded() && uses_legacy_waitcnt(model.value())) {
    if (hazard.producer.counter == WaitCounterKind::Load)
      counter_name = "vmcnt";
    else if (hazard.producer.counter == WaitCounterKind::Store)
      counter_name = "vscnt";
    else if (hazard.producer.counter == WaitCounterKind::Ds)
      counter_name = "lgkmcnt";
  }
  const auto max_wait =
      WaitCounterPolicy::maximum_dependency_wait(cu.arch(), hazard.producer.counter);
  const auto required = max_wait.succeeded() ? std::min(hazard.required_wait, max_wait.value())
                                             : hazard.required_wait;
  const auto register_name = [&] {
    switch (hazard.reg.cls) {
    case RegClass::VGPR:
      return std::format("v{}", hazard.reg.index);
    case RegClass::SGPR:
      return std::format("s{}", hazard.reg.index);
    case RegClass::TTMP:
      return std::format("ttmp{}", hazard.reg.index);
    case RegClass::ACC_VGPR:
      return std::format("acc{}", hazard.reg.index);
    case RegClass::EXEC:
      return std::string("exec");
    case RegClass::VCC:
      return std::string("vcc");
    case RegClass::SCC:
      return std::string("scc");
    case RegClass::M0:
      return std::string("m0");
    case RegClass::FLAT_SCRATCH:
      return std::string("flat_scratch");
    case RegClass::PC:
      return std::string("pc");
    }
    return std::string("unknown register");
  }();
  if (replay) {
    util::Logger::warn(std::format(
        "xcnt-wait: {} wg={} wave={} pc={:#x}: overwrite of {} before the replay source from "
        "pc={:#x} is known safe to reuse. XNACK replay may need the original value. "
        "s_wait_xcnt <= {} is required; set top-level wait_checking to \"off\" "
        "(rocjitsu --wait-checking=off) to disable wait checking.",
        cu.full_path(), wf.wg_id(), wf.wf_id(), hazard.consumer_pc, register_name,
        hazard.producer.pc, required));
  } else {
    util::Logger::warn(std::format(
        "memory-wait: {} wg={} wave={} pc={:#x}: {} of {} before memory result from pc={:#x} is "
        "known ready ({}). A wait threshold <= {} is required; set top-level wait_checking "
        "to \"off\" (rocjitsu --wait-checking=off) to disable wait checking.",
        cu.full_path(), wf.wg_id(), wf.wf_id(), hazard.consumer_pc,
        hazard.write ? "overwrite" : "read", register_name, hazard.producer.pc, counter_name,
        required));
  }
  if (count == kMaxMemoryWaitDiagnostics)
    util::Logger::warn(replay ? "xcnt-wait: further diagnostics on this CU are suppressed"
                              : "memory-wait: further diagnostics on this CU are suppressed");
}

void ComputeUnitCore::track_memory_wait(Instruction &inst, Wavefront &wf) {
  using namespace waitcheck_detail;
  // Common single-counter loads have one decoded full-dword destination.
  // Resolve and issue that dependency before instruction execution.
  assert(inst.is_memory_wait_producer());
  const auto *info = inst.amdgpu_memory_issue_info();
  if ((inst.flags() & SIMPLE_MEMORY_RESULT) && !config_.xcnt_checks_enabled() && info &&
      info->counter_obligations().size() == 1) {
    const auto obligation = info->counter_obligations().front();
    const auto completion = obligation.completion_class();
    const bool scalar = completion == MemoryCompletionClass::UNORDERED;
    if (scalar || completion == MemoryCompletionClass::VMEM ||
        completion == MemoryCompletionClass::LDS) {
      const auto model = waitcnt_model(config_.arch).value();
      const auto counter = scalar                                     ? smem_wait_counter(model)
                           : completion == MemoryCompletionClass::LDS ? WaitCounterKind::Ds
                                                                      : WaitCounterKind::Load;
      const auto kind = scalar ? WaitEventKind::Smem
                        : completion == MemoryCompletionClass::LDS
                            ? WaitEventKind::Ds
                            : WaitEventKind::VmemNoSamplerLoad;
      auto &scoreboard = wf.ensure_memory_wait_scoreboard();
      scoreboard.bind(wf.pc, &wf, &ComputeUnitCore::report_memory_wait);
      uint64_t lanes = scalar ? ~uint64_t{0}
                       : info->exec_masked
                           ? wf.exec()
                           : (wf.wf_size() == 64 ? ~uint64_t{0} : uint64_t{0xffffffff});
      uint64_t issue_lanes = lanes;
      if (!scalar && (inst.flags() & CONDITIONAL_MEMORY_LANES)) {
        const auto masks = MemoryWaitScoreboard::memory_lanes(inst, wf);
        issue_lanes = masks.issue;
        lanes = masks.result;
      }
      if (!issue_lanes && !scoreboard.outstanding(counter))
        return;
      // These qualified loads have one known FIFO class. The legacy VM
      // counter shares one class across ordinary loads; SMEM stays unordered.
      const uint32_t ordering_kind =
          scalar || (model == WaitcntModel::LegacyNoVscnt && counter == WaitCounterKind::Load)
              ? 0
              : uint32_t{1} << static_cast<unsigned>(kind);
      const auto sequence =
          scoreboard.issue(counter, scalar, ordering_kind, obligation.counter_increment());
      if (config_.memory_wait_diagnostics == MemoryWaitDiagnostics::Off)
        return;
      assert(inst.num_dst_operands() == 1);
      const auto &operand = *inst.dst_operand(0);
      if (scalar) {
        if (const auto range = resolve_scalar_destination_range(
                wf, operand.encoding_value(), std::max(1, operand.size_bits() / 32)))
          if (const auto reg = range->register_ref())
            scoreboard.add(
                {sequence, wf.pc, lanes, *reg, counter, MemoryWaitScoreboard::kFullDwordByteMask});
      } else {
        auto reg = operand.decoded_vgpr();
        if (reg && !wf.gpr_idx_en() &&
            !(inst.flags() & (RESERVES_UNWRITTEN_VGPR | COMPACT_WAVE64_MEMORY_RESULT))) {
          reg->index +=
              reg->index < ACC_VGPR_OFFSET ? wf.vgpr_msb_for_role(operand.vgpr_msb_role()) << 8 : 0;
          if (!owns_memory_result_range(wf, wf.vgpr_alloc().base + reg->index, reg->width))
            reg = std::nullopt;
        } else {
          reg = RegisterAccess(wf).memory_destination_register(inst, operand);
        }
        if (reg)
          scoreboard.add(
              {sequence, wf.pc, lanes, *reg, counter, MemoryWaitScoreboard::kFullDwordByteMask});
      }
      return;
    }
  }
  const auto model = waitcnt_model(config_.arch);
  if (model.failed())
    return;
  auto &classified = memory_wait_classification_;
  if (WaitcheckTarget::classify_events_into(inst, config_.arch, classified).failed() ||
      classified.empty())
    return;
  auto &scoreboard = wf.ensure_memory_wait_scoreboard();
  scoreboard.bind(wf.pc, &wf, &ComputeUnitCore::report_memory_wait);
  const auto xevent =
      config_.xcnt_checks_enabled()
          ? std::ranges::find(classified, WaitCounterKind::X, &ClassifiedEvent::counter)
          : classified.end();
  const bool xscalar = xevent != classified.end() && xevent->kind == WaitEventKind::Smem;
  // The qualified VMEM policy is LLVM's multi-group replay mode. MODE[25]
  // selects VMEM grouping, not whether XNACK is enabled. SMEM is independent.
  const bool track_xcnt = xevent != classified.end() && (xscalar || (wf.mode_raw() & (1u << 25)));
  // Even a single-group VMEM instruction drains preceding SMEM translations.
  if (xevent != classified.end())
    scoreboard.xcnt_group(xscalar);
  const auto memory_lanes = MemoryWaitScoreboard::memory_lanes(inst, wf);
  const uint64_t vector_lanes = memory_lanes.result;

  // Source checks precede this pure FLAT domain calculation. Resolve admission,
  // overwrite ordering and result readiness before execution.
  const bool flat = inst.mnemonic().starts_with("flat_");
  const auto planned_flat =
      flat ? MemoryWaitScoreboard::flat_lanes(
                 inst, wf,
                 wf.use_wave_apertures() ? wf.shared_aperture_base() : shared_aperture_base_,
                 wf.use_wave_apertures() ? wf.shared_aperture_limit() : shared_aperture_limit_)
           : MemoryWaitScoreboard::FlatLanes{};
  const uint64_t flat_shared_lanes = planned_flat.shared;
  const uint64_t flat_requests = planned_flat.requests;
  const auto flat_counter_lanes = [&](WaitCounterKind counter) {
    return counter == WaitCounterKind::Ds ? flat_requests & flat_shared_lanes
                                          : flat_requests & ~flat_shared_lanes;
  };
  const auto is_flat_memory_counter = [&](WaitCounterKind counter) {
    return flat && (counter == WaitCounterKind::Load || counter == WaitCounterKind::Store ||
                    counter == WaitCounterKind::Ds);
  };
  for (const auto &event : classified) {
    if (!is_flat_memory_counter(event.counter) || !flat_counter_lanes(event.counter))
      continue;
    const auto maximum = WaitCounterPolicy::maximum_dependency_wait(config_.arch, event.counter);
    if (maximum.succeeded())
      scoreboard.backpressure(event.counter, maximum.value() + 1,
                              scoreboard.issue_units(inst, event, config_.arch));
  }

  struct Destination {
    RegisterRef reg;
    uint64_t lanes;
    uint8_t bytes;
  };
  // A block load can select any subset of its DWORD destinations.
  std::array<Destination, kBlockDwordCount> destinations;
  size_t num_destinations = 0;
  auto append = [&](RegisterRef reg, uint64_t lanes, uint8_t bytes) {
    if (reg.cls == RegClass::TTMP && !wf.shader_can_write_ttmp())
      return;
    assert(num_destinations < destinations.size());
    destinations[num_destinations++] = {reg, lanes, bytes};
  };
  // Named scalar destinations such as VCC need their architectural selector,
  // even when static liveness does not include their register class.
  std::optional<RegisterRef> scalar_result;
  auto scalar_event = std::ranges::find_if(classified, [](const auto &event) {
    return event.kind == WaitEventKind::Smem || event.kind == WaitEventKind::SqMessage;
  });
  if (scalar_event != classified.end() && inst.num_dst_operands() == 1) {
    const auto *operand = inst.dst_operand(0);
    if (operand->has_register_selector() && !operand->is_fieldless())
      if (const auto range = resolve_scalar_register_range(wf, operand->encoding_value(),
                                                           std::max(1, operand->size_bits() / 32)))
        scalar_result = range->register_ref();
    if (scalar_result)
      scalar_event->registers = TrackedRegisterSource::Defs;
  }
  const auto defs =
      std::ranges::find(classified, TrackedRegisterSource::Defs, &ClassifiedEvent::registers);
  if (defs != classified.end()) {
    if (scalar_result) {
      append(*scalar_result, ~uint64_t{0}, MemoryWaitScoreboard::kFullDwordByteMask);
    } else {
      std::array<RegisterRef, kBlockDwordCount> registers{};
      unsigned count = 0;
      const RegisterAccess access(wf);
      RegisterModifiers modifiers;
      inst.amdgpu_register_modifiers(modifiers);
      for (int i = 0; i < inst.num_dst_operands(); ++i) {
        const auto &operand = *inst.dst_operand(i);
        if (&operand == modifiers.block_data) {
          for (uint32_t word = 0; word < kBlockDwordCount; ++word)
            if (block_dword_enabled(wf.m0(), word))
              if (const auto reg = access.block_data_register(operand, word, /*write=*/true)) {
                assert(count < registers.size());
                registers[count++] = *reg;
              }
          continue;
        }
        const auto reg = access.memory_destination_register(inst, operand);
        if (operand.decoded_vgpr() && !reg) {
          // Completion validates every vector destination together.
          count = 0;
          break;
        }
        if (reg) {
          assert(count < registers.size());
          registers[count++] = *reg;
        }
      }
      for (unsigned i = 0; i < count; ++i) {
        const auto reg = registers[i];
        const auto lanes = reg.cls == RegClass::VGPR ? vector_lanes : ~uint64_t{0};
        for_each_memory_result_range(
            reg, modifiers, sram_ecc(),
            [&](RegisterRef range, uint8_t bytes) { append(range, lanes, bytes); });
      }
    }
  }
  // Check all writes before adding any new dependencies. FLAT has the same
  // destination on two counters; those entries are not writes racing each other.
  const bool flat_result = defs != classified.end() && defs->kind == WaitEventKind::FlatLoad;
  const auto vmem_write_order = has_ordered_vmem_writeback(config_.arch) &&
                                        defs != classified.end() &&
                                        defs->counter == WaitCounterKind::Load
                                    ? scoreboard.ordered_write_order(*defs, config_.arch)
                                    : MemoryWaitScoreboard::kUnordered;
  if (flat) {
    const auto ds_event =
        std::ranges::find(classified, WaitCounterKind::Ds, &ClassifiedEvent::counter);
    const auto lds_write_order =
        has_ordered_flat_lds_writeback(config_.arch) && ds_event != classified.end()
            ? scoreboard.ordered_write_order(*ds_event, config_.arch)
            : MemoryWaitScoreboard::kUnordered;
    for (size_t i = 0; i < num_destinations; ++i) {
      const auto &d = destinations[i];
      if (track_xcnt && !xscalar)
        scoreboard.xcnt_ordered_write(d.reg, d.lanes, d.bytes);
      // The incoming class determines VMEM writeback ordering before its issue.
      // Shared FLAT lanes use their own DS completion class. Only architectures
      // with ordered LDS-route writeback can share that class with indexed LDS.
      const uint64_t shared_lanes = flat_result ? d.lanes & flat_shared_lanes : 0;
      scoreboard.access(d.reg, d.lanes & ~shared_lanes, d.bytes, true, vmem_write_order);
      scoreboard.access(d.reg, shared_lanes, d.bytes, true, lds_write_order);
    }
  }
  // FLAT's actual route determines which completion proves translation. Only
  // map X to a position issued below for this instruction: zero-EXEC operations
  // may be skipped by an empty completion queue while older X entries remain.
  const bool flat_local_route =
      flat && flat_requests != 0 && (flat_requests & ~flat_shared_lanes) == 0;
  const auto xcnt_completion_counter =
      flat_local_route ? WaitCounterKind::Ds : classified.front().counter;
  std::optional<WaitCounterKind> xcnt_completion;
  for (const auto &event : classified) {
    const auto counter = event.counter;
    if (is_flat_memory_counter(counter) && !flat_counter_lanes(counter))
      continue;
    if (counter == WaitCounterKind::X) {
      if (track_xcnt && (xscalar || vector_lanes || scoreboard.outstanding(counter))) {
        const auto sequence = scoreboard.issue_xcnt(xcnt_completion, xscalar);
        const RegisterAccess registers(wf);
        auto add_source = [&](RegisterRef reg) {
          scoreboard.add({sequence, wf.pc, reg.cls == RegClass::VGPR ? vector_lanes : ~uint64_t{0},
                          reg, counter, MemoryWaitScoreboard::kFullDwordByteMask});
        };
        // Zero-EXEC VMEM still occupies a queue entry but has no data sources.
        // EXEC itself must survive until the instruction can no longer replay.
        if (xscalar || vector_lanes) {
          RegisterModifiers modifiers;
          inst.amdgpu_register_modifiers(modifiers);
          for (int i = 0; i < inst.num_src_operands(); ++i) {
            if (!inst.tracks_explicit_source(i))
              continue;
            const auto *operand = inst.src_operand(i);
            if (operand && operand == modifiers.block_data) {
              // Resolve data before whole-operand ownership validation. Address
              // operands remain separate, even when they overlap masked data.
              for (uint32_t word = 0; word < kBlockDwordCount; ++word)
                if (block_dword_enabled(wf.m0(), word))
                  if (const auto reg =
                          registers.block_data_register(*operand, word, /*write=*/false))
                    add_source(*reg);
            } else if (operand && operand == modifiers.buffer_resource) {
              for (const auto reg :
                   registers.buffer_resource_registers(*operand, modifiers.scalar_buffer_words))
                if (reg)
                  add_source(*reg);
            } else if (operand) {
              if (auto reg = registers.source_register(*operand,
                                                       operand == modifiers.wordwise_source0 ||
                                                           operand == modifiers.wordwise_source1))
                add_source(*reg);
            }
          }
          RegisterSet implicit;
          inst.implicit_uses(implicit);
          if (!implicit.none())
            implicit.for_each(add_source);
        }
        if (!xscalar)
          add_source({RegClass::EXEC, 0, static_cast<uint8_t>(wf.wf_size() / 32)});
      }
      continue;
    }
    // Expert scheduling dependencies are independent of completion and replay.
    if (counter == WaitCounterKind::VmVsrc || counter == WaitCounterKind::VaVdst ||
        counter == WaitCounterKind::Depctr)
      continue;
    const bool scalar = event.kind == WaitEventKind::Smem ||
                        event.kind == WaitEventKind::SqMessage ||
                        event.kind == WaitEventKind::SccWrite;
    // Zero-EXEC instructions still occupy positions behind pending requests.
    if (!scalar && !memory_lanes.issue && !scoreboard.outstanding(counter))
      continue;
    const auto sequence =
        scoreboard.issue(event, config_.arch, scoreboard.issue_units(inst, event, config_.arch));
    if (counter == xcnt_completion_counter)
      xcnt_completion = counter;
    // Replay-only checking still needs completion positions to retire X, but
    // must not add result dependencies or report completion hazards.
    if (config_.memory_wait_diagnostics == MemoryWaitDiagnostics::Off)
      continue;
    if (event.special_reg && MemoryWaitScoreboard::result_is_written(inst, wf)) {
      scoreboard.add({sequence, wf.pc, ~uint64_t{0}, *event.special_reg, counter,
                      MemoryWaitScoreboard::kFullDwordByteMask});
    }
    if (event.registers == TrackedRegisterSource::Defs)
      for (size_t i = 0; i < num_destinations; ++i) {
        const auto &d = destinations[i];
        uint64_t lanes = d.lanes;
        if (event.kind == WaitEventKind::FlatLoad)
          lanes &= counter == WaitCounterKind::Ds ? flat_shared_lanes : ~flat_shared_lanes;
        MemoryWaitScoreboard::Event result{sequence, wf.pc, lanes, d.reg, counter, d.bytes};
        // FLAT's LDS-serviced lanes share native LDS's order class, so partial
        // waits retire them together. Only qualified targets also write them
        // to registers in order with native LDS results.
        if (event.kind == WaitEventKind::FlatLoad && counter == WaitCounterKind::Ds)
          result.ordered_overwrite = has_ordered_flat_lds_writeback(config_.arch);
        scoreboard.add(result);
      }
  }
}

// The observation's route is MemPipelineTag under another name; keep the two
// numberings pinned so a new tag cannot pick up an existing route's value.
static_assert(static_cast<uint8_t>(MemoryRoute::SCALAR) == SCALAR_MEM);
static_assert(static_cast<uint8_t>(MemoryRoute::GLOBAL) == GLOBAL_MEM);
static_assert(static_cast<uint8_t>(MemoryRoute::LOCAL) == LOCAL_MEM);

namespace {

DecodedMemorySpace decoded_memory_space(std::string_view mnemonic, uint8_t decoded_route_tag) {
  // Canonical AMDGPU mnemonics carry the encoding family. Keep this decision
  // separate from the route tag: FLAT and explicit SCRATCH both initially use
  // the global pipeline, but a plugin must be able to tell them apart.
  if (mnemonic.starts_with("flat_"))
    return DecodedMemorySpace::FLAT;
  if (mnemonic.starts_with("scratch_") || mnemonic.starts_with("s_scratch_"))
    return DecodedMemorySpace::SCRATCH;
  if (mnemonic.starts_with("ds_"))
    return DecodedMemorySpace::LOCAL;
  if (mnemonic.starts_with("global_") || mnemonic.starts_with("buffer_") ||
      mnemonic.starts_with("tbuffer_") || mnemonic.starts_with("image_"))
    return DecodedMemorySpace::GLOBAL;
  if (mnemonic.starts_with("s_"))
    return DecodedMemorySpace::SCALAR;

  // Synthetic instructions and future families still get the least-specific
  // fact routing already knew before it made any changes.
  switch (decoded_route_tag) {
  case SCALAR_MEM:
    return DecodedMemorySpace::SCALAR;
  case GLOBAL_MEM:
    return DecodedMemorySpace::GLOBAL;
  case LOCAL_MEM:
    return DecodedMemorySpace::LOCAL;
  default:
    return DecodedMemorySpace::UNKNOWN;
  }
}

} // namespace

void ComputeUnitCore::report_routed_access(const Instruction &inst, Wavefront &wf,
                                           uint8_t route_tag, uint8_t decoded_route_tag,
                                           bool normalized_to_local,
                                           std::span<const uint64_t> pre_routing_addresses,
                                           uint64_t flat_local_lane_mask,
                                           uint64_t flat_dds_lane_mask) {
  std::vector<ImageSampleFootprint> image_footprints;
  std::array<uint64_t, 16> scalar_element_masks;
  MemoryAccessObservation access;
  access.mnemonic = inst.mnemonic();
  access.pc = wf.pc;
  access.compute_unit_id = id();
  access.dispatch_id = wf.dispatch_id();
  access.queue_id = wf.queue_id();
  access.workgroup_id = wf.wg_id();
  access.wavefront_id = wf.wf_id();
  access.process_id = wf.process_id();
  access.decoded_space = decoded_memory_space(inst.mnemonic(), decoded_route_tag);
  access.normalized_to_local = normalized_to_local;
  access.pre_routing_addresses = pre_routing_addresses;

  switch (route_tag) {
  case SCALAR_MEM: {
    const auto &state = *inst.data_as<ScalarMemState>();
    access.route = MemoryRoute::SCALAR;
    access.is_load = state.is_load;
    access.atomic_op = state.atomic_op;
    access.mtype = state.mtype;
    access.wait_counter = state.wait_counter_type;
    access.element_size_bytes = state.elem_size;
    // An atomic is one element of elem_size bytes, matching vector atomic
    // observations; num_dwords counts its returned dwords.
    access.elements_per_lane = state.atomic_op != AtomicOp::NONE ? 1 : state.num_dwords;
    // A scalar access is one address, so it is a one-lane wavefront as far as
    // the memory system is concerned. Saying so lets a consumer treat both
    // routes with the same per-lane arithmetic.
    access.wavefront_size = 1;
    access.active_lane_mask = 1;
    access.architectural_exec_lane_mask = 1;
    // The pipeline skips a buffer access whose bounds mask clears every DWORD.
    // Like an out-of-range vector lane, its lane is active but not valid.
    const bool accessed = !state.memviol && state.load_dword_mask != 0;
    access.valid_lane_mask = accessed;
    access.request_lane_mask = accessed;
    if (accessed && state.atomic_op == AtomicOp::NONE && state.load_dword_mask != 0xffff) {
      assert(state.num_dwords <= scalar_element_masks.size());
      auto masks = std::span(scalar_element_masks).first(state.num_dwords);
      for (size_t word = 0; word < masks.size(); ++word)
        masks[word] = (state.load_dword_mask >> word) & 1u;
      if (std::ranges::find(masks, uint64_t{0}) != masks.end())
        access.element_lane_masks = masks;
    }
    access.addresses = std::span<const uint64_t>(&state.addr, 1);
    break;
  }
  case GLOBAL_MEM:
  case LOCAL_MEM: {
    const auto &state = *inst.data_as<VectorMemState>();
    const uint32_t wf_size = wf.wf_size();
    access.route = route_tag == LOCAL_MEM ? MemoryRoute::LOCAL : MemoryRoute::GLOBAL;
    access.is_load = state.is_load;
    access.atomic_op = state.atomic_op;
    access.mtype = state.mtype;
    access.wait_counter = state.wait_counter_type;
    access.wavefront_size = wf_size;
    access.element_size_bytes = state.elem_size;
    access.elements_per_lane = state.num_elems;
    access.active_lane_mask = state.exec_mask;
    access.architectural_exec_lane_mask = wf.exec() & access.wavefront_lane_mask();
    const uint64_t requests = transpose_request_lane_mask(state, wf_size);
    const bool rejected = state.rejects_routed_access();
    access.valid_lane_mask = state.lane_mask;
    // A prequalified synchronous fault rejects every lane before either pipeline accesses bytes.
    if (rejected)
      access.valid_lane_mask = 0;
    access.request_lane_mask = rejected ? 0 : requests;
    access.flat_local_lane_mask = flat_local_lane_mask & access.request_lane_mask;
    access.flat_dds_lane_mask = flat_dds_lane_mask & access.request_lane_mask;
    access.flat_local_oob_lane_mask = rejected ? 0 : state.flat_local_oob_lane_mask;
    access.flat_local_allocation_end = state.flat_local_allocation_end;
    // Intersected with the requesting lanes, which is how the global pipeline
    // splits the wave: a swizzled lane that never requests costs nothing.
    access.scratch_lane_mask =
        state.scratch_swizzle ? state.scratch_lane_mask & access.request_lane_mask : 0;
    access.scratch_element_stride_bytes = state.scratch_swizzle ? state.scratch_addr_stride : 0;
    access.non_temporal = state.non_temporal;
    access.force_l1_bypass = state.request_force_l1_bypass;
    access.lds_destination = state.lds_dst;
    if (state.image_sample) {
      for (const auto &tap : state.image_sample->taps)
        image_footprints.push_back({std::span(tap.addresses).first(wf_size), tap.lane_mask});
      access.image_sample_footprints = image_footprints;
    } else
      access.addresses = std::span<const uint64_t>(state.per_lane_addr.data(), wf_size);
    if (!rejected)
      access.element_lane_masks = state.element_lane_masks.view();
    if (state.ds2_active)
      access.secondary_addresses =
          std::span<const uint64_t>(state.ds2_per_lane_addr.data(), wf_size);
    break;
  }
  default:
    // Left UNKNOWN, so a consumer counting the kernel's memory traffic sees an
    // access it cannot account for rather than never hearing about it.
    break;
  }

  plugin_group_->onAmdgpuMemoryAccessRouted(access, inst, wf);
}

void ComputeUnitCore::update_wf_states() {
  ++cycle_counter_;

#ifndef NDEBUG
  // A missed activity transition would otherwise leave a wave outside every
  // hot-path scan and silently prevent its dispatch from completing.
  for (size_t slot = active_slot_limit_; slot < wfs_.size(); ++slot)
    assert((!wfs_[slot] || wfs_[slot]->is_halted()) &&
           "excluded wavefront slot must be halted or empty");
#endif

  for (uint32_t slot = 0; slot < active_slot_limit_; ++slot) {
    auto &wf = wfs_[slot];
    if (!wf)
      continue;
    // A debugger resume can release HALT, but FATAL_HALT blocks issue even in
    // privileged mode until the host clears the architectural cause.
    if (arch() == ROCJITSU_CODE_ARCH_RDNA3_5 && (wf->status_raw() & (1u << 23)) &&
        !wf->is_halted() && !wf->debug_halted())
      wf->set_debug_halted(true);
    if (wf->state() == WfState::MEMORY_TRAP) {
      deliver_memory_trap(*wf);
    } else if (wf->state() == WfState::WAITCNT && wf->wait_satisfied()) {
      wf->set_state(WfState::RUNNING);
      wf->set_ready_cycle(cycle_counter_);
    } else if (wf->state() == WfState::ENDING && wf->wait_counters().empty()) {
      wf->halt();
    }
  }

  for (uint32_t slot = 0; slot < active_slot_limit_; ++slot) {
    auto &wf = wfs_[slot];
    if (!wf)
      continue;
    if (wf->state() != WfState::BARRIER || wf->waiting_barrier_bit_ != Wavefront::kNoBarrierWait)
      continue;
    uint32_t did = wf->dispatch_id();
    uint32_t wg = wf->wg_id();
    bool all_at_barrier = true;
    for (uint32_t other = 0; other < active_slot_limit_; ++other) {
      auto &other_wf = wfs_[other];
      if (other_wf && other_wf->dispatch_id() == did && other_wf->wg_id() == wg &&
          other_wf->state() != WfState::HALTED &&
          (other_wf->state() != WfState::BARRIER ||
           other_wf->waiting_barrier_bit_ != Wavefront::kNoBarrierWait)) {
        all_at_barrier = false;
        break;
      }
    }
    if (all_at_barrier) {
      std::vector<Wavefront *> barrier_wfs;
      for (uint32_t other = 0; other < active_slot_limit_; ++other) {
        auto &other_wf = wfs_[other];
        if (other_wf && other_wf->dispatch_id() == did && other_wf->wg_id() == wg &&
            other_wf->state() == WfState::BARRIER &&
            other_wf->waiting_barrier_bit_ == Wavefront::kNoBarrierWait)
          barrier_wfs.push_back(other_wf.get());
      }
      notify_barrier_complete(barrier_wfs, kWorkgroupBarrierBit);
      for (auto *bwf : barrier_wfs) {
        bwf->set_state(WfState::RUNNING);
        bwf->set_ready_cycle(cycle_counter_);
      }
    }
  }

  // Ordinary kernels have no GWS waiters. Avoid an extra wavefront scan on
  // every step until a synchronization operation actually parks a wave.
  if (!gws_device_->any_parked()) {
    gws_step_progress_processes_.clear();
    return;
  }

  // Poll real releases before charging the wait budget. Neither quiescence nor
  // insufficient residency is permission to release a synchronization operation.
  gws_device_->poll_waits(*this);
  bool has_waiter = false, has_other_wave = false;
  for (const auto &wave : wfs_) {
    if (!wave || wave->is_halted() || wave->debug_paused() || wave->status_halt())
      continue;
    if (wave->state() != WfState::GWS_WAIT) {
      // Workgroup barriers cannot progress while all remaining waves are parked.
      // In a cooperative grid, sibling waves wait here for wave 0's GWS arrival.
      has_other_wave |= wave->state() != WfState::BARRIER;
      continue;
    }
    has_waiter = true;
    // Local progress is visible on the next step even before the quantum batch
    // is published to peers, so small budgets do not reject co-resident producers.
    if (std::ranges::find(gws_step_progress_processes_, wave->process_id()) !=
        gws_step_progress_processes_.end())
      wave->gws_wait_steps_ = 0;
    if (config_.gws_wait_limit != 0 && ++wave->gws_wait_steps_ >= config_.gws_wait_limit)
      fail_gws_wait(*wave, "GWS inactivity limit exceeded");
  }
  gws_step_progress_processes_.clear();
  if (has_waiter && !has_other_wave) {
    // Return even for unlimited quanta, without shortening a co-resident producer's
    // quantum. The drivers poll a stalled CU at the normal quantum cadence.
    gws_wait_yield_requested_ = true;
    request_functional_yield();
  }
}

AsyncInstructionWindow::AsyncInstructionWindow(ComputeUnitCore &cu, Wavefront &wf,
                                               bool has_accvgprs)
    : cu_(cu), wf_(wf), pool_(cu.async_pool()), has_accvgprs_(has_accvgprs) {}

matrix_coexecution::SharedPool &ComputeUnitCore::async_pool() {
  if (!async_pool_)
    async_pool_ = &config_.async_resources->pool();
  return *async_pool_;
}

void ComputeUnitCore::fail_wave_instruction(Wavefront &wf, std::string_view instruction,
                                            uint64_t pc, InstructionExecutionError error,
                                            std::string_view failure_detail) {
  std::string reason(instruction_execution_error_name(error));
  if (!failure_detail.empty())
    reason += std::format(" ({})", failure_detail);
  const std::string failure = std::format(
      "CU {}: wf{} could not execute {} at pc={:#x} (pid={} qid={} dispatch={} wg={}): {}",
      this->name(), wf.wf_id(), instruction, pc, wf.process_id(), wf.queue_id(), wf.dispatch_id(),
      wf.wg_id(), reason);
  const bool exception_claimed = error == InstructionExecutionError::IllegalInstruction &&
                                 !wf.fail_pm4_submission() && illegal_inst_handler_ &&
                                 illegal_inst_handler_(wf);
  if (!exception_claimed || !wf.debug_halted())
    handle_instruction_failure(wf, failure, exception_claimed);
}

void AsyncInstructionWindow::materialize() {
  if (materialized_)
    return;
  // Inline instructions can touch new registers before jobs finish. Allocate
  // all lazy chunks before that can race with worker register-file access.
  for (uint32_t reg = 0; reg != wf_.num_vgprs(); ++reg)
    (void)cu_.raw_vgpr_data(wf_.vgpr_alloc().base + reg);
  if (has_accvgprs_)
    for (uint32_t reg = 256; reg != 512; ++reg)
      (void)cu_.raw_vgpr_data(wf_.vgpr_alloc().base + reg);
  materialized_ = true;
}

void ComputeUnitCore::issue_async_instruction(Wavefront *active, MmaAdmissionCache *admission,
                                              uint32_t wave_size, bool has_accvgprs) {
  // CDNA4 MFMA and the default gfx1250 allowlist have cheap encoding filters.
  // For retained dispatch bindings and legacy memory, resident non-candidates
  // use the ordinary issue body, including its fetchability and
  // debugger checks. A cache miss uses full decoding below.
  // This hint never executes a cached word or bypasses instruction validation.
  if (async_mma_policy::supported(arch())) {
    uint32_t word;
    const GpuVmAccess *vm_access = active->vm_access();
    const uint64_t pc =
        vm_access || active->process_id() ? active->vm_instruction_pc() : active->pc;
    const bool resident = vm_access ? inst_cache_.peek_word(*vm_access, pc, word)
                                    : inst_cache_.peek_word(pc, active->process_id(), word);
    if (resident) {
      if (!async_mma_policy::encoding_may_be_candidate(arch(), word)) {
        issue_instruction(active);
        if (active->is_halted())
          async_execution::stats.flush();
        return;
      }
    }
  }
  const uint64_t full_exec = wave_size == 64 ? ~uint64_t{0} : uint64_t{0xFFFFFFFF};
  if (active->wf_size() != wave_size || active->exec() != full_exec ||
      active->vgpr_msb_mode() != 0 || active->gpr_idx_en() || debug_active() ||
      active->debug_single_step() || active->in_trap_handler() ||
      !plugin_group_->supports_async_instructions()) {
    issue_instruction_impl<true>(active);
    return;
  }
  AsyncInstructionWindowStorage storage;
  storage.admission = admission;
  storage.has_accvgprs = has_accvgprs;
  try {
    unsigned issued = 0;
    do {
      issue_instruction_impl<true>(active, storage.window ? &*storage.window : nullptr, &storage);
      if (!storage.window) {
        if (active->is_halted())
          async_execution::stats.flush();
        return;
      }
      storage.window->poll();
    } while (++issued < async_execution::issue_limit() && storage.window->pending() &&
             !storage.window->stopped() && active->state() == WfState::RUNNING);
    storage.window->drain();
  } catch (const util::UnimplementedInst &error) {
    // Helper threads execute the same guest callbacks as the inline path. Join
    // the remaining jobs before translating an unsupported guest combination
    // into the wave-owned execution failure used by synchronous execution.
    if (storage.window)
      storage.window->abandon();
    active->report_instruction_execution_error(InstructionExecutionError::UnimplementedInstruction);
    const std::string instruction =
        storage.window && !storage.window->failed_mnemonic().empty()
            ? std::format("asynchronous {}", storage.window->failed_mnemonic())
            : std::string("asynchronous instruction");
    const uint64_t failure_pc = storage.window && !storage.window->failed_mnemonic().empty()
                                    ? storage.window->failed_pc()
                                    : active->pc;
    fail_wave_instruction(*active, instruction, failure_pc, active->instruction_execution_error(),
                          error.what());
    async_execution::stats.flush();
    return;
  } catch (...) {
    if (storage.window)
      storage.window->abandon();
    throw;
  }
  if (active->is_halted())
    async_execution::stats.flush();
}

constinit thread_local ComputeUnitCore::InstructionVmSnapshot
    *ComputeUnitCore::instruction_vm_snapshot_ = nullptr;

template <bool EnableAsync>
[[gnu::always_inline]] inline ComputeUnitCore::IssueOutcome
ComputeUnitCore::issue_instruction_impl(Wavefront *active, MaybeAsyncWindow<EnableAsync> window,
                                        MaybeAsyncWindowStorage<EnableAsync> storage) {
  uint32_t vmid = active->process_id();
  const auto drain_async_window = [&]() {
    if constexpr (EnableAsync) {
      if (window)
        window->drain();
    }
  };

  // A completed FLAT exception interrupts between instructions. An enabled CDNA2
  // range exception enters a configured shader. A private atomic on GFX11+
  // must also reach the queue or debugger when no shader can take the trap.
  if (active->memory_trap_pending()) [[unlikely]] {
    const auto config = trap_handler_resolver_ ? trap_handler_resolver_(*active) : std::nullopt;
    const bool enters = config && config->tba && !active->in_trap_handler();
    const bool unmaskable =
        active->pending_memory_trap_ == Wavefront::MemoryTrapDelivery::Unmaskable;
    if (enters || unmaskable) {
      drain_async_window();
      if (!active->wait_counters().empty()) {
        active->set_wait_all();
        return IssueOutcome::NotIssued;
      }
    }
    active->pending_memory_trap_ = Wavefront::MemoryTrapDelivery::None;
    if (enters) {
      enter_trap_handler(*active, *config, 0, active->pc);
      return IssueOutcome::NotIssued;
    }
    if (unmaskable) {
      if (active->fail_pm4_submission()) {
        abort_dispatch(active->dispatch_id());
      } else if (!(aperture_violation_handler_ && aperture_violation_handler_(*active))) {
        const bool claimed = defer_queue_exception(
            nullptr, active->queue_id(), active->process_id(), kAqlQueueApertureViolation, false,
            false, QueueExceptionDeliveryPolicy::PublishAndCancel);
        handle_instruction_failure(*active, "unhandled FLAT private atomic memory violation",
                                   claimed);
      }
      return IssueOutcome::NotIssued;
    }
  }

  std::optional<GpuVmAccess> fresh_vm_access;
  const GpuVmAccess *vm_access = active->vm_access();
  const bool using_retained_vm_access = vm_access != nullptr;
  InstructionVmSnapshot *snapshot = instruction_vm_snapshot_;
  // A callback may recursively issue another wave. Keep its fetch from
  // replacing the snapshot borrowed by this instruction's later debug probes.
  ScopedInstructionVmSnapshot nested_scope(nullptr);
  if (active->address_space() || vmid != 0) {
    if (gpu_vm_ == nullptr) {
      drain_async_window();
      util::Logger::vm("CU ", this->name(), ": wf", active->wf_id(), " HALT(MissingGpuVm) pc=0x",
                       std::hex, active->pc, std::dec, " vmid=", vmid);
      handle_terminal_vm_fault(*active, VmAccessOutcome::Faulted, active->pc,
                               "translated wave has no GPU VM");
      return IssueOutcome::NotIssued;
    }
    if (vm_access == nullptr) {
      const AddressSpaceHandle address_space = active->address_space();
      if (snapshot && snapshot->compute_unit == this) {
        auto &cached = *snapshot;
        if (cached.owner != gpu_vm_ || cached.address_space != address_space ||
            cached.vmid != vmid || !cached.access || !cached.access->is_current()) {
          cached.access =
              address_space ? gpu_vm_->snapshot(address_space) : gpu_vm_->snapshot_vmid(vmid);
          cached.owner = gpu_vm_;
          cached.address_space = address_space;
          cached.vmid = vmid;
        }
        if (cached.access)
          vm_access = &*cached.access;
      } else {
        fresh_vm_access =
            address_space ? gpu_vm_->snapshot(address_space) : gpu_vm_->snapshot_vmid(vmid);
        if (fresh_vm_access)
          vm_access = &*fresh_vm_access;
      }
    }
    if (!vm_access) {
      drain_async_window();
      util::Logger::vm("CU ", this->name(), ": wf", active->wf_id(),
                       " HALT(StaleAddressSpace) pc=0x", std::hex, active->pc, std::dec,
                       " vmid=", vmid);
      handle_terminal_vm_fault(*active, VmAccessOutcome::Faulted, active->pc,
                               "address space retired before fetch");
      return IssueOutcome::NotIssued;
    }
    if (!using_retained_vm_access && !active->same_execution_root(*vm_access)) {
      drain_async_window();
      handle_terminal_vm_fault(*active, VmAccessOutcome::Revoked, active->pc,
                               "address space root replaced before fetch");
      return IssueOutcome::NotIssued;
    }
  }
  const bool vm_address_space = vm_access != nullptr;
  const uint64_t fetch_pc = vm_address_space ? active->vm_instruction_pc() : active->pc;

  std::array<rj_code_binary_inst_t, DecodedInstructionCache::kNumWords> words{};
  constexpr uint32_t kFirstFetchBytes = InstructionCache::kFetchBytes;
  constexpr uint32_t kFirstFetchWords = kFirstFetchBytes / sizeof(words[0]);
  static_assert(kFirstFetchWords + 1 == words.size());
  const bool debugger_active = debug_active();
  if (!debugger_active)
    sync_inst_cache_debug_epoch();
  const auto fetch = [&](uint64_t pc, uint32_t *dst, uint32_t byte_count) {
    if (vm_address_space) {
      if (debugger_active)
        return vm_access->read(pc, {reinterpret_cast<std::byte *>(dst), byte_count},
                               VmAccessKind::Execute);
      return inst_cache_.fetch(*vm_access, pc, reinterpret_cast<uint8_t *>(dst), byte_count);
    }
    if (debugger_active) {
      for (uint32_t word = 0; word < byte_count / sizeof(*dst); ++word)
        dst[word] = memory_->fetch32(pc + word * sizeof(*dst));
    } else {
      inst_cache_.fetch(*memory_, pc, reinterpret_cast<uint8_t *>(dst), byte_count);
    }
    return VmAccessOutcome::Complete;
  };
  VmAccessOutcome fetch_outcome = fetch(fetch_pc, words.data(), kFirstFetchBytes);
  // GFX10 MIMG has a two-bit NSA count at bits 2:1. Only NSA=3 needs
  // a fifth word; shorter instructions must not fetch beyond the normal window.
  constexpr uint32_t kMimgEncodingAndNsaMask = 0xfc000006u;
  constexpr uint32_t kMimgWithThreeNsaWords = 0xf0000006u;
  if (fetch_outcome == VmAccessOutcome::Complete &&
      (arch() == ROCJITSU_CODE_ARCH_RDNA1 || arch() == ROCJITSU_CODE_ARCH_RDNA2) &&
      (words[0] & kMimgEncodingAndNsaMask) == kMimgWithThreeNsaWords)
    fetch_outcome = fetch(fetch_pc + kFirstFetchBytes, &words.back(), sizeof(words.back()));

  if (fetch_outcome != VmAccessOutcome::Complete) {
    drain_async_window();
    if (fetch_outcome == VmAccessOutcome::Revoked) {
      // Invalidation deliberately revokes pinned snapshots. Drop this wave's
      // retained fast path so the next issue can capture the new translation
      // epoch, or report a terminal fault if the binding is gone.
      if (using_retained_vm_access)
        active->set_vm_access({});
      request_functional_yield();
      return IssueOutcome::NotIssued;
    }
    if (fetch_outcome == VmAccessOutcome::Unavailable) {
      request_functional_yield();
      return IssueOutcome::NotIssued;
    }
    if (fetch_outcome == VmAccessOutcome::Faulted && memory_violation_handler_ &&
        memory_violation_handler_(*active, fetch_pc, false, active->pc)) {
      return IssueOutcome::NotIssued;
    }
    util::Logger::vm("CU ", this->name(), ": wf", active->wf_id(),
                     " HALT(InstructionVmAccess) pc=0x", std::hex, active->pc, std::dec,
                     " vmid=", vmid, " outcome=", static_cast<unsigned>(fetch_outcome));
    handle_terminal_vm_fault(*active, fetch_outcome, active->pc);
    return IssueOutcome::NotIssued;
  }

  const bool reuse_decoded = !debugger_active;
  DecodeResult decoded = decoded_inst_cache_.decode(*decoder_, active->pc, vmid, words,
                                                    decode_error_.emitter(), reuse_decoded);
  if (decoded.failed()) {
    ++active->trace_inst_count_;
    drain_async_window();
    // Preserve debugger-first handling of illegal instructions. PM4 failures
    // remain exclusive to their submission and are handled below.
    const bool exception_claimed =
        !active->fail_pm4_submission() && illegal_inst_handler_ && illegal_inst_handler_(*active);
    // Only a debugger-owned stop can be repaired and resumed. Runtime ownership
    // covers notification delivery, not recovery of the simulator.
    if (exception_claimed && active->debug_halted())
      return IssueOutcome::NotIssued;
    active->report_instruction_execution_error(InstructionExecutionError::DecodeFailure);
    const std::string failure = std::format(
        "CU {}: wf{} could not decode instruction at pc={:#x} (pid={} qid={} dispatch={} wg={}) "
        "words=[{:#x},{:#x},{:#x},{:#x}]: {}",
        this->name(), active->wf_id(), active->pc, active->process_id(), active->queue_id(),
        active->dispatch_id(), active->wg_id(), words[0], words[1], words[2], words[3],
        decode_error_.message());
    decode_error_({});
    handle_instruction_failure(*active, failure, exception_claimed);
    return IssueOutcome::NotIssued;
  }
  Instruction *inst = decoded.value().get();
  DecodedInstructionCache::ScopedReturn return_decoded(decoded_inst_cache_, decoded.value(),
                                                       active->pc, vmid, words, reuse_decoded);
  const auto fail_current_instruction = [&]() {
    return_decoded.discard();
    if constexpr (EnableAsync) {
      if (window)
        window->drain();
    }
    fail_wave_instruction(*active, inst->mnemonic(), active->pc,
                          active->instruction_execution_error(),
                          // Taken before reporting, so a throwing failure handler
                          // cannot leave it for the next instruction.
                          std::exchange(execution_failure_detail_, {}));
  };
  // Operand resolution outside execute_instruction() participates in the same
  // per-instruction failure contract as the generated execution callback.
  active->begin_instruction_execution();

  const bool gfx10 =
      config_.arch == ROCJITSU_CODE_ARCH_RDNA1 || config_.arch == ROCJITSU_CODE_ARCH_RDNA2;
  const bool code_end =
      (gfx10 || config_.arch == ROCJITSU_CODE_ARCH_RDNA3 ||
       config_.arch == ROCJITSU_CODE_ARCH_RDNA3_5 || config_.arch == ROCJITSU_CODE_ARCH_RDNA4 ||
       config_.arch == ROCJITSU_CODE_ARCH_CDNA5) &&
      inst->mnemonic() == "s_code_end";
  const bool software_trap = inst->flags() & SOFTWARE_TRAP;
  const auto trap_config =
      (code_end || software_trap) && !active->in_trap_handler() && trap_handler_resolver_
          ? trap_handler_resolver_(*active)
          : std::nullopt;
  const bool enters_trap = trap_config && trap_config->tba != 0;
  const bool ordered_end = gfx10 && inst->mnemonic() == "s_endpgm_ordered_ps_done";
  if (enters_trap || code_end || ordered_end) {
    // Drain older requests before publishing the trap or ordered pixel completion.
    drain_async_window();
    active->set_wait_all();
    if (!active->wait_satisfied())
      return IssueOutcome::NotIssued;
  }
  ++active->trace_inst_count_;

  int inst_size_signed = inst->size();
  assert(inst_size_signed > 0 && "instruction size must be positive");
  auto inst_size = static_cast<uint64_t>(inst_size_signed);
  // Skipped vector instructions never issue: no register/plugin effects, wait
  // counter changes, or async matrix submission. Scalar instructions still run.
  if ((active->mode_raw() & Wavefront::VSKIP_BIT) && inst->is_vskip_affected()) {
    active->pc += inst_size;
    return IssueOutcome::NotIssued;
  }
  // VINTERP consumes its operands only after WAIT_EXP is satisfied. Park on
  // the instruction itself so a late LDS parameter return is visible on retry.
  if (inst->has_embedded_memory_wait() &&
      (config_.arch == ROCJITSU_CODE_ARCH_RDNA3 || config_.arch == ROCJITSU_CODE_ARCH_RDNA3_5) &&
      active->wait_counters().expcnt) {
    const auto threshold = waitcheck_detail::WaitcheckTarget::vinterp_wait_exp(*inst);
    if (threshold && active->wait_counters().expcnt > *threshold) {
      active->wait_for({.expcnt = static_cast<uint8_t>(*threshold)});
      --active->trace_inst_count_;
      return IssueOutcome::NotIssued;
    }
  }

  // A decoded instruction may publish memory through a cache, a translated
  // access, or an async pipeline. NOP and ENDPGM create no new memory effect;
  // all other instructions conservatively block completion rebinding.
  if (active->needs_execution_effect_mark() && inst->mnemonic() != "s_nop" &&
      inst->mnemonic() != "s_endpgm")
    active->note_potential_execution_effect();
  auto *wait_state =
      config_.memory_wait_checks_enabled() ? active->memory_wait_scoreboard() : nullptr;
  if (wait_state) {
    if (inst->is_waitcnt() || inst->has_embedded_memory_wait() || inst->is_memory_wait_producer() ||
        (config_.arch == ROCJITSU_CODE_ARCH_CDNA5 &&
         waitcheck_detail::WaitcheckTarget::is_xcnt_drain(*inst)))
      wait_state->before(*inst, config_.arch);
    wait_state->bind(active->pc, active, &ComputeUnitCore::report_memory_wait);
  }
  const bool wait_for_buffer_lanes =
      wait_state && !wait_state->empty() && inst->is_memory_wait_producer() &&
      MemoryWaitScoreboard::needs_buffer_payload_lanes(*inst, config_.arch);
  bool wait_checked = false;

  if constexpr (EnableAsync) {
    bool may_submit = true;
    std::optional<uint64_t> issuer;
    auto *admission = storage ? storage->admission : nullptr;
    if (window && window->take_issuer(fetch_pc)) {
      may_submit = false;
      if (admission)
        ++admission->stats.issuer;
    } else if (admission && matrix_coexecution::async_candidate(inst->mnemonic())) {
      // Capacity can become available between this check and submit_mma().
      // Skipping lookahead must keep this instruction inline in that case.
      may_submit = false;
      if (async_pool().available()) {
        MmaAdmissionCache::Words first;
        std::ranges::copy_n(words.begin(), first.size(), first.begin());
        issuer = admission->inspect(*decoder_, inst_cache_, *memory_, vm_access, fetch_pc, vmid,
                                    active->num_vgprs(), storage->has_accvgprs, first);
        may_submit = issuer.has_value();
      }
    }
    if (may_submit && !window && storage && matrix_coexecution::async_candidate(inst->mnemonic()) &&
        async_pool().available())
      window = &storage->window.emplace(*this, *active, storage->has_accvgprs);
    if (window) {
      window->before(*inst);
      if (wait_state && !wait_for_buffer_lanes) {
        wait_state->check_instruction(*inst, *active);
        wait_checked = true;
      }
      const bool submitted = may_submit && window->submit_mma(decoded.value());
      if (submitted) {
        check_static_isa_diagnostics(*inst, *active);
        if (issuer)
          window->reserve_issuer(*issuer);
        // Async execution bypasses execute_instruction(), but the submitted
        // instruction still ends the immediately-adjacent setreg hazard.
        active->clear_setreg_vgpr_msb_hazard();
        if (observes_async_instruction_issued_)
          plugin_group_->onAmdgpuAsyncInstructionIssued(active->pc, *inst, *active);
        active->pc += inst_size;
        record_gws_instruction_progress(active->process_id());
        return IssueOutcome::Issued;
      }
    }
  }

  if (wait_state && !wait_checked && !wait_for_buffer_lanes)
    wait_state->check_instruction(*inst, *active);

  struct ProbeWaitIssue {
    MemoryWaitScoreboard::IssueSnapshot before;
    uint64_t pc;
    uint64_t wave_generation;
  };
  std::optional<ProbeWaitIssue> probe_wait_issue;
  if (inst->is_memory_wait_producer() && config_.memory_wait_checks_enabled()) {
    if (debugger_active)
      probe_wait_issue = ProbeWaitIssue{active->ensure_memory_wait_scoreboard().issue_snapshot(),
                                        active->pc, active->dispatch_generation()};
    if (!wait_for_buffer_lanes)
      track_memory_wait(*inst, *active);
  }

  // The cause classifiers report into this as they run; see alu_exceptions.h.
  active->clear_pending_alu_causes();

  if constexpr (util::Logger::group_enabled(util::Logger::GROUP_VM)) {
    if (active->num_vgprs_ > 0) {
      util::Logger::vm([&](auto &os) {
        uint32_t vb = active->vgpr_alloc().base;
        os << std::format("{} wg[{}] wf[{}] EXECUTE #{} pc={:#x} {} sz={}", full_path(),
                          active->wg_id(), active->wf_id(), active->trace_inst_count_, active->pc,
                          inst->mnemonic(), inst_size);
        os << " enc=";
        for (uint64_t w = 0; w < inst_size / 4; ++w)
          os << std::format("{}{:08x}", w ? "," : "", words[w]);
        os << std::format(" scc={} vcc={:x} exec={:x}", active->read_scc(), active->vcc(),
                          active->exec());
        uint32_t nvr = std::min(active->num_vgprs_, 16u);
        for (uint32_t ln = 0; ln < active->wf_size_; ++ln) {
          os << std::format("\n[rj log VM]  PRE L{}: v[0:{}]=", ln, nvr - 1);
          for (uint32_t r = 0; r < nvr; ++r)
            os << std::format("{}{:x}", r ? "," : "", read_vgpr(vb + r, ln));
        }
      });
    }
  }

  const auto observe_before_execute = [&] {
    if (observes_before_execute_instruction_)
      plugin_group_->onAmdgpuBeforeExecuteInstruction(
          active->pc, *inst, *active,
          std::span(words).first(std::max<size_t>(kFirstFetchWords, inst_size / 4)));
  };

  // s_trap enters the per-process handler configured by SET_TRAP_HANDLER. The
  // hardware saves the interrupted PC/status in TTMPs and begins fetching at
  // TBA. The handler advances TTMP0:1 for software traps, sends the KFD
  // interrupt message, restores STATUS, and returns through s_rfe_b64.
  if (code_end || software_trap) {
    if (code_end)
      active->set_trapsts(active->trapsts() | (1u << 11)); // EXCP_FLAG_PRIV.ILLEGAL_INST
    // A trap stalled on older memory has not issued yet. Deliver the before
    // hook and end the setreg hazard slot only when the instruction proceeds.
    const auto begin_trap = [&] {
      observe_before_execute();
      active->clear_setreg_vgpr_msb_hazard();
    };
    const uint32_t trap_id =
        code_end ? 0u : words[0] & (active->uses_separate_trap_ctrl() ? 0xFu : 0xFFu);
    if (enters_trap) {
      begin_trap();
      if (wait_state)
        wait_state->clear();
      WaitFields drained{};
      for (auto counter : {WaitCounterKind::Load, WaitCounterKind::Store, WaitCounterKind::Ds,
                           WaitCounterKind::Exp, WaitCounterKind::Sample, WaitCounterKind::Bvh,
                           WaitCounterKind::Async, WaitCounterKind::Tensor, WaitCounterKind::Km})
        drained[static_cast<size_t>(counter)] = 0;
      plugin_group_->onAmdgpuImplicitMemoryWait(*active, drained);
      const auto saved_pc = active->pc;
      enter_trap_handler(*active, *trap_config, trap_id, saved_pc);
      if (software_trap && observes_after_execute_instruction_)
        plugin_group_->onAmdgpuAfterExecuteInstruction(saved_pc, *inst, *active);
      return IssueOutcome::Issued;
    }
    begin_trap();
    if (code_end) {
      const bool delivered =
          !active->fail_pm4_submission() && illegal_inst_handler_ && illegal_inst_handler_(*active);
      if (delivered && active->debug_halted())
        return IssueOutcome::NotIssued;
      active->set_status_raw(active->status_raw() | Wavefront::kStatusFatalHaltMask);
      active->report_instruction_execution_error(InstructionExecutionError::IllegalInstruction);
      handle_instruction_failure(*active, "s_code_end issued", delivered);
      return IssueOutcome::NotIssued;
    }
    active->pc += inst_size;
    return IssueOutcome::Issued;
  }

  observe_before_execute();

  if (!gfx10 && (inst->flags() & (INDIRECT_BRANCH | INDIRECT_CALL))) {
    auto mn = std::string_view(inst->mnemonic());
    if (mn.find("s_setpc") != std::string_view::npos ||
        mn.find("s_swappc") != std::string_view::npos) {
      const Operand *target_operand = inst->src_operand(0);
      assert(target_operand && "indirect PC instruction must have a target operand");
      uint64_t target = RegisterAccess(*active).read_scalar64(*target_operand);
      if (active->instruction_execution_failed()) [[unlikely]] {
        fail_current_instruction();
        return IssueOutcome::Issued;
      }
      if (target == 0) {
        // The null-PC return sentinel terminates like s_endpgm, including its
        // implicit memory wait before registers and the wave slot are freed.
        active->end();
        return IssueOutcome::Issued;
      }
    }
  }

  // Sampled before execute so the trap-return test below can be a state
  // transition rather than a per-ISA mnemonic list. See its use.
  const bool was_in_trap_handler = active->in_trap_handler();

  const uint32_t executing_process = active->process_id();
  const util::Result execution_result =
      execute_instruction(inst, *active, &execution_failure_detail_);
  if (wait_for_buffer_lanes) {
    wait_state->check_instruction(*inst, *active);
    track_memory_wait(*inst, *active);
  }

  if (execution_result.failed()) [[unlikely]] {
    fail_current_instruction();
    return IssueOutcome::Issued;
  }

  record_gws_instruction_progress(executing_process);

  // A terminating instruction (s_endpgm with no pending waits) halts the wave
  // inside execute_instruction, which frees and resets its slot. Its registers,
  // pc, and allocations are now zeroed, so the after-execute hook, result logging,
  // and pc-advance below must not run on the dead slot. The dedicated
  // onAmdgpuWavefrontHalted hook already fired (with live state) from halt().
  // The completed s_endpgm can return to the decoded cache on scope exit.
  //
  // Note the intentional asymmetry: an s_endpgm that defers to ENDING (pending
  // memory waits) is NOT halted here, so it DOES fire onAmdgpuAfterExecuteInstruction
  // below; the immediate-halt case does not. onAmdgpuWavefrontHalted is the
  // authoritative terminal hook and fires in both cases — consumers should observe
  // termination there, not via the after-execute hook.
  if (active->is_halted()) {
    return IssueOutcome::Issued;
  }

  if (observes_after_execute_instruction_)
    plugin_group_->onAmdgpuAfterExecuteInstruction(active->pc, *inst, *active);

  if constexpr (util::Logger::group_enabled(util::Logger::GROUP_VM)) {
    if (active->num_vgprs_ > 0) {
      util::Logger::vm([&](auto &os) {
        uint32_t vb = active->vgpr_alloc().base;
        os << std::format("RESULT #{} scc={} vcc={:x} exec={:x}", active->trace_inst_count_,
                          active->read_scc(), active->vcc(), active->exec());
        uint32_t nvr = std::min(active->num_vgprs_, 16u);
        for (uint32_t ln = 0; ln < active->wf_size_; ++ln) {
          os << std::format("\n[rj log VM]  POST L{}: v[0:{}]=", ln, nvr - 1);
          for (uint32_t r = 0; r < nvr; ++r)
            os << std::format("{}{:x}", r ? "," : "", read_vgpr(vb + r, ln));
        }
      });
    }
  }

  if (is_tensor_dma_instruction(*inst)) {
    const VmAccessOutcome tensor_outcome = tensor_dma_outcome(*inst);
    if (tensor_outcome == VmAccessOutcome::Unavailable) {
      tensor_dma_pipeline_.defer_unavailable(decoded.value().release(), *active);
      active->pc += inst_size;
      return IssueOutcome::Issued;
    }
    if (tensor_outcome != VmAccessOutcome::Complete) {
      drain_async_window();
      util::Logger::vm("CU ", this->name(), ": wf", active->wf_id(),
                       " HALT(TensorDmaVmAccess) pc=0x", std::hex, active->pc, std::dec,
                       " vmid=", vmid, " outcome=", static_cast<unsigned>(tensor_outcome));
      const uint64_t issue_pc = active->pc;
      active->pc += inst_size;
      handle_memory_access_failure(*active, tensor_outcome, issue_pc);
      return IssueOutcome::Issued;
    }
    active->pc += inst_size;
    return IssueOutcome::Issued;
  }

  // Capture debugger probe info before the pipeline consumes the instruction.
  // The checks run after the PC advances (below) so that a wave stopped on a
  // watchpoint or memory fault resumes past the access instead of re-executing
  // it. Gated on debug_active_ so non-debugged runs pay no per-access cost.
  const bool debug_probe = debugger_active && inst->is_memory_op() && inst->data() &&
                           (inst->data()->tag() == GLOBAL_MEM || inst->data()->tag() == SCALAR_MEM);
  struct DebugMemoryRange : MemoryByteRange {
    bool shared;
  };
  std::vector<DebugMemoryRange> debug_ranges;
  bool dbg_is_write = false;
  bool dbg_is_atomic = false;
  if (debug_probe) {
    const uint64_t shared_base =
        active->use_wave_apertures() ? active->shared_aperture_base() : shared_aperture_base_;
    const uint64_t shared_limit =
        active->use_wave_apertures() ? active->shared_aperture_limit() : shared_aperture_limit_;
    const auto add_range = [&](MemoryByteRange range,
                               std::optional<bool> flat_shared = std::nullopt) {
      debug_ranges.push_back({range, flat_shared.value_or(flat_in_shared_aperture(
                                         range.address, shared_base, shared_limit))});
    };
    if (inst->data()->tag() == SCALAR_MEM) {
      auto &d = *inst->data_as<ScalarMemState>();
      dbg_is_atomic = d.atomic_op != AtomicOp::NONE;
      dbg_is_write = !d.is_load || dbg_is_atomic;
      if (!d.memviol && d.load_dword_mask) {
        if (!dbg_is_atomic && d.elem_size >= sizeof(uint32_t) && d.load_dword_mask != 0xffff) {
          for (uint32_t i = 0; i < d.num_dwords; ++i)
            if (d.load_dword_mask & (1u << i))
              add_range({d.addr + i * sizeof(uint32_t), sizeof(uint32_t)});
        } else {
          add_range(
              {d.addr, std::max(1u, dbg_is_atomic ? d.elem_size : d.num_dwords * d.elem_size)});
        }
      }
    } else {
      auto &d = *inst->data_as<VectorMemState>();
      dbg_is_atomic = (d.atomic_op != AtomicOp::NONE);
      dbg_is_write = !d.is_load || dbg_is_atomic;
      // std::max on both arms: VectorMemState::elem_size defaults to 0 and the
      // range check below reports a zero-sized range as *unmapped*, so an
      // atomic whose decoder left elem_size unset would fault on every lane of
      // a perfectly mapped buffer.
      const uint32_t bytes = std::max(1u, dbg_is_atomic ? d.elem_size : d.num_elems * d.elem_size);
      // The debugger and the issued transfer must agree even if a callback
      // republishes the sparse ranges or replaces the address-space root.
      const bool check_residency = d.check_texel_residency && active->address_space() &&
                                   vm_access && vm_access->info().ready;
      const bool check_buffer = d.reports_buffer_residency() && active->address_space() &&
                                vm_access && vm_access->info().ready;
      const auto ranges = check_residency ? vm_access->nonresident_prt_ranges() : nullptr;
      if (check_residency || check_buffer) {
        d.translated.access = *vm_access;
        d.translated.initialized = true;
      }
      if (check_buffer)
        qualify_buffer_residency(d, *vm_access);
      const auto &element_lanes = d.element_lane_masks;
      debug_ranges.reserve(d.wf_size);
      const bool flat = inst->mnemonic().starts_with("flat_");
      auto record = [&](uint64_t lanes, std::span<const uint64_t> addresses,
                        TranslatedMemoryProgress &progress) {
        if (d.rejects_private_atomic(lanes))
          return;
        if (check_residency) {
          progress.nonresident_lanes = nonresident_texel_lanes(d, addresses, lanes, ranges.get());
          lanes &= ~*progress.nonresident_lanes;
        }
        while (lanes) {
          const auto lane = std::countr_zero(lanes);
          lanes &= lanes - 1;
          std::optional<bool> flat_shared;
          if (flat) {
            const uint64_t pointer =
                d.per_lane_addr[lane] -
                (flat_selects_segment_before_offset(arch()) ? d.flat_aperture_offset : 0);
            flat_shared = !(d.scratch_lane_mask & (uint64_t{1} << lane)) &&
                          flat_in_shared_aperture(pointer, shared_base, shared_limit);
            if (*flat_shared) {
              const uint64_t offset = d.per_lane_addr[lane] - shared_base;
              if (flat_offset_is_dds(arch(), offset) ||
                  flat_shared_atomic_misaligned(arch(), d, offset)) {
                debug_ranges.clear();
                return;
              }
            }
          }
          if (d.scratch_swizzle || !element_lanes.empty()) {
            any_texel_byte_range(d, lane, addresses[lane], element_lanes,
                                 [&](MemoryByteRange range) {
                                   add_range(range, flat_shared);
                                   return false;
                                 });
          } else {
            add_range({addresses[lane], bytes}, flat_shared);
          }
        }
      };
      const uint64_t lanes =
          transpose_request_lane_mask(d, active->wf_size()) & active->lane_mask();
      if (d.image_sample) {
        for (auto &tap : d.image_sample->taps)
          record(tap.lane_mask & lanes, tap.addresses, tap.translated);
      } else {
        record(lanes, d.per_lane_addr, d.translated);
      }
    }
  }
  // One predicate for the whole access, used both to decide that this
  // instruction faulted and to pick the address reported to the debugger below.
  // The two must agree: if the decision said "faulted" and the report loop then
  // found no faulting address, the access would be dropped with nothing
  // delivered. Checking only the first byte let a multi-byte load, store or
  // atomic that starts near the end of a mapped page run off it unnoticed, so
  // each actual byte range is validated.
  const uint32_t dbg_vmid = active->process_id();
  auto access_faults = [&](const DebugMemoryRange &range) {
    const uint64_t addr = range.address;
    if (range.shared)
      return false;
    if (vm_address_space) {
      const VmAccessKind access = dbg_is_atomic  ? VmAccessKind::Atomic
                                  : dbg_is_write ? VmAccessKind::Write
                                                 : VmAccessKind::Read;
      return vm_access->query_access(addr, range.bytes, access) == VmAccessOutcome::Faulted;
    }
    return dbg_vmid != 0;
  };
  // Locate the faulting address once. The report loop below resumes from this
  // iterator rather than rescanning: each access_faults() call is a page-table
  // walk per touched page.
  auto first_fault = debug_ranges.end();
  if (debug_probe && memory_violation_handler_)
    first_fault = std::ranges::find_if(debug_ranges, access_faults);
  const bool debug_memory_fault = first_fault != debug_ranges.end();

  // The trap return is whatever instruction just left the handler. Asking the
  // wave rather than matching mnemonics keeps this ISA-agnostic: the spelling
  // differs per ISA (gfx1250 uses s_rfe_i64), and a hard-coded list silently
  // stops calling notify_trap_complete() for any spelling it misses -- KFD is
  // never told the handler returned and the debugger hangs. Only the generated
  // s_rfe body clears the flag.
  const bool trap_return = was_in_trap_handler && !active->in_trap_handler();

  // A frontend may claim an architectural fault for either a debugger or a
  // runtime exception owner. The probe is CU-wide, so an attached debugger
  // also enables it for other processes. Only Faulted queries are offered:
  // transient translation failures, revoked snapshots and invalid requests
  // remain on the typed memory pipeline's retry/cancellation path.
  //
  // Report against the resume PC. The frontend can restore the issue PC for
  // XNACK; if it declines, undo only our advance before routing the instruction
  // normally. A declining handler may itself have changed the PC.
  const uint64_t issue_pc = active->pc;
  bool fault_claimed = false;
  if (debug_memory_fault && !active->debug_halted()) {
    active->pc += inst_size;
    // first_fault is already known to fault, so it is reported without a second
    // range walk; only the lanes after it still have to be tested.
    for (auto it = first_fault; it != debug_ranges.end(); ++it) {
      if ((it == first_fault || access_faults(*it)) &&
          memory_violation_handler_(*active, it->address, dbg_is_write, issue_pc)) {
        fault_claimed = true;
        break;
      }
    }
    // Only undo our own advance. A declining handler is still free to have
    // moved the wave (entering a trap handler, for instance); clobbering that
    // would resume the application with the handler's state half-installed.
    if (!fault_claimed && active->pc == issue_pc + inst_size)
      active->pc = issue_pc;
  }

  if (fault_claimed) {
    if (probe_wait_issue && active->dispatch_generation() == probe_wait_issue->wave_generation) {
      // The probe bypassed pipeline issue. Cancel exactly the positions tracked
      // before execution, including X and the selected FLAT domains, even when
      // the claimant changed MODE or registers. A reset wave already cleared them.
      [[maybe_unused]] const bool cancelled = active->memory_wait_scoreboard()->cancel_last_issue(
          probe_wait_issue->pc, probe_wait_issue->before);
      assert(cancelled && "Claimed memory probe has no matching wait-check issue");
    }
    return IssueOutcome::Issued;
  }
  if (inst->is_memory_op()) {
    if (!inst->data()) {
      // A memory execute path can intentionally reject an invalid complete
      // register operand before constructing pipeline state. Treat that as a
      // fully suppressed instruction: no route callback, wait-counter update,
      // or memory transaction is permitted.
      decoded.value().reset();
    } else {
      if (inst->data()->tag() == GLOBAL_MEM) {
        auto *d = inst->data_as<VectorMemState>();
        d->issue_pc = active->pc;
      }
      const VmAccessOutcome memory_outcome = route_memory_inst(decoded.value().release(), *active);
      if (memory_outcome != VmAccessOutcome::Complete) {
        if (memory_outcome == VmAccessOutcome::Unavailable) {
          request_functional_yield();
          return IssueOutcome::Issued;
        }
        // Ordinary memory instructions may overlap independent matrix work.
        // Join that work before a terminal fault releases the wave's registers.
        drain_async_window();
        util::Logger::vm("CU ", this->name(), ": wf", active->wf_id(), " HALT(DataVmAccess) pc=0x",
                         std::hex, active->pc, std::dec, " vmid=", vmid,
                         " outcome=", static_cast<unsigned>(memory_outcome));
        active->pc += inst_size;
        handle_memory_access_failure(*active, memory_outcome, issue_pc);
        return IssueOutcome::Issued;
      }
      if (active->state() == WfState::MEMORY_TRAP) {
        active->pc += inst_size;
        deliver_memory_trap(*active);
        return IssueOutcome::Issued;
      }
    }
  }

  active->pc += inst_size;

  // Deliver on the causes this instruction raised, not on TRAPSTS changing.
  // TRAPSTS.EXCP is sticky and nothing in the model clears it between
  // instructions, so a rising-edge test went permanently quiet for any cause
  // whose bit was already latched: raise a cause with its MODE.EXCP_EN bit
  // clear, enable it, repeat the operation, and the second occurrence -- the
  // one hardware traps on -- was never reported. The same silence followed a
  // first occurrence the handler declined, which is every occurrence before a
  // debugger attaches.
  // Masks come from alu_exceptions.h, which is also what the classifiers and
  // the generated call sites use. A local copy here would keep checking the
  // old bits if the EXCP set ever widened.
  const uint32_t new_alu_causes = active->pending_alu_causes() & kAluExceptionTrapstsMask;
  if (new_alu_causes != 0) {
    const uint32_t enabled_alu_causes = alu_exception_trap_enables(*active);
    if ((new_alu_causes & enabled_alu_causes) != 0 && alu_exception_handler_ &&
        alu_exception_handler_(*active))
      return IssueOutcome::Issued;
  }

  // s_rfe follows the same target-minus-size convention as other control-flow
  // instructions. Publish the handler-driven stop only after the common PC
  // increment has produced the architectural return PC for CWSR serialization.
  // On GFX10 the trap handler explicitly restores application EXEC before s_rfe.
  if (trap_return && active->debug_halted() && active->trap_interrupt_sent())
    notify_trap_complete(*active);

  // Watchpoints, after the access completed and the PC advanced (so the
  // serialized wave resumes at the next instruction). A memory fault is more
  // severe than a watchpoint and wins if both would fire on the same
  // instruction, which it does by returning above before reaching here.
  if (debug_probe && !active->debug_halted() && watchpoint_handler_) {
    for (const auto range : debug_ranges)
      if (watchpoint_handler_(*active, range.address, range.bytes, dbg_is_write, dbg_is_atomic))
        break;
  }
  return IssueOutcome::Issued;
}

// Keep ordinary issue and step as concrete entry points. Async execution
// uses a separate entry selected when constructing the CU.
ComputeUnitCore::IssueOutcome ComputeUnitCore::issue_instruction(Wavefront *active) {
  return issue_instruction_impl<false>(active);
}

template <bool EnableAsync>
[[gnu::always_inline]] inline bool
ComputeUnitCore::step_impl(MmaAdmissionCache *admission, uint32_t async_wave_size,
                           bool has_accvgprs, uint64_t selected_slots, uint64_t *issued) {
  // A wave reaching s_endpgm in this loop retires its workgroup; the guard sends
  // the CP its completion after the lock is released. See WaveStateGuard.
  WaveStateGuard wave_state_lock(*this);
  tick_pipelines();
  update_wf_states();

  for (uint32_t slot = 0; slot < active_slot_limit_; ++slot) {
    auto &wf = wfs_[slot];
    if (!wf)
      continue;
    if (wf->state() == WfState::RUNNING && !wf->debug_paused()) {
      // Burn down an in-flight S_SLEEP before issuing anything else. A
      // single-step request cancels the remainder instead of spending the
      // debugger's one step on it, which would look like a hung wave.
      if (wf->nop_ready_cycle_ > cycle_counter_ && !wf->debug_single_step())
        continue;
      if (wf->sleep_cycles() != 0) {
        if (!wf->debug_single_step()) {
          wf->tick_sleep();
          continue;
        }
        wf->set_sleep_cycles(0);
      }
      const uint64_t slot_bit = uint64_t{1} << wf->wf_id();
      if ((selected_slots & slot_bit) == 0)
        continue;
      // A pending unmaskable exception owns the stop, including one raised by this issue.
      const auto unmaskable_trap_pending = [&] {
        return wf->pending_memory_trap_ == Wavefront::MemoryTrapDelivery::Unmaskable;
      };
      const bool single_step = wf->debug_single_step() && !unmaskable_trap_pending();
      const uint64_t step_pc = wf->pc;
      if constexpr (EnableAsync) {
        issue_async_instruction(wf.get(), admission, async_wave_size, has_accvgprs);
        if (wf->is_halted()) {
          if (admission)
            admission->flush();
        }
      } else if (issue_instruction(wf.get()) == IssueOutcome::Issued && issued) {
        *issued |= slot_bit;
      }
      if (single_step && !unmaskable_trap_pending() && !wf->in_trap_handler() &&
          !wf->debug_halted() && single_step_handler_) {
        // A trap can park the instruction before it issues.
        // S_WAITCNT itself advances the PC even when it parks the wave.
        const auto state = wf->state();
        const bool waiting_to_issue = state == WfState::WAITCNT;
        if (!waiting_to_issue || wf->pc != step_pc)
          single_step_handler_(*wf);
      }
    }
  }

  if (!gws_progress_batch_active_)
    publish_gws_instruction_progress();

  ++step_count_;
  if constexpr (util::Logger::group_enabled(util::Logger::GROUP_CP)) {
    if ((step_count_ & 0xFFFFF) == 0) {
      util::Logger::cp([&](auto &os) {
        os << std::format("CU[{}] steps={}M", full_path(), step_count_ >> 20);
        for (auto &wf : wfs_) {
          if (!wf)
            continue;
          auto st = wf->state();
          // Include GWS_WAIT: a CU keeps stepping with parked GWS waves, so omitting
          // them hides their PCs exactly when a stalled rendezvous is being probed.
          if (st == WfState::RUNNING || st == WfState::WAITCNT || st == WfState::BARRIER ||
              st == WfState::GWS_WAIT)
            os << std::format(" wf{}:pc={:#x}:{}", wf->wf_id(), wf->pc,
                              st == WfState::RUNNING    ? "R"
                              : st == WfState::WAITCNT  ? "W"
                              : st == WfState::GWS_WAIT ? "G"
                                                        : "B");
        }
      });
    }
  }

  return has_runnable_wfs();
}

bool ComputeUnitCore::step() { return step_impl<false>(); }

uint64_t ComputeUnitCore::run_issue_cycle(uint64_t selected_slots) {
  assert(wfs_.size() <= 64 && "issue-cycle slot masks cover at most 64 wave slots");
  // The ownership a driver can lose without noticing: CommandProcessor's
  // add_compute_unit() and set_dispatch_threads() rewrite pool-driven mode,
  // after which the serial tick or the dispatch pool also issues for this CU.
  assert((!engine() || pool_driven()) &&
         "a clocked driver must own the CU's execution: set_pool_driven(true)");
  assert((!cp_ || cp_->dispatch_threads() <= 1) &&
         "a clocked driver needs a single CP dispatch thread, or the pool runs this CU");
  const InstructionFetchScope fetch_scope(*this);
  // The clocked driver times each instruction itself, so it takes the
  // ordinary issue path rather than the batched asynchronous MMA window.
  uint64_t issued = 0;
  (void)step_impl<false>(nullptr, 0, false, selected_slots, &issued);
  return issued;
}

// Clang 17 and 18 do not emit this specialization for the local class in
// create_functional_cu; keep it explicit for the supported toolchain floor.
template bool ComputeUnitCore::step_impl<true>(MmaAdmissionCache *, uint32_t, bool, uint64_t,
                                               uint64_t *);

// Explicit template instantiations for all AMDGPU ISAs and execution modes.
#define ROCJITSU_CU_INSTANTIATE(ISA_TYPE)                                                          \
  template class IsaExecComputeUnit<simdojo::ExecMode::FUNCTIONAL, ISA_TYPE>;                      \
  template class IsaExecComputeUnit<simdojo::ExecMode::CLOCKED, ISA_TYPE>

ROCJITSU_CU_INSTANTIATE(cdna1::Isa);
ROCJITSU_CU_INSTANTIATE(cdna2::Isa);
ROCJITSU_CU_INSTANTIATE(cdna3::Isa);
ROCJITSU_CU_INSTANTIATE(cdna4::Isa);
ROCJITSU_CU_INSTANTIATE(rdna1::Isa);
ROCJITSU_CU_INSTANTIATE(rdna2::Isa);
ROCJITSU_CU_INSTANTIATE(rdna3::Isa);
ROCJITSU_CU_INSTANTIATE(rdna3_5::Isa);
ROCJITSU_CU_INSTANTIATE(rdna4::Isa);
ROCJITSU_CU_INSTANTIATE(cdna5::Isa);

#undef ROCJITSU_CU_INSTANTIATE

} // namespace amdgpu
} // namespace rocjitsu
