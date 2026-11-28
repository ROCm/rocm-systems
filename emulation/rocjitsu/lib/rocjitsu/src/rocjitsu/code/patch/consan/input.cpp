// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "rocjitsu/code/patch/consan/input.h"
#include "rocjitsu/code/consan/image/metadata.h"
#include "rocjitsu/code/patch/consan/local_helper.h"
#include "rocjitsu/isa/isa_family.h"

#include "rocjitsu/code/amdgpu_code_object.h"
#include "rocjitsu/code/amdgpu_elf.h"
#include "rocjitsu/code/analysis/def_use_chain.h"
#include "rocjitsu/code/analysis/exec_state.h"
#include "rocjitsu/code/basic_block.h"
#include "rocjitsu/code/patch/code_object_patcher.h"
#include "rocjitsu/code/patch/kernel_text_layout.h"
#include "rocjitsu/isa/decoder.h"
#include "rocjitsu/isa/instruction.h"
#include "rocjitsu/isa/target_registry.h"

#include <algorithm>
#include <array>
#include <cstring>
#include <limits>
#include <unordered_map>
#include <utility>

namespace rocjitsu::consan {
namespace {

namespace kd = rocjitsu::amdhsa;

Result reject(InputError &error, InputErrorCode code, std::string message,
              std::optional<uint64_t> offset = std::nullopt) {
  error = {.code = code, .text_offset = offset, .message = std::move(message)};
  return Result::failure();
}

Result validate_image_header(std::span<const uint8_t> image, const TargetProfile &profile,
                             InputError &error) {
  Elf64_Ehdr header{};
  if (image.size() < sizeof(header)) {
    return reject(error, InputErrorCode::InvalidImage, "truncated ELF header");
  }
  std::memcpy(&header, image.data(), sizeof(header));
  if (std::memcmp(header.e_ident, EI_MAGIC, EI_MAGIC_SIZE) != 0 ||
      header.e_ident[EI_CLASS] != ELFCLASS64 || header.e_ident[EI_DATA] != ELFDATA2LSB ||
      header.e_ident[EI_OSABI] != ELFOSABI_AMDGPU_HSA || header.e_type != ET_DYN ||
      header.e_machine != EM_AMDGPU) {
    return reject(error, InputErrorCode::InvalidImage,
                  "expected little-endian AMDHSA ET_DYN ELF64");
  }
  if (header.e_ident[EI_ABIVERSION] != ELFABIVERSION_AMDGPU_HSA_V5 ||
      (header.e_flags & EF_AMDGPU_MACH) != profile.elf_machine ||
      (header.e_flags & ~uint32_t{0xfff}) != 0) {
    return reject(error, InputErrorCode::UnsupportedTarget,
                  "concrete target requires matching code-object v5");
  }
  return Result::success();
}

Result validate_entry_abi(const kd::kernel_descriptor_t &descriptor, const TargetProfile &target,
                          InputError &error) {
  constexpr uint32_t kProperties = kd::KERNEL_CODE_PROPERTY_ENABLE_SGPR_KERNARG_SEGMENT_PTR |
                                   kd::KERNEL_CODE_PROPERTY_ENABLE_WAVEFRONT_SIZE32;
  if ((descriptor.kernel_code_properties & ~kProperties) != 0) {
    return reject(error, InputErrorCode::UnsupportedEntry, "unsupported kernel entry properties");
  }
  if ((descriptor.kernel_code_properties &
       kd::KERNEL_CODE_PROPERTY_ENABLE_SGPR_KERNARG_SEGMENT_PTR) == 0) {
    return reject(error, InputErrorCode::UnsupportedEntry,
                  "entry requires the kernarg SGPR pointer");
  }

  const uint32_t user_sgprs = kernel_descriptor_user_sgpr_count(target.arch, descriptor);
  if (user_sgprs < 2 || user_sgprs > 16) {
    return reject(error, InputErrorCode::UnsupportedEntry, "unsupported entry user SGPR count");
  }
  const uint32_t wave_size = kernel_wavefront_size(target.arch, descriptor);
  if (!supports_wave(target, wave_size)) {
    return reject(error, InputErrorCode::UnsupportedEntry, "unsupported kernel wave mode");
  }
  // The gfx1100 wave32 CP initialization workaround needs sixteen inputs even
  // when the program uses fewer; scratch must also start beyond those inputs.
  if (target.id == 1100 && wave_size == 32 &&
      kernel_descriptor_initial_sgpr_count(target.arch, descriptor) < 16) {
    return reject(error, InputErrorCode::UnsupportedEntry,
                  "gfx1100 wave32 requires input SGPR padding");
  }

  // The shared bootstrap restores a conventional argument pointer; preload
  // and private-frame setup are outside this entry protocol.
  if (kernarg_preload_length(descriptor) != 0 || descriptor.private_segment_fixed_size != 0 ||
      descriptor.kernarg_size == 0) {
    return reject(error, InputErrorCode::UnsupportedEntry,
                  "entry requires nonempty kernargs, no preload and no private frame");
  }
  if (RJ_AMDHSA_BITS_GET(descriptor.compute_pgm_rsrc2,
                         kd::COMPUTE_PGM_RSRC2_ENABLE_VGPR_WORKITEM_ID) != 0 ||
      RJ_AMDHSA_BITS_GET(descriptor.compute_pgm_rsrc2,
                         kd::COMPUTE_PGM_RSRC2_ENABLE_SGPR_WORKGROUP_ID_Y) != 0 ||
      RJ_AMDHSA_BITS_GET(descriptor.compute_pgm_rsrc2,
                         kd::COMPUTE_PGM_RSRC2_ENABLE_SGPR_WORKGROUP_ID_Z) != 0 ||
      RJ_AMDHSA_BITS_GET(descriptor.compute_pgm_rsrc2,
                         kd::COMPUTE_PGM_RSRC2_ENABLE_SGPR_WORKGROUP_INFO) != 0) {
    return reject(error, InputErrorCode::UnsupportedEntry,
                  "entry supports workitem X and optional workgroup X only");
  }
  return Result::success();
}

// Track a shared or private aperture high word. A vector
// definition under unknown EXEC proves this only for lanes active at that
// point. A later EXEC change can expose older values in the other lanes.
enum class ApertureSelector { SharedBase = 235, PrivateBase = 237 };

struct ApertureWords {
  RegisterSet active;
  RegisterSet all_lanes;

  void transfer(const Instruction &inst, const InstDefUse &effects, ExecState exec,
                std::optional<ApertureSelector> base) {
    using Op = amdgpu::Operation;
    const Op op = inst.amdgpu_operation();
    const auto *source = inst.num_src_operands() ? inst.src_operand(0) : nullptr;
    const auto *destination = inst.num_dst_operands() ? inst.dst_operand(0) : nullptr;
    std::optional<RegisterRef> proven;
    bool all = false;
    bool writes_exec =
        (inst.flags() & WRITES_EXEC) || effects.defs.contains({RegClass::EXEC, 0, 1});
    if (source && destination) {
      const auto dst = destination->to_register_ref();
      const auto src = source->to_register_ref();
      writes_exec |= dst && dst->cls == RegClass::EXEC;
      if (dst && op == Op::S_MOV_B64 && dst->cls == RegClass::SGPR && dst->width == 2 && base &&
          source->has_register_selector() && source->encoding_value() == static_cast<int>(*base)) {
        proven = RegisterRef{RegClass::SGPR, static_cast<uint16_t>(dst->index + 1), 1};
        all = true;
      } else if (op == Op::S_MOV_B64 && dst && src && dst->cls == RegClass::SGPR &&
                 src->cls == RegClass::SGPR && dst->width == 2 && src->width == 2 &&
                 active.contains({RegClass::SGPR, static_cast<uint16_t>(src->index + 1), 1})) {
        proven = RegisterRef{RegClass::SGPR, static_cast<uint16_t>(dst->index + 1), 1};
        all = true;
      } else if (op == Op::S_CSELECT_B32 && dst && dst->cls == RegClass::SGPR && src &&
                 active.contains(*src) && inst.num_src_operands() >= 2) {
        // Both SGPR inputs retain their common aperture class whichever SCC selects.
        const auto rhs = inst.src_operand(1)->to_register_ref();
        if (rhs && active.contains(*rhs)) {
          proven = dst;
          all = true;
        }
      } else if (dst && src && dst->width == 1 && src->width == 1 && active.contains(*src) &&
                 ((op == Op::S_MOV_B32 && dst->cls == RegClass::SGPR &&
                   src->cls == RegClass::SGPR) ||
                  (op == Op::V_MOV_B32 && inst.size() == 4 && dst->cls == RegClass::VGPR))) {
        proven = dst;
        all = dst->cls == RegClass::SGPR || (exec == ExecState::Full && all_lanes.contains(*src));
      }
    }

    active -= effects.defs;
    all_lanes -= effects.defs;
    if (effects.conditional_vgpr_def) {
      active.erase(*effects.conditional_vgpr_def);
      all_lanes.erase(*effects.conditional_vgpr_def);
    }
    if (writes_exec && op != Op::S_AND_SAVEEXEC_B32 && op != Op::S_AND_SAVEEXEC_B64 &&
        op != Op::V_CMPX_EQ_U32)
      active = all_lanes;
    if (proven) {
      active.expand(*proven);
      if (all)
        all_lanes.expand(*proven);
    }
  }
};

// Track bytes that hold one value across the current EXEC lanes. The
// original-entry whitelist excludes VGPR bank and indexing mode changes.
class BlockUniformBytes {
public:
  UniformStoreProof store_proof(const Instruction &store, const NativeLdsAccess &site) const {
    if (site.origin != AccessOrigin::Native || site.kind != AccessKind::Write ||
        site.lds_ranges.shape != LdsRangeShape::Single)
      return UniformStoreProof::None;
    // The native classifier checked the ADDR and DATA0 allocation and tuple
    // widths. The decoder identifies the selected DATA0 bytes in each word.
    const Operand &payload = *store.src_operand(1);
    const RegisterRef address = *store.src_operand(0)->decoded_vgpr();
    const RegisterRef data = *payload.decoded_vgpr();
    const unsigned selected = payload.register_byte_mask();
    return uniform_bytes(address) == 0xf && selected && (uniform_bytes(data) & selected) == selected
               ? UniformStoreProof::BlockLocalBroadcast
               : UniformStoreProof::None;
  }

  void transfer(const Instruction &inst, const InstDefUse &effects) {
    if (writes_execution_mask(inst) && !intersects_execution_mask(inst))
      bytes_ = {};
    const std::optional<RegisterRef> dst =
        inst.num_dst_operands() ? inst.dst_operand(0)->decoded_vgpr() : std::nullopt;
    // Read the old destination or an aliased shift source before killing defs.
    const unsigned derived = dst ? defined_bytes(inst, *dst) : 0;
    for (unsigned byte = 0; byte < bytes_.size(); ++byte) {
      bytes_[byte] -= effects.defs;
      if (effects.conditional_vgpr_def)
        bytes_[byte].erase(*effects.conditional_vgpr_def);
      if (derived & (1u << byte))
        bytes_[byte].expand(*dst);
    }
  }

private:
  unsigned uniform_bytes(RegisterRef reg) const {
    unsigned mask = 0;
    for (unsigned byte = 0; byte < bytes_.size(); ++byte) {
      if (bytes_[byte].contains(reg))
        mask |= 1u << byte;
    }
    return mask;
  }

  unsigned defined_bytes(const Instruction &inst, RegisterRef dst) const {
    if (inst.has_d16_memory_data() &&
        inst.amdgpu_memory_addressing() == amdgpu::MemoryAddressing::Lds &&
        inst.amdgpu_memory_kind() == amdgpu::MemoryKind::Load && dst.width == 1) {
      amdgpu::RegisterModifiers modifiers;
      inst.amdgpu_register_modifiers(modifiers);
      // The other half is preserved or cleared uniformly on SRAM-ECC targets.
      // Neither case introduces variation into a previously uniform byte.
      return uniform_bytes(dst) & ~modifiers.memory_result_bytes;
    }
    // Only plain four-byte vector operations have these lane and operand
    // semantics; extended encodings can select other lanes or partial bytes.
    if (inst.size() != 4 || dst.width != 1)
      return 0;
    using Op = amdgpu::Operation;
    switch (inst.amdgpu_operation()) {
    case Op::V_MOV_B32: {
      const Operand &source = *inst.src_operand(0);
      const std::optional<RegisterRef> reg = source.to_register_ref();
      return source.const_value() || (reg && reg->cls == RegClass::SGPR) ? 0xf : 0;
    }
    case Op::V_LSHLREV_B32:
    case Op::V_LSHRREV_B32: {
      const std::optional<RegisterRef> source = inst.src_operand(1)->decoded_vgpr();
      if (!source || source->width != 1 || inst.src_operand(0)->const_value() != 8)
        return 0;
      const unsigned input = uniform_bytes(*source);
      return inst.amdgpu_operation() == Op::V_LSHLREV_B32 ? (input << 1) & 0xf : input >> 1;
    }
    default:
      return 0;
    }
  }

  static bool intersects_execution_mask(const Instruction &inst) {
    using Op = amdgpu::Operation;
    const std::optional<RegisterRef> dst =
        inst.num_dst_operands() ? inst.dst_operand(0)->to_register_ref() : std::nullopt;
    switch (inst.amdgpu_operation()) {
    case Op::S_AND_SAVEEXEC_B32:
    case Op::S_AND_SAVEEXEC_B64:
      // A separate SGPR destination receives the saved mask.
      return dst && dst->cls == RegClass::SGPR;
    case Op::S_AND_B32:
    case Op::S_AND_B64:
      // The intersection must read the same EXEC bits it writes. Copying
      // bits from exec_lo into exec_hi could activate different lanes.
      return dst && dst->cls == RegClass::EXEC &&
             (dst == inst.src_operand(0)->to_register_ref() ||
              dst == inst.src_operand(1)->to_register_ref());
    default:
      return false;
    }
  }

  std::array<RegisterSet, 4> bytes_;
};

struct FlatPointers {
  ApertureWords shared, non_group, selected;
  void transfer(const Instruction &inst, const InstDefUse &effects, ExecState exec) {
    // Qualify a shared high word paired with an unknown SGPR or inline zero
    // (the null arm of a generic LDS pointer select).
    std::optional<RegisterRef> selected_dst;
    if (inst.amdgpu_operation() == amdgpu::Operation::S_CSELECT_B32 &&
        inst.num_src_operands() >= 2 && inst.num_dst_operands() == 1) {
      const auto shared_with_alternative = [&](const Operand &source, const Operand &alternative) {
        const auto reg = source.to_register_ref();
        if (!reg || reg->cls != RegClass::SGPR || !shared.active.contains(*reg))
          return false;
        if (const auto other = alternative.to_register_ref())
          return other->cls == RegClass::SGPR && !shared.active.contains(*other) &&
                 !non_group.active.contains(*other);
        return alternative.has_register_selector() && alternative.const_value() == 0;
      };
      const Operand &lhs = *inst.src_operand(0);
      const Operand &rhs = *inst.src_operand(1);
      const auto dst = inst.dst_operand(0)->to_register_ref();
      if (dst && dst->cls == RegClass::SGPR &&
          (shared_with_alternative(lhs, rhs) || shared_with_alternative(rhs, lhs)))
        selected_dst = dst;
    }
    shared.transfer(inst, effects, exec, ApertureSelector::SharedBase);
    non_group.transfer(inst, effects, exec, ApertureSelector::PrivateBase);
    selected.transfer(inst, effects, exec, {});
    if (selected_dst) {
      selected.active.expand(*selected_dst);
      selected.all_lanes.expand(*selected_dst);
    }
  }

  void intersect(const FlatPointers &other) {
    shared.active &= other.shared.active;
    shared.all_lanes &= other.shared.all_lanes;
    non_group.active &= other.non_group.active;
    non_group.all_lanes &= other.non_group.all_lanes;
    selected.active &= other.selected.active;
    selected.all_lanes &= other.selected.all_lanes;
  }
};

Result collect_original_code(const std::vector<std::unique_ptr<BasicBlock>> &blocks,
                             const AmdGpuCodeObject &object, InputProgram &program,
                             RegisterSet &used, const InputLimits &limits, InputError &error) {
  std::vector<BasicBlock *> scope;
  for (const auto &block : blocks)
    scope.push_back(block.get());
  const auto helper = qualify_local_helper(object, scope, program.wave_size, error);
  if (helper.failed())
    return Result::failure();
  const LocalHelper &call = helper.value();
  program.has_local_helper = call.caller != nullptr;
  const auto ordered = qualify_lds_race_cfg(blocks, call, error);
  if (ordered.failed()) {
    return Result::failure();
  }
  std::unordered_map<const BasicBlock *, const Instruction *> latest_observation;
  latest_observation.reserve(ordered.value().size());
  for (const BasicBlock *block : ordered.value()) {
    latest_observation.emplace(block, nullptr);
  }
  const auto root =
      std::ranges::find(scope, program.kernel.entry_text_offset, &BasicBlock::start_offset);
  if (root == scope.end())
    return reject(error, InputErrorCode::UnsupportedFlow,
                  "selected entry is absent from original CFG");
  const std::array<const BasicBlock *, 1> entries{*root};
  const ExecMaskAnalysis exec(scope, program.wave_size, call.edges, entries);
  std::unordered_map<const BasicBlock *, FlatPointers> outgoing;
  size_t instruction_count = 0;
  for (const BasicBlock *block : ordered.value()) {
    if (block->has_implicit_terminator() ||
        (block->successor_issue() != BasicBlock::SuccessorIssue::None &&
         !(block == call.callee &&
           block->successor_issue() == BasicBlock::SuccessorIssue::IndirectControlFlow)) ||
        (!block->call_edges().empty() && block != call.caller)) {
      return reject(error, InputErrorCode::UnsupportedFlow, "original control flow is not closed",
                    block->start_offset());
    }
    BlockUniformBytes uniform; // No definitions flow across a branch, call, or backedge.
    FlatPointers aperture;
    if (block->start_offset() == program.kernel.entry_text_offset &&
        block->predecessors().empty()) {
      // validate_entry_abi admits the kernarg pointer as the only enabled user pointer.
      // The high dword of its global address cannot be a shared aperture.
      const RegisterRef kernarg_high{RegClass::SGPR, 1, 1};
      aperture.non_group.active.expand(kernarg_high);
      aperture.non_group.all_lanes.expand(kernarg_high);
    }
    bool first = true;
    for (const BasicBlock *predecessor : call.predecessors(*block)) {
      // A deferred backedge cannot inherit first-iteration proofs.
      const auto it = outgoing.find(predecessor);
      if (it == outgoing.end()) {
        aperture = {};
        break;
      }
      if (first)
        aperture = it->second;
      else
        aperture.intersect(it->second);
      first = false;
    }
    const Instruction *latest = latest_observation.at(block);
    for (const Instruction &instruction : block->instructions()) {
      if (instruction_count++ >= limits.instructions) {
        return reject(error, InputErrorCode::ResourceLimit, "instruction analysis budget exceeded");
      }
      const uint64_t offset = instruction.src_loc();
      if ((instruction.flags() & (INDIRECT_BRANCH | INDIRECT_CALL)) != 0 &&
          !call.is_transfer(instruction)) {
        return reject(error, InputErrorCode::UnsupportedFlow, "indirect transfer is outside MVP",
                      offset);
      }
      const size_t previous_accesses = program.accesses.size();
      const FlatSitePolicy flat{
          aperture.shared.active,   aperture.non_group.active,
          aperture.selected.active, program.flat_policy,
          program.flat_exclusions,  &instruction == call.terminal_access ? call.api : std::nullopt};
      if (!call.is_transfer(instruction) &&
          classify_lds_race_instruction(instruction, program.target->arch, program.original_vgprs,
                                        program.accesses, error, &flat, program.original_sgprs)
              .failed()) {
        return Result::failure();
      }
      // Replay orders LDS sites by PC and counts preceding barrier completions
      // to derive each site's phase. Follow both across the call and return.
      const amdgpu::Operation operation = instruction.amdgpu_operation();
      const bool caller_barrier = call.caller && (operation == amdgpu::Operation::S_BARRIER ||
                                                  operation == amdgpu::Operation::S_BARRIER_WAIT);
      if (program.accesses.size() != previous_accesses || caller_barrier) {
        if (latest && offset <= latest->src_loc()) {
          if (caller_barrier || latest->is_barrier())
            return reject(
                error, InputErrorCode::UnsupportedFlow,
                "observed LDS accesses and barrier completions must follow source PC order",
                offset);
          return reject(error, InputErrorCode::UnsupportedFlow,
                        "MVP requires increasing LDS access order along every CFG path", offset);
        }
        latest = &instruction;
      }
      const InstDefUse effects(instruction, program.wave_size);
      if (program.accesses.size() != previous_accesses)
        program.accesses.back().uniform_store_proof =
            uniform.store_proof(instruction, program.accesses.back());
      uniform.transfer(instruction, effects);
      aperture.transfer(instruction, effects, exec.before(instruction));
      used |= effects.uses;
      used |= effects.defs;
      const std::optional<int64_t> branch = instruction.branch_offset_bytes();
      if (branch) {
        const int64_t destination = static_cast<int64_t>(offset) + instruction.size() + *branch;
        if (destination < 0 || static_cast<uint64_t>(destination) >= program.text_bytes) {
          return reject(error, InputErrorCode::UnsupportedFlow, "branch leaves original text",
                        offset);
        }
      }
      if ((instruction.flags() & PROGRAM_TERMINATOR) != 0) {
        program.exit_offsets.push_back(offset);
      }
    }
    outgoing.emplace(block, std::move(aperture));
    for (const BasicBlock *successor : call.successors(*block)) {
      // The deferred self-edge starts another repeated epoch if the later
      // barrier gate qualifies it; propagate this epoch's maximum only to exits.
      if (successor == block) {
        continue;
      }
      const Instruction *&incoming = latest_observation.at(successor);
      if (latest && (!incoming || latest->src_loc() > incoming->src_loc())) {
        incoming = latest;
      }
    }
  }
  std::ranges::sort(program.flat_exclusions, {}, &FlatExclusion::original_offset);
  if (program.accesses.empty()) {
    return reject(error, InputErrorCode::UnsupportedAccess,
                  program.flat_exclusions.empty()
                      ? "selected entry has no native LDS sites"
                      : "selected entry has no observed LDS sites after FLAT policy exclusions");
  }
  std::ranges::sort(program.exit_offsets);
  std::ranges::sort(program.accesses, {}, &NativeLdsAccess::original_offset);
  for (size_t index = 0; index < program.accesses.size(); ++index) {
    program.accesses[index].site = static_cast<uint32_t>(index);
  }
  return Result::success();
}

Result validate_original_registers(const RegisterSet &used, uint32_t declared_vgprs,
                                   const InputProgram &program, InputError &error) {
  bool invalid_register = false;
  used.for_each_ordinary([&](RegisterRef reg) {
    if ((reg.cls == RegClass::SGPR && reg.index >= program.original_sgprs) ||
        (reg.cls == RegClass::VGPR && reg.index >= declared_vgprs) ||
        reg.cls == RegClass::ACC_VGPR) {
      invalid_register = true;
    }
  });
  if (invalid_register) {
    return reject(error, InputErrorCode::UnsupportedEntry,
                  "original operands exceed declared registers");
  }
  return Result::success();
}

FailureOr<InputProgram> build_entry_program(std::span<const uint8_t> image,
                                            const TargetProfile &target, std::string_view entry,
                                            InputError &error, const InputLimits &limits,
                                            FlatPolicy flat_policy, bool allow_siblings) {
  error = {};
  if (image.size() > limits.image_bytes) {
    return reject(error, InputErrorCode::ResourceLimit, "code object exceeds image budget");
  }

  // Validate concrete identity before interpreting target-dependent bytes.
  // A caller may supply a temporary aggregate or a non-static name view.
  // Validate all identity fields and retain the canonical table entry, never
  // the caller's storage. This is profile validation, not string resolution.
  const TargetProfile *profile = nullptr;
  for (const TargetProfile &candidate : kNativeLdsTargets) {
    if (candidate == target) {
      profile = &candidate;
      break;
    }
  }
  if (profile == nullptr) {
    return reject(error, InputErrorCode::UnsupportedTarget,
                  "unsupported or inconsistent native LDS target profile");
  }
  const rocjitsu::IsaFamily arch = profile->arch;
  const IsaTargetRegistry &registry = default_isa_target_registry();
  const IsaTargetBinding *binding = registry.find_target_by_code_object_id(profile->name);
  const IsaTargetDescriptor *provider = registry.find(profile->public_id);
  if (!registry.ok() || binding == nullptr || provider == nullptr ||
      binding->elf_machine != profile->elf_machine || provider->family != arch) {
    return reject(error, InputErrorCode::UnsupportedTarget, "selected ISA provider is unavailable");
  }
  if (validate_image_header(image, *profile, error).failed()) {
    return Result::failure();
  }

  // Establish the original binary/entry/metadata bounds before CFG decoding.
  AmdGpuCodeObject object(image.data(), image.size());
  if (!object.is_valid() || object.text_sections().size() != 1 ||
      object.allocated_executable_sections().size() != 1) {
    return reject(error, InputErrorCode::InvalidImage,
                  "expected one valid executable .text section");
  }
  const Section &text = *object.text_sections().front();
  if (text.size() > limits.text_bytes) {
    return reject(error, InputErrorCode::ResourceLimit, "text exceeds analysis budget");
  }
  const std::vector<KernelDescriptorInfo> kernels =
      scan_kernel_descriptors(image, text.sectionOffset(), text.size(), text.sectionHeaderIndex());
  const std::vector<KernelDescriptorInfo>::const_iterator selected =
      std::ranges::find(kernels, entry, &KernelDescriptorInfo::kernel_name);
  if (selected == kernels.end()) {
    return reject(error, InputErrorCode::UnsupportedEntry,
                  "requested kernel entry '" + std::string(entry) + "' was not found");
  }
  if (!allow_siblings && kernels.size() != 1) {
    return reject(error, InputErrorCode::UnsupportedEntry,
                  "ConSan requires a single-kernel code object; found " +
                      std::to_string(kernels.size()) + " kernels");
  }
  const KernelDescriptorInfo &kernel = *selected;
  if (validate_entry_abi(kernel.descriptor, *profile, error).failed()) {
    return Result::failure();
  }
  InputProgram program;
  program.flat_policy = flat_policy;
  program.kernel = kernel;
  program.target = profile;
  program.wave_size = kernel_wavefront_size(arch, kernel.descriptor);
  if (qualify_lds_race_entry(program, error).failed()) {
    return Result::failure();
  }
  if (text.vaddr() > std::numeric_limits<uint64_t>::max() - kernel.entry_text_offset ||
      ((text.vaddr() + kernel.entry_text_offset) % kKernelEntryAlignment) != 0) {
    return reject(error, InputErrorCode::UnsupportedEntry,
                  "hardware entry is not 256-byte aligned");
  }
  CodeObjectPatcher patcher(object);
  if (patcher.has_relocations_within_text() || patcher.has_unsupported_relocation_to_text() ||
      patcher.has_malformed_rocr_relocation_section() ||
      patcher.has_rocr_rejected_none_relocation()) {
    return reject(error, InputErrorCode::UnsupportedFlow, "unsupported code relocation");
  }
  std::string metadata_error;
  const FailureOr<MetadataResources> metadata = inspect_metadata(image, kernel, &metadata_error);
  if (metadata.failed()) {
    return reject(error, InputErrorCode::UnsupportedEntry, metadata_error);
  }
  program.original_sgprs = metadata.value().sgprs;
  program.maximum_workgroup_size = metadata.value().maximum_workgroup_size;
  program.original_vgprs =
      (RJ_AMDHSA_BITS_GET(kernel.descriptor.compute_pgm_rsrc1,
                          kd::COMPUTE_PGM_RSRC1_GRANULATED_WORKITEM_VGPR_COUNT) +
       1) *
      descriptor_vgpr_granularity_for_wavefront(arch, program.wave_size);
  program.text_file_offset = text.sectionOffset();
  program.text_bytes = text.size();

  // Decoders/blocks stay alive through classification and register-use planning;
  // the returned program owns original bytes, offsets and operations only.
  std::unique_ptr<Decoder> decoder = Decoder::create(registry, profile->public_id);
  if (decoder == nullptr) {
    return reject(error, InputErrorCode::DecodeFailure, "concrete target decoder is unavailable");
  }
  const std::array<uint64_t, 1> entries = {kernel.entry_text_offset};
  std::string decode_message;
  const auto sink = [&decode_message](std::string_view message) { decode_message = message; };
  const FailureOr<std::vector<std::unique_ptr<BasicBlock>>> blocks = BasicBlock::build_reachable(
      object, *decoder, arch, entries, util::DiagnosticEmitter{util::DiagnosticSink{sink}}, {}, {},
      {}, profile->public_id);
  if (blocks.failed()) {
    return reject(error, InputErrorCode::DecodeFailure, "invalid original CFG: " + decode_message);
  }
  if (kernels.size() > 1) {
    for (const std::unique_ptr<BasicBlock> &block : blocks.value()) {
      if (!block->call_edges().empty()) {
        return reject(error, InputErrorCode::UnsupportedFlow,
                      "LDS helper calls require a single-kernel code object",
                      block->start_offset());
      }
      for (const Instruction &instruction : block->instructions()) {
        for (const KernelDescriptorInfo &sibling : kernels) {
          if (sibling.descriptor_file_offset != kernel.descriptor_file_offset &&
              instruction.src_loc() == sibling.entry_text_offset) {
            return reject(error, InputErrorCode::UnsupportedFlow,
                          "selected control flow enters another kernel", instruction.src_loc());
          }
        }
      }
    }
  }
  RegisterSet used;
  if (collect_original_code(blocks.value(), object, program, used, limits, error).failed() ||
      validate_original_registers(used, metadata.value().vgprs, program, error).failed()) {
    return Result::failure();
  }
  std::vector<BasicBlock *> scope;
  for (const std::unique_ptr<BasicBlock> &block : blocks.value()) {
    scope.push_back(block.get());
  }
  if (qualify_lds_race_barrier(program, scope, error).failed()) {
    return Result::failure();
  }
  program.identity = {.sha256 = digest(image), .bytes = image.size()};
  program.image.assign(image.begin(), image.end());
  return program;
}

} // namespace

FailureOr<InputProgram> build_input_program(std::span<const uint8_t> image,
                                            const TargetProfile &target, std::string_view entry,
                                            InputError &error, const InputLimits &limits,
                                            FlatPolicy flat_policy) {
  return build_entry_program(image, target, entry, error, limits, flat_policy,
                             /*allow_siblings=*/false);
}

FailureOr<InputProgram> build_image_entry_program(std::span<const uint8_t> image,
                                                  const TargetProfile &target,
                                                  std::string_view entry, InputError &error,
                                                  const InputLimits &limits,
                                                  FlatPolicy flat_policy) {
  return build_entry_program(image, target, entry, error, limits, flat_policy,
                             /*allow_siblings=*/true);
}

} // namespace rocjitsu::consan
