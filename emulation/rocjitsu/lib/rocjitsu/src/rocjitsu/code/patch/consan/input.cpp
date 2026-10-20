// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "rocjitsu/code/patch/consan/input.h"
#include "rocjitsu/code/consan/image/metadata.h"

#include "rocjitsu/code/amdgpu_code_object.h"
#include "rocjitsu/code/amdgpu_elf.h"
#include "rocjitsu/code/analysis/def_use_chain.h"
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
  if (image.size() < sizeof(header))
    return reject(error, InputErrorCode::InvalidImage, "truncated ELF header");
  std::memcpy(&header, image.data(), sizeof(header));
  if (std::memcmp(header.e_ident, EI_MAGIC, EI_MAGIC_SIZE) != 0 ||
      header.e_ident[EI_CLASS] != ELFCLASS64 || header.e_ident[EI_DATA] != ELFDATA2LSB ||
      header.e_ident[EI_OSABI] != ELFOSABI_AMDGPU_HSA || header.e_type != ET_DYN ||
      header.e_machine != EM_AMDGPU)
    return reject(error, InputErrorCode::InvalidImage,
                  "expected little-endian AMDHSA ET_DYN ELF64");
  if (header.e_ident[EI_ABIVERSION] != ELFABIVERSION_AMDGPU_HSA_V5 ||
      (header.e_flags & EF_AMDGPU_MACH) != profile.elf_machine ||
      (header.e_flags & ~uint32_t{0xfff}) != 0)
    return reject(error, InputErrorCode::UnsupportedTarget,
                  "concrete target requires matching code-object v5");
  return Result::success();
}

Result validate_entry_abi(const kd::kernel_descriptor_t &descriptor, const TargetProfile &target,
                          InputError &error) {
  constexpr uint32_t kProperties = kd::KERNEL_CODE_PROPERTY_ENABLE_SGPR_KERNARG_SEGMENT_PTR |
                                   kd::KERNEL_CODE_PROPERTY_ENABLE_WAVEFRONT_SIZE32;
  if ((descriptor.kernel_code_properties & ~kProperties) != 0)
    return reject(error, InputErrorCode::UnsupportedEntry, "unsupported kernel entry properties");
  if ((descriptor.kernel_code_properties &
       kd::KERNEL_CODE_PROPERTY_ENABLE_SGPR_KERNARG_SEGMENT_PTR) == 0)
    return reject(error, InputErrorCode::UnsupportedEntry,
                  "entry requires the kernarg SGPR pointer");

  const uint32_t user_sgprs = kernel_descriptor_user_sgpr_count(target.arch, descriptor);
  if (user_sgprs < 2 || user_sgprs > 16)
    return reject(error, InputErrorCode::UnsupportedEntry, "unsupported entry user SGPR count");
  const uint32_t wave_size = kernel_wavefront_size(target.arch, descriptor);
  if (!supports_wave(target, wave_size))
    return reject(error, InputErrorCode::UnsupportedEntry, "unsupported kernel wave mode");
  // The gfx1100 wave32 CP initialization workaround needs sixteen inputs even
  // when the program uses fewer; scratch must also start beyond those inputs.
  if (target.id == 1100 && wave_size == 32 &&
      kernel_descriptor_initial_sgpr_count(target.arch, descriptor) < 16)
    return reject(error, InputErrorCode::UnsupportedEntry,
                  "gfx1100 wave32 requires input SGPR padding");

  // The shared bootstrap restores a conventional argument pointer; preload
  // and private-frame setup are outside this entry protocol.
  if (kernarg_preload_length(descriptor) != 0 || descriptor.private_segment_fixed_size != 0 ||
      descriptor.kernarg_size == 0)
    return reject(error, InputErrorCode::UnsupportedEntry,
                  "entry requires nonempty kernargs, no preload and no private frame");
  if (RJ_AMDHSA_BITS_GET(descriptor.compute_pgm_rsrc2,
                         kd::COMPUTE_PGM_RSRC2_ENABLE_VGPR_WORKITEM_ID) != 0 ||
      RJ_AMDHSA_BITS_GET(descriptor.compute_pgm_rsrc2,
                         kd::COMPUTE_PGM_RSRC2_ENABLE_SGPR_WORKGROUP_ID_Y) != 0 ||
      RJ_AMDHSA_BITS_GET(descriptor.compute_pgm_rsrc2,
                         kd::COMPUTE_PGM_RSRC2_ENABLE_SGPR_WORKGROUP_ID_Z) != 0 ||
      RJ_AMDHSA_BITS_GET(descriptor.compute_pgm_rsrc2,
                         kd::COMPUTE_PGM_RSRC2_ENABLE_SGPR_WORKGROUP_INFO) != 0)
    return reject(error, InputErrorCode::UnsupportedEntry,
                  "entry supports workitem X and optional workgroup X only");
  return Result::success();
}

Result collect_original_code(const std::vector<std::unique_ptr<BasicBlock>> &blocks,
                             InputProgram &program, RegisterSet &used, const InputLimits &limits,
                             InputError &error) {
  for (const std::unique_ptr<BasicBlock> &block : blocks) {
    if (block->has_implicit_terminator() ||
        block->successor_issue() != BasicBlock::SuccessorIssue::None ||
        !block->call_edges().empty())
      return reject(error, InputErrorCode::UnsupportedFlow, "original control flow is not closed",
                    block->start_offset());
    if (qualify_lds_race_block(*block, error).failed())
      return Result::failure();
    for (const Instruction &instruction : block->instructions()) {
      if (program.instructions.size() >= limits.instructions)
        return reject(error, InputErrorCode::ResourceLimit, "instruction analysis budget exceeded");
      const uint64_t offset = instruction.src_loc();
      if ((instruction.flags() & (INDIRECT_BRANCH | INDIRECT_CALL)) != 0)
        return reject(error, InputErrorCode::UnsupportedFlow, "indirect transfer is outside MVP",
                      offset);
      if (classify_lds_race_instruction(instruction, program.target->arch, program.original_vgprs,
                                        program.accesses, error)
              .failed())
        return Result::failure();
      const InstDefUse effects(instruction);
      used |= effects.uses;
      used |= effects.defs;
      OriginalInstruction original{.offset = offset,
                                   .words = {},
                                   .branch_target = std::nullopt,
                                   .terminates = (instruction.flags() & PROGRAM_TERMINATOR) != 0};
      original.words.assign(instruction.raw_encoding(),
                            instruction.raw_encoding() + instruction.size() / sizeof(uint32_t));
      const std::optional<int64_t> branch = instruction.branch_offset_bytes();
      if (branch) {
        const int64_t destination = static_cast<int64_t>(offset) + instruction.size() + *branch;
        if (destination < 0 || static_cast<uint64_t>(destination) >= program.text_bytes)
          return reject(error, InputErrorCode::UnsupportedFlow, "branch leaves original text",
                        offset);
        original.branch_target = static_cast<uint64_t>(destination);
      }
      program.instructions.push_back(std::move(original));
    }
  }
  if (program.accesses.empty())
    return reject(error, InputErrorCode::UnsupportedAccess,
                  "selected entry has no native LDS sites");
  std::ranges::sort(program.instructions, {}, &OriginalInstruction::offset);
  std::ranges::sort(program.accesses, {}, &NativeLdsAccess::original_offset);
  for (size_t index = 0; index < program.accesses.size(); ++index)
    program.accesses[index].site = static_cast<uint32_t>(index);
  return Result::success();
}

Result plan_context_bootstrap(const std::vector<std::unique_ptr<BasicBlock>> &blocks,
                              const RegisterSet &used, uint32_t declared_vgprs,
                              InputProgram &program, InputError &error) {
  bool invalid_register = false;
  used.for_each_ordinary([&](RegisterRef reg) {
    if ((reg.cls == RegClass::SGPR && reg.index >= program.original_sgprs) ||
        (reg.cls == RegClass::VGPR && reg.index >= declared_vgprs) || reg.cls == RegClass::ACC_VGPR)
      invalid_register = true;
  });
  if (invalid_register)
    return reject(error, InputErrorCode::UnsupportedEntry,
                  "original operands exceed declared registers");
  std::vector<BasicBlock *> scope;
  for (const std::unique_ptr<BasicBlock> &block : blocks)
    scope.push_back(block.get());
  RegisterSet reserved;
  reserved.expand({RegClass::SGPR, 0, static_cast<uint8_t>(program.original_sgprs)});
  std::string setup_error;
  const std::optional<DbiEntryProloguePlan> setup = plan_dbi_entry_prologue(
      scope, program.kernel.descriptor, program.target->arch, 106, reserved, &setup_error);
  if (!setup)
    return reject(error, InputErrorCode::ResourceLimit, "unsafe context bootstrap: " + setup_error);
  program.entry_plan = *setup;
  return Result::success();
}

} // namespace

FailureOr<InputProgram> build_input_program(std::span<const uint8_t> image,
                                            const TargetProfile &target, std::string_view entry,
                                            InputError &error, const InputLimits &limits) {
  error = {};
  if (image.size() > limits.image_bytes)
    return reject(error, InputErrorCode::ResourceLimit, "code object exceeds image budget");

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
  if (profile == nullptr)
    return reject(error, InputErrorCode::UnsupportedTarget,
                  "unsupported or inconsistent native LDS target profile");
  const rj_code_arch_t arch = profile->arch;
  const IsaTargetRegistry &registry = default_isa_target_registry();
  const IsaGpuTargetDescription *binding =
      registry.find_gpu_target_by_code_object_id(profile->name);
  const IsaTargetDescriptor *provider = registry.find(profile->public_id);
  if (!registry.ok() ||
      (profile->public_id != ROCJITSU_CODE_TARGET_INVALID &&
       (binding == nullptr || provider == nullptr || binding->elf_machine != profile->elf_machine ||
        provider->architecture_id != arch)))
    return reject(error, InputErrorCode::UnsupportedTarget, "selected ISA provider is unavailable");
  if (validate_image_header(image, *profile, error).failed())
    return Result::failure();

  // Establish the original binary/entry/metadata bounds before CFG decoding.
  AmdGpuCodeObject object(image.data(), image.size());
  if (!object.is_valid() || object.text_sections().size() != 1 ||
      object.allocated_executable_sections().size() != 1)
    return reject(error, InputErrorCode::InvalidImage,
                  "expected one valid executable .text section");
  const Section &text = *object.text_sections().front();
  if (text.size() > limits.text_bytes)
    return reject(error, InputErrorCode::ResourceLimit, "text exceeds analysis budget");
  const std::vector<KernelDescriptorInfo> kernels =
      scan_kernel_descriptors(image, text.sectionOffset(), text.size(), text.sectionHeaderIndex());
  if (kernels.size() != 1 || kernels.front().kernel_name != entry)
    return reject(error, InputErrorCode::UnsupportedEntry, "expected one selected kernel entry");
  const KernelDescriptorInfo &kernel = kernels.front();
  if (validate_entry_abi(kernel.descriptor, *profile, error).failed())
    return Result::failure();
  InputProgram program;
  program.kernel = kernel;
  program.target = profile;
  program.wave_size = kernel_wavefront_size(arch, kernel.descriptor);
  if (qualify_lds_race_entry(program, error).failed())
    return Result::failure();
  if (text.vaddr() > std::numeric_limits<uint64_t>::max() - kernel.entry_text_offset ||
      ((text.vaddr() + kernel.entry_text_offset) % kKernelEntryAlignment) != 0)
    return reject(error, InputErrorCode::UnsupportedEntry,
                  "hardware entry is not 256-byte aligned");
  CodeObjectPatcher patcher(object);
  if (patcher.has_relocations_within_text() || patcher.has_unsupported_relocation_to_text() ||
      patcher.has_malformed_rocr_relocation_section() ||
      patcher.has_rocr_rejected_none_relocation())
    return reject(error, InputErrorCode::UnsupportedFlow, "unsupported code relocation");
  std::string metadata_error;
  const FailureOr<MetadataResources> metadata = inspect_metadata(image, kernel, &metadata_error);
  if (metadata.failed())
    return reject(error, InputErrorCode::UnsupportedEntry, metadata_error);
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
  // gfx1100 has no concrete binding in this base: exact ELF identity above plus
  // the legacy RDNA3 decoder is the explicit fallback for that target alone.
  std::unique_ptr<Decoder> decoder = profile->public_id == ROCJITSU_CODE_TARGET_INVALID
                                         ? Decoder::create(registry, arch)
                                         : Decoder::create(registry, profile->public_id);
  if (decoder == nullptr)
    return reject(error, InputErrorCode::DecodeFailure, "concrete target decoder is unavailable");
  const std::array<uint64_t, 1> entries = {kernel.entry_text_offset};
  std::string decode_message;
  const auto sink = [&decode_message](std::string_view message) { decode_message = message; };
  const FailureOr<std::vector<std::unique_ptr<BasicBlock>>> blocks = BasicBlock::build_reachable(
      object, *decoder, arch, entries, util::DiagnosticEmitter{util::DiagnosticSink{sink}}, {}, {},
      {}, profile->public_id);
  if (blocks.failed())
    return reject(error, InputErrorCode::DecodeFailure, "invalid original CFG: " + decode_message);
  RegisterSet used;
  if (collect_original_code(blocks.value(), program, used, limits, error).failed() ||
      plan_context_bootstrap(blocks.value(), used, metadata.value().vgprs, program, error).failed())
    return Result::failure();
  program.identity = {.sha256 = digest(image), .bytes = image.size()};
  program.image.assign(image.begin(), image.end());
  return program;
}

} // namespace rocjitsu::consan
