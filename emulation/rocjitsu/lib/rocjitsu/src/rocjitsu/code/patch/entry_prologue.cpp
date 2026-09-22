// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "rocjitsu/code/patch/entry_prologue.h"

#include "rocjitsu/code/analysis/free_registers.h"
#include "rocjitsu/code/builders/instruction_builder.h"
#include "rocjitsu/code/builders/smem_builders.h"
#include "rocjitsu/code/kernel_descriptor_scan.h"

#include <algorithm>
#include <array>
#include <tuple>
#include <utility>

namespace rocjitsu {

namespace {
using KD = rocr::llvm::amdhsa::kernel_descriptor_t;
} // namespace

uint32_t dbi_entry_storage_floor(KernelBlockScope blocks, const KD &desc, rj_code_arch_t arch) {
  return std::max(explicit_ordinary_sgpr_bound(blocks),
                  kernel_descriptor_initial_sgpr_count(arch, desc));
}

std::optional<DbiEntryStorage> plan_dbi_entry_storage(KernelBlockScope blocks, const KD &desc,
                                                      rj_code_arch_t arch,
                                                      uint32_t kernel_sgpr_count,
                                                      const RegisterSet &reserved,
                                                      std::string *error_out) {
  const uint32_t floor = dbi_entry_storage_floor(blocks, desc, arch);
  const uint32_t bound = std::min<uint32_t>(kernel_sgpr_count, REGISTER_SET_ALLOCATABLE_SGPRS);

  // find_free_run takes a uint16_t search start. Rejecting a floor at or past
  // the bound here keeps a floor that outgrew the type from narrowing into a low
  // index that looks available.
  if (floor >= bound) {
    if (error_out != nullptr)
      *error_out = "kernel and its ABI reach s" + std::to_string(floor) + ", and it allocates " +
                   std::to_string(kernel_sgpr_count) + ", leaving no room for the " +
                   std::to_string(kDbiEntryStorageRegisters) + " SGPRs the entry prologue reserves";
    return std::nullopt;
  }

  const auto base = find_free_run(reserved, RegClass::SGPR, kDbiEntryStorageRegisters,
                                  static_cast<uint16_t>(floor), /*base_alignment=*/2, bound);
  if (!base) {
    if (error_out != nullptr)
      *error_out = "no free SGPR pair run of " + std::to_string(kDbiEntryStorageRegisters) +
                   " above s" + std::to_string(floor) + " within the kernel's " +
                   std::to_string(kernel_sgpr_count) +
                   "-SGPR allocation for the entry prologue's reserved storage";
    return std::nullopt;
  }

  return DbiEntryStorage{.persistent_base = *base,
                         .entry_temp_base = static_cast<uint16_t>(*base + 2)};
}

std::optional<DbiEntryPrologue> build_dbi_entry_prologue(const KD &desc, rj_code_arch_t arch,
                                                         DbiEntryStorage storage,
                                                         std::string *error_out) {
  // The sequence, for storage s[res:res+3] and kernarg pair s[karg:karg+1]:
  //
  //   s_load_dwordx2 s[res  :res+1], s[karg:karg+1], payload_offset
  //   s_load_dwordx2 s[res+2:res+3], s[karg:karg+1], original_ptr_offset
  //   s_waitcnt lgkmcnt(0)
  //   s_mov_b32 s[karg  ], s[res+2]
  //   s_mov_b32 s[karg+1], s[res+3]
  //
  // Both loads go first: the CP delivers a pointer to the rocjitsu wrapper, and
  // the restore overwrites it, after which nothing in the wrapper is reachable.
  // The temp pair exists because the second load cannot name the kernarg pair as
  // both destination and address. The wait has to sit ahead of the restore, not
  // merely ahead of the reads below it, since the restore writes the SBASE both
  // loads are still in flight against.
  //
  // Everything before the emission is a guard on a case that would otherwise
  // miscompile silently. The register-field ones are also enforced by the
  // builders, which throw; repeating them keeps a fail-closed caller from having
  // to catch.
  const auto fail = [error_out](std::string message) {
    if (error_out != nullptr)
      *error_out = std::move(message);
    return std::nullopt;
  };

  const std::optional<uint16_t> kernarg_base = kernarg_segment_ptr_sgpr(desc);
  if (!kernarg_base) {
    return fail("kernel does not enable ENABLE_SGPR_KERNARG_SEGMENT_PTR, so the entry "
                "prologue has no pointer to load the DBI payload through");
  }

  // The kernarg pair is the loads' address operand and the storage pairs are
  // their destinations, so the two ride different SMEM fields with different
  // widths. Checked against the field each one actually occupies.
  const std::array<std::tuple<const char *, uint16_t, uint16_t, const char *>, 3> pairs{{
      {"kernarg segment", *kernarg_base, kMaxSmemSbase, "address"},
      {"persistent storage", storage.persistent_base, kMaxSmemSdata, "data"},
      {"entry temp storage", storage.entry_temp_base, kMaxSmemSdata, "data"},
  }};
  for (const auto &[name, base, limit, field] : pairs) {
    if ((base % 2) != 0)
      return fail(std::string(name) + " pair base s" + std::to_string(base) + " is not even");
    if (base > limit) {
      return fail(std::string(name) + " pair base s" + std::to_string(base) + " exceeds the SMEM " +
                  field + " field");
    }
  }

  // One aligned run is the only shape plan_dbi_entry_storage produces and the
  // only shape the rest of this function is correct for. A hand-built or
  // value-initialized DbiEntryStorage can name the same pair twice, and then the
  // second load silently overwrites the payload pointer with the guest's.
  if (storage.entry_temp_base != storage.persistent_base + 2) {
    return fail("entry storage pairs s[" + std::to_string(storage.persistent_base) + ":" +
                std::to_string(storage.persistent_base + 1) + "] and s[" +
                std::to_string(storage.entry_temp_base) + ":" +
                std::to_string(storage.entry_temp_base + 1) + "] are not one contiguous run");
  }

  // Both loads address through the kernarg pair and the restore writes it, so a
  // run covering it would corrupt the address mid-prologue or lose the payload
  // pointer to the restore.
  const uint32_t run_last = storage.persistent_base + kDbiEntryStorageRegisters - 1u;
  if (run_last >= *kernarg_base && *kernarg_base + 1u >= storage.persistent_base) {
    return fail("entry storage s[" + std::to_string(storage.persistent_base) + ":" +
                std::to_string(run_last) + "] overlaps the kernarg segment pair s[" +
                std::to_string(*kernarg_base) + ":" + std::to_string(*kernarg_base + 1) + "]");
  }

  const std::array<KernargExtensionPayloadLayout, 1> payloads{kDbiEntryPayloadLayout};
  const std::optional<KernargExtensionLayout> layout =
      make_kernarg_extension_layout(desc.kernarg_size, payloads);
  if (!layout)
    return fail("kernarg wrapper layout overflows 32 bits for this kernel's kernarg_size");

  DbiEntryPrologue prologue;
  prologue.payload_byte_offset = layout->payload_offsets.front();
  prologue.original_kernarg_pointer_offset = layout->original_kernarg_pointer_offset;

  const uint32_t offset_max = max_smem_byte_offset(arch);
  for (const uint32_t offset :
       {prologue.payload_byte_offset, prologue.original_kernarg_pointer_offset}) {
    if (offset > offset_max) {
      return fail("kernarg wrapper offset " + std::to_string(offset) +
                  " exceeds the target's SMEM immediate range of " + std::to_string(offset_max));
    }
  }

  const auto emit_load = [&](uint16_t sdst, uint32_t byte_offset) {
    const std::array<uint32_t, 2> load =
        build_s_load_dwordx2(sdst, *kernarg_base, byte_offset, arch);
    prologue.words.push_back(load[0]);
    prologue.words.push_back(load[1]);
  };

  emit_load(storage.persistent_base, prologue.payload_byte_offset);
  emit_load(storage.entry_temp_base, prologue.original_kernarg_pointer_offset);

  prologue.words.push_back(build_wait_scalar_loads_complete(arch));

  prologue.words.push_back(build_s_mov_b32(*kernarg_base, storage.entry_temp_base, arch));
  prologue.words.push_back(build_s_mov_b32(static_cast<uint16_t>(*kernarg_base + 1),
                                           static_cast<uint16_t>(storage.entry_temp_base + 1),
                                           arch));
  return prologue;
}

std::optional<DbiEntryProloguePlan> plan_dbi_entry_prologue(KernelBlockScope blocks, const KD &desc,
                                                            rj_code_arch_t arch,
                                                            uint32_t kernel_sgpr_count,
                                                            const RegisterSet &reserved,
                                                            std::string *error_out) {
  // A kernel compiled with kernarg preloading has two hardware entries 256 bytes
  // apart. A prologue reached through only one of them leaves the storage
  // uninitialized on the other path, and nothing here can patch a firmware entry
  // that the descriptor does not name.
  if (const uint32_t preload = kernarg_preload_length(desc); preload != 0) {
    if (error_out != nullptr) {
      *error_out = "kernel preloads " + std::to_string(preload) +
                   " kernarg dwords, so it has a second hardware entry the DBI entry "
                   "prologue would not cover";
    }
    return std::nullopt;
  }

  const std::optional<DbiEntryStorage> storage =
      plan_dbi_entry_storage(blocks, desc, arch, kernel_sgpr_count, reserved, error_out);
  if (!storage)
    return std::nullopt;

  std::optional<DbiEntryPrologue> prologue =
      build_dbi_entry_prologue(desc, arch, *storage, error_out);
  if (!prologue)
    return std::nullopt;

  return DbiEntryProloguePlan{.storage = *storage, .prologue = std::move(*prologue)};
}

} // namespace rocjitsu
