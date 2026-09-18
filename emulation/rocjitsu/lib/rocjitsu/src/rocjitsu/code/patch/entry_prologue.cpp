// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "rocjitsu/code/patch/entry_prologue.h"

#include "rocjitsu/code/analysis/free_registers.h"
#include "rocjitsu/code/kernel_descriptor_scan.h"

#include <algorithm>

namespace rocjitsu {

namespace {

namespace kd = rocr::llvm::amdhsa;
using KD = kd::kernel_descriptor_t;

/// System SGPRs allocated immediately after the user-SGPR block. A disabled
/// dimension consumes no SGPR, so this is a count, not a fixed three.
[[nodiscard]] uint32_t enabled_workgroup_id_sgprs(const KD &desc) {
  const uint32_t rsrc2 = desc.compute_pgm_rsrc2;
  uint32_t count = 0;
  if (AMDHSA_BITS_GET(rsrc2, kd::COMPUTE_PGM_RSRC2_ENABLE_SGPR_WORKGROUP_ID_X))
    ++count;
  if (AMDHSA_BITS_GET(rsrc2, kd::COMPUTE_PGM_RSRC2_ENABLE_SGPR_WORKGROUP_ID_Y))
    ++count;
  if (AMDHSA_BITS_GET(rsrc2, kd::COMPUTE_PGM_RSRC2_ENABLE_SGPR_WORKGROUP_ID_Z))
    ++count;
  return count;
}

} // namespace

uint32_t dbi_entry_storage_floor(KernelBlockScope blocks, const KD &desc, rj_code_arch_t arch) {
  return std::max(explicit_ordinary_sgpr_bound(blocks),
                  kernel_descriptor_user_sgpr_count(arch, desc) + enabled_workgroup_id_sgprs(desc));
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
      *error_out = "kernel names SGPRs up to s" + std::to_string(floor) + " and allocates " +
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

} // namespace rocjitsu
