// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

/// @file consan_program_analysis.h
/// @brief Bounded decoding and refinement of ConSan's program inventory.

#pragma once

#include "rocjitsu/code/patch/consan/consan.h"

#include <memory>
#include <span>
#include <string>
#include <vector>

namespace rocjitsu {

class AmdGpuCodeObject;
class Decoder;

/// Complete forward product of code-object and synchronization analysis.
/// Lowering transaction state, observation policy, resources, patches,
/// replacement bytes, and mutation outcomes are deliberately absent.
struct ConSanProgramAnalysisResult {
  ProgramInventory program_inventory;
  std::vector<ConSanFaultSite> fault_sites;
  std::vector<ConSanBarrierMoveDestination> barrier_move_destinations;
  ConSanTransformOutcome outcome = ConSanTransformOutcome::Unchanged;
  std::vector<std::string> warnings;
  std::vector<std::string> errors;
};

/// Parse, decode, and semantically analyze one code object into its immutable
/// program inventory. The caller retains the parser because later lowering
/// stages may continue against the same image; validation callers may discard
/// it immediately after projecting the independently derived proof facts.
///
/// This is the complete analysis boundary. It deliberately stops before fault
/// or perturbation selection, observation planning, resource solving, and
/// mutation. `result` receives the published analysis product; `perturbation`
/// receives only analysis candidates used by the subsequent planner.
[[nodiscard]] bool analyze_consan_program_inventory(std::span<const uint8_t> code_object_bytes,
                                                    const ConSanOptions &options,
                                                    std::unique_ptr<AmdGpuCodeObject> &code_object,
                                                    ProgramInventoryBuilder &inventory_builder,
                                                    ConSanPerturbationPlanningState &perturbation,
                                                    ConSanProgramAnalysisResult &result);

void decode_consan_kernel_inventory(std::span<const uint8_t> code_object_bytes, Decoder &decoder,
                                    rj_code_arch_t arch, ConSanKernelInfo &kernel,
                                    std::vector<ConSanAccessInventorySite> &accesses,
                                    std::vector<std::string> &warnings);

void decode_consan_function_inventory(std::span<const uint8_t> code_object_bytes, Decoder &decoder,
                                      rj_code_arch_t arch, ConSanFunctionInfo &function,
                                      std::vector<ConSanAccessInventorySite> &accesses,
                                      std::vector<std::string> &warnings);

void refine_consan_flat_pointer_provenance(std::span<const uint8_t> code_object_bytes,
                                           const AmdGpuCodeObject &code_object, rj_code_arch_t arch,
                                           std::vector<ConSanKernelInfo> &kernels,
                                           std::vector<ConSanFunctionInfo> &functions,
                                           std::vector<ConSanAccessInventorySite> &accesses,
                                           std::vector<std::string> &warnings);

void prune_consan_unreachable_inferred_ranges(
    const AmdGpuCodeObject &code_object, Decoder &decoder, rj_code_arch_t arch,
    std::span<const ConSanPreappliedCodeRange> preapplied_ranges,
    std::span<ConSanKernelInfo> kernels, std::span<ConSanFunctionInfo> functions,
    std::vector<ConSanAccessInventorySite> &accesses);

void preflight_consan_kernel(ConSanKernelInfo &kernel, std::vector<std::string> &warnings);

} // namespace rocjitsu
