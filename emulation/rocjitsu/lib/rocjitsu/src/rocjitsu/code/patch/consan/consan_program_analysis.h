// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

/// @file consan_program_analysis.h
/// @brief Bounded decoding and refinement of ConSan's program inventory.

#pragma once

#include "rocjitsu/code/patch/consan/consan.h"

#include <span>
#include <string>
#include <vector>

namespace rocjitsu {

class AmdGpuCodeObject;
class Decoder;

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
