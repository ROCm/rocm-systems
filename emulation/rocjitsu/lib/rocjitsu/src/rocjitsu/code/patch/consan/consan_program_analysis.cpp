// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "rocjitsu/code/patch/consan/consan_program_analysis.h"

#include "rocjitsu/analysis/def_use_chain.h"
#include "rocjitsu/analysis/kernel_scope.h"
#include "rocjitsu/analysis/liveness.h"
#include "rocjitsu/code/amdgpu_code_object.h"
#include "rocjitsu/code/basic_block.h"
#include "rocjitsu/code/builders/instruction_builder.h"
#include "rocjitsu/code/major_image_ownership.h"
#include "rocjitsu/code/patch/code_object_patcher.h"
#include "rocjitsu/code/patch/consan/consan_cfg.h"
#include "rocjitsu/code/patch/consan/consan_descriptor.h"
#include "rocjitsu/code/patch/consan/consan_instruction_semantics.h"
#include "rocjitsu/code/patch/consan/consan_physical_site_alias.h"
#include "rocjitsu/code/patch/consan/consan_semantic_classifiers.h"
#include "rocjitsu/isa/arch/amdgpu/generated/cdna4/machine_insts.h"
#include "rocjitsu/isa/arch/amdgpu/generated/cdna5/machine_insts.h"
#include "rocjitsu/isa/arch/amdgpu/generated/rdna3/machine_insts.h"
#include "rocjitsu/isa/arch/amdgpu/generated/rdna4/machine_insts.h"
#include "rocjitsu/isa/arch/amdgpu/shared/gfx12_cache_flags.h"
#include "rocjitsu/isa/decoder.h"
#include "rocjitsu/isa/instruction.h"
#include "util/bit.h"

#include <algorithm>
#include <array>
#include <cctype>
#include <cstring>
#include <initializer_list>
#include <iterator>
#include <map>
#include <memory>
#include <numeric>
#include <optional>
#include <set>
#include <string>
#include <string_view>
#include <tuple>
#include <unordered_map>
#include <unordered_set>
#include <utility>

namespace rocjitsu {

#include "rocjitsu/code/patch/consan/consan_analysis.inc"

} // namespace

void decode_consan_kernel_inventory(std::span<const uint8_t> code_object_bytes, Decoder &decoder,
                                    rj_code_arch_t arch, ConSanKernelInfo &kernel,
                                    std::vector<ConSanAccessInventorySite> &accesses,
                                    std::vector<std::string> &warnings) {
  decode_kernel_stats(code_object_bytes, decoder, arch, kernel, accesses, warnings);
}

void decode_consan_function_inventory(std::span<const uint8_t> code_object_bytes, Decoder &decoder,
                                      rj_code_arch_t arch, ConSanFunctionInfo &function,
                                      std::vector<ConSanAccessInventorySite> &accesses,
                                      std::vector<std::string> &warnings) {
  decode_function_stats(code_object_bytes, decoder, arch, function, accesses, warnings);
}

void refine_consan_flat_pointer_provenance(std::span<const uint8_t> code_object_bytes,
                                           const AmdGpuCodeObject &code_object, rj_code_arch_t arch,
                                           std::vector<ConSanKernelInfo> &kernels,
                                           std::vector<ConSanFunctionInfo> &functions,
                                           std::vector<ConSanAccessInventorySite> &accesses,
                                           std::vector<std::string> &warnings) {
  relay_flat_pointer_provenance_across_calls(code_object_bytes, code_object, arch, kernels,
                                             functions, accesses, warnings);
}

void prune_consan_unreachable_inferred_ranges(
    const AmdGpuCodeObject &code_object, Decoder &decoder, rj_code_arch_t arch,
    std::span<const ConSanPreappliedCodeRange> preapplied_ranges,
    std::span<ConSanKernelInfo> kernels, std::span<ConSanFunctionInfo> functions,
    std::vector<ConSanAccessInventorySite> &accesses) {
  prune_unreachable_inferred_ranges(code_object, decoder, arch, preapplied_ranges, kernels,
                                    functions, accesses);
}

void preflight_consan_kernel(ConSanKernelInfo &kernel, std::vector<std::string> &warnings) {
  preflight_kernel(kernel, warnings);
}

} // namespace rocjitsu
