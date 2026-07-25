// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "rocjitsu/code/dbt/proxy_code_object.h"

#include "rocjitsu/code/amdgpu_code_object.h"
#include "rocjitsu/code/amdgpu_elf.h"
#include "rocjitsu/code/dbt/kernel_descriptor_translator.h"
#include "rocjitsu/code/patch/code_object_patcher.h"
#include "rocjitsu/code/patch/kernel_text_layout.h"

#include <cstdint>
#include <optional>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

namespace rocjitsu {
namespace {

void append_error(std::vector<TranslationDiagnostic> &diagnostics, DiagnosticKind kind,
                  std::string message, std::optional<uint64_t> guest_offset = std::nullopt) {
  TranslationDiagnostic diagnostic;
  diagnostic.severity = DiagnosticSeverity::Error;
  diagnostic.kind = kind;
  diagnostic.guest_offset = guest_offset;
  diagnostic.message = std::move(message);
  diagnostics.push_back(std::move(diagnostic));
}

/// @brief Leave the source bytes intact so the caller can fall back to eager.
ProxyCodeObject leave_unchanged(const AmdGpuCodeObject &obj, ProxyCodeObject result) {
  const auto *image = reinterpret_cast<const uint8_t *>(obj.image_data());
  result.elf_bytes.assign(image, image + obj.image_size());
  return result;
}

} // namespace

ProxyCodeObject build_proxy_code_object(const AmdGpuCodeObject &obj, rj_code_arch_t target_arch) {
  ProxyCodeObject result;

  CodeObjectPatcher patcher(obj);
  if (patcher.text_bytes().empty()) {
    append_error(result.diagnostics, DiagnosticKind::ResourceLimit,
                 "code object does not expose a non-empty .text section for a proxy");
    return leave_unchanged(obj, std::move(result));
  }

  // Descriptor discovery needs only the ELF + .text extent; it does not decode
  // instructions, so the proxy skips all of BinaryTranslator's CFG/liveness/
  // legalization work. Every kernel descriptor is retargeted to a private
  // s_endpgm stub and stripped to the minimal skipped-kernel resource plan.
  KernelDescriptorTranslator descriptor_translator(target_arch, target_arch);
  auto descriptor_translations = descriptor_translator.translate_image(
      patcher.image_bytes(), patcher.text_offset(), patcher.text_size(),
      KernelDescriptorTranslationOptions{});
  if (descriptor_translations.empty()) {
    append_error(result.diagnostics, DiagnosticKind::KernelDescriptor,
                 "kernel descriptors are required to build a proxy code object");
    return leave_unchanged(obj, std::move(result));
  }

  // Emit one s_endpgm stub per distinct kernel entry, then point every descriptor
  // that shares that entry at the stub. Entries can repeat across descriptor
  // variants (e.g. a normal + a virtual-LDS sidecar for one kernel), so the stub
  // offset is memoized by source entry to keep the proxy .text minimal.
  const uint64_t original_text_size = patcher.text_size();
  std::vector<uint8_t> proxy_text;
  std::unordered_map<uint64_t, uint64_t> stub_entry_by_source;
  for (KdTranslation &translation : descriptor_translations) {
    const uint64_t source_entry = translation.entry_text_offset;
    auto it = stub_entry_by_source.find(source_entry);
    if (it == stub_entry_by_source.end()) {
      auto stub = append_skipped_kernel_stub(
          proxy_text,
          {.source_entry = source_entry,
           .has_kernarg_preload_firmware_skip = translation.has_kernarg_preload_firmware_skip},
          target_arch);
      if (!stub.ok) {
        append_error(result.diagnostics, DiagnosticKind::ResourceLimit, stub.message,
                     stub.source_offset);
        return leave_unchanged(obj, std::move(result));
      }
      it = stub_entry_by_source.emplace(source_entry, stub.target_entry).first;
      translation.target_body_entry_text_offset = stub.target_body_entry;
    }
    translation.target_entry_text_offset = it->second;
    // A proxy descriptor must describe the stub's minimal resource footprint, not
    // the source kernel's (whose oversized SGPR/VGPR/LDS/private demands could make
    // the runtime reject the load even though the entry is a safe no-op).
    translation.configure_skipped_stub();
    // configure_skipped_stub() clears skipped-related resource fields but keeps the
    // entry redirect above; re-assert the target entry in case it was touched.
    translation.target_entry_text_offset = it->second;
  }

  // The proxy .text replaces the whole original .text. Pad up to the original size
  // so section-relative offsets the loader/descriptor ABI already resolved stay
  // within range (the patcher grows/shifts as needed for larger text, but a proxy
  // is always smaller, so this is pure tail padding).
  if (proxy_text.size() < original_text_size)
    append_nop_padding(proxy_text, original_text_size - proxy_text.size(), target_arch);

  std::unordered_set<uint64_t> applied_descriptors;
  for (const KdTranslation &translation : descriptor_translations) {
    if (!applied_descriptors.insert(translation.descriptor_file_offset).second)
      continue;
    if (!patcher.apply_kernel_descriptor_translation(translation, target_arch)) {
      append_error(result.diagnostics, DiagnosticKind::KernelDescriptor,
                   "proxy kernel descriptor could not be retargeted to its stub safely");
      return leave_unchanged(obj, std::move(result));
    }
  }

  if (!patcher.replace_text(proxy_text)) {
    append_error(result.diagnostics, DiagnosticKind::ResourceLimit,
                 "proxy .text could not be materialized safely");
    return leave_unchanged(obj, std::move(result));
  }

  patcher.update_elf_flags(elf_mach_for_arch(target_arch));

  // Reparse the emitted image so a malformed proxy is caught here rather than at
  // load, matching BinaryTranslator's post-materialization validation.
  const auto proxy_image = patcher.image_bytes();
  AmdGpuCodeObject proxy_layout(proxy_image.data(), proxy_image.size());
  if (!proxy_layout.is_valid()) {
    append_error(result.diagnostics, DiagnosticKind::ResourceLimit,
                 "emitted proxy ELF could not be reparsed; leaving code object unchanged");
    return leave_unchanged(obj, std::move(result));
  }

  result.elf_bytes = std::move(patcher).emit();
  return result;
}

} // namespace rocjitsu
