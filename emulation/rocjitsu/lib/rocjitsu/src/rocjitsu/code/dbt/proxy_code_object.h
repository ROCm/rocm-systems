// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

#include "rocjitsu/code/dbt/translation_diagnostic.h"
#include "rocjitsu/code/rj_code.h"

#include <cstdint>
#include <vector>

namespace rocjitsu {

class AmdGpuCodeObject;

/// @brief A cheap stand-in ELF loaded in place of a deferred (lazy) code object.
///
/// @details Lazy translation retains an eligible B0 object without running DBT at
/// load. So the loader still has stable, loadable kernel symbols and descriptors
/// to hand the runtime, rocjitsu instead loads a **proxy**: the original ELF with
/// every kernel body replaced by the minimal `s_endpgm` skipped-kernel stub and
/// every descriptor reset to that stub's minimal resource plan. The proxy is
/// never meant to execute — the dispatch interceptor rewrites a proxy dispatch to
/// the translated kernel before the command processor reads the packet. It exists
/// only to satisfy the runtime's load-time symbol/descriptor queries with stable
/// kernel-object addresses while translation is deferred to first dispatch.
///
/// This is a much cheaper pass than @ref BinaryTranslator: it does not decode
/// `.text`, build a CFG, recover indirect targets, or apply any legalization. It
/// reuses the same descriptor-translation and ELF-patching primitives so the
/// emitted proxy is byte-compatible with what the loader expects.
struct ProxyCodeObject {
  /// @brief The emitted proxy ELF (empty when @ref ok is false).
  std::vector<uint8_t> elf_bytes;

  /// @brief Diagnostics produced while building the proxy.
  std::vector<TranslationDiagnostic> diagnostics;

  /// @brief True when a valid proxy ELF was emitted.
  [[nodiscard]] bool ok() const { return !has_error_diagnostic(diagnostics); }
};

/// @brief Build an A0-loadable proxy for a gfx1250 code object.
///
/// @param obj Parsed source code object (any AMDGPU ELF with kernel descriptors).
/// @param target_arch Architecture the proxy descriptors/stubs are encoded for.
/// @returns A @ref ProxyCodeObject; on failure @ref ProxyCodeObject::ok is false
///          and the original bytes are left in @ref ProxyCodeObject::elf_bytes so
///          the caller can fall back to eager translation rather than load a proxy.
[[nodiscard]] ProxyCodeObject build_proxy_code_object(const AmdGpuCodeObject &obj,
                                                      rj_code_arch_t target_arch);

} // namespace rocjitsu
