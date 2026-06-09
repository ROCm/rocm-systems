// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

/// @file rj_hip_device_intercept.cpp
/// @brief LD_PRELOAD HIP device-property override for DBT kernel selection.
///
/// Why this exists
/// ---------------
///
/// The HSA tools hook spoofs the HSA ISA and translates code objects before
/// ROCR loads them. That is enough for ROCR compatibility, but it is not the
/// only identity ROCm libraries consult while choosing kernels. MIOpen,
/// rocBLAS, hipBLASLt, and ATen can also query HIP device properties,
/// especially `gcnArchName`, while selecting prebuilt kernels.
///
/// This hook makes the HIP-facing architecture name match the guest ISA. That
/// is a selection-time lie only: kernels still run on the real host GPU after
/// the HSA DBT hook translates their code objects.
///
/// What this intercepts
/// --------------------
///
/// The important property entry points are:
///
///   * `hipGetDevicePropertiesR0600`: current HIP 6+ / HIP 7 struct layout.
///   * `hipGetDevicePropertiesR0000`: legacy struct layout, still exported.
///   * `hipGetDeviceProperties`: legacy public symbol used by older callers.
///
/// The wrappers forward to the real HIP implementation first and then rewrite
/// only `gcnArchName`. Capacity, memory, PCI, chip, queue, and execution
/// capability fields remain the real host values.
///
/// Scope limitation
/// ----------------
///
/// PLT callers hit these wrappers directly. Callers that fetch HIP entry points
/// via a function-table API can bypass ordinary `LD_PRELOAD` symbol
/// interposition. That is acceptable for this hook because the libraries we are
/// fixing here query the exported HIP symbols, while code-object enforcement is
/// handled separately by the HSA tools hook.
///
/// Configuration
/// -------------
///
///   RJ_HIP_DEVICE_GFX_OVERRIDE=gfx950
/// If `RJ_HIP_DEVICE_GFX_OVERRIDE` is unset, `HIP_DEVICE_GCN_ARCH_OVERRIDE` and
/// then `RJ_DBT_SOURCE_ISA` are accepted as fallbacks. The gfx value may include
/// `amdgcn-amd-amdhsa--` and/or target-id features; both are stripped before
/// writing HIP's `gcnArchName` field.

#include <dlfcn.h>
#include <link.h>

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <optional>
#include <string>
#include <string_view>

namespace {

using hipError_t = int;
using HipGetDevicePropertiesFn = hipError_t (*)(void *, int);
using DlsymFn = void *(*)(void *, const char *);

// HIP device-property layouts are ABI-pinned per symbol variant. These offsets
// were checked against ROCm 7.x headers with offsetof():
//
//   hipDeviceProp_tR0600: sizeof=1472 gcnArchName=1160
//   hipDeviceProp_tR0000: sizeof=792  gcnArchName=396
//
// We write fixed offsets rather than scanning for "gfx<digit>". A scan can
// silently latch onto the wrong field if a future layout adds another gfx-like
// string before gcnArchName. The runtime check in overwrite_string_field()
// verifies that the existing bytes at the offset look like a real gfx name and
// aborts loudly if the struct layout drifts.
constexpr size_t kHipDevicePropR0600GcnArchOffset = 1160;
constexpr size_t kHipDevicePropR0000GcnArchOffset = 396;
constexpr size_t kHipGcnArchNameMax = 256;

HipGetDevicePropertiesFn g_real_hipGetDevicePropertiesR0600 = nullptr;
HipGetDevicePropertiesFn g_real_hipGetDevicePropertiesR0000 = nullptr;
HipGetDevicePropertiesFn g_real_hipGetDeviceProperties = nullptr;

extern "C" hipError_t hipGetDevicePropertiesR0600(void *prop, int device_id);
extern "C" hipError_t hipGetDevicePropertiesR0000(void *prop, int device_id);
extern "C" hipError_t hipGetDeviceProperties(void *prop, int device_id);

struct OverrideConfig {
  std::string gfx;
};

[[nodiscard]] DlsymFn real_dlsym() {
  static DlsymFn fn = [] {
    // Resolve libc's real dlsym through its versioned export. Calling
    // dlsym(RTLD_NEXT, "dlsym") during bootstrap is unsafe when another
    // LD_PRELOAD library also intercepts dlsym: RTLD_NEXT can find that hook
    // and recurse back into interposition code.
    auto *resolved = reinterpret_cast<DlsymFn>(dlvsym(RTLD_NEXT, "dlsym", "GLIBC_2.34"));
    if (resolved == nullptr)
      resolved = reinterpret_cast<DlsymFn>(dlvsym(RTLD_NEXT, "dlsym", "GLIBC_2.2.5"));
    if (resolved == nullptr) {
      std::fprintf(stderr, "rocjitsu-hip-hooks: failed to resolve libc dlsym\n");
      std::abort();
    }
    return resolved;
  }();
  return fn;
}

struct SymbolSearch {
  const char *symbol = nullptr;
  const void *self = nullptr;
  void *found = nullptr;
};

int find_symbol_in_object(dl_phdr_info *info, size_t, void *data) {
  if (info == nullptr || info->dlpi_name == nullptr || info->dlpi_name[0] == '\0')
    return 0;

  auto *search = static_cast<SymbolSearch *>(data);
  void *handle = dlopen(info->dlpi_name, RTLD_NOW | RTLD_LOCAL | RTLD_NOLOAD);
  if (handle == nullptr)
    return 0;

  void *symbol = real_dlsym()(handle, search->symbol);
  if (symbol == nullptr || symbol == search->self)
    return 0;

  search->found = symbol;
  return 1;
}

[[nodiscard]] void *resolve_symbol(const char *symbol, const void *self) {
  // The normal LD_PRELOAD case: the next object in the global lookup chain is
  // libamdhip64.so, so RTLD_NEXT resolves the real entry point.
  if (void *fn = dlsym(RTLD_NEXT, symbol); fn != nullptr && fn != self)
    return fn;

  // PyTorch and some ROCm wheels may load libamdhip64 RTLD_LOCAL. RTLD_NEXT can
  // miss local objects, so walk the link map and attach to already-loaded DSOs
  // with RTLD_NOLOAD. The self check avoids caching our own wrapper if another
  // interposer changes symbol lookup order.
  SymbolSearch search{symbol, self, nullptr};
  dl_iterate_phdr(find_symbol_in_object, &search);
  return search.found;
}

template <typename Fn>
[[nodiscard]] Fn resolve_real(const char *symbol, Fn &slot, const void *self) {
  if (slot != nullptr)
    return slot;
  slot = reinterpret_cast<Fn>(resolve_symbol(symbol, self));
  if (slot == nullptr) {
    std::fprintf(stderr, "rocjitsu-hip-hooks: failed to resolve real %s\n", symbol);
    std::abort();
  }
  return slot;
}

[[nodiscard]] std::string_view strip_hsa_prefix(std::string_view value) {
  constexpr std::string_view kPrefix = "amdgcn-amd-amdhsa--";
  if (value.starts_with(kPrefix))
    value.remove_prefix(kPrefix.size());
  return value;
}

[[nodiscard]] std::string parse_gfx_name(const char *raw) {
  if (raw == nullptr || raw[0] == '\0')
    return {};

  // Accept HSA-style values so users can share the same string between the HSA
  // hook and this HIP hook, for example "amdgcn-amd-amdhsa--gfx950:xnack-".
  std::string_view value = strip_hsa_prefix(raw);
  const size_t feature_pos = value.find(':');
  if (feature_pos != std::string_view::npos)
    value = value.substr(0, feature_pos);

  if (value.size() < 4 || value.substr(0, 3) != "gfx" || value[3] < '0' || value[3] > '9') {
    std::fprintf(stderr, "rocjitsu-hip-hooks: invalid gfx override '%s'\n", raw);
    return {};
  }
  return std::string(value);
}

[[nodiscard]] const OverrideConfig &config() {
  static const OverrideConfig cfg = [] {
    OverrideConfig result;
    result.gfx = parse_gfx_name(std::getenv("RJ_HIP_DEVICE_GFX_OVERRIDE"));
    if (result.gfx.empty())
      result.gfx = parse_gfx_name(std::getenv("HIP_DEVICE_GCN_ARCH_OVERRIDE"));
    if (result.gfx.empty())
      result.gfx = parse_gfx_name(std::getenv("RJ_DBT_SOURCE_ISA"));

    if (!result.gfx.empty())
      std::fprintf(stderr, "rocjitsu-hip-hooks: active gfx=%s\n", result.gfx.c_str());
    return result;
  }();
  return cfg;
}

void overwrite_string_field(void *buffer, size_t offset, std::string_view value) {
  if (buffer == nullptr || value.empty())
    return;
  // HIP defines gcnArchName as char[256]. Reject oversized values instead of
  // truncating, since a truncated architecture name would make library
  // selection fail in a confusing way.
  if (value.size() + 1 > kHipGcnArchNameMax) {
    std::fprintf(stderr, "rocjitsu-hip-hooks: gfx override '%.*s' is too long\n",
                 static_cast<int>(value.size()), value.data());
    std::abort();
  }

  auto *bytes = static_cast<uint8_t *>(buffer);
  // Guard the fixed-offset assumption. If this fires, the HIP ABI/layout in the
  // process no longer matches the offsets above and the hook must be updated.
  if (bytes[offset] != 'g' || bytes[offset + 1] != 'f' || bytes[offset + 2] != 'x' ||
      bytes[offset + 3] < '0' || bytes[offset + 3] > '9') {
    std::fprintf(stderr,
                 "rocjitsu-hip-hooks: hipDeviceProp_t layout mismatch at gcnArchName offset %zu\n",
                 offset);
    std::abort();
  }

  std::memcpy(bytes + offset, value.data(), value.size());
  bytes[offset + value.size()] = '\0';
}

void rewrite_r0600(void *prop) {
  // R0600 is the current HIP property layout. This is the path PyTorch/ROCm 7
  // callers normally use through the hipGetDeviceProperties macro.
  const OverrideConfig &cfg = config();
  overwrite_string_field(prop, kHipDevicePropR0600GcnArchOffset, cfg.gfx);
}

void rewrite_r0000(void *prop) {
  // R0000 is still exported for older ABI users. Keep it wired up because ROCm
  // libraries can mix direct versioned calls with public compatibility symbols.
  const OverrideConfig &cfg = config();
  overwrite_string_field(prop, kHipDevicePropR0000GcnArchOffset, cfg.gfx);
}

} // namespace

extern "C" hipError_t hipGetDevicePropertiesR0600(void *prop, int device_id) {
  auto real = resolve_real("hipGetDevicePropertiesR0600", g_real_hipGetDevicePropertiesR0600,
                           reinterpret_cast<const void *>(&hipGetDevicePropertiesR0600));
  const hipError_t err = real(prop, device_id);
  if (err == 0)
    rewrite_r0600(prop);
  return err;
}

extern "C" hipError_t hipGetDevicePropertiesR0000(void *prop, int device_id) {
  auto real = resolve_real("hipGetDevicePropertiesR0000", g_real_hipGetDevicePropertiesR0000,
                           reinterpret_cast<const void *>(&hipGetDevicePropertiesR0000));
  const hipError_t err = real(prop, device_id);
  if (err == 0)
    rewrite_r0000(prop);
  return err;
}

extern "C" hipError_t hipGetDeviceProperties(void *prop, int device_id) {
  auto real = resolve_real("hipGetDeviceProperties", g_real_hipGetDeviceProperties,
                           reinterpret_cast<const void *>(&hipGetDeviceProperties));
  const hipError_t err = real(prop, device_id);
  if (err == 0)
    rewrite_r0000(prop);
  return err;
}
