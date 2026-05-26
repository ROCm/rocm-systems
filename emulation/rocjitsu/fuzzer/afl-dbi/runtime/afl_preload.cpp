#include <hip/hip_runtime_api.h>
#include <hsa/hsa.h>
#include <hsa/hsa_ven_amd_loader.h>

#include "code_object_image.h"
#include "instrumentation_planner.h"
#include "patchability_classifier.h"
#include "rocjitsu/code/amdgpu_code_object.h"
#include "rocjitsu/code/amdgpu_elf.h"
#include "rocjitsu/code/patch/code_object_patcher.h"
#include "rocjitsu/code/rj_code.h"
#include "rocjitsu/vm/amdgpu/wavefront.h"
#include "rocjitsu_fuzzer/afl_dbi_plan.h"

#include <dlfcn.h>
#include <fcntl.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/shm.h>
#include <unistd.h>

#include <algorithm>
#include <chrono>
#include <cstddef>
#include <iterator>
#include <limits>
#include <memory>
#include <mutex>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace {

using Clock = std::chrono::steady_clock;
namespace afl_dbi = rocjitsu::fuzzer::afl_dbi;
using afl_dbi::arch_name;
using afl_dbi::kCoverageSlots;
using afl_dbi::kDeviceStart;
using afl_dbi::kDeviceStateSlots;
using afl_dbi::kEntryCounterSlot;
using afl_dbi::kFirstEdgeCounterSlot;
using afl_dbi::kMapSize;
using afl_dbi::kMaxInlineCounterSlot;
using afl_dbi::EdgePatchFailure;
using afl_dbi::EdgeSite;
using afl_dbi::EdgeSiteSelectionSample;
using afl_dbi::EdgeSlotPolicyKind;
using afl_dbi::EdgeSlotPolicySummary;
using afl_dbi::EdgeTrampolinePlacement;
using afl_dbi::FixedEdgeSlotTracker;
using afl_dbi::DeviceElfPatchPlan;
using afl_dbi::EntryProbeRegisterSelection;
using afl_dbi::InstrumentationPlan;
using afl_dbi::InstrumentationPlanOptions;
using afl_dbi::KernelEdgeSelectionSummary;
using afl_dbi::KernelDescriptorResourceSummary;
using afl_dbi::KernelPatchability;
using afl_dbi::KernelPatchabilityFilters;
using afl_dbi::KernelPatchabilitySkipSummary;
using afl_dbi::KernelSite;
using afl_dbi::kUnknownDeviceImageIndex;
using afl_dbi::LocalTextCaveAllocator;
using afl_dbi::PatchDeviceElfReport;
using afl_dbi::PlannedEdgeTrampoline;
using afl_dbi::PlannedEntryProbe;
using afl_dbi::ProbeRegisterRequirements;
using afl_dbi::Rdna4ProbeRegisters;
using afl_dbi::classify_kernel_patchability;
using afl_dbi::classify_loader_patchability;
using afl_dbi::code_object_self_contained_edge_probe_reason;
using afl_dbi::edge_count_for_site;
using afl_dbi::find_kernel_sites;
using afl_dbi::install_planned_edge_trampoline;
using afl_dbi::amdgpu_metadata_private_segment_patch_kind_name;
using afl_dbi::make_stable_fixed_counter_fallback_site;
using afl_dbi::patch_kernel_descriptor_resources;
using afl_dbi::placement_failure_can_degrade_to_fixed;
using afl_dbi::placement_fixed_fallback_has_budget;
using afl_dbi::plan_edge_trampoline;
using afl_dbi::plan_kernel_descriptor_resources;
using afl_dbi::previous_bb_branch_site;
using afl_dbi::prime_fixed_counter_placement_tracker;
using afl_dbi::record_fixed_counter_placement_slots;
using afl_dbi::record_previous_bb_branch_placement_fallback;
using afl_dbi::record_patch_plan_summary;
using afl_dbi::select_edge_sites;
using afl_dbi::select_entry_probe_registers;
using afl_dbi::stable_bb_id;

constexpr std::string_view kLazyCcobShadowPolicy = "raw_exact_kernel_elf";

using hipModuleLoad_t = hipError_t (*)(hipModule_t *, const char *);
using hipModuleLoadData_t = hipError_t (*)(hipModule_t *, const void *);
using hipModuleLoadFatBinary_t = hipError_t (*)(hipModule_t *, const void *);
using hipModuleLoadDataEx_t =
    hipError_t (*)(hipModule_t *, const void *, unsigned int, hipJitOption *, void **);
using hipModuleUnload_t = hipError_t (*)(hipModule_t);
using hipModuleGetFunction_t = hipError_t (*)(hipFunction_t *, hipModule_t, const char *);
using hipModuleLaunchKernel_t = hipError_t (*)(hipFunction_t, unsigned int, unsigned int,
                                               unsigned int, unsigned int, unsigned int,
                                               unsigned int, unsigned int, hipStream_t, void **,
                                               void **);
using hipExtModuleLaunchKernel_t = hipError_t (*)(hipFunction_t, uint32_t, uint32_t, uint32_t,
                                                  uint32_t, uint32_t, uint32_t, size_t, hipStream_t,
                                                  void **, void **, hipEvent_t, hipEvent_t,
                                                  uint32_t);
using hipDrvLaunchKernelEx_t = hipError_t (*)(const HIP_LAUNCH_CONFIG *, hipFunction_t, void **,
                                              void **);
using hipLaunchKernel_t = hipError_t (*)(const void *, dim3, dim3, void **, size_t, hipStream_t);
using hipDeviceSynchronize_t = hipError_t (*)();
using hipStreamSynchronize_t = hipError_t (*)(hipStream_t);
using hipEventSynchronize_t = hipError_t (*)(hipEvent_t);
using hipMemcpy_t = hipError_t (*)(void *, const void *, size_t, hipMemcpyKind);
using hipRegisterFatBinary_t = void **(*)(const void *);
using hipUnregisterFatBinary_t = void (*)(void **);
using hipRegisterFunction_t = void (*)(void **, const void *, char *, const char *, unsigned int,
                                       uint3 *, uint3 *, dim3 *, dim3 *, int *);
using hsaReaderCreateMemory_t =
    hsa_status_t (*)(const void *, size_t, hsa_code_object_reader_t *);
using hsaReaderCreateFile_t = hsa_status_t (*)(hsa_file_t, hsa_code_object_reader_t *);
using hsaReaderDestroy_t = hsa_status_t (*)(hsa_code_object_reader_t);
using hsaReaderCreateFileOffsetSize_t =
    hsa_status_t (*)(hsa_file_t, size_t, size_t, hsa_code_object_reader_t *);
using hsaSystemGetExtensionTable_t =
    hsa_status_t (*)(uint16_t, uint16_t, uint16_t, void *);
using hsaSystemGetMajorExtensionTable_t =
    hsa_status_t (*)(uint16_t, uint16_t, size_t, void *);

hipModuleLoad_t real_hipModuleLoad = nullptr;
hipModuleLoadData_t real_hipModuleLoadData = nullptr;
hipModuleLoadFatBinary_t real_hipModuleLoadFatBinary = nullptr;
hipModuleLoadDataEx_t real_hipModuleLoadDataEx = nullptr;
hipModuleUnload_t real_hipModuleUnload = nullptr;
hipModuleGetFunction_t real_hipModuleGetFunction = nullptr;
hipModuleLaunchKernel_t real_hipModuleLaunchKernel = nullptr;
hipExtModuleLaunchKernel_t real_hipExtModuleLaunchKernel = nullptr;
hipDrvLaunchKernelEx_t real_hipDrvLaunchKernelEx = nullptr;
hipLaunchKernel_t real_hipLaunchKernel = nullptr;
hipDeviceSynchronize_t real_hipDeviceSynchronize = nullptr;
hipStreamSynchronize_t real_hipStreamSynchronize = nullptr;
hipStreamSynchronize_t real_hipStreamSynchronize_spt = nullptr;
hipEventSynchronize_t real_hipEventSynchronize = nullptr;
hipMemcpy_t real_hipMemcpy = nullptr;
hipRegisterFatBinary_t real___hipRegisterFatBinary = nullptr;
hipUnregisterFatBinary_t real___hipUnregisterFatBinary = nullptr;
hipRegisterFunction_t real___hipRegisterFunction = nullptr;
hsaReaderCreateMemory_t real_hsa_code_object_reader_create_from_memory = nullptr;
hsaReaderCreateFile_t real_hsa_code_object_reader_create_from_file = nullptr;
hsaReaderDestroy_t real_hsa_code_object_reader_destroy = nullptr;
hsaReaderCreateFileOffsetSize_t
    real_hsa_ven_amd_loader_code_object_reader_create_from_file_with_offset_size = nullptr;
hsaSystemGetExtensionTable_t real_hsa_system_get_extension_table = nullptr;
hsaSystemGetMajorExtensionTable_t real_hsa_system_get_major_extension_table = nullptr;

std::once_flag g_symbol_once;
std::mutex g_state_mutex;
std::vector<std::unique_ptr<std::vector<uint8_t>>> g_patched_images;
std::unordered_map<uint64_t, std::unique_ptr<std::vector<uint8_t>>> g_hsa_reader_images;
std::unordered_map<uintptr_t, std::string> g_module_function_names;

struct LazyPatchedFunction {
  hipModule_t module = nullptr;
  hipFunction_t function = nullptr;
};

struct LazyCcobModule {
  std::vector<uint8_t> image;
  std::unordered_map<std::string, LazyPatchedFunction> functions;
};

std::unordered_map<uintptr_t, LazyCcobModule> g_lazy_ccob_modules;
struct LazyCcobLaunchBinding {
  uintptr_t module = 0;
  std::string kernel;
  LazyPatchedFunction target;
};

struct RuntimeShadowSource {
  uintptr_t registration_key = 0;
  uint64_t dedupe_key = 0;
  std::shared_ptr<const std::vector<uint8_t>> image;
  std::vector<std::string> kernel_names;
};

struct RuntimeShadowSourceSnapshot {
  uint64_t dedupe_key = 0;
  std::shared_ptr<const std::vector<uint8_t>> image;
  std::vector<std::string> kernel_names;
};

struct RuntimeShadowFunction {
  uintptr_t registration_key = 0;
  LazyPatchedFunction target;
  size_t source_count = 0;
  size_t matching_sources = 0;
  uint64_t source_dedupe_key = 0;
};

struct ShadowModuleUnloadStats {
  size_t attempts = 0;
  size_t failures = 0;
  hipError_t last_error = hipSuccess;
};

struct PendingLaunchSummary {
  uint64_t launches = 0;
  uint64_t hip_module_launches = 0;
  uint64_t hip_runtime_launches = 0;
  uint64_t runtime_shadow_launches = 0;
  uint64_t lazy_ccob_launches = 0;
  std::string first_api;
  std::string first_kernel;
  std::string last_api;
  std::string last_kind;
  std::string last_kernel;
  uintptr_t last_module_function = 0;
  uintptr_t last_runtime_function = 0;
  uintptr_t last_registration = 0;
};

std::vector<uint8_t> g_fallback_trace_bits;
std::vector<uint32_t> g_last_merged_counters;
uint8_t *g_trace_bits = nullptr;
uint32_t *g_device_counters = nullptr;
bool g_runtime_ready = false;
bool g_runtime_failed = false;
bool g_verbose = false;
bool g_disable_edges = false;
bool g_launch_only = false;
bool g_fixed_edge_slots = false;
bool g_branch_edge_slots = false;
EdgeSlotPolicyKind g_branch_edge_slot_policy = EdgeSlotPolicyKind::PreviousBbHash;
bool g_diagnostic_branch_edge_slot_policy_override = false;
bool g_require_liveness_registers = false;
bool g_disable_vgpr_scratch_spills = false;
bool g_force_fresh_sgprs = false;
bool g_force_fresh_vgprs = false;
bool g_allow_opaque_fresh_registers = false;
std::vector<uint32_t> g_debug_edge_patch_text_offsets;
bool g_persistent_mode = false;
bool g_require_device_edges = false;
bool g_required_device_edges_observed = false;
bool g_required_device_edges_warned = false;
bool g_skip_entry_probe = false;
bool g_runtime_shadow_modules = true;
const char *g_kernel_include = nullptr;
const char *g_kernel_exclude = nullptr;
const char *g_patch_report_path = nullptr;
thread_local const char *g_kernel_include_override = nullptr;
thread_local uintptr_t g_runtime_shadow_registration_override = 0;
uint32_t g_edge_site_limit = 12;
bool g_branch_edge_site_limit_overridden = false;
uint32_t g_debug_forced_runtime_shadow_private_segment_bytes = 0;
// Conservative explicit branch-edge cap for the current DBI prototype. The
// best-effort self-contained hashed fallback may raise the cap locally for
// entry-unsafe kernels, but making that unbounded should be driven by a
// per-kernel instrumentation plan that preflights all selected sites: choose
// scratch temporaries from descriptor/liveness information instead of fixed high
// registers, reject or relocate overwritten instructions with PC-relative
// operands or flag dependencies, account for code-cave growth and branch
// reachability, and only then select as many safe slots as the AFL device map
// can absorb.
uint32_t g_branch_edge_site_limit = 8;
uint64_t g_launches = 0;
PendingLaunchSummary g_pending_launch_summary;
thread_local uint32_t g_intercept_depth = 0;

struct ScopedInterceptionBypass {
  ScopedInterceptionBypass() { ++g_intercept_depth; }
  ScopedInterceptionBypass(const ScopedInterceptionBypass &) = delete;
  ScopedInterceptionBypass &operator=(const ScopedInterceptionBypass &) = delete;
  ~ScopedInterceptionBypass() { --g_intercept_depth; }
};

bool interception_bypassed() {
  return g_intercept_depth != 0;
}

void emit_device_edge_delta_report(const char *trigger, const PendingLaunchSummary &launches,
                                   uint32_t entry_delta, uint32_t edge_slot_delta_count,
                                   uint64_t edge_counter_delta_total,
                                   uint32_t nonzero_edge_slots_total);
void emit_required_device_edges_report(const char *trigger, bool fatal,
                                       uint32_t nonzero_edge_slots_total);

struct ScopedKernelIncludeOverride {
  explicit ScopedKernelIncludeOverride(const char *kernel_name)
      : previous_(g_kernel_include_override) {
    g_kernel_include_override = kernel_name;
  }
  ScopedKernelIncludeOverride(const ScopedKernelIncludeOverride &) = delete;
  ScopedKernelIncludeOverride &operator=(const ScopedKernelIncludeOverride &) = delete;
  ~ScopedKernelIncludeOverride() { g_kernel_include_override = previous_; }

private:
  const char *previous_ = nullptr;
};

struct ScopedRuntimeShadowRegistrationOverride {
  explicit ScopedRuntimeShadowRegistrationOverride(uintptr_t registration_key)
      : previous_(g_runtime_shadow_registration_override) {
    g_runtime_shadow_registration_override = registration_key;
  }
  ScopedRuntimeShadowRegistrationOverride(const ScopedRuntimeShadowRegistrationOverride &) =
      delete;
  ScopedRuntimeShadowRegistrationOverride &
  operator=(const ScopedRuntimeShadowRegistrationOverride &) = delete;
  ~ScopedRuntimeShadowRegistrationOverride() {
    g_runtime_shadow_registration_override = previous_;
  }

private:
  uintptr_t previous_ = 0;
};

template <typename T> T load_symbol(const char *name) {
  void *sym = dlsym(RTLD_NEXT, name);
  if (sym == nullptr && g_verbose) {
    fprintf(stderr, "rocjitsu-afl: dlsym(%s) failed: %s\n", name, dlerror());
  }
  return reinterpret_cast<T>(sym);
}

template <typename T> T load_optional_symbol(const char *name) {
  return reinterpret_cast<T>(dlsym(RTLD_NEXT, name));
}

bool env_flag(const char *name) { return getenv(name) != nullptr; }

const char *getenv_debug(const char *debug_name) { return getenv(debug_name); }

bool debug_env_flag(const char *debug_name) { return getenv_debug(debug_name) != nullptr; }

std::optional<uint32_t> parse_debug_u32_env(const char *debug_name,
                                            int base = 10) {
  const char *value = getenv_debug(debug_name);
  if (value == nullptr)
    return std::nullopt;
  char *end = nullptr;
  const unsigned long parsed = strtoul(value, &end, base);
  if (end == value)
    return std::nullopt;
  return static_cast<uint32_t>(
      std::min<unsigned long>(parsed, std::numeric_limits<uint32_t>::max()));
}

void append_debug_patch_text_offset(uint32_t offset) {
  if (std::find(g_debug_edge_patch_text_offsets.begin(),
                g_debug_edge_patch_text_offsets.end(),
                offset) == g_debug_edge_patch_text_offsets.end()) {
    g_debug_edge_patch_text_offsets.push_back(offset);
  }
}

void parse_debug_patch_text_offset_list_env(const char *debug_name) {
  const char *value = getenv_debug(debug_name);
  if (value == nullptr)
    return;
  const char *cursor = value;
  while (*cursor != '\0') {
    while (*cursor == ',' || *cursor == ':' || *cursor == ';' ||
           *cursor == ' ' || *cursor == '\t')
      ++cursor;
    if (*cursor == '\0')
      break;

    char *end = nullptr;
    const unsigned long parsed = strtoul(cursor, &end, 0);
    if (end == cursor) {
      while (*cursor != '\0' && *cursor != ',' && *cursor != ':' &&
             *cursor != ';' && *cursor != ' ' && *cursor != '\t')
        ++cursor;
      continue;
    }
    append_debug_patch_text_offset(static_cast<uint32_t>(
        std::min<unsigned long>(parsed, std::numeric_limits<uint32_t>::max())));
    cursor = end;
  }
}

void resolve_symbols() {
  std::call_once(g_symbol_once, [] {
    g_verbose = env_flag("ROCJITSU_AFL_VERBOSE");
    g_disable_edges = debug_env_flag("ROCJITSU_AFL_DEBUG_DISABLE_EDGES");
    g_launch_only = debug_env_flag("ROCJITSU_AFL_DEBUG_LAUNCH_ONLY");
    g_fixed_edge_slots = debug_env_flag("ROCJITSU_AFL_DEBUG_FIXED_EDGE_SLOTS");
    g_disable_vgpr_scratch_spills =
        debug_env_flag("ROCJITSU_AFL_DEBUG_DISABLE_VGPR_SCRATCH_SPILLS");
    // DEBUG-only allocation diagnostic. Product planning still picks allocated
    // dead registers before growing descriptor resources.
    g_force_fresh_sgprs = debug_env_flag("ROCJITSU_AFL_DEBUG_FORCE_FRESH_SGPRS");
    g_force_fresh_vgprs = debug_env_flag("ROCJITSU_AFL_DEBUG_FORCE_FRESH_VGPRS");
    // DEBUG-only dangerous repro hook for minimizers. Product planning keeps
    // fresh descriptor growth disabled in opaque/VOPD scopes.
    g_allow_opaque_fresh_registers =
        debug_env_flag("ROCJITSU_AFL_DEBUG_ALLOW_OPAQUE_FRESH_REGISTERS");
    const bool explicit_branch_edge_slots =
        debug_env_flag("ROCJITSU_AFL_DEBUG_BRANCH_EDGE_SLOTS");
    const bool explicit_liveness_registers =
        debug_env_flag("ROCJITSU_AFL_DEBUG_REQUIRE_LIVENESS_REGISTERS");
    const std::optional<uint32_t> edge_limit =
        parse_debug_u32_env("ROCJITSU_AFL_DEBUG_EDGE_LIMIT");
    const std::optional<uint32_t> branch_edge_limit =
        parse_debug_u32_env("ROCJITSU_AFL_DEBUG_BRANCH_EDGE_LIMIT");
    const std::optional<uint32_t> edge_patch_text_offset =
        parse_debug_u32_env("ROCJITSU_AFL_DEBUG_EDGE_PATCH_TEXT_OFFSET", 0);
    const std::optional<uint32_t> forced_runtime_shadow_private_segment_bytes =
        parse_debug_u32_env(
            "ROCJITSU_AFL_DEBUG_FORCE_RUNTIME_SHADOW_PRIVATE_SEGMENT_BYTES");
    g_branch_edge_slots = explicit_branch_edge_slots;
    g_branch_edge_slot_policy = EdgeSlotPolicyKind::PreviousBbHash;
    g_require_liveness_registers = explicit_liveness_registers;
    if (const char *policy = getenv_debug("ROCJITSU_AFL_DEBUG_BRANCH_EDGE_SLOT_POLICY")) {
      if (strcmp(policy, "hashed") == 0 || strcmp(policy, "previous-bb-hash") == 0) {
        g_branch_edge_slot_policy = EdgeSlotPolicyKind::PreviousBbHash;
        g_diagnostic_branch_edge_slot_policy_override = true;
      } else if (strcmp(policy, "fixed") == 0 || strcmp(policy, "fixed-counter") == 0 ||
                 strcmp(policy, "inline-fixed-counter") == 0) {
        g_branch_edge_slot_policy = EdgeSlotPolicyKind::FixedCounter;
        g_diagnostic_branch_edge_slot_policy_override = true;
      } else if (g_verbose) {
        fprintf(stderr,
                "rocjitsu-afl: ignoring unknown branch edge slot policy %s\n",
                policy);
      }
    }
    if (g_branch_edge_slots && g_branch_edge_slot_policy == EdgeSlotPolicyKind::PreviousBbHash) {
      if (!g_require_liveness_registers && g_verbose) {
        fprintf(stderr,
                "rocjitsu-afl: enabling strict liveness for hashed branch-edge probes\n");
      }
      g_require_liveness_registers = true;
    }
    g_persistent_mode =
        env_flag("ROCJITSU_AFL_PERSISTENT") || env_flag("COV_PERSISTENT");
    g_require_device_edges = env_flag("ROCJITSU_AFL_REQUIRE_DEVICE_EDGES");
    g_skip_entry_probe = debug_env_flag("ROCJITSU_AFL_DEBUG_SKIP_ENTRY_PROBE");
    if (edge_limit)
      g_edge_site_limit = *edge_limit;
    if (branch_edge_limit) {
      g_branch_edge_site_limit = *branch_edge_limit;
      g_branch_edge_site_limit_overridden = true;
    }
    if (edge_patch_text_offset)
      append_debug_patch_text_offset(*edge_patch_text_offset);
    parse_debug_patch_text_offset_list_env(
        "ROCJITSU_AFL_DEBUG_EDGE_PATCH_TEXT_OFFSETS");
    if (forced_runtime_shadow_private_segment_bytes) {
      g_debug_forced_runtime_shadow_private_segment_bytes =
          *forced_runtime_shadow_private_segment_bytes;
    }
    const bool disable_runtime_shadow = debug_env_flag(
        "ROCJITSU_AFL_DEBUG_DISABLE_RUNTIME_SHADOW_MODULES");
    const bool force_runtime_shadow = debug_env_flag(
        "ROCJITSU_AFL_DEBUG_FORCE_RUNTIME_SHADOW_MODULES");
    g_runtime_shadow_modules = force_runtime_shadow || !disable_runtime_shadow;
    g_kernel_include = getenv("ROCJITSU_AFL_KERNEL_INCLUDE");
    g_kernel_exclude = getenv("ROCJITSU_AFL_KERNEL_EXCLUDE");
    g_patch_report_path = getenv("ROCJITSU_AFL_PATCH_REPORT");
    real_hipModuleLoad = load_symbol<hipModuleLoad_t>("hipModuleLoad");
    real_hipModuleLoadData = load_symbol<hipModuleLoadData_t>("hipModuleLoadData");
    real_hipModuleLoadFatBinary =
        load_optional_symbol<hipModuleLoadFatBinary_t>("hipModuleLoadFatBinary");
    real_hipModuleLoadDataEx = load_symbol<hipModuleLoadDataEx_t>("hipModuleLoadDataEx");
    real_hipModuleUnload = load_optional_symbol<hipModuleUnload_t>("hipModuleUnload");
    real_hipModuleGetFunction = load_symbol<hipModuleGetFunction_t>("hipModuleGetFunction");
    real_hipModuleLaunchKernel = load_symbol<hipModuleLaunchKernel_t>("hipModuleLaunchKernel");
    real_hipExtModuleLaunchKernel =
        load_symbol<hipExtModuleLaunchKernel_t>("hipExtModuleLaunchKernel");
    real_hipDrvLaunchKernelEx = load_symbol<hipDrvLaunchKernelEx_t>("hipDrvLaunchKernelEx");
    real_hipLaunchKernel = load_symbol<hipLaunchKernel_t>("hipLaunchKernel");
    real_hipDeviceSynchronize = load_symbol<hipDeviceSynchronize_t>("hipDeviceSynchronize");
    real_hipStreamSynchronize = load_symbol<hipStreamSynchronize_t>("hipStreamSynchronize");
    real_hipStreamSynchronize_spt =
        load_optional_symbol<hipStreamSynchronize_t>("hipStreamSynchronize_spt");
    real_hipEventSynchronize = load_symbol<hipEventSynchronize_t>("hipEventSynchronize");
    real_hipMemcpy = load_symbol<hipMemcpy_t>("hipMemcpy");
    real___hipRegisterFatBinary =
        load_symbol<hipRegisterFatBinary_t>("__hipRegisterFatBinary");
    real___hipUnregisterFatBinary =
        load_symbol<hipUnregisterFatBinary_t>("__hipUnregisterFatBinary");
    real___hipRegisterFunction =
        load_symbol<hipRegisterFunction_t>("__hipRegisterFunction");
    real_hsa_code_object_reader_create_from_memory =
        load_symbol<hsaReaderCreateMemory_t>("hsa_code_object_reader_create_from_memory");
    real_hsa_code_object_reader_create_from_file =
        load_symbol<hsaReaderCreateFile_t>("hsa_code_object_reader_create_from_file");
    real_hsa_code_object_reader_destroy =
        load_symbol<hsaReaderDestroy_t>("hsa_code_object_reader_destroy");
    real_hsa_ven_amd_loader_code_object_reader_create_from_file_with_offset_size =
        load_optional_symbol<hsaReaderCreateFileOffsetSize_t>(
            "hsa_ven_amd_loader_code_object_reader_create_from_file_with_offset_size");
    real_hsa_system_get_extension_table =
        load_symbol<hsaSystemGetExtensionTable_t>("hsa_system_get_extension_table");
    real_hsa_system_get_major_extension_table =
        load_symbol<hsaSystemGetMajorExtensionTable_t>("hsa_system_get_major_extension_table");
  });
}

uint8_t counter_to_byte(uint32_t counter) {
  if (counter >= 65536)
    return 7;
  if (counter >= 16384)
    return 6;
  if (counter >= 4096)
    return 5;
  if (counter >= 512)
    return 4;
  if (counter >= 3)
    return 3;
  if (counter >= 2)
    return 2;
  if (counter >= 1)
    return 1;
  return 0;
}

uint8_t merge_counter(uint8_t host_counter, uint8_t device_counter) {
  const uint32_t merged = host_counter + device_counter;
  const uint8_t merged8 = static_cast<uint8_t>(merged & 0xffu);
  if (merged == 0)
    return 0;
  return merged8 == 0 ? 1 : merged8;
}

uint8_t *map_trace_bits_locked() {
  const char *shm_id_env = getenv("__AFL_SHM_ID");
  if (shm_id_env == nullptr || shm_id_env[0] == '\0') {
    g_fallback_trace_bits.assign(kMapSize, 0);
    return g_fallback_trace_bits.data();
  }

  char *end = nullptr;
  long shm_id = strtol(shm_id_env, &end, 10);
  if (end == shm_id_env || shm_id < 0) {
    if (g_verbose)
      fprintf(stderr, "rocjitsu-afl: invalid __AFL_SHM_ID=%s\n", shm_id_env);
    g_fallback_trace_bits.assign(kMapSize, 0);
    return g_fallback_trace_bits.data();
  }

  void *mapped = shmat(static_cast<int>(shm_id), nullptr, 0);
  if (mapped == reinterpret_cast<void *>(-1)) {
    if (g_verbose)
      perror("rocjitsu-afl: shmat");
    g_fallback_trace_bits.assign(kMapSize, 0);
    return g_fallback_trace_bits.data();
  }
  return static_cast<uint8_t *>(mapped);
}

bool reset_device_state_locked(const char *context) {
  if (g_device_counters == nullptr)
    return false;

  ScopedInterceptionBypass bypass;
  hipError_t err = hipMemset(g_device_counters, 0, sizeof(uint32_t) * kDeviceStateSlots);
  if (err != hipSuccess) {
    if (g_verbose) {
      fprintf(stderr, "rocjitsu-afl: hipMemset %s failed: %s\n", context, hipGetErrorString(err));
    }
    return false;
  }

  std::fill(g_last_merged_counters.begin(), g_last_merged_counters.end(), 0);
  g_pending_launch_summary = {};
  return true;
}

bool ensure_runtime_locked() {
  if (g_runtime_ready)
    return true;
  if (g_runtime_failed)
    return false;

  g_trace_bits = map_trace_bits_locked();
  g_last_merged_counters.assign(kCoverageSlots, 0);
  hipError_t err = hipSuccess;
  {
    ScopedInterceptionBypass bypass;
    err = hipMalloc(reinterpret_cast<void **>(&g_device_counters),
                    sizeof(uint32_t) * kDeviceStateSlots);
  }
  if (err != hipSuccess) {
    if (g_verbose) {
      fprintf(stderr, "rocjitsu-afl: hipMalloc coverage failed: %s\n", hipGetErrorString(err));
    }
    g_runtime_failed = true;
    return false;
  }

  if (!reset_device_state_locked("coverage state")) {
    ScopedInterceptionBypass bypass;
    (void)hipFree(g_device_counters);
    g_device_counters = nullptr;
    g_runtime_failed = true;
    return false;
  }

  g_runtime_ready = true;
  if (g_verbose) {
    fprintf(stderr,
            "rocjitsu-afl: runtime initialized trace_bits=%p device_state=%p "
            "counter_slots=%u previous_bb_slots=%u\n",
            static_cast<void *>(g_trace_bits), static_cast<void *>(g_device_counters),
            kCoverageSlots, afl_dbi::kPreviousBbSlots);
  }
  return true;
}

uintptr_t function_key(hipFunction_t function) {
  return reinterpret_cast<uintptr_t>(function);
}

uintptr_t runtime_function_key(const void *function) {
  return reinterpret_cast<uintptr_t>(function);
}

uintptr_t runtime_registration_key(void **modules) {
  return reinterpret_cast<uintptr_t>(modules);
}

std::unordered_map<uintptr_t, std::string> &runtime_function_names() {
  static auto *names = new std::unordered_map<uintptr_t, std::string>();
  return *names;
}

std::unordered_map<uintptr_t, uintptr_t> &runtime_function_registrations() {
  static auto *registrations = new std::unordered_map<uintptr_t, uintptr_t>();
  return *registrations;
}

std::vector<RuntimeShadowSource> &runtime_shadow_sources() {
  static auto *sources = new std::vector<RuntimeShadowSource>();
  return *sources;
}

std::unordered_set<uint64_t> &runtime_shadow_source_hashes() {
  static auto *hashes = new std::unordered_set<uint64_t>();
  return *hashes;
}

std::unordered_map<std::string, RuntimeShadowFunction> &runtime_shadow_functions() {
  static auto *functions = new std::unordered_map<std::string, RuntimeShadowFunction>();
  return *functions;
}

std::mutex &runtime_function_names_mutex() {
  static auto *mutex = new std::mutex();
  return *mutex;
}

uintptr_t module_key(hipModule_t module) {
  return reinterpret_cast<uintptr_t>(module);
}

std::string registered_kernel_name_locked(hipFunction_t module_function,
                                          const void *runtime_function) {
  if (module_function != nullptr) {
    auto it = g_module_function_names.find(function_key(module_function));
    if (it != g_module_function_names.end())
      return it->second;
  }
  if (runtime_function != nullptr) {
    std::lock_guard<std::mutex> lock(runtime_function_names_mutex());
    auto &names = runtime_function_names();
    auto it = names.find(runtime_function_key(runtime_function));
    if (it != names.end())
      return it->second;
  }
  return "<unknown>";
}

std::string registered_kernel_name(hipFunction_t module_function,
                                   const void *runtime_function) {
  std::lock_guard<std::mutex> lock(g_state_mutex);
  return registered_kernel_name_locked(module_function, runtime_function);
}

uintptr_t registered_runtime_function_registration(const void *runtime_function) {
  if (runtime_function == nullptr)
    return 0;
  std::lock_guard<std::mutex> lock(runtime_function_names_mutex());
  auto &registrations = runtime_function_registrations();
  auto it = registrations.find(runtime_function_key(runtime_function));
  return it == registrations.end() ? 0 : it->second;
}

bool should_scope_hsa_reader_to_launch(std::string_view kernel_name) {
  return !kernel_name.empty() && kernel_name != "<unknown>" && g_kernel_include == nullptr &&
         g_kernel_include_override == nullptr;
}

void record_launch(const char *api, const char *kind, hipFunction_t module_function,
                   const void *runtime_function, unsigned int grid_x, unsigned int grid_y,
                   unsigned int grid_z, unsigned int block_x, unsigned int block_y,
                   unsigned int block_z, std::string_view kernel_override = {},
                   uintptr_t registration_key = 0) {
  std::lock_guard<std::mutex> lock(g_state_mutex);
  if (!ensure_runtime_locked())
    return;
  ++g_launches;

  std::string kernel_name;
  if (!kernel_override.empty()) {
    kernel_name = std::string(kernel_override);
  } else {
    kernel_name = registered_kernel_name_locked(module_function, runtime_function);
  }

  PendingLaunchSummary &pending = g_pending_launch_summary;
  if (pending.launches == 0) {
    pending.first_api = api != nullptr ? api : "";
    pending.first_kernel = kernel_name;
  }
  ++pending.launches;
  const std::string_view launch_kind = kind != nullptr ? std::string_view(kind) : std::string_view();
  if (launch_kind == "runtime_shadow") {
    ++pending.runtime_shadow_launches;
  } else if (launch_kind == "lazy_ccob_shadow") {
    ++pending.lazy_ccob_launches;
  } else if (launch_kind == "hip_runtime" || launch_kind == "hip_runtime_scoped") {
    ++pending.hip_runtime_launches;
  } else {
    ++pending.hip_module_launches;
  }
  pending.last_api = api != nullptr ? api : "";
  pending.last_kind = kind != nullptr ? kind : "";
  pending.last_kernel = kernel_name;
  pending.last_module_function = function_key(module_function);
  pending.last_runtime_function = runtime_function_key(runtime_function);
  pending.last_registration = registration_key;

  if (g_verbose) {
    fprintf(stderr,
            "rocjitsu-afl: launch api=%s kind=%s kernel=%s module_function=%p "
            "runtime_function=%p grid=(%u,%u,%u) block=(%u,%u,%u)\n",
            api, kind != nullptr ? kind : "", kernel_name.c_str(),
            reinterpret_cast<void *>(module_function), runtime_function, grid_x, grid_y, grid_z,
            block_x, block_y, block_z);
  }
}

bool merge_coverage_locked(const char *trigger = "coverage merge") {
  if (!g_runtime_ready || g_device_counters == nullptr || g_trace_bits == nullptr)
    return true;

  std::vector<uint32_t> host(kCoverageSlots, 0);
  ScopedInterceptionBypass bypass;
  hipError_t err = hipSuccess;
  {
    ScopedInterceptionBypass bypass;
    err = hipMemcpy(host.data(), g_device_counters, sizeof(uint32_t) * host.size(),
                    hipMemcpyDeviceToHost);
  }
  if (err != hipSuccess) {
    if (g_verbose) {
      fprintf(stderr, "rocjitsu-afl: hipMemcpy coverage failed: %s\n", hipGetErrorString(err));
    }
    return false;
  }

  uint32_t entry_delta = 0;
  uint32_t edge_slot_delta_count = 0;
  uint64_t edge_counter_delta_total = 0;
  for (uint32_t i = 0; i < kCoverageSlots; ++i) {
    const uint32_t delta = host[i] - g_last_merged_counters[i];
    g_last_merged_counters[i] = host[i];
    if (i == kEntryCounterSlot) {
      entry_delta = delta;
    } else if (i >= kFirstEdgeCounterSlot && delta != 0) {
      ++edge_slot_delta_count;
      edge_counter_delta_total += delta;
    }
    const uint8_t device_byte = counter_to_byte(delta);
    g_trace_bits[kDeviceStart + i] = merge_counter(g_trace_bits[kDeviceStart + i], device_byte);
  }

  uint32_t nonzero_edge_slots = 0;
  for (uint32_t i = kFirstEdgeCounterSlot; i < kCoverageSlots; ++i) {
    if (host[i] != 0)
      ++nonzero_edge_slots;
  }
  if (nonzero_edge_slots != 0)
    g_required_device_edges_observed = true;

  if (g_verbose) {
    fprintf(stderr,
            "rocjitsu-afl: launches=%llu slot0=%u edge_slots=%u "
            "trace_bits[%u]=%u\n",
            static_cast<unsigned long long>(g_launches), host[0], nonzero_edge_slots, kDeviceStart,
            static_cast<unsigned>(g_trace_bits[kDeviceStart]));
  }
  if (g_pending_launch_summary.launches != 0) {
    emit_device_edge_delta_report(trigger, g_pending_launch_summary, entry_delta,
                                  edge_slot_delta_count, edge_counter_delta_total,
                                  nonzero_edge_slots);
    g_pending_launch_summary = {};
  }
  if (g_require_device_edges && g_launches != 0 && !g_required_device_edges_observed) {
    if (!g_required_device_edges_warned) {
      fprintf(stderr,
              "rocjitsu-afl: required device branch coverage has not been observed yet\n");
      emit_required_device_edges_report(trigger, /*fatal=*/false, nonzero_edge_slots);
      g_required_device_edges_warned = true;
    }
  }
  return true;
}

template <typename T>
std::optional<T> read_struct(std::span<const uint8_t> image, uint64_t offset) {
  if (offset > image.size() || sizeof(T) > image.size() - offset)
    return std::nullopt;
  T out{};
  memcpy(&out, image.data() + offset, sizeof(T));
  return out;
}

std::mutex &patch_report_mutex() {
  static auto *mutex = new std::mutex();
  return *mutex;
}

bool patch_reports_enabled() {
  return g_patch_report_path != nullptr && g_patch_report_path[0] != '\0';
}

void append_json_string(std::string &out, std::string_view value) {
  out.push_back('"');
  for (unsigned char c : value) {
    switch (c) {
    case '\\':
      out += "\\\\";
      break;
    case '"':
      out += "\\\"";
      break;
    case '\b':
      out += "\\b";
      break;
    case '\f':
      out += "\\f";
      break;
    case '\n':
      out += "\\n";
      break;
    case '\r':
      out += "\\r";
      break;
    case '\t':
      out += "\\t";
      break;
    default:
      if (c < 0x20) {
        char escaped[7];
        snprintf(escaped, sizeof(escaped), "\\u%04x", static_cast<unsigned>(c));
        out += escaped;
      } else {
        out.push_back(static_cast<char>(c));
      }
      break;
    }
  }
  out.push_back('"');
}

void append_json_field_name(std::string &out, std::string_view name) {
  out.push_back(',');
  append_json_string(out, name);
  out.push_back(':');
}

void append_json_string_field(std::string &out, std::string_view name, std::string_view value) {
  append_json_field_name(out, name);
  append_json_string(out, value);
}

void append_json_u64_field(std::string &out, std::string_view name, uint64_t value) {
  append_json_field_name(out, name);
  out += std::to_string(value);
}

void append_json_bool_field(std::string &out, std::string_view name, bool value) {
  append_json_field_name(out, name);
  out += value ? "true" : "false";
}

void append_json_u8_array_field(std::string &out, std::string_view name,
                                const std::vector<uint8_t> &values) {
  append_json_field_name(out, name);
  out.push_back('[');
  bool first = true;
  for (uint8_t value : values) {
    if (!first)
      out.push_back(',');
    first = false;
    out += std::to_string(static_cast<uint32_t>(value));
  }
  out.push_back(']');
}

std::string hex_u64(uint64_t value) {
  char buffer[32];
  snprintf(buffer, sizeof(buffer), "0x%llx", static_cast<unsigned long long>(value));
  return std::string(buffer);
}

void append_json_hex_field(std::string &out, std::string_view name, uint64_t value) {
  append_json_string_field(out, name, hex_u64(value));
}

void append_patch_report_line(const std::string &line) {
  if (!patch_reports_enabled())
    return;

  std::lock_guard<std::mutex> lock(patch_report_mutex());
  const int fd = open(g_patch_report_path, O_WRONLY | O_CREAT | O_APPEND | O_CLOEXEC, 0644);
  if (fd < 0) {
    if (g_verbose)
      perror("rocjitsu-afl: open patch report");
    return;
  }

  size_t written = 0;
  while (written < line.size()) {
    const ssize_t rc = write(fd, line.data() + written, line.size() - written);
    if (rc <= 0)
      break;
    written += static_cast<size_t>(rc);
  }
  close(fd);
}

template <typename Fn> void emit_patch_report_event(std::string_view event, Fn write_fields) {
  if (!patch_reports_enabled())
    return;

  std::string out;
  out.reserve(512);
  out += "{\"event\":";
  append_json_string(out, event);
  write_fields(out);
  out += "}\n";
  append_patch_report_line(out);
}

void emit_code_object_image_report(const char *context, size_t image_bytes,
                                   const afl_dbi::CodeObjectImageSummary &summary) {
  emit_patch_report_event("code_object_image", [&](std::string &out) {
    append_json_string_field(out, "context", context != nullptr ? context : "");
    append_json_u64_field(out, "image_bytes", image_bytes);
    append_json_bool_field(out, "top_level_raw_elf", summary.top_level_raw_elf);
    append_json_bool_field(out, "top_level_bundle", summary.top_level_bundle);
    append_json_bool_field(out, "top_level_ccob", summary.top_level_ccob);
    append_json_u64_field(out, "device_images", summary.device_image_count);
    append_json_u64_field(out, "top_level_ccob_device_images",
                          summary.top_level_ccob_device_image_count);
    append_json_bool_field(out, "raw_elf_bypass_drops_sibling_payloads",
                           summary.raw_elf_bypass_drops_sibling_payloads);
    append_json_hex_field(out, "runtime_shadow_registration",
                          g_runtime_shadow_registration_override);
  });
}

void append_device_image_index_field(std::string &out, uint32_t index) {
  if (index != kUnknownDeviceImageIndex)
    append_json_u64_field(out, "device_image_index", index);
}

void emit_ccob_rebuild_report(const char *context, std::string_view device_image_id,
                              uint32_t device_image_index, bool success, size_t input_bytes,
                              size_t output_bytes, uint32_t input_device_images,
                              uint32_t output_device_images, bool sibling_payloads_preserved) {
  emit_patch_report_event("ccob_rebuild", [&](std::string &out) {
    append_json_string_field(out, "context", context != nullptr ? context : "");
    append_json_string_field(out, "device_image", device_image_id);
    append_device_image_index_field(out, device_image_index);
    append_json_bool_field(out, "success", success);
    append_json_u64_field(out, "input_bytes", input_bytes);
    append_json_u64_field(out, "output_bytes", output_bytes);
    append_json_u64_field(out, "input_device_images", input_device_images);
    append_json_u64_field(out, "output_device_images", output_device_images);
    append_json_bool_field(out, "sibling_payloads_preserved", sibling_payloads_preserved);
  });
}

void emit_hsa_reader_patch_report(const char *context, std::string_view device_image_id,
                                  uint32_t device_image_index, bool primary_rebuilt_container,
                                  bool fallback_available, bool fallback_used,
                                  hsa_status_t primary_status, hsa_status_t final_status,
                                  uint64_t reader_handle,
                                  uint64_t primary_reader_handle,
                                  uint64_t fallback_reader_handle, size_t primary_bytes,
                                  size_t fallback_bytes) {
  emit_patch_report_event("hsa_reader_patch", [&](std::string &out) {
    append_json_string_field(out, "context", context != nullptr ? context : "");
    append_json_string_field(out, "device_image", device_image_id);
    append_device_image_index_field(out, device_image_index);
    append_json_bool_field(out, "primary_rebuilt_ccob", primary_rebuilt_container);
    append_json_bool_field(out, "primary_rebuilt_container", primary_rebuilt_container);
    append_json_bool_field(out, "fallback_available", fallback_available);
    append_json_bool_field(out, "fallback_used", fallback_used);
    append_json_u64_field(out, "primary_status", static_cast<uint64_t>(primary_status));
    append_json_u64_field(out, "final_status", static_cast<uint64_t>(final_status));
    append_json_hex_field(out, "reader", reader_handle);
    append_json_hex_field(out, "primary_reader", primary_reader_handle);
    append_json_hex_field(out, "fallback_reader", fallback_reader_handle);
    append_json_u64_field(out, "primary_bytes", primary_bytes);
    append_json_u64_field(out, "fallback_bytes", fallback_bytes);
  });
}

void emit_lazy_ccob_module_report(hipModule_t module, const char *context, size_t image_bytes,
                                  const std::optional<afl_dbi::CodeObjectImageSummary> &summary) {
  emit_patch_report_event("lazy_ccob_module", [&](std::string &out) {
    append_json_hex_field(out, "module", module_key(module));
    append_json_string_field(out, "context", context != nullptr ? context : "");
    append_json_u64_field(out, "image_bytes", image_bytes);
    append_json_string_field(out, "lazy_shadow_policy", kLazyCcobShadowPolicy);
    append_json_bool_field(out, "lazy_shadow_preserves_sibling_payloads", false);
    if (summary) {
      append_json_bool_field(out, "top_level_raw_elf", summary->top_level_raw_elf);
      append_json_bool_field(out, "top_level_bundle", summary->top_level_bundle);
      append_json_bool_field(out, "top_level_ccob", summary->top_level_ccob);
      append_json_u64_field(out, "device_images", summary->device_image_count);
      append_json_u64_field(out, "top_level_ccob_device_images",
                            summary->top_level_ccob_device_image_count);
      append_json_bool_field(out, "raw_elf_bypass_drops_sibling_payloads",
                             summary->raw_elf_bypass_drops_sibling_payloads);
    }
  });
}

void emit_lazy_ccob_function_report(hipModule_t module, std::string_view kernel,
                                    LazyPatchedFunction published, hipModule_t created_module,
                                    bool cache_hit, bool inserted, bool duplicate_unloaded,
                                    bool owner_missing, size_t total_functions) {
  emit_patch_report_event("lazy_ccob_function", [&](std::string &out) {
    append_json_hex_field(out, "module", module_key(module));
    append_json_string_field(out, "kernel", kernel);
    append_json_hex_field(out, "patched_module", module_key(published.module));
    append_json_hex_field(out, "function", function_key(published.function));
    append_json_hex_field(out, "created_module", module_key(created_module));
    append_json_string_field(out, "shadow_image_policy", kLazyCcobShadowPolicy);
    append_json_bool_field(out, "cache_hit", cache_hit);
    append_json_bool_field(out, "inserted", inserted);
    append_json_bool_field(out, "duplicate_unloaded", duplicate_unloaded);
    append_json_bool_field(out, "owner_missing", owner_missing);
    append_json_u64_field(out, "total_functions", total_functions);
  });
}

void emit_lazy_ccob_release_report(hipModule_t module, size_t shadow_modules,
                                   const ShadowModuleUnloadStats &unload_stats) {
  emit_patch_report_event("lazy_ccob_release", [&](std::string &out) {
    append_json_hex_field(out, "module", module_key(module));
    append_json_u64_field(out, "shadow_modules", shadow_modules);
    append_json_u64_field(out, "unload_attempts", unload_stats.attempts);
    append_json_u64_field(out, "unload_failures", unload_stats.failures);
    append_json_u64_field(out, "last_unload_error",
                          static_cast<uint64_t>(unload_stats.last_error));
  });
}

void emit_lazy_ccob_launch_report(const LazyCcobLaunchBinding &binding, unsigned int grid_x,
                                  unsigned int grid_y, unsigned int grid_z,
                                  unsigned int block_x, unsigned int block_y,
                                  unsigned int block_z, size_t shared_mem_bytes) {
  emit_patch_report_event("lazy_ccob_launch", [&](std::string &out) {
    append_json_hex_field(out, "module", binding.module);
    append_json_string_field(out, "kernel", binding.kernel);
    append_json_hex_field(out, "patched_module", module_key(binding.target.module));
    append_json_hex_field(out, "function", function_key(binding.target.function));
    append_json_u64_field(out, "grid_x", grid_x);
    append_json_u64_field(out, "grid_y", grid_y);
    append_json_u64_field(out, "grid_z", grid_z);
    append_json_u64_field(out, "block_x", block_x);
    append_json_u64_field(out, "block_y", block_y);
    append_json_u64_field(out, "block_z", block_z);
    append_json_u64_field(out, "shared_mem_bytes", shared_mem_bytes);
  });
}

void emit_runtime_shadow_source_report(const char *context, uintptr_t registration_key,
                                       size_t image_bytes, uint64_t image_hash,
                                       uint64_t dedupe_key, bool inserted,
                                       size_t total_sources,
                                       const std::vector<std::string> &kernel_names) {
  emit_patch_report_event("runtime_shadow_source", [&](std::string &out) {
    append_json_string_field(out, "context", context != nullptr ? context : "");
    append_json_hex_field(out, "registration", registration_key);
    append_json_u64_field(out, "image_bytes", image_bytes);
    append_json_hex_field(out, "image_hash", image_hash);
    append_json_hex_field(out, "dedupe_key", dedupe_key);
    append_json_bool_field(out, "inserted", inserted);
    append_json_u64_field(out, "total_sources", total_sources);
    append_json_u64_field(out, "kernel_count", kernel_names.size());
    append_json_string_field(out, "first_kernel",
                             kernel_names.empty() ? "" : kernel_names.front());
  });
}

void emit_runtime_shadow_function_report(std::string_view kernel, uintptr_t registration_key,
                                         LazyPatchedFunction published,
                                         hipModule_t created_module, bool cache_hit,
                                         bool inserted, bool duplicate_unloaded,
                                         size_t total_functions, size_t source_count = 0,
                                         size_t attempted_sources = 0,
                                         uint64_t source_dedupe_key = 0,
                                         uint64_t patch_us = 0, uint64_t load_us = 0,
                                         uint64_t publish_us = 0,
                                         size_t matching_sources = 0) {
  emit_patch_report_event("runtime_shadow_function", [&](std::string &out) {
    append_json_string_field(out, "kernel", kernel);
    append_json_hex_field(out, "registration", registration_key);
    append_json_hex_field(out, "module", module_key(published.module));
    append_json_hex_field(out, "function", function_key(published.function));
    append_json_hex_field(out, "created_module", module_key(created_module));
    append_json_bool_field(out, "cache_hit", cache_hit);
    append_json_bool_field(out, "inserted", inserted);
    append_json_bool_field(out, "duplicate_unloaded", duplicate_unloaded);
    append_json_u64_field(out, "total_functions", total_functions);
    append_json_u64_field(out, "source_count", source_count);
    append_json_u64_field(out, "attempted_sources", attempted_sources);
    append_json_hex_field(out, "source_dedupe_key", source_dedupe_key);
    append_json_u64_field(out, "patch_us", patch_us);
    append_json_u64_field(out, "load_us", load_us);
    append_json_u64_field(out, "publish_us", publish_us);
    append_json_u64_field(out, "matching_sources", matching_sources);
    append_json_bool_field(out, "ambiguous_source_matches", matching_sources > 1);
  });
}

void emit_runtime_shadow_launch_report(std::string_view kernel, uintptr_t registration_key,
                                       const void *runtime_function,
                                       LazyPatchedFunction shadow, dim3 num_blocks,
                                       dim3 dim_blocks, size_t shared_mem_bytes) {
  emit_patch_report_event("runtime_shadow_launch", [&](std::string &out) {
    append_json_string_field(out, "kernel", kernel);
    append_json_hex_field(out, "registration", registration_key);
    append_json_hex_field(out, "runtime_function",
                          reinterpret_cast<uintptr_t>(runtime_function));
    append_json_hex_field(out, "module", module_key(shadow.module));
    append_json_hex_field(out, "function", function_key(shadow.function));
    append_json_u64_field(out, "grid_x", num_blocks.x);
    append_json_u64_field(out, "grid_y", num_blocks.y);
    append_json_u64_field(out, "grid_z", num_blocks.z);
    append_json_u64_field(out, "block_x", dim_blocks.x);
    append_json_u64_field(out, "block_y", dim_blocks.y);
    append_json_u64_field(out, "block_z", dim_blocks.z);
    append_json_u64_field(out, "shared_mem_bytes", shared_mem_bytes);
  });
}

void emit_runtime_shadow_miss_report(std::string_view kernel, uintptr_t registration_key,
                                     std::string_view reason, size_t source_count,
                                     size_t attempted_sources, size_t patch_failures,
                                     size_t load_failures,
                                     size_t matching_sources = 0) {
  emit_patch_report_event("runtime_shadow_miss", [&](std::string &out) {
    append_json_string_field(out, "kernel", kernel);
    append_json_hex_field(out, "registration", registration_key);
    append_json_string_field(out, "reason", reason);
    append_json_u64_field(out, "source_count", source_count);
    append_json_u64_field(out, "attempted_sources", attempted_sources);
    append_json_u64_field(out, "patch_failures", patch_failures);
    append_json_u64_field(out, "load_failures", load_failures);
    append_json_u64_field(out, "matching_sources", matching_sources);
    append_json_bool_field(out, "ambiguous_source_matches", matching_sources > 1);
  });
}

void emit_runtime_shadow_release_report(uintptr_t registration_key, const char *context,
                                        size_t removed_sources, size_t removed_functions,
                                        size_t total_sources, size_t total_functions,
                                        size_t shadow_modules,
                                        const ShadowModuleUnloadStats &unload_stats) {
  emit_patch_report_event("runtime_shadow_registration_release", [&](std::string &out) {
    append_json_hex_field(out, "registration", registration_key);
    append_json_string_field(out, "context", context != nullptr ? context : "");
    append_json_u64_field(out, "removed_sources", removed_sources);
    append_json_u64_field(out, "removed_functions", removed_functions);
    append_json_u64_field(out, "total_sources_before", total_sources);
    append_json_u64_field(out, "total_functions_before", total_functions);
    append_json_u64_field(out, "shadow_modules", shadow_modules);
    append_json_u64_field(out, "unload_attempts", unload_stats.attempts);
    append_json_u64_field(out, "unload_failures", unload_stats.failures);
    append_json_u64_field(out, "last_unload_error",
                          static_cast<uint64_t>(unload_stats.last_error));
  });
}

void emit_device_edge_delta_report(const char *trigger, const PendingLaunchSummary &launches,
                                   uint32_t entry_delta, uint32_t edge_slot_delta_count,
                                   uint64_t edge_counter_delta_total,
                                   uint32_t nonzero_edge_slots_total) {
  emit_patch_report_event("device_edge_delta", [&](std::string &out) {
    append_json_string_field(out, "trigger", trigger != nullptr ? trigger : "");
    append_json_u64_field(out, "launches", launches.launches);
    append_json_u64_field(out, "hip_module_launches", launches.hip_module_launches);
    append_json_u64_field(out, "hip_runtime_launches", launches.hip_runtime_launches);
    append_json_u64_field(out, "runtime_shadow_launches", launches.runtime_shadow_launches);
    append_json_u64_field(out, "lazy_ccob_launches", launches.lazy_ccob_launches);
    append_json_string_field(out, "first_api", launches.first_api);
    append_json_string_field(out, "first_kernel", launches.first_kernel);
    append_json_string_field(out, "last_api", launches.last_api);
    append_json_string_field(out, "last_kind", launches.last_kind);
    append_json_string_field(out, "last_kernel", launches.last_kernel);
    append_json_hex_field(out, "last_module_function", launches.last_module_function);
    append_json_hex_field(out, "last_runtime_function", launches.last_runtime_function);
    append_json_hex_field(out, "last_registration", launches.last_registration);
    append_json_u64_field(out, "entry_delta", entry_delta);
    append_json_u64_field(out, "edge_slot_delta_count", edge_slot_delta_count);
    append_json_u64_field(out, "edge_counter_delta_total", edge_counter_delta_total);
    append_json_u64_field(out, "nonzero_edge_slots_total", nonzero_edge_slots_total);
    append_json_u64_field(out, "launches_total", g_launches);
  });
}

void emit_required_device_edges_report(const char *trigger, bool fatal,
                                       uint32_t nonzero_edge_slots_total) {
  emit_patch_report_event("required_device_edges", [&](std::string &out) {
    append_json_string_field(out, "trigger", trigger != nullptr ? trigger : "");
    append_json_bool_field(out, "fatal", fatal);
    append_json_bool_field(out, "observed", g_required_device_edges_observed);
    append_json_u64_field(out, "nonzero_edge_slots_total", nonzero_edge_slots_total);
    append_json_u64_field(out, "launches_total", g_launches);
  });
}

void enforce_required_device_edges_at_exit_locked() {
  if (!g_require_device_edges || g_launches == 0 || g_required_device_edges_observed)
    return;

  emit_required_device_edges_report("process_exit", /*fatal=*/true,
                                    /*nonzero_edge_slots_total=*/0);
  fprintf(stderr,
          "rocjitsu-afl: required device branch coverage was never observed; "
          "failing strict smoke gate\n");
  fflush(stderr);
  _exit(86);
}

bool summary_has_previous_bb_signal(const KernelEdgeSelectionSummary &summary) {
  return summary.slot_policy_summary.hashed_edge_sites != 0;
}

bool summary_has_fixed_counter_signal(const KernelEdgeSelectionSummary &summary) {
  return summary.slot_policy_summary.fixed_edge_sites != 0;
}

const char *coverage_signal_name(const KernelEdgeSelectionSummary &summary) {
  const bool previous_bb = summary_has_previous_bb_signal(summary);
  const bool fixed_counter = summary_has_fixed_counter_signal(summary);
  if (previous_bb && fixed_counter)
    return "hybrid-previous-bb-and-fixed";
  if (previous_bb)
    return "previous-bb-edges";
  if (fixed_counter)
    return "fixed-branch-counters";
  if (summary.block_selected != 0 || summary.branch_edges_selected != 0)
    return "edge-sites-without-slot-policy";
  return "none";
}

const char *coverage_signal_reason(const KernelEdgeSelectionSummary &summary) {
  const bool previous_bb = summary_has_previous_bb_signal(summary);
  const bool fixed_counter = summary_has_fixed_counter_signal(summary);
  if (previous_bb && fixed_counter)
    return "previous-BB probes selected where safe and fixed counters selected for degraded sites";
  if (previous_bb)
    return "selected previous-BB hashed edge sites";
  if (fixed_counter)
    return "selected fixed branch counter sites";
  if (summary.block_selected != 0 || summary.branch_edges_selected != 0)
    return "selected edge sites without a populated slot policy summary";
  return "no selected device edge sites";
}

bool report_has_previous_bb_signal(const PatchDeviceElfReport &report) {
  return report.hashed_edge_sites != 0;
}

bool report_has_fixed_counter_signal(const PatchDeviceElfReport &report) {
  return report.fixed_edge_sites != 0;
}

const char *coverage_signal_name(const PatchDeviceElfReport &report) {
  if (!report.success)
    return "none";
  const bool previous_bb = report_has_previous_bb_signal(report);
  const bool fixed_counter = report_has_fixed_counter_signal(report);
  if (previous_bb && fixed_counter)
    return "hybrid-previous-bb-and-fixed";
  if (previous_bb)
    return "previous-bb-edges";
  if (fixed_counter)
    return "fixed-branch-counters";
  if (report.entry_patched != 0)
    return "entry-counter";
  return "none";
}

const char *coverage_signal_reason(const PatchDeviceElfReport &report) {
  if (!report.success)
    return "patch event did not install device coverage";
  const bool previous_bb = report_has_previous_bb_signal(report);
  const bool fixed_counter = report_has_fixed_counter_signal(report);
  if (previous_bb && fixed_counter)
    return "previous-BB probes selected where safe and fixed counters selected for degraded or fixed-preferred sites";
  if (previous_bb)
    return "selected previous-BB hashed edge sites";
  if (fixed_counter)
    return "selected fixed branch counter sites";
  if (report.entry_patched != 0) {
    return report.edge_instrumentation_enabled
               ? "selected entry-backed coverage only"
               : "target uses the conservative entry-counter tier";
  }
  return "no patched device coverage sites";
}

void emit_patch_report(const PatchDeviceElfReport &report) {
  if (!patch_reports_enabled())
    return;

  std::string out;
  out.reserve(2048);
  out += "{\"event\":\"patch_device_elf\"";
  append_json_string_field(out, "context", report.context);
  append_json_string_field(out, "device_image", report.device_image_id);
  append_device_image_index_field(out, report.device_image_index);
  append_json_string_field(out, "kernel_filter", report.kernel_filter);
  append_json_string_field(out, "runtime_shadow_registration",
                           hex_u64(report.runtime_shadow_registration));
  append_json_string_field(out, "gfxip", report.gfxip);
  append_json_string_field(out, "arch", report.arch);
  append_json_string_field(out, "reason", report.reason);
  if (!report.failure_phase.empty())
    append_json_string_field(out, "failure_phase", report.failure_phase);
  append_json_bool_field(out, "success", report.success);
  append_json_bool_field(out, "supports_previous_bb_edges", report.supports_previous_bb_edges);
  append_json_bool_field(out, "edge_instrumentation_enabled", report.edge_instrumentation_enabled);
  append_json_string_field(out, "edge_instrumentation_reason",
                           report.edge_instrumentation_reason);
  append_json_string_field(out, "coverage_strategy", report.coverage_strategy);
  append_json_string_field(out, "coverage_strategy_reason",
                           report.coverage_strategy_reason);
  append_json_string_field(out, "coverage_signal", coverage_signal_name(report));
  append_json_string_field(out, "coverage_signal_reason",
                           coverage_signal_reason(report));
  append_json_bool_field(out, "skip_entry_probe", report.skip_entry_probe);
  append_json_bool_field(out, "fixed_edge_slots", report.fixed_edge_slots);
  append_json_bool_field(out, "branch_edge_slots", report.branch_edge_slots);
  append_json_bool_field(out, "hybrid_edge_probes", report.hybrid_edge_probes);
  append_json_string_field(out, "branch_edge_slot_policy",
                           afl_dbi::edge_slot_policy_name(report.branch_edge_slot_policy));
  append_json_bool_field(out, "require_liveness_registers", report.require_liveness_registers);
  append_json_bool_field(out, "allow_opaque_fresh_registers",
                         report.allow_opaque_fresh_registers);
  append_json_bool_field(out, "vgpr_scratch_spills_requested",
                         report.vgpr_scratch_spills_requested);
  append_json_bool_field(out, "vgpr_scratch_spills_enabled",
                         report.vgpr_scratch_spills_enabled);
  append_json_string_field(out, "vgpr_scratch_spill_reason",
                           report.vgpr_scratch_spill_reason);
  append_json_u64_field(out, "input_bytes", report.input_bytes);
  append_json_u64_field(out, "output_bytes", report.output_bytes);
  append_json_u64_field(out, "kernel_descriptors", report.kernel_descriptor_count);
  append_json_u64_field(out, "entry_candidates", report.entry_candidate_count);
  append_json_u64_field(out, "descriptor_updates", report.descriptor_updates);
  append_json_u64_field(out, "entries_patched", report.entry_patched);
  append_json_u64_field(out, "edge_sites_selected", report.edge_sites_selected);
  append_json_u64_field(out, "edge_sites_patched", report.edge_sites_patched);
  append_json_u64_field(out, "edge_patch_failures", report.edge_patch_failures);
  append_json_string_field(out, "descriptor_resource_failure_reason",
                           report.descriptor_resource_failure_reason);
  append_json_u64_field(out, "local_text_caves", report.local_text_caves);
  append_json_u64_field(out, "local_text_cave_ranges", report.local_text_cave_ranges);
  append_json_u64_field(out, "local_text_cave_bytes", report.local_text_cave_bytes);
  append_json_u64_field(out, "largest_local_text_cave_bytes",
                        report.largest_local_text_cave_bytes);
  append_json_u64_field(out, "appended_caves", report.appended_caves);
  append_json_u64_field(out, "appended_cave_bytes", report.appended_cave_bytes);
  append_json_u64_field(out, "branch_range_failures", report.branch_range_failures);
  append_json_u64_field(out, "hashed_edge_sites", report.hashed_edge_sites);
  append_json_u64_field(out, "fixed_edge_sites", report.fixed_edge_sites);
  append_json_u64_field(out, "fixed_slot_requests", report.fixed_slot_requests);
  append_json_u64_field(out, "fixed_slots_reserved", report.fixed_slots_reserved);
  append_json_u64_field(out, "fixed_slot_exhaustions", report.fixed_slot_exhaustions);
  append_json_u64_field(out, "fixed_slot_collisions", report.fixed_slot_collisions);
  append_json_u64_field(out, "inline_slot_requests", report.inline_slot_requests);
  append_json_u64_field(out, "inline_slot_exhaustions", report.inline_slot_exhaustions);
  append_json_u64_field(out, "branch_edges_degraded_to_fixed",
                        report.branch_edges_degraded_to_fixed);
  append_json_u64_field(out, "fixed_counter_branch_edge_aggregate_fallback_used",
                        report.fixed_counter_branch_edge_aggregate_fallback_used);
  append_json_u64_field(out, "fixed_counter_branch_edge_safety_fallback_used",
                        report.fixed_counter_branch_edge_safety_fallback_used);
  append_json_u64_field(out, "fixed_counter_branch_edge_liveness_fallback_used",
                        report.fixed_counter_branch_edge_liveness_fallback_used);
  append_json_u64_field(out, "fixed_counter_branch_edge_placement_fallback_used",
                        report.fixed_counter_branch_edge_placement_fallback_used);
  append_json_u64_field(out, "exec_empty_fixed_counter_edges",
                        report.exec_empty_fixed_counter_edges);
  append_json_u64_field(out, "previous_bb_branch_edges_selected",
                        report.previous_bb_branch_edges_selected);
  append_json_u64_field(out, "previous_bb_branch_sites_selected",
                        report.previous_bb_branch_sites_selected);
  append_json_u64_field(out, "previous_bb_branch_sites_degraded_to_fixed",
                        report.previous_bb_branch_sites_degraded_to_fixed);
  append_json_u64_field(out, "edge_trampolines_planned", report.edge_trampolines_planned);
  append_json_u64_field(out, "previous_bb_branch_edge_trampolines_planned",
                        report.previous_bb_branch_edge_trampolines_planned);
  append_json_u64_field(out, "planned_appended_edge_trampolines",
                        report.planned_appended_edge_trampolines);
  append_json_u64_field(out, "planned_local_edge_trampolines",
                        report.planned_local_edge_trampolines);
  append_json_u64_field(
      out, "previous_bb_branch_planned_appended_edge_trampolines",
      report.previous_bb_branch_planned_appended_edge_trampolines);
  append_json_u64_field(
      out, "previous_bb_branch_planned_local_edge_trampolines",
      report.previous_bb_branch_planned_local_edge_trampolines);
  append_json_u64_field(out, "planned_edge_trampoline_bytes",
                        report.planned_edge_trampoline_bytes);
  append_json_u64_field(out, "previous_bb_branch_edge_trampoline_bytes",
                        report.previous_bb_branch_edge_trampoline_bytes);
  append_json_u64_field(out, "planned_appended_edge_trampoline_bytes",
                        report.planned_appended_edge_trampoline_bytes);
  append_json_u64_field(out, "planned_local_edge_trampoline_bytes",
                        report.planned_local_edge_trampoline_bytes);
  append_json_u64_field(
      out, "previous_bb_branch_planned_appended_edge_trampoline_bytes",
      report.previous_bb_branch_planned_appended_edge_trampoline_bytes);
  append_json_u64_field(
      out, "previous_bb_branch_planned_local_edge_trampoline_bytes",
      report.previous_bb_branch_planned_local_edge_trampoline_bytes);
  append_json_u64_field(out, "largest_edge_trampoline_bytes",
                        report.largest_edge_trampoline_bytes);
  append_json_u64_field(out, "largest_previous_bb_branch_edge_trampoline_bytes",
                        report.largest_previous_bb_branch_edge_trampoline_bytes);
  append_json_u64_field(out, "previous_bb_branch_afl_map_budget",
                        report.previous_bb_branch_afl_map_budget);
  append_json_u64_field(out, "previous_bb_branch_afl_map_pressure_ppm",
                        report.previous_bb_branch_afl_map_pressure_ppm);
  append_json_u64_field(out, "previous_bb_branch_trampoline_avg_bytes_x100",
                        report.previous_bb_branch_trampoline_avg_bytes_x100);
  append_json_u64_field(out, "previous_bb_branch_appended_trampoline_ratio_ppm",
                        report.previous_bb_branch_appended_trampoline_ratio_ppm);
  append_json_u64_field(out, "previous_bb_branch_local_trampoline_ratio_ppm",
                        report.previous_bb_branch_local_trampoline_ratio_ppm);
  append_json_u64_field(out, "previous_bb_branch_code_growth_pressure_ppm",
                        report.previous_bb_branch_code_growth_pressure_ppm);
  append_json_string_field(out, "previous_bb_branch_overhead_status",
                           report.previous_bb_branch_overhead_status);
  append_json_string_field(out, "previous_bb_branch_overhead_reason",
                           report.previous_bb_branch_overhead_reason);
  append_json_u64_field(out, "edge_site_limit", report.edge_site_limit);
  append_json_u64_field(out, "branch_edge_site_limit", report.branch_edge_site_limit);
  append_json_bool_field(out, "branch_edge_site_limit_auto",
                         report.branch_edge_site_limit_auto);
  append_json_u64_field(out, "previous_bb_branch_site_limit",
                        report.previous_bb_branch_site_limit);
  append_json_bool_field(out, "previous_bb_branch_site_limit_auto",
                         report.previous_bb_branch_site_limit_auto);
  append_json_u64_field(out, "coverage_slots", kCoverageSlots);
  append_json_u64_field(out, "first_edge_counter_slot", kFirstEdgeCounterSlot);
  append_json_u64_field(out, "max_inline_counter_slot", kMaxInlineCounterSlot);
  append_json_u64_field(out, "max_fixed_counter_slot", afl_dbi::kMaxFixedCounterSlot);
  append_json_u64_field(out, "fixed_edge_slot_budget",
                        afl_dbi::kMaxFixedCounterSlot - kFirstEdgeCounterSlot + 1);
  append_json_u64_field(out, "inline_edge_slot_budget",
                        kMaxInlineCounterSlot - kFirstEdgeCounterSlot + 1);
  const bool inline_edge_slots_active =
      report.fixed_edge_slots ||
      (report.branch_edge_slots &&
       report.branch_edge_slot_policy == EdgeSlotPolicyKind::FixedCounter) ||
      report.fixed_edge_sites != 0;
  append_json_bool_field(out, "inline_edge_slots_active", inline_edge_slots_active);
  append_json_string_field(out, "inline_edge_slot_scope", "stable-site-hash");
  append_json_bool_field(out, "inline_edge_slot_reuse", true);
  append_json_bool_field(out, "inline_edge_slot_may_collide", inline_edge_slots_active);
  append_json_string_field(out, "inline_edge_slot_collision_policy",
                           "stable-site-hash-afl-bitmap-collisions-accepted");
  append_json_bool_field(out, "fixed_edge_slots_active", inline_edge_slots_active);
  append_json_string_field(out, "fixed_edge_slot_scope", "stable-site-hash");
  append_json_bool_field(out, "fixed_edge_slot_reuse", true);
  append_json_bool_field(out, "fixed_edge_slot_may_collide", inline_edge_slots_active);
  append_json_string_field(out, "fixed_edge_slot_collision_policy",
                           "stable-site-hash-afl-bitmap-collisions-accepted");
  append_json_u64_field(out, "probe_required_sgprs", report.probe_required_sgprs);
  append_json_u64_field(out, "probe_required_vgprs", report.probe_required_vgprs);
  append_json_u64_field(out, "probe_required_private_segment_bytes",
                        report.probe_required_private_segment_bytes);
  append_json_u64_field(out, "spill_bytes", report.spill_bytes);
  append_json_u64_field(out, "entry_liveness_register_kernels",
                        report.entry_liveness_register_kernels);
  append_json_u64_field(out, "entry_liveness_probe_points",
                        report.entry_liveness_probe_points);
  append_json_u64_field(out, "entry_backed_edge_kernels",
                        report.entry_backed_edge_kernels);
  append_json_u64_field(out, "self_contained_edge_kernels",
                        report.self_contained_edge_kernels);
  append_json_u64_field(out, "skipped_kernel_count", report.skipped_kernels.size());
  uint32_t liveness_register_kernels = 0;
  uint32_t liveness_probe_points = 0;
  uint32_t scratch_spill_probe_points = 0;
  uint32_t vgpr_scratch_spill_probe_points = 0;
  uint32_t sgpr_scratch_spill_probe_points = 0;
  for (const KernelEdgeSelectionSummary &summary : report.kernel_summaries) {
    scratch_spill_probe_points += summary.scratch_spill_probe_points;
    vgpr_scratch_spill_probe_points += summary.vgpr_scratch_spill_probe_points;
    sgpr_scratch_spill_probe_points += summary.sgpr_scratch_spill_probe_points;
    if (!summary.liveness_registers)
      continue;
    ++liveness_register_kernels;
    liveness_probe_points += summary.liveness_probe_points;
  }
  const bool uses_liveness_registers = liveness_register_kernels != 0;
  append_json_string_field(out, "register_plan",
                           uses_liveness_registers
                               ? "mixed-liveness-edge-registers"
                               : (report.entry_liveness_register_kernels != 0
                                      ? "mixed-liveness-entry-registers"
                                      : "fixed-high-registers"));
  append_json_u64_field(out, "liveness_register_kernels", liveness_register_kernels);
  append_json_u64_field(out, "liveness_probe_points", liveness_probe_points);
  append_json_u64_field(out, "scratch_spill_probe_points",
                        scratch_spill_probe_points);
  append_json_u64_field(out, "vgpr_scratch_spill_probe_points",
                        vgpr_scratch_spill_probe_points);
  append_json_u64_field(out, "sgpr_scratch_spill_probe_points",
                        sgpr_scratch_spill_probe_points);
  if (!report.cfg_failure_reason.empty())
    append_json_string_field(out, "cfg_failure_reason", report.cfg_failure_reason);

  append_json_field_name(out, "kernel_summaries");
  out.push_back('[');
  bool first = true;
  for (const KernelEdgeSelectionSummary &summary : report.kernel_summaries) {
    if (!first)
      out.push_back(',');
    first = false;
    out.push_back('{');
    append_json_string(out, "kernel");
    out.push_back(':');
    append_json_string(out, summary.kernel_name);
    append_json_string_field(out, "coverage_strategy", summary.coverage_strategy);
    append_json_string_field(out, "coverage_signal", coverage_signal_name(summary));
    append_json_string_field(out, "coverage_signal_reason",
                             coverage_signal_reason(summary));
    append_json_u64_field(out, "reachable_blocks", summary.reachable_blocks);
    append_json_u64_field(out, "block_candidates", summary.block_candidates);
    append_json_u64_field(out, "block_selected", summary.block_selected);
    append_json_u64_field(out, "inline_slots_reserved", summary.inline_slots_reserved);
    append_json_u64_field(out, "skipped_unsafe", summary.skipped_unsafe);
    append_json_u64_field(out, "skipped_liveness", summary.skipped_liveness);
    append_json_u64_field(out, "skipped_limit", summary.skipped_limit);
    append_json_u64_field(out, "skipped_fixed_slot", summary.skipped_fixed_slot);
    append_json_u64_field(out, "branch_candidates", summary.branch_candidates);
    append_json_u64_field(out, "branch_edge_candidate_edges",
                          summary.branch_edge_candidate_edges);
    append_json_u64_field(out, "previous_bb_branch_edge_candidate_edges",
                          summary.previous_bb_branch_edge_candidate_edges);
    append_json_u64_field(out, "previous_bb_branch_site_candidate_sites",
                          summary.previous_bb_branch_site_candidate_sites);
    append_json_u64_field(out, "branch_edge_budget", summary.branch_edge_budget);
    append_json_string_field(out, "branch_edge_budget_reason",
                             summary.branch_edge_budget_reason);
    append_json_u64_field(out, "previous_bb_branch_site_budget",
                          summary.previous_bb_branch_site_budget);
    append_json_string_field(out, "previous_bb_branch_site_budget_reason",
                             summary.previous_bb_branch_site_budget_reason);
    append_json_u64_field(out, "previous_bb_branch_edge_over_budget",
                          summary.previous_bb_branch_edge_over_budget);
    append_json_u64_field(out, "previous_bb_branch_site_over_budget",
                          summary.previous_bb_branch_site_over_budget);
    append_json_u64_field(out, "fixed_counter_branch_edge_fallback_budget",
                          summary.fixed_counter_branch_edge_fallback_budget);
    append_json_u64_field(
        out, "fixed_counter_branch_edge_aggregate_fallback_used",
        summary.fixed_counter_branch_edge_aggregate_fallback_used);
    append_json_u64_field(out, "fixed_counter_branch_edge_safety_fallback_used",
                          summary.fixed_counter_branch_edge_safety_fallback_used);
    append_json_u64_field(
        out, "fixed_counter_branch_edge_liveness_fallback_used",
        summary.fixed_counter_branch_edge_liveness_fallback_used);
    append_json_u64_field(
        out, "fixed_counter_branch_edge_placement_fallback_used",
        summary.fixed_counter_branch_edge_placement_fallback_used);
    append_json_u64_field(out, "exec_empty_fixed_counter_edges",
                          summary.exec_empty_fixed_counter_edges);
    append_json_u64_field(out, "fixed_counter_branch_edge_fallback_used",
                          summary.fixed_counter_branch_edge_fallback_used);
    append_json_string_field(
        out, "fixed_counter_branch_edge_fallback_budget_reason",
        summary.fixed_counter_branch_edge_fallback_budget_reason);
    append_json_string_field(out, "previous_bb_branch_aggregate_limit_kind",
                             summary.previous_bb_branch_aggregate_limit_kind);
    append_json_string_field(out, "previous_bb_branch_aggregate_safety",
                             summary.previous_bb_branch_aggregate_safety);
    append_json_string_field(out, "previous_bb_branch_aggregate_safety_reason",
                             summary.previous_bb_branch_aggregate_safety_reason);
    append_json_string_field(out, "branch_edge_slot_policy_reason",
                             summary.branch_edge_slot_policy_reason);
    append_json_u64_field(out, "branch_edges_selected", summary.branch_edges_selected);
    append_json_u64_field(out, "previous_bb_branch_edges_selected",
                          summary.previous_bb_branch_edges_selected);
    append_json_u64_field(out, "branch_edges_degraded_to_fixed",
                          summary.branch_edges_degraded_to_fixed);
    append_json_u64_field(out, "previous_bb_branch_sites_selected",
                          summary.previous_bb_branch_sites_selected);
    append_json_u64_field(out, "previous_bb_branch_sites_degraded_to_fixed",
                          summary.previous_bb_branch_sites_degraded_to_fixed);
    append_json_u64_field(out, "edge_trampolines_planned",
                          summary.edge_trampolines_planned);
    append_json_u64_field(out, "previous_bb_branch_edge_trampolines_planned",
                          summary.previous_bb_branch_edge_trampolines_planned);
    append_json_u64_field(out, "planned_appended_edge_trampolines",
                          summary.planned_appended_edge_trampolines);
    append_json_u64_field(out, "planned_local_edge_trampolines",
                          summary.planned_local_edge_trampolines);
    append_json_u64_field(
        out, "previous_bb_branch_planned_appended_edge_trampolines",
        summary.previous_bb_branch_planned_appended_edge_trampolines);
    append_json_u64_field(
        out, "previous_bb_branch_planned_local_edge_trampolines",
        summary.previous_bb_branch_planned_local_edge_trampolines);
    append_json_u64_field(out, "planned_edge_trampoline_bytes",
                          summary.planned_edge_trampoline_bytes);
    append_json_u64_field(out, "previous_bb_branch_edge_trampoline_bytes",
                          summary.previous_bb_branch_edge_trampoline_bytes);
    append_json_u64_field(out, "planned_appended_edge_trampoline_bytes",
                          summary.planned_appended_edge_trampoline_bytes);
    append_json_u64_field(out, "planned_local_edge_trampoline_bytes",
                          summary.planned_local_edge_trampoline_bytes);
    append_json_u64_field(
        out, "previous_bb_branch_planned_appended_edge_trampoline_bytes",
        summary.previous_bb_branch_planned_appended_edge_trampoline_bytes);
    append_json_u64_field(
        out, "previous_bb_branch_planned_local_edge_trampoline_bytes",
        summary.previous_bb_branch_planned_local_edge_trampoline_bytes);
    append_json_u64_field(out, "largest_edge_trampoline_bytes",
                          summary.largest_edge_trampoline_bytes);
    append_json_u64_field(out, "largest_previous_bb_branch_edge_trampoline_bytes",
                          summary.largest_previous_bb_branch_edge_trampoline_bytes);
    append_json_u64_field(out, "previous_bb_branch_afl_map_budget",
                          summary.previous_bb_branch_afl_map_budget);
    append_json_u64_field(out, "previous_bb_branch_afl_map_pressure_ppm",
                          summary.previous_bb_branch_afl_map_pressure_ppm);
    append_json_u64_field(out, "previous_bb_branch_trampoline_avg_bytes_x100",
                          summary.previous_bb_branch_trampoline_avg_bytes_x100);
    append_json_u64_field(out, "previous_bb_branch_appended_trampoline_ratio_ppm",
                          summary.previous_bb_branch_appended_trampoline_ratio_ppm);
    append_json_u64_field(out, "previous_bb_branch_local_trampoline_ratio_ppm",
                          summary.previous_bb_branch_local_trampoline_ratio_ppm);
    append_json_string_field(out, "previous_bb_branch_overhead_status",
                             summary.previous_bb_branch_overhead_status);
    append_json_string_field(out, "previous_bb_branch_overhead_reason",
                             summary.previous_bb_branch_overhead_reason);
    append_json_u64_field(out, "skipped_branch_unsafe", summary.skipped_branch_unsafe);
    append_json_u64_field(out, "skipped_branch_liveness", summary.skipped_branch_liveness);
    append_json_u64_field(out, "skipped_branch_limit", summary.skipped_branch_limit);
    append_json_u64_field(out, "opaque_instruction_count",
                          summary.opaque_instruction_count);
    append_json_u64_field(out, "unmodeled_opaque_instruction_count",
                          summary.unmodeled_opaque_instruction_count);
    append_json_u64_field(out, "liveness_probe_points", summary.liveness_probe_points);
    append_json_u64_field(out, "fresh_register_probe_points",
                          summary.fresh_register_probe_points);
    append_json_u64_field(out, "opaque_fresh_register_candidate_probe_points",
                          summary.opaque_fresh_register_candidate_probe_points);
    append_json_u64_field(
        out, "opaque_fresh_register_candidate_sgpr_growth_probe_points",
        summary.opaque_fresh_register_candidate_sgpr_growth_probe_points);
    append_json_u64_field(
        out, "opaque_fresh_register_candidate_vgpr_growth_probe_points",
        summary.opaque_fresh_register_candidate_vgpr_growth_probe_points);
    append_json_u64_field(out, "opaque_fresh_register_candidate_required_sgprs",
                          summary.opaque_fresh_register_candidate_required_sgprs);
    append_json_u64_field(out, "opaque_fresh_register_candidate_required_vgprs",
                          summary.opaque_fresh_register_candidate_required_vgprs);
    append_json_u64_field(out, "scratch_spill_probe_points",
                          summary.scratch_spill_probe_points);
    append_json_u64_field(out, "vgpr_scratch_spill_probe_points",
                          summary.vgpr_scratch_spill_probe_points);
    append_json_u64_field(out, "sgpr_scratch_spill_probe_points",
                          summary.sgpr_scratch_spill_probe_points);
    append_json_u64_field(
        out, "fresh_register_growth_disabled_by_opaque_probe_points",
        summary.fresh_register_growth_disabled_by_opaque_probe_points);
    append_json_u64_field(
        out, "sgpr_scratch_spill_disabled_by_opaque_probe_points",
        summary.sgpr_scratch_spill_disabled_by_opaque_probe_points);
    append_json_u64_field(
        out, "sgpr_scratch_spill_disabled_by_exec_condition_probe_points",
        summary.sgpr_scratch_spill_disabled_by_exec_condition_probe_points);
    append_json_u64_field(
        out, "direct_exec_fixed_scratch_disabled_by_opaque_probe_points",
        summary.direct_exec_fixed_scratch_disabled_by_opaque_probe_points);
    append_json_bool_field(out, "liveness_registers", summary.liveness_registers);
    append_json_bool_field(out, "fresh_registers", summary.fresh_registers);
    append_json_bool_field(out, "self_contained_probe", summary.self_contained_probe);
    append_json_u64_field(out, "hashed_edge_sites",
                          summary.slot_policy_summary.hashed_edge_sites);
    append_json_u64_field(out, "fixed_edge_sites",
                          summary.slot_policy_summary.fixed_edge_sites);
    append_json_u64_field(out, "fixed_slot_requests",
                          summary.slot_policy_summary.fixed_slot_requests);
    append_json_u64_field(out, "fixed_slots_reserved",
                          summary.slot_policy_summary.fixed_slots_reserved);
    append_json_u64_field(out, "fixed_slot_exhaustions",
                          summary.slot_policy_summary.fixed_slot_exhaustions);
    append_json_u64_field(out, "fixed_slot_collisions",
                          summary.slot_policy_summary.fixed_slot_collisions);
    append_json_u64_field(out, "inline_slot_requests",
                          summary.slot_policy_summary.inline_slot_requests);
    append_json_u64_field(out, "inline_slot_exhaustions",
                          summary.slot_policy_summary.inline_slot_exhaustions);
    append_json_field_name(out, "degradation_reason_counts");
    out.push_back('[');
    bool first_degradation_count = true;
    for (const afl_dbi::EdgeSiteSkipReasonCount &count : summary.degradation_reason_counts) {
      if (!first_degradation_count)
        out.push_back(',');
      first_degradation_count = false;
      out.push_back('{');
      append_json_string(out, "kind");
      out.push_back(':');
      append_json_string(out, count.kind);
      append_json_string_field(out, "reason", count.reason);
      append_json_u64_field(out, "count", count.count);
      out.push_back('}');
    }
    out.push_back(']');
    append_json_field_name(out, "skip_reason_counts");
    out.push_back('[');
    bool first_skip_count = true;
    for (const afl_dbi::EdgeSiteSkipReasonCount &count : summary.skip_reason_counts) {
      if (!first_skip_count)
        out.push_back(',');
      first_skip_count = false;
      out.push_back('{');
      append_json_string(out, "kind");
      out.push_back(':');
      append_json_string(out, count.kind);
      append_json_string_field(out, "reason", count.reason);
      append_json_u64_field(out, "count", count.count);
      out.push_back('}');
    }
    out.push_back(']');
    append_json_field_name(out, "sampled_skips");
    out.push_back('[');
    bool first_skip = true;
    for (const afl_dbi::EdgeSiteSkipSample &skip : summary.sampled_skips) {
      if (!first_skip)
        out.push_back(',');
      first_skip = false;
      out.push_back('{');
      append_json_string(out, "kind");
      out.push_back(':');
      append_json_string(out, skip.kind);
      append_json_u64_field(out, "text_offset", skip.text_offset);
      append_json_string_field(out, "reason", skip.reason);
      if (!skip.mnemonic.empty()) {
        append_json_string_field(out, "mnemonic", skip.mnemonic);
        append_json_u64_field(out, "instruction_size", skip.instruction_size);
        append_json_u64_field(out, "instruction_flags", skip.instruction_flags);
        append_json_field_name(out, "words");
        out.push_back('[');
        for (size_t i = 0; i < skip.words.size(); ++i) {
          if (i != 0)
            out.push_back(',');
          out += std::to_string(skip.words[i]);
        }
        out.push_back(']');
      }
      out.push_back('}');
    }
    out.push_back(']');
    append_json_field_name(out, "sampled_opaque_instructions");
    out.push_back('[');
    bool first_opaque = true;
    for (const afl_dbi::OpaqueInstructionSample &sample :
         summary.sampled_opaque_instructions) {
      if (!first_opaque)
        out.push_back(',');
      first_opaque = false;
      out.push_back('{');
      append_json_string(out, "mnemonic");
      out.push_back(':');
      append_json_string(out, sample.mnemonic);
      append_json_u64_field(out, "text_offset", sample.text_offset);
      append_json_bool_field(out, "liveness_modeled", sample.liveness_modeled);
      append_json_field_name(out, "words");
      out.push_back('[');
      for (size_t i = 0; i < sample.words.size(); ++i) {
        if (i != 0)
          out.push_back(',');
        out += std::to_string(sample.words[i]);
      }
      out.push_back(']');
      out.push_back('}');
    }
    out.push_back(']');
    append_json_field_name(out, "sampled_opaque_fresh_register_candidates");
    out.push_back('[');
    bool first_fresh_candidate = true;
    for (const afl_dbi::OpaqueFreshRegisterCandidateSample &sample :
         summary.sampled_opaque_fresh_register_candidates) {
      if (!first_fresh_candidate)
        out.push_back(',');
      first_fresh_candidate = false;
      out.push_back('{');
      append_json_string(out, "kind");
      out.push_back(':');
      append_json_string(out, sample.kind);
      append_json_u64_field(out, "patch_text_offset", sample.patch_text_offset);
      append_json_string_field(out, "mnemonic", sample.mnemonic);
      append_json_u64_field(out, "required_sgprs", sample.required_sgprs);
      append_json_u64_field(out, "required_vgprs", sample.required_vgprs);
      append_json_u64_field(out, "allocated_sgprs", sample.allocated_sgprs);
      append_json_u64_field(out, "allocated_vgprs", sample.allocated_vgprs);
      append_json_bool_field(out, "sgpr_growth", sample.sgpr_growth);
      append_json_bool_field(out, "vgpr_growth", sample.vgpr_growth);
      append_json_bool_field(out, "previous_bb_probe_registers",
                             sample.previous_bb_probe_registers);
      append_json_bool_field(out, "stable_state_sgpr", sample.stable_state_sgpr);
      append_json_string_field(out, "slot_policy", sample.slot_policy);
      append_json_u64_field(out, "state_sgpr", sample.state_sgpr);
      append_json_u64_field(out, "saved_exec_sgpr", sample.saved_exec_sgpr);
      append_json_u64_field(out, "tmp0_sgpr", sample.tmp0_sgpr);
      append_json_u64_field(out, "tmp1_sgpr", sample.tmp1_sgpr);
      append_json_u64_field(out, "scc_sgpr", sample.scc_sgpr);
      append_json_u64_field(out, "workitem_vgpr", sample.workitem_vgpr);
      append_json_u64_field(out, "tmp0_vgpr", sample.tmp0_vgpr);
      append_json_u64_field(out, "tmp1_vgpr", sample.tmp1_vgpr);
      append_json_u64_field(out, "tmp2_vgpr", sample.tmp2_vgpr);
      append_json_field_name(out, "words");
      out.push_back('[');
      for (size_t i = 0; i < sample.words.size(); ++i) {
        if (i != 0)
          out.push_back(',');
        out += std::to_string(sample.words[i]);
      }
      out.push_back(']');
      out.push_back('}');
    }
    out.push_back(']');
    out.push_back('}');
  }
  out.push_back(']');

  append_json_field_name(out, "skipped_kernels");
  out.push_back('[');
  first = true;
  for (const KernelPatchabilitySkipSummary &summary : report.skipped_kernels) {
    if (!first)
      out.push_back(',');
    first = false;
    out.push_back('{');
    append_json_string(out, "kernel");
    out.push_back(':');
    append_json_string(out, summary.kernel_name);
    append_json_string_field(out, "reason", summary.reason);
    append_json_bool_field(out, "entry_probe_safe", summary.entry_probe_safe);
    append_json_bool_field(out, "self_contained_probe_safe",
                           summary.self_contained_probe_safe);
    append_json_bool_field(out, "branch_probe_safe", summary.branch_probe_safe);
    append_json_bool_field(out, "prefers_self_contained_edge_probes",
                           summary.prefers_self_contained_edge_probes);
    append_json_bool_field(out, "prefers_fixed_branch_counters",
                           summary.prefers_fixed_branch_counters);
    out.push_back('}');
  }
  out.push_back(']');

  append_json_field_name(out, "descriptor_resources");
  out.push_back('[');
  first = true;
  for (const KernelDescriptorResourceSummary &summary : report.descriptor_resources) {
    if (!first)
      out.push_back(',');
    first = false;
    out.push_back('{');
    append_json_string(out, "kernel");
    out.push_back(':');
    append_json_string(out, summary.kernel_name);
    append_json_u64_field(out, "descriptor_file_offset", summary.descriptor_file_offset);
    append_json_bool_field(out, "wave32", summary.wave32);
    append_json_bool_field(out, "has_metadata_sgpr_count",
                           summary.has_metadata_sgpr_count);
    append_json_bool_field(out, "descriptor_sgpr_count_effective",
                           summary.descriptor_sgpr_count_effective);
    append_json_bool_field(out, "fresh_sgpr_growth_supported",
                           summary.fresh_sgpr_growth_supported);
    append_json_bool_field(out, "old_private_segment_enabled",
                           summary.old_private_segment_enabled);
    append_json_bool_field(out, "patched_private_segment_enabled",
                           summary.patched_private_segment_enabled);
    append_json_u64_field(out, "vgpr_granularity", summary.vgpr_granularity);
    append_json_u64_field(out, "sgpr_granularity", summary.sgpr_granularity);
    append_json_u64_field(out, "old_vgpr_granulated", summary.old_vgpr_granulated);
    append_json_u64_field(out, "old_sgpr_granulated", summary.old_sgpr_granulated);
    append_json_u64_field(out, "patched_vgpr_granulated",
                          summary.patched_vgpr_granulated);
    append_json_u64_field(out, "patched_sgpr_granulated",
                          summary.patched_sgpr_granulated);
    append_json_u64_field(out, "descriptor_sgpr_count",
                          summary.descriptor_sgpr_count);
    append_json_u64_field(out, "metadata_sgpr_count",
                          summary.metadata_sgpr_count);
    append_json_u64_field(out, "old_vgpr_count", summary.old_vgpr_count);
    append_json_u64_field(out, "old_sgpr_count", summary.old_sgpr_count);
    append_json_u64_field(out, "patched_vgpr_count", summary.patched_vgpr_count);
    append_json_u64_field(out, "patched_sgpr_count", summary.patched_sgpr_count);
    append_json_u64_field(out, "old_private_segment_fixed_size",
                          summary.old_private_segment_fixed_size);
    append_json_u64_field(out, "patched_private_segment_fixed_size",
                          summary.patched_private_segment_fixed_size);
    append_json_u64_field(out, "spill_bytes", summary.spill_bytes);
    append_json_string_field(out, "sgpr_count_metadata_patch",
                             summary.sgpr_count_metadata_patch);
    append_json_string_field(out, "private_segment_metadata_patch",
                             summary.private_segment_metadata_patch);
    append_json_bool_field(out, "resource_fields_changed",
                           summary.resource_fields_changed);
    out.push_back('}');
  }
  out.push_back(']');

  append_json_field_name(out, "sampled_failures");
  out.push_back('[');
  first = true;
  for (const EdgePatchFailure &failure : report.sampled_failures) {
    if (!first)
      out.push_back(',');
    first = false;
    out.push_back('{');
    append_json_string(out, "kernel");
    out.push_back(':');
    append_json_string(out, failure.kernel_name);
    append_json_string_field(out, "kind", failure.kind);
    append_json_u64_field(out, "patch_text_offset", failure.patch_text_offset);
    append_json_u64_field(out, "return_text_offset", failure.return_text_offset);
    append_json_string_field(out, "reason", failure.reason);
    out.push_back('}');
  }
  out.push_back(']');

  append_json_field_name(out, "sampled_selected_edges");
  out.push_back('[');
  first = true;
  for (const EdgeSiteSelectionSample &sample : report.sampled_selected_edges) {
    if (!first)
      out.push_back(',');
    first = false;
    out.push_back('{');
    append_json_string(out, "kernel");
    out.push_back(':');
    append_json_string(out, sample.kernel_name);
    append_json_string_field(out, "kind", sample.kind);
    append_json_u64_field(out, "pred_text_offset", sample.pred_text_offset);
    append_json_u64_field(out, "block_text_offset", sample.block_text_offset);
    append_json_u64_field(out, "patch_text_offset", sample.patch_text_offset);
    append_json_u64_field(out, "return_text_offset", sample.return_text_offset);
    append_json_u64_field(out, "cave_text_offset", sample.cave_text_offset);
    append_json_u64_field(out, "trampoline_bytes", sample.trampoline_bytes);
    append_json_u64_field(out, "bb_id", sample.bb_id);
    append_json_u64_field(out, "fallthrough_bb_id", sample.fallthrough_bb_id);
    append_json_string_field(out, "slot_policy", sample.slot_policy);
    append_json_u64_field(out, "fixed_slot", sample.fixed_slot);
    append_json_u64_field(out, "fallthrough_slot", sample.fallthrough_slot);
    append_json_u64_field(out, "fixed_slot_collisions", sample.fixed_slot_collisions);
    append_json_bool_field(out, "self_contained_probe", sample.self_contained_probe);
    append_json_bool_field(out, "force_lane0_exec_for_fixed_counter",
                           sample.force_lane0_exec_for_fixed_counter);
    append_json_bool_field(out, "scratch_spill", sample.scratch_spill);
    append_json_bool_field(out, "vgpr_scratch_spill", sample.vgpr_scratch_spill);
    append_json_bool_field(out, "sgpr_scratch_spill", sample.sgpr_scratch_spill);
    append_json_u64_field(out, "state_sgpr", sample.state_sgpr);
    append_json_u64_field(out, "saved_exec_sgpr", sample.saved_exec_sgpr);
    append_json_u64_field(out, "tmp0_sgpr", sample.tmp0_sgpr);
    append_json_u64_field(out, "tmp1_sgpr", sample.tmp1_sgpr);
    append_json_u64_field(out, "scc_sgpr", sample.scc_sgpr);
    append_json_u64_field(out, "workitem_vgpr", sample.workitem_vgpr);
    append_json_u64_field(out, "tmp0_vgpr", sample.tmp0_vgpr);
    append_json_u64_field(out, "tmp1_vgpr", sample.tmp1_vgpr);
    append_json_u64_field(out, "tmp2_vgpr", sample.tmp2_vgpr);
    append_json_u64_field(out, "scratch_address_vgpr", sample.scratch_address_vgpr);
    append_json_u8_array_field(out, "scratch_spilled_vgprs",
                               sample.scratch_spilled_vgprs);
    append_json_u8_array_field(out, "scratch_spilled_sgprs",
                               sample.scratch_spilled_sgprs);
    append_json_string_field(out, "scratch_address_exec_source",
                             sample.scratch_address_exec_source);
    append_json_string_field(out, "placement", sample.placement);
    out.push_back('}');
  }
  out.push_back(']');
  out += "}\n";
  append_patch_report_line(out);
}

std::optional<std::string> write_temp_elf(std::span<const uint8_t> elf) {
  std::string path = "/tmp/rocjitsu-afl-elf-XXXXXX";
  std::vector<char> writable(path.begin(), path.end());
  writable.push_back('\0');
  int fd = mkstemp(writable.data());
  if (fd < 0)
    return std::nullopt;
  size_t written = 0;
  while (written < elf.size()) {
    const ssize_t rc = write(fd, elf.data() + written, static_cast<size_t>(elf.size() - written));
    if (rc <= 0) {
      close(fd);
      unlink(writable.data());
      return std::nullopt;
    }
    written += static_cast<size_t>(rc);
  }
  close(fd);
  return std::string(writable.data());
}

const afl_dbi::ProbeTarget *detect_probe_target(std::span<const uint8_t> elf) {
  auto ehdr = read_struct<rocjitsu::Elf64_Ehdr>(elf, 0);
  if (!ehdr)
    return nullptr;
  const uint32_t mach = ehdr->e_flags & rocjitsu::EF_AMDGPU_MACH;
  return afl_dbi::probe_target_for_elf_mach(mach);
}

KernelPatchabilityFilters current_patchability_filters() {
  const char *include_filter =
      g_kernel_include_override != nullptr ? g_kernel_include_override : g_kernel_include;
  return KernelPatchabilityFilters{include_filter, g_kernel_exclude};
}

KernelPatchability classify_current_kernel(std::string_view name) {
  return classify_kernel_patchability(name, current_patchability_filters());
}

bool should_instrument_kernel(std::string_view name) {
  return classify_current_kernel(name).instrumentable;
}

bool entry_probe_is_unsafe(const KernelSite &site) {
  return !classify_current_kernel(site.name).entry_probe_safe;
}

std::optional<std::string_view> fixed_branch_counter_reason(const KernelSite &site) {
  const KernelPatchability patchability = classify_current_kernel(site.name);
  if (patchability.prefers_fixed_branch_counters) {
    return patchability.fixed_branch_strategy_reason != nullptr
               ? std::string_view(patchability.fixed_branch_strategy_reason)
               : std::string_view("kernel-fixed-branch-preferred");
  }
  return std::nullopt;
}

bool kernel_site_prefers_fixed_branch_counters(const KernelSite &site) {
  return fixed_branch_counter_reason(site).has_value();
}

const char *coverage_strategy_name(const InstrumentationPlanOptions &options,
                                   bool edge_instrumentation_enabled) {
  if (!edge_instrumentation_enabled)
    return "entry-counter-only";
  if (!options.branch_edge_slots)
    return "entry-previous-bb-block";
  if (options.self_contained_edge_probes) {
    return options.branch_terminator_slot_policy == EdgeSlotPolicyKind::PreviousBbHash
               ? "self-contained-previous-bb-branch"
               : "self-contained-fixed-branch";
  }
  if (options.block_entry_site_limit != 0) {
    return options.branch_terminator_slot_policy == EdgeSlotPolicyKind::PreviousBbHash
               ? "entry-previous-bb-block-and-previous-bb-branch"
               : "entry-previous-bb-block-and-fixed-branch";
  }
  return "branch-edge-diagnostic";
}

bool diagnostic_coverage_override_enabled() {
  // The product path is the adaptive planner. These DEBUG knobs intentionally
  // disable parts of that planner for regression tests and ablation runs.
  return g_skip_entry_probe || g_branch_edge_slots || g_fixed_edge_slots ||
         g_diagnostic_branch_edge_slot_policy_override || g_require_liveness_registers ||
         g_allow_opaque_fresh_registers ||
         !g_debug_edge_patch_text_offsets.empty();
}

std::vector<KernelSite> collect_edge_planning_kernels(std::span<const KernelSite> sites,
                                                      bool self_contained_edge_probes) {
  std::vector<KernelSite> edge_kernels;
  for (const KernelSite &site : sites) {
    if (!should_instrument_kernel(site.name))
      continue;
    if (!self_contained_edge_probes && entry_probe_is_unsafe(site))
      continue;
    edge_kernels.push_back(site);
  }
  return edge_kernels;
}

std::unordered_set<std::string>
entry_liveness_kernel_names(std::span<const EntryProbeRegisterSelection> selections) {
  std::unordered_set<std::string> selected;
  selected.reserve(selections.size());
  for (const EntryProbeRegisterSelection &selection : selections)
    selected.insert(selection.kernel_name);
  return selected;
}

std::vector<KernelSite>
collect_kernels_by_name(std::span<const KernelSite> sites,
                        const std::unordered_set<std::string> &names) {
  std::vector<KernelSite> selected;
  for (const KernelSite &site : sites) {
    if (names.find(site.name) != names.end())
      selected.push_back(site);
  }
  return selected;
}

std::unordered_set<std::string>
fixed_branch_counter_kernel_names(std::span<const KernelSite> sites) {
  std::unordered_set<std::string> selected;
  for (const KernelSite &site : sites) {
    if (kernel_site_prefers_fixed_branch_counters(site))
      selected.insert(site.name);
  }
  return selected;
}

std::unordered_map<std::string, std::string>
fixed_branch_counter_kernel_reasons(std::span<const KernelSite> sites) {
  std::unordered_map<std::string, std::string> reasons;
  for (const KernelSite &site : sites) {
    std::optional<std::string_view> reason = fixed_branch_counter_reason(site);
    if (reason)
      reasons.emplace(site.name, std::string(*reason));
  }
  return reasons;
}

void annotate_branch_slot_policy_reasons(
    InstrumentationPlan &plan,
    const std::unordered_map<std::string, std::string> &reasons) {
  for (KernelEdgeSelectionSummary &summary : plan.kernel_summaries) {
    if (!summary.branch_edge_slot_policy_reason.empty())
      continue;
    auto reason = reasons.find(summary.kernel_name);
    if (reason != reasons.end())
      summary.branch_edge_slot_policy_reason = reason->second;
  }
}

bool all_kernels_prefer_fixed_branch_counters(std::span<const KernelSite> sites) {
  bool saw_kernel = false;
  for (const KernelSite &site : sites) {
    saw_kernel = true;
    if (!kernel_site_prefers_fixed_branch_counters(site))
      return false;
  }
  return saw_kernel;
}

void append_unique_kernel_site(std::vector<KernelSite> &sites,
                               std::unordered_set<std::string> &seen,
                               const KernelSite &site) {
  if (seen.insert(site.name).second)
    sites.push_back(site);
}

void merge_instrumentation_plan(InstrumentationPlan &dst, InstrumentationPlan &&src) {
  dst.sites.insert(dst.sites.end(), std::make_move_iterator(src.sites.begin()),
                   std::make_move_iterator(src.sites.end()));
  dst.kernel_summaries.insert(dst.kernel_summaries.end(),
                              std::make_move_iterator(src.kernel_summaries.begin()),
                              std::make_move_iterator(src.kernel_summaries.end()));
  dst.slot_policy_summary.hashed_edge_sites +=
      src.slot_policy_summary.hashed_edge_sites;
  dst.slot_policy_summary.fixed_edge_sites +=
      src.slot_policy_summary.fixed_edge_sites;
  dst.slot_policy_summary.fixed_slot_requests +=
      src.slot_policy_summary.fixed_slot_requests;
  dst.slot_policy_summary.fixed_slots_reserved +=
      src.slot_policy_summary.fixed_slots_reserved;
  dst.slot_policy_summary.fixed_slot_exhaustions +=
      src.slot_policy_summary.fixed_slot_exhaustions;
  dst.slot_policy_summary.fixed_slot_collisions +=
      src.slot_policy_summary.fixed_slot_collisions;
  dst.slot_policy_summary.inline_slot_requests +=
      src.slot_policy_summary.inline_slot_requests;
  dst.slot_policy_summary.inline_slots_reserved +=
      src.slot_policy_summary.inline_slots_reserved;
  dst.slot_policy_summary.inline_slot_exhaustions +=
      src.slot_policy_summary.inline_slot_exhaustions;
  if (dst.failure_reason.empty())
    dst.failure_reason = std::move(src.failure_reason);
}

uint32_t selected_edge_count(const KernelEdgeSelectionSummary &summary) {
  return summary.block_selected + summary.branch_edges_selected;
}

void add_slot_policy_summary(EdgeSlotPolicySummary &dst,
                             const EdgeSlotPolicySummary &src) {
  dst.hashed_edge_sites += src.hashed_edge_sites;
  dst.fixed_edge_sites += src.fixed_edge_sites;
  dst.fixed_slot_requests += src.fixed_slot_requests;
  dst.fixed_slots_reserved += src.fixed_slots_reserved;
  dst.fixed_slot_exhaustions += src.fixed_slot_exhaustions;
  dst.fixed_slot_collisions += src.fixed_slot_collisions;
  dst.inline_slot_requests += src.inline_slot_requests;
  dst.inline_slots_reserved += src.inline_slots_reserved;
  dst.inline_slot_exhaustions += src.inline_slot_exhaustions;
}

InstrumentationPlan drop_kernels_from_plan(InstrumentationPlan &&plan,
                                           const std::unordered_set<std::string> &drop) {
  if (drop.empty())
    return std::move(plan);

  InstrumentationPlan kept;
  kept.failure_reason = std::move(plan.failure_reason);
  for (EdgeSite &site : plan.sites) {
    if (drop.find(site.kernel_name) == drop.end())
      kept.sites.push_back(std::move(site));
  }
  for (KernelEdgeSelectionSummary &summary : plan.kernel_summaries) {
    if (drop.find(summary.kernel_name) == drop.end()) {
      add_slot_policy_summary(kept.slot_policy_summary, summary.slot_policy_summary);
      kept.kernel_summaries.push_back(std::move(summary));
    }
  }
  return kept;
}

bool debug_edge_patch_text_offset_filter_enabled() {
  return !g_debug_edge_patch_text_offsets.empty();
}

bool debug_edge_patch_text_offset_matches(uint32_t patch_text_offset) {
  return std::find(g_debug_edge_patch_text_offsets.begin(),
                   g_debug_edge_patch_text_offsets.end(),
                   patch_text_offset) != g_debug_edge_patch_text_offsets.end();
}

void add_site_slot_policy_summary(EdgeSlotPolicySummary &summary,
                                  const EdgeSite &site) {
  const uint32_t edge_count = edge_count_for_site(site);
  if (site.slot_policy == EdgeSlotPolicyKind::PreviousBbHash) {
    summary.hashed_edge_sites += edge_count;
    return;
  }
  summary.fixed_edge_sites += edge_count;
  summary.fixed_slot_requests += edge_count;
  summary.fixed_slots_reserved += edge_count;
  summary.fixed_slot_collisions += site.fixed_slot_collisions;
}

InstrumentationPlan apply_debug_edge_patch_text_offset_filter(InstrumentationPlan &&plan) {
  if (!debug_edge_patch_text_offset_filter_enabled())
    return std::move(plan);

  struct KernelStats {
    uint32_t block_selected = 0;
    uint32_t branch_edges_selected = 0;
    uint32_t previous_bb_branch_edges_selected = 0;
    uint32_t previous_bb_branch_sites_selected = 0;
    EdgeSlotPolicySummary slot_policy_summary;
  };

  std::unordered_map<std::string, KernelStats> stats_by_kernel;
  InstrumentationPlan kept;
  kept.failure_reason = std::move(plan.failure_reason);
  for (EdgeSite &site : plan.sites) {
    if (!debug_edge_patch_text_offset_matches(site.patch_text_offset))
      continue;
    KernelStats &stats = stats_by_kernel[site.kernel_name];
    if (edge_patch_kind_is_block_entry(site.kind))
      ++stats.block_selected;
    else {
      stats.branch_edges_selected += edge_count_for_site(site);
      if (previous_bb_branch_site(site)) {
        stats.previous_bb_branch_edges_selected += edge_count_for_site(site);
        ++stats.previous_bb_branch_sites_selected;
      }
    }
    add_site_slot_policy_summary(stats.slot_policy_summary, site);
    kept.sites.push_back(std::move(site));
  }

  for (KernelEdgeSelectionSummary &summary : plan.kernel_summaries) {
    auto it = stats_by_kernel.find(summary.kernel_name);
    if (it == stats_by_kernel.end())
      continue;
    KernelEdgeSelectionSummary filtered = std::move(summary);
    filtered.block_selected = it->second.block_selected;
    filtered.branch_edges_selected = it->second.branch_edges_selected;
    filtered.previous_bb_branch_edges_selected =
        it->second.previous_bb_branch_edges_selected;
    filtered.previous_bb_branch_sites_selected =
        it->second.previous_bb_branch_sites_selected;
    filtered.slot_policy_summary = it->second.slot_policy_summary;
    filtered.inline_slots_reserved =
        it->second.slot_policy_summary.inline_slots_reserved;
    add_slot_policy_summary(kept.slot_policy_summary,
                            it->second.slot_policy_summary);
    kept.kernel_summaries.push_back(std::move(filtered));
  }
  return kept;
}

std::vector<KernelSite>
collect_kernels_by_summary_names(std::span<const KernelSite> kernels,
                                 const std::unordered_set<std::string> &names) {
  std::vector<KernelSite> selected;
  for (const KernelSite &kernel : kernels) {
    if (names.find(kernel.name) != names.end())
      selected.push_back(kernel);
  }
  return selected;
}

void merge_probe_requirements(ProbeRegisterRequirements &lhs,
                              const ProbeRegisterRequirements &rhs) {
  lhs.sgprs = std::max(lhs.sgprs, rhs.sgprs);
  lhs.vgprs = std::max(lhs.vgprs, rhs.vgprs);
  lhs.private_segment_bytes =
      std::max(lhs.private_segment_bytes, rhs.private_segment_bytes);
}

void record_probe_requirements(PatchDeviceElfReport &report,
                               const ProbeRegisterRequirements &requirements) {
  report.probe_required_sgprs = std::max(report.probe_required_sgprs, requirements.sgprs);
  report.probe_required_vgprs = std::max(report.probe_required_vgprs, requirements.vgprs);
  report.probe_required_private_segment_bytes =
      std::max(report.probe_required_private_segment_bytes,
               requirements.private_segment_bytes);
}

std::unordered_map<std::string, Rdna4ProbeRegisters>
entry_register_map(std::span<const EntryProbeRegisterSelection> selections) {
  std::unordered_map<std::string, Rdna4ProbeRegisters> by_kernel;
  by_kernel.reserve(selections.size());
  for (const EntryProbeRegisterSelection &selection : selections)
    by_kernel.emplace(selection.kernel_name, selection.probe_registers);
  return by_kernel;
}

void merge_kernel_probe_requirements(
    std::unordered_map<std::string, ProbeRegisterRequirements> &by_kernel,
    std::string_view kernel_name, const ProbeRegisterRequirements &requirements) {
  ProbeRegisterRequirements &merged = by_kernel[std::string(kernel_name)];
  merge_probe_requirements(merged, requirements);
}

void apply_debug_forced_private_segment_bytes(
    std::string_view context,
    std::unordered_map<std::string, ProbeRegisterRequirements> &by_kernel) {
  if (g_debug_forced_runtime_shadow_private_segment_bytes == 0)
    return;
  if (context != "runtime KPACK shadow")
    return;
  // DEBUG-only loader validation hook. Product coverage planning should grow
  // private segment metadata only from actual probe scratch requirements.
  for (auto &entry : by_kernel) {
    ProbeRegisterRequirements &requirements = entry.second;
    requirements.private_segment_bytes =
        std::max(requirements.private_segment_bytes,
                 g_debug_forced_runtime_shadow_private_segment_bytes);
  }
}

std::vector<uint8_t> patch_device_elf(std::span<const uint8_t> elf, uint64_t state_pointer,
                                      const char *context, const char *device_image_id,
                                      uint32_t device_image_index = kUnknownDeviceImageIndex) {
  PatchDeviceElfReport report;
  report.context = context != nullptr ? context : "<unknown>";
  report.device_image_id = device_image_id != nullptr ? device_image_id : "";
  report.device_image_index = device_image_index;
  const char *active_filter =
      g_kernel_include_override != nullptr ? g_kernel_include_override : g_kernel_include;
  const auto loader_patchability =
      classify_loader_patchability(context != nullptr ? std::string_view(context)
                                                      : std::string_view());
  const bool context_self_contained_edge_probes =
      loader_patchability.prefers_self_contained_edge_probes;
  const bool context_fixed_branch_counters =
      loader_patchability.prefers_fixed_branch_counters;
  const bool consider_vgpr_scratch_spills = !g_disable_vgpr_scratch_spills;
  const bool allow_vgpr_scratch_spills =
      consider_vgpr_scratch_spills && loader_patchability.allows_vgpr_scratch_spills;
  bool self_contained_edge_probes = g_skip_entry_probe || context_self_contained_edge_probes;
  std::string coverage_strategy_reason =
      g_skip_entry_probe
          ? "forced-skip-entry-env"
          : (context_self_contained_edge_probes ? loader_patchability.self_contained_strategy_reason
                                                : "default-entry-preferred");
  report.vgpr_scratch_spills_requested = consider_vgpr_scratch_spills;
  report.vgpr_scratch_spills_enabled = allow_vgpr_scratch_spills;
  report.vgpr_scratch_spill_reason =
      g_disable_vgpr_scratch_spills ? "disabled-by-debug-env"
                                     : loader_patchability.vgpr_scratch_spill_reason;
  report.kernel_filter = active_filter != nullptr ? active_filter : "";
  report.runtime_shadow_registration = g_runtime_shadow_registration_override;
  report.input_bytes = elf.size();
  InstrumentationPlanOptions plan_options;
  auto make_plan_options = [&](bool use_self_contained_edge_probes) {
    InstrumentationPlanOptions options;
    const bool diagnostic_coverage_override =
        diagnostic_coverage_override_enabled();
    const bool branch_edge_slots = g_branch_edge_slots || use_self_contained_edge_probes;
    auto default_branch_edge_slot_policy = [&] {
      if (!use_self_contained_edge_probes ||
          g_diagnostic_branch_edge_slot_policy_override)
        return g_branch_edge_slot_policy;
      if (context_fixed_branch_counters)
        return EdgeSlotPolicyKind::FixedCounter;
      return EdgeSlotPolicyKind::PreviousBbHash;
    };
    const EdgeSlotPolicyKind branch_edge_slot_policy = default_branch_edge_slot_policy();
    const bool default_loader_fixed_branch_counters =
        use_self_contained_edge_probes && context_fixed_branch_counters &&
        !g_diagnostic_branch_edge_slot_policy_override;
    const bool default_safe_edge_liveness =
        report.edge_instrumentation_enabled && !diagnostic_coverage_override &&
        !default_loader_fixed_branch_counters;
    const bool require_liveness_registers =
        g_require_liveness_registers || default_safe_edge_liveness ||
        (branch_edge_slots && branch_edge_slot_policy == EdgeSlotPolicyKind::PreviousBbHash &&
         use_self_contained_edge_probes);
    const bool auto_branch_edge_budget =
        use_self_contained_edge_probes &&
        branch_edge_slot_policy == EdgeSlotPolicyKind::PreviousBbHash &&
        !g_branch_edge_site_limit_overridden;
    const uint32_t branch_edge_site_limit =
        auto_branch_edge_budget ? afl_dbi::kAdaptiveBranchEdgeSiteLimit
                                : g_branch_edge_site_limit;
    const uint32_t previous_bb_branch_site_limit =
        auto_branch_edge_budget ? afl_dbi::kAdaptivePreviousBbBranchSiteLimit
                                : std::numeric_limits<uint32_t>::max();
    options.block_entry_site_limit = use_self_contained_edge_probes ? 0 : g_edge_site_limit;
    options.branch_edge_site_limit = branch_edge_site_limit;
    options.branch_edge_site_limit_auto = auto_branch_edge_budget;
    options.previous_bb_branch_site_limit = previous_bb_branch_site_limit;
    options.previous_bb_branch_site_limit_auto = auto_branch_edge_budget;
    options.block_entry_slot_policy = g_fixed_edge_slots
                                               ? EdgeSlotPolicyKind::FixedCounter
                                               : EdgeSlotPolicyKind::PreviousBbHash;
    options.branch_terminator_slot_policy = branch_edge_slot_policy;
    options.fixed_edge_slots = g_fixed_edge_slots;
    options.branch_edge_slots = branch_edge_slots;
    options.require_liveness_registers = require_liveness_registers;
    options.fixed_counter_fallback_for_branch_liveness =
        branch_edge_slots &&
        branch_edge_slot_policy == EdgeSlotPolicyKind::PreviousBbHash &&
        !diagnostic_coverage_override;
    options.fixed_counter_fallback_for_branch_budget =
        auto_branch_edge_budget && !diagnostic_coverage_override;
    options.fixed_counter_branch_edge_fallback_limit =
        options.fixed_counter_fallback_for_branch_budget
            ? afl_dbi::kAdaptiveFixedCounterBranchEdgeFallbackLimit
            : 0;
    options.allow_vgpr_scratch_spills = allow_vgpr_scratch_spills;
    options.force_fresh_sgprs = g_force_fresh_sgprs;
    options.force_fresh_vgprs = g_force_fresh_vgprs;
    options.allow_opaque_fresh_registers = g_allow_opaque_fresh_registers;
    options.self_contained_edge_probes = use_self_contained_edge_probes;
    options.verbose = g_verbose;
    return options;
  };
  auto configure_plan = [&] {
    plan_options = make_plan_options(self_contained_edge_probes);
    report.edge_site_limit = plan_options.block_entry_site_limit;
    report.branch_edge_site_limit = plan_options.branch_edge_site_limit;
    report.branch_edge_site_limit_auto = plan_options.branch_edge_site_limit_auto;
    report.previous_bb_branch_site_limit =
        plan_options.previous_bb_branch_site_limit;
    report.previous_bb_branch_site_limit_auto =
        plan_options.previous_bb_branch_site_limit_auto;
    report.skip_entry_probe = self_contained_edge_probes;
    report.fixed_edge_slots = plan_options.fixed_edge_slots;
    report.branch_edge_slots = plan_options.branch_edge_slots;
    report.branch_edge_slot_policy = plan_options.branch_terminator_slot_policy;
    report.require_liveness_registers = plan_options.require_liveness_registers;
    report.allow_opaque_fresh_registers = plan_options.allow_opaque_fresh_registers;
    report.coverage_strategy =
        coverage_strategy_name(plan_options, report.edge_instrumentation_enabled);
    report.coverage_strategy_reason = coverage_strategy_reason;
  };
  configure_plan();
  auto finish_empty = [&](std::string_view reason,
                          std::string_view failure_phase) -> std::vector<uint8_t> {
    report.reason = reason;
    report.failure_phase = failure_phase;
    emit_patch_report(report);
    return {};
  };

  if (g_launch_only) {
    if (g_verbose)
      fprintf(stderr, "rocjitsu-afl: launch-only mode, leaving device ELF unpatched\n");
    return finish_empty("launch_only", "preflight");
  }

  const afl_dbi::ProbeTarget *target = detect_probe_target(elf);
  if (target == nullptr || !afl_dbi::probe_target_supports_entry_counter(*target)) {
    if (g_verbose)
      fprintf(stderr,
              "rocjitsu-afl: unsupported code-object arch for probe patching "
              "(detected=%s)\n",
              target == nullptr ? "unknown" : arch_name(target->arch));
    if (target != nullptr) {
      report.gfxip = target->gfxip;
      report.arch = arch_name(target->arch);
    }
    report.edge_instrumentation_reason = "unsupported_arch";
    return finish_empty("unsupported_arch", "preflight");
  }
  const rj_code_arch_t arch = target->arch;
  report.gfxip = target->gfxip;
  report.arch = arch_name(arch);
  const bool supports_previous_bb_edges = afl_dbi::probe_target_supports_previous_bb_edges(*target);
  report.supports_previous_bb_edges = supports_previous_bb_edges;
  report.edge_instrumentation_enabled = !g_disable_edges && supports_previous_bb_edges;
  configure_plan();
  if (g_disable_edges) {
    report.edge_instrumentation_reason = "disabled_by_env";
  } else {
    report.edge_instrumentation_reason = afl_dbi::probe_target_edge_support_reason(*target);
  }
  if (g_verbose && !supports_previous_bb_edges) {
    fprintf(stderr,
            "rocjitsu-afl: %s uses entry-counter-only instrumentation for now "
            "(gfxip=%s reason=%s)\n",
            arch_name(arch), target->gfxip, report.edge_instrumentation_reason.c_str());
  }

  const auto sites = find_kernel_sites(elf);
  report.kernel_descriptor_count = static_cast<uint32_t>(
      std::min<size_t>(sites.size(), std::numeric_limits<uint32_t>::max()));
  if (sites.empty()) {
    if (g_verbose)
      fprintf(stderr, "rocjitsu-afl: no kernel descriptors found\n");
    return finish_empty("no_kernel_descriptors", "preflight");
  }
  std::unordered_set<std::string> reported_skipped_kernel_names;
  auto record_skipped_kernel = [&](const KernelSite &site,
                                   const KernelPatchability &patchability) {
    if (patchability.instrumentable)
      return;
    if (!reported_skipped_kernel_names.insert(site.name).second)
      return;

    KernelPatchabilitySkipSummary skipped;
    skipped.kernel_name = site.name;
    skipped.reason = patchability.instrumentation_reason != nullptr
                         ? patchability.instrumentation_reason
                         : "not-instrumented";
    skipped.entry_probe_safe = patchability.entry_probe_safe;
    skipped.self_contained_probe_safe = patchability.self_contained_probe_safe;
    skipped.branch_probe_safe = patchability.branch_probe_safe;
    skipped.prefers_self_contained_edge_probes =
        patchability.prefers_self_contained_edge_probes;
    skipped.prefers_fixed_branch_counters = patchability.prefers_fixed_branch_counters;
    report.skipped_kernels.push_back(std::move(skipped));
  };
  for (const KernelSite &site : sites)
    record_skipped_kernel(site, classify_current_kernel(site.name));
  if (!self_contained_edge_probes) {
    std::optional<std::string_view> code_object_self_contained_reason =
        code_object_self_contained_edge_probe_reason(sites, current_patchability_filters());
    if (code_object_self_contained_reason) {
      self_contained_edge_probes = true;
      coverage_strategy_reason = std::string(*code_object_self_contained_reason);
      configure_plan();
      if (g_verbose) {
        fprintf(stderr,
                "rocjitsu-afl: using self-contained edge probes for %s because no "
                "instrumentable kernel has an entry-safe prologue or the payload is entry-unsafe\n",
                report.context.c_str());
      }
    }
  }

  auto temp_path = write_temp_elf(elf);
  if (!temp_path)
    return finish_empty("temp_elf_write_failed", "preflight");
  rocjitsu::AmdGpuCodeObject co(*temp_path);
  unlink(temp_path->c_str());
  if (!co.is_valid()) {
    if (g_verbose)
      fprintf(stderr, "rocjitsu-afl: AmdGpuCodeObject rejected extracted ELF\n");
    return finish_empty("code_object_rejected", "preflight");
  }

  const bool entry_uses_masked_probe_registers = supports_previous_bb_edges && !g_disable_edges;
  const bool entry_uses_stable_state_sgpr = supports_previous_bb_edges && !g_disable_edges;
  std::unordered_set<std::string> self_contained_edge_kernel_names;
  std::unordered_set<std::string> self_contained_branch_kernel_names;
  std::optional<std::vector<EntryProbeRegisterSelection>> preflight_entry_liveness_selections;
  auto select_self_contained_edge_plan =
      [&](std::span<const KernelSite> kernels, bool verbose) -> InstrumentationPlan {
    InstrumentationPlanOptions branch_options = make_plan_options(true);
    branch_options.verbose = verbose;
    InstrumentationPlan plan = select_edge_sites(co, arch, kernels, branch_options);
    if (!diagnostic_coverage_override_enabled() &&
        branch_options.branch_terminator_slot_policy == EdgeSlotPolicyKind::PreviousBbHash) {
      const std::unordered_map<std::string, std::string> fixed_preferred_kernel_reasons =
          fixed_branch_counter_kernel_reasons(kernels);
      std::unordered_set<std::string> fixed_preferred_kernel_names;
      for (const auto &entry : fixed_preferred_kernel_reasons)
        fixed_preferred_kernel_names.insert(entry.first);
      if (!fixed_preferred_kernel_names.empty()) {
        if (verbose) {
          fprintf(stderr,
                  "rocjitsu-afl: planning %zu self-contained branch kernel(s) with "
                  "fixed counters based on patchability/resource classification\n",
                  fixed_preferred_kernel_names.size());
        }
        std::vector<KernelSite> fixed_kernels =
            collect_kernels_by_name(kernels, fixed_preferred_kernel_names);
        InstrumentationPlanOptions fixed_options = branch_options;
        fixed_options.branch_terminator_slot_policy = EdgeSlotPolicyKind::FixedCounter;
        fixed_options.branch_edge_site_limit = g_branch_edge_site_limit;
        fixed_options.branch_edge_site_limit_auto = false;
        fixed_options.previous_bb_branch_site_limit =
            std::numeric_limits<uint32_t>::max();
        fixed_options.previous_bb_branch_site_limit_auto = false;
        fixed_options.require_liveness_registers = false;
        fixed_options.fixed_counter_fallback_for_branch_liveness = false;
        fixed_options.verbose = verbose;
        InstrumentationPlan fixed_plan =
            select_edge_sites(co, arch, fixed_kernels, fixed_options);
        annotate_branch_slot_policy_reasons(fixed_plan,
                                            fixed_preferred_kernel_reasons);
        plan = drop_kernels_from_plan(std::move(plan), fixed_preferred_kernel_names);
        merge_instrumentation_plan(plan, std::move(fixed_plan));
      }
    }
    if (diagnostic_coverage_override_enabled() ||
        branch_options.branch_terminator_slot_policy != EdgeSlotPolicyKind::PreviousBbHash) {
      return apply_debug_edge_patch_text_offset_filter(std::move(plan));
    }

    std::unordered_map<std::string, std::string> fallback_reasons;
    for (const KernelEdgeSelectionSummary &summary : plan.kernel_summaries) {
      if (selected_edge_count(summary) == 0 && summary.branch_candidates != 0 &&
          summary.skipped_branch_liveness != 0) {
        fallback_reasons.emplace(summary.kernel_name,
                                 "previous-bb-branch-liveness-rejected");
      }
    }
    std::unordered_set<std::string> fixed_fallback_kernel_names;
    for (const auto &entry : fallback_reasons)
      fixed_fallback_kernel_names.insert(entry.first);
    if (fixed_fallback_kernel_names.empty())
      return apply_debug_edge_patch_text_offset_filter(std::move(plan));

    if (verbose) {
      fprintf(stderr,
              "rocjitsu-afl: retrying %zu self-contained branch kernel(s) with "
              "fixed counters after previous-BB planning requested conservative fallback\n",
              fixed_fallback_kernel_names.size());
    }

    std::vector<KernelSite> fallback_kernels =
        collect_kernels_by_summary_names(kernels, fixed_fallback_kernel_names);
    if (fallback_kernels.empty())
      return apply_debug_edge_patch_text_offset_filter(std::move(plan));

    InstrumentationPlanOptions fixed_options = branch_options;
    fixed_options.branch_terminator_slot_policy = EdgeSlotPolicyKind::FixedCounter;
    fixed_options.branch_edge_site_limit = g_branch_edge_site_limit;
    fixed_options.branch_edge_site_limit_auto = false;
    fixed_options.previous_bb_branch_site_limit = std::numeric_limits<uint32_t>::max();
    fixed_options.previous_bb_branch_site_limit_auto = false;
    fixed_options.require_liveness_registers = true;
    fixed_options.fixed_counter_fallback_for_branch_liveness = false;
    fixed_options.verbose = verbose;
    InstrumentationPlan fixed_plan = select_edge_sites(co, arch, fallback_kernels, fixed_options);
    if (fixed_plan.sites.empty())
      return apply_debug_edge_patch_text_offset_filter(std::move(plan));
    annotate_branch_slot_policy_reasons(fixed_plan, fallback_reasons);

    InstrumentationPlan kept = drop_kernels_from_plan(std::move(plan), fixed_fallback_kernel_names);
    merge_instrumentation_plan(kept, std::move(fixed_plan));
    return apply_debug_edge_patch_text_offset_filter(std::move(kept));
  };
  auto branch_preflight_has_sites = [&](std::span<const KernelSite> kernels) -> bool {
    if (kernels.empty())
      return false;
    const InstrumentationPlan branch_preflight =
        select_self_contained_edge_plan(kernels, /*verbose=*/false);
    return !branch_preflight.sites.empty();
  };
  auto switch_to_self_contained_if_branch_preflight_succeeds =
      [&](const char *reason) -> bool {
    const std::vector<KernelSite> branch_edge_kernels =
        collect_edge_planning_kernels(sites, /*self_contained_edge_probes=*/true);
    if (!branch_preflight_has_sites(branch_edge_kernels))
      return false;
    self_contained_edge_probes = true;
    coverage_strategy_reason = reason;
    configure_plan();
    return true;
  };
  if (!diagnostic_coverage_override_enabled() && !self_contained_edge_probes &&
      report.edge_instrumentation_enabled && context_fixed_branch_counters) {
    if (switch_to_self_contained_if_branch_preflight_succeeds(
            loader_patchability.fixed_branch_strategy_reason)) {
      if (g_verbose) {
        fprintf(stderr,
                "rocjitsu-afl: using fixed self-contained branch probes for %s "
                "because entry-backed and previous-BB probes are not the safe default "
                "for this loader path\n",
                report.context.c_str());
      }
    }
  }
  if (!diagnostic_coverage_override_enabled() && !self_contained_edge_probes &&
      report.edge_instrumentation_enabled) {
    const std::vector<KernelSite> entry_edge_kernels =
        collect_edge_planning_kernels(sites, /*self_contained_edge_probes=*/false);
    InstrumentationPlanOptions entry_preflight_options = make_plan_options(false);
    entry_preflight_options.verbose = false;
    const InstrumentationPlan entry_preflight =
        select_edge_sites(co, arch, entry_edge_kernels, entry_preflight_options);

    for (const KernelSite &site : sites) {
      if (should_instrument_kernel(site.name) && entry_probe_is_unsafe(site))
        self_contained_edge_kernel_names.insert(site.name);
    }

    std::vector<EntryProbeRegisterSelection> entry_liveness =
        select_entry_probe_registers(co, arch, entry_edge_kernels,
                                     entry_uses_masked_probe_registers,
                                     entry_uses_stable_state_sgpr,
                                     plan_options.probe_registers);
    const std::unordered_set<std::string> entry_liveness_names =
        entry_liveness_kernel_names(entry_liveness);
    for (const KernelSite &kernel : entry_edge_kernels) {
      if (entry_liveness_names.find(kernel.name) == entry_liveness_names.end())
        self_contained_edge_kernel_names.insert(kernel.name);
    }
    if (!entry_liveness.empty())
      preflight_entry_liveness_selections = std::move(entry_liveness);

    if (!self_contained_edge_kernel_names.empty()) {
      const std::vector<KernelSite> self_contained_candidates =
          collect_kernels_by_name(sites, self_contained_edge_kernel_names);
      if (branch_preflight_has_sites(self_contained_candidates)) {
        coverage_strategy_reason =
            preflight_entry_liveness_selections ? "per-kernel-entry-safety"
                                                : "entry-liveness-preflight-needs-fixed-registers";
        if (g_verbose) {
          fprintf(stderr,
                  "rocjitsu-afl: planning self-contained edge probes for %zu kernel(s) "
                  "in %s because entry redirection is unsafe or needs fixed high "
                  "registers\n",
                  self_contained_edge_kernel_names.size(), report.context.c_str());
        }
      } else if (!preflight_entry_liveness_selections) {
        self_contained_edge_probes = true;
        coverage_strategy_reason = "entry-liveness-preflight-rejected-fixed-registers";
        configure_plan();
        if (g_verbose) {
          fprintf(stderr,
                  "rocjitsu-afl: leaving %s entries unpatched because entry redirection "
                  "would need fixed high-register fallback and branch preflight selected "
                  "no safe hashed sites\n",
                  report.context.c_str());
        }
      }
    } else if (entry_preflight.sites.empty()) {
      if (switch_to_self_contained_if_branch_preflight_succeeds(
              "entry-preflight-found-no-sites")) {
        if (g_verbose) {
          fprintf(stderr,
                  "rocjitsu-afl: switching %s to self-contained edge probes after "
                  "entry preflight selected no edge sites\n",
                  report.context.c_str());
        }
      }
    }
  }

  rocjitsu::CodeObjectPatcher patcher(co);
  patcher.set_cave_start(patcher.text_size());

  std::vector<EntryProbeRegisterSelection> entry_liveness_selections;
  if (!self_contained_edge_probes && preflight_entry_liveness_selections) {
    entry_liveness_selections = std::move(*preflight_entry_liveness_selections);
  } else if (!self_contained_edge_probes && plan_options.liveness_registers) {
    const std::vector<KernelSite> entry_probe_kernels =
        collect_edge_planning_kernels(sites, /*self_contained_edge_probes=*/false);
    entry_liveness_selections =
        select_entry_probe_registers(co, arch, entry_probe_kernels,
                                     entry_uses_masked_probe_registers,
                                     entry_uses_stable_state_sgpr, plan_options.probe_registers);
  }
  const std::unordered_map<std::string, Rdna4ProbeRegisters> entry_liveness_registers =
      entry_register_map(entry_liveness_selections);

  DeviceElfPatchPlan install_plan;
  install_plan.planned_entry_cave_body_size = patcher.cave_body_size();
  for (const KernelSite &site : sites) {
    const KernelPatchability patchability = classify_current_kernel(site.name);
    if (!patchability.instrumentable) {
      if (g_verbose) {
        fprintf(stderr, "rocjitsu-afl: skipping %s: %s\n", site.name.c_str(),
                patchability.instrumentation_reason != nullptr
                    ? patchability.instrumentation_reason
                    : "not-instrumented");
      }
      continue;
    }
    ++install_plan.entry_candidate_count;
    install_plan.edge_probe_sites.push_back(site);
    if (self_contained_edge_probes || entry_probe_is_unsafe(site) ||
        self_contained_edge_kernel_names.find(site.name) !=
            self_contained_edge_kernel_names.end()) {
      if (g_verbose) {
        fprintf(stderr,
                "rocjitsu-afl: leaving %s entry unchanged; edge trampolines will "
                "load coverage state directly\n",
                site.name.c_str());
      }
      continue;
    }
    Rdna4ProbeRegisters entry_probe_registers = plan_options.probe_registers;
    const auto liveness_regs = entry_liveness_registers.find(site.name);
    if (liveness_regs != entry_liveness_registers.end())
      entry_probe_registers = liveness_regs->second;
    std::vector<uint32_t> entry_probe;
    ProbeRegisterRequirements entry_requirements;
    if (!supports_previous_bb_edges) {
      auto entry_only_probe = afl_dbi::rdna4_flagless_counter_probe_with_state_pointer(
          kEntryCounterSlot, state_pointer, arch, entry_probe_registers);
      if (!entry_only_probe)
        return finish_empty("entry_probe_not_supported", "plan");
      entry_probe = std::move(*entry_only_probe);
      entry_requirements =
          afl_dbi::flagless_counter_probe_register_requirements(entry_probe_registers);
    } else if (g_disable_edges) {
      auto entry_only_probe = afl_dbi::rdna4_flagless_counter_probe_with_state_pointer(
          kEntryCounterSlot, state_pointer, arch, entry_probe_registers);
      if (!entry_only_probe)
        return finish_empty("entry_probe_not_supported", "plan");
      entry_probe = std::move(*entry_only_probe);
      entry_requirements =
          afl_dbi::flagless_counter_probe_register_requirements(entry_probe_registers);
    } else if (g_fixed_edge_slots) {
      entry_probe = afl_dbi::rdna4_edge_entry_probe(arch, entry_probe_registers);
      afl_dbi::replace_rdna4_initial_s_load_b64_with_state_pointer(
          entry_probe, entry_probe_registers.state_sgpr, state_pointer);
      entry_requirements =
          afl_dbi::first_active_counter_probe_register_requirements(entry_probe_registers);
    } else {
      entry_probe = afl_dbi::rdna4_previous_bb_edge_probe_with_state_pointer(
          stable_bb_id(site.name, site.entry_text_offset), state_pointer, arch,
          entry_probe_registers);
      entry_requirements =
          afl_dbi::previous_bb_probe_register_requirements(entry_probe_registers);
    }
    std::optional<rocjitsu::KernelEntryProloguePlan> prologue_plan =
        rocjitsu::plan_kernel_entry_prologue(patcher.cave_start(),
                                             install_plan.planned_entry_cave_body_size,
                                             site.entry_text_offset, entry_probe, arch);
    if (!prologue_plan)
      continue;
    install_plan.planned_entry_cave_body_size +=
        static_cast<uint64_t>(prologue_plan->cave_words.size()) * sizeof(uint32_t);
    merge_kernel_probe_requirements(install_plan.kernel_probe_requirements, site.name,
                                    entry_requirements);
    PlannedEntryProbe planned_entry;
    planned_entry.site = site;
    planned_entry.prologue = std::move(*prologue_plan);
    planned_entry.liveness_registers = liveness_regs != entry_liveness_registers.end();
    install_plan.entry_probes.push_back(std::move(planned_entry));
    install_plan.entry_backed_edge_sites.push_back(site);
    if (!diagnostic_coverage_override_enabled() && report.edge_instrumentation_enabled) {
      if (patchability.branch_probe_safe) {
        self_contained_branch_kernel_names.insert(site.name);
      } else if (g_verbose) {
        fprintf(stderr,
                "rocjitsu-afl: keeping %s branch terminators unpatched: %s\n",
                site.name.c_str(), patchability.branch_probe_reason);
      }
    }
    if (g_verbose) {
      fprintf(stderr,
              "rocjitsu-afl: planned %s entry_text=%llu new_entry=%llu "
              "kd_file=%llu\n",
              site.name.c_str(), static_cast<unsigned long long>(site.entry_text_offset),
              static_cast<unsigned long long>(install_plan.entry_probes.back().prologue
                                                  .new_entry_text_offset),
              static_cast<unsigned long long>(site.descriptor_file_offset));
    }
  }

  install_plan.text.assign(patcher.text_bytes().begin(), patcher.text_bytes().end());
  std::vector<KernelSite> full_self_contained_edge_sites =
      self_contained_edge_probes
          ? install_plan.edge_probe_sites
          : collect_kernels_by_name(install_plan.edge_probe_sites,
                                    self_contained_edge_kernel_names);
  std::vector<KernelSite> branch_only_self_contained_edge_sites;
  if (!self_contained_edge_probes && !self_contained_branch_kernel_names.empty()) {
    std::unordered_set<std::string> full_self_contained_names;
    for (const KernelSite &site : full_self_contained_edge_sites)
      full_self_contained_names.insert(site.name);
    for (const KernelSite &site :
         collect_kernels_by_name(install_plan.edge_probe_sites,
                                 self_contained_branch_kernel_names)) {
      if (full_self_contained_names.find(site.name) == full_self_contained_names.end())
        branch_only_self_contained_edge_sites.push_back(site);
    }
  }
  install_plan.self_contained_edge_sites = full_self_contained_edge_sites;
  std::unordered_set<std::string> self_contained_site_names;
  for (const KernelSite &site : install_plan.self_contained_edge_sites)
    self_contained_site_names.insert(site.name);
  for (const KernelSite &site : branch_only_self_contained_edge_sites)
    append_unique_kernel_site(install_plan.self_contained_edge_sites,
                              self_contained_site_names, site);
  install_plan.hybrid_edge_probes =
      !self_contained_edge_probes && !install_plan.entry_backed_edge_sites.empty() &&
      !install_plan.self_contained_edge_sites.empty();
  if (!install_plan.self_contained_edge_sites.empty()) {
    InstrumentationPlanOptions branch_options = make_plan_options(true);
    if (!diagnostic_coverage_override_enabled() &&
        all_kernels_prefer_fixed_branch_counters(install_plan.self_contained_edge_sites)) {
      branch_options.branch_terminator_slot_policy = EdgeSlotPolicyKind::FixedCounter;
      branch_options.branch_edge_site_limit = g_branch_edge_site_limit;
      branch_options.branch_edge_site_limit_auto = false;
      branch_options.previous_bb_branch_site_limit =
          std::numeric_limits<uint32_t>::max();
      branch_options.previous_bb_branch_site_limit_auto = false;
      branch_options.require_liveness_registers = false;
    }
    report.branch_edge_slots = true;
    report.branch_edge_slot_policy = branch_options.branch_terminator_slot_policy;
    report.branch_edge_site_limit = branch_options.branch_edge_site_limit;
    report.branch_edge_site_limit_auto = branch_options.branch_edge_site_limit_auto;
    report.previous_bb_branch_site_limit =
        branch_options.previous_bb_branch_site_limit;
    report.previous_bb_branch_site_limit_auto =
        branch_options.previous_bb_branch_site_limit_auto;
    report.require_liveness_registers =
        report.require_liveness_registers || branch_options.require_liveness_registers;
    if (install_plan.hybrid_edge_probes) {
      report.coverage_strategy = "hybrid-entry-previous-bb-and-self-contained-branch";
    } else if (!self_contained_edge_probes && install_plan.entry_backed_edge_sites.empty()) {
      report.skip_entry_probe = true;
      report.edge_site_limit = branch_options.block_entry_site_limit;
      report.coverage_strategy =
          coverage_strategy_name(branch_options, report.edge_instrumentation_enabled);
    }
    report.coverage_strategy_reason = coverage_strategy_reason;
  }

  if (!g_disable_edges && supports_previous_bb_edges) {
    if (!install_plan.entry_backed_edge_sites.empty()) {
      InstrumentationPlan entry_backed_selection =
          select_edge_sites(co, arch, install_plan.entry_backed_edge_sites, plan_options);
      entry_backed_selection =
          apply_debug_edge_patch_text_offset_filter(std::move(entry_backed_selection));
      merge_instrumentation_plan(install_plan.edge_selection,
                                 std::move(entry_backed_selection));
    }
    if (!install_plan.self_contained_edge_sites.empty()) {
      InstrumentationPlan self_contained_selection =
          select_self_contained_edge_plan(install_plan.self_contained_edge_sites, g_verbose);
      merge_instrumentation_plan(install_plan.edge_selection,
                                 std::move(self_contained_selection));
    }
  }
  install_plan.edge_sites = install_plan.edge_selection.sites;
  if (install_plan.edge_selection.slot_policy_summary.fixed_edge_sites != 0 &&
      install_plan.edge_selection.slot_policy_summary.hashed_edge_sites == 0) {
    report.branch_edge_slot_policy = EdgeSlotPolicyKind::FixedCounter;
    if (report.coverage_strategy == "self-contained-previous-bb-branch")
      report.coverage_strategy = "self-contained-fixed-branch";
    else if (report.coverage_strategy == "entry-previous-bb-block-and-previous-bb-branch")
      report.coverage_strategy = "entry-previous-bb-block-and-fixed-branch";
  } else if (install_plan.edge_selection.slot_policy_summary.fixed_edge_sites != 0 &&
             install_plan.edge_selection.slot_policy_summary.hashed_edge_sites != 0) {
    if (report.coverage_strategy == "self-contained-previous-bb-branch")
      report.coverage_strategy = "self-contained-hybrid-previous-bb-and-fixed-branch";
    else if (report.coverage_strategy == "entry-previous-bb-block-and-previous-bb-branch")
      report.coverage_strategy = "entry-previous-bb-block-and-hybrid-branch";
  }
  for (const EdgeSite &site : install_plan.edge_sites) {
    merge_kernel_probe_requirements(install_plan.kernel_probe_requirements, site.kernel_name,
                                    afl_dbi::edge_site_probe_register_requirements(site));
  }
  apply_debug_forced_private_segment_bytes(report.context,
                                           install_plan.kernel_probe_requirements);
  record_patch_plan_summary(report, install_plan);
  {
    // Descriptor resources are planned after entry and edge selection so each
    // kernel is sized from the union of the probes it may run. The byte-level
    // descriptor mutation is delayed until trampoline placement also succeeds.
    std::vector<KernelSite> descriptor_sites;
    std::unordered_set<std::string> descriptor_site_names;
    for (const KernelSite &site : install_plan.entry_backed_edge_sites)
      append_unique_kernel_site(descriptor_sites, descriptor_site_names, site);
    std::unordered_set<std::string> self_contained_edge_site_names;
    for (const EdgeSite &site : install_plan.edge_sites) {
      if (site.self_contained_probe)
        self_contained_edge_site_names.insert(site.kernel_name);
    }
    for (const KernelSite &site : collect_kernels_by_name(install_plan.edge_probe_sites,
                                                          self_contained_edge_site_names))
      append_unique_kernel_site(descriptor_sites, descriptor_site_names, site);
    std::unordered_set<std::string> descriptor_ready_kernels;
    for (const KernelSite &site : descriptor_sites) {
      auto requirements = install_plan.kernel_probe_requirements.find(site.name);
      if (requirements == install_plan.kernel_probe_requirements.end())
        continue;
      const char *descriptor_plan_failure = nullptr;
      std::optional<KernelDescriptorResourceSummary> descriptor_summary =
          plan_kernel_descriptor_resources(patcher.image_bytes(), site, requirements->second,
                                           &descriptor_plan_failure);
      if (!descriptor_summary) {
        if (report.descriptor_resource_failure_reason.empty()) {
          report.descriptor_resource_failure_reason =
              descriptor_plan_failure != nullptr ? descriptor_plan_failure
                                                 : "descriptor resource planning failed";
        }
        continue;
      }
      install_plan.descriptor_resources.push_back(std::move(*descriptor_summary));
      descriptor_ready_kernels.insert(site.name);
    }
    for (const KernelSite &site : install_plan.entry_backed_edge_sites) {
      if (install_plan.kernel_probe_requirements.find(site.name) !=
              install_plan.kernel_probe_requirements.end() &&
          descriptor_ready_kernels.find(site.name) == descriptor_ready_kernels.end())
        return finish_empty("descriptor_resource_patch_failed", "plan");
    }
    install_plan.edge_sites.erase(
        std::remove_if(install_plan.edge_sites.begin(), install_plan.edge_sites.end(),
                       [&](const EdgeSite &site) {
                         return site.self_contained_probe &&
                                descriptor_ready_kernels.find(site.kernel_name) ==
                                    descriptor_ready_kernels.end();
                       }),
        install_plan.edge_sites.end());
  }
  LocalTextCaveAllocator local_caves(install_plan.text);
  install_plan.local_text_cave_summary = local_caves.summary();
  FixedEdgeSlotTracker placement_fixed_slot_tracker;
  prime_fixed_counter_placement_tracker(placement_fixed_slot_tracker,
                                        install_plan.edge_sites);
  install_plan.edge_trampolines.reserve(install_plan.edge_sites.size());
  install_plan.planned_cave_body_size = install_plan.planned_entry_cave_body_size;
  for (const EdgeSite &edge : install_plan.edge_sites) {
    const char *failure_reason = nullptr;
    auto planned =
        plan_edge_trampoline(edge, install_plan.text, install_plan.planned_cave_body_size,
                             patcher.cave_start(), local_caves, arch, state_pointer,
                             &failure_reason);
    if (!planned && previous_bb_branch_site(edge)) {
      const std::string_view reason =
          failure_reason != nullptr ? std::string_view(failure_reason)
                                    : std::string_view();
      if (placement_failure_can_degrade_to_fixed(reason) &&
          placement_fixed_fallback_has_budget(install_plan.edge_selection, edge)) {
        EdgeSite fallback_edge = make_stable_fixed_counter_fallback_site(edge);
        const char *fallback_failure_reason = nullptr;
        auto fallback_planned = plan_edge_trampoline(
            fallback_edge, install_plan.text, install_plan.planned_cave_body_size,
            patcher.cave_start(), local_caves, arch, state_pointer,
            &fallback_failure_reason);
        if (fallback_planned) {
          const uint32_t fixed_slot_collisions =
              record_fixed_counter_placement_slots(placement_fixed_slot_tracker,
                                                   fallback_planned->site);
          fallback_planned->site.fixed_slot_collisions = fixed_slot_collisions;
          if (!record_previous_bb_branch_placement_fallback(
                  install_plan.edge_selection, fallback_planned->site,
                  fixed_slot_collisions, reason)) {
            continue;
          }
          if (fallback_planned->result.placement ==
              EdgeTrampolinePlacement::AppendedCave) {
            install_plan.planned_cave_body_size +=
                fallback_planned->trampoline.cave_words.size() * sizeof(uint32_t);
          }
          if (g_verbose) {
            fprintf(stderr,
                    "rocjitsu-afl: degraded previous-BB branch edge bb_id=0x%x "
                    "kernel=%s patch=%llu to fixed counter after placement "
                    "failure: %s\n",
                    edge.bb_id, edge.kernel_name.c_str(),
                    static_cast<unsigned long long>(edge.patch_text_offset),
                    failure_reason != nullptr ? failure_reason : "unknown");
          }
          install_plan.edge_trampolines.push_back(std::move(*fallback_planned));
          continue;
        }
      }
    }
    if (!planned) {
      ++install_plan.edge_patch_failures;
      const char *reason =
          failure_reason != nullptr ? failure_reason : "edge trampoline emission failed";
      if (std::string_view(reason).find("range") != std::string_view::npos)
        ++install_plan.branch_range_failures;
      if (install_plan.sampled_edge_failures.size() < 32) {
        install_plan.sampled_edge_failures.push_back(
            {edge.kernel_name, afl_dbi::edge_patch_kind_name(edge.kind), edge.patch_text_offset,
             edge.return_text_offset, reason});
      }
      if (g_verbose) {
        fprintf(stderr,
                "rocjitsu-afl: skipping edge bb_id=0x%x kernel=%s patch=%llu "
                "target=%llu reason=%s\n",
                edge.bb_id, edge.kernel_name.c_str(),
                static_cast<unsigned long long>(edge.patch_text_offset),
                static_cast<unsigned long long>(edge.return_text_offset),
                reason);
      }
      continue;
    }
    if (planned->result.placement == EdgeTrampolinePlacement::AppendedCave)
      install_plan.planned_cave_body_size +=
          planned->trampoline.cave_words.size() * sizeof(uint32_t);
    install_plan.edge_trampolines.push_back(std::move(*planned));
  }
  record_patch_plan_summary(report, install_plan);

  std::unordered_set<std::string> kernels_needing_descriptor_updates;
  for (const KernelSite &site : install_plan.entry_backed_edge_sites) {
    if (install_plan.kernel_probe_requirements.find(site.name) !=
        install_plan.kernel_probe_requirements.end())
      kernels_needing_descriptor_updates.insert(site.name);
  }
  for (const PlannedEdgeTrampoline &planned : install_plan.edge_trampolines) {
    if (install_plan.kernel_probe_requirements.find(planned.site.kernel_name) !=
        install_plan.kernel_probe_requirements.end())
      kernels_needing_descriptor_updates.insert(planned.site.kernel_name);
  }
  for (const KernelDescriptorResourceSummary &summary : install_plan.descriptor_resources) {
    if (kernels_needing_descriptor_updates.find(summary.kernel_name) ==
        kernels_needing_descriptor_updates.end())
      continue;
    auto requirements = install_plan.kernel_probe_requirements.find(summary.kernel_name);
    if (requirements == install_plan.kernel_probe_requirements.end())
      continue;
    const char *descriptor_patch_failure = nullptr;
    std::optional<afl_dbi::AmdgpuMetadataPrivateSegmentPatch::Kind> metadata_patch_kind;
    std::optional<afl_dbi::AmdgpuMetadataPrivateSegmentPatch::Kind>
        sgpr_metadata_patch_kind;
    if (!patch_kernel_descriptor_resources(patcher, summary, &descriptor_patch_failure,
                                           &metadata_patch_kind,
                                           &sgpr_metadata_patch_kind)) {
      return finish_empty(descriptor_patch_failure != nullptr
                              ? descriptor_patch_failure
                              : "descriptor_resource_patch_failed",
                          "install");
    }
    record_probe_requirements(report, requirements->second);
    report.spill_bytes = std::max(report.spill_bytes, summary.spill_bytes);
    ++report.descriptor_updates;
    KernelDescriptorResourceSummary reported_summary = summary;
    if (sgpr_metadata_patch_kind)
      reported_summary.sgpr_count_metadata_patch =
          amdgpu_metadata_private_segment_patch_kind_name(*sgpr_metadata_patch_kind);
    if (metadata_patch_kind)
      reported_summary.private_segment_metadata_patch =
          amdgpu_metadata_private_segment_patch_kind_name(*metadata_patch_kind);
    report.descriptor_resources.push_back(std::move(reported_summary));
  }

  uint32_t patched = 0;
  for (const PlannedEntryProbe &planned : install_plan.entry_probes) {
    patcher.append_cave_body(planned.prologue.cave_words);
    if (!patcher.redirect_kernel_entry(planned.site.descriptor_file_offset,
                                       planned.site.entry_text_offset,
                                       planned.prologue.new_entry_text_offset)) {
      return finish_empty("entry_redirect_failed", "install");
    }
    if (planned.liveness_registers) {
      ++report.entry_liveness_register_kernels;
      ++report.entry_liveness_probe_points;
    }
    ++patched;
    if (g_verbose) {
      fprintf(stderr,
              "rocjitsu-afl: patched %s entry_text=%llu new_entry=%llu "
              "kd_file=%llu\n",
              planned.site.name.c_str(),
              static_cast<unsigned long long>(planned.site.entry_text_offset),
              static_cast<unsigned long long>(planned.prologue.new_entry_text_offset),
              static_cast<unsigned long long>(planned.site.descriptor_file_offset));
    }
  }

  uint32_t edge_patched = 0;
  uint32_t local_edge_patched = 0;
  uint32_t appended_edge_patched = 0;
  for (const PlannedEdgeTrampoline &planned : install_plan.edge_trampolines) {
    install_planned_edge_trampoline(planned, install_plan.text, patcher, arch);
    ++edge_patched;
    if (planned.result.placement == EdgeTrampolinePlacement::LocalTextCave)
      ++local_edge_patched;
    else
      ++appended_edge_patched;
    if (g_verbose) {
      const EdgeSite &edge = planned.site;
      const char *kind = afl_dbi::edge_patch_kind_name(edge.kind);
      const char *placement =
          planned.result.placement == EdgeTrampolinePlacement::LocalTextCave ? "local-cave"
                                                                             : "appended-cave";
      fprintf(stderr,
              "rocjitsu-afl: patched %s edge bb_id=0x%x fixed_slot=%u "
              "fallthrough_slot=%u kernel=%s pred=%llu pred_count=%u patch=%llu cave=%llu "
              "placement=%s target=%llu inst_size=%u\n",
              kind, edge.bb_id, edge.fixed_slot, edge.fallthrough_slot, edge.kernel_name.c_str(),
              static_cast<unsigned long long>(edge.pred_text_offset), edge.predecessor_count,
              static_cast<unsigned long long>(edge.patch_text_offset),
              static_cast<unsigned long long>(planned.result.cave_text_offset), placement,
              static_cast<unsigned long long>(edge.return_text_offset), edge.first_inst_size);
    }
  }
  if (edge_patched != 0)
    patcher.overwrite_text(install_plan.text);

  report.entry_patched = patched;
  report.edge_sites_patched = edge_patched;
  report.local_text_caves = local_edge_patched;
  report.appended_caves = appended_edge_patched;
  report.appended_cave_bytes = patcher.cave_body_size();

  if (patched == 0 && edge_patched == 0)
    return finish_empty("no_patchable_sites", "plan");
  if (!patcher.append_cave_section(".rj_afl_entry"))
    return finish_empty("append_cave_section_failed", "emit");
  if (g_verbose) {
    fprintf(stderr,
            "rocjitsu-afl: patched %u kernel entries and %u edge sites "
            "(local_text_caves=%u)\n",
            patched, edge_patched, local_edge_patched);
  }
  std::vector<uint8_t> emitted = patcher.emit();
  if (emitted.empty())
    return finish_empty("emit_failed", "emit");
  report.output_bytes = emitted.size();
  report.success = true;
  report.reason = "patched";
  emit_patch_report(report);
  return emitted;
}

std::vector<afl_dbi::DeviceImage>
order_device_images_for_current_device(const std::vector<afl_dbi::DeviceImage> &images,
                                       const char *context);
void remember_runtime_shadow_source(std::span<const uint8_t> image, const char *context);
bool device_image_has_candidate_kernel(std::span<const uint8_t> image);

struct PatchedCodeObjectImage {
  std::vector<uint8_t> image;
  std::vector<uint8_t> raw_elf_fallback;
  std::string device_image_id;
  uint32_t device_image_index = kUnknownDeviceImageIndex;
  bool rebuilt_container = false;

  bool empty() const { return image.empty(); }
};

enum class CcobPatchPolicy {
  RawElfBypass,
  RebuildPreservingSiblings,
};

PatchedCodeObjectImage
patch_selected_device_images(std::span<const uint8_t> image, uint64_t state_pointer,
                             const char *context,
                             const afl_dbi::CodeObjectImageSummary &summary,
                             std::vector<afl_dbi::DeviceImage> device_images,
                             CcobPatchPolicy ccob_policy) {
  PatchedCodeObjectImage result;
  if (image.empty() || state_pointer == 0 || device_images.empty())
    return result;

  device_images = order_device_images_for_current_device(device_images, context);

  for (const afl_dbi::DeviceImage &device_image : device_images) {
    std::vector<uint8_t> patched =
        patch_device_elf(device_image.bytes, state_pointer, context, device_image.id.c_str(),
                         device_image.index);
    if (!patched.empty()) {
      if (g_verbose) {
        fprintf(stderr, "rocjitsu-afl: selected device image %s from %s\n",
                device_image.id.c_str(), context);
      }
      remember_runtime_shadow_source(device_image.bytes, context);
      result.device_image_id = device_image.id;
      result.device_image_index = device_image.index;
      if (ccob_policy == CcobPatchPolicy::RebuildPreservingSiblings &&
          summary.raw_elf_bypass_drops_sibling_payloads) {
        std::optional<std::vector<uint8_t>> rebuilt =
            afl_dbi::rebuild_code_object_image_with_replaced_device_image(
                image, device_image, patched);
        if (rebuilt) {
          if (g_verbose) {
            fprintf(stderr,
                    "rocjitsu-afl: rebuilt code-object container for %s preserving sibling "
                    "payloads\n",
                    device_image.id.c_str());
          }
          const afl_dbi::CodeObjectImageSummary rebuilt_summary =
              afl_dbi::summarize_code_object_image(*rebuilt);
          emit_ccob_rebuild_report(
              context, device_image.id, device_image.index, /*success=*/true, image.size(),
              rebuilt->size(), summary.device_image_count, rebuilt_summary.device_image_count,
              rebuilt_summary.device_image_count == summary.device_image_count);
          result.image = std::move(*rebuilt);
          result.raw_elf_fallback = std::move(patched);
          result.rebuilt_container = true;
          return result;
        }
        emit_ccob_rebuild_report(context, device_image.id, device_image.index,
                                 /*success=*/false, image.size(), /*output_bytes=*/0,
                                 summary.device_image_count, /*output_device_images=*/0,
                                 /*sibling_payloads_preserved=*/false);
        if (g_verbose) {
          fprintf(stderr,
                  "rocjitsu-afl: failed to rebuild code-object container for %s; falling back "
                  "to raw ELF bypass\n",
                  device_image.id.c_str());
        }
      }
      result.image = std::move(patched);
      return result;
    }
  }

  if (g_verbose) {
    fprintf(stderr, "rocjitsu-afl: no patchable AMDGPU ELF found among %zu image(s) in %s\n",
            device_images.size(), context);
  }
  return result;
}

PatchedCodeObjectImage patch_code_object_image(std::span<const uint8_t> image,
                                               uint64_t state_pointer, const char *context,
                                               bool require_candidate_kernel,
                                               CcobPatchPolicy ccob_policy) {
  PatchedCodeObjectImage result;
  if (image.empty())
    return result;

  std::vector<afl_dbi::DeviceImage> device_images = afl_dbi::extract_device_images(image);
  const afl_dbi::CodeObjectImageSummary summary = afl_dbi::summarize_code_object_image(image);
  emit_code_object_image_report(context, image.size(), summary);
  if (device_images.empty()) {
    if (g_verbose)
      fprintf(stderr, "rocjitsu-afl: no supported AMDGPU ELF found in %s\n", context);
    return {};
  }

  if (require_candidate_kernel) {
    std::vector<afl_dbi::DeviceImage> candidates;
    for (afl_dbi::DeviceImage &device_image : device_images) {
      if (device_image_has_candidate_kernel(device_image.bytes))
        candidates.push_back(std::move(device_image));
    }
    device_images = std::move(candidates);
    if (device_images.empty())
      return result;
  }

  if (state_pointer == 0)
    return result;

  return patch_selected_device_images(image, state_pointer, context, summary,
                                      std::move(device_images), ccob_policy);
}

std::vector<uint8_t> patch_module_image(std::span<const uint8_t> image, uint64_t state_pointer,
                                         const char *context) {
  // HIP lazy shadow modules load one exact-kernel replacement at a time. Keeping
  // the raw ELF bypass here avoids duplicate rebuilt-CCOB shadow modules, which
  // currently load but are not launch-stable on the branchy multi-payload smoke.
  PatchedCodeObjectImage patched =
      patch_code_object_image(image, state_pointer, context, /*require_candidate_kernel=*/false,
                              CcobPatchPolicy::RawElfBypass);
  return std::move(patched.image);
}

std::vector<uint8_t> patch_module_file(const char *path, uint64_t state_pointer) {
  std::vector<uint8_t> image = afl_dbi::read_file_bytes(path);
  if (image.empty())
    return {};
  return patch_module_image(image, state_pointer, path != nullptr ? path : "<null>");
}

std::vector<uint8_t> patch_module_data(const void *image, uint64_t state_pointer) {
  std::vector<uint8_t> copied = afl_dbi::copy_module_data_image(image);
  if (copied.empty())
    return {};
  return patch_module_image(copied, state_pointer, "in-memory code object");
}

uint64_t device_state_pointer_for_patching();
ShadowModuleUnloadStats unload_shadow_modules(std::vector<hipModule_t> modules,
                                              const char *context);

uint64_t env_bounded_delay_us(const char *name) {
  const char *delay = getenv(name);
  if (delay == nullptr || delay[0] == '\0')
    return 0;
  char *end = nullptr;
  const unsigned long long parsed = strtoull(delay, &end, 10);
  if (end == delay || parsed == 0)
    return 0;
  return std::min<unsigned long long>(parsed, 1000000ull);
}

uint64_t runtime_shadow_race_delay_us() {
  return env_bounded_delay_us("ROCFUZZ_RUNTIME_SHADOW_RACE_DELAY_US");
}

uint64_t lazy_ccob_race_delay_us() {
  return env_bounded_delay_us("ROCFUZZ_LAZY_CCOB_RACE_DELAY_US");
}

bool should_defer_ccob_module_patch(std::span<const uint8_t> image) {
  if (image.empty() || !afl_dbi::is_ccob_image(image))
    return false;
  if (real_hipModuleLoadData == nullptr)
    return false;
  return !g_launch_only && g_kernel_include == nullptr && g_kernel_include_override == nullptr;
}

void remember_lazy_ccob_module(hipModule_t module, std::vector<uint8_t> image,
                               const char *context) {
  if (module == nullptr || image.empty())
    return;

  const size_t image_bytes = image.size();
  std::optional<afl_dbi::CodeObjectImageSummary> summary;
  if (g_verbose || patch_reports_enabled())
    summary = afl_dbi::summarize_code_object_image(image);
  {
    std::lock_guard<std::mutex> lock(g_state_mutex);
    LazyCcobModule lazy;
    lazy.image = std::move(image);
    g_lazy_ccob_modules[module_key(module)] = std::move(lazy);
  }
  emit_lazy_ccob_module_report(module, context, image_bytes, summary);
  if (g_verbose && summary) {
    fprintf(stderr,
            "rocjitsu-afl: deferring CCOB instrumentation for module=%p from %s "
            "device_images=%u raw_elf_bypass_drops_sibling_payloads=%u\n",
            reinterpret_cast<void *>(module), context, summary->device_image_count,
            summary->raw_elf_bypass_drops_sibling_payloads ? 1u : 0u);
  }
}

std::optional<LazyCcobLaunchBinding> lazy_ccob_launch_binding(hipFunction_t function) {
  if (function == nullptr || !patch_reports_enabled())
    return std::nullopt;

  const uintptr_t launched_function = function_key(function);
  std::lock_guard<std::mutex> lock(g_state_mutex);
  for (const auto &[module, lazy] : g_lazy_ccob_modules) {
    for (const auto &[kernel, target] : lazy.functions) {
      if (function_key(target.function) == launched_function)
        return LazyCcobLaunchBinding{module, kernel, target};
    }
  }
  return std::nullopt;
}

std::optional<hipError_t> try_get_lazy_ccob_function(hipFunction_t *function, hipModule_t module,
                                                     const char *kname) {
  if (function == nullptr || module == nullptr || kname == nullptr || kname[0] == '\0' ||
      real_hipModuleLoadData == nullptr || real_hipModuleGetFunction == nullptr) {
    return std::nullopt;
  }

  std::vector<uint8_t> image;
  LazyPatchedFunction cached;
  bool cache_hit = false;
  size_t cached_total_functions = 0;
  {
    std::lock_guard<std::mutex> lock(g_state_mutex);
    auto module_it = g_lazy_ccob_modules.find(module_key(module));
    if (module_it == g_lazy_ccob_modules.end())
      return std::nullopt;
    auto function_it = module_it->second.functions.find(kname);
    if (function_it != module_it->second.functions.end()) {
      cached = function_it->second;
      cache_hit = true;
      cached_total_functions = module_it->second.functions.size();
    } else {
      image = module_it->second.image;
    }
  }
  if (cache_hit) {
    *function = cached.function;
    emit_lazy_ccob_function_report(module, kname, cached, nullptr, /*cache_hit=*/true,
                                   /*inserted=*/false, /*duplicate_unloaded=*/false,
                                   /*owner_missing=*/false,
                                   cached_total_functions);
    return hipSuccess;
  }

  const uint64_t state_pointer = device_state_pointer_for_patching();
  if (state_pointer == 0)
    return std::nullopt;

  std::vector<uint8_t> patched;
  {
    ScopedKernelIncludeOverride include(kname);
    patched = patch_module_image(image, state_pointer, "lazy CCOB module");
  }
  if (patched.empty())
    return std::nullopt;

  auto owned = std::make_unique<std::vector<uint8_t>>(std::move(patched));
  const void *data = owned->data();
  hipModule_t patched_module = nullptr;
  hipError_t err = hipSuccess;
  {
    ScopedInterceptionBypass bypass;
    err = real_hipModuleLoadData(&patched_module, data);
  }
  if (err != hipSuccess)
    return err;

  hipFunction_t patched_function = nullptr;
  {
    ScopedInterceptionBypass bypass;
    err = real_hipModuleGetFunction(&patched_function, patched_module, kname);
  }
  if (err != hipSuccess || patched_function == nullptr) {
    unload_shadow_modules({patched_module}, "failed lazy CCOB");
    return err;
  }
  if (const uint64_t delay_us = lazy_ccob_race_delay_us(); delay_us != 0)
    usleep(static_cast<useconds_t>(delay_us));

  LazyPatchedFunction shadow{patched_module, patched_function};
  LazyPatchedFunction published = shadow;
  bool inserted = false;
  bool owner_missing = false;
  size_t total_functions = 0;
  {
    std::lock_guard<std::mutex> lock(g_state_mutex);
    auto module_it = g_lazy_ccob_modules.find(module_key(module));
    if (module_it == g_lazy_ccob_modules.end()) {
      owner_missing = true;
    } else {
      auto existing = module_it->second.functions.find(kname);
      if (existing != module_it->second.functions.end()) {
        published = existing->second;
      } else {
        g_patched_images.push_back(std::move(owned));
        module_it->second.functions[kname] = shadow;
        g_module_function_names[function_key(patched_function)] = kname;
        inserted = true;
      }
      total_functions = module_it->second.functions.size();
    }
  }
  if (owner_missing) {
    emit_lazy_ccob_function_report(module, kname, shadow, patched_module, /*cache_hit=*/false,
                                   /*inserted=*/false, /*duplicate_unloaded=*/false,
                                   /*owner_missing=*/true, total_functions);
    unload_shadow_modules({patched_module}, "orphan lazy CCOB");
    return std::nullopt;
  }
  if (!inserted)
    unload_shadow_modules({patched_module}, "duplicate lazy CCOB");

  *function = published.function;
  emit_lazy_ccob_function_report(module, kname, published, patched_module, /*cache_hit=*/false,
                                 inserted, /*duplicate_unloaded=*/!inserted,
                                 /*owner_missing=*/false, total_functions);
  if (!inserted)
    return hipSuccess;

  if (g_verbose) {
    fprintf(stderr,
            "rocjitsu-afl: lazy patched CCOB kernel module=%p patched_module=%p "
            "function=%p name=%s\n",
            reinterpret_cast<void *>(module), reinterpret_cast<void *>(published.module),
            reinterpret_cast<void *>(published.function), kname);
  }
  return hipSuccess;
}

bool device_image_has_candidate_kernel(std::span<const uint8_t> image) {
  for (const KernelSite &site : find_kernel_sites(image)) {
    if (should_instrument_kernel(site.name))
      return true;
  }
  return false;
}

uint64_t hash_bytes(std::span<const uint8_t> image) {
  uint64_t hash = 1469598103934665603ull;
  for (uint8_t byte : image) {
    hash ^= byte;
    hash *= 1099511628211ull;
  }
  return hash;
}

uint64_t elapsed_us(Clock::time_point start, Clock::time_point end) {
  return static_cast<uint64_t>(
      std::chrono::duration_cast<std::chrono::microseconds>(end - start).count());
}

uint64_t runtime_shadow_source_dedupe_key(uintptr_t registration_key, uint64_t image_hash) {
  uint64_t key = image_hash;
  key ^= registration_key + 0x9e3779b97f4a7c15ull + (key << 6) + (key >> 2);
  return key;
}

std::string runtime_shadow_function_key(uintptr_t registration_key, std::string_view kernel_name) {
  std::string key = std::to_string(registration_key);
  key.push_back('\n');
  key.append(kernel_name.data(), kernel_name.size());
  return key;
}

std::vector<std::string> runtime_shadow_kernel_names(std::span<const uint8_t> image) {
  std::vector<std::string> names;
  std::unordered_set<std::string> seen;
  for (const KernelSite &site : find_kernel_sites(image)) {
    if (seen.insert(site.name).second)
      names.push_back(site.name);
  }
  return names;
}

bool runtime_shadow_source_has_kernel(const RuntimeShadowSourceSnapshot &source,
                                      std::string_view kernel_name) {
  return std::any_of(source.kernel_names.begin(), source.kernel_names.end(),
                     [&](const std::string &name) { return name == kernel_name; });
}

void remember_runtime_shadow_source(std::span<const uint8_t> image, const char *context) {
  if (!g_runtime_shadow_modules || image.empty())
    return;
  const uint64_t hash = hash_bytes(image);
  const uintptr_t registration_key = g_runtime_shadow_registration_override;
  const uint64_t dedupe_key = runtime_shadow_source_dedupe_key(registration_key, hash);
  std::vector<std::string> kernel_names = runtime_shadow_kernel_names(image);
  bool inserted = false;
  size_t total_sources = 0;
  {
    std::lock_guard<std::mutex> lock(g_state_mutex);
    inserted = runtime_shadow_source_hashes().insert(dedupe_key).second;
    if (inserted) {
      auto owned = std::make_shared<const std::vector<uint8_t>>(image.begin(), image.end());
      runtime_shadow_sources().push_back(RuntimeShadowSource{registration_key, dedupe_key,
                                                             std::move(owned), kernel_names});
    }
    total_sources = runtime_shadow_sources().size();
  }
  emit_runtime_shadow_source_report(context, registration_key, image.size(), hash, dedupe_key,
                                    inserted, total_sources, kernel_names);
  if (!inserted)
    return;
  if (g_verbose) {
    fprintf(stderr,
            "rocjitsu-afl: cached runtime shadow source from %s bytes=%zu "
            "registration=%p kernels=%zu\n",
            context, image.size(), reinterpret_cast<void *>(registration_key),
            kernel_names.size());
  }
}

std::vector<RuntimeShadowSourceSnapshot>
runtime_shadow_sources_snapshot(uintptr_t registration_key) {
  std::lock_guard<std::mutex> lock(g_state_mutex);
  std::vector<RuntimeShadowSourceSnapshot> sources;
  for (const RuntimeShadowSource &source : runtime_shadow_sources()) {
    if (source.registration_key == registration_key ||
        (registration_key != 0 && source.registration_key == 0)) {
      sources.push_back(
          RuntimeShadowSourceSnapshot{source.dedupe_key, source.image, source.kernel_names});
    }
  }
  if (!sources.empty() || registration_key == 0)
    return sources;
  for (const RuntimeShadowSource &source : runtime_shadow_sources()) {
    if (source.registration_key == 0)
      sources.push_back(
          RuntimeShadowSourceSnapshot{source.dedupe_key, source.image, source.kernel_names});
  }
  return sources;
}

uint64_t device_state_pointer_for_patching() {
  std::lock_guard<std::mutex> lock(g_state_mutex);
  if (!ensure_runtime_locked())
    return 0;
  return reinterpret_cast<uint64_t>(g_device_counters);
}

std::vector<uint8_t> read_whole_fd(int fd) {
  if (fd < 0)
    return {};
  const off_t original = lseek(fd, 0, SEEK_CUR);
  const off_t end = lseek(fd, 0, SEEK_END);
  if (original >= 0)
    (void)lseek(fd, original, SEEK_SET);
  if (end <= 0)
    return {};
  return afl_dbi::read_fd_bytes(fd, 0, static_cast<uint64_t>(end));
}

hsa_status_t create_hsa_memory_reader_from_bytes(std::vector<uint8_t> patched,
                                                 hsa_code_object_reader_t *reader,
                                                 const char *context) {
  if (real_hsa_code_object_reader_create_from_memory == nullptr)
    return HSA_STATUS_ERROR_NOT_INITIALIZED;
  auto owned = std::make_unique<std::vector<uint8_t>>(std::move(patched));
  const void *data = owned->data();
  const size_t size = owned->size();
  hsa_status_t status = HSA_STATUS_SUCCESS;
  {
    ScopedInterceptionBypass bypass;
    status = real_hsa_code_object_reader_create_from_memory(data, size, reader);
  }
  if (status == HSA_STATUS_SUCCESS && reader != nullptr) {
    std::lock_guard<std::mutex> lock(g_state_mutex);
    g_hsa_reader_images[reader->handle] = std::move(owned);
    if (g_verbose) {
      fprintf(stderr, "rocjitsu-afl: created patched HSA code-object reader %llu for %s\n",
              static_cast<unsigned long long>(reader->handle), context);
    }
  }
  return status;
}

PatchedCodeObjectImage patch_hsa_code_object(std::span<const uint8_t> image,
                                             const char *context) {
  if (image.empty())
    return {};

  std::vector<afl_dbi::DeviceImage> device_images = afl_dbi::extract_device_images(image);
  const afl_dbi::CodeObjectImageSummary summary = afl_dbi::summarize_code_object_image(image);
  emit_code_object_image_report(context, image.size(), summary);
  if (device_images.empty())
    return {};

  std::vector<afl_dbi::DeviceImage> candidates;
  for (afl_dbi::DeviceImage &device_image : device_images) {
    if (device_image_has_candidate_kernel(device_image.bytes))
      candidates.push_back(std::move(device_image));
  }
  if (candidates.empty())
    return {};

  const uint64_t state_pointer = device_state_pointer_for_patching();
  if (state_pointer == 0)
    return {};

  return patch_selected_device_images(image, state_pointer, context, summary,
                                      std::move(candidates),
                                      CcobPatchPolicy::RebuildPreservingSiblings);
}

hsa_status_t create_hsa_memory_reader_from_patched(PatchedCodeObjectImage patched,
                                                   hsa_code_object_reader_t *reader,
                                                   const char *context) {
  if (patched.empty())
    return HSA_STATUS_ERROR_INVALID_ARGUMENT;

  const bool fallback_available = !patched.raw_elf_fallback.empty();
  const size_t primary_bytes = patched.image.size();
  const size_t fallback_bytes = patched.raw_elf_fallback.size();
  hsa_status_t status =
      create_hsa_memory_reader_from_bytes(std::move(patched.image), reader, context);
  const uint64_t primary_reader_handle =
      (status == HSA_STATUS_SUCCESS && reader != nullptr) ? reader->handle : 0;
  if (status == HSA_STATUS_SUCCESS || !fallback_available) {
    emit_hsa_reader_patch_report(context, patched.device_image_id,
                                 patched.device_image_index, patched.rebuilt_container,
                                 fallback_available, /*fallback_used=*/false, status, status,
                                 primary_reader_handle, primary_reader_handle,
                                 /*fallback_reader_handle=*/0, primary_bytes, fallback_bytes);
    return status;
  }

  if (g_verbose) {
    fprintf(stderr,
            "rocjitsu-afl: HSA reader rejected rebuilt code-object container for %s in %s "
            "(status=%u); retrying raw patched ELF fallback\n",
            patched.device_image_id.c_str(), context, static_cast<unsigned>(status));
  }
  hsa_status_t fallback_status = create_hsa_memory_reader_from_bytes(
      std::move(patched.raw_elf_fallback), reader, context);
  const uint64_t fallback_reader_handle =
      (fallback_status == HSA_STATUS_SUCCESS && reader != nullptr) ? reader->handle : 0;
  emit_hsa_reader_patch_report(context, patched.device_image_id, patched.device_image_index,
                               patched.rebuilt_container, fallback_available,
                               /*fallback_used=*/true, status, fallback_status,
                               fallback_reader_handle, primary_reader_handle,
                               fallback_reader_handle, primary_bytes, fallback_bytes);
  return fallback_status;
}

std::optional<LazyPatchedFunction> runtime_shadow_function(std::string_view kernel_name,
                                                           uintptr_t registration_key) {
  if (!g_runtime_shadow_modules || kernel_name.empty() || kernel_name == "<unknown>")
    return std::nullopt;
  if (real_hipModuleLoadData == nullptr || real_hipModuleGetFunction == nullptr) {
    emit_runtime_shadow_miss_report(kernel_name, registration_key, "missing_hip_module_api",
                                    /*source_count=*/0, /*attempted_sources=*/0,
                                    /*patch_failures=*/0, /*load_failures=*/0);
    return std::nullopt;
  }

  const std::string shadow_key = runtime_shadow_function_key(registration_key, kernel_name);
  LazyPatchedFunction cached;
  RuntimeShadowFunction cached_metadata;
  bool cache_hit = false;
  size_t cached_total_functions = 0;
  {
    std::lock_guard<std::mutex> lock(g_state_mutex);
    auto it = runtime_shadow_functions().find(shadow_key);
    if (it != runtime_shadow_functions().end()) {
      cached_metadata = it->second;
      cached = cached_metadata.target;
      cache_hit = true;
      cached_total_functions = runtime_shadow_functions().size();
    }
  }
  if (cache_hit) {
    emit_runtime_shadow_function_report(kernel_name, registration_key, cached, nullptr,
                                        /*cache_hit=*/true, /*inserted=*/false,
                                        /*duplicate_unloaded=*/false, cached_total_functions,
                                        cached_metadata.source_count, /*attempted_sources=*/0,
                                        cached_metadata.source_dedupe_key, /*patch_us=*/0,
                                        /*load_us=*/0, /*publish_us=*/0,
                                        cached_metadata.matching_sources);
    return cached;
  }

  std::vector<RuntimeShadowSourceSnapshot> sources =
      runtime_shadow_sources_snapshot(registration_key);
  if (sources.empty()) {
    emit_runtime_shadow_miss_report(kernel_name, registration_key, "no_cached_source",
                                    /*source_count=*/0, /*attempted_sources=*/0,
                                    /*patch_failures=*/0, /*load_failures=*/0);
    return std::nullopt;
  }

  std::vector<RuntimeShadowSourceSnapshot> matching_sources;
  for (const RuntimeShadowSourceSnapshot &source : sources) {
    if (runtime_shadow_source_has_kernel(source, kernel_name))
      matching_sources.push_back(source);
  }
  if (matching_sources.empty()) {
    emit_runtime_shadow_miss_report(kernel_name, registration_key, "no_matching_source",
                                    sources.size(), /*attempted_sources=*/0,
                                    /*patch_failures=*/0, /*load_failures=*/0,
                                    /*matching_sources=*/0);
    return std::nullopt;
  }

  const uint64_t state_pointer = device_state_pointer_for_patching();
  if (state_pointer == 0) {
    emit_runtime_shadow_miss_report(kernel_name, registration_key, "no_device_state",
                                    sources.size(), /*attempted_sources=*/0,
                                    /*patch_failures=*/0, /*load_failures=*/0,
                                    matching_sources.size());
    return std::nullopt;
  }

  std::string kernel_name_string(kernel_name);
  size_t attempted_sources = 0;
  size_t patch_failures = 0;
  size_t load_failures = 0;
  for (const RuntimeShadowSourceSnapshot &source : matching_sources) {
    if (!source.image || source.image->empty())
      continue;
    ++attempted_sources;

    std::vector<uint8_t> patched;
    const auto patch_start = Clock::now();
    {
      ScopedKernelIncludeOverride include(kernel_name_string.c_str());
      ScopedRuntimeShadowRegistrationOverride registration(registration_key);
      patched = patch_device_elf(*source.image, state_pointer, "runtime KPACK shadow",
                                 kernel_name_string.c_str());
    }
    const uint64_t patch_us = elapsed_us(patch_start, Clock::now());
    if (patched.empty()) {
      ++patch_failures;
      continue;
    }

    auto owned = std::make_unique<std::vector<uint8_t>>(std::move(patched));
    const void *data = owned->data();
    hipModule_t module = nullptr;
    hipFunction_t function = nullptr;
    hipError_t err = hipSuccess;
    const auto load_start = Clock::now();
    {
      ScopedInterceptionBypass bypass;
      err = real_hipModuleLoadData(&module, data);
      if (err == hipSuccess)
        err = real_hipModuleGetFunction(&function, module, kernel_name_string.c_str());
    }
    const uint64_t load_us = elapsed_us(load_start, Clock::now());
    if (err != hipSuccess || module == nullptr || function == nullptr) {
      ++load_failures;
      if (g_verbose) {
        fprintf(stderr,
                "rocjitsu-afl: runtime shadow patch for %s failed after load err=%d\n",
                kernel_name_string.c_str(), static_cast<int>(err));
      }
      unload_shadow_modules({module}, "failed runtime KPACK");
      continue;
    }

    LazyPatchedFunction shadow{module, function};
    LazyPatchedFunction published = shadow;
    bool inserted = false;
    size_t total_functions = 0;
    size_t published_source_count = sources.size();
    size_t published_matching_sources = matching_sources.size();
    uint64_t published_source_dedupe_key = source.dedupe_key;
    if (const uint64_t delay_us = runtime_shadow_race_delay_us(); delay_us != 0)
      usleep(static_cast<useconds_t>(delay_us));
    const auto publish_start = Clock::now();
    {
      std::lock_guard<std::mutex> lock(g_state_mutex);
      // Another thread may have published the same registration+kernel shadow
      // while this thread was patching and loading the duplicate module.
      auto existing = runtime_shadow_functions().find(shadow_key);
      if (existing != runtime_shadow_functions().end()) {
        published = existing->second.target;
        published_source_count = existing->second.source_count;
        published_matching_sources = existing->second.matching_sources;
        published_source_dedupe_key = existing->second.source_dedupe_key;
      } else {
        g_patched_images.push_back(std::move(owned));
        runtime_shadow_functions().emplace(
            shadow_key, RuntimeShadowFunction{registration_key, shadow, sources.size(),
                                              matching_sources.size(), source.dedupe_key});
        g_module_function_names[function_key(function)] = kernel_name_string;
        inserted = true;
      }
      if (g_verbose) {
        fprintf(stderr,
                "rocjitsu-afl: cached runtime shadow function registration=%p "
                "inserted=%d total_functions=%zu\n",
                reinterpret_cast<void *>(registration_key), inserted ? 1 : 0,
                runtime_shadow_functions().size());
      }
      total_functions = runtime_shadow_functions().size();
    }
    const uint64_t publish_us = elapsed_us(publish_start, Clock::now());
    if (!inserted) {
      unload_shadow_modules({module}, "duplicate runtime KPACK");
      emit_runtime_shadow_function_report(kernel_name_string, registration_key, published, module,
                                          /*cache_hit=*/false, /*inserted=*/false,
                                          /*duplicate_unloaded=*/true, total_functions,
                                          published_source_count, attempted_sources,
                                          published_source_dedupe_key, patch_us, load_us,
                                          publish_us, published_matching_sources);
      if (g_verbose) {
        fprintf(stderr,
                "rocjitsu-afl: discarded duplicate runtime shadow module=%p "
                "registration=%p name=%s\n",
                reinterpret_cast<void *>(module), reinterpret_cast<void *>(registration_key),
                kernel_name_string.c_str());
      }
      return published;
    }
    emit_runtime_shadow_function_report(kernel_name_string, registration_key, published, module,
                                        /*cache_hit=*/false, /*inserted=*/true,
                                        /*duplicate_unloaded=*/false, total_functions,
                                        published_source_count, attempted_sources,
                                        published_source_dedupe_key, patch_us, load_us,
                                        publish_us, published_matching_sources);
    if (g_verbose) {
      fprintf(stderr,
              "rocjitsu-afl: created runtime shadow module=%p function=%p "
              "registration=%p name=%s\n",
              reinterpret_cast<void *>(module), reinterpret_cast<void *>(function),
              reinterpret_cast<void *>(registration_key), kernel_name_string.c_str());
    }
    return published;
  }

  emit_runtime_shadow_miss_report(kernel_name_string, registration_key, "no_patchable_source",
                                  sources.size(), attempted_sources, patch_failures,
                                  load_failures, matching_sources.size());
  return std::nullopt;
}

std::optional<hipError_t> try_launch_runtime_shadow_kernel(std::string_view kernel_name,
                                                           uintptr_t registration_key,
                                                           const void *runtime_function,
                                                           dim3 num_blocks, dim3 dim_blocks,
                                                           void **args, size_t shared_mem_bytes,
                                                           hipStream_t stream) {
  if (!g_runtime_shadow_modules || real_hipModuleLaunchKernel == nullptr ||
      shared_mem_bytes > std::numeric_limits<unsigned int>::max())
    return std::nullopt;
  std::optional<LazyPatchedFunction> shadow =
      runtime_shadow_function(kernel_name, registration_key);
  if (!shadow)
    return std::nullopt;
  if (g_verbose) {
    fprintf(stderr, "rocjitsu-afl: launching runtime shadow kernel registration=%p %.*s\n",
            reinterpret_cast<void *>(registration_key), static_cast<int>(kernel_name.size()),
            kernel_name.data());
  }
  emit_runtime_shadow_launch_report(kernel_name, registration_key, runtime_function, *shadow,
                                    num_blocks, dim_blocks, shared_mem_bytes);
  record_launch("hipLaunchKernel", "runtime_shadow", nullptr, runtime_function, num_blocks.x,
                num_blocks.y, num_blocks.z, dim_blocks.x, dim_blocks.y, dim_blocks.z,
                kernel_name, registration_key);
  ScopedInterceptionBypass bypass;
  return real_hipModuleLaunchKernel(
      shadow->function, num_blocks.x, num_blocks.y, num_blocks.z, dim_blocks.x, dim_blocks.y,
      dim_blocks.z, static_cast<unsigned int>(shared_mem_bytes), stream, args, nullptr);
}

ShadowModuleUnloadStats unload_shadow_modules(std::vector<hipModule_t> modules,
                                              const char *context) {
  ShadowModuleUnloadStats stats;
  if (real_hipModuleUnload == nullptr || modules.empty())
    return stats;
  std::unordered_set<uintptr_t> seen;
  for (hipModule_t module : modules) {
    if (module == nullptr || !seen.insert(module_key(module)).second)
      continue;
    hipError_t err = hipSuccess;
    {
      ScopedInterceptionBypass bypass;
      err = real_hipModuleUnload(module);
    }
    ++stats.attempts;
    if (err != hipSuccess) {
      ++stats.failures;
      stats.last_error = err;
    }
    if (g_verbose) {
      fprintf(stderr, "rocjitsu-afl: unloaded %s shadow module=%p err=%d\n", context,
              reinterpret_cast<void *>(module), static_cast<int>(err));
    }
  }
  return stats;
}

void release_lazy_ccob_module(hipModule_t module) {
  if (module == nullptr)
    return;
  std::vector<hipModule_t> shadow_modules;
  {
    std::lock_guard<std::mutex> lock(g_state_mutex);
    auto it = g_lazy_ccob_modules.find(module_key(module));
    if (it == g_lazy_ccob_modules.end())
      return;
    for (const auto &[name, target] : it->second.functions) {
      (void)name;
      if (target.function != nullptr)
        g_module_function_names.erase(function_key(target.function));
      if (target.module != nullptr)
        shadow_modules.push_back(target.module);
    }
    g_lazy_ccob_modules.erase(it);
  }
  const size_t shadow_module_count = shadow_modules.size();
  const ShadowModuleUnloadStats unload_stats =
      unload_shadow_modules(std::move(shadow_modules), "lazy CCOB");
  emit_lazy_ccob_release_report(module, shadow_module_count, unload_stats);
}

void release_all_lazy_ccob_modules(const char *context) {
  std::vector<hipModule_t> modules;
  {
    std::lock_guard<std::mutex> lock(g_state_mutex);
    modules.reserve(g_lazy_ccob_modules.size());
    for (const auto &[module_key_value, lazy] : g_lazy_ccob_modules) {
      (void)lazy;
      modules.push_back(reinterpret_cast<hipModule_t>(module_key_value));
    }
  }
  for (hipModule_t module : modules)
    release_lazy_ccob_module(module);
  if (g_verbose && !modules.empty()) {
    fprintf(stderr, "rocjitsu-afl: released %zu lazy CCOB modules during %s\n",
            modules.size(), context != nullptr ? context : "cleanup");
  }
}

void release_runtime_registration_key(uintptr_t registration_key, const char *context) {
  if (registration_key == 0)
    return;

  {
    std::lock_guard<std::mutex> lock(runtime_function_names_mutex());
    auto &names = runtime_function_names();
    auto &registrations = runtime_function_registrations();
    for (auto it = registrations.begin(); it != registrations.end();) {
      if (it->second != registration_key) {
        ++it;
        continue;
      }
      names.erase(it->first);
      it = registrations.erase(it);
    }
  }

  std::vector<hipModule_t> shadow_modules;
  size_t removed_sources = 0;
  size_t removed_functions = 0;
  size_t total_sources = 0;
  size_t total_functions = 0;
  {
    std::lock_guard<std::mutex> lock(g_state_mutex);
    total_sources = runtime_shadow_sources().size();
    total_functions = runtime_shadow_functions().size();
    runtime_shadow_sources().erase(
        std::remove_if(runtime_shadow_sources().begin(), runtime_shadow_sources().end(),
                       [&](const RuntimeShadowSource &source) {
                         if (source.registration_key != registration_key)
                           return false;
                         runtime_shadow_source_hashes().erase(source.dedupe_key);
                         ++removed_sources;
                         return true;
                       }),
        runtime_shadow_sources().end());

    for (auto it = runtime_shadow_functions().begin(); it != runtime_shadow_functions().end();) {
      if (it->second.registration_key != registration_key) {
        ++it;
        continue;
      }
      if (it->second.target.function != nullptr)
        g_module_function_names.erase(function_key(it->second.target.function));
      if (it->second.target.module != nullptr)
        shadow_modules.push_back(it->second.target.module);
      ++removed_functions;
      it = runtime_shadow_functions().erase(it);
    }
  }

  const size_t shadow_module_count = shadow_modules.size();
  const ShadowModuleUnloadStats unload_stats =
      unload_shadow_modules(std::move(shadow_modules), "runtime KPACK");
  emit_runtime_shadow_release_report(registration_key, context, removed_sources,
                                     removed_functions, total_sources, total_functions,
                                     shadow_module_count,
                                     unload_stats);
  if (g_verbose) {
    fprintf(stderr,
            "rocjitsu-afl: released runtime registration=%p during %s shadow_sources=%zu "
            "shadow_functions=%zu total_sources=%zu total_functions=%zu\n",
            reinterpret_cast<void *>(registration_key), context != nullptr ? context : "cleanup",
            removed_sources, removed_functions,
            total_sources, total_functions);
  }
}

void release_runtime_registration(void **modules, const char *context) {
  release_runtime_registration_key(runtime_registration_key(modules), context);
}

void release_all_runtime_registrations(const char *context) {
  std::unordered_set<uintptr_t> registrations;
  {
    std::lock_guard<std::mutex> lock(g_state_mutex);
    for (const RuntimeShadowSource &source : runtime_shadow_sources()) {
      if (source.registration_key != 0)
        registrations.insert(source.registration_key);
    }
    for (const auto &[key, function] : runtime_shadow_functions()) {
      (void)key;
      if (function.registration_key != 0)
        registrations.insert(function.registration_key);
    }
  }
  for (uintptr_t registration_key : registrations)
    release_runtime_registration_key(registration_key, context);
  if (g_verbose && !registrations.empty()) {
    fprintf(stderr, "rocjitsu-afl: released %zu runtime shadow registrations during %s\n",
            registrations.size(), context != nullptr ? context : "cleanup");
  }
}

struct HipFatBinaryWrapper {
  unsigned int magic = 0;
  unsigned int version = 0;
  const void *binary = nullptr;
  const void *dummy1 = nullptr;
};

const char *hip_fatbin_magic_name(unsigned int magic) {
  constexpr unsigned kHipfMagic = 0x48495046; // "HIPF"
  constexpr unsigned kHipkMagic = 0x4b504948; // "HIPK"
  if (magic == kHipfMagic)
    return "HIPF";
  if (magic == kHipkMagic)
    return "HIPK";
  return "unknown";
}

void log_hip_fat_binary_registration(const void *data) {
  if (!g_verbose || data == nullptr)
    return;
  const auto *wrapper = reinterpret_cast<const HipFatBinaryWrapper *>(data);
  fprintf(stderr,
          "rocjitsu-afl: __hipRegisterFatBinary magic=%s(0x%x) version=%u "
          "binary=%p reserved=%p\n",
          hip_fatbin_magic_name(wrapper->magic), wrapper->magic, wrapper->version,
          wrapper->binary, wrapper->dummy1);
}

std::string current_device_target_id() {
  int device = 0;
  hipError_t err = hipSuccess;
  {
    ScopedInterceptionBypass bypass;
    err = hipGetDevice(&device);
  }
  if (err != hipSuccess)
    return {};

  hipDeviceProp_t props{};
  {
    ScopedInterceptionBypass bypass;
    err = hipGetDeviceProperties(&props, device);
  }
  if (err != hipSuccess || props.gcnArchName[0] == '\0')
    return {};
  return afl_dbi::normalize_amdgpu_target_id(props.gcnArchName);
}

std::vector<afl_dbi::DeviceImage>
order_device_images_for_current_device(const std::vector<afl_dbi::DeviceImage> &images,
                                       const char *context) {
  const std::string target_id = current_device_target_id();
  if (target_id.empty())
    return images;
  std::vector<afl_dbi::DeviceImage> ordered =
      afl_dbi::order_device_images_for_target(images, target_id);
  if (g_verbose && ordered.size() > 1) {
    fprintf(stderr, "rocjitsu-afl: ordered %zu device images for %s target=%s first=%s\n",
            ordered.size(), context, target_id.c_str(), ordered.front().id.c_str());
  }
  return ordered;
}

bool amd_loader_table_has_file_offset_reader(size_t table_length) {
  constexpr size_t reader_offset = offsetof(
      hsa_ven_amd_loader_1_02_pfn_t,
      hsa_ven_amd_loader_code_object_reader_create_from_file_with_offset_size);
  return table_length >= reader_offset + sizeof(hsaReaderCreateFileOffsetSize_t);
}

void patch_amd_loader_extension_table(uint16_t extension, uint16_t version_major,
                                      size_t table_length, void *table, const char *source) {
  if (extension != HSA_EXTENSION_AMD_LOADER || version_major != 1 || table == nullptr ||
      !amd_loader_table_has_file_offset_reader(table_length)) {
    return;
  }

  auto *loader = static_cast<hsa_ven_amd_loader_1_02_pfn_t *>(table);
  hsaReaderCreateFileOffsetSize_t original =
      loader->hsa_ven_amd_loader_code_object_reader_create_from_file_with_offset_size;
  if (original == nullptr)
    return;

  if (original != hsa_ven_amd_loader_code_object_reader_create_from_file_with_offset_size)
    real_hsa_ven_amd_loader_code_object_reader_create_from_file_with_offset_size = original;
  loader->hsa_ven_amd_loader_code_object_reader_create_from_file_with_offset_size =
      hsa_ven_amd_loader_code_object_reader_create_from_file_with_offset_size;

  if (g_verbose) {
    fprintf(stderr,
            "rocjitsu-afl: patched AMD loader extension table from %s "
            "for file-offset code-object reader\n",
            source);
  }
}

size_t deprecated_amd_loader_table_length(uint16_t version_minor) {
  if (version_minor >= 3)
    return sizeof(hsa_ven_amd_loader_1_03_pfn_t);
  if (version_minor >= 2)
    return sizeof(hsa_ven_amd_loader_1_02_pfn_t);
  return 0;
}

hipError_t merge_after_successful_sync(const char *trigger, hipError_t err) {
  if (err == hipSuccess && !g_persistent_mode) {
    std::lock_guard<std::mutex> lock(g_state_mutex);
    if (!merge_coverage_locked(trigger))
      return hipErrorUnknown;
  }
  return err;
}

} // namespace

extern "C" {

hsa_status_t hsa_system_get_extension_table(uint16_t extension, uint16_t version_major,
                                            uint16_t version_minor, void *table) {
  resolve_symbols();
  auto real = real_hsa_system_get_extension_table;
  if (real == nullptr)
    return HSA_STATUS_ERROR_NOT_INITIALIZED;

  hsa_status_t status = real(extension, version_major, version_minor, table);
  if (status == HSA_STATUS_SUCCESS) {
    patch_amd_loader_extension_table(extension, version_major,
                                     deprecated_amd_loader_table_length(version_minor), table,
                                     "hsa_system_get_extension_table");
  }
  return status;
}

hsa_status_t hsa_system_get_major_extension_table(uint16_t extension, uint16_t version_major,
                                                  size_t table_length, void *table) {
  resolve_symbols();
  auto real = real_hsa_system_get_major_extension_table;
  if (real == nullptr)
    return HSA_STATUS_ERROR_NOT_INITIALIZED;

  hsa_status_t status = real(extension, version_major, table_length, table);
  if (status == HSA_STATUS_SUCCESS) {
    patch_amd_loader_extension_table(extension, version_major, table_length, table,
                                     "hsa_system_get_major_extension_table");
  }
  return status;
}

void **__hipRegisterFatBinary(const void *data) {
  resolve_symbols();
  log_hip_fat_binary_registration(data);
  if (real___hipRegisterFatBinary == nullptr)
    return nullptr;
  return real___hipRegisterFatBinary(data);
}

void __hipUnregisterFatBinary(void **modules) {
  resolve_symbols();
  if (g_verbose)
    fprintf(stderr, "rocjitsu-afl: __hipUnregisterFatBinary modules=%p\n", modules);
  if (getenv("ROCFUZZ_SKIP_RUNTIME_UNREGISTER_RELEASE") == nullptr) {
    release_runtime_registration(modules, "__hipUnregisterFatBinary");
  } else if (g_verbose) {
    fprintf(stderr,
            "rocjitsu-afl: skipping runtime registration release at unregister "
            "for modules=%p\n",
            modules);
  }
  if (real___hipUnregisterFatBinary != nullptr)
    real___hipUnregisterFatBinary(modules);
}

void __hipRegisterFunction(void **modules, const void *hostFunction, char *deviceFunction,
                           const char *deviceName, unsigned int threadLimit, uint3 *tid,
                           uint3 *bid, dim3 *blockDim, dim3 *gridDim, int *wSize) {
  resolve_symbols();
  const char *registered_name =
      deviceName != nullptr && deviceName[0] != '\0'
          ? deviceName
          : (deviceFunction != nullptr && deviceFunction[0] != '\0' ? deviceFunction : nullptr);
  if (hostFunction != nullptr && registered_name != nullptr) {
    std::lock_guard<std::mutex> lock(runtime_function_names_mutex());
    const uintptr_t key = runtime_function_key(hostFunction);
    runtime_function_names()[key] = registered_name;
    runtime_function_registrations()[key] = runtime_registration_key(modules);
  }
  if (g_verbose) {
    fprintf(stderr,
            "rocjitsu-afl: __hipRegisterFunction host=%p device=%s device_name=%s modules=%p\n",
            hostFunction, deviceFunction != nullptr ? deviceFunction : "<null>",
            deviceName != nullptr ? deviceName : "<null>", modules);
  }
  if (real___hipRegisterFunction != nullptr) {
    real___hipRegisterFunction(modules, hostFunction, deviceFunction, deviceName, threadLimit,
                               tid, bid, blockDim, gridDim, wSize);
  }
}

hsa_status_t hsa_code_object_reader_create_from_memory(const void *code_object, size_t size,
                                                       hsa_code_object_reader_t *reader) {
  resolve_symbols();
  if (real_hsa_code_object_reader_create_from_memory == nullptr)
    return HSA_STATUS_ERROR_NOT_INITIALIZED;
  if (interception_bypassed() || code_object == nullptr || size == 0)
    return real_hsa_code_object_reader_create_from_memory(code_object, size, reader);

  const auto *base = static_cast<const uint8_t *>(code_object);
  PatchedCodeObjectImage patched = patch_hsa_code_object({base, size}, "HSA memory reader");
  if (!patched.empty())
    return create_hsa_memory_reader_from_patched(std::move(patched), reader, "HSA memory reader");

  return real_hsa_code_object_reader_create_from_memory(code_object, size, reader);
}

hsa_status_t hsa_code_object_reader_create_from_file(hsa_file_t file,
                                                     hsa_code_object_reader_t *reader) {
  resolve_symbols();
  if (real_hsa_code_object_reader_create_from_file == nullptr)
    return HSA_STATUS_ERROR_NOT_INITIALIZED;
  if (interception_bypassed())
    return real_hsa_code_object_reader_create_from_file(file, reader);

  std::vector<uint8_t> image = read_whole_fd(file);
  PatchedCodeObjectImage patched = patch_hsa_code_object(image, "HSA file reader");
  if (!patched.empty())
    return create_hsa_memory_reader_from_patched(std::move(patched), reader, "HSA file reader");

  return real_hsa_code_object_reader_create_from_file(file, reader);
}

hsa_status_t hsa_ven_amd_loader_code_object_reader_create_from_file_with_offset_size(
    hsa_file_t file, size_t offset, size_t size, hsa_code_object_reader_t *reader) {
  resolve_symbols();
  auto real = real_hsa_ven_amd_loader_code_object_reader_create_from_file_with_offset_size;
  if (real == nullptr)
    return HSA_STATUS_ERROR_NOT_INITIALIZED;
  if (interception_bypassed())
    return real(file, offset, size, reader);

  std::vector<uint8_t> image =
      afl_dbi::read_fd_bytes(file, static_cast<uint64_t>(offset), static_cast<uint64_t>(size));
  PatchedCodeObjectImage patched = patch_hsa_code_object(image, "HSA file-offset reader");
  if (!patched.empty()) {
    return create_hsa_memory_reader_from_patched(std::move(patched), reader,
                                                 "HSA file-offset reader");
  }

  return real(file, offset, size, reader);
}

hsa_status_t hsa_code_object_reader_destroy(hsa_code_object_reader_t reader) {
  resolve_symbols();
  if (real_hsa_code_object_reader_destroy == nullptr)
    return HSA_STATUS_ERROR_NOT_INITIALIZED;
  {
    std::lock_guard<std::mutex> lock(g_state_mutex);
    g_hsa_reader_images.erase(reader.handle);
  }
  return real_hsa_code_object_reader_destroy(reader);
}

hipError_t hipModuleLoad(hipModule_t *module, const char *fname) {
  resolve_symbols();
  if (real_hipModuleLoad == nullptr)
    return hipErrorSharedObjectInitFailed;
  if (interception_bypassed())
    return real_hipModuleLoad(module, fname);

  if (fname != nullptr && real_hipModuleLoadData != nullptr) {
    uint64_t state_pointer = 0;
    {
      std::lock_guard<std::mutex> lock(g_state_mutex);
      if (ensure_runtime_locked())
        state_pointer = reinterpret_cast<uint64_t>(g_device_counters);
    }
    std::vector<uint8_t> image = afl_dbi::read_file_bytes(fname);
    if (state_pointer != 0 && should_defer_ccob_module_patch(image)) {
      hipError_t err = hipSuccess;
      {
        ScopedInterceptionBypass bypass;
        err = real_hipModuleLoad(module, fname);
      }
      if (err == hipSuccess && module != nullptr) {
        remember_lazy_ccob_module(*module, std::move(image), fname);
        return err;
      }
      if (g_verbose) {
        fprintf(stderr,
                "rocjitsu-afl: original CCOB hipModuleLoad(%s) failed err=%d; "
                "trying eager raw-ELF patch\n",
                fname, static_cast<int>(err));
      }
    }
    std::vector<uint8_t> patched =
        state_pointer == 0 || image.empty() ? std::vector<uint8_t>{}
                                            : patch_module_image(image, state_pointer, fname);
    if (!patched.empty()) {
      auto owned = std::make_unique<std::vector<uint8_t>>(std::move(patched));
      const void *data = owned->data();
      if (g_verbose)
        fprintf(stderr, "rocjitsu-afl: loading patched raw device ELF for %s\n", fname);
      hipError_t err = hipSuccess;
      {
        ScopedInterceptionBypass bypass;
        err = real_hipModuleLoadData(module, data);
      }
      if (err == hipSuccess && module != nullptr) {
        std::lock_guard<std::mutex> lock(g_state_mutex);
        g_patched_images.push_back(std::move(owned));
      }
      return err;
    }
  }

  if (g_verbose)
    fprintf(stderr, "rocjitsu-afl: falling back to unpatched hipModuleLoad(%s)\n",
            fname ? fname : "<null>");
  return real_hipModuleLoad(module, fname);
}

hipError_t hipModuleLoadData(hipModule_t *module, const void *image) {
  resolve_symbols();
  if (real_hipModuleLoadData == nullptr)
    return hipErrorSharedObjectInitFailed;
  if (interception_bypassed())
    return real_hipModuleLoadData(module, image);

  uint64_t state_pointer = 0;
  {
    std::lock_guard<std::mutex> lock(g_state_mutex);
    if (ensure_runtime_locked())
      state_pointer = reinterpret_cast<uint64_t>(g_device_counters);
  }

  std::vector<uint8_t> copied = afl_dbi::copy_module_data_image(image);
  if (state_pointer != 0 && should_defer_ccob_module_patch(copied)) {
    hipError_t err = hipSuccess;
    {
      ScopedInterceptionBypass bypass;
      err = real_hipModuleLoadData(module, image);
    }
    if (err == hipSuccess && module != nullptr) {
      remember_lazy_ccob_module(*module, std::move(copied), "in-memory CCOB");
      return err;
    }
    if (g_verbose) {
      fprintf(stderr,
              "rocjitsu-afl: original CCOB hipModuleLoadData(%p) failed err=%d; "
              "trying eager raw-ELF patch\n",
              image, static_cast<int>(err));
    }
  }

  std::vector<uint8_t> patched =
      state_pointer == 0 || copied.empty() ? std::vector<uint8_t>{}
                                           : patch_module_image(copied, state_pointer,
                                                                "in-memory code object");
  if (!patched.empty()) {
    auto owned = std::make_unique<std::vector<uint8_t>>(std::move(patched));
    const void *data = owned->data();
    if (g_verbose)
      fprintf(stderr, "rocjitsu-afl: loading patched in-memory device ELF\n");
    hipError_t err = hipSuccess;
    {
      ScopedInterceptionBypass bypass;
      err = real_hipModuleLoadData(module, data);
    }
    if (err == hipSuccess && module != nullptr) {
      std::lock_guard<std::mutex> lock(g_state_mutex);
      g_patched_images.push_back(std::move(owned));
    }
    return err;
  }

  if (g_verbose)
    fprintf(stderr, "rocjitsu-afl: falling back to unpatched hipModuleLoadData(%p)\n", image);
  return real_hipModuleLoadData(module, image);
}

hipError_t hipModuleLoadFatBinary(hipModule_t *module, const void *fatbin) {
  resolve_symbols();
  if (real_hipModuleLoadFatBinary == nullptr)
    return hipErrorSharedObjectInitFailed;
  if (interception_bypassed())
    return real_hipModuleLoadFatBinary(module, fatbin);

  uint64_t state_pointer = 0;
  {
    std::lock_guard<std::mutex> lock(g_state_mutex);
    if (ensure_runtime_locked())
      state_pointer = reinterpret_cast<uint64_t>(g_device_counters);
  }

  std::vector<uint8_t> patched =
      state_pointer == 0 ? std::vector<uint8_t>{} : patch_module_data(fatbin, state_pointer);
  if (!patched.empty() && real_hipModuleLoadData != nullptr) {
    auto owned = std::make_unique<std::vector<uint8_t>>(std::move(patched));
    const void *data = owned->data();
    if (g_verbose)
      fprintf(stderr, "rocjitsu-afl: loading patched fat binary as raw device ELF\n");
    hipError_t err = hipSuccess;
    {
      ScopedInterceptionBypass bypass;
      err = real_hipModuleLoadData(module, data);
    }
    if (err == hipSuccess && module != nullptr) {
      std::lock_guard<std::mutex> lock(g_state_mutex);
      g_patched_images.push_back(std::move(owned));
    }
    return err;
  }

  if (g_verbose)
    fprintf(stderr, "rocjitsu-afl: falling back to unpatched hipModuleLoadFatBinary(%p)\n",
            fatbin);
  return real_hipModuleLoadFatBinary(module, fatbin);
}

hipError_t hipModuleLoadDataEx(hipModule_t *module, const void *image, unsigned int numOptions,
                               hipJitOption *options, void **optionsValues) {
  resolve_symbols();
  if (real_hipModuleLoadDataEx == nullptr)
    return hipErrorSharedObjectInitFailed;
  if (interception_bypassed())
    return real_hipModuleLoadDataEx(module, image, numOptions, options, optionsValues);

  uint64_t state_pointer = 0;
  {
    std::lock_guard<std::mutex> lock(g_state_mutex);
    if (ensure_runtime_locked())
      state_pointer = reinterpret_cast<uint64_t>(g_device_counters);
  }

  std::vector<uint8_t> copied = afl_dbi::copy_module_data_image(image);
  if (state_pointer != 0 && should_defer_ccob_module_patch(copied)) {
    hipError_t err = hipSuccess;
    {
      ScopedInterceptionBypass bypass;
      err = real_hipModuleLoadDataEx(module, image, numOptions, options, optionsValues);
    }
    if (err == hipSuccess && module != nullptr) {
      remember_lazy_ccob_module(*module, std::move(copied), "in-memory CCOB via LoadDataEx");
      return err;
    }
    if (g_verbose) {
      fprintf(stderr,
              "rocjitsu-afl: original CCOB hipModuleLoadDataEx(%p) failed err=%d; "
              "trying eager raw-ELF patch\n",
              image, static_cast<int>(err));
    }
  }

  std::vector<uint8_t> patched =
      state_pointer == 0 || copied.empty() ? std::vector<uint8_t>{}
                                           : patch_module_image(copied, state_pointer,
                                                                "in-memory code object");
  if (!patched.empty()) {
    auto owned = std::make_unique<std::vector<uint8_t>>(std::move(patched));
    const void *data = owned->data();
    if (g_verbose)
      fprintf(stderr, "rocjitsu-afl: loading patched in-memory device ELF via LoadDataEx\n");
    hipError_t err = hipSuccess;
    {
      ScopedInterceptionBypass bypass;
      err = real_hipModuleLoadDataEx(module, data, numOptions, options, optionsValues);
    }
    if (err == hipSuccess && module != nullptr) {
      std::lock_guard<std::mutex> lock(g_state_mutex);
      g_patched_images.push_back(std::move(owned));
    }
    return err;
  }

  if (g_verbose)
    fprintf(stderr, "rocjitsu-afl: falling back to unpatched hipModuleLoadDataEx(%p)\n", image);
  return real_hipModuleLoadDataEx(module, image, numOptions, options, optionsValues);
}

hipError_t hipModuleUnload(hipModule_t module) {
  resolve_symbols();
  if (real_hipModuleUnload == nullptr)
    return hipErrorSharedObjectInitFailed;
  if (!interception_bypassed())
    release_lazy_ccob_module(module);
  return real_hipModuleUnload(module);
}

hipError_t hipModuleGetFunction(hipFunction_t *function, hipModule_t module, const char *kname) {
  resolve_symbols();
  if (real_hipModuleGetFunction == nullptr)
    return hipErrorSharedObjectInitFailed;

  if (!interception_bypassed()) {
    if (std::optional<hipError_t> lazy = try_get_lazy_ccob_function(function, module, kname)) {
      if (*lazy == hipSuccess && function != nullptr && *function != nullptr) {
        {
          std::lock_guard<std::mutex> lock(g_state_mutex);
          g_module_function_names[function_key(*function)] = kname;
        }
        if (g_verbose) {
          fprintf(stderr,
                  "rocjitsu-afl: hipModuleGetFunction module=%p function=%p name=%s "
                  "(lazy CCOB)\n",
                  reinterpret_cast<void *>(module), reinterpret_cast<void *>(*function), kname);
        }
        return hipSuccess;
      }
      if (g_verbose) {
        fprintf(stderr,
                "rocjitsu-afl: lazy CCOB patch failed for module=%p name=%s err=%d; "
                "falling back to original module\n",
                reinterpret_cast<void *>(module), kname != nullptr ? kname : "<null>",
                static_cast<int>(*lazy));
      }
    }
  }

  hipError_t err = real_hipModuleGetFunction(function, module, kname);
  if (err == hipSuccess && function != nullptr && *function != nullptr && kname != nullptr &&
      kname[0] != '\0') {
    {
      std::lock_guard<std::mutex> lock(g_state_mutex);
      g_module_function_names[function_key(*function)] = kname;
    }
    if (g_verbose) {
      fprintf(stderr, "rocjitsu-afl: hipModuleGetFunction module=%p function=%p name=%s\n",
              reinterpret_cast<void *>(module), reinterpret_cast<void *>(*function), kname);
    }
  }
  return err;
}

hipError_t hipModuleLaunchKernel(hipFunction_t f, unsigned int gridDimX, unsigned int gridDimY,
                                 unsigned int gridDimZ, unsigned int blockDimX,
                                 unsigned int blockDimY, unsigned int blockDimZ,
                                 unsigned int sharedMemBytes, hipStream_t stream,
                                 void **kernelParams, void **extra) {
  resolve_symbols();
  if (real_hipModuleLaunchKernel == nullptr)
    return hipErrorSharedObjectInitFailed;
  if (interception_bypassed())
    return real_hipModuleLaunchKernel(f, gridDimX, gridDimY, gridDimZ, blockDimX, blockDimY,
                                      blockDimZ, sharedMemBytes, stream, kernelParams, extra);

  std::optional<LazyCcobLaunchBinding> lazy = lazy_ccob_launch_binding(f);
  record_launch("hipModuleLaunchKernel", lazy ? "lazy_ccob_shadow" : "hip_module", f, nullptr,
                gridDimX, gridDimY, gridDimZ, blockDimX, blockDimY, blockDimZ,
                lazy ? std::string_view(lazy->kernel) : std::string_view());
  if (lazy) {
    emit_lazy_ccob_launch_report(*lazy, gridDimX, gridDimY, gridDimZ, blockDimX, blockDimY,
                                 blockDimZ, sharedMemBytes);
  }

  return real_hipModuleLaunchKernel(f, gridDimX, gridDimY, gridDimZ, blockDimX, blockDimY,
                                    blockDimZ, sharedMemBytes, stream, kernelParams, extra);
}

hipError_t hipExtModuleLaunchKernel(hipFunction_t f, uint32_t globalWorkSizeX,
                                    uint32_t globalWorkSizeY, uint32_t globalWorkSizeZ,
                                    uint32_t localWorkSizeX, uint32_t localWorkSizeY,
                                    uint32_t localWorkSizeZ, size_t sharedMemBytes,
                                    hipStream_t hStream, void **kernelParams, void **extra,
                                    hipEvent_t startEvent, hipEvent_t stopEvent, uint32_t flags) {
  resolve_symbols();
  if (real_hipExtModuleLaunchKernel == nullptr)
    return hipErrorSharedObjectInitFailed;
  if (interception_bypassed()) {
    return real_hipExtModuleLaunchKernel(
        f, globalWorkSizeX, globalWorkSizeY, globalWorkSizeZ, localWorkSizeX, localWorkSizeY,
        localWorkSizeZ, sharedMemBytes, hStream, kernelParams, extra, startEvent, stopEvent, flags);
  }

  record_launch("hipExtModuleLaunchKernel", "hip_module", f, nullptr, globalWorkSizeX,
                globalWorkSizeY, globalWorkSizeZ, localWorkSizeX, localWorkSizeY,
                localWorkSizeZ);

  return real_hipExtModuleLaunchKernel(
      f, globalWorkSizeX, globalWorkSizeY, globalWorkSizeZ, localWorkSizeX, localWorkSizeY,
      localWorkSizeZ, sharedMemBytes, hStream, kernelParams, extra, startEvent, stopEvent, flags);
}

hipError_t hipDrvLaunchKernelEx(const HIP_LAUNCH_CONFIG *config, hipFunction_t f, void **params,
                                void **extra) {
  resolve_symbols();
  if (real_hipDrvLaunchKernelEx == nullptr)
    return hipErrorSharedObjectInitFailed;
  if (interception_bypassed())
    return real_hipDrvLaunchKernelEx(config, f, params, extra);

  if (config != nullptr) {
    record_launch("hipDrvLaunchKernelEx", "hip_module", f, nullptr, config->gridDimX,
                  config->gridDimY, config->gridDimZ, config->blockDimX, config->blockDimY,
                  config->blockDimZ);
  } else {
    record_launch("hipDrvLaunchKernelEx", "hip_module", f, nullptr, 0, 0, 0, 0, 0, 0);
  }

  return real_hipDrvLaunchKernelEx(config, f, params, extra);
}

hipError_t hipLaunchKernel(const void *function_address, dim3 numBlocks, dim3 dimBlocks,
                           void **args, size_t sharedMemBytes, hipStream_t stream) {
  resolve_symbols();
  if (real_hipLaunchKernel == nullptr)
    return hipErrorSharedObjectInitFailed;
  if (interception_bypassed())
    return real_hipLaunchKernel(function_address, numBlocks, dimBlocks, args, sharedMemBytes,
                                stream);

  const std::string launch_kernel_name = registered_kernel_name(nullptr, function_address);
  const uintptr_t registration_key = registered_runtime_function_registration(function_address);
  if (should_scope_hsa_reader_to_launch(launch_kernel_name)) {
    if (std::optional<hipError_t> shadow = try_launch_runtime_shadow_kernel(
            launch_kernel_name, registration_key, function_address, numBlocks, dimBlocks, args,
            sharedMemBytes, stream))
      return *shadow;
    if (g_verbose) {
      fprintf(stderr,
              "rocjitsu-afl: scoping HSA reader patching to launched kernel "
              "registration=%p %s\n",
              reinterpret_cast<void *>(registration_key), launch_kernel_name.c_str());
    }
    ScopedKernelIncludeOverride include(launch_kernel_name.c_str());
    ScopedRuntimeShadowRegistrationOverride registration(registration_key);
    record_launch("hipLaunchKernel", "hip_runtime_scoped", nullptr, function_address, numBlocks.x,
                  numBlocks.y, numBlocks.z, dimBlocks.x, dimBlocks.y, dimBlocks.z,
                  launch_kernel_name, registration_key);
    return real_hipLaunchKernel(function_address, numBlocks, dimBlocks, args, sharedMemBytes,
                                stream);
  }

  record_launch("hipLaunchKernel", "hip_runtime", nullptr, function_address, numBlocks.x,
                numBlocks.y, numBlocks.z, dimBlocks.x, dimBlocks.y, dimBlocks.z,
                launch_kernel_name, registration_key);
  return real_hipLaunchKernel(function_address, numBlocks, dimBlocks, args, sharedMemBytes, stream);
}

hipError_t hipDeviceSynchronize() {
  resolve_symbols();
  if (real_hipDeviceSynchronize == nullptr)
    return hipErrorSharedObjectInitFailed;
  if (interception_bypassed())
    return real_hipDeviceSynchronize();
  return merge_after_successful_sync("hipDeviceSynchronize", real_hipDeviceSynchronize());
}

hipError_t hipStreamSynchronize(hipStream_t stream) {
  resolve_symbols();
  if (real_hipStreamSynchronize == nullptr)
    return hipErrorSharedObjectInitFailed;
  if (interception_bypassed())
    return real_hipStreamSynchronize(stream);
  return merge_after_successful_sync("hipStreamSynchronize", real_hipStreamSynchronize(stream));
}

hipError_t hipStreamSynchronize_spt(hipStream_t stream) {
  resolve_symbols();
  if (real_hipStreamSynchronize_spt == nullptr)
    return hipErrorSharedObjectInitFailed;
  if (interception_bypassed())
    return real_hipStreamSynchronize_spt(stream);
  return merge_after_successful_sync("hipStreamSynchronize_spt",
                                     real_hipStreamSynchronize_spt(stream));
}

hipError_t hipEventSynchronize(hipEvent_t event) {
  resolve_symbols();
  if (real_hipEventSynchronize == nullptr)
    return hipErrorSharedObjectInitFailed;
  if (interception_bypassed())
    return real_hipEventSynchronize(event);
  return merge_after_successful_sync("hipEventSynchronize", real_hipEventSynchronize(event));
}

hipError_t hipMemcpy(void *dst, const void *src, size_t size_bytes, hipMemcpyKind kind) {
  resolve_symbols();
  if (real_hipMemcpy == nullptr)
    return hipErrorSharedObjectInitFailed;
  if (interception_bypassed())
    return real_hipMemcpy(dst, src, size_bytes, kind);
  return merge_after_successful_sync("hipMemcpy", real_hipMemcpy(dst, src, size_bytes, kind));
}

int rocjitsu_afl_persistent_begin() {
  resolve_symbols();
  std::lock_guard<std::mutex> lock(g_state_mutex);
  if (!ensure_runtime_locked())
    return 1;
  if (!reset_device_state_locked("persistent coverage state"))
    return 2;
  if (g_verbose)
    fprintf(stderr, "rocjitsu-afl: persistent iteration begin\n");
  return 0;
}

int rocjitsu_afl_persistent_end() {
  resolve_symbols();
  if (real_hipDeviceSynchronize != nullptr) {
    hipError_t err = real_hipDeviceSynchronize();
    if (err != hipSuccess) {
      if (g_verbose) {
        fprintf(stderr, "rocjitsu-afl: persistent sync failed: %s\n", hipGetErrorString(err));
      }
      return static_cast<int>(err);
    }
  }

  std::lock_guard<std::mutex> lock(g_state_mutex);
  if (!merge_coverage_locked("persistent_end"))
    return 5;
  if (g_verbose)
    fprintf(stderr, "rocjitsu-afl: persistent iteration end\n");
  return 0;
}

__attribute__((destructor)) void rocjitsu_afl_fini() {
  resolve_symbols();
  release_all_lazy_ccob_modules("destructor");
  release_all_runtime_registrations("destructor");
  std::lock_guard<std::mutex> lock(g_state_mutex);
  (void)merge_coverage_locked("destructor");
  enforce_required_device_edges_at_exit_locked();
  if (g_device_counters != nullptr) {
    ScopedInterceptionBypass bypass;
    (void)hipFree(g_device_counters);
    g_device_counters = nullptr;
  }
  if (real_hipDeviceSynchronize != nullptr) {
    ScopedInterceptionBypass bypass;
    (void)real_hipDeviceSynchronize();
  }
}

} // extern "C"
