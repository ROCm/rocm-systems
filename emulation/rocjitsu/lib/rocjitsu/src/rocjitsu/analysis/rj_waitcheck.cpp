// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "rocjitsu/analysis/rj_waitcheck.h"

#include "rocjitsu/analysis/waitcheck.h"
#include "rocjitsu/code/amdgpu_code_object.h"

#include <algorithm>
#include <cstring>
#include <exception>
#include <new>
#include <optional>

namespace {

using namespace rocjitsu;

constexpr size_t kOptionsV1Size = offsetof(rj_waitcheck_options_t, user_data) +
                                  sizeof(static_cast<rj_waitcheck_options_t *>(nullptr)->user_data);
constexpr size_t kResultV1Size =
    offsetof(rj_waitcheck_result_t, stopped_early) +
    sizeof(static_cast<rj_waitcheck_result_t *>(nullptr)->stopped_early);

[[nodiscard]] rj_waitcheck_options_t default_options() {
  rj_waitcheck_options_t options{};
  options.struct_size = sizeof(options);
  options.abi_version = ROCJITSU_WAITCHECK_ABI_VERSION;
  return options;
}

[[nodiscard]] rj_status_t read_options(const rj_waitcheck_options_t *options,
                                       rj_waitcheck_options_t &local_options) {
  local_options = default_options();
  if (!options)
    return ROCJITSU_STATUS_SUCCESS;
  if (options->struct_size < kOptionsV1Size ||
      options->abi_version != ROCJITSU_WAITCHECK_ABI_VERSION)
    return ROCJITSU_STATUS_INVALID_ARGUMENT;

  const size_t copy_size = std::min(options->struct_size, sizeof(local_options));
  std::memcpy(&local_options, options, copy_size);
  local_options.struct_size = sizeof(local_options);
  local_options.abi_version = ROCJITSU_WAITCHECK_ABI_VERSION;
  return ROCJITSU_STATUS_SUCCESS;
}

[[nodiscard]] bool result_header_is_valid(const rj_waitcheck_result_t *result) {
  return result && result->struct_size >= kResultV1Size &&
         result->abi_version == ROCJITSU_WAITCHECK_ABI_VERSION;
}

void store_result(const rj_waitcheck_result_t &local_result, rj_waitcheck_result_t *result) {
  const size_t copy_size = std::min(result->struct_size, sizeof(local_result));
  std::memset(result, 0, copy_size);
  std::memcpy(result, &local_result, copy_size);
}

void report_error(const rj_waitcheck_options_t &options, const char *message) noexcept {
  if (!options.error_callback)
    return;
  try {
    options.error_callback(message, options.user_data);
  } catch (...) {
    // Exceptions must never cross the public C ABI.
  }
}

[[nodiscard]] rj_waitcheck_counter_t public_counter(WaitCounterKind counter) {
  switch (counter) {
  case WaitCounterKind::Load:
    return ROCJITSU_WAITCHECK_COUNTER_LOAD;
  case WaitCounterKind::Store:
    return ROCJITSU_WAITCHECK_COUNTER_STORE;
  case WaitCounterKind::Ds:
    return ROCJITSU_WAITCHECK_COUNTER_DS;
  case WaitCounterKind::Km:
    return ROCJITSU_WAITCHECK_COUNTER_KM;
  case WaitCounterKind::Sample:
    return ROCJITSU_WAITCHECK_COUNTER_SAMPLE;
  case WaitCounterKind::Bvh:
    return ROCJITSU_WAITCHECK_COUNTER_BVH;
  case WaitCounterKind::Exp:
    return ROCJITSU_WAITCHECK_COUNTER_EXP;
  case WaitCounterKind::X:
    return ROCJITSU_WAITCHECK_COUNTER_X;
  case WaitCounterKind::Async:
    return ROCJITSU_WAITCHECK_COUNTER_ASYNC;
  case WaitCounterKind::Tensor:
    return ROCJITSU_WAITCHECK_COUNTER_TENSOR;
  case WaitCounterKind::VmVsrc:
    return ROCJITSU_WAITCHECK_COUNTER_VM_VSRC;
  case WaitCounterKind::VaVdst:
    return ROCJITSU_WAITCHECK_COUNTER_VA_VDST;
  case WaitCounterKind::Depctr:
    return ROCJITSU_WAITCHECK_COUNTER_DEPCTR;
  case WaitCounterKind::Count:
    return ROCJITSU_WAITCHECK_COUNTER_INVALID;
  }
  return ROCJITSU_WAITCHECK_COUNTER_INVALID;
}

[[nodiscard]] rj_waitcheck_access_t public_access(WaitcheckAccessKind access) {
  switch (access) {
  case WaitcheckAccessKind::Use:
    return ROCJITSU_WAITCHECK_ACCESS_USE;
  case WaitcheckAccessKind::Def:
    return ROCJITSU_WAITCHECK_ACCESS_DEF;
  case WaitcheckAccessKind::MemoryOrder:
    return ROCJITSU_WAITCHECK_ACCESS_MEMORY_ORDER;
  case WaitcheckAccessKind::ProgramEnd:
    return ROCJITSU_WAITCHECK_ACCESS_PROGRAM_END;
  case WaitcheckAccessKind::ControlTransfer:
    return ROCJITSU_WAITCHECK_ACCESS_CONTROL_TRANSFER;
  }
  return ROCJITSU_WAITCHECK_ACCESS_INVALID;
}

[[nodiscard]] rj_waitcheck_register_class_t public_register_class(RegClass reg_class) {
  switch (reg_class) {
  case RegClass::SGPR:
    return ROCJITSU_WAITCHECK_REGISTER_SGPR;
  case RegClass::VGPR:
    return ROCJITSU_WAITCHECK_REGISTER_VGPR;
  case RegClass::ACC_VGPR:
    return ROCJITSU_WAITCHECK_REGISTER_ACC_VGPR;
  case RegClass::EXEC:
    return ROCJITSU_WAITCHECK_REGISTER_EXEC;
  case RegClass::VCC:
    return ROCJITSU_WAITCHECK_REGISTER_VCC;
  case RegClass::SCC:
    return ROCJITSU_WAITCHECK_REGISTER_SCC;
  case RegClass::M0:
    return ROCJITSU_WAITCHECK_REGISTER_M0;
  case RegClass::FLAT_SCRATCH:
    return ROCJITSU_WAITCHECK_REGISTER_FLAT_SCRATCH;
  case RegClass::TTMP:
    return ROCJITSU_WAITCHECK_REGISTER_TTMP;
  case RegClass::PC:
    return ROCJITSU_WAITCHECK_REGISTER_PC;
  }
  return ROCJITSU_WAITCHECK_REGISTER_INVALID;
}

[[nodiscard]] rj_waitcheck_target_t public_target(rj_code_target_id_t target) {
  switch (target) {
  case ROCJITSU_CODE_TARGET_GFX90A:
    return ROCJITSU_WAITCHECK_TARGET_GFX90A;
  case ROCJITSU_CODE_TARGET_GFX942:
    return ROCJITSU_WAITCHECK_TARGET_GFX942;
  case ROCJITSU_CODE_TARGET_GFX950:
    return ROCJITSU_WAITCHECK_TARGET_GFX950;
  case ROCJITSU_CODE_TARGET_GFX1100:
    return ROCJITSU_WAITCHECK_TARGET_GFX1100;
  case ROCJITSU_CODE_TARGET_GFX1150:
    return ROCJITSU_WAITCHECK_TARGET_GFX1150;
  case ROCJITSU_CODE_TARGET_GFX1151:
    return ROCJITSU_WAITCHECK_TARGET_GFX1151;
  case ROCJITSU_CODE_TARGET_GFX1200:
    return ROCJITSU_WAITCHECK_TARGET_GFX1200;
  case ROCJITSU_CODE_TARGET_GFX1201:
    return ROCJITSU_WAITCHECK_TARGET_GFX1201;
  case ROCJITSU_CODE_TARGET_GFX1250:
    return ROCJITSU_WAITCHECK_TARGET_GFX1250;
  default:
    return ROCJITSU_WAITCHECK_TARGET_UNKNOWN;
  }
}

[[nodiscard]] rj_waitcheck_diagnostic_code_t public_diagnostic_code(WaitcheckDiagnosticKind kind) {
  switch (kind) {
  case WaitcheckDiagnosticKind::WaitCounter:
    return ROCJITSU_WAITCHECK_DIAGNOSTIC_WAIT_COUNTER;
  case WaitcheckDiagnosticKind::SgprDepctr:
    return ROCJITSU_WAITCHECK_DIAGNOSTIC_SGPR_DEPCTR;
  case WaitcheckDiagnosticKind::AsyncBarrierPreWait:
    return ROCJITSU_WAITCHECK_DIAGNOSTIC_ASYNC_BARRIER_PRE_WAIT;
  case WaitcheckDiagnosticKind::AsyncBarrierPostWait:
    return ROCJITSU_WAITCHECK_DIAGNOSTIC_ASYNC_BARRIER_POST_WAIT;
  case WaitcheckDiagnosticKind::VaVdst:
    return ROCJITSU_WAITCHECK_DIAGNOSTIC_VA_VDST;
  }
  return ROCJITSU_WAITCHECK_DIAGNOSTIC_UNKNOWN;
}

[[nodiscard]] rj_waitcheck_diagnostic_t public_diagnostic(const WaitcheckDiagnostic &diagnostic) {
  return rj_waitcheck_diagnostic_t{
      .struct_size = sizeof(rj_waitcheck_diagnostic_t),
      .abi_version = ROCJITSU_WAITCHECK_ABI_VERSION,
      .has_kernel = diagnostic.has_kernel ? 1U : 0U,
      .kernel_name = diagnostic.has_kernel ? diagnostic.kernel_name.c_str() : nullptr,
      .kernel_entry_offset = diagnostic.kernel_entry_offset,
      .counter = public_counter(diagnostic.counter),
      .access = public_access(diagnostic.access),
      .reg =
          rj_waitcheck_register_t{
              .register_class = public_register_class(diagnostic.reg.cls),
              .index = diagnostic.reg.index,
              .width = diagnostic.reg.width,
          },
      .section_name = diagnostic.section_name.c_str(),
      .section_offset = diagnostic.section_offset,
      .file_offset = diagnostic.file_offset,
      .instruction = diagnostic.instruction.c_str(),
      .producer_section_offset = diagnostic.producer_section_offset,
      .producer_file_offset = diagnostic.producer_file_offset,
      .producer_instruction = diagnostic.producer_instruction.c_str(),
      .required_count = diagnostic.required_count,
      .message = diagnostic.message.c_str(),
      .code = public_diagnostic_code(diagnostic.kind),
  };
}

[[nodiscard]] rj_waitcheck_counter_parity_diagnostic_t
public_counter_parity_diagnostic(const WaitcheckCounterUnderaccountingDiagnostic &diagnostic) {
  return rj_waitcheck_counter_parity_diagnostic_t{
      .struct_size = sizeof(rj_waitcheck_counter_parity_diagnostic_t),
      .abi_version = ROCJITSU_WAITCHECK_ABI_VERSION,
      .kind = diagnostic.has_required_dependency
                  ? ROCJITSU_WAITCHECK_COUNTER_PARITY_MODELED_UNDERACCOUNTING
                  : ROCJITSU_WAITCHECK_COUNTER_PARITY_UNMODELED_EMITTED_WAIT,
      .has_kernel = diagnostic.has_kernel ? 1U : 0U,
      .kernel_name = diagnostic.has_kernel ? diagnostic.kernel_name.c_str() : nullptr,
      .kernel_entry_offset = diagnostic.kernel_entry_offset,
      .counter = public_counter(diagnostic.counter),
      .emitted_count = diagnostic.emitted_count,
      .required_count = diagnostic.required_count,
      .has_required_dependency = diagnostic.has_required_dependency ? 1U : 0U,
      .access = diagnostic.has_required_dependency ? public_access(diagnostic.access)
                                                   : ROCJITSU_WAITCHECK_ACCESS_INVALID,
      .reg =
          rj_waitcheck_register_t{
              .register_class = diagnostic.has_required_dependency
                                    ? public_register_class(diagnostic.reg.cls)
                                    : ROCJITSU_WAITCHECK_REGISTER_INVALID,
              .index = diagnostic.has_required_dependency ? diagnostic.reg.index : uint16_t{0},
              .width = diagnostic.has_required_dependency ? diagnostic.reg.width : uint8_t{0},
          },
      .section_name = diagnostic.section_name.c_str(),
      .wait_section_offset = diagnostic.wait_section_offset,
      .wait_file_offset = diagnostic.wait_file_offset,
      .wait_instruction = diagnostic.wait_instruction.c_str(),
      .consumer_section_offset = diagnostic.consumer_section_offset,
      .consumer_file_offset = diagnostic.consumer_file_offset,
      .consumer_instruction = diagnostic.consumer_instruction.c_str(),
      .producer_section_offset = diagnostic.producer_section_offset,
      .producer_file_offset = diagnostic.producer_file_offset,
      .producer_instruction =
          diagnostic.has_required_dependency ? diagnostic.producer_instruction.c_str() : nullptr,
      .message = diagnostic.message.c_str(),
  };
}

void populate_result(const WaitcheckReport &report, size_t diagnostics_reported,
                     size_t counter_parity_diagnostics_reported, bool counter_parity_evaluated,
                     rj_waitcheck_result_t &result) {
  result.instructions_analyzed = report.instructions_analyzed;
  result.memory_events_tracked = report.memory_events_tracked;
  result.kernels_discovered = report.kernels_discovered;
  result.kernels_analyzed = report.kernels_analyzed;
  result.diagnostics_observed = report.diagnostics_observed;
  result.diagnostics_reported = diagnostics_reported;
  result.passed = report.passed() ? 1U : 0U;
  result.diagnostics_truncated = report.diagnostics_truncated ? 1U : 0U;
  result.stopped_early = report.stopped_early ? 1U : 0U;
  result.counter_parity_wait_groups = report.counter_parity_wait_groups;
  result.counter_parity_fields_checked = report.counter_parity_fields_checked;
  result.counter_parity_exact = report.counter_parity_exact;
  result.counter_underaccounting_observed = report.counter_underaccounting_observed;
  result.counter_unmodeled_wait_observed = report.counter_unmodeled_wait_observed;
  result.counter_parity_indeterminate_groups = report.counter_parity_indeterminate_groups;
  result.counter_parity_diagnostics_reported = counter_parity_diagnostics_reported;
  result.counter_parity_diagnostics_truncated =
      report.counter_parity_diagnostics_truncated ? 1U : 0U;
  result.counter_parity_evaluated = counter_parity_evaluated ? 1U : 0U;
  result.analysis_complete = report.analysis_complete ? 1U : 0U;
  result.incomplete_observations = report.incomplete_observations;
}

} // namespace

rj_status_t rj_waitcheck_options_init(rj_waitcheck_options_t *options, size_t options_size) {
  if (!options || options_size < kOptionsV1Size)
    return ROCJITSU_STATUS_INVALID_ARGUMENT;
  std::memset(options, 0, options_size);
  options->struct_size = options_size;
  options->abi_version = ROCJITSU_WAITCHECK_ABI_VERSION;
  return ROCJITSU_STATUS_SUCCESS;
}

rj_status_t rj_waitcheck_result_init(rj_waitcheck_result_t *result, size_t result_size) {
  if (!result || result_size < kResultV1Size)
    return ROCJITSU_STATUS_INVALID_ARGUMENT;
  std::memset(result, 0, result_size);
  result->struct_size = result_size;
  result->abi_version = ROCJITSU_WAITCHECK_ABI_VERSION;
  result->target = ROCJITSU_WAITCHECK_TARGET_UNKNOWN;
  return ROCJITSU_STATUS_SUCCESS;
}

#if defined(__clang__)
#define RJ_NO_SANITIZE_ENUM __attribute__((no_sanitize("enum")))
#else
#define RJ_NO_SANITIZE_ENUM
#endif

RJ_NO_SANITIZE_ENUM const char *rj_waitcheck_target_name(rj_waitcheck_target_t target) {
  switch (target) {
  case ROCJITSU_WAITCHECK_TARGET_GFX90A:
    return "gfx90a";
  case ROCJITSU_WAITCHECK_TARGET_GFX942:
    return "gfx942";
  case ROCJITSU_WAITCHECK_TARGET_GFX950:
    return "gfx950";
  case ROCJITSU_WAITCHECK_TARGET_GFX1100:
    return "gfx1100";
  case ROCJITSU_WAITCHECK_TARGET_GFX1150:
    return "gfx1150";
  case ROCJITSU_WAITCHECK_TARGET_GFX1151:
    return "gfx1151";
  case ROCJITSU_WAITCHECK_TARGET_GFX1200:
    return "gfx1200";
  case ROCJITSU_WAITCHECK_TARGET_GFX1201:
    return "gfx1201";
  case ROCJITSU_WAITCHECK_TARGET_GFX1250:
    return "gfx1250";
  case ROCJITSU_WAITCHECK_TARGET_UNKNOWN:
    break;
  }
  return "unknown";
}

RJ_NO_SANITIZE_ENUM const char *
rj_waitcheck_diagnostic_code_name(rj_waitcheck_diagnostic_code_t code) {
  switch (code) {
  case ROCJITSU_WAITCHECK_DIAGNOSTIC_WAIT_COUNTER:
    return "wait-counter";
  case ROCJITSU_WAITCHECK_DIAGNOSTIC_SGPR_DEPCTR:
    return "sgpr-depctr";
  case ROCJITSU_WAITCHECK_DIAGNOSTIC_ASYNC_BARRIER_PRE_WAIT:
    return "async-barrier-pre-wait";
  case ROCJITSU_WAITCHECK_DIAGNOSTIC_ASYNC_BARRIER_POST_WAIT:
    return "async-barrier-post-wait";
  case ROCJITSU_WAITCHECK_DIAGNOSTIC_VA_VDST:
    return "va-vdst";
  case ROCJITSU_WAITCHECK_DIAGNOSTIC_UNKNOWN:
    break;
  }
  return "unknown";
}

RJ_NO_SANITIZE_ENUM const char *
rj_waitcheck_counter_parity_kind_name(rj_waitcheck_counter_parity_kind_t kind) {
  switch (kind) {
  case ROCJITSU_WAITCHECK_COUNTER_PARITY_MODELED_UNDERACCOUNTING:
    return "modeled-counter-underaccounting";
  case ROCJITSU_WAITCHECK_COUNTER_PARITY_UNMODELED_EMITTED_WAIT:
    return "unmodeled-emitted-wait";
  case ROCJITSU_WAITCHECK_COUNTER_PARITY_UNKNOWN:
    break;
  }
  return "unknown";
}

#undef RJ_NO_SANITIZE_ENUM

namespace {

[[nodiscard]] rj_status_t analyze(const void *code_object, size_t code_object_size,
                                  std::optional<uint64_t> kernel_entry_offset,
                                  const rj_waitcheck_options_t *options,
                                  rj_waitcheck_result_t *result) {
  rj_waitcheck_options_t local_options;
  if (read_options(options, local_options) != ROCJITSU_STATUS_SUCCESS)
    return ROCJITSU_STATUS_INVALID_ARGUMENT;
  if (!result_header_is_valid(result)) {
    report_error(local_options, "result must be initialized for this waitcheck ABI version");
    return ROCJITSU_STATUS_INVALID_ARGUMENT;
  }

  rj_waitcheck_result_t local_result{};
  local_result.struct_size = result->struct_size;
  local_result.abi_version = ROCJITSU_WAITCHECK_ABI_VERSION;
  local_result.target = ROCJITSU_WAITCHECK_TARGET_UNKNOWN;
  const auto finish = [&](rj_status_t status) {
    store_result(local_result, result);
    return status;
  };
  store_result(local_result, result);

  if (!code_object || code_object_size == 0) {
    report_error(local_options, "code object bytes, a nonzero size, and a result are required");
    return finish(ROCJITSU_STATUS_INVALID_ARGUMENT);
  }

  try {
    AmdGpuCodeObject parsed(static_cast<const uint8_t *>(code_object), code_object_size);
    if (!parsed.is_valid()) {
      report_error(local_options, "buffer is not a valid AMDGPU HSA code object");
      return finish(ROCJITSU_STATUS_INVALID_CODE_OBJECT);
    }
    local_result.target = public_target(parsed.target_id());

    const rj_code_arch_t arch = waitcheck_arch_for_target(parsed.target_id());
    if (arch == ROCJITSU_CODE_ARCH_INVALID) {
      report_error(local_options, "code object target is not supported by waitcheck");
      return finish(ROCJITSU_STATUS_INVALID_CODE_OBJECT);
    }

    WaitcheckOptions internal_options;
    internal_options.stop_after_first_diagnostic = local_options.stop_after_first_diagnostic != 0;
    internal_options.check_counter_parity = local_options.check_counter_parity != 0;
    if (local_options.max_reachability_cache_bytes != 0)
      internal_options.max_reachability_cache_bytes = local_options.max_reachability_cache_bytes;
    if (!local_options.diagnostic_callback) {
      internal_options.max_diagnostics = 0;
    } else if (local_options.max_diagnostics != 0) {
      internal_options.max_diagnostics = local_options.max_diagnostics;
    }
    if (!local_options.counter_parity_callback) {
      internal_options.max_counter_parity_diagnostics = 0;
    } else if (local_options.max_counter_parity_diagnostics != 0) {
      internal_options.max_counter_parity_diagnostics =
          local_options.max_counter_parity_diagnostics;
    }

    WaitcheckReport report;
    if (kernel_entry_offset) {
      const std::vector<WaitcheckKernelInfo> kernels = waitcheck_kernels(parsed);
      const auto kernel_it =
          std::ranges::find(kernels, *kernel_entry_offset, &WaitcheckKernelInfo::entry_offset);
      if (kernel_it == kernels.end()) {
        report_error(local_options, "kernel entry offset is not present in the code object");
        return finish(ROCJITSU_STATUS_INVALID_ARGUMENT);
      }
      report = analyze_waitcnts_for_kernel(parsed, arch, *kernel_it, internal_options);
      report.kernels_discovered = kernels.size();
    } else {
      report = analyze_waitcnts(parsed, arch, internal_options);
    }
    if (!report.supported) {
      const char *message = report.analysis_error.empty() ? "waitcheck analysis failed"
                                                          : report.analysis_error.c_str();
      report_error(local_options, message);
      return finish(ROCJITSU_STATUS_INVALID_CODE_OBJECT);
    }

    size_t diagnostics_reported = 0;
    if (local_options.diagnostic_callback) {
      for (const WaitcheckDiagnostic &diagnostic : report.diagnostics) {
        const rj_waitcheck_diagnostic_t public_view = public_diagnostic(diagnostic);
        ++diagnostics_reported;
        try {
          local_options.diagnostic_callback(&public_view, local_options.user_data);
        } catch (...) {
          // Callback failures belong to the consumer, not the analysis result.
        }
      }
    }
    size_t counter_parity_diagnostics_reported = 0;
    if (local_options.counter_parity_callback) {
      for (const auto &diagnostic : report.counter_underaccounting_diagnostics) {
        const rj_waitcheck_counter_parity_diagnostic_t public_view =
            public_counter_parity_diagnostic(diagnostic);
        ++counter_parity_diagnostics_reported;
        try {
          local_options.counter_parity_callback(&public_view, local_options.user_data);
        } catch (...) {
          // Callback failures belong to the consumer, not the analysis result.
        }
      }
    }
    const bool counter_parity_evaluated = local_options.check_counter_parity != 0 &&
                                          parsed.target_id() == ROCJITSU_CODE_TARGET_GFX950;
    populate_result(report, diagnostics_reported, counter_parity_diagnostics_reported,
                    counter_parity_evaluated, local_result);
    if (!report.analysis_complete) {
      std::string message = "waitcheck analysis incomplete";
      if (!report.incomplete_reason.empty())
        message += ": " + report.incomplete_reason;
      report_error(local_options, message.c_str());
      return finish(ROCJITSU_STATUS_UNSUPPORTED);
    }
    return finish(ROCJITSU_STATUS_SUCCESS);
  } catch (const std::bad_alloc &) {
    report_error(local_options, "waitcheck analysis ran out of memory");
    return finish(ROCJITSU_STATUS_OUT_OF_RESOURCES);
  } catch (const std::exception &error) {
    report_error(local_options, error.what());
    return finish(ROCJITSU_STATUS_ERROR);
  } catch (...) {
    report_error(local_options, "unexpected waitcheck analysis error");
    return finish(ROCJITSU_STATUS_ERROR);
  }
}

} // namespace

rj_status_t rj_waitcheck_analyze(const void *code_object, size_t code_object_size,
                                 const rj_waitcheck_options_t *options,
                                 rj_waitcheck_result_t *result) {
  return analyze(code_object, code_object_size, std::nullopt, options, result);
}

rj_status_t rj_waitcheck_analyze_kernel(const void *code_object, size_t code_object_size,
                                        uint64_t kernel_entry_offset,
                                        const rj_waitcheck_options_t *options,
                                        rj_waitcheck_result_t *result) {
  return analyze(code_object, code_object_size, kernel_entry_offset, options, result);
}
