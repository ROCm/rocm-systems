// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "rocjitsu/analysis/rj_waitcheck.h"

#include "rocjitsu/analysis/waitcheck.h"
#include "rocjitsu/code/amdgpu_code_object.h"

#include <exception>
#include <new>

namespace {

using namespace rocjitsu;

[[nodiscard]] rj_waitcheck_options_t default_options() {
  rj_waitcheck_options_t options{};
  options.kernel_entry_offset = ROCJITSU_WAITCHECK_ALL_KERNELS;
  return options;
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

[[nodiscard]] rj_waitcheck_diagnostic_t public_diagnostic(const WaitcheckDiagnostic &diagnostic) {
  return rj_waitcheck_diagnostic_t{
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
  };
}

void populate_result(const WaitcheckReport &report, size_t diagnostics_reported,
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
}

} // namespace

void rj_waitcheck_options_init(rj_waitcheck_options_t *options) {
  if (options)
    *options = default_options();
}

rj_status_t rj_waitcheck_analyze(const void *code_object, size_t code_object_size,
                                 const rj_waitcheck_options_t *options,
                                 rj_waitcheck_result_t *result) {
  const rj_waitcheck_options_t local_options = options ? *options : default_options();
  if (!code_object || code_object_size == 0 || !result) {
    report_error(local_options, "code object bytes, a nonzero size, and a result are required");
    return ROCJITSU_STATUS_INVALID_ARGUMENT;
  }
  *result = {};

  try {
    AmdGpuCodeObject parsed(static_cast<const uint8_t *>(code_object), code_object_size);
    if (!parsed.is_valid()) {
      report_error(local_options, "buffer is not a valid AMDGPU HSA code object");
      return ROCJITSU_STATUS_INVALID_CODE_OBJECT;
    }

    const rj_code_arch_t arch = waitcheck_arch_for_target(parsed.target_id());
    if (arch == ROCJITSU_CODE_ARCH_INVALID) {
      report_error(local_options, "code object target is not supported by waitcheck");
      return ROCJITSU_STATUS_INVALID_CODE_OBJECT;
    }

    WaitcheckOptions internal_options;
    internal_options.stop_after_first_diagnostic = local_options.stop_after_first_diagnostic != 0;
    if (local_options.max_reachability_cache_bytes != 0)
      internal_options.max_reachability_cache_bytes = local_options.max_reachability_cache_bytes;
    if (!local_options.diagnostic_callback) {
      internal_options.max_diagnostics = 0;
    } else if (local_options.max_diagnostics != 0) {
      internal_options.max_diagnostics = local_options.max_diagnostics;
    }

    WaitcheckReport report =
        local_options.kernel_entry_offset == ROCJITSU_WAITCHECK_ALL_KERNELS
            ? analyze_waitcnts(parsed, arch, internal_options)
            : analyze_waitcnts_for_kernel(parsed, arch, local_options.kernel_entry_offset,
                                          internal_options);
    if (!report.supported) {
      const char *message = report.analysis_error.empty() ? "waitcheck analysis failed"
                                                          : report.analysis_error.c_str();
      report_error(local_options, message);
      if (report.analysis_error == "kernel entry offset is not present in the code object")
        return ROCJITSU_STATUS_INVALID_ARGUMENT;
      return ROCJITSU_STATUS_INVALID_CODE_OBJECT;
    }

    size_t diagnostics_reported = 0;
    if (local_options.diagnostic_callback) {
      for (const WaitcheckDiagnostic &diagnostic : report.diagnostics) {
        const rj_waitcheck_diagnostic_t public_view = public_diagnostic(diagnostic);
        local_options.diagnostic_callback(&public_view, local_options.user_data);
        ++diagnostics_reported;
      }
    }
    populate_result(report, diagnostics_reported, *result);
    return ROCJITSU_STATUS_SUCCESS;
  } catch (const std::bad_alloc &) {
    report_error(local_options, "waitcheck analysis ran out of memory");
    return ROCJITSU_STATUS_OUT_OF_RESOURCES;
  } catch (const std::exception &error) {
    report_error(local_options, error.what());
    return ROCJITSU_STATUS_ERROR;
  } catch (...) {
    report_error(local_options, "unexpected waitcheck analysis error");
    return ROCJITSU_STATUS_ERROR;
  }
}
