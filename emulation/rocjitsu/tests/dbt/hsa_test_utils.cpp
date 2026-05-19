// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "hsa_test_utils.h"

#ifdef HAS_HOST_AMDGPU

#include "rocjitsu/base/rj_compiler.h"
RJ_DIAGNOSTIC_PUSH
RJ_DIAGNOSTIC_IGNORE_PEDANTIC
#include <hsa/hsa.h>
RJ_DIAGNOSTIC_POP

#include <cstring>

namespace rocjitsu::dbt_test {

namespace {

hsa_agent_t find_gpu_agent() {
  hsa_agent_t gpu{};
  hsa_iterate_agents(
      [](hsa_agent_t agent, void *data) -> hsa_status_t {
        hsa_device_type_t type;
        hsa_agent_get_info(agent, HSA_AGENT_INFO_DEVICE, &type);
        if (type == HSA_DEVICE_TYPE_GPU) {
          *static_cast<hsa_agent_t *>(data) = agent;
          return HSA_STATUS_INFO_BREAK;
        }
        return HSA_STATUS_SUCCESS;
      },
      &gpu);
  return gpu;
}

} // namespace

std::string kernel_path(const char *name) { return std::string(KERNEL_DIR) + "/" + name + ".o"; }

HostTarget detect_hsa_host_target() {
  HostTarget target;
  if (hsa_init() != HSA_STATUS_SUCCESS)
    return target;

  hsa_agent_t gpu = find_gpu_agent();
  if (gpu.handle != 0) {
    hsa_isa_t isa{};
    char isa_name[128] = {};
    hsa_agent_get_info(gpu, HSA_AGENT_INFO_ISA, &isa);
    hsa_isa_get_info_alt(isa, HSA_ISA_INFO_NAME, isa_name);
    target.isa_name = isa_name;

    if (std::strstr(isa_name, "gfx1100")) {
      target.arch = ROCJITSU_CODE_ARCH_RDNA3;
      target.mach = 0x41;
    } else if (std::strstr(isa_name, "gfx1201")) {
      target.arch = ROCJITSU_CODE_ARCH_RDNA4;
      target.mach = 0x4E;
    } else if (std::strstr(isa_name, "gfx1200")) {
      target.arch = ROCJITSU_CODE_ARCH_RDNA4;
      target.mach = 0x48;
    }
  }

  hsa_shut_down();
  return target;
}

tools::KernelArgPatch ptr_arg(size_t offset, std::string buffer_name) {
  tools::KernelArgPatch patch;
  patch.offset = offset;
  patch.kind = tools::KernelArgKind::Pointer;
  patch.buffer_name = std::move(buffer_name);
  return patch;
}

tools::KernelArgPatch u32_arg(size_t offset, uint32_t value) {
  tools::KernelArgPatch patch;
  patch.offset = offset;
  patch.kind = tools::KernelArgKind::U32;
  patch.bytes = scalar_arg_bytes(value);
  return patch;
}

const tools::HsaBufferOutput *find_output(const tools::HsaRunOutput &output,
                                          const std::string &name) {
  for (const auto &buffer : output.outputs) {
    if (buffer.name == name)
      return &buffer;
  }
  return nullptr;
}

} // namespace rocjitsu::dbt_test

#endif // HAS_HOST_AMDGPU
