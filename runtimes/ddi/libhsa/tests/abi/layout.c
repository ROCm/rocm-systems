// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

/* Independent public-header layout and legacy finalizer ABI probe. */
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <hsa/hsa_ext_finalize.h>
#include <hsa/hsa_ext_amd.h>
#include <hsa/hsa_ext_image.h>
#include <hsa/hsa_ven_amd_loader.h>

_Static_assert(sizeof(hsa_amd_ext_kernel_dispatch_packet_t) == 64,
               "AMD dispatch packets must occupy one AQL slot");
_Static_assert(offsetof(hsa_amd_ext_kernel_dispatch_packet_t, private_segment_size) ==
                   offsetof(hsa_kernel_dispatch_packet_t, private_segment_size),
               "scratch handling shares the private-size field");
_Static_assert(offsetof(hsa_amd_ext_kernel_dispatch_packet_t, workgroup_size_x) ==
                   offsetof(hsa_kernel_dispatch_packet_t, workgroup_size_x),
               "scratch handling shares the workgroup fields");

#define TYPE(label, type) \
  printf("type %s %zu %zu\n", label, sizeof(type), _Alignof(type))
#define FIELD(label, type, member) \
  printf("field %s %zu %zu\n", label, offsetof(type, member), \
         sizeof(((type *)0)->member))

int main(void) {
  TYPE("HsaFile", hsa_file_t);
  TYPE("HsaHandle", hsa_handle_t);
  TYPE("HsaSignal", hsa_signal_t);
  FIELD("HsaSignal.handle", hsa_signal_t, handle);
  TYPE("HsaDim3", hsa_dim3_t);
  FIELD("HsaDim3.x", hsa_dim3_t, x);
  FIELD("HsaDim3.y", hsa_dim3_t, y);
  FIELD("HsaDim3.z", hsa_dim3_t, z);
  TYPE("HsaQueue", hsa_queue_t);
  FIELD("HsaQueue.queue_type", hsa_queue_t, type);
  FIELD("HsaQueue.features", hsa_queue_t, features);
  FIELD("HsaQueue.base_address", hsa_queue_t, base_address);
  FIELD("HsaQueue.doorbell_signal", hsa_queue_t, doorbell_signal);
  FIELD("HsaQueue.size", hsa_queue_t, size);
  FIELD("HsaQueue.reserved", hsa_queue_t, reserved1);
  FIELD("HsaQueue.id", hsa_queue_t, id);
  TYPE("HsaKernelDispatchPacket", hsa_kernel_dispatch_packet_t);
  FIELD("HsaKernelDispatchPacket.header", hsa_kernel_dispatch_packet_t, header);
  FIELD("HsaKernelDispatchPacket.setup", hsa_kernel_dispatch_packet_t, setup);
  FIELD("HsaKernelDispatchPacket.private_segment_size", hsa_kernel_dispatch_packet_t, private_segment_size);
  FIELD("HsaKernelDispatchPacket.kernarg_address", hsa_kernel_dispatch_packet_t, kernarg_address);
  FIELD("HsaKernelDispatchPacket.completion_signal", hsa_kernel_dispatch_packet_t, completion_signal);
  TYPE("HsaExtControlDirectives", hsa_ext_control_directives_t);
  FIELD("HsaExtControlDirectives.control_directives_mask",
        hsa_ext_control_directives_t, control_directives_mask);
  FIELD("HsaExtControlDirectives.break_exceptions_mask",
        hsa_ext_control_directives_t, break_exceptions_mask);
  FIELD("HsaExtControlDirectives.detect_exceptions_mask",
        hsa_ext_control_directives_t, detect_exceptions_mask);
  FIELD("HsaExtControlDirectives.max_dynamic_group_size",
        hsa_ext_control_directives_t, max_dynamic_group_size);
  FIELD("HsaExtControlDirectives.max_flat_grid_size",
        hsa_ext_control_directives_t, max_flat_grid_size);
  FIELD("HsaExtControlDirectives.max_flat_workgroup_size",
        hsa_ext_control_directives_t, max_flat_workgroup_size);
  FIELD("HsaExtControlDirectives.reserved1",
        hsa_ext_control_directives_t, reserved1);
  FIELD("HsaExtControlDirectives.required_grid_size",
        hsa_ext_control_directives_t, required_grid_size);
  FIELD("HsaExtControlDirectives.required_workgroup_size",
        hsa_ext_control_directives_t, required_workgroup_size);
  FIELD("HsaExtControlDirectives.required_dim",
        hsa_ext_control_directives_t, required_dim);
  FIELD("HsaExtControlDirectives.reserved2",
        hsa_ext_control_directives_t, reserved2);

  TYPE("HsaExtSamplerDescriptorV3", hsa_ext_sampler_descriptor_v3_t);
  FIELD("HsaExtSamplerDescriptorV3.coordinate_mode", hsa_ext_sampler_descriptor_v3_t, coordinate_mode);
  FIELD("HsaExtSamplerDescriptorV3.filter_mode", hsa_ext_sampler_descriptor_v3_t, filter_mode);
  FIELD("HsaExtSamplerDescriptorV3.mipmap_filter_mode", hsa_ext_sampler_descriptor_v3_t, mipmap_filter_mode);
  FIELD("HsaExtSamplerDescriptorV3.address_modes", hsa_ext_sampler_descriptor_v3_t, address_modes);
  FIELD("HsaExtSamplerDescriptorV3.max_anisotropy", hsa_ext_sampler_descriptor_v3_t, max_anisotropy);
  TYPE("HsaAmdMemoryCopyRectPtr", hsa_amd_memory_copy_rect_ptr_t);
  FIELD("HsaAmdMemoryCopyRectPtr.pitched_ptr", hsa_amd_memory_copy_rect_ptr_t, pitched_ptr);
  FIELD("HsaAmdMemoryCopyRectPtr.offset", hsa_amd_memory_copy_rect_ptr_t, offset);

  hsa_ext_control_directives_t directives = {0};
  hsa_ext_program_t program = {0};
  hsa_isa_t isa = {0};
  hsa_code_object_t code_object = {0};
  if (hsa_ext_program_finalize(program, isa, 0, directives, NULL, 0,
                               &code_object) != HSA_STATUS_ERROR_NOT_INITIALIZED) {
    return 2;
  }
  hsa_agent_t agent = {1};
  size_t bytes = 0;
  void* pointer = NULL;
#ifdef _WIN32
  /* A non-null native pointer whose low 32 bits would be a negative fd. */
  hsa_handle_t handle = (void*)(uintptr_t)UINT64_C(0x180000000);
#else
  hsa_handle_t handle = 0;
#endif
  if (hsa_amd_interop_map_buffer(1, &agent, handle, 0, &bytes, &pointer,
                                 NULL, NULL) != HSA_STATUS_ERROR_NOT_INITIALIZED) {
    return 3;
  }
  hsa_code_object_reader_t reader = {0};
  if (hsa_ven_amd_loader_code_object_reader_get_load_size(reader, &bytes) != HSA_STATUS_ERROR_NOT_INITIALIZED) return 4;
  hsa_ext_sampler_t sampler = {0};
  if (hsa_ext_sampler_create_v3(agent, NULL, &sampler) != HSA_STATUS_ERROR_NOT_INITIALIZED) return 5;
  return 0;
}
