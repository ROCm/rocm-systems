// SPDX-License-Identifier: MIT

/* Independent public-header layout and legacy finalizer ABI probe. */
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <hsa/hsa_ext_finalize.h>

#define TYPE(label, type) \
  printf("type %s %zu %zu\n", label, sizeof(type), _Alignof(type))
#define FIELD(label, type, member) \
  printf("field %s %zu %zu\n", label, offsetof(type, member), \
         sizeof(((type *)0)->member))

int main(void) {
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

  hsa_ext_control_directives_t directives = {0};
  hsa_ext_program_t program = {0};
  hsa_isa_t isa = {0};
  hsa_code_object_t code_object = {0};
  if (hsa_ext_program_finalize(program, isa, 0, directives, NULL, 0,
                               &code_object) != HSA_STATUS_ERROR_NOT_INITIALIZED) {
    return 2;
  }
  return 0;
}
