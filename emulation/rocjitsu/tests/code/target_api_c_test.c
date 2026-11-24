// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "rocjitsu/code/rj_code.h"

#include <stddef.h>
#include <stdlib.h>

_Static_assert(sizeof(rj_code_target_id_t) == sizeof(int32_t), "target IDs use a 32-bit ABI");
_Static_assert(ROCJITSU_CODE_TARGET_GFX90A == 0 && ROCJITSU_CODE_TARGET_GFX1100 == 7,
               "existing target IDs must remain stable");
_Static_assert(ROCJITSU_CODE_TARGET_INVALID == INT32_MAX, "invalid target must remain stable");

int main(void) {
  rj_code_decoder_t *decoder = NULL;
  if (rj_code_decoder_create(ROCJITSU_CODE_TARGET_RV64I, &decoder) != ROCJITSU_STATUS_SUCCESS)
    return EXIT_FAILURE;
  const rj_code_binary_inst_t addi = 0x00100093;
  rj_code_inst_t *instruction = NULL;
  const rj_status_t decoded = rj_code_decoder_decode(decoder, &addi, &instruction);
  rj_code_decoder_destroy(decoder);
  if (decoded != ROCJITSU_STATUS_SUCCESS || instruction == NULL)
    return EXIT_FAILURE;
  rj_code_inst_destroy(instruction);

  const rj_code_dbt_options_t options = {
      .guest_target = ROCJITSU_CODE_TARGET_GFX950,
      .host_target = ROCJITSU_CODE_TARGET_GFX1201,
  };
  rj_code_object_t *output = NULL;
  if (rj_code_translate(NULL, &options, &output) != ROCJITSU_STATUS_INVALID_ARGUMENT || output)
    return EXIT_FAILURE;
  return EXIT_SUCCESS;
}
