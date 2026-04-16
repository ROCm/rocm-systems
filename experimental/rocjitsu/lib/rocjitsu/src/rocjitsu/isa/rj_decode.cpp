// Copyright (c) 2025-2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "rocjitsu/code/rj_code.h"

#include "rocjitsu/isa/decoder.h"
#include "rocjitsu/isa/instruction.h"
#include "rocjitsu/refcount.h"

#include <memory>

using namespace rocjitsu;

struct rj_code_decoder_t : RefCounted {
  std::unique_ptr<Decoder> decoder;
};

rj_status_t rj_code_decoder_create(rj_code_arch_t arch, rj_code_decoder_t **decoder) {
  if (!decoder || arch < 0 || arch >= ROCJITSU_CODE_ARCH_NUM_ARCHS)
    return ROCJITSU_STATUS_INVALID_ARGUMENT;

  auto d = Decoder::create(arch);
  if (!d)
    return ROCJITSU_STATUS_ERROR;

  *decoder = new rj_code_decoder_t{};
  (*decoder)->decoder = std::move(d);
  return ROCJITSU_STATUS_SUCCESS;
}

void rj_code_decoder_retain(rj_code_decoder_t *decoder) {
  if (decoder)
    decoder->retain();
}

void rj_code_decoder_release(rj_code_decoder_t *decoder) {
  if (!decoder)
    return;
  if (decoder->release())
    delete decoder;
}

void rj_code_decoder_destroy(rj_code_decoder_t *decoder) {
  if (!decoder)
    return;
  if (decoder->destroy())
    delete decoder;
}

rj_status_t rj_code_decoder_decode(rj_code_decoder_t *decoder,
                                   const rj_code_binary_inst_t *binary_inst,
                                   rj_code_inst_t **inst) {
  if (!decoder || !decoder->decoder || !binary_inst || !inst)
    return ROCJITSU_STATUS_INVALID_ARGUMENT;

  auto decoded = decoder->decoder->decode(binary_inst);
  if (!decoded)
    return ROCJITSU_STATUS_ERROR;

  *inst = reinterpret_cast<rj_code_inst_t *>(decoded);
  return ROCJITSU_STATUS_SUCCESS;
}
