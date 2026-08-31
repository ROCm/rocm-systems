// Copyright (c) 2025-2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "rocjitsu/code/rj_code.h"

#include "rocjitsu/isa/decoder.h"
#include "rocjitsu/isa/instruction.h"
#include "rocjitsu/isa/target_registry.h"
#include "rocjitsu/refcount.h"

#include <memory>
#include <new>

using namespace rocjitsu;

struct rj_code_decoder_t : RefCounted {
  std::unique_ptr<Decoder> decoder;
};

namespace {

template <typename Factory>
rj_status_t create_decoder_handle(Factory &&factory, rj_code_decoder_t **decoder) {
  try {
    auto implementation = factory();
    if (!implementation)
      return ROCJITSU_STATUS_ERROR;

    auto handle = std::make_unique<rj_code_decoder_t>();
    handle->decoder = std::move(implementation);
    *decoder = handle.release();
    return ROCJITSU_STATUS_SUCCESS;
  } catch (const std::bad_alloc &) {
    return ROCJITSU_STATUS_OUT_OF_RESOURCES;
  } catch (...) {
    return ROCJITSU_STATUS_ERROR;
  }
}

} // namespace

rj_status_t rj_code_decoder_create(rj_code_arch_t arch, rj_code_decoder_t **decoder) {
  if (!decoder)
    return ROCJITSU_STATUS_INVALID_ARGUMENT;
  *decoder = nullptr;
  if (arch < ROCJITSU_CODE_ARCH_CDNA1 || arch >= ROCJITSU_CODE_ARCH_NUM_ARCHS)
    return ROCJITSU_STATUS_INVALID_ARGUMENT;

  return create_decoder_handle([arch] { return Decoder::create(arch); }, decoder);
}

rj_status_t rj_code_decoder_create_for_target(const char *target_id, rj_code_decoder_t **decoder) {
  if (decoder == nullptr)
    return ROCJITSU_STATUS_INVALID_ARGUMENT;
  *decoder = nullptr;
  if (target_id == nullptr || target_id[0] == '\0')
    return ROCJITSU_STATUS_INVALID_ARGUMENT;

  return create_decoder_handle(
      [target_id] { return Decoder::create(default_isa_target_registry(), target_id); }, decoder);
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
  if (!inst)
    return ROCJITSU_STATUS_INVALID_ARGUMENT;
  *inst = nullptr;
  if (!decoder || !decoder->decoder || !binary_inst)
    return ROCJITSU_STATUS_INVALID_ARGUMENT;

  try {
    Instruction::ScopedHeapAllocation heap_allocation;
    DecodeResult decoded = decoder->decoder->decode(binary_inst);
    if (decoded.failed())
      return ROCJITSU_STATUS_ERROR;

    *inst = reinterpret_cast<rj_code_inst_t *>(decoded.value().release());
    return ROCJITSU_STATUS_SUCCESS;
  } catch (const std::bad_alloc &) {
    return ROCJITSU_STATUS_OUT_OF_RESOURCES;
  } catch (...) {
    return ROCJITSU_STATUS_ERROR;
  }
}

void rj_code_inst_destroy(rj_code_inst_t *inst) {
  Instruction::ScopedHeapAllocation heap_allocation;
  delete reinterpret_cast<Instruction *>(inst);
}
