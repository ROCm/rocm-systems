// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

#include "rocjitsu/isa/decoder.h"

#include <array>
#include <cstdint>

namespace rocjitsu::test {

/// @brief Common GFX11 MIMG fields used by image decoding regressions.
struct Gfx11MimgEncoding {
  bool nsa = false;
  uint8_t dim = 0;
  bool unorm = false;
  uint8_t dmask = 0;
  bool r128 = false;
  bool a16 = false;
  bool d16 = false;
  uint8_t vaddr = 0;
  uint8_t vdata = 0;
  uint8_t srsrc = 0;
  bool tfe = false;
  uint32_t nsa_word = 0;
};

/// @brief Build and decode an RDNA3/RDNA3.5 MIMG instruction.
template <typename BuilderFields, typename Builder>
DecodeResult decode_gfx11_mimg(Decoder &decoder, Builder builder, uint16_t opcode,
                               const Gfx11MimgEncoding &encoding,
                               const DecodeErrorEmitter &emit_error = {}) {
  const auto base = builder(opcode, BuilderFields{.nsa = encoding.nsa,
                                                  .dim = encoding.dim,
                                                  .unorm = encoding.unorm,
                                                  .dmask = encoding.dmask,
                                                  .r128 = encoding.r128,
                                                  .a16 = encoding.a16,
                                                  .d16 = encoding.d16,
                                                  .vaddr = encoding.vaddr,
                                                  .vdata = encoding.vdata,
                                                  .srsrc = encoding.srsrc,
                                                  .tfe = encoding.tfe});
  if (!encoding.nsa)
    return decoder.decode(base.data(), emit_error);
  const std::array<uint32_t, 3> words{base[0], base[1], encoding.nsa_word};
  return decoder.decode(words.data(), emit_error);
}

} // namespace rocjitsu::test
