#pragma once

#include "rocjitsu/isa/register_set.h"

#include <stdint.h>

#include <optional>

namespace rocjitsu::fuzzer::afl_dbi {

struct Vopd64LivenessModel {
  RegisterSet uses;
  RegisterSet defs;
};

std::optional<Vopd64LivenessModel> decode_vopd64_liveness_model(uint32_t word0,
                                                                uint32_t word1);
uint32_t decode_vopd64_word_count(uint32_t word0, uint32_t word1);

} // namespace rocjitsu::fuzzer::afl_dbi
