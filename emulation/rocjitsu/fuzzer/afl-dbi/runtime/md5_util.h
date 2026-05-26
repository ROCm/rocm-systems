#pragma once

#include <stdint.h>

#include <optional>
#include <span>

namespace rocjitsu::fuzzer::afl_dbi {

std::optional<uint64_t> truncated_md5_hash64(std::span<const uint8_t> data);

} // namespace rocjitsu::fuzzer::afl_dbi
