#pragma once

#include <stdint.h>

namespace rocjitsu::fuzzer::afl {

inline constexpr uint32_t kMapSize = 65536;
inline constexpr uint32_t kDeviceStart = kMapSize / 2;
inline constexpr uint32_t kCoverageSlots = kMapSize / 2;
inline constexpr uint32_t kEntryCounterSlot = 0;

} // namespace rocjitsu::fuzzer::afl

extern "C" {

int rocjitsu_afl_persistent_begin();
int rocjitsu_afl_persistent_end();
}
