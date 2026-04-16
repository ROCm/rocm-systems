#ifndef AEGISBIT_TRAMPOLINE_TYPES_H
#define AEGISBIT_TRAMPOLINE_TYPES_H

#include <cstdint>
#include <vector>

namespace aegisbit {

/// One trampoline slot: describes the patch at one instrumentation site.
struct TrampolineSlot {
  uint64_t OriginalPC = 0;
  uint64_t TrampolineOffset = 0;
  std::vector<uint8_t> PatchBytes;
  std::vector<uint8_t> TrampolineBytes;
  uint64_t DisplacedSize = 0;
  bool UsedLongJump = false;
};

/// One trampoline island: a contiguous block of trampoline slots placed at
/// a specific offset in .text.
struct TrampolineIsland {
  std::vector<uint8_t> Bytes;
  uint64_t Offset = 0;
};

/// Complete result of building a trampoline-patched kernel.
struct BridgeResult {
  std::vector<TrampolineSlot> Slots;
  std::vector<TrampolineIsland> Islands;
  uint32_t PatchedCount = 0;
  uint32_t LongJumpCount = 0;
  std::vector<uint8_t> PrologueBytes;
};

} // namespace aegisbit

#endif // AEGISBIT_TRAMPOLINE_TYPES_H
