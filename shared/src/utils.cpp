#include "shared/include/utils.h"

namespace wsl {
namespace thunk {

namespace {
static const struct GfxipTable kGfxipTable[] = {
  { 0x7448, 11, 0, 0 },
  { 0x744C, 11, 0, 0 },
  { 0x745E, 11, 0, 0 },
  { 0x7449, 11, 0, 0 },
  { 0x744a, 11, 0, 0 },
  { 0x744b, 11, 0, 0 },
  { 0x7470, 11, 0, 1 },
  { 0x747E, 11, 0, 1 },
  { 0x7590, 12, 0, 0 },
  { 0x7550, 12, 0, 1 },
  { 0x7551, 12, 0, 1 },
  { 0x150E, 11, 5, 0 },
  { 0x1586, 11, 5, 1 },
  { 0x1114, 11, 5, 2 },
  { 0x1900, 11, 0, 3 },
};

const int kGfxipTableSize = sizeof(kGfxipTable) / sizeof(kGfxipTable[0]);
} // namespace

bool QueryAdapterSupported(unsigned int device_id) {
  for (int i = 0; i < kGfxipTableSize; i++) {
    if (device_id == kGfxipTable[i].device_id)
      return true;
  }
  return false;
}

bool LookupGfxipEntry(uint16_t device_id, GfxipTable *out) {
  if (out == nullptr)
    return false;

  for (int i = 0; i < kGfxipTableSize; i++) {
    if (kGfxipTable[i].device_id == device_id) {
      *out = kGfxipTable[i];
      return true;
    }
  }
  return false;
}

} // namespace thunk
} // namespace wsl
