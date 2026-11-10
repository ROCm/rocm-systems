#ifndef DDA_DONE_H_
#define DDA_DONE_H_

#include <cstdint>

// Completion word for a DDA-IPC launch. seq is signal memory; the kernel's last block stores `value`
// there so a later stream can wait without a per-launch event record.
struct DdaDoneMark {
  uint64_t* seq;
  unsigned* grid;
  uint64_t value;
};

struct rcclAddonDoneMem {
  uint64_t seq;
  unsigned grid;
};

#endif
