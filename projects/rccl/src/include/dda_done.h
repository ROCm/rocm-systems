#ifndef DDA_DONE_H_
#define DDA_DONE_H_

#include <cstdint>

// Completion word carried on the local IpcGpuBarrier copy. seq is signal memory; the kernel's last
// block stores `value` there so a later stream can wait without a per-launch event record.
struct DdaDoneMark {
  uint64_t* seq;
  unsigned* grid;
  uint64_t value;
};

// Host state for that word. wait is the sequence the next stream must observe; 0 means the last
// completion was doneEvent. HIP signal memory is 8 bytes, so the block counter is separate.
struct rcclDdaDoneState {
  uint64_t* seq;
  unsigned* grid;
  uint64_t seqValue;
  uint64_t wait;
  bool signaled;
};

#endif
