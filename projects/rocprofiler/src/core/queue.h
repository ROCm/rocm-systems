#ifndef _SRC_CORE_QUEUE_H
#define _SRC_CORE_QUEUE_H

#include "core/types.h"

namespace rocprofiler {

class Queue {
  public:
  Queue() {}
  virtual ~Queue() {}
  virtual void Submit(const packet_t* packet) = 0;
  virtual void Submit(const packet_t* packet, const size_t& count) {
    for (const packet_t* p = packet; p < packet + count; ++p) Submit(p);
  }
};

} // namespace rocprofiler

#endif // _SRC_CORE_QUEUE_H
