#ifndef CORE_HIP_ACT_CB_TRACKER_H_
#define CORE_HIP_ACT_CB_TRACKER_H_

#include <map>

namespace roctracer {
enum { API_CB_MASK = 0x1, ACT_CB_MASK = 0x2 };

class hip_act_cb_tracker_t {
  private:
  std::map<uint32_t, uint32_t> data;

 public:
  uint32_t enable_check(uint32_t op, uint32_t mask) { return data[op] |= mask; }

  uint32_t disable_check(uint32_t op, uint32_t mask) { return data[op] &= ~mask; }
};  // hip_act_cb_tracker_t
};  // namespace roctracer

#endif  // CORE_HIP_ACT_CB_TRACKER_H_
