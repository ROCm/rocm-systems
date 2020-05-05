#ifndef CORE_HIP_ACT_CB_TRACKER_H_
#define CORE_HIP_ACT_CB_TRACKER_H_

#include <map>

namespace roctracer {
enum {
  API_CB_MASK = 0x1,
  ACT_CB_MASK = 0x2
};

class hip_act_cb_tracker_t {
  struct info_t {
    uint32_t mask;
    info_t() : mask(0) {}
  };

  typedef std::map<uint32_t, info_t>  map_t;
  map_t dara;

  public:
  uint32_t enable_check(const uint32_t& op, const uint32_t& mask) {
    uint32_t& val = dara[op].mask;
    val |= mask;
    return val;
  }

  uint32_t disable_check(const uint32_t& op, const uint32_t& mask) {
    uint32_t& val = dara[op].mask;
    val &= ~mask;
    return val;
  }
};  // hip_act_cb_tracker_t
};  // namespace roctracer

#endif  // CORE_HIP_ACT_CB_TRACKER_H_
