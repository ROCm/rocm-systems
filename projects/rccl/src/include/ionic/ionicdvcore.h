#ifndef NCCL_IONICDV_CORE_H_
#define NCCL_IONICDV_CORE_H_

/* Basic ionic direct verbs structs.
 * Needed to dynamically load ionic direct verbs functions without
 * explicit including of ionic direct verbs header.
 */

#include <stddef.h>
#include <stdint.h>
#include <sys/types.h>
#include <unistd.h>
#include "ibvwrap.h"

enum ionicdv_reg_udma_mask {
  IONIC_UDMA_MASK_LOW = 1,
  IONIC_UDMA_MASK_HIGH = 2
};

struct ionic_dv_puec_route {
  union ibv_gid dgid;
  union ibv_gid sgid;
  uint32_t flow_label;
  uint8_t hop_limit;
  uint8_t sl;
  uint8_t traffic_class;
  uint32_t flags;
};

#endif  // NCCL_IONICDV_CORE_H_
