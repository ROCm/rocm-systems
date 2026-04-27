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
  IONIC_UDMA_MASK_LOW    = 1,
  IONIC_UDMA_MASK_HIGH   = 2
};

enum ionic_dv_qp_init_attr_mask {
  IONIC_DV_QP_INIT_ATTR_MASK_FLAGS            = 1 << 0,
  IONIC_DV_QP_INIT_ATTR_MASK_TRANSPORT_MODE   = 1 << 1,
  IONIC_DV_QP_INIT_ATTR_MASK_NUM_RCQ_PATHS    = 1 << 2
};

enum ionic_dv_qp_init_attr_flags {
  IONIC_DV_CREATE_QP_TYPE_RCCL        = 1 << 16,
  IONIC_DV_CREATE_QP_RCCL_DATA        = 1 << 17,
  IONIC_DV_CREATE_QP_RCCL_RDFENCE     = 1 << 18,
  IONIC_DV_CREATE_QP_RCCL_RX_OFFLOAD  = 1 << 19
};

enum ionic_dv_qp_transport_mode {
  IONIC_DV_QPT_TRANSPORT_DEFAULT,
  IONIC_DV_QPT_TRANSPORT_ROCE_V2,
  IONIC_DV_QPT_TRANSPORT_MRC
};

struct ionic_dv_qp_init_attr_ex {
  /* One or more flags of enum ionic_qp_init_attr_mask */
  uint32_t comp_mask;
  /* One or more flags of enum ionic_qp_init_attr_flags */
  uint32_t ionic_flags;
  /* enum ionic_dv_qp_transport_mode */
  enum ionic_dv_qp_transport_mode     transport_mode;
  /* number of RCQ paths */
  uint8_t     num_rcq_paths;
};

#endif  // NCCL_IONICDV_CORE_H_
