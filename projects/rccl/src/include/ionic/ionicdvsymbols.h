#ifndef NCCL_IONICDV_SYMBOLS_H_
#define NCCL_IONICDV_SYMBOLS_H_

#include "ionic/ionicdvcore.h"
#include "nccl.h"

/* Ionic Direct Verbs Function Pointers*/
struct ncclIonicdvSymbols {
  int (*ionicdv_internal_qp_set_gda)(struct ibv_qp* qp, bool enable_send, bool enable_recv);
  int (*ionicdv_internal_pd_set_udma_mask)(struct ibv_pd* ibpd, uint8_t udma_mask);
  int (*ionicdv_internal_qp_set_puec_plane_route)(struct ibv_qp* qp, uint8_t plane_idx, struct ionic_dv_puec_route* route);
  uint8_t (*ionicdv_internal_ctx_get_udma_count)(struct ibv_context* ibctx);
  uint8_t (*ionicdv_internal_qp_get_udma_idx)(struct ibv_qp* ibqp);
};

/* Constructs ionic direct verbs symbols per rdma-core linking or dynamic loading mode */
ncclResult_t buildIonicdvSymbols(struct ncclIonicdvSymbols* ionicdvSymbols);

#endif  // NCCL_IONICDV_SYMBOLS_H_
