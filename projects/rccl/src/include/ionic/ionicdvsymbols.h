#ifndef NCCL_IONICDV_SYMBOLS_H_
#define NCCL_IONICDV_SYMBOLS_H_

#include "ionic/ionicdvcore.h"
#include "nccl.h"

/* Ionic Direct Verbs Function Pointers*/
struct ncclIonicdvSymbols  {
  int (*ionicdv_internal_qp_set_gda)(struct ibv_qp *qp, bool enable_send, bool enable_recv);
  int (*ionicdv_internal_pd_set_udma_mask)(struct ibv_pd *ibpd, uint8_t udma_mask);
};

/* Constructs ionic direct verbs symbols per rdma-core linking or dynamic loading mode */
ncclResult_t buildIonicdvSymbols(struct ncclIonicdvSymbols* ionicdvSymbols);

#endif  // NCCL_IONICDV_SYMBOLS_H_
