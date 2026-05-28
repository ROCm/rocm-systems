#ifndef NCCL_IONICDVWRAP_H_
#define NCCL_IONICDVWRAP_H_

#include <arpa/inet.h>
#include <netinet/in.h>
#include "ionic/ionicdvcore.h"
#include "core.h"
#include "ibvwrap.h"
#include <sys/types.h>
#include <unistd.h>

ncclResult_t wrap_ionicdv_symbols(void);
/* NCCL wrappers of ionic direct verbs functions */
ncclResult_t wrap_ionicdv_qp_set_gda(struct ibv_qp *ibqp, bool enable_send, bool enable_recv);
ncclResult_t wrap_ionicdv_pd_set_udma_mask(struct ibv_pd *ibpd, uint8_t udma_mask);

#endif // NCCL_IONICDVWRAP_H_
