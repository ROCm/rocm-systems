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
/* Tracks whether ionic_dv_create_qp_ex is available in the loaded libionic version */
extern bool ionicdvCreateQpExSupported;
/* NCCL wrappers of ionic direct verbs functions */
ncclResult_t wrap_ionicdv_qp_set_gda(struct ibv_qp *ibqp, bool enable_send, bool enable_recv);
ncclResult_t wrap_ionicdv_pd_set_udma_mask(struct ibv_pd *ibpd, uint8_t udma_mask);
ncclResult_t wrap_ionicdv_create_qp_ex(struct ibv_context *context, struct ibv_qp_init_attr_ex *qp_attr, struct ionic_dv_qp_init_attr_ex *ionic_qp_attr, struct ibv_qp **qp);

#endif // NCCL_IONICDVWRAP_H_
