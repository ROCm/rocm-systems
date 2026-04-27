#include "ionic/ionicdvwrap.h"
#include <sys/types.h>
#include <unistd.h>
#include "param.h"

#include "ionic/ionicdvcore.h"
#include "ionic/ionicdvsymbols.h"

static pthread_once_t initOnceControl = PTHREAD_ONCE_INIT;
static ncclResult_t initResult;
struct ncclIonicdvSymbols ionicdvSymbols;
bool ionicdvCreateQpExSupported = false;

extern bool rcclUseAinic();

ncclResult_t wrap_ionicdv_symbols(void) {
#if defined(__HIP_PLATFORM_AMD__) || defined(__HIPCC__)
  if (rcclUseAinic()) {
    pthread_once(&initOnceControl,
                 [](){ initResult = buildIonicdvSymbols(&ionicdvSymbols); });
    if (initResult == ncclSuccess) {
      ionicdvCreateQpExSupported = (ionicdvSymbols.ionicdv_internal_create_qp_ex != NULL);
      INFO(NCCL_NET, "ionic_dv_create_qp_ex support: %s", ionicdvCreateQpExSupported ? "yes" : "no");
    }
    return initResult;
  }
#endif
  // simply return for unsupported platform/NIC.
  return ncclSuccess;
}

/* CHECK_NOT_NULL: helper macro to check for NULL symbol */
#define CHECK_NOT_NULL(container, internal_name) \
  if (container.internal_name == NULL) { \
     WARN("lib wrapper not initialized."); \
     return ncclInternalError; \
  }

#define IONICDV_PTR_CHECK_ERRNO(container, internal_name, call, retval, error_retval, name) \
  CHECK_NOT_NULL(container, internal_name); \
  retval = container.call; \
  if (retval == error_retval) { \
    WARN("Call to " name " failed with error %s", strerror(errno)); \
    return ncclSystemError; \
  } \
  return ncclSuccess;

#define IONICDV_INT_CHECK_RET_ERRNO(container, internal_name, call, success_retval, name) \
  CHECK_NOT_NULL(container, internal_name); \
  int ret = container.call; \
  if (ret != success_retval) { \
    INFO(NCCL_NET, "Call to " name " failed with error %s errno %d", strerror(ret), ret); \
    return ncclSystemError; \
  } else { \
    INFO(NCCL_NET, "Call to " name " success with ret %d", ret); \
  } \
  return ncclSuccess;

ncclResult_t wrap_ionicdv_qp_set_gda(struct ibv_qp *qp, bool enable_send, bool enable_recv) {
  if (ionicdvSymbols.ionicdv_internal_qp_set_gda == NULL) {
    errno = EOPNOTSUPP;
    return ncclSystemError;
  }
  IONICDV_INT_CHECK_RET_ERRNO(ionicdvSymbols, ionicdv_internal_qp_set_gda, ionicdv_internal_qp_set_gda(qp, enable_send, enable_recv), 0, "ionic_dv_qp_set_gda");
}

ncclResult_t wrap_ionicdv_create_qp_ex(struct ibv_context *context, struct ibv_qp_init_attr_ex *qp_attr, struct ionic_dv_qp_init_attr_ex *ionic_qp_attr, struct ibv_qp **qp) {
  INFO(NCCL_NET, "wrap_ionicdv_create_qp_ex: sz ibv_qp_init_attr_ex: %d, ionic_dv_qp_init_attr_ex: %d", sizeof(struct ibv_qp_init_attr_ex), sizeof(struct ionic_dv_qp_init_attr_ex));
  IONICDV_PTR_CHECK_ERRNO(ionicdvSymbols, ionicdv_internal_create_qp_ex, ionicdv_internal_create_qp_ex(context, qp_attr, ionic_qp_attr), *qp, NULL, "ionic_dv_create_qp_ex");
}

ncclResult_t wrap_ionicdv_pd_set_udma_mask(struct ibv_pd *ibpd, uint8_t udma_mask) {
  if (ionicdvSymbols.ionicdv_internal_pd_set_udma_mask == NULL) {
    errno = EOPNOTSUPP;
    return ncclSystemError;
  }
  IONICDV_INT_CHECK_RET_ERRNO(ionicdvSymbols, ionicdv_internal_pd_set_udma_mask, ionicdv_internal_pd_set_udma_mask(ibpd, udma_mask), 0, "ionic_dv_pd_set_udma_mask");
}
