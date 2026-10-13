#include <sys/types.h>
#include <unistd.h>

#include "ionic/ionicdvsymbols.h"

/* ionicdv dynamic loading mode. Symbols are loaded from shared objects. */
#include <dlfcn.h>
#include "core.h"
#include "dv_dlopen.h"

// IONICDV Library versioning
#define IONIC_VERSION "IONIC_1.0"

ncclResult_t buildIonicdvSymbols(struct ncclIonicdvSymbols* ionicdvSymbols) {
  static void* ionicdvhandle = NULL;
  void* tmp;
  void** cast;

  ionicdvhandle = ncclDvDlopen("libionic.so", RTLD_NOW);
  if (!ionicdvhandle) {
    ionicdvhandle = ncclDvDlopen("libionic.so.1", RTLD_NOW);
    if (!ionicdvhandle) {
      INFO(NCCL_INIT, "Failed to open libionic.so[.1]");
      goto teardown;
    }
  }

#define LOAD_SYM(handle, symbol, funcptr) \
  do { \
    cast = (void**)&funcptr; \
    tmp = dlvsym(handle, symbol, IONIC_VERSION); \
    if (tmp == NULL) { \
      WARN("dlvsym failed on %s - %s version %s", symbol, dlerror(), IONIC_VERSION); \
      goto teardown; \
    } else { \
      INFO(NCCL_INIT, "dlvsym loaded successfully for %s - version %s", symbol, IONIC_VERSION); \
    } \
    *cast = tmp; \
  } while (0)

// Optional symbol: on failure only funcptr is left NULL; no teardown
#define LOAD_SYM_VERSION(handle, symbol, funcptr, version) \
  do { \
    cast = (void**)&funcptr; \
    *cast = dlvsym(handle, symbol, version); \
    if (*cast == NULL) { \
      INFO(NCCL_NET, "dlvsym failed on %s - %s version %s", symbol, dlerror(), version); \
    } \
  } while (0)

  LOAD_SYM(ionicdvhandle, "ionic_dv_pd_set_udma_mask", ionicdvSymbols->ionicdv_internal_pd_set_udma_mask);
  // Not exported by older libionic releases; IB-CAST then creates its QPs without GDA receive mode.
  LOAD_SYM_VERSION(ionicdvhandle, "ionic_dv_qp_set_gda", ionicdvSymbols->ionicdv_internal_qp_set_gda, IONIC_VERSION);
  LOAD_SYM_VERSION(ionicdvhandle, "ionic_dv_qp_set_puec_plane_route",
                   ionicdvSymbols->ionicdv_internal_qp_set_puec_plane_route, IONIC_VERSION);
  INFO(NCCL_INIT, "Loaded dlvsym from libionic.so[.1]");

  return ncclSuccess;

teardown:
  ionicdvSymbols->ionicdv_internal_qp_set_gda = NULL;
  ionicdvSymbols->ionicdv_internal_pd_set_udma_mask = NULL;
  ionicdvSymbols->ionicdv_internal_qp_set_puec_plane_route = NULL;

  if (ionicdvhandle != NULL) dlclose(ionicdvhandle);
  return ncclSystemError;
}
