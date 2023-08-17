
#pragma once

#include <dlfcn.h>

#ifdef __cplusplus
extern "C" {
#endif

#if defined(ROCP_REG_TEST_WEAK) && ROCP_REG_TEST_WEAK > 0
#    pragma weak hip_init
#    pragma weak hsa_init
#    pragma weak roctxRangePush
#    pragma weak roctxRangePop
#endif

extern void
hip_init(void);

extern void
hsa_init(void);

extern void
roctxRangePush(const char*);

extern void
roctxRangePop(const char*);

#ifdef __cplusplus
}
#endif
