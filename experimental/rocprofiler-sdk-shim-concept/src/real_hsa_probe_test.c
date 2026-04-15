/*
 * real_hsa_probe_test.c
 *
 * Standalone helper for validating the real HSA table path exposed by the
 * shim's test-only export. Public hsa_* calls do not reliably exercise the
 * registered dispatch table on all runtime paths, so this test explicitly
 * calls through the registered HsaApiTable after the shim has installed its
 * wrappers.
 */
#define _GNU_SOURCE
#include <dlfcn.h>
#include <stdio.h>
#include <unistd.h>

#if defined(__has_include)
#    if __has_include(<hsa/hsa.h>)
#        include <hsa/hsa.h>
#    else
#        include <hsa.h>
#    endif
#else
#    include <hsa.h>
#endif

#include "shim_real_probe.h"

typedef int (*shim_hsa_probe_fn_t)(int*);

int main(void)
{
    shim_hsa_probe_fn_t probe = NULL;
    int                 count = 0;

    if(hsa_init() != HSA_STATUS_SUCCESS) {
        fprintf(stderr, "hsa_init failed\n");
        return 10;
    }

    probe = (shim_hsa_probe_fn_t) dlsym(RTLD_DEFAULT, "shim_real_probe_invoke_hsa_iterate_agents");
    if(probe == NULL) {
        fprintf(stderr, "shim_real_probe_invoke_hsa_iterate_agents not found\n");
        hsa_shut_down();
        return 11;
    }

    for(int i = 0; i < 2000; ++i) {
        count = 0;
        if(probe(&count) != HSA_STATUS_SUCCESS) {
            fprintf(stderr, "probe iterate failed\n");
            hsa_shut_down();
            return 12;
        }
        usleep(1000);
    }

    fprintf(stderr, "hsa probe loop done (agents=%d)\n", count);

    if(hsa_shut_down() != HSA_STATUS_SUCCESS) {
        fprintf(stderr, "hsa_shut_down failed\n");
        return 13;
    }

    sleep(8);
    return 0;
}
