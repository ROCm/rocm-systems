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

static hsa_status_t
count_agents(hsa_agent_t agent, void* data)
{
    (void) agent;
    if(data != NULL) (*(int*) data)++;
    return HSA_STATUS_SUCCESS;
}

int
main(void)
{
    typedef hsa_status_t (*hsa_init_fn_t)(void);
    typedef hsa_status_t (*hsa_iterate_agents_fn_t)(hsa_status_t (*)(hsa_agent_t, void*), void*);
    typedef hsa_status_t (*hsa_shut_down_fn_t)(void);

    int                    count     = 0;
    void*                  handle    = NULL;
    hsa_init_fn_t          hsa_init_fn = NULL;
    hsa_iterate_agents_fn_t hsa_iterate_agents_fn = NULL;
    hsa_shut_down_fn_t     hsa_shut_down_fn = NULL;

    handle = dlopen("libhsa-runtime64.so", RTLD_NOW | RTLD_LOCAL);
    if(handle == NULL)
    {
        fprintf(stderr, "dlopen(libhsa-runtime64.so) failed: %s\n", dlerror());
        return 10;
    }

    hsa_init_fn = (hsa_init_fn_t) dlsym(handle, "hsa_init");
    hsa_iterate_agents_fn = (hsa_iterate_agents_fn_t) dlsym(handle, "hsa_iterate_agents");
    hsa_shut_down_fn = (hsa_shut_down_fn_t) dlsym(handle, "hsa_shut_down");

    if(hsa_init_fn == NULL || hsa_iterate_agents_fn == NULL || hsa_shut_down_fn == NULL)
    {
        fprintf(stderr, "failed to resolve HSA entry points\n");
        dlclose(handle);
        return 11;
    }

    if(hsa_init_fn() != HSA_STATUS_SUCCESS)
    {
        fprintf(stderr, "hsa_init failed\n");
        dlclose(handle);
        return 12;
    }

    for(int i = 0; i < 2000; ++i)
    {
        count = 0;
        if(hsa_iterate_agents_fn(&count_agents, &count) != HSA_STATUS_SUCCESS)
        {
            fprintf(stderr, "hsa_iterate_agents failed\n");
            hsa_shut_down_fn();
            dlclose(handle);
            return 13;
        }
        usleep(1000);
    }

    fprintf(stderr, "hsa probe loop done (agents=%d)\n", count);

    if(hsa_shut_down_fn() != HSA_STATUS_SUCCESS)
    {
        fprintf(stderr, "hsa_shut_down failed\n");
        dlclose(handle);
        return 14;
    }

    dlclose(handle);
    sleep(8);
    return 0;
}
