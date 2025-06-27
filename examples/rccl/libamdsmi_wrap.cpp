#define _GNU_SOURCE
#include <dlfcn.h>
#include <link.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

extern "C" void
__cxa_finalize(void* dso)
{
    static void (*original_finalize)(void*) = nullptr;
    if(!original_finalize)
    {
        original_finalize = (void (*)(void*)) dlsym(RTLD_NEXT, "__cxa_finalize");
        if(!original_finalize)
        {
            fprintf(stderr, "Failed to resolve __cxa_finalize\n");
            _exit(1);
        }
    }

    Dl_info info;
    if(dso && dladdr(dso, &info) && info.dli_fname)
    {
        const char* dso_name = info.dli_fname;
        if(strstr(dso_name, "libamd_smi.so.25"))
        {
            fprintf(stderr, "[wrap] Skipping __cxa_finalize for %s — force _exit(0)\n",
                    dso_name);
            _exit(0);  // Hard stop: prevents double-free in global destructors
        }
    }

    original_finalize(dso);
}
