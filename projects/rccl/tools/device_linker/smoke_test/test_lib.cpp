#include <stdio.h>
#include <dlfcn.h>
int main() {
    printf("Library path check\n");
    Dl_info info;
    void* handle = dlopen("librccl.so.1", RTLD_NOW);
    if (handle) {
        printf("Loaded librccl.so.1\n");
        void* sym = dlsym(handle, "ncclAllReduce");
        if (dladdr(sym, &info)) {
            printf("ncclAllReduce from: %s\n", info.dli_fname);
        }
    }
    return 0;
}
