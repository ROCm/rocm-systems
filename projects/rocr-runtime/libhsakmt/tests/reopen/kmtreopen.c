#include <dlfcn.h>
#include <stdio.h>
#include <stdlib.h>
#include <hsakmt.h>

HSAKMT_STATUS HSAKMTAPI (*pfn_hsaKmtOpenKFD)(void);
HSAKMT_STATUS HSAKMTAPI (*pfn_hsaKmtCloseKFD)(void);
HSAKMT_STATUS HSAKMTAPI (*pfn_hsaKmtGetVersion)(HsaVersionInfo* VersionInfo);
HSAKMT_STATUS HSAKMTAPI (*pfn_hsaKmtAcquireSystemProperties)(HsaSystemProperties* SystemProperties);
HSAKMT_STATUS HSAKMTAPI (*pfn_hsaKmtReleaseSystemProperties)(void);

HsaVersionInfo g_versionInfo;
HsaSystemProperties g_systemProperties;

static void hsa_perror(const char *s, HSAKMT_STATUS status)
{
    static const char *errorStrings[] = {
        [HSAKMT_STATUS_SUCCESS] = "Success",
        [HSAKMT_STATUS_ERROR] = "General error",
        [HSAKMT_STATUS_DRIVER_MISMATCH] = "Driver mismatch",
        [HSAKMT_STATUS_INVALID_PARAMETER] = "Invalid parameter",
        [HSAKMT_STATUS_INVALID_HANDLE] = "Invalid handle",
        [HSAKMT_STATUS_INVALID_NODE_UNIT] = "Invalid node or unit",
        [HSAKMT_STATUS_NO_MEMORY] = "No memory",
        [HSAKMT_STATUS_BUFFER_TOO_SMALL] = "Buffer too small",
        [HSAKMT_STATUS_NOT_IMPLEMENTED] = "Not implemented",
        [HSAKMT_STATUS_NOT_SUPPORTED] = "Not supported",
        [HSAKMT_STATUS_UNAVAILABLE] = "Unavailable",
        [HSAKMT_STATUS_KERNEL_IO_CHANNEL_NOT_OPENED] = "Kernel IO channel not opened",
        [HSAKMT_STATUS_KERNEL_COMMUNICATION_ERROR] = "Kernel communication error",
        [HSAKMT_STATUS_KERNEL_ALREADY_OPENED] = "Kernel already opened",
        [HSAKMT_STATUS_HSAMMU_UNAVAILABLE] = "HSA MMU unavailable",
        [HSAKMT_STATUS_WAIT_FAILURE] = "Wait failure",
        [HSAKMT_STATUS_WAIT_TIMEOUT] = "Wait timeout",
        [HSAKMT_STATUS_MEMORY_ALREADY_REGISTERED] = "Memory already registered",
        [HSAKMT_STATUS_MEMORY_NOT_REGISTERED] = "Memory not registered",
        [HSAKMT_STATUS_MEMORY_ALIGNMENT] = "Memory alignment error"
    };

    if (status >= 0 && status <= HSAKMT_STATUS_MEMORY_ALIGNMENT)
        fprintf(stderr, "%s: %s\n", s, errorStrings[status]);
    else
        fprintf(stderr, "%s: Unknown error %d\n", s, status);
}

#define HSA_CHECK_RETURN(call) do {             \
        HSAKMT_STATUS __ret;                    \
        printf("  Calling %s\n", #call);        \
        __ret = pfn_##call;                     \
        if (__ret != HSAKMT_STATUS_SUCCESS) {   \
            hsa_perror(#call, __ret);           \
            return __ret;                       \
        }                                       \
    } while(0)

#define HSA_DLSYM(handle, func) do {                            \
        pfn_##func = dlsym(handle, #func);                      \
        if (pfn_##func == NULL) {                               \
            fprintf(stderr, "dlsym failed: %s\n", dlerror());   \
            return HSAKMT_STATUS_ERROR;                         \
        }                                                       \
    } while(0)

static int runTest(void *handle)
{
    HSA_DLSYM(handle, hsaKmtOpenKFD);
    HSA_DLSYM(handle, hsaKmtCloseKFD);
    HSA_DLSYM(handle, hsaKmtGetVersion);
    HSA_DLSYM(handle, hsaKmtAcquireSystemProperties);
    HSA_DLSYM(handle, hsaKmtReleaseSystemProperties);

    HSA_CHECK_RETURN(hsaKmtOpenKFD());
    HSA_CHECK_RETURN(hsaKmtGetVersion(&g_versionInfo));
    HSA_CHECK_RETURN(hsaKmtAcquireSystemProperties(&g_systemProperties));

    HSA_CHECK_RETURN(hsaKmtReleaseSystemProperties());
    HSA_CHECK_RETURN(hsaKmtCloseKFD());

    return HSAKMT_STATUS_SUCCESS;
}

int main(int argc, char *argv[])
{
    void *handle;
    int i;

    for (i = 0; i < 5; i++) {
        printf("Iteration %d:\n  Loading libhsakmt.so\n", i+1);

        handle = dlopen("libhsakmt.so", RTLD_LAZY);
        if (handle == NULL) {
            fprintf(stderr, "dlopen failed: %s\n", dlerror());
            exit(1);
        }

        if (runTest(handle) != HSAKMT_STATUS_SUCCESS)
            exit(1);

        printf("  Unloading libhsakmt.so\n");
        if (dlclose(handle) != 0) {
            fprintf(stderr, "dlclose failed: %s\n", dlerror());
            exit(1);
        }
    }
}
