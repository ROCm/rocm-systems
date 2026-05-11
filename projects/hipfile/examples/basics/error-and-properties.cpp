/* Copyright (c) Advanced Micro Devices, Inc. All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 */

/* error-and-properties - Smoke-test hipFile's error-string mapping and the
 * driver-properties round-trip. No files or GPU memory are touched.
 *
 * Usage: ./error-and-properties
 *
 *   No arguments. Exits 0 if all checks pass, non-zero on any mismatch.
 *
 * Steps:
 *   1. Verify hipFileGetOpErrorString for known error codes
 *   2. Configure driver (poll mode, max cache size, max pinned mem)
 *   3. hipFileDriverOpen
 *   4. hipFileDriverGetProperties + print
 *   5. Assert properties match configuration
 */

#include <hipfile.h>
#include <hip/hip_runtime_api.h>

#include <cinttypes>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>

/// @brief Configured driver values asserted in step 5. Multiples of 4 KiB to
/// satisfy the NVIDIA-side alignment note in hipFileDriverProps_t.
#ifndef EAP_CACHE_KB
#define EAP_CACHE_KB ((size_t)(16UL * 1024UL)) /* 16 MiB */
#endif
#ifndef EAP_PINNED_KB
#define EAP_PINNED_KB ((size_t)(8UL * 1024UL)) /* 8 MiB */
#endif

/// @brief Test if `bit_index` is set in `flags`. Driver flag enums in
/// hipfile.h are bit indices, not bitmasks.
static inline bool
flag_set(unsigned flags, unsigned bit_index)
{
    return (flags & (1u << bit_index)) != 0;
}

static const char *
yesno(bool b)
{
    return b ? "yes" : "no";
}

/* ------------------------------------------------------------------------- */
/* Step 1: error string sanity                                               */
/* ------------------------------------------------------------------------- */

static int
check_error_strings(void)
{
    static const hipFileOpError_t codes[] = {
        hipFileSuccess,
        hipFileDriverNotInitialized,
        hipFileDIONotSet,
        hipFileMemoryAlreadyRegistered,
        hipFileMemoryNotRegistered,
        hipFileInvalidValue,
        hipFileHandleAlreadyRegistered,
    };
    const size_t n = sizeof(codes) / sizeof(codes[0]);

    int failures = 0;
    printf("=== Step 1: hipFileGetOpErrorString ===\n");
    for (size_t i = 0; i < n; ++i) {
        const char *s = hipFileGetOpErrorString(codes[i]);
        if (s == NULL || s[0] == '\0') {
            fprintf(stderr, "[FAIL] hipFileGetOpErrorString(%d) returned %s\n", (int)codes[i],
                    s == NULL ? "NULL" : "\"\"");
            ++failures;
            continue;
        }
        printf("  [%d] -> \"%s\"\n", (int)codes[i], s);
    }
    return failures;
}

/* ------------------------------------------------------------------------- */
/* Step 2: configure driver (must be done while the driver is closed)         */
/* ------------------------------------------------------------------------- */

static int
configure_driver(void)
{
    hipFileError_t err;

    err = hipFileSetParameterBool(hipFileParamPropertiesUsePollMode, true);
    if (hipFileSuccess != err.err) {
        fprintf(stderr, "Could not enable poll mode (%s)\n", HIPFILE_ERRSTR(err.err));
        return 1;
    }

    err = hipFileSetParameterSizeT(hipFileParamPropertiesMaxDeviceCacheSizeKB, EAP_CACHE_KB);
    if (hipFileSuccess != err.err) {
        fprintf(stderr, "Could not set max device cache size (%s)\n", HIPFILE_ERRSTR(err.err));
        return 1;
    }

    err = hipFileSetParameterSizeT(hipFileParamPropertiesMaxDevicePinnedMemSizeKB, EAP_PINNED_KB);
    if (hipFileSuccess != err.err) {
        fprintf(stderr, "Could not set max device pinned mem size (%s)\n", HIPFILE_ERRSTR(err.err));
        return 1;
    }

    return 0;
}

/* ------------------------------------------------------------------------- */
/* Step 4: print every field of hipFileDriverProps_t                          */
/* ------------------------------------------------------------------------- */

static void
print_properties(const hipFileDriverProps_t *p)
{
    printf("\n=== Step 4: hipFileDriverGetProperties ===\n");
    printf("nvfs.major_version          : %u\n", p->nvfs.major_version);
    printf("nvfs.minor_version          : %u\n", p->nvfs.minor_version);
    printf("nvfs.poll_thresh_size       : %" PRIu64 " KiB\n", p->nvfs.poll_thresh_size);
    printf("nvfs.max_direct_io_size     : %" PRIu64 " KiB\n", p->nvfs.max_direct_io_size);

    printf("nvfs.driver_status_flags    : 0x%x\n", p->nvfs.driver_status_flags);
    printf("  Lustre                    : %s\n",
           yesno(flag_set(p->nvfs.driver_status_flags, hipFileLustreSupported)));
    printf("  WekaFS                    : %s\n",
           yesno(flag_set(p->nvfs.driver_status_flags, hipFileWekaFSSupported)));
    printf("  NFS                       : %s\n",
           yesno(flag_set(p->nvfs.driver_status_flags, hipFileNFSSupported)));
    printf("  GPFS                      : %s\n",
           yesno(flag_set(p->nvfs.driver_status_flags, hipFileGPFSSupported)));
    printf("  Local NVMe                : %s\n",
           yesno(flag_set(p->nvfs.driver_status_flags, hipFileNVMeSupported)));
    printf("  NVMeoF                    : %s\n",
           yesno(flag_set(p->nvfs.driver_status_flags, hipFileNVMeoFSupported)));
    printf("  SCSI                      : %s\n",
           yesno(flag_set(p->nvfs.driver_status_flags, hipFileSCSISupported)));
    printf("  ScaleFlux CSD             : %s\n",
           yesno(flag_set(p->nvfs.driver_status_flags, hipFileScaleFluxCSDSupported)));
    printf("  NVMesh                    : %s\n",
           yesno(flag_set(p->nvfs.driver_status_flags, hipFileNVMeshSupported)));
    printf("  BeeGFS                    : %s\n",
           yesno(flag_set(p->nvfs.driver_status_flags, hipFileBeeGFSSupported)));
    printf("  NVMeP2P                   : %s\n",
           yesno(flag_set(p->nvfs.driver_status_flags, hipFileNVMeP2PSupported)));
    printf("  ScateFS                   : %s\n",
           yesno(flag_set(p->nvfs.driver_status_flags, hipFileScatefsSupported)));

    printf("nvfs.driver_control_flags   : 0x%x\n", p->nvfs.driver_control_flags);
    printf("  poll mode                 : %s\n",
           yesno(flag_set(p->nvfs.driver_control_flags, hipFileUsePollMode)));
    printf("  compat mode               : %s\n",
           yesno(flag_set(p->nvfs.driver_control_flags, hipFileAllowCompatMode)));

    printf("feature_flags               : 0x%x\n", p->feature_flags);
    printf("  dynamic routing           : %s\n",
           yesno(flag_set(p->feature_flags, hipFileDynRoutingSupported)));
    printf("  batch IO                  : %s\n", yesno(flag_set(p->feature_flags, hipFileBatchIOSupported)));
    printf("  streams                   : %s\n", yesno(flag_set(p->feature_flags, hipFileStreamsSupported)));
    printf("  parallel IO               : %s\n",
           yesno(flag_set(p->feature_flags, hipFileParallelIOSupported)));

    printf("max_device_cache_size       : %" PRIu64 " KiB\n", p->max_device_cache_size);
    printf("per_buffer_cache_size       : %" PRIu64 " KiB\n", p->per_buffer_cache_size);
    printf("max_device_pinned_mem_size  : %" PRIu64 " KiB\n", p->max_device_pinned_mem_size);
    printf("max_batch_io_count          : %u\n", p->max_batch_io_count);
    printf("max_batch_io_timeout_msecs  : %u ms\n", p->max_batch_io_timeout_msecs);
}

/* ------------------------------------------------------------------------- */
/* Step 5: assert reported values match what we configured                    */
/* ------------------------------------------------------------------------- */

static int
assert_properties(const hipFileDriverProps_t *p)
{
    int failures = 0;
    printf("\n=== Step 5: configuration assertions ===\n");

    if (flag_set(p->nvfs.driver_control_flags, hipFileUsePollMode)) {
        printf("[OK]   poll mode enabled in driver_control_flags\n");
    }
    else {
        fprintf(stderr, "[FAIL] poll mode NOT set in driver_control_flags (0x%x)\n",
                p->nvfs.driver_control_flags);
        ++failures;
    }

    if (p->max_device_cache_size == (uint64_t)EAP_CACHE_KB) {
        printf("[OK]   max_device_cache_size == %zu KiB\n", (size_t)EAP_CACHE_KB);
    }
    else {
        fprintf(stderr, "[FAIL] max_device_cache_size: got %" PRIu64 " KiB, expected %zu KiB\n",
                p->max_device_cache_size, (size_t)EAP_CACHE_KB);
        ++failures;
    }

    if (p->max_device_pinned_mem_size == (uint64_t)EAP_PINNED_KB) {
        printf("[OK]   max_device_pinned_mem_size == %zu KiB\n", (size_t)EAP_PINNED_KB);
    }
    else {
        fprintf(stderr, "[FAIL] max_device_pinned_mem_size: got %" PRIu64 " KiB, expected %zu KiB\n",
                p->max_device_pinned_mem_size, (size_t)EAP_PINNED_KB);
        ++failures;
    }

    return failures;
}

int
main(void)
{
    int            failures    = 0;
    bool           driver_open = false;
    int            exit_status = EXIT_FAILURE;
    hipFileError_t err;

    /* 1. Verify hipFileGetOpErrorString for known error codes */
    failures += check_error_strings();

    /* 2. Configure driver (parameters must be set while driver is closed) */
    printf("\n=== Step 2: configure driver ===\n");
    if (configure_driver())
        return EXIT_FAILURE;
    printf("  poll mode                  : on\n");
    printf("  max_device_cache_size      : %zu KiB\n", (size_t)EAP_CACHE_KB);
    printf("  max_device_pinned_mem_size : %zu KiB\n", (size_t)EAP_PINNED_KB);

    /* 3. hipFileDriverOpen */
    printf("\n=== Step 3: hipFileDriverOpen ===\n");
    err = hipFileDriverOpen();
    if (hipFileSuccess != err.err) {
        fprintf(stderr, "hipFileDriverOpen failed (%s)\n", HIPFILE_ERRSTR(err.err));
        return EXIT_FAILURE;
    }
    driver_open = true;
    printf("  driver opened (use count = %" PRId64 ")\n", hipFileUseCount());

    /* 4. hipFileDriverGetProperties + print */
    {
        hipFileDriverProps_t props;
        memset(&props, 0, sizeof(props));
        err = hipFileDriverGetProperties(&props);
        if (hipFileSuccess != err.err) {
            fprintf(stderr, "hipFileDriverGetProperties failed (%s)\n", HIPFILE_ERRSTR(err.err));
            goto driver_close;
        }
        print_properties(&props);

        /* 5. Assert properties match configuration */
        failures += assert_properties(&props);
    }

    exit_status = (failures == 0) ? EXIT_SUCCESS : EXIT_FAILURE;

driver_close:
    if (driver_open) {
        err = hipFileDriverClose();
        if (hipFileSuccess != err.err) {
            fprintf(stderr, "hipFileDriverClose failed (%s)\n", HIPFILE_ERRSTR(err.err));
            exit_status = EXIT_FAILURE;
        }
    }

    printf("\n");
    if (exit_status == EXIT_SUCCESS) {
        printf("All checks passed.\n");
    }
    else {
        fprintf(stderr, "%d check(s) failed.\n", failures);
    }
    return exit_status;
}
