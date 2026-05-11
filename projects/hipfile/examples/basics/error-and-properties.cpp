/* Copyright (c) Advanced Micro Devices, Inc. All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 */

/* error-and-properties - Smoke-test hipFile's error-string mapping.
 * No files, GPU memory, or driver state are touched.
 *
 * Usage: ./error-and-properties
 *
 *   No arguments. Exits 0 if every known error code maps to a non-empty
 *   string, non-zero on any miss.
 */

#include <hipfile.h>

#include <cstdio>
#include <cstdlib>

int
main(void)
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
    printf("=== hipFileGetOpErrorString ===\n");
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

    printf("\n");
    if (failures == 0) {
        printf("All checks passed.\n");
        return EXIT_SUCCESS;
    }
    fprintf(stderr, "%d check(s) failed.\n", failures);
    return EXIT_FAILURE;
}
