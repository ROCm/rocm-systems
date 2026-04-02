/* Copyright(C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * Smoke test for the local-library-path resolution logic used by the
 * ICD Loader on Windows (khrIcdGetLocalLibraryPath in icd_windows.c).
 *
 * The test uses the executable's own directory as a stand-in for
 * OpenCL.dll's directory:
 *   - creates a dummy file there,
 *   - verifies that basename extraction, path construction, and the
 *     GetFileAttributesA existence check produce correct results,
 *   - cleans up.
 *
 * No registry access is required.
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <windows.h>

static int g_failures = 0;
static int g_passes   = 0;

static void check(int cond, const char *description) {
    if (cond) {
        printf("  PASS: %s\n", description);
        ++g_passes;
    } else {
        printf("  FAIL: %s\n", description);
        ++g_failures;
    }
}

/* ------------------------------------------------------------------ */
/* Replicates the basename-extraction loop from                        */
/* khrIcdGetLocalLibraryPath (icd_windows.c).                          */
/* ------------------------------------------------------------------ */
static const char *extractBaseName(const char *libraryName) {
    const char *baseName = libraryName;
    const char *c;
    for (c = libraryName; *c; ++c) {
        if (*c == '\\' || *c == '/') {
            baseName = c + 1;
        }
    }
    return baseName;
}

/* ------------------------------------------------------------------ */
/* Replicates the "construct local path and check existence" portion.   */
/* dir       – directory with trailing backslash                       */
/* dirLen    – strlen(dir)                                             */
/* inputPath – full path as it would come from the registry            */
/* Returns 1 if a local copy exists, 0 otherwise.                      */
/* outPath receives the constructed path (must be MAX_PATH bytes).     */
/* ------------------------------------------------------------------ */
static int resolveLocalPath(const char *dir, size_t dirLen,
                            const char *inputPath, char *outPath) {
    const char *baseName = extractBaseName(inputPath);

    if (*baseName == '\0')
        return 0;

    if (dirLen + strlen(baseName) >= MAX_PATH)
        return 0;

    memcpy(outPath, dir, dirLen);
    strcpy(outPath + dirLen, baseName);

    return GetFileAttributesA(outPath) != INVALID_FILE_ATTRIBUTES;
}

/* ------------------------------------------------------------------ */
/* Tests                                                               */
/* ------------------------------------------------------------------ */

static void test_basename_extraction(void) {
    printf("basename extraction:\n");

    check(strcmp(extractBaseName("C:\\Windows\\System32\\vendor.dll"),
                "vendor.dll") == 0,
          "backslash path");

    check(strcmp(extractBaseName("C:/unix/style/vendor.dll"),
                "vendor.dll") == 0,
          "forward-slash path");

    check(strcmp(extractBaseName("C:\\mixed/slashes\\vendor.dll"),
                "vendor.dll") == 0,
          "mixed separators");

    check(strcmp(extractBaseName("vendor.dll"),
                "vendor.dll") == 0,
          "bare filename (no directory)");

    check(*extractBaseName("C:\\trailing\\") == '\0',
          "trailing backslash yields empty basename");

    check(*extractBaseName("") == '\0',
          "empty string yields empty basename");
}

static void test_local_resolution(const char *exeDir, size_t exeDirLen,
                                  const char *dummyFileName) {
    char resolved[MAX_PATH];
    char inputFull[MAX_PATH];

    printf("local path resolution:\n");

    /* File that EXISTS in the exe directory. */
    snprintf(inputFull, MAX_PATH,
             "C:\\DriverStore\\SomeVendor\\%s", dummyFileName);
    check(resolveLocalPath(exeDir, exeDirLen, inputFull, resolved) == 1,
          "local copy found when file exists");

    /* The resolved path must point into exeDir. */
    check(strncmp(resolved, exeDir, exeDirLen) == 0,
          "resolved path starts with exe directory");
    check(strcmp(resolved + exeDirLen, dummyFileName) == 0,
          "resolved path ends with base filename");

    /* File that does NOT exist locally. */
    check(resolveLocalPath(exeDir, exeDirLen,
                           "C:\\no\\such\\nonexistent_98765.dll",
                           resolved) == 0,
          "returns 0 for nonexistent local file");

    /* Empty basename (trailing separator). */
    check(resolveLocalPath(exeDir, exeDirLen,
                           "C:\\path\\", resolved) == 0,
          "returns 0 for empty basename");

    /* Forward-slash input with an existing local file. */
    snprintf(inputFull, MAX_PATH,
             "C:/unix/path/%s", dummyFileName);
    check(resolveLocalPath(exeDir, exeDirLen, inputFull, resolved) == 1,
          "forward-slash input still finds local copy");
}

static void test_max_path_guard(const char *exeDir, size_t exeDirLen) {
    char resolved[MAX_PATH];
    char longInput[MAX_PATH + 64];

    printf("MAX_PATH overflow guard:\n");

    /* Construct a basename that, combined with exeDir, exceeds MAX_PATH. */
    size_t need = MAX_PATH - exeDirLen + 1;   /* guaranteed overflow */
    memset(longInput, 'x', need);
    memcpy(longInput + need - 4, ".dll", 4);
    longInput[need] = '\0';

    check(resolveLocalPath(exeDir, exeDirLen, longInput, resolved) == 0,
          "overlong basename rejected");
}

/* ------------------------------------------------------------------ */

int main(void) {
    /* Determine this executable's directory (stand-in for OpenCL.dll dir). */
    char exeDir[MAX_PATH] = {0};
    DWORD pathLen = GetModuleFileNameA(NULL, exeDir, MAX_PATH);
    if (pathLen == 0 || pathLen >= MAX_PATH) {
        printf("FATAL: cannot determine exe directory\n");
        return 1;
    }
    char *lastSlash = strrchr(exeDir, '\\');
    if (!lastSlash) {
        printf("FATAL: exe path has no backslash\n");
        return 1;
    }
    *(lastSlash + 1) = '\0';
    size_t exeDirLen = (size_t)(lastSlash + 1 - exeDir);

    /* Create a dummy file to simulate a vendor DLL placed alongside OpenCL.dll. */
    const char *dummyName = "icd_test_local_dummy.dll";
    char dummyPath[MAX_PATH];
    snprintf(dummyPath, MAX_PATH, "%s%s", exeDir, dummyName);

    HANDLE hFile = CreateFileA(dummyPath, GENERIC_WRITE, 0, NULL,
                               CREATE_ALWAYS, FILE_ATTRIBUTE_NORMAL, NULL);
    if (hFile == INVALID_HANDLE_VALUE) {
        printf("FATAL: cannot create dummy file %s (error %lu)\n",
               dummyPath, GetLastError());
        return 1;
    }
    CloseHandle(hFile);

    /* Run tests. */
    test_basename_extraction();
    test_local_resolution(exeDir, exeDirLen, dummyName);
    test_max_path_guard(exeDir, exeDirLen);

    /* Cleanup. */
    DeleteFileA(dummyPath);

    printf("\n%d passed, %d failed\n",
           g_passes, g_failures);
    return g_failures == 0 ? 0 : 1;
}
