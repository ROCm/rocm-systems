// MIT License
//
// Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
// THE SOFTWARE.
//
// Reproducer R5: a failed snapshot or restore copy is logged and ignored.
//
// memory_snapshot.cpp treats a failed hsa_memory_copy as a warning and `continue`s, in both
// snap() and restore(). The replay then proceeds with a partial snapshot, so later passes
// run against inputs that were never reverted, and the counter records are emitted as if
// the passes were equivalent. The tool is told nothing.
//
// This shim makes the Nth hsa_memory_copy fail so the path can be exercised deliberately.
//
//   gcc -shared -fPIC -O2 -o libcopyfail.so r5_copy_failure_shim.c -ldl
//   KR_FAIL_COPY_AFTER=4 LD_PRELOAD=./libcopyfail.so:./librepro_client.so rocprofv3
//     --pmc SQ_WAVES SQ_INSTS_VALU GRBM_COUNT
//     --pmc SQ_WAVES SQ_INSTS_VALU GRBM_GUI_ACTIVE
//     --kernel-replay-beta-enabled --output-format json -d out -o r5 -- ./your_app
//
// Expected on a correct implementation: the replay fails loudly, or the affected records are
// marked invalid, so a consumer cannot mistake them for sound measurements.
// Observed: "replay snapshot: device->host copy failed" / "replay restore: host->device copy
// failed" in the log at INFO/WARNING level, exit status 0, and a results file that looks
// entirely normal.
//
// Set KR_FAIL_COPY_AFTER to the number of successful copies to allow first. Small values hit
// the snapshot, larger ones hit a restore between passes.

#define _GNU_SOURCE
#include <dlfcn.h>
#include <stdatomic.h>
#include <stdio.h>
#include <stdlib.h>

// Mirrors the HSA signature without pulling in the headers: both are opaque here.
typedef int (*hsa_memory_copy_fn)(void* dst, const void* src, unsigned long size);

static atomic_long g_calls          = 0;
static long        g_fail_after     = -1;
static int         g_reported       = 0;
static const int   HSA_STATUS_ERROR = 0x1000;  // any non-success value

__attribute__((constructor)) static void
init(void)
{
    const char* env = getenv("KR_FAIL_COPY_AFTER");
    g_fail_after    = env ? strtol(env, NULL, 10) : -1;
    fprintf(stderr, "[copyfail] arming: fail hsa_memory_copy after %ld successful calls\n",
            g_fail_after);
}

int
hsa_memory_copy(void* dst, const void* src, unsigned long size)
{
    static hsa_memory_copy_fn real = NULL;
    if(!real) real = (hsa_memory_copy_fn) dlsym(RTLD_NEXT, "hsa_memory_copy");

    long n = atomic_fetch_add(&g_calls, 1) + 1;
    if(g_fail_after >= 0 && n > g_fail_after)
    {
        if(!g_reported)
        {
            fprintf(stderr,
                    "[copyfail] failing hsa_memory_copy call #%ld (dst=%p size=%lu). "
                    "A correct implementation must not silently continue.\n",
                    n, dst, size);
            g_reported = 1;
        }
        return HSA_STATUS_ERROR;
    }
    if(!real)
    {
        fprintf(stderr, "[copyfail] could not resolve the real hsa_memory_copy\n");
        return HSA_STATUS_ERROR;
    }
    return real(dst, src, size);
}
