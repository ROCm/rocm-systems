/*************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

// Exported wrapper around the LLVM source-based coverage profile runtime,
// compiled into librccl.so only when ENABLE_CODE_COVERAGE is on.
//
// librccl.so and the test binaries each statically link libclang_rt.profile.a,
// giving each its own private profile runtime instance. The test binary cannot
// reach librccl's __llvm_profile_write_file via dlsym because the runtime
// symbols have hidden visibility. This default-visibility wrapper lets the
// process-isolated test harness explicitly flush librccl's coverage before
// calling _exit().

extern "C" int __llvm_profile_write_file(void);

extern "C" __attribute__((visibility("default")))
int rcclCoverageWriteFile(void)
{
    return __llvm_profile_write_file();
}
