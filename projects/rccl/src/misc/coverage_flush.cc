/*************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

// Exported wrappers around the LLVM source-based coverage profile runtime,
// compiled into librccl.so only when ENABLE_CODE_COVERAGE is on.
//
// Both librccl.so and the unit-test executables link libclang_rt.profile.a
// statically, so each binary owns a *private* instance of the profile
// runtime (with its own counters, its own __llvm_profile_write_file, and its
// own filename buffer). The runtime's symbols are pulled in with hidden
// visibility from the static archive, so the test binary cannot reach
// librccl's writer via dlsym.
//
// To support the process-isolated test harness (which calls _exit() in the
// child and therefore bypasses the runtime's atexit() handler), we expose
// these explicit, default-visibility wrappers so the test side can flush
// librccl's coverage data for each forked-and-_exited child process.

extern "C" int  __llvm_profile_write_file(void);
extern "C" void __llvm_profile_set_filename(const char* name);

extern "C" __attribute__((visibility("default")))
int rcclCoverageWriteFile(void)
{
    return __llvm_profile_write_file();
}

extern "C" __attribute__((visibility("default")))
void rcclCoverageSetFilename(const char* name)
{
    __llvm_profile_set_filename(name);
}
