// Regression test for AIPROFSYST-787.
//
// What this tests
// ---------------
// ROCPROFILER_REGISTER_DEFINE_IMPORT(rocshmem, VERSION) in api_trace.cc
// (gated on #if defined(ROCSHMEM_ROCPROFILER_REGISTER)) generates:
//
//   extern "C" uint32_t rocprofiler_register_import_rocshmem() { return VERSION; }
//
// If CMakeLists.txt ever drops set(ROCSHMEM_ROCPROFILER_REGISTER 1), or
// #cmakedefine01 ROCSHMEM_ROCPROFILER_REGISTER is dropped from rocshmem_config.h.in,
// that symbol disappears from librocshmem.a and two things break:
//
//   (a) This program fails to LINK:
//         undefined reference to `rocprofiler_register_import_rocshmem'
//
//   (b) Even if the linker somehow succeeds, the RUNTIME check below fails
//       because the function would return 0.
//
// No GPUs, MPI, or rocshmem_init() required.

#include <rocprofiler-register/rocprofiler-register.h>
#include <rocshmem/rocshmem_config.h>   // ROCSHMEM_VENDOR_*_VERSION, ROCSHMEM_VERSION
                                         // also defines ROCSHMEM_ROCPROFILER_REGISTER

// Compile-time guard: if ROCSHMEM_ROCPROFILER_REGISTER is absent from
// rocshmem_config.h the root cause is a missing #cmakedefine or a missing
// set(ROCSHMEM_ROCPROFILER_REGISTER 1) in CMakeLists.txt.
#if !defined(ROCSHMEM_ROCPROFILER_REGISTER)
#    error "ROCSHMEM_ROCPROFILER_REGISTER not defined in rocshmem_config.h. "\
           "Restore set(ROCSHMEM_ROCPROFILER_REGISTER 1) in CMakeLists.txt "\
           "and ensure #cmakedefine01 ROCSHMEM_ROCPROFILER_REGISTER is in "\
           "cmake/rocshmem_config.h.in"
#endif

#include <cstdint>
#include <cstdio>
#include <cstdlib>

// Declare via the same macro used inside rocprofiler-register so the name is
// not duplicated: ROCPROFILER_REGISTER_IMPORT_FUNC(rocshmem) expands to
// rocprofiler_register_import_rocshmem.
extern "C" uint32_t ROCPROFILER_REGISTER_IMPORT_FUNC(rocshmem)(void);

int main()
{
    uint32_t version = ROCPROFILER_REGISTER_IMPORT_FUNC(rocshmem)();

    // The version must be non-zero.
    if(version == 0)
    {
        fprintf(stderr,
                "FAIL: rocprofiler_register_import_rocshmem() returned 0\n"
                "      Likely cause: ROCSHMEM_ROCPROFILER_REGISTER=1 was not set "
                "at compile time.\n");
        return EXIT_FAILURE;
    }

    // The version must match what was compiled into the rocshmem headers.
    // ROCPROFILER_REGISTER_COMPUTE_VERSION_3 = 10000*MAJOR + 100*MINOR + PATCH
    uint32_t expected = ROCPROFILER_REGISTER_COMPUTE_VERSION_3(
        ROCSHMEM_VENDOR_MAJOR_VERSION,
        ROCSHMEM_VENDOR_MINOR_VERSION,
        ROCSHMEM_VENDOR_PATCH_VERSION);

    if(version != expected)
    {
        fprintf(stderr,
                "FAIL: rocprofiler_register_import_rocshmem() returned %u, "
                "expected %u (rocshmem %s)\n",
                version, expected, ROCSHMEM_VERSION);
        return EXIT_FAILURE;
    }

    printf("PASS: rocprofiler_register_import_rocshmem() = %u  (rocshmem %s)\n",
           version, ROCSHMEM_VERSION);
    return EXIT_SUCCESS;
}
