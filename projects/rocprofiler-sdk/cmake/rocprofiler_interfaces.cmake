#
#
# Forward declaration of all INTERFACE targets
#
#

include(rocprofiler_utilities)

#
# interfaces for build flags
#
rocprofiler_add_interface_library(
    rocprofiler-headers
    "Provides minimal set of include flags to compile with rocprofiler")
rocprofiler_add_interface_library(
    rocprofiler-build-flags "Provides generalized build flags for rocprofiler" INTERNAL)
rocprofiler_add_interface_library(rocprofiler-threading "Enables multithreading support"
                                  INTERNAL)
rocprofiler_add_interface_library(rocprofiler-perfetto "Enables Perfetto support"
                                  INTERNAL)
rocprofiler_add_interface_library(rocprofiler-compile-definitions "Compile definitions"
                                  INTERNAL)
rocprofiler_add_interface_library(rocprofiler-static-libgcc
                                  "Link to static version of libgcc" INTERNAL)
rocprofiler_add_interface_library(rocprofiler-static-libstdcxx
                                  "Link to static version of libstdc++" INTERNAL)
rocprofiler_add_interface_library(
    rocprofiler-developer-flags "Compiler flags for developers (more warnings, etc.)"
    INTERNAL)
rocprofiler_add_interface_library(rocprofiler-debug-flags
                                  "Compiler flags for more debug info" INTERNAL)
rocprofiler_add_interface_library(rocprofiler-release-flags
                                  "Compiler flags for more debug info" INTERNAL)
rocprofiler_add_interface_library(rocprofiler-stack-protector
                                  "Adds stack-protector compiler flags" INTERNAL)
rocprofiler_add_interface_library(rocprofiler-memcheck INTERFACE INTERNAL)

#
# interfaces for libraries
#
rocprofiler_add_interface_library(rocprofiler-dl
                                  "Build flags for dynamic linking library" INTERNAL)
rocprofiler_add_interface_library(rocprofiler-rt "Build flags for runtime library"
                                  INTERNAL)
rocprofiler_add_interface_library(rocprofiler-hip "HIP library" INTERNAL)
rocprofiler_add_interface_library(rocprofiler-hsa-runtime "HSA runtime library" INTERNAL)
rocprofiler_add_interface_library(rocprofiler-amd-comgr "AMD comgr library" INTERNAL)
rocprofiler_add_interface_library(rocprofiler-gtest "Google Test library" INTERNAL)
rocprofiler_add_interface_library(rocprofiler-glog "Google Log library" INTERNAL)
rocprofiler_add_interface_library(rocprofiler-fmt "C++ format string library" INTERNAL)
rocprofiler_add_interface_library(rocprofiler-stdcxxfs "C++ filesystem library" INTERNAL)
rocprofiler_add_interface_library(rocprofiler-ptl "Parallel Tasking Library" INTERNAL)
