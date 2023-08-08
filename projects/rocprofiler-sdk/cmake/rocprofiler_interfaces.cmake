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
rocprofiler_add_interface_library(rocprofiler-build-flags
                                  "Provides generalized build flags for rocprofiler")
rocprofiler_add_interface_library(rocprofiler-threading "Enables multithreading support")
rocprofiler_add_interface_library(rocprofiler-perfetto "Enables Perfetto support")
rocprofiler_add_interface_library(rocprofiler-compile-definitions "Compile definitions")
rocprofiler_add_interface_library(rocprofiler-static-libgcc
                                  "Link to static version of libgcc")
rocprofiler_add_interface_library(rocprofiler-static-libstdcxx
                                  "Link to static version of libstdc++")
rocprofiler_add_interface_library(rocprofiler-developer-flags
                                  "Compiler flags for developers (more warnings, etc.)")
rocprofiler_add_interface_library(rocprofiler-debug-flags
                                  "Compiler flags for more debug info")
rocprofiler_add_interface_library(rocprofiler-release-flags
                                  "Compiler flags for more debug info")
rocprofiler_add_interface_library(rocprofiler-stack-protector
                                  "Adds stack-protector compiler flags")
rocprofiler_add_interface_library(rocprofiler-memcheck INTERFACE)

#
# interfaces for libraries
#
rocprofiler_add_interface_library(rocprofiler-dl
                                  "Build flags for dynamic linking library")
rocprofiler_add_interface_library(rocprofiler-rt "Build flags for runtime library")
rocprofiler_add_interface_library(rocprofiler-hip "HIP library")
rocprofiler_add_interface_library(rocprofiler-hsa-runtime "HSA runtime library")
rocprofiler_add_interface_library(rocprofiler-amd-comgr "AMD comgr library")
rocprofiler_add_interface_library(rocprofiler-googletest "Google Test library" INTERNAL)
rocprofiler_add_interface_library(rocprofiler-stdcxxfs "C++ filesystem library" INTERNAL)
