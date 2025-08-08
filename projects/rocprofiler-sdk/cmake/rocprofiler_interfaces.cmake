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
    rocprofiler-sdk-headers
    "Provides minimal set of include flags to compile with rocprofiler")
rocprofiler_add_interface_library(
    rocprofiler-sdk-build-flags "Provides generalized build flags for rocprofiler"
    INTERNAL)
rocprofiler_add_interface_library(rocprofiler-sdk-threading
                                  "Enables multithreading support" INTERNAL)
rocprofiler_add_interface_library(rocprofiler-sdk-perfetto "Enables Perfetto support"
                                  INTERNAL)
rocprofiler_add_interface_library(rocprofiler-sdk-otf2 "Enables OTF2 support" INTERNAL)
rocprofiler_add_interface_library(rocprofiler-sdk-cereal "Enables Cereal support"
                                  INTERNAL)
rocprofiler_add_interface_library(rocprofiler-sdk-sqlite3 "Enables SQLite3 support"
                                  INTERNAL)
rocprofiler_add_interface_library(rocprofiler-sdk-pybind11 "Enables PyBind11 support"
                                  INTERNAL)
rocprofiler_add_interface_library(rocprofiler-sdk-gotcha "Enables GOTCHA support"
                                  INTERNAL)
rocprofiler_add_interface_library(rocprofiler-sdk-compile-definitions
                                  "Compile definitions" INTERNAL)
rocprofiler_add_interface_library(rocprofiler-sdk-static-libgcc
                                  "Link to static version of libgcc" INTERNAL)
rocprofiler_add_interface_library(rocprofiler-sdk-static-libstdcxx
                                  "Link to static version of libstdc++" INTERNAL)
rocprofiler_add_interface_library(
    rocprofiler-sdk-developer-flags "Compiler flags for developers (more warnings, etc.)"
    INTERNAL)
rocprofiler_add_interface_library(rocprofiler-sdk-debug-flags
                                  "Compiler flags for more debug info" INTERNAL)
rocprofiler_add_interface_library(rocprofiler-sdk-release-flags
                                  "Compiler flags for more debug info" INTERNAL)
rocprofiler_add_interface_library(rocprofiler-sdk-stack-protector
                                  "Adds stack-protector compiler flags" INTERNAL)
rocprofiler_add_interface_library(rocprofiler-sdk-memcheck INTERFACE INTERNAL)
rocprofiler_add_interface_library(
    rocprofiler-sdk-experimental-flags
    "Compiler flags for experimental feature compilation" INTERNAL)
rocprofiler_add_interface_library(
    rocprofiler-sdk-deprecated-flags
    "Compiler flags for deprecated feature usage warnings" INTERNAL)

#
# interfaces for libraries (general)
#
rocprofiler_add_interface_library(rocprofiler-sdk-dl
                                  "Build flags for dynamic linking library" INTERNAL)
rocprofiler_add_interface_library(rocprofiler-sdk-rt "Build flags for runtime library"
                                  INTERNAL)
rocprofiler_add_interface_library(rocprofiler-sdk-atomic "atomic library" INTERNAL)
rocprofiler_add_interface_library(rocprofiler-sdk-gtest "Google Test library" INTERNAL)
rocprofiler_add_interface_library(rocprofiler-sdk-glog "Google Log library" INTERNAL)
rocprofiler_add_interface_library(rocprofiler-sdk-fmt "C++ format string library"
                                  INTERNAL)
rocprofiler_add_interface_library(rocprofiler-sdk-cxx-filesystem "C++ filesystem library"
                                  INTERNAL)
rocprofiler_add_interface_library(rocprofiler-sdk-ptl "Parallel Tasking Library" INTERNAL)
rocprofiler_add_interface_library(rocprofiler-sdk-elf "ElfUtils elf library" INTERNAL)
rocprofiler_add_interface_library(rocprofiler-sdk-dw "ElfUtils dw library" INTERNAL)
rocprofiler_add_interface_library(rocprofiler-sdk-elfio "ELFIO header-only C++ library"
                                  INTERNAL)
rocprofiler_add_interface_library(rocprofiler-sdk-yaml-cpp "YAML CPP Parser" INTERNAL)
rocprofiler_add_interface_library(rocprofiler-sdk-json "nlohmann json" INTERNAL)

#
# interface for libraries (ROCm-specific)
#
rocprofiler_add_interface_library(rocprofiler-sdk-hip "HIP library" INTERNAL)
rocprofiler_add_interface_library(rocprofiler-sdk-hsa-runtime "HSA runtime library"
                                  INTERNAL)
rocprofiler_add_interface_library(rocprofiler-sdk-amd-comgr "AMD comgr library" INTERNAL)
rocprofiler_add_interface_library(rocprofiler-sdk-hsa-aql "AQL library" INTERNAL)
rocprofiler_add_interface_library(rocprofiler-sdk-hsakmt
                                  "HSAKMT library for AMD KFD support" INTERNAL)
rocprofiler_add_interface_library(rocprofiler-sdk-drm "drm (amdgpu) library" INTERNAL)

#
# "nolink" interface targets emulate another interface target but do not link to the
# library. E.g. rocprofiler-sdk-hip-nolink has the include directories, compile
# definitions, and compile options of rocprofiler-sdk-hip but does not link to the HIP
# runtime library
#

rocprofiler_add_interface_library(
    rocprofiler-sdk-hip-nolink "rocprofiler-sdk-hip without linking to HIP library"
    IMPORTED)
rocprofiler_add_interface_library(
    rocprofiler-sdk-hsa-runtime-nolink
    "rocprofiler-sdk-hsa-runtime without linking to HSA library" IMPORTED)
rocprofiler_add_interface_library(
    rocprofiler-sdk-hsakmt-nolink
    "rocprofiler-sdk-hsakmt without linking to HSAKMT library" IMPORTED)
rocprofiler_add_interface_library(rocprofiler-sdk-rccl-nolink
                                  "RCCL headers without linking to RCCL library" IMPORTED)
rocprofiler_add_interface_library(
    rocprofiler-sdk-rocdecode-nolink
    "ROCDECODE headers without linking to ROCDECODE library" IMPORTED)

rocprofiler_add_interface_library(
    rocprofiler-sdk-rocjpeg-nolink "ROCJPEG headers without linking to ROCJPEG library"
    IMPORTED)
