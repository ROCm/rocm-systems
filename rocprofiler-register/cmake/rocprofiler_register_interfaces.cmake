#
#
# Forward declaration of all INTERFACE targets
#
#

include(rocprofiler_register_utilities)

#
# interfaces for build flags
#
rocprofiler_register_add_interface_library(
    rocprofiler-register-headers
    "Provides minimal set of include flags to compile with rocprofiler-register")
rocprofiler_register_add_interface_library(
    rocprofiler-register-build-flags
    "Provides generalized build flags for rocprofiler-register" INTERNAL)
rocprofiler_register_add_interface_library(rocprofiler-register-threading
                                           "Enables multithreading support" INTERNAL)
rocprofiler_register_add_interface_library(
    rocprofiler-register-developer-flags
    "Compiler flags for developers (more warnings, etc.)" INTERNAL)
rocprofiler_register_add_interface_library(rocprofiler-register-memcheck "Sanitizer"
                                           INTERNAL)

#
# interfaces for libraries
#
rocprofiler_register_add_interface_library(
    rocprofiler-register-dl "Build flags for dynamic linking library" INTERNAL)
rocprofiler_register_add_interface_library(rocprofiler-register-rt
                                           "Build flags for runtime library" INTERNAL)
rocprofiler_register_add_interface_library(rocprofiler-register-stdcxxfs
                                           "C++ filesystem library" INTERNAL)
