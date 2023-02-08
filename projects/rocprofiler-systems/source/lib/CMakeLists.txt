# ----------------------------------------------------------------------------- #
#
# omnitrace: contains all instrumentation functionality
#
# omnitrace-dl: contains minimal symbols and dlopen's omnitrace
#
# omnitrace-user: contains symbols for user API
#
# ----------------------------------------------------------------------------- #

if(CMAKE_VERSION VERSION_GREATER_EQUAL 3.20)
    cmake_policy(SET CMP0115 NEW)
endif()

if(OMNITRACE_USE_ROCPROFILER
   AND rocprofiler_LIBRARY_DIR
   AND ROCmVersion_TRIPLE_VERSION VERSION_LESS 5.2.0
   AND NOT CMAKE_INSTALL_RPATH_USE_LINK_PATH)
    set(OMNITRACE_LIB_INSTALL_RPATH
        "\$ORIGIN:\$ORIGIN/omnitrace:${rocprofiler_LIBRARY_DIR}")
else()
    set(OMNITRACE_LIB_INSTALL_RPATH "\$ORIGIN:\$ORIGIN/omnitrace")
endif()

# ------------------------------------------------------------------------------#
#
# omnitrace interface library
#
# ------------------------------------------------------------------------------#

add_library(omnitrace-interface-library INTERFACE)
add_library(omnitrace::omnitrace-interface-library ALIAS omnitrace-interface-library)

target_include_directories(
    omnitrace-interface-library INTERFACE ${CMAKE_CURRENT_SOURCE_DIR}
                                          ${CMAKE_CURRENT_SOURCE_DIR}/omnitrace)

target_link_libraries(
    omnitrace-interface-library
    INTERFACE
        $<BUILD_INTERFACE:omnitrace::omnitrace-headers>
        $<BUILD_INTERFACE:omnitrace::omnitrace-threading>
        $<BUILD_INTERFACE:omnitrace::omnitrace-common-library>
        $<BUILD_INTERFACE:omnitrace::omnitrace-compile-options>
        $<BUILD_INTERFACE:omnitrace::omnitrace-compile-definitions>
        $<BUILD_INTERFACE:omnitrace::omnitrace-perfetto>
        $<BUILD_INTERFACE:omnitrace::omnitrace-timemory>
        $<BUILD_INTERFACE:omnitrace::omnitrace-elfutils>
        $<BUILD_INTERFACE:omnitrace::omnitrace-bfd>
        $<BUILD_INTERFACE:omnitrace::omnitrace-mpi>
        $<BUILD_INTERFACE:omnitrace::omnitrace-ptl>
        $<BUILD_INTERFACE:omnitrace::omnitrace-hip>
        $<BUILD_INTERFACE:omnitrace::omnitrace-roctracer>
        $<BUILD_INTERFACE:omnitrace::omnitrace-rocprofiler>
        $<BUILD_INTERFACE:omnitrace::omnitrace-rocm-smi>
        $<BUILD_INTERFACE:omnitrace::omnitrace-rccl>
        $<BUILD_INTERFACE:omnitrace::omnitrace-static-libgcc-optional>
        $<BUILD_INTERFACE:omnitrace::omnitrace-static-libstdcxx-optional>
        $<BUILD_INTERFACE:omnitrace::omnitrace-sanitizer>
        $<BUILD_INTERFACE:$<IF:$<BOOL:${OMNITRACE_BUILD_LTO}>,omnitrace::omnitrace-lto,>>)

# ------------------------------------------------------------------------------#
#
# omnitrace internal libraries
#
# ------------------------------------------------------------------------------#

add_subdirectory(common)
add_subdirectory(core)
add_subdirectory(binary)

# ------------------------------------------------------------------------------#
#
# omnitrace exported libraries
#
# ------------------------------------------------------------------------------#

add_subdirectory(omnitrace)
add_subdirectory(omnitrace-dl)
add_subdirectory(omnitrace-user)
