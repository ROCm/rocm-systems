# ----------------------------------------------------------------------------- #
#
# rocprofiler-systems: contains all instrumentation functionality
#
# rocprofiler-systems-dl: contains minimal symbols and dlopen's rocprofiler-systems
#
# rocprofiler-systems-user: contains symbols for user API
#
# ----------------------------------------------------------------------------- #

if(CMAKE_VERSION VERSION_GREATER_EQUAL 3.20)
    cmake_policy(SET CMP0115 NEW)
endif()

set(ROCPROFSYS_LIB_INSTALL_RPATH "\$ORIGIN:\$ORIGIN/${PROJECT_NAME}")

# ------------------------------------------------------------------------------#
#
# rocprofiler-systems interface library
#
# ------------------------------------------------------------------------------#

add_library(rocprofiler-systems-interface-library INTERFACE)
add_library(
    rocprofiler-systems::rocprofiler-systems-interface-library
    ALIAS rocprofiler-systems-interface-library
)

target_include_directories(
    rocprofiler-systems-interface-library
    INTERFACE ${CMAKE_CURRENT_SOURCE_DIR} ${CMAKE_CURRENT_SOURCE_DIR}/rocprof-sys
)

target_link_libraries(
    rocprofiler-systems-interface-library
    INTERFACE
        $<BUILD_INTERFACE:rocprofiler-systems::rocprofiler-systems-headers>
        $<BUILD_INTERFACE:rocprofiler-systems::rocprofiler-systems-threading>
        $<BUILD_INTERFACE:rocprofiler-systems::rocprofiler-systems-common-library>
        $<BUILD_INTERFACE:rocprofiler-systems::rocprofiler-systems-compile-options>
        $<BUILD_INTERFACE:rocprofiler-systems::rocprofiler-systems-compile-definitions>
        $<BUILD_INTERFACE:rocprofiler-systems::rocprofiler-systems-perfetto>
        $<BUILD_INTERFACE:rocprofiler-systems::rocprofiler-systems-sqlite3>
        $<BUILD_INTERFACE:rocprofiler-systems::rocprofiler-systems-timemory>
        $<BUILD_INTERFACE:rocprofiler-systems::rocprofiler-systems-elfutils>
        $<BUILD_INTERFACE:rocprofiler-systems::rocprofiler-systems-bfd>
        $<BUILD_INTERFACE:rocprofiler-systems::rocprofiler-systems-mpi>
        $<BUILD_INTERFACE:rocprofiler-systems::rocprofiler-systems-libva>
        $<BUILD_INTERFACE:rocprofiler-systems::rocprofiler-systems-ptl>
        $<BUILD_INTERFACE:rocprofiler-systems::rocprofiler-systems-rocm>
        $<BUILD_INTERFACE:rocprofiler-systems::rocprofiler-systems-static-libgcc-optional>
        $<BUILD_INTERFACE:rocprofiler-systems::rocprofiler-systems-static-libstdcxx-optional>
        $<BUILD_INTERFACE:rocprofiler-systems::rocprofiler-systems-sanitizer>
        $<BUILD_INTERFACE:$<IF:$<BOOL:${ROCPROFSYS_BUILD_LTO}>,rocprofiler-systems::rocprofiler-systems-lto,>>
)

# ------------------------------------------------------------------------------#
#
# rocprofiler-systems internal libraries
#
# ------------------------------------------------------------------------------#

add_subdirectory(common)
add_subdirectory(core)
add_subdirectory(binary)

# ------------------------------------------------------------------------------#
#
# rocprofiler-systems exported libraries
#
# ------------------------------------------------------------------------------#

add_subdirectory(rocprof-sys)
add_subdirectory(rocprof-sys-dl)
add_subdirectory(rocprof-sys-rt)
add_subdirectory(rocprof-sys-user)
