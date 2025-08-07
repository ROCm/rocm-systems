# include guard
include_guard(DIRECTORY)

# ########################################################################################
#
# Handles the build settings
#
# ########################################################################################

include(GNUInstallDirs)
include(Compilers)
include(FindPackageHandleStandardArgs)
include(MacroUtilities)

rocprofiler_systems_add_option(
    ROCPROFSYS_BUILD_DEVELOPER "Extra build flags for development like -Werror"
    ${ROCPROFSYS_BUILD_CI}
)
rocprofiler_systems_add_option(ROCPROFSYS_BUILD_RELEASE
                               "Build with minimal debug line info" OFF
)
rocprofiler_systems_add_option(ROCPROFSYS_BUILD_EXTRA_OPTIMIZATIONS
                               "Extra optimization flags" OFF
)
rocprofiler_systems_add_option(ROCPROFSYS_BUILD_LTO "Build with link-time optimization"
                               OFF
)
rocprofiler_systems_add_option(ROCPROFSYS_USE_COMPILE_TIMING
                               "Build with timing metrics for compilation" OFF
)
rocprofiler_systems_add_option(ROCPROFSYS_USE_SANITIZER
                               "Build with -fsanitze=\${ROCPROFSYS_SANITIZER_TYPE}" OFF
)
rocprofiler_systems_add_option(ROCPROFSYS_BUILD_STATIC_LIBGCC
                               "Build with -static-libgcc if possible" OFF
)
rocprofiler_systems_add_option(ROCPROFSYS_BUILD_STATIC_LIBSTDCXX
                               "Build with -static-libstdc++ if possible" OFF
)
rocprofiler_systems_add_option(ROCPROFSYS_BUILD_STACK_PROTECTOR
                               "Build with -fstack-protector" ON
)
rocprofiler_systems_add_cache_option(
    ROCPROFSYS_BUILD_LINKER
    "If set to a non-empty value, pass -fuse-ld=\${ROCPROFSYS_BUILD_LINKER}" STRING "bfd"
)
rocprofiler_systems_add_cache_option(ROCPROFSYS_BUILD_NUMBER "Internal CI use" STRING "0"
                                     ADVANCED NO_FEATURE
)

rocprofiler_systems_add_interface_library(rocprofiler-systems-static-libgcc
                                          "Link to static version of libgcc"
)
rocprofiler_systems_add_interface_library(rocprofiler-systems-static-libstdcxx
                                          "Link to static version of libstdc++"
)
rocprofiler_systems_add_interface_library(rocprofiler-systems-static-libgcc-optional
                                          "Link to static version of libgcc"
)
rocprofiler_systems_add_interface_library(rocprofiler-systems-static-libstdcxx-optional
                                          "Link to static version of libstdc++"
)

target_compile_definitions(
    rocprofiler-systems-compile-options
    INTERFACE $<$<CONFIG:DEBUG>:DEBUG>
)

set(ROCPROFSYS_SANITIZER_TYPE "leak" CACHE STRING "Sanitizer type")
if(ROCPROFSYS_USE_SANITIZER)
    rocprofiler_systems_add_feature(
        ROCPROFSYS_SANITIZER_TYPE
        "Sanitizer type, e.g. leak, thread, address, memory, etc."
    )
endif()

if(ROCPROFSYS_BUILD_CI)
    rocprofiler_systems_target_compile_definitions(${LIBNAME}-compile-options
                                                   INTERFACE ROCPROFSYS_CI
    )
endif()

# ----------------------------------------------------------------------------------------#
# dynamic linking and runtime libraries
#
if(CMAKE_DL_LIBS AND NOT "${CMAKE_DL_LIBS}" STREQUAL "dl")
    # if cmake provides dl library, use that
    set(dl_LIBRARY ${CMAKE_DL_LIBS} CACHE FILEPATH "dynamic linking system library")
endif()

foreach(_TYPE dl rt dw)
    if(NOT ${_TYPE}_LIBRARY)
        find_library(${_TYPE}_LIBRARY NAMES ${_TYPE})
    endif()
endforeach()

find_package_handle_standard_args(dl-library REQUIRED_VARS dl_LIBRARY)
find_package_handle_standard_args(rt-library REQUIRED_VARS rt_LIBRARY)
# find_package_handle_standard_args(dw-library REQUIRED_VARS dw_LIBRARY)

if(dl_LIBRARY)
    target_link_libraries(rocprofiler-systems-compile-options INTERFACE ${dl_LIBRARY})
endif()

# ----------------------------------------------------------------------------------------#
# set the compiler flags
#
add_flag_if_avail(
    "-W" "-Wall" "-Wno-unknown-pragmas" "-Wno-unused-function" "-Wno-ignored-attributes"
    "-Wno-attributes" "-Wno-missing-field-initializers" "-Wno-interference-size"
)

if(ROCPROFSYS_BUILD_DEBUG)
    add_flag_if_avail("-g3" "-fno-omit-frame-pointer")
endif()

if(WIN32)
    # suggested by MSVC for spectre mitigation in rapidjson implementation
    add_cxx_flag_if_avail("/Qspectre")
endif()

if(CMAKE_CXX_COMPILER_IS_CLANG)
    add_cxx_flag_if_avail("-Wno-mismatched-tags")
endif()

# ----------------------------------------------------------------------------------------#
# extra flags for debug information in debug or optimized binaries
#
rocprofiler_systems_add_interface_library(
    rocprofiler-systems-compile-debuginfo
    "Attempts to set best flags for more expressive profiling information in debug or optimized binaries"
)

add_target_flag_if_avail(rocprofiler-systems-compile-debuginfo "-g3"
                         "-fno-omit-frame-pointer" "-fno-optimize-sibling-calls"
)

if(CMAKE_CUDA_COMPILER_IS_NVIDIA)
    add_target_cuda_flag(rocprofiler-systems-compile-debuginfo "-lineinfo")
endif()

target_compile_options(
    rocprofiler-systems-compile-debuginfo
    INTERFACE
        $<$<COMPILE_LANGUAGE:C>:$<$<C_COMPILER_ID:GNU>:-rdynamic>>
        $<$<COMPILE_LANGUAGE:CXX>:$<$<CXX_COMPILER_ID:GNU>:-rdynamic>>
)

if(NOT APPLE)
    target_link_options(
        rocprofiler-systems-compile-debuginfo
        INTERFACE $<$<CXX_COMPILER_ID:GNU>:-rdynamic>
    )
endif()

if(CMAKE_CUDA_COMPILER_IS_NVIDIA)
    target_compile_options(
        rocprofiler-systems-compile-debuginfo
        INTERFACE
            $<$<COMPILE_LANGUAGE:CUDA>:$<$<CXX_COMPILER_ID:GNU>:-Xcompiler=-rdynamic>>
    )
endif()

if(dl_LIBRARY)
    target_link_libraries(rocprofiler-systems-compile-debuginfo INTERFACE ${dl_LIBRARY})
endif()

if(rt_LIBRARY)
    target_link_libraries(rocprofiler-systems-compile-debuginfo INTERFACE ${rt_LIBRARY})
endif()

# ----------------------------------------------------------------------------------------#
# non-debug optimizations
#
rocprofiler_systems_add_interface_library(rocprofiler-systems-compile-extra
                                          "Extra optimization flags"
)
if(NOT ROCPROFSYS_BUILD_CODECOV AND ROCPROFSYS_BUILD_EXTRA_OPTIMIZATIONS)
    add_target_flag_if_avail(
        rocprofiler-systems-compile-extra "-finline-functions" "-funroll-loops"
        "-ftree-vectorize" "-ftree-loop-optimize" "-ftree-loop-vectorize"
    )
endif()

if(
    NOT "${CMAKE_BUILD_TYPE}" STREQUAL "Debug"
    AND ROCPROFSYS_BUILD_EXTRA_OPTIMIZATIONS
    AND NOT ROCPROFSYS_BUILD_CODECOV
)
    target_link_libraries(
        rocprofiler-systems-compile-options
        INTERFACE $<BUILD_INTERFACE:rocprofiler-systems-compile-extra>
    )
    add_flag_if_avail(
        "-fno-signaling-nans" "-fno-trapping-math" "-fno-signed-zeros"
        "-ffinite-math-only" "-fno-math-errno" "-fpredictive-commoning"
        "-fvariable-expansion-in-unroller"
    )
    # add_flag_if_avail("-freciprocal-math" "-fno-signed-zeros" "-mfast-fp")
endif()

# ----------------------------------------------------------------------------------------#
# debug-safe optimizations
#
add_cxx_flag_if_avail("-faligned-new")

rocprofiler_systems_add_interface_library(rocprofiler-systems-lto
                                          "Adds link-time-optimization flags"
)

if(NOT ROCPROFSYS_BUILD_CODECOV)
    rocprofiler_systems_save_variables(FLTO VARIABLES CMAKE_CXX_FLAGS)
    set(_CXX_FLAGS "${CMAKE_CXX_FLAGS}")
    set(CMAKE_CXX_FLAGS "-flto=thin ${_CXX_FLAGS}")

    add_target_flag_if_avail(rocprofiler-systems-lto "-flto=thin")
    if(NOT cxx_rocprofiler_systems_lto_flto_thin)
        set(CMAKE_CXX_FLAGS "-flto ${_CXX_FLAGS}")
        add_target_flag_if_avail(rocprofiler-systems-lto "-flto")
        if(NOT cxx_rocprofiler_systems_lto_flto)
            set(ROCPROFSYS_BUILD_LTO OFF)
        else()
            target_link_options(rocprofiler-systems-lto INTERFACE -flto)
        endif()
        add_target_flag_if_avail(rocprofiler-systems-lto "-fno-fat-lto-objects")
        if(cxx_rocprofiler_systems_lto_fno_fat_lto_objects)
            target_link_options(rocprofiler-systems-lto INTERFACE -fno-fat-lto-objects)
        endif()
    else()
        target_link_options(rocprofiler-systems-lto INTERFACE -flto=thin)
    endif()

    rocprofiler_systems_restore_variables(FLTO VARIABLES CMAKE_CXX_FLAGS)
endif()

# ----------------------------------------------------------------------------------------#
# print compilation timing reports (Clang compiler)
#
rocprofiler_systems_add_interface_library(
    rocprofiler-systems-compile-timing
    "Adds compiler flags which report compilation timing metrics"
)
if(CMAKE_CXX_COMPILER_IS_CLANG)
    add_target_flag_if_avail(rocprofiler-systems-compile-timing "-ftime-trace")
    if(NOT cxx_rocprofiler_systems_compile_timing_ftime_trace)
        add_target_flag_if_avail(rocprofiler-systems-compile-timing "-ftime-report")
    endif()
else()
    add_target_flag_if_avail(rocprofiler-systems-compile-timing "-ftime-report")
endif()

if(ROCPROFSYS_USE_COMPILE_TIMING)
    target_link_libraries(
        rocprofiler-systems-compile-options
        INTERFACE rocprofiler-systems-compile-timing
    )
endif()

# ----------------------------------------------------------------------------------------#
# fstack-protector
#
rocprofiler_systems_add_interface_library(rocprofiler-systems-stack-protector
                                          "Adds stack-protector compiler flags"
)
add_target_flag_if_avail(rocprofiler-systems-stack-protector "-fstack-protector-strong"
                         "-Wstack-protector"
)

if(ROCPROFSYS_BUILD_STACK_PROTECTOR)
    target_link_libraries(
        rocprofiler-systems-compile-options
        INTERFACE rocprofiler-systems-stack-protector
    )
endif()

# ----------------------------------------------------------------------------------------#
# developer build flags
#
if(ROCPROFSYS_BUILD_DEVELOPER)
    add_target_flag_if_avail(
        rocprofiler-systems-compile-options "-Werror" "-Wdouble-promotion" "-Wshadow"
        "-Wextra" "-Wpedantic" "-Wstack-usage=524288" # 512 KB
        "/showIncludes"
    )
    if(ROCPROFSYS_BUILD_NUMBER GREATER 2)
        add_target_flag_if_avail(rocprofiler-systems-compile-options "-gsplit-dwarf")
    endif()
endif()

if(ROCPROFSYS_BUILD_LINKER)
    target_link_options(
        rocprofiler-systems-compile-options
        INTERFACE
            $<$<C_COMPILER_ID:GNU>:-fuse-ld=${ROCPROFSYS_BUILD_LINKER}>
            $<$<CXX_COMPILER_ID:GNU>:-fuse-ld=${ROCPROFSYS_BUILD_LINKER}>
    )
endif()

# ----------------------------------------------------------------------------------------#
# release build flags
#
if(ROCPROFSYS_BUILD_RELEASE AND NOT ROCPROFSYS_BUILD_DEBUG)
    add_target_flag_if_avail(
        rocprofiler-systems-compile-options "-g1" "-feliminate-unused-debug-symbols"
        "-gno-column-info" "-gno-variable-location-views" "-gline-tables-only"
    )
endif()

# ----------------------------------------------------------------------------------------#
# visibility build flags
#
rocprofiler_systems_add_interface_library(rocprofiler-systems-default-visibility
                                          "Adds -fvisibility=default compiler flag"
)
rocprofiler_systems_add_interface_library(rocprofiler-systems-hidden-visibility
                                          "Adds -fvisibility=hidden compiler flag"
)

add_target_flag_if_avail(rocprofiler-systems-default-visibility "-fvisibility=default")
add_target_flag_if_avail(rocprofiler-systems-hidden-visibility "-fvisibility=hidden"
                         "-fvisibility-inlines-hidden"
)

# ----------------------------------------------------------------------------------------#
# developer build flags
#
if(dl_LIBRARY)
    # This instructs the linker to add all symbols, not only used ones, to the dynamic
    # symbol table. This option is needed for some uses of dlopen or to allow obtaining
    # backtraces from within a program.
    add_flag_if_avail("-rdynamic")
endif()

# ----------------------------------------------------------------------------------------#
# sanitizer
#
set(ROCPROFSYS_SANITIZER_TYPES
    address
    memory
    thread
    leak
    undefined
    unreachable
    null
    bounds
    alignment
)
set_property(
    CACHE ROCPROFSYS_SANITIZER_TYPE
    PROPERTY STRINGS "${ROCPROFSYS_SANITIZER_TYPES}"
)
rocprofiler_systems_add_interface_library(rocprofiler-systems-sanitizer-compile-options
                                          "Adds compiler flags for sanitizers"
)
rocprofiler_systems_add_interface_library(
    rocprofiler-systems-sanitizer
    "Adds compiler flags to enable ${ROCPROFSYS_SANITIZER_TYPE} sanitizer (-fsanitizer=${ROCPROFSYS_SANITIZER_TYPE})"
)

set(COMMON_SANITIZER_FLAGS
    "-fno-optimize-sibling-calls"
    "-fno-omit-frame-pointer"
    "-fno-inline-functions"
)
add_target_flag(rocprofiler-systems-sanitizer-compile-options ${COMMON_SANITIZER_FLAGS})

foreach(_TYPE ${ROCPROFSYS_SANITIZER_TYPES})
    set(_FLAG "-fsanitize=${_TYPE}")
    rocprofiler_systems_add_interface_library(
        rocprofiler-systems-${_TYPE}-sanitizer
        "Adds compiler flags to enable ${_TYPE} sanitizer (${_FLAG})"
    )
    add_target_flag(rocprofiler-systems-${_TYPE}-sanitizer ${_FLAG})
    target_link_libraries(
        rocprofiler-systems-${_TYPE}-sanitizer
        INTERFACE rocprofiler-systems-sanitizer-compile-options
    )
    set_property(
        TARGET rocprofiler-systems-${_TYPE}-sanitizer
        PROPERTY INTERFACE_LINK_OPTIONS ${_FLAG} ${COMMON_SANITIZER_FLAGS}
    )
endforeach()

unset(_FLAG)
unset(COMMON_SANITIZER_FLAGS)

if(ROCPROFSYS_USE_SANITIZER)
    foreach(_TYPE ${ROCPROFSYS_SANITIZER_TYPE})
        if(TARGET rocprofiler-systems-${_TYPE}-sanitizer)
            target_link_libraries(
                rocprofiler-systems-sanitizer
                INTERFACE rocprofiler-systems-${_TYPE}-sanitizer
            )
        else()
            message(
                FATAL_ERROR
                "Error! Target 'rocprofiler-systems-${_TYPE}-sanitizer' does not exist!"
            )
        endif()
    endforeach()
else()
    set(ROCPROFSYS_USE_SANITIZER OFF)
endif()

# ----------------------------------------------------------------------------------------#
# static lib flags
#
target_compile_options(
    rocprofiler-systems-static-libgcc
    INTERFACE
        $<$<COMPILE_LANGUAGE:C>:$<$<C_COMPILER_ID:GNU>:-static-libgcc>>
        $<$<COMPILE_LANGUAGE:CXX>:$<$<CXX_COMPILER_ID:GNU>:-static-libgcc>>
)
target_link_options(
    rocprofiler-systems-static-libgcc
    INTERFACE
        $<$<COMPILE_LANGUAGE:C>:$<$<C_COMPILER_ID:GNU,Clang>:-static-libgcc>>
        $<$<COMPILE_LANGUAGE:CXX>:$<$<CXX_COMPILER_ID:GNU,Clang>:-static-libgcc>>
)

target_compile_options(
    rocprofiler-systems-static-libstdcxx
    INTERFACE $<$<COMPILE_LANGUAGE:CXX>:$<$<CXX_COMPILER_ID:GNU>:-static-libstdc++>>
)
target_link_options(
    rocprofiler-systems-static-libstdcxx
    INTERFACE $<$<COMPILE_LANGUAGE:CXX>:$<$<CXX_COMPILER_ID:GNU,Clang>:-static-libstdc++>>
)

if(ROCPROFSYS_BUILD_STATIC_LIBGCC)
    target_link_libraries(
        rocprofiler-systems-static-libgcc-optional
        INTERFACE rocprofiler-systems-static-libgcc
    )
endif()

if(ROCPROFSYS_BUILD_STATIC_LIBSTDCXX)
    target_link_libraries(
        rocprofiler-systems-static-libstdcxx-optional
        INTERFACE rocprofiler-systems-static-libstdcxx
    )
endif()

# ----------------------------------------------------------------------------------------#
# user customization
#
get_property(LANGUAGES GLOBAL PROPERTY ENABLED_LANGUAGES)

if(NOT APPLE OR "$ENV{CONDA_PYTHON_EXE}" STREQUAL "")
    add_user_flags(rocprofiler-systems-compile-options "CXX")
endif()
