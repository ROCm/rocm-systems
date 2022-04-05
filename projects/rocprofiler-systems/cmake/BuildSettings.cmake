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

omnitrace_add_option(OMNITRACE_BUILD_DEVELOPER
                     "Extra build flags for development like -Werror" OFF)
omnitrace_add_option(OMNITRACE_BUILD_EXTRA_OPTIMIZATIONS "Extra optimization flags" OFF)
omnitrace_add_option(OMNITRACE_BUILD_LTO "Build with link-time optimization" OFF)
omnitrace_add_option(OMNITRACE_USE_COMPILE_TIMING
                     "Build with timing metrics for compilation" OFF)
omnitrace_add_option(OMNITRACE_USE_COVERAGE "Build with code-coverage flags" OFF)
omnitrace_add_option(OMNITRACE_USE_SANITIZER
                     "Build with -fsanitze=\${OMNITRACE_SANITIZER_TYPE}" OFF)
omnitrace_add_option(OMNITRACE_BUILD_STATIC_LIBGCC
                     "Build with -static-libgcc if possible" OFF)
omnitrace_add_option(OMNITRACE_BUILD_STATIC_LIBSTDCXX
                     "Build with -static-libstdc++ if possible" OFF)

omnitrace_add_interface_library(omnitrace-static-libgcc
                                "Link to static version of libgcc")
omnitrace_add_interface_library(omnitrace-static-libstdcxx
                                "Link to static version of libstdc++")

target_compile_definitions(omnitrace-compile-options INTERFACE $<$<CONFIG:DEBUG>:DEBUG>)

set(OMNITRACE_SANITIZER_TYPE
    "leak"
    CACHE STRING "Sanitizer type")
if(OMNITRACE_USE_SANITIZER)
    omnitrace_add_feature(OMNITRACE_SANITIZER_TYPE
                          "Sanitizer type, e.g. leak, thread, address, memory, etc.")
endif()

if(OMNITRACE_BUILD_CI)
    omnitrace_target_compile_definitions(${LIBNAME}-compile-options
                                         INTERFACE OMNITRACE_CI)
endif()

# ----------------------------------------------------------------------------------------#
# dynamic linking and runtime libraries
#
if(CMAKE_DL_LIBS AND NOT "${CMAKE_DL_LIBS}" STREQUAL "dl")
    # if cmake provides dl library, use that
    set(dl_LIBRARY
        ${CMAKE_DL_LIBS}
        CACHE FILEPATH "dynamic linking system library")
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
    target_link_libraries(omnitrace-compile-options INTERFACE ${dl_LIBRARY})
endif()

# ----------------------------------------------------------------------------------------#
# set the compiler flags
#
add_flag_if_avail(
    "-W" "-Wall" "-Wno-unknown-pragmas" "-Wno-unused-function" "-Wno-ignored-attributes"
    "-Wno-attributes" "-Wno-missing-field-initializers")

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
omnitrace_add_interface_library(
    omnitrace-compile-debuginfo
    "Attempts to set best flags for more expressive profiling information in debug or optimized binaries"
    )

add_target_flag_if_avail(omnitrace-compile-debuginfo "-g" "-fno-omit-frame-pointer"
                         "-fno-optimize-sibling-calls")

if(CMAKE_CUDA_COMPILER_IS_NVIDIA)
    add_target_cuda_flag(omnitrace-compile-debuginfo "-lineinfo")
endif()

target_compile_options(
    omnitrace-compile-debuginfo
    INTERFACE $<$<COMPILE_LANGUAGE:C>:$<$<C_COMPILER_ID:GNU>:-rdynamic>>
              $<$<COMPILE_LANGUAGE:CXX>:$<$<CXX_COMPILER_ID:GNU>:-rdynamic>>)

if(NOT APPLE)
    target_link_options(omnitrace-compile-debuginfo INTERFACE
                        $<$<CXX_COMPILER_ID:GNU>:-rdynamic>)
endif()

if(CMAKE_CUDA_COMPILER_IS_NVIDIA)
    target_compile_options(
        omnitrace-compile-debuginfo
        INTERFACE
            $<$<COMPILE_LANGUAGE:CUDA>:$<$<CXX_COMPILER_ID:GNU>:-Xcompiler=-rdynamic>>)
endif()

if(dl_LIBRARY)
    target_link_libraries(omnitrace-compile-debuginfo INTERFACE ${dl_LIBRARY})
endif()

if(rt_LIBRARY)
    target_link_libraries(omnitrace-compile-debuginfo INTERFACE ${rt_LIBRARY})
endif()

# ----------------------------------------------------------------------------------------#
# non-debug optimizations
#
omnitrace_add_interface_library(omnitrace-compile-extra "Extra optimization flags")
if(NOT OMNITRACE_USE_COVERAGE)
    add_target_flag_if_avail(
        omnitrace-compile-extra "-finline-functions" "-funroll-loops" "-ftree-vectorize"
        "-ftree-loop-optimize" "-ftree-loop-vectorize")
endif()

if(NOT "${CMAKE_BUILD_TYPE}" STREQUAL "Debug" AND OMNITRACE_BUILD_EXTRA_OPTIMIZATIONS)
    target_link_libraries(omnitrace-compile-options
                          INTERFACE $<BUILD_INTERFACE:omnitrace-compile-extra>)
    add_flag_if_avail(
        "-fno-signaling-nans" "-fno-trapping-math" "-fno-signed-zeros"
        "-ffinite-math-only" "-fno-math-errno" "-fpredictive-commoning"
        "-fvariable-expansion-in-unroller")
    # add_flag_if_avail("-freciprocal-math" "-fno-signed-zeros" "-mfast-fp")
endif()

# ----------------------------------------------------------------------------------------#
# debug-safe optimizations
#
add_cxx_flag_if_avail("-faligned-new")

omnitrace_save_variables(FLTO VARIABLES CMAKE_CXX_FLAGS)
set(_CXX_FLAGS "${CMAKE_CXX_FLAGS}")
set(CMAKE_CXX_FLAGS "-flto=thin ${_CXX_FLAGS}")

omnitrace_add_interface_library(omnitrace-lto "Adds link-time-optimization flags")
add_target_flag_if_avail(omnitrace-lto "-flto=thin")
if(NOT cxx_omnitrace_lto_flto_thin)
    set(CMAKE_CXX_FLAGS "-flto ${_CXX_FLAGS}")
    add_target_flag_if_avail(omnitrace-lto "-flto")
    if(NOT cxx_omnitrace_lto_flto)
        set(OMNITRACE_BUILD_LTO OFF)
    else()
        target_link_options(omnitrace-lto INTERFACE -flto)
    endif()
    add_target_flag_if_avail(omnitrace-lto "-fno-fat-lto-objects")
    if(cxx_omnitrace_lto_fno_fat_lto_objects)
        target_link_options(omnitrace-lto INTERFACE -fno-fat-lto-objects)
    endif()
else()
    target_link_options(omnitrace-lto INTERFACE -flto=thin)
endif()

if(OMNITRACE_BUILD_LTO)
    target_link_libraries(omnitrace-compile-options INTERFACE omnitrace::omnitrace-lto)
endif()

omnitrace_restore_variables(FLTO VARIABLES CMAKE_CXX_FLAGS)

# ----------------------------------------------------------------------------------------#
# print compilation timing reports (Clang compiler)
#
omnitrace_add_interface_library(
    omnitrace-compile-timing
    "Adds compiler flags which report compilation timing metrics")
if(CMAKE_CXX_COMPILER_IS_CLANG)
    add_target_flag_if_avail(omnitrace-compile-timing "-ftime-trace")
    if(NOT cxx_omnitrace_compile_timing_ftime_trace)
        add_target_flag_if_avail(omnitrace-compile-timing "-ftime-report")
    endif()
else()
    add_target_flag_if_avail(omnitrace-compile-timing "-ftime-report")
endif()

if(OMNITRACE_USE_COMPILE_TIMING)
    target_link_libraries(omnitrace-compile-options INTERFACE omnitrace-compile-timing)
endif()

# ----------------------------------------------------------------------------------------#
# developer build flags
#
omnitrace_add_interface_library(omnitrace-develop-options "Adds developer compiler flags")
if(OMNITRACE_BUILD_DEVELOPER)
    add_target_flag_if_avail(
        omnitrace-develop-options
        # "-Wabi"
        "-Wdouble-promotion" "-Wshadow" "-Wextra" "-Wpedantic" "-Werror" "/showIncludes")
endif()

# ----------------------------------------------------------------------------------------#
# visibility build flags
#
omnitrace_add_interface_library(omnitrace-default-visibility
                                "Adds -fvisibility=default compiler flag")
omnitrace_add_interface_library(omnitrace-hidden-visibility
                                "Adds -fvisibility=hidden compiler flag")

add_target_flag_if_avail(omnitrace-default-visibility "-fvisibility=default")
add_target_flag_if_avail(omnitrace-hidden-visibility "-fvisibility=hidden"
                         "-fvisibility-inlines-hidden")

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
set(OMNITRACE_SANITIZER_TYPES
    address
    memory
    thread
    leak
    undefined
    unreachable
    null
    bounds
    alignment)
set_property(CACHE OMNITRACE_SANITIZER_TYPE PROPERTY STRINGS
                                                     "${OMNITRACE_SANITIZER_TYPES}")
omnitrace_add_interface_library(omnitrace-sanitizer-compile-options
                                "Adds compiler flags for sanitizers")
omnitrace_add_interface_library(
    omnitrace-sanitizer
    "Adds compiler flags to enable ${OMNITRACE_SANITIZER_TYPE} sanitizer (-fsanitizer=${OMNITRACE_SANITIZER_TYPE})"
    )

set(COMMON_SANITIZER_FLAGS "-fno-optimize-sibling-calls" "-fno-omit-frame-pointer"
                           "-fno-inline-functions")
add_target_flag(omnitrace-sanitizer-compile-options ${COMMON_SANITIZER_FLAGS})

foreach(_TYPE ${OMNITRACE_SANITIZER_TYPES})
    set(_FLAG "-fsanitize=${_TYPE}")
    omnitrace_add_interface_library(
        omnitrace-${_TYPE}-sanitizer
        "Adds compiler flags to enable ${_TYPE} sanitizer (${_FLAG})")
    add_target_flag(omnitrace-${_TYPE}-sanitizer ${_FLAG})
    target_link_libraries(omnitrace-${_TYPE}-sanitizer
                          INTERFACE omnitrace-sanitizer-compile-options)
    set_property(TARGET omnitrace-${_TYPE}-sanitizer
                 PROPERTY INTERFACE_LINK_OPTIONS ${_FLAG} ${COMMON_SANITIZER_FLAGS})
endforeach()

unset(_FLAG)
unset(COMMON_SANITIZER_FLAGS)

if(OMNITRACE_USE_SANITIZER)
    foreach(_TYPE ${OMNITRACE_SANITIZER_TYPE})
        if(TARGET omnitrace-${_TYPE}-sanitizer)
            target_link_libraries(omnitrace-sanitizer
                                  INTERFACE omnitrace-${_TYPE}-sanitizer)
        else()
            message(
                FATAL_ERROR "Error! Target 'omnitrace-${_TYPE}-sanitizer' does not exist!"
                )
        endif()
    endforeach()
else()
    set(OMNITRACE_USE_SANITIZER OFF)
endif()

if(MSVC)
    # VTune is much more helpful when debug information is included in the generated
    # release code.
    add_flag_if_avail("/Zi")
    add_flag_if_avail("/DEBUG")
endif()

# ----------------------------------------------------------------------------------------#
# static lib flags
#
target_compile_options(
    omnitrace-static-libgcc
    INTERFACE $<$<COMPILE_LANGUAGE:C>:$<$<C_COMPILER_ID:GNU>:-static-libgcc>>
              $<$<COMPILE_LANGUAGE:CXX>:$<$<CXX_COMPILER_ID:GNU>:-static-libgcc>>)
target_compile_options(
    omnitrace-static-libstdcxx
    INTERFACE $<$<COMPILE_LANGUAGE:CXX>:$<$<CXX_COMPILER_ID:GNU>:-static-libstdc++>>)

# ----------------------------------------------------------------------------------------#
# user customization
#
get_property(LANGUAGES GLOBAL PROPERTY ENABLED_LANGUAGES)

if(NOT APPLE OR "$ENV{CONDA_PYTHON_EXE}" STREQUAL "")
    add_user_flags(omnitrace-compile-options "CXX")
endif()
