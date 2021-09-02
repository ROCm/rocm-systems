# include guard
include_guard(DIRECTORY)

##########################################################################################
#
#        Handles the build settings
#
##########################################################################################

include(GNUInstallDirs)
include(Compilers)
include(FindPackageHandleStandardArgs)
include(MacroUtilities)

option(HOSTTRACE_BUILD_DEVELOPER "Extra build flags for development like -Werror" OFF)
option(HOSTTRACE_BUILD_EXTRA_OPTIMIZATIONS "Extra optimization flags" OFF)
option(HOSTTRACE_BUILD_LTO "Build with link-time optimization" OFF)
option(HOSTTRACE_USE_COMPILE_TIMING "" OFF)
option(HOSTTRACE_USE_COVERAGE "" OFF)
option(HOSTTRACE_USE_SANITIZER "" OFF)

target_compile_definitions(hosttrace-compile-options INTERFACE $<$<CONFIG:DEBUG>:DEBUG>)

set(HOSTTRACE_SANITIZER_TYPE "leak" CACHE STRING "Sanitizer type")

#----------------------------------------------------------------------------------------#
#   dynamic linking and runtime libraries
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
    target_link_libraries(hosttrace-compile-options INTERFACE ${dl_LIBRARY})
endif()

#----------------------------------------------------------------------------------------#
# set the compiler flags
#
add_flag_if_avail(
    "-W"
    "-Wall"
    "-Wno-unknown-pragmas"
    "-Wno-unused-function"
    "-Wno-ignored-attributes"
    "-Wno-attributes"
    "-Wno-missing-field-initializers")

if(WIN32)
    # suggested by MSVC for spectre mitigation in rapidjson implementation
    add_cxx_flag_if_avail("/Qspectre")
endif()

if(CMAKE_CXX_COMPILER_IS_CLANG)
    add_cxx_flag_if_avail(
        "-Wno-mismatched-tags")
endif()

#----------------------------------------------------------------------------------------#
# extra flags for debug information in debug or optimized binaries
#
add_interface_library(hosttrace-compile-debuginfo
    "Attempts to set best flags for more expressive profiling information in debug or optimized binaries")

add_target_flag_if_avail(hosttrace-compile-debuginfo
    "-g"
    "-fno-omit-frame-pointer"
    "-fno-optimize-sibling-calls")

if(CMAKE_CUDA_COMPILER_IS_NVIDIA)
    add_target_cuda_flag(hosttrace-compile-debuginfo "-lineinfo")
endif()

target_compile_options(hosttrace-compile-debuginfo INTERFACE
    $<$<COMPILE_LANGUAGE:C>:$<$<C_COMPILER_ID:GNU>:-rdynamic>>
    $<$<COMPILE_LANGUAGE:CXX>:$<$<CXX_COMPILER_ID:GNU>:-rdynamic>>)

if(NOT APPLE)
    target_link_options(hosttrace-compile-debuginfo INTERFACE
        $<$<CXX_COMPILER_ID:GNU>:-rdynamic>)
endif()

if(CMAKE_CUDA_COMPILER_IS_NVIDIA)
    target_compile_options(hosttrace-compile-debuginfo INTERFACE
        $<$<COMPILE_LANGUAGE:CUDA>:$<$<CXX_COMPILER_ID:GNU>:-Xcompiler=-rdynamic>>)
endif()

if(dl_LIBRARY)
    target_link_libraries(hosttrace-compile-debuginfo INTERFACE ${dl_LIBRARY})
endif()

if(rt_LIBRARY)
    target_link_libraries(hosttrace-compile-debuginfo INTERFACE ${rt_LIBRARY})
endif()

#----------------------------------------------------------------------------------------#
# non-debug optimizations
#
add_interface_library(hosttrace-compile-extra "Extra optimization flags")
if(NOT HOSTTRACE_USE_COVERAGE)
    add_target_flag_if_avail(hosttrace-compile-extra
        "-finline-functions"
        "-funroll-loops"
        "-ftree-vectorize"
        "-ftree-loop-optimize"
        "-ftree-loop-vectorize")
endif()

if(NOT "${CMAKE_BUILD_TYPE}" STREQUAL "Debug" AND HOSTTRACE_BUILD_EXTRA_OPTIMIZATIONS)
    target_link_libraries(hosttrace-compile-options INTERFACE
        $<BUILD_INTERFACE:hosttrace-compile-extra>)
    add_flag_if_avail(
        "-fno-signaling-nans"
        "-fno-trapping-math"
        "-fno-signed-zeros"
        "-ffinite-math-only"
        "-fno-math-errno"
        "-fpredictive-commoning"
        "-fvariable-expansion-in-unroller")
    # add_flag_if_avail("-freciprocal-math" "-fno-signed-zeros" "-mfast-fp")
endif()

#----------------------------------------------------------------------------------------#
# debug-safe optimizations
#
add_cxx_flag_if_avail("-faligned-new")

if(HOSTTRACE_BUILD_LTO)
    set(CMAKE_INTERPROCEDURAL_OPTIMIZATION ON)
endif()

hosttrace_save_variables(FLTO
    VARIABLES CMAKE_CXX_FLAGS)
set(CMAKE_CXX_FLAGS "-flto=thin ${CMAKE_CXX_FLAGS}")

add_interface_library(hosttrace-lto "Adds link-time-optimization flags")
add_target_flag_if_avail(hosttrace-lto "-flto=thin")
if(NOT cxx_hosttrace_lto_flto_thin)
    set(CMAKE_CXX_FLAGS "-flto ${CMAKE_CXX_FLAGS}")
    add_target_flag_if_avail(hosttrace-lto "-flto")
    if(NOT cxx_hosttrace_lto_flto)
        add_disabled_interface(hosttrace-lto)
        set(hosttrace_BUILD_LTO OFF)
    else()
        target_link_options(hosttrace-lto INTERFACE -flto)
    endif()
else()
    target_link_options(hosttrace-lto INTERFACE -flto=thin)
endif()

if(HOSTTRACE_BUILD_LTO)
    target_link_libraries(hosttrace-compile-options INTERFACE hosttrace::hosttrace-lto)
endif()

hosttrace_restore_variables(FLTO
    VARIABLES CMAKE_CXX_FLAGS)

#----------------------------------------------------------------------------------------#
# print compilation timing reports (Clang compiler)
#
add_interface_library(hosttrace-compile-timing
    "Adds compiler flags which report compilation timing metrics")
if(CMAKE_CXX_COMPILER_IS_CLANG)
    add_target_flag_if_avail(hosttrace-compile-timing "-ftime-trace")
    if(NOT cxx_hosttrace_compile_timing_ftime_trace)
        add_target_flag_if_avail(hosttrace-compile-timing "-ftime-report")
    endif()
else()
    add_target_flag_if_avail(hosttrace-compile-timing "-ftime-report")
endif()

if(HOSTTRACE_USE_COMPILE_TIMING)
    target_link_libraries(hosttrace-compile-options INTERFACE hosttrace-compile-timing)
endif()

if(NOT cxx_hosttrace_compile_timing_ftime_report AND NOT cxx_hosttrace_compile_timing_ftime_trace)
    add_disabled_interface(hosttrace-compile-timing)
endif()

#----------------------------------------------------------------------------------------#
# developer build flags
#
add_interface_library(hosttrace-develop-options "Adds developer compiler flags")
if(HOSTTRACE_BUILD_DEVELOPER)
    add_target_flag_if_avail(hosttrace-develop-options
        # "-Wabi"
        "-Wdouble-promotion"
        "-Wshadow"
        "-Wextra"
        "-Wpedantic"
        "-Werror"
        "/showIncludes")
endif()

#----------------------------------------------------------------------------------------#
# visibility build flags
#
add_interface_library(hosttrace-default-visibility
    "Adds -fvisibility=default compiler flag")
add_interface_library(hosttrace-hidden-visibility
    "Adds -fvisibility=hidden compiler flag")

add_target_flag_if_avail(hosttrace-default-visibility
    "-fvisibility=default")
add_target_flag_if_avail(hosttrace-hidden-visibility
    "-fvisibility=hidden" "-fvisibility-inlines-hidden")

foreach(_TYPE default hidden)
    if(NOT cxx_hosttrace_${_TYPE}_visibility_fvisibility_${_TYPE})
        add_disabled_interface(hosttrace-${_TYPE}-visibility)
    endif()
endforeach()

#----------------------------------------------------------------------------------------#
# developer build flags
#
if(dl_LIBRARY)
    # This instructs the linker to add all symbols, not only used ones, to the dynamic
    # symbol table. This option is needed for some uses of dlopen or to allow obtaining
    # backtraces from within a program.
    add_flag_if_avail("-rdynamic")
endif()

#----------------------------------------------------------------------------------------#
# sanitizer
#
set(HOSTTRACE_SANITIZER_TYPES address memory thread leak undefined unreachable null bounds alignment)
set_property(CACHE HOSTTRACE_SANITIZER_TYPE PROPERTY STRINGS "${HOSTTRACE_SANITIZER_TYPES}")
add_interface_library(hosttrace-sanitizer-compile-options "Adds compiler flags for sanitizers")
add_interface_library(hosttrace-sanitizer
    "Adds compiler flags to enable ${HOSTTRACE_SANITIZER_TYPE} sanitizer (-fsanitizer=${HOSTTRACE_SANITIZER_TYPE})")

set(COMMON_SANITIZER_FLAGS "-fno-optimize-sibling-calls" "-fno-omit-frame-pointer" "-fno-inline-functions")
add_target_flag(hosttrace-sanitizer-compile-options ${COMMON_SANITIZER_FLAGS})

foreach(_TYPE ${HOSTTRACE_SANITIZER_TYPES})
    set(_FLAG "-fsanitize=${_TYPE}")
    add_interface_library(hosttrace-${_TYPE}-sanitizer
        "Adds compiler flags to enable ${_TYPE} sanitizer (${_FLAG})")
    add_target_flag(hosttrace-${_TYPE}-sanitizer ${_FLAG})
    target_link_libraries(hosttrace-${_TYPE}-sanitizer INTERFACE
        hosttrace-sanitizer-compile-options)
    set_property(TARGET hosttrace-${_TYPE}-sanitizer PROPERTY
        INTERFACE_LINK_OPTIONS ${_FLAG} ${COMMON_SANITIZER_FLAGS})
endforeach()

unset(_FLAG)
unset(COMMON_SANITIZER_FLAGS)

if(HOSTTRACE_USE_SANITIZER)
    foreach(_TYPE ${HOSTTRACE_SANITIZER_TYPE})
        if(TARGET hosttrace-${_TYPE}-sanitizer)
            target_link_libraries(hosttrace-sanitizer INTERFACE hosttrace-${_TYPE}-sanitizer)
        else()
            message(FATAL_ERROR "Error! Target 'hosttrace-${_TYPE}-sanitizer' does not exist!")
        endif()
    endforeach()
else()
    set(HOSTTRACE_USE_SANITIZER OFF)
    inform_empty_interface(hosttrace-sanitizer "${HOSTTRACE_SANITIZER_TYPE} sanitizer")
endif()

if (MSVC)
    # VTune is much more helpful when debug information is included in the
    # generated release code.
    add_flag_if_avail("/Zi")
    add_flag_if_avail("/DEBUG")
endif()

#----------------------------------------------------------------------------------------#
# user customization
#
get_property(LANGUAGES GLOBAL PROPERTY ENABLED_LANGUAGES)

if(NOT APPLE OR "$ENV{CONDA_PYTHON_EXE}" STREQUAL "")
    add_user_flags(hosttrace-compile-options "CXX")
endif()
