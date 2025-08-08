# ########################################################################################
#
# Handles the build settings
#
# ########################################################################################

include_guard(DIRECTORY)

include(GNUInstallDirs)
include(FindPackageHandleStandardArgs)
include(rocprofiler_compilers)
include(rocprofiler_utilities)

target_compile_definitions(rocprofiler-sdk-build-flags INTERFACE $<$<CONFIG:DEBUG>:DEBUG>)

if(ROCPROFILER_BUILD_CI)
    rocprofiler_target_compile_definitions(rocprofiler-sdk-build-flags
                                           INTERFACE ROCPROFILER_CI)
endif()

if(ROCPROFILER_BUILD_CODECOV)
    target_link_libraries(rocprofiler-sdk-build-flags INTERFACE gcov)
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

foreach(_TYPE dl rt)
    if(NOT ${_TYPE}_LIBRARY)
        find_library(${_TYPE}_LIBRARY NAMES ${_TYPE})
        find_package_handle_standard_args(${_TYPE}-library REQUIRED_VARS ${_TYPE}_LIBRARY)
        if(${_TYPE}-library_FOUND)
            string(TOUPPER "${_TYPE}" _TYPE_UC)
            rocprofiler_target_compile_definitions(rocprofiler-sdk-${_TYPE}
                                                   INTERFACE ROCPROFILER_${_TYPE_UC}=1)
            target_link_libraries(rocprofiler-sdk-${_TYPE} INTERFACE ${${_TYPE}_LIBRARY})
            if("${_TYPE}" STREQUAL "dl" AND NOT ROCPROFILER_ENABLE_CLANG_TIDY)
                # This instructs the linker to add all symbols, not only used ones, to the
                # dynamic symbol table. This option is needed for some uses of dlopen or
                # to allow obtaining backtraces from within a program.
                rocprofiler_target_compile_options(
                    rocprofiler-sdk-${_TYPE}
                    LANGUAGES C CXX
                    LINK_LANGUAGES C CXX
                    INTERFACE "-rdynamic")
            endif()
        else()
            rocprofiler_target_compile_definitions(rocprofiler-sdk-${_TYPE}
                                                   INTERFACE ROCPROFILER_${_TYPE_UC}=0)
        endif()
    endif()
endforeach()

target_link_libraries(rocprofiler-sdk-build-flags
                      INTERFACE rocprofiler-sdk::rocprofiler-sdk-dl)

# ----------------------------------------------------------------------------------------#
# set the compiler flags
#
rocprofiler_target_compile_options(rocprofiler-sdk-build-flags
                                   INTERFACE "-W" "-Wall" "-Wno-unknown-pragmas")

# compiler version specific flags
function(set_compiler_options compiler_id version)
    if(CMAKE_CXX_COMPILER_ID STREQUAL ${compiler_id} AND CMAKE_CXX_COMPILER_VERSION
                                                         VERSION_LESS_EQUAL ${version})
        rocprofiler_target_compile_options(
            rocprofiler-sdk-build-flags
            INTERFACE "-Wno-error=extra" "-Wno-unused-variable"
                      "-Wno-error=unused-but-set-variable"
                      "-Wno-error=unused-but-set-parameter" "-Wno-error=shadow")
    endif()
endfunction()

# Check GCC version
set_compiler_options("GNU" "8.5")

# Check Clang version
set_compiler_options("Clang" "13.0")

# ----------------------------------------------------------------------------------------#
# extra flags for debug information in debug or optimized binaries
#

rocprofiler_target_compile_options(
    rocprofiler-sdk-debug-flags INTERFACE "-g3" "-fno-omit-frame-pointer"
                                          "-fno-optimize-sibling-calls")

target_compile_options(
    rocprofiler-sdk-debug-flags
    INTERFACE $<$<COMPILE_LANGUAGE:C>:$<$<C_COMPILER_ID:GNU>:-rdynamic>>
              $<$<COMPILE_LANGUAGE:CXX>:$<$<CXX_COMPILER_ID:GNU>:-rdynamic>>)

if(NOT APPLE AND NOT ROCPROFILER_ENABLE_CLANG_TIDY)
    target_link_options(rocprofiler-sdk-debug-flags INTERFACE
                        $<$<CXX_COMPILER_ID:GNU>:-rdynamic>)
endif()

if(dl_LIBRARY)
    target_link_libraries(rocprofiler-sdk-debug-flags INTERFACE ${dl_LIBRARY})
endif()

if(rt_LIBRARY)
    target_link_libraries(rocprofiler-sdk-debug-flags INTERFACE ${rt_LIBRARY})
endif()

if(ROCPROFILER_BUILD_DEBUG)
    target_link_libraries(rocprofiler-sdk-build-flags
                          INTERFACE rocprofiler-sdk::rocprofiler-sdk-debug-flags)
endif()

# ----------------------------------------------------------------------------------------#
# debug-safe optimizations
#
rocprofiler_target_compile_options(
    rocprofiler-sdk-build-flags
    LANGUAGES CXX
    INTERFACE "-faligned-new")

# ----------------------------------------------------------------------------------------#
# fstack-protector
#
rocprofiler_target_compile_options(
    rocprofiler-sdk-stack-protector
    LANGUAGES C CXX
    INTERFACE "-fstack-protector-strong" "-Wstack-protector")

if(ROCPROFILER_BUILD_STACK_PROTECTOR)
    target_link_libraries(rocprofiler-sdk-build-flags
                          INTERFACE rocprofiler-sdk::rocprofiler-sdk-stack-protector)
endif()

# ----------------------------------------------------------------------------------------#
# developer build flags
#
rocprofiler_target_compile_options(
    rocprofiler-sdk-developer-flags
    LANGUAGES C CXX
    INTERFACE "-Werror" "-Wdouble-promotion" "-Wshadow" "-Wextra" "-Wvla")

if(ROCPROFILER_BUILD_DEVELOPER)
    target_link_libraries(rocprofiler-sdk-build-flags
                          INTERFACE rocprofiler-sdk::rocprofiler-sdk-developer-flags)
endif()

# ----------------------------------------------------------------------------------------#
# release build flags
#
rocprofiler_target_compile_options(
    rocprofiler-sdk-release-flags
    LANGUAGES C CXX
    INTERFACE "-g1" "-feliminate-unused-debug-symbols" "-gno-column-info"
              "-gno-variable-location-views" "-gline-tables-only")

if(ROCPROFILER_BUILD_RELEASE)
    target_link_libraries(rocprofiler-sdk-build-flags
                          INTERFACE rocprofiler-sdk::rocprofiler-sdk-release-flags)
endif()

# ----------------------------------------------------------------------------------------#
# static lib flags
#
target_compile_options(
    rocprofiler-sdk-static-libgcc
    INTERFACE $<$<COMPILE_LANGUAGE:C>:$<$<C_COMPILER_ID:GNU>:-static-libgcc>>
              $<$<COMPILE_LANGUAGE:CXX>:$<$<CXX_COMPILER_ID:GNU>:-static-libgcc>>)
target_link_options(
    rocprofiler-sdk-static-libgcc INTERFACE
    $<$<COMPILE_LANGUAGE:C>:$<$<C_COMPILER_ID:GNU,Clang>:-static-libgcc>>
    $<$<COMPILE_LANGUAGE:CXX>:$<$<CXX_COMPILER_ID:GNU,Clang>:-static-libgcc>>)

target_compile_options(
    rocprofiler-sdk-static-libstdcxx
    INTERFACE $<$<COMPILE_LANGUAGE:CXX>:$<$<CXX_COMPILER_ID:GNU>:-static-libstdc++>>)
target_link_options(
    rocprofiler-sdk-static-libstdcxx INTERFACE
    $<$<COMPILE_LANGUAGE:CXX>:$<$<CXX_COMPILER_ID:GNU,Clang>:-static-libstdc++>>)

if(ROCPROFILER_BUILD_STATIC_LIBGCC)
    target_link_libraries(rocprofiler-sdk-build-flags
                          INTERFACE rocprofiler-sdk::rocprofiler-sdk-static-libgcc)
endif()

if(ROCPROFILER_BUILD_STATIC_LIBSTDCXX)
    target_link_libraries(rocprofiler-sdk-build-flags
                          INTERFACE rocprofiler-sdk::rocprofiler-sdk-static-libstdcxx)
endif()

if(ROCPROFILER_UNSAFE_NO_VERSION_CHECK)
    rocprofiler_target_compile_definitions(rocprofiler-sdk-build-flags
                                           INTERFACE ROCPROFILER_UNSAFE_NO_VERSION_CHECK)
endif()

if(ROCPROFILER_BUILD_CI_STRICT_TIMESTAMPS)
    rocprofiler_target_compile_definitions(rocprofiler-sdk-build-flags
                                           INTERFACE ROCPROFILER_CI_STRICT_TIMESTAMPS)
endif()

# ----------------------------------------------------------------------------------------#
# extra flags for compiling with experimental warnings
#
target_compile_definitions(rocprofiler-sdk-experimental-flags
                           INTERFACE ROCPROFILER_SDK_EXPERIMENTAL_WARNINGS=1)
rocprofiler_target_compile_options(rocprofiler-sdk-experimental-flags
                                   INTERFACE "-Wno-deprecated-declarations")

if(ROCPROFILER_BUILD_EXPERIMENTAL_WARNINGS)
    target_link_libraries(rocprofiler-sdk-build-flags
                          INTERFACE rocprofiler-sdk::rocprofiler-sdk-experimental-flags)
endif()

# ----------------------------------------------------------------------------------------#
# extra flags for compiling with deprecated warnings
#
target_compile_definitions(rocprofiler-sdk-deprecated-flags
                           INTERFACE ROCPROFILER_SDK_DEPRECATED_WARNINGS=1)
rocprofiler_target_compile_options(rocprofiler-sdk-deprecated-flags
                                   INTERFACE "-Wdeprecated-declarations")

if(ROCPROFILER_BUILD_DEPRECATED_WARNINGS)
    target_link_libraries(rocprofiler-sdk-build-flags
                          INTERFACE rocprofiler-sdk::rocprofiler-sdk-deprecated-flags)
endif()

# ----------------------------------------------------------------------------------------#
# user customization
#
get_property(LANGUAGES GLOBAL PROPERTY ENABLED_LANGUAGES)

if(NOT APPLE OR "$ENV{CONDA_PYTHON_EXE}" STREQUAL "")
    rocprofiler_target_user_flags(rocprofiler-sdk-build-flags "CXX")
endif()
