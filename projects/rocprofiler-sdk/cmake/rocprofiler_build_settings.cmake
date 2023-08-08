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

target_compile_definitions(rocprofiler-build-flags INTERFACE $<$<CONFIG:DEBUG>:DEBUG>)

if(ROCPROFILER_BUILD_CI)
    rocprofiler_target_compile_definitions(rocprofiler-build-flags
                                           INTERFACE ROCPROFILER_CI)
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
            rocprofiler_target_compile_definitions(rocprofiler-${_TYPE}
                                                   INTERFACE ROCPROFILER_${_TYPE_UC}=1)
            target_link_libraries(rocprofiler-${_TYPE} INTERFACE ${${_TYPE}_LIBRARY})
            if("${_TYPE}" STREQUAL "dl")
                # This instructs the linker to add all symbols, not only used ones, to the
                # dynamic symbol table. This option is needed for some uses of dlopen or
                # to allow obtaining backtraces from within a program.
                rocprofiler_target_compile_options(
                    rocprofiler-${_TYPE}
                    LANGUAGES C CXX
                    LINK_LANGUAGES C CXX
                    INTERFACE "-rdynamic")
            endif()
        else()
            rocprofiler_target_compile_definitions(rocprofiler-${_TYPE}
                                                   INTERFACE ROCPROFILER_${_TYPE_UC}=0)
        endif()
    endif()
endforeach()

target_link_libraries(rocprofiler-build-flags INTERFACE rocprofiler::rocprofiler-dl)

# ----------------------------------------------------------------------------------------#
# set the compiler flags
#
rocprofiler_target_compile_options(rocprofiler-build-flags
                                   INTERFACE "-W" "-Wall" "-Wno-unknown-pragmas")

# ----------------------------------------------------------------------------------------#
# extra flags for debug information in debug or optimized binaries
#

rocprofiler_target_compile_options(
    rocprofiler-debug-flags INTERFACE "-g3" "-fno-omit-frame-pointer"
                                      "-fno-optimize-sibling-calls")

target_compile_options(
    rocprofiler-debug-flags
    INTERFACE $<$<COMPILE_LANGUAGE:C>:$<$<C_COMPILER_ID:GNU>:-rdynamic>>
              $<$<COMPILE_LANGUAGE:CXX>:$<$<CXX_COMPILER_ID:GNU>:-rdynamic>>)

if(NOT APPLE)
    target_link_options(rocprofiler-debug-flags INTERFACE
                        $<$<CXX_COMPILER_ID:GNU>:-rdynamic>)
endif()

if(dl_LIBRARY)
    target_link_libraries(rocprofiler-debug-flags INTERFACE ${dl_LIBRARY})
endif()

if(rt_LIBRARY)
    target_link_libraries(rocprofiler-debug-flags INTERFACE ${rt_LIBRARY})
endif()

if(ROCPROFILER_BUILD_DEBUG)
    target_link_libraries(rocprofiler-build-flags
                          INTERFACE rocprofiler::rocprofiler-debug-flags)
endif()

# ----------------------------------------------------------------------------------------#
# debug-safe optimizations
#
rocprofiler_target_compile_options(
    rocprofiler-build-flags
    LANGUAGES CXX
    INTERFACE "-faligned-new")

# ----------------------------------------------------------------------------------------#
# fstack-protector
#
rocprofiler_target_compile_options(
    rocprofiler-stack-protector
    LANGUAGES C CXX
    INTERFACE "-fstack-protector-strong" "-Wstack-protector")

if(ROCPROFILER_BUILD_STACK_PROTECTOR)
    target_link_libraries(rocprofiler-build-flags
                          INTERFACE rocprofiler::rocprofiler-stack-protector)
endif()

# ----------------------------------------------------------------------------------------#
# developer build flags
#
rocprofiler_target_compile_options(
    rocprofiler-developer-flags
    LANGUAGES C CXX
    INTERFACE "-Werror" "-Wdouble-promotion" "-Wshadow" "-Wextra"
              "-Wstack-usage=524288" # 512 KB
    )

if(ROCPROFILER_BUILD_DEVELOPER)
    target_link_libraries(rocprofiler-build-flags
                          INTERFACE rocprofiler::rocprofiler-developer-flags)
endif()

# ----------------------------------------------------------------------------------------#
# release build flags
#
rocprofiler_target_compile_options(
    rocprofiler-release-flags
    LANGUAGES C CXX
    INTERFACE "-g1" "-feliminate-unused-debug-symbols" "-gno-column-info"
              "-gno-variable-location-views" "-gline-tables-only")

if(ROCPROFILER_BUILD_RELEASE)
    target_link_libraries(rocprofiler-build-flags
                          INTERFACE rocprofiler::rocprofiler-release-flags)
endif()

# ----------------------------------------------------------------------------------------#
# static lib flags
#
target_compile_options(
    rocprofiler-static-libgcc
    INTERFACE $<$<COMPILE_LANGUAGE:C>:$<$<C_COMPILER_ID:GNU>:-static-libgcc>>
              $<$<COMPILE_LANGUAGE:CXX>:$<$<CXX_COMPILER_ID:GNU>:-static-libgcc>>)
target_link_options(
    rocprofiler-static-libgcc INTERFACE
    $<$<COMPILE_LANGUAGE:C>:$<$<C_COMPILER_ID:GNU,Clang>:-static-libgcc>>
    $<$<COMPILE_LANGUAGE:CXX>:$<$<CXX_COMPILER_ID:GNU,Clang>:-static-libgcc>>)

target_compile_options(
    rocprofiler-static-libstdcxx
    INTERFACE $<$<COMPILE_LANGUAGE:CXX>:$<$<CXX_COMPILER_ID:GNU>:-static-libstdc++>>)
target_link_options(
    rocprofiler-static-libstdcxx INTERFACE
    $<$<COMPILE_LANGUAGE:CXX>:$<$<CXX_COMPILER_ID:GNU,Clang>:-static-libstdc++>>)

if(ROCPROFILER_BUILD_STATIC_LIBGCC)
    target_link_libraries(rocprofiler-build-flags
                          INTERFACE rocprofiler::rocprofiler-static-libgcc)
endif()

if(ROCPROFILER_BUILD_STATIC_LIBSTDCXX)
    target_link_libraries(rocprofiler-build-flags
                          INTERFACE rocprofiler::rocprofiler-static-libstdcxx)
endif()

# ----------------------------------------------------------------------------------------#
# user customization
#
get_property(LANGUAGES GLOBAL PROPERTY ENABLED_LANGUAGES)

if(NOT APPLE OR "$ENV{CONDA_PYTHON_EXE}" STREQUAL "")
    rocprofiler_target_user_flags(rocprofiler-build-flags "CXX")
endif()
