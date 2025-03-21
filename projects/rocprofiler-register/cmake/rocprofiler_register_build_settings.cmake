# ########################################################################################
#
# Handles the build settings
#
# ########################################################################################

include_guard(DIRECTORY)

include(GNUInstallDirs)
include(FindPackageHandleStandardArgs)
include(rocprofiler_register_compilers)
include(rocprofiler_register_utilities)

target_compile_definitions(rocprofiler-register-build-flags
                           INTERFACE $<$<CONFIG:DEBUG>:DEBUG>)

if(ROCPROFILER_REGISTER_BUILD_CI)
    rocprofiler_register_target_compile_definitions(rocprofiler-register-build-flags
                                                    INTERFACE ROCPROFILER_REGISTER_CI)
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
            rocprofiler_register_target_compile_definitions(
                rocprofiler-register-${_TYPE}
                INTERFACE ROCPROFILER_REGISTER_${_TYPE_UC}=1)
            target_link_libraries(rocprofiler-register-${_TYPE}
                                  INTERFACE ${${_TYPE}_LIBRARY})
            if("${_TYPE}" STREQUAL "dl" AND NOT ROCPROFILER_REGISTER_ENABLE_CLANG_TIDY)
                # This instructs the linker to add all symbols, not only used ones, to the
                # dynamic symbol table. This option is needed for some uses of dlopen or
                # to allow obtaining backtraces from within a program.
                rocprofiler_register_target_compile_options(
                    rocprofiler-register-${_TYPE}
                    LANGUAGES C CXX
                    LINK_LANGUAGES C CXX
                    INTERFACE "-rdynamic")
            endif()
        else()
            rocprofiler_register_target_compile_definitions(
                rocprofiler-register-${_TYPE}
                INTERFACE ROCPROFILER_REGISTER_${_TYPE_UC}=0)
        endif()
    endif()
endforeach()

target_link_libraries(rocprofiler-register-build-flags INTERFACE rocprofiler-register::dl)

# ----------------------------------------------------------------------------------------#
# set the compiler flags
#

set(_WARN_STACK_USAGE)
if(NOT ROCPROFILER_REGISTER_ENABLE_CLANG_TIDY)
    set(_WARN_STACK_USAGE "-Wstack-usage=8192") # 2 KB
endif()

rocprofiler_register_target_compile_options(
    rocprofiler-register-build-flags
    INTERFACE "-W" "-Wall" "-Wno-unknown-pragmas" "-fstack-protector-strong"
              "-Wstack-protector" ${_WARN_STACK_USAGE})

# ----------------------------------------------------------------------------------------#
# debug-safe optimizations
#
rocprofiler_register_target_compile_options(
    rocprofiler-register-build-flags
    LANGUAGES CXX
    INTERFACE "-faligned-new")

# ----------------------------------------------------------------------------------------#
# developer build flags
#
rocprofiler_register_target_compile_options(
    rocprofiler-register-developer-flags
    LANGUAGES C CXX
    INTERFACE "-Werror" "-Wdouble-promotion" "-Wshadow" "-Wextra"
              "-Wno-deprecated-declarations")

if(ROCPROFILER_REGISTER_BUILD_DEVELOPER)
    target_link_libraries(rocprofiler-register-build-flags
                          INTERFACE rocprofiler-register::developer-flags)
endif()

# ----------------------------------------------------------------------------------------#
# user customization
#
get_property(LANGUAGES GLOBAL PROPERTY ENABLED_LANGUAGES)

if(NOT APPLE OR "$ENV{CONDA_PYTHON_EXE}" STREQUAL "")
    rocprofiler_register_target_user_flags(rocprofiler-register-build-flags "CXX")
endif()
