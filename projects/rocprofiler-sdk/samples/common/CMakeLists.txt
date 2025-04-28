#
# common utilities for samples
#

find_package(rocprofiler-sdk REQUIRED)

# rocprofiler-sdk provides a Findlibdw.cmake
find_package(libdw REQUIRED)

# default FAIL_REGULAR_EXPRESSION for tests
set(ROCPROFILER_DEFAULT_FAIL_REGEX
    "threw an exception|Permission denied|Could not create logging file|failed with error code"
    CACHE INTERNAL "Default FAIL_REGULAR_EXPRESSION for tests")

# build flags
add_library(rocprofiler-sdk-samples-build-flags INTERFACE)
add_library(rocprofiler-sdk::samples-build-flags ALIAS
            rocprofiler-sdk-samples-build-flags)
target_compile_options(rocprofiler-sdk-samples-build-flags INTERFACE -W -Wall -Wextra
                                                                     -Wshadow)
target_compile_features(rocprofiler-sdk-samples-build-flags INTERFACE cxx_std_17)

if(ROCPROFILER_BUILD_CI OR ROCPROFILER_BUILD_WERROR)
    target_compile_options(rocprofiler-sdk-samples-build-flags INTERFACE -Werror)
endif()

# common utilities
cmake_path(GET CMAKE_CURRENT_SOURCE_DIR PARENT_PATH COMMON_LIBRARY_INCLUDE_DIR)

add_library(rocprofiler-sdk-samples-common-library INTERFACE)
add_library(rocprofiler-sdk::samples-common-library ALIAS
            rocprofiler-sdk-samples-common-library)
target_link_libraries(rocprofiler-sdk-samples-common-library
                      INTERFACE rocprofiler-sdk::samples-build-flags libdw::libdw)
target_compile_features(rocprofiler-sdk-samples-common-library INTERFACE cxx_std_17)
target_include_directories(rocprofiler-sdk-samples-common-library
                           INTERFACE ${COMMON_LIBRARY_INCLUDE_DIR})

set(EXTERNAL_SUBMODULE_DIR "${PROJECT_SOURCE_DIR}/../external")
cmake_path(ABSOLUTE_PATH EXTERNAL_SUBMODULE_DIR NORMALIZE)

if(EXISTS ${EXTERNAL_SUBMODULE_DIR}/filesystem/include/ghc/filesystem.hpp)
    target_compile_definitions(
        rocprofiler-sdk-samples-common-library
        INTERFACE $<BUILD_INTERFACE:ROCPROFILER_SAMPLES_HAS_GHC_LIB_FILESYSTEM=1>)
    target_include_directories(
        rocprofiler-sdk-samples-common-library SYSTEM
        INTERFACE $<BUILD_INTERFACE:${EXTERNAL_SUBMODULE_DIR}/filesystem/include>)
endif()

# function for getting the LD_PRELOAD environment variable
function(rocprofiler_samples_get_preload_env _VAR)
    set(_PRELOAD_ENV_LIBS ${ROCPROFILER_MEMCHECK_PRELOAD_ENV_VALUE} $ENV{LD_PRELOAD})

    foreach(_TARG ${ARGN})
        if(NOT TARGET ${_TARG})
            message(
                FATAL_ERROR
                    "rocprofiler_samples_get_preload_env: '${_TARG}' is not a valid target"
                )
        endif()
        list(APPEND _PRELOAD_ENV_LIBS $<TARGET_FILE:${_TARG}>)
    endforeach()

    if(_PRELOAD_ENV_LIBS)
        string(REPLACE ";" ":" _PRELOAD_ENV "LD_PRELOAD=${_PRELOAD_ENV_LIBS}")
    endif()

    set(${_VAR}
        "${_PRELOAD_ENV}"
        PARENT_SCOPE)
endfunction()

# function for getting the LD_LIBRARY_PATH environment variable
function(rocprofiler_samples_get_ld_library_path_env _VAR)

    set(_LDLIB_PATH "LD_LIBRARY_PATH=")
    foreach(_TARG ${ARGN})
        if(NOT TARGET ${_TARG})
            message(
                FATAL_ERROR
                    "rocprofiler_samples_get_ld_library_path_env: '${_TARG}' is not a valid target"
                )
        endif()
        string(APPEND _LDLIB_PATH "$<TARGET_FILE_DIR:${_TARG}>:")
    endforeach()
    # append the environments current LD_LIBRARY_PATH
    string(APPEND _LDLIB_PATH "$ENV{LD_LIBRARY_PATH}")

    set(${_VAR}
        "${_LDLIB_PATH}"
        PARENT_SCOPE)
endfunction()
