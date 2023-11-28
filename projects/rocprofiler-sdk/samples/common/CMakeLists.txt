#
# common utilities for samples
#

# build flags
add_library(rocprofiler-samples-build-flags INTERFACE)
add_library(rocprofiler::samples-build-flags ALIAS rocprofiler-samples-build-flags)
target_compile_options(rocprofiler-samples-build-flags INTERFACE -W -Wall -Wextra
                                                                 -Wshadow)
target_compile_features(rocprofiler-samples-build-flags INTERFACE cxx_std_17)

if(ROCPROFILER_BUILD_CI OR ROCPROFILER_BUILD_WERROR)
    target_compile_options(rocprofiler-samples-build-flags INTERFACE -Werror)
endif()

# common utilities
cmake_path(GET CMAKE_CURRENT_SOURCE_DIR PARENT_PATH COMMON_LIBRARY_INCLUDE_DIR)

add_library(rocprofiler-samples-common-library INTERFACE)
add_library(rocprofiler::samples-common-library ALIAS rocprofiler-samples-common-library)
target_link_libraries(rocprofiler-samples-common-library
                      INTERFACE rocprofiler::samples-build-flags)
target_compile_features(rocprofiler-samples-common-library INTERFACE cxx_std_17)
target_include_directories(rocprofiler-samples-common-library
                           INTERFACE ${COMMON_LIBRARY_INCLUDE_DIR})

set(EXTERNAL_SUBMODULE_DIR "${PROJECT_SOURCE_DIR}/../external")
cmake_path(ABSOLUTE_PATH EXTERNAL_SUBMODULE_DIR NORMALIZE)

if(EXISTS ${EXTERNAL_SUBMODULE_DIR}/filesystem/include/ghc/filesystem.hpp)
    target_compile_definitions(
        rocprofiler-samples-common-library
        INTERFACE $<BUILD_INTERFACE:ROCPROFILER_SAMPLES_HAS_GHC_LIB_FILESYSTEM=1>)
    target_include_directories(
        rocprofiler-samples-common-library SYSTEM
        INTERFACE $<BUILD_INTERFACE:${EXTERNAL_SUBMODULE_DIR}/filesystem/include>)
endif()
