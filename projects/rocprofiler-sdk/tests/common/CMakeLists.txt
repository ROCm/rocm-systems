#
# common utilities for tests
#

# build flags
add_library(rocprofiler-tests-build-flags INTERFACE)
add_library(rocprofiler::tests-build-flags ALIAS rocprofiler-tests-build-flags)
target_compile_options(rocprofiler-tests-build-flags INTERFACE -W -Wall -Wextra -Wshadow)
target_compile_features(rocprofiler-tests-build-flags INTERFACE cxx_std_17)

if(ROCPROFILER_BUILD_CI OR ROCPROFILER_BUILD_WERROR)
    target_compile_options(rocprofiler-tests-build-flags INTERFACE -Werror)
endif()

# serialization library
if(NOT TARGET rocprofiler::cereal)
    get_filename_component(ROCPROFILER_SOURCE_DIR "${PROJECT_SOURCE_DIR}/.." REALPATH)

    add_library(rocprofiler-cereal INTERFACE)
    add_library(rocprofiler::cereal ALIAS rocprofiler-cereal)
    target_compile_definitions(rocprofiler-cereal
                               INTERFACE $<BUILD_INTERFACE:CEREAL_THREAD_SAFE=1>)

    if(EXISTS ${ROCPROFILER_SOURCE_DIR}/external AND COMMAND
                                                     rocprofiler_checkout_git_submodule)
        rocprofiler_checkout_git_submodule(
            RECURSIVE
            RELATIVE_PATH external/cereal
            WORKING_DIRECTORY ${ROCPROFILER_SOURCE_DIR}
            REPO_URL https://github.com/jrmadsen/cereal.git
            REPO_BRANCH "rocprofiler")

        target_include_directories(
            rocprofiler-cereal SYSTEM
            INTERFACE $<BUILD_INTERFACE:${ROCPROFILER_SOURCE_DIR}/external/cereal/include>
            )
    else()
        include(FetchContent)

        set(FETCHCONTENT_BASE_DIR ${PROJECT_BINARY_DIR}/external)

        fetchcontent_declare(
            cereal
            GIT_REPOSITORY https://github.com/jrmadsen/cereal.git
            GIT_TAG rocprofiler
            SOURCE_DIR ${PROJECT_BINARY_DIR}/external/cereal BINARY_DIR
            ${PROJECT_BINARY_DIR}/external/build/cereal-build SUBBUILD_DIR
            ${PROJECT_BINARY_DIR}/external/build/cereal-subdir)

        # This particular version of projD requires workarounds
        fetchcontent_getproperties(cereal)

        if(NOT cereal_POPULATED)
            fetchcontent_populate(cereal)
        endif()

        target_include_directories(
            rocprofiler-cereal SYSTEM
            INTERFACE $<BUILD_INTERFACE:${PROJECT_BINARY_DIR}/external/cereal/include>)
    endif()
endif()

# common utilities
cmake_path(GET CMAKE_CURRENT_SOURCE_DIR PARENT_PATH COMMON_LIBRARY_INCLUDE_DIR)

add_library(rocprofiler-tests-common-library INTERFACE)
add_library(rocprofiler::tests-common-library ALIAS rocprofiler-tests-common-library)
target_link_libraries(rocprofiler-tests-common-library
                      INTERFACE rocprofiler::tests-build-flags rocprofiler::cereal)
target_compile_features(rocprofiler-tests-common-library INTERFACE cxx_std_17)
target_include_directories(rocprofiler-tests-common-library
                           INTERFACE ${COMMON_LIBRARY_INCLUDE_DIR})

set(EXTERNAL_SUBMODULE_DIR "${PROJECT_SOURCE_DIR}/../external")
cmake_path(ABSOLUTE_PATH EXTERNAL_SUBMODULE_DIR NORMALIZE)

if(EXISTS ${EXTERNAL_SUBMODULE_DIR}/filesystem/include/ghc/filesystem.hpp)
    target_compile_definitions(
        rocprofiler-tests-common-library
        INTERFACE $<BUILD_INTERFACE:ROCPROFILER_SAMPLES_HAS_GHC_LIB_FILESYSTEM=1>)
    target_include_directories(
        rocprofiler-tests-common-library SYSTEM
        INTERFACE $<BUILD_INTERFACE:${EXTERNAL_SUBMODULE_DIR}/filesystem/include>)
endif()
