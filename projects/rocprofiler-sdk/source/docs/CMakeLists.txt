#
#

if(NOT ROCPROFILER_BUILD_DOCS)
    return()
endif()

set(PACKAGE_NAME ${PROJECT_NAME}-sdk)

include(FetchContent)

if(CMAKE_VERSION VERSION_GREATER_EQUAL 3.24)
    cmake_policy(SET CMP0135 NEW)
endif()

if(NOT EXISTS ${PROJECT_BINARY_DIR}/external/miniconda.sh)
    file(DOWNLOAD https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh
         ${PROJECT_BINARY_DIR}/external/miniconda.sh)
endif()

function(DOCS_EXECUTE_PROCESS)
    string(REPLACE ";" " " _MSG "${ARGN}")
    message(STATUS "[rocprofiler][docs] Executing: ${_MSG}")

    execute_process(
        COMMAND ${ARGN}
        RESULT_VARIABLE _RET
        OUTPUT_VARIABLE _OUT
        ERROR_VARIABLE _ERR
        WORKING_DIRECTORY ${PROJECT_BINARY_DIR}/external COMMAND_ERROR_IS_FATAL ANY)

    if(NOT _RET EQUAL 0)
        message(STATUS "docs command failed: ${_RET}")
        message(STATUS "stderr:\n${_ERR}")
        message(STATUS "stdout:\n${_OUT}")
    endif()
endfunction()

if(NOT EXISTS ${PROJECT_BINARY_DIR}/external/miniconda)
    docs_execute_process(/bin/bash ${PROJECT_BINARY_DIR}/external/miniconda.sh -b -p
                         ${PROJECT_BINARY_DIR}/external/miniconda)
    docs_execute_process(${PROJECT_BINARY_DIR}/external/miniconda/bin/conda config --set
                         always_yes yes)
    docs_execute_process(${PROJECT_BINARY_DIR}/external/miniconda/bin/conda update -c
                         defaults -n base conda)
endif()

if(NOT EXISTS ${PROJECT_BINARY_DIR}/external/miniconda/envs/rocprofiler-docs)
    docs_execute_process(${PROJECT_BINARY_DIR}/external/miniconda/bin/conda env create -n
                         rocprofiler-docs -f ${CMAKE_CURRENT_LIST_DIR}/environment.yml)
endif()

file(
    WRITE "${CMAKE_CURRENT_BINARY_DIR}/build-docs.sh"
    "#!/bin/bash -e

export PATH=${PROJECT_BINARY_DIR}/external/miniconda/bin:\${PATH}
source activate
conda activate rocprofiler-docs
${PROJECT_SOURCE_DIR}/source/scripts/update-docs.sh 1> /dev/null
rm -r ${PROJECT_SOURCE_DIR}/build-docs
")

add_custom_target(docs ALL /bin/bash ${CMAKE_CURRENT_BINARY_DIR}/build-docs.sh)

install(
    DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/_build/html/
    DESTINATION ${CMAKE_INSTALL_DATAROOTDIR}/html/${PACKAGE_NAME}
    COMPONENT docs
    USE_SOURCE_PERMISSIONS FILES_MATCHING
    PATTERN "*")
