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

set(DOCS_WD ${PROJECT_BINARY_DIR}/external)

if(NOT EXISTS ${PROJECT_BINARY_DIR}/external/miniconda.sh)
    file(DOWNLOAD https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh
         ${PROJECT_BINARY_DIR}/external/miniconda.sh)
endif()

find_program(SHELL_CMD NAME bash sh REQUIRED)

function(DOCS_EXECUTE_PROCESS)
    string(REPLACE ";" " " _MSG "${ARGN}")
    message(STATUS "[rocprofiler][docs] Executing: ${_MSG}")

    execute_process(
        COMMAND ${CMAKE_COMMAND} -E env HOME=${DOCS_WD} -- ${ARGN}
        RESULT_VARIABLE _RET
        OUTPUT_VARIABLE _OUT
        ERROR_VARIABLE _ERR
        WORKING_DIRECTORY ${PROJECT_BINARY_DIR}/external)

    if(NOT _RET EQUAL 0)
        message(STATUS "docs command failed: ${_RET}")
        message(STATUS "stderr:\n${_ERR}")
        message(STATUS "stdout:\n${_OUT}")
        string(REPLACE ";" " " _CMD "${ARGN}")
        message(WARNING "command failure: ${_CMD}")
    endif()
endfunction()

set(CONDA_ROOT ${PROJECT_BINARY_DIR}/external/miniconda)

if(NOT EXISTS ${CONDA_ROOT})
    docs_execute_process(${SHELL_CMD} ${PROJECT_BINARY_DIR}/external/miniconda.sh -b -p
                         ${CONDA_ROOT})
    docs_execute_process(${CONDA_ROOT}/bin/conda config --set always_yes yes)
    docs_execute_process(${CONDA_ROOT}/bin/conda update -c defaults -n base conda)
endif()

if(NOT EXISTS ${CONDA_ROOT}/envs/rocprofiler-docs)
    docs_execute_process(${CONDA_ROOT}/bin/conda env create -n rocprofiler-docs -f
                         ${CMAKE_CURRENT_LIST_DIR}/environment.yml)
    docs_execute_process(${CONDA_ROOT}/envs/rocprofiler-docs/bin/python -m pip install -r
                         ${CMAKE_CURRENT_LIST_DIR}/requirements.txt)
endif()

file(
    WRITE "${CMAKE_CURRENT_BINARY_DIR}/build-docs.sh"
    "#!${SHELL_CMD} -e

# this is where conda thinks the HOME directory is
HOME=${DOCS_WD}
export HOME

PATH=${CONDA_ROOT}/bin:\${PATH}
export PATH

source activate
conda activate rocprofiler-docs
${PROJECT_SOURCE_DIR}/source/scripts/update-docs.sh 1> /dev/null
rm -r ${PROJECT_SOURCE_DIR}/build-docs
")

add_custom_target(docs ALL ${SHELL_CMD} ${CMAKE_CURRENT_BINARY_DIR}/build-docs.sh)

install(
    DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/_build/html/
    DESTINATION ${CMAKE_INSTALL_DATAROOTDIR}/html/${PACKAGE_NAME}
    COMPONENT docs
    USE_SOURCE_PERMISSIONS FILES_MATCHING
    PATTERN "*")
