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

set(DOCS_WD ${CMAKE_CURRENT_BINARY_DIR})
set(CONDA_ROOT ${PROJECT_BINARY_DIR}/external/miniconda)

find_program(SHELL_CMD NAMES bash sh REQUIRED)
find_program(CHMOD_CMD NAMES chmod)

if(NOT EXISTS ${PROJECT_BINARY_DIR}/external/miniconda.sh)
    message(
        STATUS
            "Downloading https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh"
        )
    file(
        DOWNLOAD https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh
        ${PROJECT_BINARY_DIR}/external/miniconda.sh
        STATUS MINICONDA_DOWNLOAD_RET
        INACTIVITY_TIMEOUT 60
        SHOW_PROGRESS)

    if(NOT MINICONDA_DOWNLOAD_RET EQUAL 0)
        message(
            FATAL_ERROR
                "Download to ${PROJECT_BINARY_DIR}/external/miniconda.sh from https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh failed"
            )
    endif()
endif()

function(DOCS_EXECUTE_PROCESS)
    string(REPLACE ";" " " _MSG "${ARGN}")
    message(STATUS "[rocprofiler][docs] Executing: ${_MSG}")

    execute_process(
        COMMAND ${CMAKE_COMMAND} -E env HOME=${DOCS_WD} ${ARGN}
        RESULT_VARIABLE _RET
        OUTPUT_VARIABLE _OUT
        ERROR_VARIABLE _ERR
        WORKING_DIRECTORY ${PROJECT_BINARY_DIR}/external)

    if(NOT _RET EQUAL 0)
        message(STATUS "docs command failed: ${_RET}")
        message(STATUS "stderr:\n${_ERR}")
        message(STATUS "stdout:\n${_OUT}")
        string(REPLACE ";" " " _CMD "${ARGN}")
        message(FATAL_ERROR "command failure: ${_CMD}")
    endif()
endfunction()

file(
    WRITE "${CMAKE_CURRENT_BINARY_DIR}/build-docs.sh"
    "#!${SHELL_CMD}

set -e
set -x

HOME=${DOCS_WD}
export HOME

env

if [ ! -d '${CONDA_ROOT}' ]; then
    chmod +x ${PROJECT_BINARY_DIR}/external/miniconda.sh
    ${PROJECT_BINARY_DIR}/external/miniconda.sh -b -p ${CONDA_ROOT}
fi

PATH=${PROJECT_BINARY_DIR}/external/miniconda/bin:\${PATH}
export PATH

source ${CONDA_ROOT}/bin/activate

conda config --set always_yes yes
conda update -c defaults -n base conda

if [ ! -d '${CONDA_ROOT}/envs/rocprofiler-docs' ]; then
    ${CONDA_ROOT}/bin/conda env create -n rocprofiler-docs -f ${CMAKE_CURRENT_LIST_DIR}/environment.yml
fi

which python

conda activate rocprofiler-docs

which python

python -m pip install -r ${CMAKE_CURRENT_LIST_DIR}/requirements.txt

WORK_DIR=${PROJECT_SOURCE_DIR}/source/docs
SOURCE_DIR=${PROJECT_SOURCE_DIR}

cd \${SOURCE_DIR}

cmake -B build-docs \${SOURCE_DIR} -DROCPROFILER_INTERNAL_BUILD_DOCS=ON

cd \${WORK_DIR}

cmake -DSOURCE_DIR=\${SOURCE_DIR} -P \${WORK_DIR}/generate-doxyfile.cmake

doxygen rocprofiler-sdk.dox
doxygen rocprofiler-sdk.dox

doxysphinx build \${WORK_DIR} \${WORK_DIR}/_build/html \${WORK_DIR}/_doxygen/html

make html SPHINXOPTS=\"-W --keep-going -n\"

rm -rf ${PROJECT_SOURCE_DIR}/build-docs
")

if(CHMOD_CMD)
    docs_execute_process(${CHMOD_CMD} +x ${CMAKE_CURRENT_BINARY_DIR}/build-docs.sh)
    add_custom_target(docs ALL ${CMAKE_COMMAND} -E env HOME=${DOCS_WD}
                               ${CMAKE_CURRENT_BINARY_DIR}/build-docs.sh)
else()
    add_custom_target(docs ALL ${CMAKE_COMMAND} -E env HOME=${DOCS_WD} ${SHELL_CMD}
                               ${CMAKE_CURRENT_BINARY_DIR}/build-docs.sh)
endif()

install(
    DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/_build/html/
    DESTINATION ${CMAKE_INSTALL_DATAROOTDIR}/html/${PACKAGE_NAME}
    COMPONENT docs
    USE_SOURCE_PERMISSIONS FILES_MATCHING
    PATTERN "*")
