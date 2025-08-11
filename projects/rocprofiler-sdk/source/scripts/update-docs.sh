#!/bin/bash -e

message()
{
    echo -e "\n\n##### ${@}... #####\n"
}

WORK_DIR=$(cd $(dirname ${BASH_SOURCE[0]})/../docs &> /dev/null && pwd)
SOURCE_DIR=$(cd ${WORK_DIR}/../.. &> /dev/null && pwd)

message "Working directory is ${WORK_DIR}"
message "Source directory is ${SOURCE_DIR}"

message "Changing directory to ${SOURCE_DIR}"
cd ${SOURCE_DIR}

message "Configurating cmake..."
cmake -B build-docs ${SOURCE_DIR} -DROCPROFILER_INTERNAL_BUILD_DOCS=ON

message "Changing directory to ${WORK_DIR}"
cd ${WORK_DIR}

message "Generating rocprofiler-sdk.dox"
cmake -DSOURCE_DIR=${SOURCE_DIR} -DPROJECT_NAME="ROCprofiler-SDK" -P ${WORK_DIR}/generate-doxyfile.cmake

message "Generating doxygen xml files"
mkdir -p _doxygen
doxygen rocprofiler-sdk.dox
doxygen rocprofiler-sdk-roctx.dox

message "Running doxysphinx"
doxysphinx build ${WORK_DIR} ${WORK_DIR}/_build/html ${WORK_DIR}/_doxygen/rocprofiler-sdk/html
doxysphinx build ${WORK_DIR} ${WORK_DIR}/_build/html ${WORK_DIR}/_doxygen/roctx/html

message "Building html documentation"
make html SPHINXOPTS="--keep-going -n -q -T"

if [ -d ${SOURCE_DIR}/docs ]; then
    message "Removing stale documentation in ${SOURCE_DIR}/docs/"
    rm -rf ${SOURCE_DIR}/docs/*

    message "Adding nojekyll to docs/"
    cp -r ${WORK_DIR}/.nojekyll ${SOURCE_DIR}/docs/.nojekyll

    message "Copying source/docs/_build/html/* to docs/"
    cp -r ${WORK_DIR}/_build/html/* ${SOURCE_DIR}/docs/
fi
