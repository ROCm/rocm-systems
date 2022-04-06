#!/bin/bash -e

message()
{
    echo -e "\n\n##### ${@}... #####\n"
}

WORK_DIR=$(cd $(dirname ${BASH_SOURCE[0]}) && pwd)
SOURCE_DIR=$(cd ${WORK_DIR}/../.. &> /dev/null && pwd)

message "Working directory is ${WORK_DIR}"
message "Source directory is ${SOURCE_DIR}"

message "Changing directory to ${WORK_DIR}"
cd ${WORK_DIR}

message "Generating omnitrace.dox"
cmake -DSOURCE_DIR=${SOURCE_DIR} -P ${WORK_DIR}/generate-doxyfile.cmake

message "Generating doxygen xml files"
doxygen omnitrace.dox

message "Building html documentation"
make html

if [ -d ${SOURCE_DIR}/docs ]; then
    message "Removing stale documentation in ${SOURCE_DIR}/docs/"
    echo rm -rf ${SOURCE_DIR}/docs/*

    message "Copying source/docs/_build/html/* to docs/"
    echo cp -r ${WORK_DIR}/_build/html/* ${SOURCE_DIR}/docs/
fi
