#!/bin/bash -e

message()
{
    echo -e "\n\n##### ${@}... #####\n"
}

WORK_DIR=$(dirname ${BASH_SOURCE[0]})

message "Changing directory to ${WORK_DIR}"
cd ${WORK_DIR}

SOURCE_DIR=$(cd ${WORK_DIR}/.. &> /dev/null && pwd)
message "Source directory is ${SOURCE_DIR}"

message "Generating omnitrace.dox"
cmake -DSOURCE_DIR=${SOURCE_DIR} -P ${WORK_DIR}/generate-doxyfile.cmake

message "Generating doxygen xml files"
doxygen omnitrace.dox

message "Building html documentation"
make html

message "Removing stale documentation in ${SOURCE_DIR}/docs/"
rm -rf ${SOURCE_DIR}/docs/*

message "Copying docs-source/_build/html/* to docs/"
cp -r ${WORK_DIR}/_build/html/* ${SOURCE_DIR}/docs/
