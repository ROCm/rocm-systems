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

message "Building html documentation"
make html

if [ -d ${SOURCE_DIR}/docs ]; then
    message "Removing stale documentation in ${SOURCE_DIR}/docs/"
    rm -rf ${SOURCE_DIR}/docs/*

    message "Adding nojekyll to docs/"
    cp -r ${WORK_DIR}/.nojekyll ${SOURCE_DIR}/docs/.nojekyll

    message "Copying source/docs/_build/html/* to docs/"
    cp -r ${WORK_DIR}/_build/html/* ${SOURCE_DIR}/docs/
fi