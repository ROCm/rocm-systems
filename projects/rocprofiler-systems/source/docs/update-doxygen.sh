#!/bin/bash -e

WORK_DIR=$(dirname ${BASH_SOURCE[0]})

SOURCE_DIR=$(cd ${WORK_DIR}/.. &> /dev/null && pwd)

cmake -DSOURCE_DIR=${SOURCE_DIR} -P generate-doxyfile.cmake

doxygen omnitrace.dox
