#!/bin/bash -e

WORK_DIR=$(cd $(dirname ${BASH_SOURCE[0]})/../docs &> /dev/null && pwd)
SOURCE_DIR=$(cd ${WORK_DIR}/../.. &> /dev/null && pwd)

pushd ${SOURCE_DIR}
cmake -B build-docs ${SOURCE_DIR} -DROCPROFILER_INTERNAL_BUILD_DOCS=ON
popd

pushd ${WORK_DIR}
cmake -DSOURCE_DIR=${SOURCE_DIR} -P generate-doxyfile.cmake

doxygen rocprofiler-sdk.dox

doxysphinx build ${WORK_DIR} ${WORK_DIR}/_build/html ${WORK_DIR}/_doxygen/html
popd
