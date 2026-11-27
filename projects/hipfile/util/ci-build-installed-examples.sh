#!/usr/bin/env bash
# Copyright (c) Advanced Micro Devices, Inc. All rights reserved.
#
# SPDX-License-Identifier: MIT

# Used by CI to check the examples that hipFile installs. Every example
# directory in the source tree has to be installed with a CMakeLists.txt, and
# each one has to configure and build standalone against the installed hipFile
# without warnings. Then runs api/get-version, the one example that needs
# neither a GPU nor an AIS-capable filesystem, and checks that the version in
# the header it was compiled with matches the library's.
#
# The example directories are built in place in the installed tree, since
# some of them pull in ../common, with their build directories under BUILD_DIR.
#
# Usage:
# ci-build-installed-examples.sh SOURCE_DIR INSTALLED_DIR BUILD_DIR [CMAKE_ARG...]
#   SOURCE_DIR:    The examples directory in the hipFile source tree, which
#                  says which example directories should be installed
#   INSTALLED_DIR: The installed examples directory,
#                  <prefix>/share/doc/hipfile/examples
#   BUILD_DIR:     The directory to build in, outside INSTALLED_DIR
#   CMAKE_ARG:     Extra arguments for configuring each example directory,
#                  such as -DCMAKE_CXX_COMPILER=... or -DCMAKE_PREFIX_PATH=...

set -euo pipefail
shopt -s nullglob

if (( $# < 3 )); then
    echo "Usage: $0 SOURCE_DIR INSTALLED_DIR BUILD_DIR [CMAKE_ARG...]" >&2
    exit 2
fi

SOURCE_DIR=$1
INSTALLED_DIR=$2
BUILD_DIR=$3
shift 3

example_dirs=()
for dir in "${SOURCE_DIR}"/*/; do
    example_dirs+=("$(basename "${dir}")")
done
if (( ${#example_dirs[@]} == 0 )); then
    echo "No example directories in ${SOURCE_DIR}" >&2
    exit 1
fi

# Report every missing CMakeLists.txt before building anything. A directory
# missing from HIPFILE_EXAMPLE_DIRS is installed without one.
missing=0
for dir in "${example_dirs[@]}"; do
    if [[ ! -f "${INSTALLED_DIR}/${dir}/CMakeLists.txt" ]]; then
        echo "Not installed: ${INSTALLED_DIR}/${dir}/CMakeLists.txt" >&2
        missing=1
    fi
done
if (( missing )); then
    exit 1
fi

for dir in "${example_dirs[@]}"; do
    echo "Building the installed ${dir} examples"
    # A -DCMAKE_COMPILE_WARNING_AS_ERROR=OFF in CMAKE_ARG comes later on the
    # command line, so it overrides this one
    cmake \
        -S "${INSTALLED_DIR}/${dir}" \
        -B "${BUILD_DIR}/${dir}" \
        -DCMAKE_COMPILE_WARNING_AS_ERROR=ON \
        "$@"
    cmake --build "${BUILD_DIR}/${dir}" --parallel
done

echo "Running the installed get-version example"
output="$("${BUILD_DIR}/api/get-version")"
echo "${output}"

# A mismatch means the examples found another hipFile's header or library,
# such as one in a different prefix
header_version="$(sed -n 's/^Version from the header symbols.*: //p' <<< "${output}")"
library_version="$(sed -n 's/^Version from hipFileGetVersion().*: //p' <<< "${output}")"
if [[ -z "${header_version}" || -z "${library_version}" ]]; then
    echo "Can't find the versions in get-version's output" >&2
    exit 1
fi
if [[ "${header_version}" != "${library_version}" ]]; then
    echo "Header version ${header_version} != library version ${library_version}" >&2
    exit 1
fi
