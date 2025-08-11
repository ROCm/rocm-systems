#!/bin/bash -e

# MIT License
#
# Copyright (c) 2025 Advanced Micro Devices, Inc. All rights reserved.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.

SCRIPT_DIR=$(realpath $(dirname ${BASH_SOURCE[0]}))
cd $(dirname ${SCRIPT_DIR})
echo -e "Working directory: $(pwd)"

error-message()
{
    echo -e "\nError! ${@}\n"
    exit -1
}

verbose-run()
{
    echo -e "\n##### Executing \"${@}\"... #####\n"
    eval $@
}

if [ -d "$(realpath /tmp)" ]; then
    : ${TMPDIR:=/tmp}
    export TMPDIR
fi

: ${EXAMPLE_DIR:=examples}
: ${EXAMPLE_NAME:=user-api}
: ${SOURCE_DIR:=$(mktemp -t -d rocprof-sys-test-source-XXXX)}
: ${BINARY_DIR:=$(mktemp -t -d rocprof-sys-test-build-XXXX)}
: ${INSTALL_DIR:=$(mktemp -t -d rocprof-sys-install-XXXX)}
: ${INSTALL_SCRIPT:=""}

usage()
{
    print_option() { printf "    --%-10s %-24s     %s (default: %s)\n" "${1}" "${2}" "${3}" "${4}"; }
    echo "Options:"
    print_option example-name "<NAME>" "Name of the example project" "${EXAMPLE_NAME}"
    print_option example-dir "<PATH>" "Directory containing example" "${EXAMPLE_DIR}"
    print_option source-dir "<PATH>" "Location to copy example to" "${SOURCE_DIR}"
    print_option binary-dir "<PATH>" "Location to build" "${BINARY_DIR}"
    print_option install-dir "<PATH>" "Location of rocprofiler-systems installation" "${INSTALL_DIR}"
    print_option install-script "<FILEPATH>" "Absolute path to the installer script" ""
}

while [[ $# -gt 0 ]]
do
    ARG=${1}
    shift

    VAL=""
    while [[ $# -gt 0 ]]
    do
        VAL=${1}
        shift
        break
    done

    if [ -z "${VAL}" ]; then
        echo "Error! Missing value for argument \"${ARG}\""
        usage
        exit -1
    fi

    case "${ARG}" in
        --example-name)
            EXAMPLE_NAME=${VAL}
            continue
            ;;
        --example-dir)
            EXAMPLE_DIR=${VAL}
            continue
            ;;
        --source-dir)
            SOURCE_DIR=${VAL}
            continue
            ;;
        --binary-dir)
            BINARY_DIR=${VAL}
            continue
            ;;
        --install-dir)
            INSTALL_DIR=${VAL}
            continue
            ;;
        --install-script)
            INSTALL_SCRIPT=${VAL}
            continue
            ;;
        *)
            echo -e "Error! Unknown option : ${ARG}"
            usage
            exit -1
            ;;
    esac
done

if [ ! -f "${INSTALL_DIR}/include/rocprofiler-systems/user.h" ]; then
    if [ -z "${INSTALL_SCRIPT}" ]; then
        error-message "Unable to find \"rocprofiler-systems/user.h\" in \"${INSTALL_DIR}/include\" and installation script not provided"
    elif [ ! -f "${INSTALL_SCRIPT}" ]; then
        error-message "Unable to locate \"${INSTALL_SCRIPT}\" in directory \"${PWD}\""
    else
        verbose-run mkdir -p ${INSTALL_DIR}
        verbose-run $(realpath ${INSTALL_SCRIPT}) --prefix=${INSTALL_DIR} --skip-license --exclude-subdir
    fi
fi

if [ ! -d ${SOURCE_DIR} ]; then
    verbose-run mkdir -p ${SOURCE_DIR}
fi

verbose-run cp -v -r ${EXAMPLE_DIR}/${EXAMPLE_NAME}/* ${SOURCE_DIR}/

verbose-run pushd ${SOURCE_DIR}

cat << EOF > CMakeLists.txt
cmake_minimum_required(VERSION 3.18.4.0 FATAL_ERROR)

project(test LANGUAGES C CXX)

set(CMAKE_C_STANDARD 11)
set(CMAKE_CXX_STANDARD 17)

set(CMAKE_C_STANDARD_REQUIRED ON)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(rocprofiler-systems REQUIRED COMPONENTS user)
find_package(Threads REQUIRED)

get_target_property(LIBS rocprofiler-systems::rocprofiler-systems INTERFACE_LINK_LIBRARIES)
foreach(_LIB \${LIBS})
    foreach(_VAR LINK_LIBRARIES INCLUDE_DIRECTORIES)
        get_target_property(_VAL \${_LIB} INTERFACE_\${_VAR})
        if(_VAL)
            message(STATUS "\${_LIB} :: \${_VAR} :: \${_VAL}")
        endif()
    endforeach()
endforeach()

file(GLOB sources \${CMAKE_CURRENT_LIST_DIR}/*.cpp \${CMAKE_CURRENT_LIST_DIR}/*.c)
add_executable(app \${sources})
target_link_libraries(app PRIVATE Threads::Threads rocprofiler-systems::rocprofiler-systems)
EOF

export CMAKE_PREFIX_PATH=${INSTALL_DIR}:${CMAKE_PREFIX_PATH}

verbose-run find .
verbose-run cmake -B ${BINARY_DIR} ${SOURCE_DIR}
verbose-run cmake --build ${BINARY_DIR} --target all --parallel 2 -- VERBOSE=1

set +e

verbose-run pushd ${BINARY_DIR}
verbose-run ./app
verbose-run popd
verbose-run popd
