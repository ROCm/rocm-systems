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

tolower()
{
    echo "$@" | awk -F '\\|~\\|' '{print tolower($1)}';
}

toupper()
{
    echo "$@" | awk -F '\\|~\\|' '{print toupper($1)}';
}

: ${CMAKE_BUILD_PARALLEL_LEVEL:=$(nproc)}
: ${DASHBOARD_MODE:="Continuous"}
: ${DASHBOARD_STAGES:="Start Update Configure Build Test MemCheck Coverage Submit"}
: ${SOURCE_DIR:=${PWD}}
: ${BINARY_DIR:=${PWD}/build}
: ${SITE:=$(hostname)}
: ${NAME:=""}
: ${SUBMIT_URL:="my.cdash.org/submit.php?project=rocprofiler-systems"}
: ${CODECOV:=0}

usage()
{
    print_option() { printf "    --%-20s %-24s     %s\n" "${1}" "${2}" "${3}"; }
    print_default_option() { printf "    --%-20s %-24s     %s (default: %s)\n" "${1}" "${2}" "${3}" "$(tolower ${4})"; }

    echo "Options:"
    print_option "help -h" "" "This message"

    echo ""
    print_option         "coverage -c" "" "Enable code coverage"
    print_default_option "name -n" "<NAME>" "Job name" ""
    print_default_option "site -s" "<NAME>" "Site name" "${SITE}"
    print_default_option "source-dir -S" "<N>" "Source directory" "${SOURCE_DIR}"
    print_default_option "binary-dir -B" "<N>" "Build directory" "${BINARY_DIR}"
    print_default_option "build-jobs -j" "<N>" "Number of build jobs" "${CMAKE_BUILD_PARALLEL_LEVEL}"
    print_default_option cmake-args "<ARGS...>" "CMake configuration args" "none"
    print_default_option ctest-args "<ARGS...>" "CTest command args" "none"
    print_default_option cdash-mode "<ARGS...>" "CDash mode" "${DASHBOARD_MODE}"
    print_default_option cdash-stages "<ARGS...>" "CDash stages" "${DASHBOARD_STAGES}"
    print_default_option submit-url "<URL>" "CDash submission URL" "${SUBMIT_URL}"
    #print_default_option lto "[on|off]" "Enable LTO" "${LTO}"
}

send-error()
{
    usage
    echo -e "\nError: ${@}"
    exit 1
}

verbose-run()
{
    echo -e "\n### Executing \"${@}\"... ###\n"
    eval "${@}"
}

reset-last()
{
    last() { send-error "Unsupported argument :: \"${1}\""; }
}

reset-last

n=0
while [[ $# -gt 0 ]]
do
    case "${1}" in
        -h|--help)
            usage
            exit 0
            ;;
        -c|--coverage)
            CODECOV=1
            reset-last
            ;;
        -n|--name)
            shift
            NAME=$(echo ${1} | sed 's/g++/gcc/g' | sed 's/\/merge//1')
            reset-last
            ;;
        -s|--site)
            shift
            SITE=${1}
            reset-last
            ;;
        -S|--source-dir)
            shift
            SOURCE_DIR=${1}
            reset-last
            ;;
        -B|--binary-dir)
            shift
            BINARY_DIR=${1}
            reset-last
            ;;
        -j|--build-jobs)
            shift
            CMAKE_BUILD_PARALLEL_LEVEL=${1}
            reset-last
            ;;
        --cmake-args)
            if [ -n "${2}" ]; then
                shift
                CMAKE_ARGS=${1}
            fi
            last() { CMAKE_ARGS="${CMAKE_ARGS} ${1}"; }
            ;;
        --ctest-args)
            if [ -n "${2}" ]; then
                shift
                CTEST_ARGS=${1}
            fi
            last() { CTEST_ARGS="${CTEST_ARGS} \"${1}\""; }
            ;;
        --cdash-mode)
            shift
            DASHBOARD_MODE=${1}
            reset-last
            ;;
        --cdash-stages)
            shift
            DASHBOARD_STAGES=${1}
            last() { DASHBOARD_STAGES="${DASHBOARD_STAGES} ${1}"; }
            ;;
        --submit-url)
            shift
            SUBMIT_URL=${1}
            reset-last
            ;;
        --*)
            send-error "Unsupported argument at position $((${n} + 1)) :: \"${1}\""
            ;;
        *)
            last ${1}
            ;;
    esac
    n=$((${n} + 1))
    shift
done

if [ -z "${NAME}" ]; then send-error "--name option required"; fi

CDASH_ARGS=""
for i in ${DASHBOARD_STAGES}
do
    if [ -z "${CDASH_ARGS}" ]; then
        CDASH_ARGS="-D ${DASHBOARD_MODE}${i}"
    else
        CDASH_ARGS="${CDASH_ARGS} -D ${DASHBOARD_MODE}${i}"
    fi
done

export CMAKE_BUILD_PARALLEL_LEVEL

if [ "${CODECOV}" -gt 0 ]; then
    GCOV_CMD=$(which gcov)
    CMAKE_ARGS="${CMAKE_ARGS} -DROCPROFSYS_BUILD_CODECOV=ON -DROCPROFSYS_STRIP_LIBRARIES=OFF"
fi

GIT_CMD=$(which git)
CMAKE_CMD=$(which cmake)
CTEST_CMD=$(which ctest)
SOURCE_DIR=$(realpath ${SOURCE_DIR})
BINARY_DIR=$(realpath ${BINARY_DIR})

verbose-run mkdir -p ${BINARY_DIR}

cat << EOF > ${BINARY_DIR}/CTestCustom.cmake

set(CTEST_PROJECT_NAME "rocprofiler-systems")
set(CTEST_NIGHTLY_START_TIME "05:00:00 UTC")

set(CTEST_DROP_METHOD "http")
set(CTEST_DROP_SITE_CDASH TRUE)
set(CTEST_SUBMIT_URL "https://${SUBMIT_URL}")

set(CTEST_UPDATE_TYPE git)
set(CTEST_UPDATE_VERSION_ONLY TRUE)
set(CTEST_GIT_INIT_SUBMODULES TRUE)

set(CTEST_OUTPUT_ON_FAILURE TRUE)
set(CTEST_USE_LAUNCHERS TRUE)
set(CMAKE_CTEST_ARGUMENTS --output-on-failure ${CTEST_ARGS})

set(CTEST_CUSTOM_MAXIMUM_NUMBER_OF_ERRORS "100")
set(CTEST_CUSTOM_MAXIMUM_NUMBER_OF_WARNINGS "100")
set(CTEST_CUSTOM_MAXIMUM_PASSED_TEST_OUTPUT_SIZE "51200")
set(CTEST_CUSTOM_COVERAGE_EXCLUDE "/usr/.*;.*external/.*;.*examples/.*")

set(CTEST_SITE "${SITE}")
set(CTEST_BUILD_NAME "${NAME}")

set(CTEST_SOURCE_DIRECTORY ${SOURCE_DIR})
set(CTEST_BINARY_DIRECTORY ${BINARY_DIR})

set(CTEST_UPDATE_COMMAND ${GIT_CMD})
set(CTEST_CONFIGURE_COMMAND "${CMAKE_CMD} -B ${BINARY_DIR} ${SOURCE_DIR} -DROCPROFSYS_BUILD_CI=ON ${CMAKE_ARGS}")
set(CTEST_BUILD_COMMAND "${CMAKE_CMD} --build ${BINARY_DIR} --target all --parallel ${CMAKE_BUILD_PARALLEL_LEVEL}")
set(CTEST_COVERAGE_COMMAND ${GCOV_CMD})
EOF

verbose-run cd ${BINARY_DIR}

cat << EOF > dashboard.cmake
cmake_minimum_required(VERSION 3.18.4 FATAL_ERROR)

include("\${CMAKE_CURRENT_LIST_DIR}/CTestCustom.cmake")

set(_STAGES ${DASHBOARD_STAGES})

macro(handle_submit)
    if("Submit" IN_LIST _STAGES)
        ctest_submit(
            ${ARGN}
            CAPTURE_CMAKE_ERROR _submit_err)
        if(NOT \${_submit_err} EQUAL 0)
            message(WARNING "Submission failed: ctest_submit(\${ARGN})")
        endif()
    endif()
endmacro()

macro(handle_error _message _ret)
    if(NOT \${\${_ret}} EQUAL 0)
        handle_submit(PARTS Done RETURN_VALUE _submit_ret)
        message(FATAL_ERROR "\${_message} failed: \${\${_ret}}")
    endif()
endmacro()

ctest_start(${DASHBOARD_MODE})
ctest_update(SOURCE "${SOURCE_DIR}")
ctest_configure(BUILD "${BINARY_DIR}" RETURN_VALUE _configure_ret)

handle_submit(PARTS Start Update Configure RETURN_VALUE _submit_ret)
handle_error("Configure" _configure_ret)

ctest_build(BUILD "${BINARY_DIR}" RETURN_VALUE _build_ret)
handle_submit(PARTS Build RETURN_VALUE _submit_ret)
handle_error("Build" _build_ret)

ctest_test(BUILD "${BINARY_DIR}" RETURN_VALUE _test_ret)
handle_submit(PARTS Test RETURN_VALUE _submit_ret)

if("${CODECOV}" GREATER 0)
    ctest_coverage(
        BUILD "${BINARY_DIR}"
        RETURN_VALUE _coverage_ret
        CAPTURE_CMAKE_ERROR _coverage_err)
    handle_submit(PARTS Coverage RETURN_VALUE _submit_ret)
endif()

handle_error("Testing" _test_ret)

handle_submit(PARTS Done RETURN_VALUE _submit_ret)
EOF

verbose-run cat CTestCustom.cmake
verbose-run cat dashboard.cmake
verbose-run ctest ${CDASH_ARGS} -S dashboard.cmake --output-on-failure -V ${CTEST_ARGS}
