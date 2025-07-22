#!/bin/bash -e
# MIT License
#
# Copyright (c) 2023-2025 Advanced Micro Devices, Inc. All rights reserved.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

#
#   This script allows CMAKE_CXX_COMPILER to be a standard
#   C++ compiler and rocprofiler-sdk sets RULE_LAUNCH_COMPILE and
#   RULE_LAUNCH_LINK in CMake so that all compiler and link
#   commands are prefixed with this script followed by the
#   C++ compiler. Thus if $1 == $2 then we know the command
#   was intended for the C++ compiler and we discard both
#   $1 and $2 and redirect the command to linker.
#   If $1 != $2 then we know that the command was not intended
#   for the C++ compiler and we just discard $1 and launch
#   the original command. Examples of when $2 will not equal
#   $1 are 'ar', 'cmake', etc. during the linking phase
#

# emit a message about the underlying command executed
: ${DEBUG:=0}
: ${ROCPROFILER_SDK_LAUNCH_COMPILER_DEBUG:=${DEBUG}}

debug_message()
{
    if [ "${ROCPROFILER_SDK_LAUNCH_COMPILER_DEBUG}" -ne 0 ]; then
        echo -e "##### $(basename ${BASH_SOURCE[0]}) executing: \"$@\"... #####"
    fi
}

# if rocprofiler-sdk compiler is not passed, someone is probably trying to invoke it directly
if [ -z "${1}" ]; then
    echo -e "\n${BASH_SOURCE[0]} was invoked without the rocprofiler-sdk compiler as the first argument."
    echo "This script is not indended to be directly invoked by any mechanism other"
    echo -e "than through a RULE_LAUNCH_COMPILE or RULE_LAUNCH_LINK property set in CMake.\n"
    exit 1
fi

# if rocprofiler-sdk compiler is not passed, someone is probably trying to invoke it directly
if [ -z "${2}" ]; then
    echo -e "\n${BASH_SOURCE[0]} was invoked without the C++ compiler as the second argument."
    echo "This script is not indended to be directly invoked by any mechanism other"
    echo -e "than through a RULE_LAUNCH_COMPILE or RULE_LAUNCH_LINK property set in CMake.\n"
    exit 1
fi

# if there aren't two args, this isn't necessarily invalid, just a bit strange
if [ -z "${3}" ]; then exit 0; fi

# store the rocprofiler-sdk compiler
ROCPROFILER_SDK_COMPILER=${1}

# remove the rocprofiler-sdk compiler from the arguments
shift

# store the expected C++ compiler
CXX_COMPILER=${1}

# remove the expected C++ compiler from the arguments
shift

# discards the clang-tidy arguments
if [ "$(basename ${1})" = "cmake" ] && [ "${2}" = "-E" ] && [ "${3}" = "__run_co_compile" ]; then
    c=1
    n=1
    for i in "${@}"
    do
        if [ "${i}" = "--" ]; then
            break;
        fi
        if [ "${c}" -gt 3 ]; then
            n=$((${n} + 1))
        fi
        c=$((${c} + 1))
    done

    if [[ $# -gt ${c} ]]; then
        n=$((${n} + 3)) # add three because of the first 3 args
        for i in $(seq 1 1 ${n})
        do
            shift
        done
    fi
fi

if [[ "${CXX_COMPILER}" != "${1}" ]]; then
    debug_message $@
    # the command does not depend on rocprofiler-sdk so just execute the command w/o re-directing to ${ROCPROFILER_SDK_COMPILER}
    exec $@
else
    # the executable is the C++ compiler, so we need to re-direct to ${ROCPROFILER_SDK_COMPILER}
    if [ ! -f "${ROCPROFILER_SDK_COMPILER}" ]; then
        echo -e "\nError: the compiler redirect for rocprofiler-sdk was not found at ${ROCPROFILER_SDK_COMPILER}\n"
        exit 1
    fi

    # discard the compiler from the command
    shift

    if [ -n "$(basename ${ROCPROFILER_SDK_COMPILER} | egrep 'amdclang|amdllvm')" ]; then
        # this ensures the libomptarget-amdgpu-gfx*.bc files are found
        LLVM_LIB_DIR=$(cd $(dirname $(realpath ${ROCPROFILER_SDK_COMPILER}))/../lib && pwd)
        debug_message export LIBRARY_PATH=${LLVM_LIB_DIR}:${LIBRARY_PATH}
        export LIBRARY_PATH=${LLVM_LIB_DIR}:${LIBRARY_PATH}
    fi

    debug_message ${ROCPROFILER_SDK_COMPILER} $@
    # execute ${ROCPROFILER_SDK_COMPILER}
    exec ${ROCPROFILER_SDK_COMPILER} $@
fi
