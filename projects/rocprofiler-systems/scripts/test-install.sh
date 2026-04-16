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

: ${SLEEP_TIME:=0}

error-message()
{
    echo -e "\nError! ${@}\n"
    exit -1
}

verbose-run()
{
    echo -e "\n##### Executing \"${@}\"... #####\n"
    sleep ${SLEEP_TIME}
    eval $@
}

toupper()
{
    echo "$@" | awk -F '\\|~\\|' '{print toupper($1)}';
}

get-bool()
{
    echo "${1}" | egrep -i '^(y|on|yes|true|[1-9])$' &> /dev/null && echo 1 || echo 0
}

if [ -d "$(realpath /tmp)" ]; then
    : ${TMPDIR:=/tmp}
    export TMPDIR
fi

: ${CONFIG_DIR:=$(mktemp -t -d rocprof-sys-test-install-XXXX)}
: ${SOURCE_DIR:=$(dirname ${SCRIPT_DIR})}
: ${ENABLE_ROCPROFSYS_INSTRUMENT:=1}
: ${ENABLE_ROCPROFSYS_AVAIL:=1}
: ${ENABLE_ROCPROFSYS_SAMPLE:=1}
: ${ENABLE_ROCPROFSYS_PYTHON:=0}
: ${ENABLE_ROCPROFSYS_REWRITE:=1}
: ${ENABLE_ROCPROFSYS_RUNTIME:=1}

usage()
{
    print_option() { printf "    --%-10s %-30s     %s (default: %s)\n" "${1}" "${2}" "${3}" "${4}"; }
    echo "Options:"
    print_option source-dir "<PATH>" "Location of source directory" "${SOURCE_DIR}"
    print_option test-rocprof-sys-instrument "0|1" "Enable testing rocprof-sys-instrument" "${ENABLE_ROCPROFSYS_INSTRUMENT}"
    print_option test-rocprof-sys-avail "0|1" "Enable testing rocprof-sys-avail" "${ENABLE_ROCPROFSYS_AVAIL}"
    print_option test-rocprof-sys-sample "0|1" "Enable testing rocprof-sys-sample" "${ENABLE_ROCPROFSYS_SAMPLE}"
    print_option test-rocprof-sys-python "0|1" "Enable testing rocprof-sys-python" "${ENABLE_ROCPROFSYS_PYTHON}"
    print_option test-rocprof-sys-rewrite "0|1" "Enable testing rocprof-sys-instrument binary rewrite" "${ENABLE_ROCPROFSYS_REWRITE}"
    print_option test-rocprof-sys-runtime "0|1" "Enable testing rocprof-sys-instrument runtime instrumentation" "${ENABLE_ROCPROFSYS_RUNTIME}"
}

cat << EOF > ${CONFIG_DIR}/rocprof-sys.cfg
ROCPROFSYS_VERBOSE               = 2
ROCPROFSYS_PROFILE               = ON
ROCPROFSYS_TRACE                 = ON
ROCPROFSYS_USE_SAMPLING          = ON
ROCPROFSYS_USE_PROCESS_SAMPLING  = ON
ROCPROFSYS_OUTPUT_PATH           = %env{CONFIG_DIR}%/rocprof-sys-tests-output
ROCPROFSYS_OUTPUT_PREFIX         = %tag%/
ROCPROFSYS_SAMPLING_FREQ         = 100
ROCPROFSYS_SAMPLING_DELAY        = 0.05
ROCPROFSYS_COUT_OUTPUT           = ON
ROCPROFSYS_TIME_OUTPUT           = OFF
ROCPROFSYS_USE_PID               = OFF
EOF

export CONFIG_DIR
export ROCPROFSYS_CONFIG_FILE=${CONFIG_DIR}/rocprof-sys.cfg
verbose-run cat ${ROCPROFSYS_CONFIG_FILE}

while [[ $# -gt 0 ]]
do
    ARG=${1}
    shift

    VAL="$(echo ${ARG} | sed 's/=/ /1' | awk '{print $2}')"
    if [ -z "${VAL}" ]; then
        while [[ $# -gt 0 ]]
        do
            VAL=$(get-bool ${1})
            shift
            break
        done
    else
        VAL=$(get-bool ${VAL})
        ARG="$(echo ${ARG} | sed 's/=/ /1' | awk '{print $1}')"
    fi

    if [ -z "${VAL}" ]; then
        echo "Error! Missing value for argument \"${ARG}\""
        usage
        exit -1
    fi

    case "${ARG}" in
        --test-rocprof-sys-instrument)
            ENABLE_ROCPROFSYS_INSTRUMENT=${VAL}
            continue
            ;;
        --test-rocprof-sys-avail)
            ENABLE_ROCPROFSYS_AVAIL=${VAL}
            continue
            ;;
        --test-rocprof-sys-sample)
            ENABLE_ROCPROFSYS_SAMPLE=${VAL}
            continue
            ;;
        --test-rocprof-sys-python)
            ENABLE_ROCPROFSYS_PYTHON=${VAL}
            continue
            ;;
        --test-rocprof-sys-rewrite)
            ENABLE_ROCPROFSYS_REWRITE=${VAL}
            continue
            ;;
        --test-rocprof-sys-runtime)
            ENABLE_ROCPROFSYS_RUNTIME=${VAL}
            continue
            ;;
        --source-dir)
            SOURCE_DIR=${VAL}
            continue
            ;;
        *)
            echo -e "Error! Unknown option : ${ARG}"
            usage
            exit -1
            ;;
    esac
done

test-rocprof-sys-instrument()
{
    verbose-run which rocprof-sys-instrument
    verbose-run ldd $(which rocprof-sys-instrument)
    verbose-run rocprof-sys-instrument --help
}

test-rocprof-sys-avail()
{
    verbose-run which rocprof-sys-avail
    verbose-run ldd $(which rocprof-sys-avail)
    verbose-run rocprof-sys-avail --help
    verbose-run rocprof-sys-avail -a
}

test-rocprof-sys-sample()
{
    verbose-run which rocprof-sys-sample
    verbose-run ldd $(which rocprof-sys-sample)
    verbose-run rocprof-sys-sample --help
    verbose-run rocprof-sys-sample --cputime 100 --realtime 50 --hsa-interrupt 0 -TPH -- python3 ${SOURCE_DIR}/examples/python/external.py -n 5 -v 20
}

test-rocprof-sys-python()
{
    verbose-run which rocprof-sys-python
    verbose-run rocprof-sys-python --help
    verbose-run rocprof-sys-python -b -- ${SOURCE_DIR}/examples/python/builtin.py -n 5 -v 5
    verbose-run rocprof-sys-python -b -- ${SOURCE_DIR}/examples/python/noprofile.py -n 5 -v 5
    verbose-run rocprof-sys-python -- ${SOURCE_DIR}/examples/python/external.py -n 5 -v 5
    verbose-run python3 ${SOURCE_DIR}/examples/python/source.py -n 5 -v 5
}

test-rocprof-sys-rewrite()
{
    if [ -f /usr/bin/coreutils ]; then
        local LS_NAME=coreutils
        local LS_ARGS="--coreutils-prog=ls"
    else
        local LS_NAME=ls
        local LS_ARGS=""
    fi
    verbose-run rocprof-sys-instrument -e -v 1 -o ${CONFIG_DIR}/ls.inst --simulate -- ${LS_NAME}
    for i in $(find ${CONFIG_DIR}/rocprof-sys-tests-output/ls.inst -type f); do verbose-run ls ${i}; done
    verbose-run rocprof-sys-instrument -e -v 1 -o ${CONFIG_DIR}/ls.inst -- ${LS_NAME}
    verbose-run rocprof-sys-run -- ${CONFIG_DIR}/ls.inst ${LS_ARGS}
}

test-rocprof-sys-runtime()
{
    if [ -f /usr/bin/coreutils ]; then
        local LS_NAME=coreutils
        local LS_ARGS="--coreutils-prog=ls"
    else
        local LS_NAME=ls
        local LS_ARGS=""
    fi
    verbose-run rocprof-sys-instrument -e -v 1 --simulate -- ${LS_NAME} ${LS_ARGS}
    for i in $(find ${CONFIG_DIR}/rocprof-sys-tests-output/$(basename ${LS_NAME}) -type f); do verbose-run ls ${i}; done
    verbose-run rocprof-sys-instrument -e -v 1 -- ${LS_NAME} ${LS_ARGS}
}

if [ "${ENABLE_ROCPROFSYS_INSTRUMENT}" -ne 0 ]; then verbose-run test-rocprof-sys-instrument; fi
if [ "${ENABLE_ROCPROFSYS_AVAIL}" -ne 0 ]; then verbose-run test-rocprof-sys-avail; fi
if [ "${ENABLE_ROCPROFSYS_SAMPLE}" -ne 0 ]; then verbose-run test-rocprof-sys-sample; fi
if [ "${ENABLE_ROCPROFSYS_PYTHON}" -ne 0 ]; then verbose-run test-rocprof-sys-python; fi
if [ "${ENABLE_ROCPROFSYS_REWRITE}" -ne 0 ]; then verbose-run test-rocprof-sys-rewrite; fi
if [ "${ENABLE_ROCPROFSYS_RUNTIME}" -ne 0 ]; then verbose-run test-rocprof-sys-runtime; fi
