#!/bin/bash

EXE=$(basename ${1})
DIR=cpu.prof.${EXE}
mkdir -p ${DIR}

# gperf settings
: ${N:=0}
: ${GPERF_PROFILE:=""}
: ${GPERF_PROFILE_BASE:=${DIR}/gperf}
: ${MALLOCSTATS:=1}
: ${CPUPROFILE_FREQUENCY:=250}
: ${CPUPROFILE_REALTIME:=1}

# rendering settings
: ${INTERACTIVE:=0}
: ${IMG_FORMAT:="png"}
#: ${DOT_ARGS:='-Gsize=24,24\! -Gdpi=200'}
: ${DOT_ARGS:=""}
: ${PPROF_ARGS:="--no_strip_temp --functions"}

if [ "$(uname)" = "Darwin" ]; then
    if [ "${IMG_FORMAT}" = "jpeg" ]; then
        IMG_FORMAT="jpg"
    fi
fi
run-verbose()
{
    echo "${@}" 1>&2
    eval ${@}
}

while [ -z "${GPERF_PROFILE}" ]
do
    TEST_FILE=${GPERF_PROFILE_BASE}.${N}
    if [ ! -f "${TEST_FILE}" ]; then
        GPERF_PROFILE=${TEST_FILE}
    fi
    N=$((${N}+1))
done

export MALLOCSTATS
export CPUPROFILE_FREQUENCY
export CPUPROFILE_REALTIME

echo -e "\n\t--> Outputting profile to '${GPERF_PROFILE}'...\n"

# remove profile file if unsucessful execution
cleanup-failure() { set +v ; echo "failure"; rm -f ${GPERF_PROFILE}; exit 1; }
trap cleanup-failure SIGHUP SIGINT SIGQUIT SIGILL SIGABRT SIGKILL

ADD_LIBS()
{
    for i in $@
    do
        if [ -z "${ADD_LIB_LIST}" ]; then
            ADD_LIB_LIST="--add_lib=${i}"
        else
            ADD_LIB_LIST="${ADD_LIB_LIST} --add_lib=${i}"
        fi
    done
}

ADD_PRELOAD()
{
    for i in $@
    do
        if [ -z "${LIBS}" ]; then
            LIBS=${i}
        else
            LIBS="${LIBS}:${i}"
        fi
    done
}

# configure pre-loading of profiler library
PROJECT_LIBRARIES="$(find ${PWD} -type f | egrep 'libtimemory|libctimemory' | egrep -v '\.a$|\.dSYM' | egrep '\.so$|\.dylib$')"
run-verbose ADD_LIBS ${PROJECT_LIBRARIES}
if [ "$(uname)" = "Darwin" ]; then
    run-verbose ADD_PRELOAD $(otool -L ${1} | egrep 'profiler' | awk '{print $1}')
    LIBS=$(echo ${LIBS} | sed 's/^://g')
    if [ -n "${LIBS}" ]; then
        export DYLD_FORCE_FLAT_NAMESPACE=1
        echo "DYLD_INSERT_LIBRARIES=${LIBS}"
    fi
else
    run-verbose ADD_PRELOAD $(ldd ${1} | egrep 'profiler' | awk '{print $(NF-1)}') /usr/lib/x86_64-linux-gnu/libprofiler.so
    LIBS=$(echo ${LIBS} | sed 's/^://g')
    if [ -n "${LIB}" ]; then
        echo "LD_PRELOAD=${LIBS}"
    fi
fi

set -e
# run the application
if [ "$(uname)" = "Darwin" ]; then
    eval DYLD_INSERT_LIBRARIES=${LIBS} CPUPROFILE_FREQUENCY=${CPUPROFILE_FREQUENCY} CPUPROFILE=${GPERF_PROFILE} $@ | tee ${GPERF_PROFILE}.log
else
    eval LD_PRELOAD=${LIBS} CPUPROFILE_FREQUENCY=${CPUPROFILE_FREQUENCY} CPUPROFILE=${GPERF_PROFILE} $@ | tee ${GPERF_PROFILE}.log
fi
set +e

echo-dart-measurement()
{
    local _NAME=${1}
    local _TYPE=${2}
    local _PATH=${3}
    echo "<DartMeasurementFile name=\"${_NAME}\" type=\"image/${_TYPE}\">${_PATH}</DartMeasurementFile>"
}

# generate the results
EXT=so
if [ "$(uname)" = "Darwin" ]; then EXT=dylib; fi
if [ -f "${GPERF_PROFILE}" ]; then
    : ${PPROF:=$(which google-pprof)}
    : ${PPROF:=$(which pprof)}
    if [ -n "${PPROF}" ]; then
        run-verbose ${PPROF} --text ${ADD_LIB_LIST} ${PPROF_ARGS} ${1} ${GPERF_PROFILE} 1> ${GPERF_PROFILE}.txt.tmp
        run-verbose cat ${GPERF_PROFILE}.txt.tmp | c++filt -n -t &> ${GPERF_PROFILE}.txt
        run-verbose ${PPROF} --text --cum ${ADD_LIB_LIST} ${PPROF_ARGS} ${1} ${GPERF_PROFILE} 1> ${GPERF_PROFILE}.cum.txt.tmp
        run-verbose cat ${GPERF_PROFILE}.cum.txt.tmp | c++filt -n -t &> ${GPERF_PROFILE}.cum.txt
        rm -f *.txt.tmp
        # if dot is available
        if [ -n "$(which dot)" ]; then
            run-verbose ${PPROF} --dot ${ADD_LIB_LIST} ${PPROF_ARGS} ${1} ${GPERF_PROFILE} 1> ${GPERF_PROFILE}.dot
            run-verbose dot ${DOT_ARGS} -T${IMG_FORMAT} ${GPERF_PROFILE}.dot -o ${GPERF_PROFILE}.${IMG_FORMAT}
            echo-dart-measurement ${GPERF_PROFILE}.${IMG_FORMAT} ${IMG_FORMAT} ${PWD}/${GPERF_PROFILE}.${IMG_FORMAT}
        fi
        if [ "${INTERACTIVE}" -gt 0 ]; then
            run-verbose ${PPROF} ${ADD_LIB_LIST} ${PPROF_ARGS} ${1} ${GPERF_PROFILE}
        fi
    else
        echo -e "google-pprof/pprof not found!"
    fi
else
    echo -e "profile file \"${GPERF_PROFILE}\" not found!"
    ls -la
fi
