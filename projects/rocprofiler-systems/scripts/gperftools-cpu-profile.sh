#!/bin/bash

EXE=$(basename ${1})
DIR=gperftools-output
mkdir -p ${DIR}

# gperftools settings
: ${N:=0}
: ${GPERFTOOLS_PROFILE:=""}
: ${GPERFTOOLS_PROFILE_BASE:=${DIR}/prof.${EXE}}
: ${MALLOCSTATS:=1}
: ${CPUPROFILE_FREQUENCY:=250}
: ${CPUPROFILE_REALTIME:=1}
: ${PPROF:=$(which google-pprof)}
: ${PPROF:=$(which pprof)}

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
    echo "### ${@} ###" 1>&2
    eval ${@}
}

while [ -z "${GPERFTOOLS_PROFILE}" ]
do
    TEST_FILE=${GPERFTOOLS_PROFILE_BASE}.${N}
    if [ ! -f "${TEST_FILE}" ]; then
        GPERFTOOLS_PROFILE=${TEST_FILE}
    fi
    N=$((${N}+1))
done

export MALLOCSTATS
export CPUPROFILE_FREQUENCY
export CPUPROFILE_REALTIME

echo -e "\n\t--> Outputting profile to '${GPERFTOOLS_PROFILE}'...\n"

# remove profile file if unsucessful execution
cleanup-failure() { set +v ; echo "failure"; rm -f ${GPERFTOOLS_PROFILE}; exit 1; }
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

run-pprof()
{
    if [ -n "${PPROF}" ]; then
        run-verbose ${PPROF} ${ADD_LIB_LIST} ${PPROF_ARGS} "${@}"
    else
        echo -e "neither google-pprof nor pprof were found!"
        exit 1
    fi
}

# configure pre-loading of profiler library
for i in $(find ${PWD} -type f | egrep 'librocprof-sys' | egrep -v '\.a$' | egrep '\.so$') $(ldd ${1} | awk '{print $(NF-1)}')
do
    if [ -f "${i}" ]; then run-verbose ADD_LIBS "${i}"; fi
done
run-verbose ADD_PRELOAD $(ldd ${1} | egrep 'profiler' | awk '{print $(NF-1)}') /usr/lib/$(uname -m)-linux-gnu/libprofiler.so
LIBS=$(echo ${LIBS} | sed 's/^://g')

set -e
# run the application
LD_PRELOAD=${LIBS} CPUPROFILE_FREQUENCY=${CPUPROFILE_FREQUENCY} CPUPROFILE=${GPERFTOOLS_PROFILE} ${@} | tee ${GPERFTOOLS_PROFILE}.log
set +e

# generate the results
EXT=so
if [ -f "${GPERFTOOLS_PROFILE}" ]; then
    run-pprof --text ${1} ${GPERFTOOLS_PROFILE} | c++filt -n -t 1> ${GPERFTOOLS_PROFILE}.txt
    run-pprof ${PPROF} --text --cum ${1} ${GPERFTOOLS_PROFILE} | c++filt -n -t 1> ${GPERFTOOLS_PROFILE}.cum.txt
    # if dot is available
    if [ -n "$(which dot)" ]; then
        run-pprof ${PPROF} --dot ${1} ${GPERFTOOLS_PROFILE} 1> ${GPERFTOOLS_PROFILE}.dot
        run-verbose $(which dot) ${DOT_ARGS} -T${IMG_FORMAT} ${GPERFTOOLS_PROFILE}.dot -o ${GPERFTOOLS_PROFILE}.${IMG_FORMAT}
    fi
    if [ "${INTERACTIVE}" -gt 0 ]; then
        run-pprof ${PPROF} ${1} ${GPERFTOOLS_PROFILE}
    fi
else
    echo -e "profile file \"${GPERFTOOLS_PROFILE}\" not found!"
    ls -la
    exit 1
fi
