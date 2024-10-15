#!/bin/bash

cleanup()
{
    kill -s 9 ${_PID}
}

trap cleanup SIGABRT SIGQUIT

ROCPROFSYS_COMMAND=""
APPLICATION_COMMAND=""

while [[ $# -gt 0 ]]
do
    if [ "${1}" == "--" ]; then
        shift
        break
    else
        ROCPROFSYS_COMMAND="${ROCPROFSYS_COMMAND}${1} "
        shift
    fi
done

${@} &
_PID=$!

echo ""
ps ax | grep "${1}"
echo ""

if [ "${_PID}" = "" ]; then
    echo "Error! No PID for \"${@}\""
    exit -1
fi

echo "PID: ${_PID}"
echo ""

${ROCPROFSYS_COMMAND} -p ${_PID} -- ${@}
RET=$?

killall $(basename ${1}) &> /dev/null

echo "Exiting with code: ${RET}"
exit ${RET}
