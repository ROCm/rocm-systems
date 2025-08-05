#!/bin/bash -l

if [ -f /etc/profile.d/modules.sh ]; then
    source /etc/profile.d/modules.sh
    module load mpi &> /dev/null
fi

if [ -z "${1}" ]; then
    : ${SHELL:=/bin/bash}
    exec ${SHELL}
else
    set -e
    eval $@
fi
