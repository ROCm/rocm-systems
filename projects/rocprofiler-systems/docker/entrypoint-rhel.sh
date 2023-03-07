#!/bin/bash

source /etc/profile.d/modules.sh
module load mpi

if [ -z "${1}" ]; then
    exec bash
else
    set -e
    eval $@
fi
