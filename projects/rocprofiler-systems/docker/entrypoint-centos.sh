#!/bin/bash

source scl_source enable devtoolset-9
source /etc/profile.d/modules.sh
module load mpi

export LC_ALL=en_US.UTF-8

if [ -z "${1}" ]; then
    exec bash
else
    set -e
    eval $@
fi
