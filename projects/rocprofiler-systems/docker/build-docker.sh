#!/usr/bin/env bash

: ${ROCM_VERSIONS:="4.5 4.3 4.3.1"}

for i in ${ROCM_VERSIONS}
do
    docker build . --tag jrmadsen/omnitrace-base-rocm-${i} --build-arg ROCM_REPO_VERSION=${i}
done
