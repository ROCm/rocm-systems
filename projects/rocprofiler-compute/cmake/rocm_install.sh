#!/usr/bin/env bash

# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier:  MIT

declare -a rocm_versions=("4.3.1" "4.5.2" "5.0.2" "5.1.3" "5.2.3")
wget https://repo.radeon.com/amdgpu-install/22.10/ubuntu/focal/amdgpu-install_22.10.50100-1_all.deb
apt-get install -y ./amdgpu-install_22.10.50100-1_all.deb
for rocm_version in ${rocm_versions[@]}; do
    echo "deb [arch=amd64] https://repo.radeon.com/rocm/apt/$rocm_version ubuntu main" | tee /etc/apt/sources.list.d/rocm.list
    apt update
    amdgpu-install -y --usecase=rocm --rocmrelease=$rocm_version --no-dkms
done
