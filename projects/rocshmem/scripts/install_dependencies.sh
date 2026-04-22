###############################################################################
# Copyright (c) Advanced Micro Devices, Inc. All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to
# deal in the Software without restriction, including without limitation the
# rights to use, copy, modify, merge, publish, distribute, sublicense, and/or
# sell copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
# FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
# IN THE SOFTWARE.
###############################################################################

#!/bin/bash

set -e
set -o pipefail

if [[ -z "${_ROCM_DIR}" ]]; then
  export _ROCM_DIR=/opt/rocm
fi

# Location of dependencies source code
_BUILD_DIR=${BUILD_DIR:-$PWD}
export _INSTALL_DIR=${INSTALL_DIR:-$_BUILD_DIR/install}
echo "rocSHMEM dependencies UCX and Open MPI will install in $_INSTALL_DIR"

export _DEPS_SRC_DIR=$_BUILD_DIR/deps-src
mkdir -p $_DEPS_SRC_DIR

#Adjust branches and installation location as necessary
export _UCX_INSTALL_DIR=${INSTALL_DIR:-$_INSTALL_DIR/ucx}
export _UCX_REPO=https://github.com/ROCm/ucx.git
export _UCX_COMMIT_HASH=18770fdc1c3b5de202d14a088a14b734d2c4bbf3

export _OMPI_INSTALL_DIR=${INSTALL_DIR:-$_INSTALL_DIR/ompi}
export _OMPI_REPO=https://github.com/ROCm/ompi.git
export _OMPI_COMMIT_HASH=697a596dde68815fe50db3c2a75a42ddb41b5ef4

# Step 1: Build UCX with ROCm support
cd $_DEPS_SRC_DIR
rm -rf ucx
git clone $_UCX_REPO
cd ucx
git checkout $_UCX_COMMIT_HASH
./autogen.sh
./contrib/configure-release --prefix=$_UCX_INSTALL_DIR \
                            --with-rocm=$_ROCM_DIR     \
                            --enable-mt                \
                            --without-go               \
                            --without-java             \
                            --without-cuda             \
                            --without-knem
make -j
make install

# Step 2: Install OpenMPI with UCX support
cd $_DEPS_SRC_DIR
rm -rf ompi
git clone --recursive $_OMPI_REPO
cd ompi
git checkout $_OMPI_COMMIT_HASH
git submodule update --init --recursive
python3 -m ensurepip && python3 -m venv venv || python3 -m venv --system-site-packages --without-pip venv
. venv/bin/activate
python3 -m pip install -r docs/requirements.txt
./autogen.pl
./configure --prefix=$_OMPI_INSTALL_DIR  \
            --with-rocm=$_ROCM_DIR       \
            --with-ucx=$_UCX_INSTALL_DIR \
            --disable-oshmem             \
            --with-prrte=internal        \
            --with-hwloc=internal        \
            --with-libevent=internal     \
            --without-cuda               \
            --disable-mpi-fortran        \
            --without-ofi
make -j
make install

rm -rf $_DEPS_SRC_DIR

echo "Dependencies for rocSHMEM are now installed"
echo ""
echo "UCX ($_UCX_COMMIT_HASH) Installed to $_UCX_INSTALL_DIR"
echo "OpenMPI ($_OMPI_COMMIT_HASH) Installed to $_OMPI_INSTALL_DIR"
echo ""
echo "Please update your PATH and LD_LIBRARY_PATH"
