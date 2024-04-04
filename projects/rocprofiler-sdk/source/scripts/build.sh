#!/bin/bash

sudo apt-get update
sudo apt-get install -y cmake clang-tidy g++-11 g++-12 python3-pip
sudo update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-11 10 --slave /usr/bin/g++ g++ /usr/bin/g++-11
sudo update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-12 20 --slave /usr/bin/g++ g++ /usr/bin/g++-12
#python3 -m pip install -r requirements.txt
python3 -m pip install pytest pandas pyyaml
python3 -m pip install 'cmake>=3.22.0'

ROCPROFILER_SDK_PATH="$(pwd)/$(dirname ${BASH_SOURCE[0]})/../.."

cd ${ROCPROFILER_SDK_PATH}

echo -e "Redirecting to location: $ROCPROFILER_SDK_PATH"

cmake -B build -DROCPROFILER_BUILD_CI=ON -DROCPROFILER_BUILD_TESTS=ON -DROCPROFILER_BUILD_SAMPLES=ON -DROCPROFILER_ENABLE_CLANG_TIDY=ON $*
cmake --build build --target all --parallel $(nproc)
#cd --
