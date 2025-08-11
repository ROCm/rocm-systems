#!/bin/bash -e

apt-get update
apt-get install -y cmake clang-tidy-15 g++-11 g++-12 python3-pip libdw-dev libsqlite3-dev
update-alternatives --install /usr/bin/clang-tidy clang-tidy /usr/bin/clang-tidy-17 10
update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-11 10 --slave /usr/bin/g++ g++ /usr/bin/g++-11
update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-12 20 --slave /usr/bin/g++ g++ /usr/bin/g++-12
