#!/bin/bash -xe
  set -xe
  cmake   -DCMAKE_BUILD_TYPE=Debug   -DCMAKE_INSTALL_PREFIX=/opt/rocm-6.5.0   -DBUILD_TESTS=ON   ..
  #-DBUILD_WRAPPER=ON   #
  
