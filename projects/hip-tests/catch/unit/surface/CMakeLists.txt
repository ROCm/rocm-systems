# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
#
# SPDX-License-Identifier: MIT

# Common Tests - Test independent of all platforms
set(TEST_SRC
    hipCreateSurfaceObject.cc
    hipDestroySurfaceObject.cc
    surf1D.cc
    surf1DLayered.cc
    surf2D.cc
    surf2DLayered.cc
    surf3D.cc
    surfCubemap.cc
)

if(HIP_PLATFORM MATCHES "nvidia") # Disabled on AMD due to defect EXSWHTEC-377
  set(TEST_SRC
    ${TEST_SRC}
    surfCubemapLayered.cc)
endif()

hip_add_exe_to_target(NAME SurfaceTest
                      TEST_SRC ${TEST_SRC}
                      TEST_TARGET_NAME build_tests)
