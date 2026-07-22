# Copyright (c) 2025 Advanced Micro Devices, Inc. All Rights Reserved.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.

# Accepted archs
set(ACCEPTED_GFX1250_ARCH gfx1250)

function(CheckAcceptedArchs OFFLOAD_ARCH_STR_LOCAL)
  set(ARCH_GFX1250 -1 PARENT_SCOPE)
  string(REGEX MATCHALL "--offload-arch=gfx[0-9a-z]+" OFFLOAD_ARCH_LIST ${OFFLOAD_ARCH_STR_LOCAL})
  foreach(OFFLOAD_ARCH IN LISTS OFFLOAD_ARCH_LIST)
    string(REGEX MATCHALL "--offload-arch=(gfx[0-9a-z]+)" matches ${OFFLOAD_ARCH})
    if (CMAKE_MATCH_COUNT EQUAL 1)
      # CMAKE_MATCH_1
    if (CMAKE_MATCH_1 IN_LIST ACCEPTED_GFX1250_ARCH)
        set(ARCH_GFX1250 1 PARENT_SCOPE)
      endif()
    endif()        # CMAKE_MATCH_COUNT
  endforeach()   # OFFLOAD_ARCH_LIST
endfunction()  # CheckAcceptedArchs

if (DEFINED OFFLOAD_ARCH_STR)
  CheckAcceptedArchs(${OFFLOAD_ARCH_STR})
elseif(DEFINED $ENV{HCC_AMDGPU_TARGET})
  CheckAcceptedArchs($ENV{HCC_AMDGPU_TARGET})
endif()

if(${ARCH_GFX1250} GREATER_EQUAL 0)
    set(TEST_SRC
        tdm_basic_load_store.cc
    )
    hip_add_exe_to_target(NAME tdm_basic_load_store
                        TEST_SRC ${TEST_SRC}
                        TEST_TARGET_NAME build_tests)
endif()