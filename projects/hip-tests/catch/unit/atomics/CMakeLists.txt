# Copyright (c) 2023 Advanced Micro Devices, Inc. All Rights Reserved.
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

if(HIP_PLATFORM MATCHES "amd")
    set(TEST_SRC
      atomicAnd.cc
      atomicAnd_system.cc
      atomicOr.cc
      atomicOr_system.cc
      atomicXor.cc
      atomicXor_system.cc
      atomicMin.cc
      atomicMin_system.cc
      atomicMax.cc
      atomicMax_system.cc
      safeAtomicMin.cc
      unsafeAtomicMin.cc
      safeAtomicMax.cc
      unsafeAtomicMax.cc
      __hip_atomic_fetch_min.cc
      __hip_atomic_fetch_max.cc
      atomic_builtins.cc
      acquire_release.cc
      sequential_consistency.cc
      atomicAdd.cc
      atomicAdd_system.cc
      unsafeAtomicAdd.cc
      safeAtomicAdd.cc
      atomicSub.cc
      atomicSub_system.cc
      atomicCAS.cc
      atomicCAS_system.cc
      __hip_atomic_fetch_add.cc
      __hip_atomic_compare_exchange_strong.cc
      atomicExch.cc
      atomicExch_system.cc
      __hip_atomic_fetch_and.cc
      __hip_atomic_fetch_or.cc
      __hip_atomic_fetch_xor.cc
      __hip_atomic_exchange.cc
      atomicInc.cc
      atomicDec.cc
    )

  hip_add_exe_to_target(NAME AtomicsTest
                        TEST_SRC ${TEST_SRC}
                        TEST_TARGET_NAME build_tests
                        LINKER_LIBS hiprtc)

if(UNIX)
  set(EXPECTED_ERRORS 40)

  file(GLOB NEGATIVE_TEST_SRC
        "atomicAnd_negative_kernels.cc"
        "atomicOr_negative_kernels.cc"
        "atomicXor_negative_kernels.cc"
        "atomicMin_negative_kernels.cc"
        "atomicMax_negative_kernels.cc"
        "atomic_builtin_kernels.cc"
        "atomicAdd_negative_kernels.cc"
        "atomicSub_negative_kernels.cc"
        "atomicInc_negative_kernels.cc"
        "atomicDec_negative_kernels.cc"
        "atomicCAS_negative_kernels.cc"
        "atomicExch_negative_kernels.cc"
        "atomicExch_system_negative_kernels.cc")

  file(COPY ${NEGATIVE_TEST_SRC} DESTINATION ${CMAKE_CURRENT_BINARY_DIR}/src)
  set_property(GLOBAL APPEND PROPERTY G_INSTALL_SRC_FILES ${NEGATIVE_TEST_SRC})
 # add_test(NAME Unit_atomicAnd_Negative_Parameters
 #         COMMAND ${Python3_EXECUTABLE} ../compileAndCaptureOutput.py
 #         ./src ${HIP_PLATFORM} ${HIP_PATH}
 #         atomicAnd_negative_kernels.cc ${EXPECTED_ERRORS})

 # add_test(NAME Unit_atomicOr_Negative_Parameters
 #         COMMAND ${Python3_EXECUTABLE} ../compileAndCaptureOutput.py
 #         ./src ${HIP_PLATFORM} ${HIP_PATH}
 #         atomicOr_negative_kernels.cc ${EXPECTED_ERRORS})

 # add_test(NAME Unit_atomicXor_Negative_Parameters
 #         COMMAND ${Python3_EXECUTABLE} ../compileAndCaptureOutput.py
 #         ./src ${HIP_PLATFORM} ${HIP_PATH}
 #         atomicXor_negative_kernels.cc ${EXPECTED_ERRORS})

 # add_test(NAME Unit_atomicMin_Negative_Parameters
 #         COMMAND ${Python3_EXECUTABLE} ../compileAndCaptureOutput.py
 #         ./src ${HIP_PLATFORM} ${HIP_PATH}
 #         atomicMin_negative_kernels.cc 42)

 # add_test(NAME Unit_atomicMax_Negative_Parameters
 #         COMMAND ${Python3_EXECUTABLE} ../compileAndCaptureOutput.py
 #         ./src ${HIP_PLATFORM} ${HIP_PATH}
 #         atomicMax_negative_kernels.cc 42)

	#add_test(NAME Unit_AtomicBuiltins_Negative_Parameters
 #         COMMAND ${Python3_EXECUTABLE} ../compileAndCaptureOutput.py
 #         ./src ${HIP_PLATFORM} ${HIP_PATH}
 #         atomic_builtin_kernels.cc 60 27)

 #         add_test(NAME Unit_atomicAdd_Negative_Parameters
 #         COMMAND ${Python3_EXECUTABLE} ../compileAndCaptureOutput.py
 #         ./src ${HIP_PLATFORM} ${HIP_PATH}
 #         atomicAdd_negative_kernels.cc 48)
 # add_test(NAME Unit_atomicSub_Negative_Parameters
 #         COMMAND ${Python3_EXECUTABLE} ../compileAndCaptureOutput.py
 #         ./src ${HIP_PLATFORM} ${HIP_PATH}
 #         atomicSub_negative_kernels.cc 48)
 # add_test(NAME Unit_atomicInc_Negative_Parameters
 #         COMMAND ${Python3_EXECUTABLE} ../compileAndCaptureOutput.py
 #         ./src ${HIP_PLATFORM} ${HIP_PATH}
 #         atomicInc_negative_kernels.cc 8)

 # add_test(NAME Unit_atomicDec_Negative_Parameters
 #         COMMAND ${Python3_EXECUTABLE} ../compileAndCaptureOutput.py
 #         ./src ${HIP_PLATFORM} ${HIP_PATH}
 #         atomicDec_negative_kernels.cc 8)

 # add_test(NAME Unit_atomicCAS_Negative_Parameters
 #         COMMAND ${Python3_EXECUTABLE} ../compileAndCaptureOutput.py
 #         ./src ${HIP_PLATFORM} ${HIP_PATH}
 #         atomicCAS_negative_kernels.cc 48)

 # add_test(NAME Unit_atomicExch_Negative_Parameters
 #          COMMAND ${Python3_EXECUTABLE} ../compileAndCaptureOutput.py
 #          ./src ${HIP_PLATFORM} ${HIP_PATH}
 #          atomicExch_negative_kernels.cc 40)

 # add_test(NAME Unit_atomicExch_system_Negative_Parameters
 #          COMMAND ${Python3_EXECUTABLE} ../compileAndCaptureOutput.py
 #          ./src ${HIP_PLATFORM} ${HIP_PATH}
 #          atomicExch_system_negative_kernels.cc 40)
endif()
endif()