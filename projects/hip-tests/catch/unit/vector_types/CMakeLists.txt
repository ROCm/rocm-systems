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

# Common Tests - Test independent of all platforms

set(TEST_SRC
    vector_types.cc
    dim3.cc
)

# the last argument linker libraries is required for this test but optional to the function
if(HIP_PLATFORM MATCHES "nvidia")
hip_add_exe_to_target(NAME VectorTypesTest
                      TEST_SRC ${TEST_SRC}
                      TEST_TARGET_NAME build_tests
                      LINKER_LIBS nvrtc)
elseif(HIP_PLATFORM MATCHES "amd")
hip_add_exe_to_target(NAME VectorTypesTest
                      TEST_SRC ${TEST_SRC}
                      TEST_TARGET_NAME build_tests
                      LINKER_LIBS hiprtc)
endif()

if(UNIX)
if(HIP_PLATFORM MATCHES "amd")
    file(GLOB NEGATIVE_TEST_SRC
         "negative_negate_unsigned.cc"
         "negative_bitwise_float.cc"
         "negative_bitwise_double.cc"
         "negative_calculate_assign_with_value_1D.cc"
         "negative_calculate_assign_with_value_2D.cc"
         "negative_calculate_assign_with_value_3D.cc"
         "negative_calculate_assign_with_value_4D.cc"
         "negative_calculate_assign_with_unsigned_value_1D.cc"
         "negative_calculate_assign_with_unsigned_value_2D.cc"
         "negative_calculate_assign_with_unsigned_value_3D.cc"
         "negative_calculate_assign_with_unsigned_value_4D.cc"
         "negative_macros_common.hh" )

    file(COPY ${NEGATIVE_TEST_SRC} DESTINATION ${CMAKE_CURRENT_BINARY_DIR}/src)
    set_property(GLOBAL APPEND PROPERTY G_INSTALL_SRC_FILES ${NEGATIVE_TEST_SRC})

    #add_test(NAME Unit_NegateUnsigned_Negative_Parameters
    #        COMMAND ${Python3_EXECUTABLE} ../compileAndCaptureOutput.py
    #        ./src ${HIP_PLATFORM} ${HIP_PATH}
    #        negative_negate_unsigned.cc 40)

    #add_test(NAME Unit_BitwiseFloat_Negative_Parameters
    #        COMMAND ${Python3_EXECUTABLE} ../compileAndCaptureOutput.py
    #        ./src ${HIP_PLATFORM} ${HIP_PATH}
    #        negative_bitwise_float.cc 96)

    #add_test(NAME Unit_BitwiseDouble_Negative_Parameters
    #        COMMAND ${Python3_EXECUTABLE} ../compileAndCaptureOutput.py
    #        ./src ${HIP_PLATFORM} ${HIP_PATH}
    #        negative_bitwise_double.cc 96)

    #add_test(NAME Unit_CalculateAssign1D_Negative_Parameters
    #        COMMAND ${Python3_EXECUTABLE} ../compileAndCaptureOutput.py
    #        ./src ${HIP_PLATFORM} ${HIP_PATH}
    #        negative_calculate_assign_with_value_1D.cc 60)

    #add_test(NAME Unit_CalculateAssign2D_Negative_Parameters
    #        COMMAND ${Python3_EXECUTABLE} ../compileAndCaptureOutput.py
    #        ./src ${HIP_PLATFORM} ${HIP_PATH}
    #        negative_calculate_assign_with_value_2D.cc 60)

    #add_test(NAME Unit_CalculateAssign3D_Negative_Parameters
    #        COMMAND ${Python3_EXECUTABLE} ../compileAndCaptureOutput.py
    #        ./src ${HIP_PLATFORM} ${HIP_PATH}
    #        negative_calculate_assign_with_value_3D.cc 60)

    #add_test(NAME Unit_CalculateAssign4D_Negative_Parameters
    #        COMMAND ${Python3_EXECUTABLE} ../compileAndCaptureOutput.py
    #        ./src ${HIP_PLATFORM} ${HIP_PATH}
    #        negative_calculate_assign_with_value_4D.cc 60)

    #add_test(NAME Unit_CalculateAssignUnsigned1D_Negative_Parameters
    #        COMMAND ${Python3_EXECUTABLE} ../compileAndCaptureOutput.py
    #        ./src ${HIP_PLATFORM} ${HIP_PATH}
    #        negative_calculate_assign_with_unsigned_value_1D.cc 60)

    #add_test(NAME Unit_CalculateAssignUnsigned2D_Negative_Parameters
    #        COMMAND ${Python3_EXECUTABLE} ../compileAndCaptureOutput.py
    #        ./src ${HIP_PLATFORM} ${HIP_PATH}
    #        negative_calculate_assign_with_unsigned_value_2D.cc 60)

    #add_test(NAME Unit_CalculateAssignUnsigned3D_Negative_Parameters
    #        COMMAND ${Python3_EXECUTABLE} ../compileAndCaptureOutput.py
    #        ./src ${HIP_PLATFORM} ${HIP_PATH}
    #        negative_calculate_assign_with_unsigned_value_3D.cc 60)

    #add_test(NAME Unit_CalculateAssignUnsigned4D_Negative_Parameters
    #        COMMAND ${Python3_EXECUTABLE} ../compileAndCaptureOutput.py
    #        ./src ${HIP_PLATFORM} ${HIP_PATH}
    #        negative_calculate_assign_with_unsigned_value_4D.cc 60)
endif()
endif()
