# ##############################################################################
# Copyright (c) 2024 - 2026 Advanced Micro Devices, Inc.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
#
# ##############################################################################
cmake_minimum_required(VERSION 3.10)

if(NOT DEFINED ENHANCED_MESSAGE)
  string(ASCII 27 Esc)
  set(ColourReset "${Esc}[m")
  set(Red         "${Esc}[31m")
  set(Green       "${Esc}[32m")
  set(Yellow      "${Esc}[33m")
  set(White       "${Esc}[37m")
endif()

# ROCM Path
if(DEFINED ENV{ROCM_PATH})
  set(ROCM_PATH $ENV{ROCM_PATH} CACHE PATH "${White}${PROJECT_NAME}: Default ROCm installation path${ColourReset}")
elseif(ROCM_PATH)
  message("-- ${White}${PROJECT_NAME} :ROCM_PATH Set -- ${ROCM_PATH}${ColourReset}")
else()
  set(ROCM_PATH /opt/rocm CACHE PATH "${White}${PROJECT_NAME}: Default ROCm installation path${ColourReset}")
endif()
# Set AMD Clang as default compiler
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED On)
set(CMAKE_CXX_EXTENSIONS ON)
if(NOT DEFINED CMAKE_CXX_COMPILER)
  set(CMAKE_C_COMPILER ${ROCM_PATH}/lib/llvm/bin/amdclang)
  set(CMAKE_CXX_COMPILER ${ROCM_PATH}/lib/llvm/bin/amdclang++)
endif()

project(rocjpeg-test)

# make test with CTest
enable_testing()
include(CTest)

# add ROCm find Config location
list(APPEND CMAKE_PREFIX_PATH ${ROCM_PATH}/lib/cmake)

# find rocjpeg
if(BUILD_FROM_SOURCE)
  message("-- ${White}${PROJECT_NAME}: CTests requires rocjpeg installed before running CTests")
else()
  find_package(rocjpeg QUIET)

  if(rocjpeg_FOUND)
    message("-- ${White}${PROJECT_NAME}: rocjpeg found with find_package(rocjpeg QUIET)${ColourReset}")
    message("-- \t${White}rocjpeg_INCLUDE_DIR -- ${rocjpeg_INCLUDE_DIR}${ColourReset}")
    message("-- \t${White}rocjpeg_LIB_DIR -- ${rocjpeg_LIB_DIR}${ColourReset}")
    message("-- \t${White}rocjpeg_LIBRARY -- ${rocjpeg_LIBRARY}${ColourReset}")
    message("-- \t${White}rocjpeg_FOUND -- ${rocjpeg_FOUND}${ColourReset}")
    message("-- \t${White}rocjpeg_VERSION -- ${rocjpeg_VERSION}${ColourReset}")
    message("-- \t${White}rocjpeg_VERSION_MAJOR -- ${rocjpeg_VERSION_MAJOR}${ColourReset}")
    message("-- \t${White}rocjpeg_VERSION_MINOR -- ${rocjpeg_VERSION_MINOR}${ColourReset}")
    message("-- \t${White}rocjpeg_VERSION_PATCH -- ${rocjpeg_VERSION_PATCH}${ColourReset}")
  else()
    message("-- ${Yellow}${PROJECT_NAME} requires rocjpeg. Install rocjpeg before running CTests")
  endif(rocjpeg_FOUND)
endif(BUILD_FROM_SOURCE)

# Check if lib/rocm_sysdeps/lib exists in the ROCm path which indicates ROCm installation via TheRock
set(USING_THE_ROCK OFF)
if(EXISTS "${ROCM_PATH}/lib/rocm_sysdeps/lib")
  set(USING_THE_ROCK ON)
endif()

add_test(
  NAME
    jpeg-decode-fmt-native
  COMMAND
    "${CMAKE_CTEST_COMMAND}"
            --build-and-test "${ROCM_PATH}/share/rocjpeg/samples/jpegDecode"
                              "${CMAKE_CURRENT_BINARY_DIR}/jpegDecode"
            --build-generator "${CMAKE_GENERATOR}"
            --test-command "jpegdecode"
            -i ${ROCM_PATH}/share/rocjpeg/images/
)
if(USING_THE_ROCK)
  set_property(TEST jpeg-decode-fmt-native PROPERTY ENVIRONMENT "LIBVA_DRIVERS_PATH=${ROCM_PATH}/lib/rocm_sysdeps/lib")
endif()

add_test(
  NAME
    jpeg-decode-fmt-yuv-planar
  COMMAND
    "${CMAKE_CTEST_COMMAND}"
            --build-and-test "${ROCM_PATH}/share/rocjpeg/samples/jpegDecode"
                              "${CMAKE_CURRENT_BINARY_DIR}/jpegDecode"
            --build-generator "${CMAKE_GENERATOR}"
            --test-command "jpegdecode"
            -i ${ROCM_PATH}/share/rocjpeg/images/ -fmt yuv_planar
)
if(USING_THE_ROCK)
  set_property(TEST jpeg-decode-fmt-yuv-planar PROPERTY ENVIRONMENT "LIBVA_DRIVERS_PATH=${ROCM_PATH}/lib/rocm_sysdeps/lib")
endif()

add_test(
  NAME
    jpeg-decode-fmt-y
  COMMAND
    "${CMAKE_CTEST_COMMAND}"
            --build-and-test "${ROCM_PATH}/share/rocjpeg/samples/jpegDecode"
                              "${CMAKE_CURRENT_BINARY_DIR}/jpegDecode"
            --build-generator "${CMAKE_GENERATOR}"
            --test-command "jpegdecode"
            -i ${ROCM_PATH}/share/rocjpeg/images/ -fmt y
)
if(USING_THE_ROCK)
  set_property(TEST jpeg-decode-fmt-y PROPERTY ENVIRONMENT "LIBVA_DRIVERS_PATH=${ROCM_PATH}/lib/rocm_sysdeps/lib")
endif()

add_test(
  NAME
    jpeg-decode-fmt-rgb
  COMMAND
    "${CMAKE_CTEST_COMMAND}"
            --build-and-test "${ROCM_PATH}/share/rocjpeg/samples/jpegDecode"
                              "${CMAKE_CURRENT_BINARY_DIR}/jpegDecode"
            --build-generator "${CMAKE_GENERATOR}"
            --test-command "jpegdecode"
            -i ${ROCM_PATH}/share/rocjpeg/images/ -fmt rgb
)
if(USING_THE_ROCK)
  set_property(TEST jpeg-decode-fmt-rgb PROPERTY ENVIRONMENT "LIBVA_DRIVERS_PATH=${ROCM_PATH}/lib/rocm_sysdeps/lib")
endif()

add_test(
  NAME
    jpeg-decode-fmt-rgb-planar
  COMMAND
    "${CMAKE_CTEST_COMMAND}"
            --build-and-test "${ROCM_PATH}/share/rocjpeg/samples/jpegDecode"
                              "${CMAKE_CURRENT_BINARY_DIR}/jpegDecode"
            --build-generator "${CMAKE_GENERATOR}"
            --test-command "jpegdecode"
            -i ${ROCM_PATH}/share/rocjpeg/images/ -fmt rgb_planar
)
if(USING_THE_ROCK)
  set_property(TEST jpeg-decode-fmt-rgb-planar PROPERTY ENVIRONMENT "LIBVA_DRIVERS_PATH=${ROCM_PATH}/lib/rocm_sysdeps/lib")
endif()

add_test(
  NAME
    jpeg-decode-crop-fmt-native
  COMMAND
    "${CMAKE_CTEST_COMMAND}"
            --build-and-test "${ROCM_PATH}/share/rocjpeg/samples/jpegDecode"
                              "${CMAKE_CURRENT_BINARY_DIR}/jpegDecode"
            --build-generator "${CMAKE_GENERATOR}"
            --test-command "jpegdecode"
            -i ${ROCM_PATH}/share/rocjpeg/images/ -crop 960,540,2880,1620
)
if(USING_THE_ROCK)
  set_property(TEST jpeg-decode-crop-fmt-native PROPERTY ENVIRONMENT "LIBVA_DRIVERS_PATH=${ROCM_PATH}/lib/rocm_sysdeps/lib")
endif()

add_test(
  NAME
    jpeg-decode-crop-fmt-yuv-planar
  COMMAND
    "${CMAKE_CTEST_COMMAND}"
            --build-and-test "${ROCM_PATH}/share/rocjpeg/samples/jpegDecode"
                              "${CMAKE_CURRENT_BINARY_DIR}/jpegDecode"
            --build-generator "${CMAKE_GENERATOR}"
            --test-command "jpegdecode"
            -i ${ROCM_PATH}/share/rocjpeg/images/ -fmt yuv_planar -crop 960,540,2880,1620
)
if(USING_THE_ROCK)
  set_property(TEST jpeg-decode-crop-fmt-yuv-planar PROPERTY ENVIRONMENT "LIBVA_DRIVERS_PATH=${ROCM_PATH}/lib/rocm_sysdeps/lib")
endif()

add_test(
  NAME
    jpeg-decode-crop-fmt-y
  COMMAND
    "${CMAKE_CTEST_COMMAND}"
            --build-and-test "${ROCM_PATH}/share/rocjpeg/samples/jpegDecode"
                              "${CMAKE_CURRENT_BINARY_DIR}/jpegDecode"
            --build-generator "${CMAKE_GENERATOR}"
            --test-command "jpegdecode"
            -i ${ROCM_PATH}/share/rocjpeg/images/ -fmt y -crop 960,540,2880,1620
)
if(USING_THE_ROCK)
  set_property(TEST jpeg-decode-crop-fmt-y PROPERTY ENVIRONMENT "LIBVA_DRIVERS_PATH=${ROCM_PATH}/lib/rocm_sysdeps/lib")
endif()

add_test(
  NAME
    jpeg-decode-crop-fmt-rgb
  COMMAND
    "${CMAKE_CTEST_COMMAND}"
            --build-and-test "${ROCM_PATH}/share/rocjpeg/samples/jpegDecode"
                              "${CMAKE_CURRENT_BINARY_DIR}/jpegDecode"
            --build-generator "${CMAKE_GENERATOR}"
            --test-command "jpegdecode"
            -i ${ROCM_PATH}/share/rocjpeg/images/ -fmt rgb -crop 960,540,2880,1620
)
if(USING_THE_ROCK)
  set_property(TEST jpeg-decode-crop-fmt-rgb PROPERTY ENVIRONMENT "LIBVA_DRIVERS_PATH=${ROCM_PATH}/lib/rocm_sysdeps/lib")
endif()

add_test(
  NAME
    jpeg-decode-crop-fmt-rgb-planar
  COMMAND
    "${CMAKE_CTEST_COMMAND}"
            --build-and-test "${ROCM_PATH}/share/rocjpeg/samples/jpegDecode"
                              "${CMAKE_CURRENT_BINARY_DIR}/jpegDecode"
            --build-generator "${CMAKE_GENERATOR}"
            --test-command "jpegdecode"
            -i ${ROCM_PATH}/share/rocjpeg/images/ -fmt rgb_planar -crop 960,540,2880,1620
)
if(USING_THE_ROCK)
  set_property(TEST jpeg-decode-crop-fmt-rgb-planar PROPERTY ENVIRONMENT "LIBVA_DRIVERS_PATH=${ROCM_PATH}/lib/rocm_sysdeps/lib")
endif()

add_test(
  NAME
    jpeg-decode-batch-fmt-native
  COMMAND
    "${CMAKE_CTEST_COMMAND}"
            --build-and-test "${ROCM_PATH}/share/rocjpeg/samples/jpegDecodeBatched"
                              "${CMAKE_CURRENT_BINARY_DIR}/jpegDecodeBatched"
            --build-generator "${CMAKE_GENERATOR}"
            --test-command "jpegdecodebatched"
            -i ${ROCM_PATH}/share/rocjpeg/images/
)
if(USING_THE_ROCK)
  set_property(TEST jpeg-decode-batch-fmt-native PROPERTY ENVIRONMENT "LIBVA_DRIVERS_PATH=${ROCM_PATH}/lib/rocm_sysdeps/lib")
endif()

add_test(
  NAME
    jpeg-decode-batch-fmt-yuv-planar
  COMMAND
    "${CMAKE_CTEST_COMMAND}"
            --build-and-test "${ROCM_PATH}/share/rocjpeg/samples/jpegDecodeBatched"
                              "${CMAKE_CURRENT_BINARY_DIR}/jpegDecodeBatched"
            --build-generator "${CMAKE_GENERATOR}"
            --test-command "jpegdecodebatched"
            -i ${ROCM_PATH}/share/rocjpeg/images/ -fmt yuv_planar
)
if(USING_THE_ROCK)
  set_property(TEST jpeg-decode-batch-fmt-yuv-planar PROPERTY ENVIRONMENT "LIBVA_DRIVERS_PATH=${ROCM_PATH}/lib/rocm_sysdeps/lib")
endif()

add_test(
  NAME
    jpeg-decode-batch-fmt-y
  COMMAND
    "${CMAKE_CTEST_COMMAND}"
            --build-and-test "${ROCM_PATH}/share/rocjpeg/samples/jpegDecodeBatched"
                              "${CMAKE_CURRENT_BINARY_DIR}/jpegDecodeBatched"
            --build-generator "${CMAKE_GENERATOR}"
            --test-command "jpegdecodebatched"
            -i ${ROCM_PATH}/share/rocjpeg/images/ -fmt y
)
if(USING_THE_ROCK)
  set_property(TEST jpeg-decode-batch-fmt-y PROPERTY ENVIRONMENT "LIBVA_DRIVERS_PATH=${ROCM_PATH}/lib/rocm_sysdeps/lib")
endif()

add_test(
  NAME
    jpeg-decode-batch-fmt-rgb
  COMMAND
    "${CMAKE_CTEST_COMMAND}"
            --build-and-test "${ROCM_PATH}/share/rocjpeg/samples/jpegDecodeBatched"
                              "${CMAKE_CURRENT_BINARY_DIR}/jpegDecodeBatched"
            --build-generator "${CMAKE_GENERATOR}"
            --test-command "jpegdecodebatched"
            -i ${ROCM_PATH}/share/rocjpeg/images/ -fmt rgb
)
if(USING_THE_ROCK)
  set_property(TEST jpeg-decode-batch-fmt-rgb PROPERTY ENVIRONMENT "LIBVA_DRIVERS_PATH=${ROCM_PATH}/lib/rocm_sysdeps/lib")
endif()

add_test(
  NAME
    jpeg-decode-batch-fmt-rgb-planar
  COMMAND
    "${CMAKE_CTEST_COMMAND}"
            --build-and-test "${ROCM_PATH}/share/rocjpeg/samples/jpegDecodeBatched"
                              "${CMAKE_CURRENT_BINARY_DIR}/jpegDecodeBatched"
            --build-generator "${CMAKE_GENERATOR}"
            --test-command "jpegdecodebatched"
            -i ${ROCM_PATH}/share/rocjpeg/images/ -fmt rgb_planar
)
if(USING_THE_ROCK)
  set_property(TEST jpeg-decode-batch-fmt-rgb-planar PROPERTY ENVIRONMENT "LIBVA_DRIVERS_PATH=${ROCM_PATH}/lib/rocm_sysdeps/lib")
endif()

add_test(
  NAME
    jpeg-decode-batch-crop-fmt-native
  COMMAND
    "${CMAKE_CTEST_COMMAND}"
            --build-and-test "${ROCM_PATH}/share/rocjpeg/samples/jpegDecodeBatched"
                              "${CMAKE_CURRENT_BINARY_DIR}/jpegDecodeBatched"
            --build-generator "${CMAKE_GENERATOR}"
            --test-command "jpegdecodebatched"
            -i ${ROCM_PATH}/share/rocjpeg/images/ -crop 960,540,2880,1620
)
if(USING_THE_ROCK)
  set_property(TEST jpeg-decode-batch-crop-fmt-native PROPERTY ENVIRONMENT "LIBVA_DRIVERS_PATH=${ROCM_PATH}/lib/rocm_sysdeps/lib")
endif()

add_test(
  NAME
    jpeg-decode-batch-crop-fmt-yuv-planar
  COMMAND
    "${CMAKE_CTEST_COMMAND}"
            --build-and-test "${ROCM_PATH}/share/rocjpeg/samples/jpegDecodeBatched"
                              "${CMAKE_CURRENT_BINARY_DIR}/jpegDecodeBatched"
            --build-generator "${CMAKE_GENERATOR}"
            --test-command "jpegdecodebatched"
            -i ${ROCM_PATH}/share/rocjpeg/images/ -fmt yuv_planar -crop 960,540,2880,1620
)
if(USING_THE_ROCK)
  set_property(TEST jpeg-decode-batch-crop-fmt-yuv-planar PROPERTY ENVIRONMENT "LIBVA_DRIVERS_PATH=${ROCM_PATH}/lib/rocm_sysdeps/lib")
endif()

add_test(
  NAME
    jpeg-decode-batch-crop-fmt-y
  COMMAND
    "${CMAKE_CTEST_COMMAND}"
            --build-and-test "${ROCM_PATH}/share/rocjpeg/samples/jpegDecodeBatched"
                              "${CMAKE_CURRENT_BINARY_DIR}/jpegDecodeBatched"
            --build-generator "${CMAKE_GENERATOR}"
            --test-command "jpegdecodebatched"
            -i ${ROCM_PATH}/share/rocjpeg/images/ -fmt y -crop 960,540,2880,1620
)
if(USING_THE_ROCK)
  set_property(TEST jpeg-decode-batch-crop-fmt-y PROPERTY ENVIRONMENT "LIBVA_DRIVERS_PATH=${ROCM_PATH}/lib/rocm_sysdeps/lib")
endif()

add_test(
  NAME
    jpeg-decode-batch-crop-fmt-rgb
  COMMAND
    "${CMAKE_CTEST_COMMAND}"
            --build-and-test "${ROCM_PATH}/share/rocjpeg/samples/jpegDecodeBatched"
                              "${CMAKE_CURRENT_BINARY_DIR}/jpegDecodeBatched"
            --build-generator "${CMAKE_GENERATOR}"
            --test-command "jpegdecodebatched"
            -i ${ROCM_PATH}/share/rocjpeg/images/ -fmt rgb -crop 960,540,2880,1620
)
if(USING_THE_ROCK)
  set_property(TEST jpeg-decode-batch-crop-fmt-rgb PROPERTY ENVIRONMENT "LIBVA_DRIVERS_PATH=${ROCM_PATH}/lib/rocm_sysdeps/lib")
endif()

add_test(
  NAME
    jpeg-decode-batch-crop-fmt-rgb-planar
  COMMAND
    "${CMAKE_CTEST_COMMAND}"
            --build-and-test "${ROCM_PATH}/share/rocjpeg/samples/jpegDecodeBatched"
                              "${CMAKE_CURRENT_BINARY_DIR}/jpegDecodeBatched"
            --build-generator "${CMAKE_GENERATOR}"
            --test-command "jpegdecodebatched"
            -i ${ROCM_PATH}/share/rocjpeg/images/ -fmt rgb_planar -crop 960,540,2880,1620
)
if(USING_THE_ROCK)
  set_property(TEST jpeg-decode-batch-crop-fmt-rgb-planar PROPERTY ENVIRONMENT "LIBVA_DRIVERS_PATH=${ROCM_PATH}/lib/rocm_sysdeps/lib")
endif()

add_test(
  NAME
    jpeg-decode-perf-fmt-native
  COMMAND
    "${CMAKE_CTEST_COMMAND}"
            --build-and-test "${ROCM_PATH}/share/rocjpeg/samples/jpegDecodePerf"
                              "${CMAKE_CURRENT_BINARY_DIR}/jpegDecodePerf"
            --build-generator "${CMAKE_GENERATOR}"
            --test-command "jpegdecodeperf"
            -i ${ROCM_PATH}/share/rocjpeg/images/
)
if(USING_THE_ROCK)
  set_property(TEST jpeg-decode-perf-fmt-native PROPERTY ENVIRONMENT "LIBVA_DRIVERS_PATH=${ROCM_PATH}/lib/rocm_sysdeps/lib")
endif()

add_test(
  NAME
    jpeg-negative-api-tests
  COMMAND
    "${CMAKE_CTEST_COMMAND}"
            --build-and-test "${ROCM_PATH}/share/rocjpeg/test/rocjpeg_negative_api_tests"
                              "${CMAKE_CURRENT_BINARY_DIR}/rocjpeg_negative_api_tests"
            --build-generator "${CMAKE_GENERATOR}"
            --test-command "rocjpegnegativetest"
)
if(USING_THE_ROCK)
  set_property(TEST jpeg-negative-api-tests PROPERTY ENVIRONMENT "LIBVA_DRIVERS_PATH=${ROCM_PATH}/lib/rocm_sysdeps/lib")
endif()