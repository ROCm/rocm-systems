# ##############################################################################
# Copyright (c) 2023 - 2026 Advanced Micro Devices, Inc.
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
  set(ROCM_PATH $ENV{ROCM_PATH} CACHE PATH "Default ROCm installation path")
elseif(ROCM_PATH)
  message("-- INFO:ROCM_PATH Set -- ${ROCM_PATH}")
else()
  set(ROCM_PATH /opt/rocm CACHE PATH "Default ROCm installation path")
endif()
# Set AMD Clang as default compiler
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED On)
set(CMAKE_CXX_EXTENSIONS ON)
if (NOT DEFINED CMAKE_CXX_COMPILER)
  set(CMAKE_C_COMPILER ${ROCM_PATH}/lib/llvm/bin/amdclang)
  set(CMAKE_CXX_COMPILER ${ROCM_PATH}/lib/llvm/bin/amdclang++)
endif()

project(rocdecode-test)

# make test with CTest
enable_testing()
include(CTest)

# add ROCm find Config location
list(APPEND CMAKE_PREFIX_PATH ${ROCM_PATH}/lib/cmake)
list(APPEND CMAKE_MODULE_PATH ${ROCM_PATH}/share/rocdecode/cmake)

# find rocdecode
if(BUILD_FROM_SOURCE)
  message("-- ${White}${PROJECT_NAME}: CTests requires rocdecode installed before running CTests")
else()
  find_package(rocdecode QUIET)
  if(rocdecode_FOUND)
    message("-- ${White}${PROJECT_NAME}: rocdecode found with find_package(rocdecode QUIET)${ColourReset}")
    message("-- \t${White}rocdecode_INCLUDE_DIR -- ${rocdecode_INCLUDE_DIR}${ColourReset}")
    message("-- \t${White}rocdecode_LIB_DIR -- ${rocdecode_LIB_DIR}${ColourReset}")
    message("-- \t${White}rocdecode_LIBRARY -- ${rocdecode_LIBRARY}${ColourReset}")
    message("-- \t${White}rocdecode_FOUND -- ${rocdecode_FOUND}${ColourReset}")
    message("-- \t${White}rocdecode_VERSION -- ${rocdecode_VERSION}${ColourReset}")
    message("-- \t${White}rocdecode_VERSION_MAJOR -- ${rocdecode_VERSION_MAJOR}${ColourReset}")
    message("-- \t${White}rocdecode_VERSION_MINOR -- ${rocdecode_VERSION_MINOR}${ColourReset}")
    message("-- \t${White}rocdecode_VERSION_PATCH -- ${rocdecode_VERSION_PATCH}${ColourReset}")
  else()
    message("-- ${Yellow}${PROJECT_NAME} requires rocdecode. Install rocdecode before running CTests")
  endif(rocdecode_FOUND)
  find_package(rocdecode-host QUIET)
  if(rocdecode-host_FOUND)
    message("-- ${White}${PROJECT_NAME}: rocdecode-host found with find_package(rocdecode-host QUIET)${ColourReset}")
    message("-- \t${White}rocdecode-host_INCLUDE_DIR -- ${rocdecode-host_INCLUDE_DIR}${ColourReset}")
    message("-- \t${White}rocdecode-host_LIB_DIR -- ${rocdecode-host_LIB_DIR}${ColourReset}")
    message("-- \t${White}rocdecode-host_LIBRARY -- ${rocdecode-host_LIBRARY}${ColourReset}")
    message("-- \t${White}rocdecode-host_FOUND -- ${rocdecode-host_FOUND}${ColourReset}")
    message("-- \t${White}rocdecode-host_VERSION -- ${rocdecode-host_VERSION}${ColourReset}")
    message("-- \t${White}rocdecode-host_VERSION_MAJOR -- ${rocdecode-host_VERSION_MAJOR}${ColourReset}")
    message("-- \t${White}rocdecode-host_VERSION_MINOR -- ${rocdecode-host_VERSION_MINOR}${ColourReset}")
    message("-- \t${White}rocdecode-host_VERSION_PATCH -- ${rocdecode-host_VERSION_PATCH}${ColourReset}")
  else()
    message("-- ${Yellow}${PROJECT_NAME} rocdecode-host is optional. Install rocdecode-host to run additional tests before running CTests")
  endif(rocdecode-host_FOUND)
endif(BUILD_FROM_SOURCE)
find_package(FFmpeg QUIET)

# Check if lib/rocm_sysdeps/lib exists in the ROCm path which indicates ROCm installation via TheRock
set(USING_THE_ROCK OFF)
if(EXISTS "${ROCM_PATH}/lib/rocm_sysdeps/lib")
  set(USING_THE_ROCK ON)
endif()

# 1 - videoDecodeRaw HEVC
add_test(
  NAME
    video_decodeRaw-HEVC
  COMMAND
    "${CMAKE_CTEST_COMMAND}"
            --build-and-test "${ROCM_PATH}/share/rocdecode/samples/videoDecodeRaw"
                              "${CMAKE_CURRENT_BINARY_DIR}/videoDecodeRaw-HEVC"
            --build-generator "${CMAKE_GENERATOR}"
            --test-command "videodecoderaw"
            -i ${ROCM_PATH}/share/rocdecode/video/AMD_driving_virtual_20-H265.265
)
if(USING_THE_ROCK)
  set_property(TEST video_decodeRaw-HEVC PROPERTY ENVIRONMENT "LIBVA_DRIVERS_PATH=${ROCM_PATH}/lib/rocm_sysdeps/lib")
endif()

# 2 - videoDecodeRaw AVC
add_test(
  NAME
    video_decodeRaw-AVC
  COMMAND
    "${CMAKE_CTEST_COMMAND}"
            --build-and-test "${ROCM_PATH}/share/rocdecode/samples/videoDecodeRaw"
                              "${CMAKE_CURRENT_BINARY_DIR}/videoDecodeRaw-AVC"
            --build-generator "${CMAKE_GENERATOR}"
            --test-command "videodecoderaw"
            -i ${ROCM_PATH}/share/rocdecode/video/AMD_driving_virtual_20-H264.264
)
if(USING_THE_ROCK)
  set_property(TEST video_decodeRaw-AVC PROPERTY ENVIRONMENT "LIBVA_DRIVERS_PATH=${ROCM_PATH}/lib/rocm_sysdeps/lib")
endif()

# 3 - videoDecodeRaw AV1
add_test(
  NAME
    video_decodeRaw-AV1
  COMMAND
    "${CMAKE_CTEST_COMMAND}"
            --build-and-test "${ROCM_PATH}/share/rocdecode/samples/videoDecodeRaw"
                              "${CMAKE_CURRENT_BINARY_DIR}/videoDecodeRaw-AV1"
            --build-generator "${CMAKE_GENERATOR}"
            --test-command "videodecoderaw"
            -i ${ROCM_PATH}/share/rocdecode/video/AMD_driving_virtual_20-AV1.ivf
)
if(USING_THE_ROCK)
  set_property(TEST video_decodeRaw-AV1 PROPERTY ENVIRONMENT "LIBVA_DRIVERS_PATH=${ROCM_PATH}/lib/rocm_sysdeps/lib")
endif()

# 4 - videoDecodeRaw VP9
add_test(
  NAME
    video_decodeRaw-VP9
  COMMAND
    "${CMAKE_CTEST_COMMAND}"
            --build-and-test "${ROCM_PATH}/share/rocdecode/samples/videoDecodeRaw"
                              "${CMAKE_CURRENT_BINARY_DIR}/videoDecodeRaw-VP9"
            --build-generator "${CMAKE_GENERATOR}"
            --test-command "videodecoderaw"
            -i ${ROCM_PATH}/share/rocdecode/video/AMD_driving_virtual_20-VP9.ivf
)
if(USING_THE_ROCK)
  set_property(TEST video_decodeRaw-VP9 PROPERTY ENVIRONMENT "LIBVA_DRIVERS_PATH=${ROCM_PATH}/lib/rocm_sysdeps/lib")
endif()

# 5 - rocdecDecode
add_test(
  NAME
    rocdec_Decode-HEVC
  COMMAND
    "${CMAKE_CTEST_COMMAND}"
            --build-and-test "${ROCM_PATH}/share/rocdecode/samples/rocdecDecode"
                              "${CMAKE_CURRENT_BINARY_DIR}/rocdecDecode"
            --build-generator "${CMAKE_GENERATOR}"
            --test-command "rocdecdecode"
            -i ${ROCM_PATH}/share/rocdecode/frames
)
if(USING_THE_ROCK)
  set_property(TEST rocdec_Decode-HEVC PROPERTY ENVIRONMENT "LIBVA_DRIVERS_PATH=${ROCM_PATH}/lib/rocm_sysdeps/lib")
endif()

# 6 - rocDecodeNegativeApiTests
add_test(
  NAME
  rocDecode_Negative_API_Tests
  COMMAND
    "${CMAKE_CTEST_COMMAND}"
            --build-and-test "${ROCM_PATH}/share/rocdecode/test/rocDecodeNegativeApiTests"
                              "${CMAKE_CURRENT_BINARY_DIR}/rocdecodenegativetest"
            --build-generator "${CMAKE_GENERATOR}"
            --test-command "rocdecodenegativetest"
)
if(USING_THE_ROCK)
  set_property(TEST rocDecode_Negative_API_Tests PROPERTY ENVIRONMENT "LIBVA_DRIVERS_PATH=${ROCM_PATH}/lib/rocm_sysdeps/lib")
endif()

if(FFMPEG_FOUND)
  message("-- ${Green}${PROJECT_NAME} FFmpeg found - rocdecode tests requiring FFmpeg added")
  # 7 - videoDecode HEVC
  add_test(
    NAME
      video_decode-HEVC
    COMMAND
      "${CMAKE_CTEST_COMMAND}"
            --build-and-test "${ROCM_PATH}/share/rocdecode/samples/videoDecode"
                              "${CMAKE_CURRENT_BINARY_DIR}/videoDecode-HEVC"
            --build-generator "${CMAKE_GENERATOR}"
            --test-command "videodecode"
            -i ${ROCM_PATH}/share/rocdecode/video/AMD_driving_virtual_20-H265.mp4
  )
  if(USING_THE_ROCK)
    set_property(TEST video_decode-HEVC PROPERTY ENVIRONMENT "LIBVA_DRIVERS_PATH=${ROCM_PATH}/lib/rocm_sysdeps/lib")
  endif()

  # 8 - videoDecode AVC
  add_test(
    NAME
      video_decode-AVC
    COMMAND
      "${CMAKE_CTEST_COMMAND}"
            --build-and-test "${ROCM_PATH}/share/rocdecode/samples/videoDecode"
                              "${CMAKE_CURRENT_BINARY_DIR}/videoDecode-AVC"
            --build-generator "${CMAKE_GENERATOR}"
            --test-command "videodecode"
            -i ${ROCM_PATH}/share/rocdecode/video/AMD_driving_virtual_20-H264.mp4
  )
  if(USING_THE_ROCK)
    set_property(TEST video_decode-AVC PROPERTY ENVIRONMENT "LIBVA_DRIVERS_PATH=${ROCM_PATH}/lib/rocm_sysdeps/lib")
  endif()

  # 9 - videoDecode AV1
  add_test(
    NAME
      video_decode-AV1
    COMMAND
      "${CMAKE_CTEST_COMMAND}"
            --build-and-test "${ROCM_PATH}/share/rocdecode/samples/videoDecode"
                              "${CMAKE_CURRENT_BINARY_DIR}/videoDecode-AV1"
            --build-generator "${CMAKE_GENERATOR}"
            --test-command "videodecode"
            -i ${ROCM_PATH}/share/rocdecode/video/AMD_driving_virtual_20-AV1.mp4
  )
  if(USING_THE_ROCK)
    set_property(TEST video_decode-AV1 PROPERTY ENVIRONMENT "LIBVA_DRIVERS_PATH=${ROCM_PATH}/lib/rocm_sysdeps/lib")
  endif()

  # 10 - videoDecode VP9
  add_test(
    NAME
      video_decode-VP9
    COMMAND
      "${CMAKE_CTEST_COMMAND}"
            --build-and-test "${ROCM_PATH}/share/rocdecode/samples/videoDecode"
                              "${CMAKE_CURRENT_BINARY_DIR}/videoDecode-VP9"
            --build-generator "${CMAKE_GENERATOR}"
            --test-command "videodecode"
            -i ${ROCM_PATH}/share/rocdecode/video/AMD_driving_virtual_20-VP9.ivf
  )
  if(USING_THE_ROCK)
    set_property(TEST video_decode-VP9 PROPERTY ENVIRONMENT "LIBVA_DRIVERS_PATH=${ROCM_PATH}/lib/rocm_sysdeps/lib")
  endif()

  # 11 - videoDecodePerf
  add_test(
    NAME
      video_decodePerf-HEVC
    COMMAND
      "${CMAKE_CTEST_COMMAND}"
            --build-and-test "${ROCM_PATH}/share/rocdecode/samples/videoDecodePerf"
                              "${CMAKE_CURRENT_BINARY_DIR}/videoDecodePerf"
            --build-generator "${CMAKE_GENERATOR}"
            --test-command "videodecodeperf"
            -i ${ROCM_PATH}/share/rocdecode/video/AMD_driving_virtual_20-H265.mp4
  )
  if(USING_THE_ROCK)
    set_property(TEST video_decodePerf-HEVC PROPERTY ENVIRONMENT "LIBVA_DRIVERS_PATH=${ROCM_PATH}/lib/rocm_sysdeps/lib")
  endif()

  # 12 - videoDecodeBatch
  add_test(
    NAME
      video_decodeBatch
    COMMAND
      "${CMAKE_CTEST_COMMAND}"
            --build-and-test "${ROCM_PATH}/share/rocdecode/samples/videoDecodeBatch"
                              "${CMAKE_CURRENT_BINARY_DIR}/videoDecodeBatch"
            --build-generator "${CMAKE_GENERATOR}"
            --test-command "videodecodebatch"
            -i ${ROCM_PATH}/share/rocdecode/video/ -t 2
  )
  if(USING_THE_ROCK)
    set_property(TEST video_decodeBatch PROPERTY ENVIRONMENT "LIBVA_DRIVERS_PATH=${ROCM_PATH}/lib/rocm_sysdeps/lib")
  endif()

  # 13 - videoDecodeRGB
  add_test(
    NAME
      video_decodeRGB-HEVC
    COMMAND
      "${CMAKE_CTEST_COMMAND}"
            --build-and-test "${ROCM_PATH}/share/rocdecode/samples/videoDecodeRGB"
                              "${CMAKE_CURRENT_BINARY_DIR}/videoDecodeRGB"
            --build-generator "${CMAKE_GENERATOR}"
            --test-command "videodecodergb"
            -i ${ROCM_PATH}/share/rocdecode/video/AMD_driving_virtual_20-H265.mp4 -of rgb
  )
  if(USING_THE_ROCK)
    set_property(TEST video_decodeRGB-HEVC PROPERTY ENVIRONMENT "LIBVA_DRIVERS_PATH=${ROCM_PATH}/lib/rocm_sysdeps/lib")
  endif()

  # 14 - videoDecodeMem
  add_test(
    NAME
      video_decodeMem-HEVC
    COMMAND
      "${CMAKE_CTEST_COMMAND}"
            --build-and-test "${ROCM_PATH}/share/rocdecode/samples/videoDecodeMem"
                              "${CMAKE_CURRENT_BINARY_DIR}/videoDecodeMem"
            --build-generator "${CMAKE_GENERATOR}"
            --test-command "videodecodemem"
            -i ${ROCM_PATH}/share/rocdecode/video/AMD_driving_virtual_20-H265.mp4
  )
  if(USING_THE_ROCK)
    set_property(TEST video_decodeMem-HEVC PROPERTY ENVIRONMENT "LIBVA_DRIVERS_PATH=${ROCM_PATH}/lib/rocm_sysdeps/lib")
  endif()

  # 15 - videoDecodeRGBResize
  add_test(
    NAME
      video_decodeRGB-Resize
    COMMAND
      "${CMAKE_CTEST_COMMAND}"
            --build-and-test "${ROCM_PATH}/share/rocdecode/samples/videoDecodeRGB"
                              "${CMAKE_CURRENT_BINARY_DIR}/videoDecodeRGB-Resize"
            --build-generator "${CMAKE_GENERATOR}"
            --test-command "videodecodergb"
            -i "${ROCM_PATH}/share/rocdecode/video/AMD_driving_virtual_20-H265.mp4" -resize 640x360 -of rgb
  )
  if(USING_THE_ROCK)
    set_property(TEST video_decodeRGB-Resize PROPERTY ENVIRONMENT "LIBVA_DRIVERS_PATH=${ROCM_PATH}/lib/rocm_sysdeps/lib")
  endif()

  if(rocdecode-host_FOUND)
    # 16 - videoDecode Host backend
    add_test(
      NAME
        video_decode-Host-Backend
      COMMAND
        "${CMAKE_CTEST_COMMAND}"
            --build-and-test "${ROCM_PATH}/share/rocdecode/samples/videoDecode"
                              "${CMAKE_CURRENT_BINARY_DIR}/videoDecode-Host"
            --build-generator "${CMAKE_GENERATOR}"
            --test-command "videodecode"
            -i ${ROCM_PATH}/share/rocdecode/video/AMD_driving_virtual_20-H265.mp4
            -backend 1
    )
    if(USING_THE_ROCK)
      set_property(TEST video_decode-Host-Backend PROPERTY ENVIRONMENT "LIBVA_DRIVERS_PATH=${ROCM_PATH}/lib/rocm_sysdeps/lib")
    endif()
  else()
    message("-- ${Yellow}${PROJECT_NAME} rocdecode-host NOT found. rocdecode tests requiring rocdecode-host excluded")
  endif(rocdecode-host_FOUND)
else()
  message("-- ${Yellow}${PROJECT_NAME} FFmpeg NOT found. rocdecode tests requiring FFmpeg excluded")
endif(FFMPEG_FOUND)
