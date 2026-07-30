################################################################################
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
################################################################################

cmake_minimum_required(VERSION 3.10)
# ROCM Path
if(DEFINED ENV{ROCM_PATH})
  set(ROCM_PATH $ENV{ROCM_PATH} CACHE PATH "Default ROCm installation path")
elseif(ROCM_PATH)
  message("-- INFO:ROCM_PATH Set -- ${ROCM_PATH}")
else()
  set(ROCM_PATH /opt/rocm CACHE PATH "Default ROCm installation path")
endif()

# Check if lib/rocm_sysdeps/lib exists in the ROCm path which indicates ROCm installation via TheRock
set(USING_THE_ROCK OFF)
if(EXISTS "${ROCM_PATH}/lib/rocm_sysdeps/lib")
  set(USING_THE_ROCK ON)
endif()

if(USING_THE_ROCK)
  if(NOT DEFINED ENV{HIP_DEVICE_LIB_PATH})
    set(ENV{HIP_DEVICE_LIB_PATH} ${ROCM_PATH}/lib/llvm/amdgcn/bitcode)
    set_property(GLOBAL PROPERTY RULE_LAUNCH_COMPILE "HIP_DEVICE_LIB_PATH=${ROCM_PATH}/lib/llvm/amdgcn/bitcode")
  endif()
endif()

# Set AMD Clang as default compiler
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED On)
set(CMAKE_CXX_EXTENSIONS ON)
if(NOT WIN32)
  if (NOT DEFINED CMAKE_CXX_COMPILER)
    set(CMAKE_C_COMPILER ${ROCM_PATH}/lib/llvm/bin/amdclang)
    set(CMAKE_CXX_COMPILER ${ROCM_PATH}/lib/llvm/bin/amdclang++)
  endif()
endif()

project(videodecodergb)

list(APPEND CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/../../cmake)
list(APPEND CMAKE_PREFIX_PATH ${ROCM_PATH}/lib/cmake ${ROCM_PATH}/share/rocmcmakebuildtools/cmake)

# rocdecode sample build type
set(DEFAULT_BUILD_TYPE "Release")
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE "${DEFAULT_BUILD_TYPE}" CACHE STRING "rocdecode Default Build Type" FORCE)
  set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS "Debug" "Release")
endif()
if(NOT WIN32)
  if(CMAKE_BUILD_TYPE MATCHES Debug)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O0 -gdwarf-4")
  else()
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -DNDEBUG -fPIC")
  endif()
endif()

# Set supported GPU Targets
if(NOT GPU_TARGETS AND NOT AMDGPU_TARGETS)
    find_package(ROCmCMakeBuildTools QUIET)
    if(NOT ROCmCMakeBuildTools_FOUND)
        find_package(ROCM QUIET)
    endif()
    include(ROCMCheckTargetIds OPTIONAL RESULT_VARIABLE HAS_ROCM_CHECK_TARGET_IDS)

    set(OPTIONAL_GPU_TARGETS "gfx950;gfx1150;gfx1151;gfx1152;gfx1153;gfx1200;gfx1201;gfx1250")
    if(HAS_ROCM_CHECK_TARGET_IDS AND NOT WIN32)
        rocm_check_target_ids(OPTIONAL_GPU_TARGETS_AVAILABLE TARGETS ${OPTIONAL_GPU_TARGETS})
    else() # if we don't have rocm_check_target_ids, just assume the targets are available
        set(OPTIONAL_GPU_TARGETS_AVAILABLE "${OPTIONAL_GPU_TARGETS}")
    endif()
    set(DEFAULT_GPU_TARGETS "gfx908;gfx90a;gfx942;gfx1030;gfx1031;gfx1032;gfx1100;gfx1101;gfx1102;${OPTIONAL_GPU_TARGETS_AVAILABLE}")
endif()

# Set AMD GPU_TARGETS
if((AMDGPU_TARGETS OR DEFINED ENV{AMDGPU_TARGETS}) AND (NOT GPU_TARGETS))
  message("-- ${Red}${PROJECT_NAME} DEPRECATION -- AMDGPU_TARGETS use is deprecated. Use GPU_TARGETS${ColourReset}")
  if(DEFINED ENV{AMDGPU_TARGETS})
    set(GPU_TARGETS $ENV{AMDGPU_TARGETS} CACHE STRING "List of specific machine types for library to target")
  else()
    set(GPU_TARGETS ${AMDGPU_TARGETS})
  endif()
endif()

if(DEFINED ENV{GPU_ARCHS})
  set(GPU_TARGETS $ENV{GPU_ARCHS} CACHE STRING "List of specific machine types for library to target")
elseif(GPU_TARGETS)
  message("-- ${White}${PROJECT_NAME} -- GPU_TARGETS set with -D option${ColourReset}")
else()
  set(GPU_TARGETS "${DEFAULT_GPU_TARGETS}" CACHE STRING "List of specific machine types for library to target")
endif()
message("-- ${White}${PROJECT_NAME} -- AMD GPU_TARGETS: ${GPU_TARGETS}${ColourReset}")

set(HIP_PLATFORM amd CACHE STRING "HIP platform")
find_package(HIP QUIET)
find_package(rocdecode QUIET)
if(NOT WIN32)
    find_package(rocprofiler-register QUIET)
endif()
find_package(FFmpeg QUIET)
find_package(Threads QUIET)

if(HIP_FOUND AND FFMPEG_FOUND AND rocdecode_FOUND AND (WIN32 OR (Threads_FOUND AND rocprofiler-register_FOUND)))
    # HIP
    if(WIN32)
        set(LINK_LIBRARY_LIST ${LINK_LIBRARY_LIST} hip::host)
    else()
        set(LINK_LIBRARY_LIST ${LINK_LIBRARY_LIST} hip::device)
    endif()
    # FFMPEG
    include_directories(${AVUTIL_INCLUDE_DIR} ${AVCODEC_INCLUDE_DIR})
    set(LINK_LIBRARY_LIST ${LINK_LIBRARY_LIST} ${FFMPEG_LIBRARIES})
    # rocdecode and utils
    include_directories (${CMAKE_CURRENT_SOURCE_DIR}/../../utils ${rocdecode_INCLUDE_DIR} ${CMAKE_CURRENT_SOURCE_DIR}/../../utils/rocvideodecode)
    set(LINK_LIBRARY_LIST ${LINK_LIBRARY_LIST} rocdecode::rocdecode)
    # threads
    if(Threads_FOUND)
        set(THREADS_PREFER_PTHREAD_FLAG ON)
        set(LINK_LIBRARY_LIST ${LINK_LIBRARY_LIST} Threads::Threads)
    endif()
    # rocprofiler-register
    if(rocprofiler-register_FOUND)
        set(LINK_LIBRARY_LIST ${LINK_LIBRARY_LIST} rocprofiler-register::rocprofiler-register)
    endif()
    if(WIN32)
        add_compile_definitions(NOMINMAX _CRT_SECURE_NO_WARNINGS)
    endif()

    # sample app exe
    list(APPEND SOURCES ${PROJECT_SOURCE_DIR}
                        videodecrgb.cpp
                        ${CMAKE_CURRENT_SOURCE_DIR}/../../utils/rocvideodecode/roc_video_dec.cpp)
    if(WIN32)
        # HIP kernel files must be compiled with hipcc on Windows (VS generator does not support HIP language)
        set(HIPCC ${ROCM_PATH}/bin/hipcc.exe)
        set(KERNEL_SOURCES
            ${CMAKE_CURRENT_SOURCE_DIR}/../../utils/colorspace_kernels.cpp
            ${CMAKE_CURRENT_SOURCE_DIR}/../../utils/resize_kernels.cpp)
        set(KERNEL_OBJS)
        foreach(KSRC ${KERNEL_SOURCES})
            get_filename_component(KNAME ${KSRC} NAME_WE)
            set(KOBJ ${CMAKE_CURRENT_BINARY_DIR}/${KNAME}.obj)
            add_custom_command(
                OUTPUT ${KOBJ}
                COMMAND ${HIPCC} -c ${KSRC} -o ${KOBJ}
                    --offload-arch=${GPU_TARGETS}
                    --rocm-path=${ROCM_PATH}
                    --rocm-device-lib-path=${ROCM_PATH}/lib/llvm/amdgcn/bitcode
                    -I${CMAKE_CURRENT_SOURCE_DIR}/../../utils
                    -I${CMAKE_CURRENT_SOURCE_DIR}/../../utils/rocvideodecode
                    -I${rocdecode_INCLUDE_DIR}
                    -I${ROCM_PATH}/include
                    -DNOMINMAX -D_CRT_SECURE_NO_WARNINGS
                    -fms-runtime-lib=dll
                    -O2
                DEPENDS ${KSRC}
                COMMENT "Compiling HIP kernel ${KNAME}"
            )
            list(APPEND KERNEL_OBJS ${KOBJ})
        endforeach()
        add_custom_target(hip_kernels DEPENDS ${KERNEL_OBJS})
    else()
        list(APPEND SOURCES
            ${CMAKE_CURRENT_SOURCE_DIR}/../../utils/colorspace_kernels.cpp
            ${CMAKE_CURRENT_SOURCE_DIR}/../../utils/resize_kernels.cpp)
    endif()

    add_executable(${PROJECT_NAME} ${SOURCES})
    if(WIN32)
        add_dependencies(${PROJECT_NAME} hip_kernels)
        target_link_libraries(${PROJECT_NAME} ${KERNEL_OBJS})
    # On TheRock installs libva is vendored under lib/rocm_sysdeps, but FFmpeg still
    # links the system libva, so both load at runtime. librocdecode already depends
    # on the vendored libva, so force librocdecode to be the FIRST direct dependency
    # of the executable -- ahead of FFmpeg -- so the vendored libva enters the global
    # symbol scope first and wins VA symbol interposition, loading the matching
    # vendored VA driver. It must be injected as a raw path because CMake
    # topologically reorders normal link entries, placing FFmpeg ahead otherwise.
    elseif(EXISTS "${ROCM_PATH}/lib/rocm_sysdeps/lib")
      set(LINK_LIBRARY_LIST -Wl,--push-state,--no-as-needed ${rocdecode_LIBRARY} -Wl,--pop-state ${LINK_LIBRARY_LIST})
    endif()
    target_link_libraries(${PROJECT_NAME} ${LINK_LIBRARY_LIST})
    # FFMPEG multi-version support
    if(_FFMPEG_AVCODEC_VERSION VERSION_LESS_EQUAL 58.134.100)
      target_compile_definitions(${PROJECT_NAME} PUBLIC USE_AVCODEC_GREATER_THAN_58_134=0)
    else()
      target_compile_definitions(${PROJECT_NAME} PUBLIC USE_AVCODEC_GREATER_THAN_58_134=1)
    endif()
else()
    message("-- ERROR!: ${PROJECT_NAME} excluded! unmet dependencies")
    if (NOT HIP_FOUND)
        message(FATAL_ERROR "-- ERROR!: HIP Not Found!")
    endif()
    if (NOT FFMPEG_FOUND)
        message(FATAL_ERROR "-- ERROR!: FFMPEG Not Found!")
    endif()
    if (NOT rocdecode_FOUND)
        message(FATAL_ERROR "-- ERROR!: rocdecode Not Found!")
    endif()
    if (NOT WIN32 AND NOT Threads_FOUND)
        message(FATAL_ERROR "-- ERROR!: Threads Not Found!")
    endif()
    if (NOT WIN32 AND NOT rocprofiler-register_FOUND)
        message(FATAL_ERROR "-- ERROR!: rocprofiler-register Not Found!")
    endif()
endif()
