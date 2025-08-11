# Copyright (c) 2020 - 2021 Advanced Micro Devices, Inc. All rights reserved.
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

# ROCclr abstracts the usage of multiple AMD compilers and runtimes.
# It is possible to support multiple backends concurrently in the same binary.
option(ROCCLR_ENABLE_HSAIL "Enable support for HSAIL compiler" OFF)
option(ROCCLR_ENABLE_LC    "Enable support for LC compiler"    ON)
option(ROCCLR_ENABLE_HSA   "Enable support for HSA runtime"    ON)
option(ROCCLR_ENABLE_PAL   "Enable support for PAL runtime"    OFF)

if((NOT ROCCLR_ENABLE_HSAIL) AND (NOT ROCCLR_ENABLE_LC))
  message(FATAL "Support for at least one compiler needs to be enabled!")
endif()

if((NOT ROCCLR_ENABLE_HSA) AND (NOT ROCCLR_ENABLE_PAL))
  message(FATAL "Support for at least one runtime needs to be enabled!")
endif()

set(THREADS_PREFER_PTHREAD_FLAG ON)
find_package(Threads REQUIRED)

find_package(AMD_OPENCL)

add_library(rocclr STATIC)

include(ROCclrCompilerOptions)

# To Fix path issue due to current dir (cmake folder - cmake/../) in debuginfo
get_filename_component(_ROCCLR_SRC_DIR_PATH "${CMAKE_CURRENT_LIST_DIR}/../" REALPATH)
set(ROCCLR_SRC_DIR "${_ROCCLR_SRC_DIR_PATH}")

mark_as_advanced(ROCCLR_SRC_DIR)
set(ROCCLR_INCLUDE_DIR "${ROCCLR_SRC_DIR}/include" PARENT_SCOPE)
mark_as_advanced(ROCCLR_INCLUDE_DIR)

set_target_properties(rocclr PROPERTIES
    CXX_STANDARD 17
    CXX_STANDARD_REQUIRED ON
    CXX_EXTENSIONS OFF
    POSITION_INDEPENDENT_CODE ON)

target_sources(rocclr PRIVATE
  ${ROCCLR_SRC_DIR}/compiler/lib/utils/options.cpp
  ${ROCCLR_SRC_DIR}/device/appprofile.cpp
  ${ROCCLR_SRC_DIR}/device/blit.cpp
  ${ROCCLR_SRC_DIR}/device/blitcl.cpp
  ${ROCCLR_SRC_DIR}/device/comgrctx.cpp
  ${ROCCLR_SRC_DIR}/device/devhcmessages.cpp
  ${ROCCLR_SRC_DIR}/device/devhcprintf.cpp
  ${ROCCLR_SRC_DIR}/device/devhostcall.cpp
  ${ROCCLR_SRC_DIR}/device/device.cpp
  ${ROCCLR_SRC_DIR}/device/devkernel.cpp
  ${ROCCLR_SRC_DIR}/device/devprogram.cpp
  ${ROCCLR_SRC_DIR}/device/hsailctx.cpp
  ${ROCCLR_SRC_DIR}/elf/elf.cpp
  ${ROCCLR_SRC_DIR}/os/alloc.cpp
  ${ROCCLR_SRC_DIR}/os/os_posix.cpp
  ${ROCCLR_SRC_DIR}/os/os_win32.cpp
  ${ROCCLR_SRC_DIR}/os/os.cpp
  ${ROCCLR_SRC_DIR}/platform/activity.cpp
  ${ROCCLR_SRC_DIR}/platform/agent.cpp
  ${ROCCLR_SRC_DIR}/platform/command.cpp
  ${ROCCLR_SRC_DIR}/platform/commandqueue.cpp
  ${ROCCLR_SRC_DIR}/platform/context.cpp
  ${ROCCLR_SRC_DIR}/platform/kernel.cpp
  ${ROCCLR_SRC_DIR}/platform/vmheap.cpp
  ${ROCCLR_SRC_DIR}/platform/memory.cpp
  ${ROCCLR_SRC_DIR}/platform/ndrange.cpp
  ${ROCCLR_SRC_DIR}/platform/program.cpp
  ${ROCCLR_SRC_DIR}/platform/runtime.cpp
  ${ROCCLR_SRC_DIR}/platform/interop_gl.cpp
  ${ROCCLR_SRC_DIR}/thread/thread.cpp
  ${ROCCLR_SRC_DIR}/utils/debug.cpp
  ${ROCCLR_SRC_DIR}/utils/flags.cpp)

find_package(Git)

set(ROCCLR_VERSION_GITHASH "not_found")
if(GIT_FOUND)
  # Get commit short hash
  execute_process(COMMAND ${GIT_EXECUTABLE} rev-parse --short HEAD
    WORKING_DIRECTORY ${PROJECT_SOURCE_DIR}
    RESULT_VARIABLE git_result
    OUTPUT_VARIABLE git_output
    OUTPUT_STRIP_TRAILING_WHITESPACE)
  if (git_result EQUAL 0)
    set(ROCCLR_VERSION_GITHASH ${git_output})
  endif()
endif()

target_compile_definitions(rocclr PRIVATE ROCCLR_VERSION_GITHASH="${ROCCLR_VERSION_GITHASH}")

if(WIN32)
  target_sources(rocclr PRIVATE
  ${ROCCLR_SRC_DIR}/platform/interop_d3d9.cpp
  ${ROCCLR_SRC_DIR}/platform/interop_d3d10.cpp
  ${ROCCLR_SRC_DIR}/platform/interop_d3d11.cpp)
  target_compile_definitions(rocclr PUBLIC ATI_OS_WIN)
else()
  target_compile_definitions(rocclr PUBLIC ATI_OS_LINUX)
endif()

if(CMAKE_SIZEOF_VOID_P EQUAL 4)
  target_compile_definitions(rocclr PUBLIC ATI_BITS_32)
endif()

target_compile_definitions(rocclr PUBLIC
  LITTLEENDIAN_CPU
  ${AMD_OPENCL_DEFS})

target_include_directories(rocclr PUBLIC
  ${ROCCLR_SRC_DIR}/../
  ${ROCCLR_SRC_DIR}
  ${ROCCLR_SRC_DIR}/compiler/lib
  ${ROCCLR_SRC_DIR}/compiler/lib/include
  ${ROCCLR_SRC_DIR}/compiler/lib/backends/common
  ${ROCCLR_SRC_DIR}/device
  ${ROCCLR_SRC_DIR}/elf
  ${ROCCLR_SRC_DIR}/include
  ${AMD_OPENCL_INCLUDE_DIRS})

target_link_libraries(rocclr PUBLIC Threads::Threads)
# IPC on Windows is not supported
if(UNIX)
  target_link_libraries(rocclr PUBLIC rt)
endif()

if(ROCCLR_ENABLE_HSAIL)
  include(ROCclrHSAIL)
endif()

if(ROCCLR_ENABLE_LC)
  include(ROCclrLC)
endif()

if(ROCCLR_ENABLE_HSA)
  include(ROCclrHSA)
endif()

if(ROCCLR_ENABLE_PAL)
  include(ROCclrPAL)
endif()
