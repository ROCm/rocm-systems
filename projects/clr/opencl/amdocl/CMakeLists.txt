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

project(amdocl)

option(BUILD_SHARED_LIBS "Build the shared library" ON)

if(ADDRESS_SANITIZER)
  set(ASAN_LINKER_FLAGS "-fsanitize=address")
  set(ASAN_COMPILER_FLAGS "-fno-omit-frame-pointer -fsanitize=address")

  if(NOT CMAKE_COMPILER_IS_GNUCC)
    if(BUILD_SHARED_LIBS)
      set(ASAN_COMPILER_FLAGS "${ASAN_COMPILER_FLAGS} -shared-libsan")
      set(ASAN_LINKER_FLAGS "${ASAN_LINKER_FLAGS} -shared-libsan")
    else()
      set(ASAN_LINKER_FLAGS "${ASAN_LINKER_FLAGS} -static-libsan")
    endif()
  endif()

  set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${ASAN_COMPILER_FLAGS}")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${ASAN_COMPILER_FLAGS}")
  set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${ASAN_LINKER_FLAGS} -s -Wl,--build-id=sha1")
  set(CMAKE_SHARED_LINKER_FLAGS "${CMAKE_SHARED_LINKER_FLAGS} ${ASAN_LINKER_FLAGS} -Wl,--build-id=sha1")
endif()

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_LIST_DIR}/cmake")

if(BUILD_SHARED_LIBS)
  add_library(amdocl SHARED)
  # Windows doesn't have a strip utility, so CMAKE_STRIP won't be set.
  if((CMAKE_BUILD_TYPE STREQUAL "Release") AND NOT ("${CMAKE_STRIP}" STREQUAL ""))
    add_custom_command(TARGET amdocl POST_BUILD COMMAND ${CMAKE_STRIP} $<TARGET_FILE:amdocl>)
  endif()
else()
  add_library(amdocl STATIC)
endif()

set_target_properties(amdocl PROPERTIES
  CXX_STANDARD 17
  CXX_STANDARD_REQUIRED ON
  CXX_EXTENSIONS OFF
  POSITION_INDEPENDENT_CODE ON)

# set amdocl library version
set(AMDOCL_LIB_VERSION_MAJOR "2")
set(AMDOCL_LIB_VERSION_MINOR "1")
if(DEFINED ENV{ROCM_LIBPATCH_VERSION})
  set(AMDOCL_LIB_VERSION_PATCH "${AMDOCL_LIB_VERSION_MAJOR}.${AMDOCL_LIB_VERSION_MINOR}.$ENV{ROCM_LIBPATCH_VERSION}")
endif()

set_target_properties(amdocl PROPERTIES
                      VERSION "${AMDOCL_LIB_VERSION_PATCH}"
                      SOVERSION ${AMDOCL_LIB_VERSION_MAJOR})

if(CMAKE_SIZEOF_VOID_P EQUAL 8)
  set_target_properties(amdocl PROPERTIES OUTPUT_NAME "amdocl64")
else()
  set_target_properties(amdocl PROPERTIES OUTPUT_NAME "amdocl32")
endif()

target_include_directories(amdocl PUBLIC $<TARGET_PROPERTY:rocclr,INCLUDE_DIRECTORIES>)

target_sources(amdocl PRIVATE
  cl_command.cpp
  cl_context.cpp
  cl_counter.cpp
  cl_d3d9.cpp
  cl_d3d10.cpp
  cl_d3d11.cpp
  cl_device.cpp
  cl_event.cpp
  cl_execute.cpp
  cl_gl.cpp
  cl_icd.cpp
  cl_kernel_info_amd.cpp
  cl_memobj.cpp
  cl_p2p_amd.cpp
  cl_pipe.cpp
  cl_platform_amd.cpp
  cl_profile_amd.cpp
  cl_program.cpp
  cl_sampler.cpp
  cl_sdi_amd.cpp
  cl_svm.cpp
  cl_thread_trace_amd.cpp)

if(WIN32)
  target_sources(amdocl PRIVATE
  cl_runtime.cpp)
endif()

if(BUILD_SHARED_LIBS)
  if(WIN32)
    target_sources(amdocl PRIVATE amdocl.def)
  else()
    # -Bsymbolic is required to make sure all AMD OpenCL runtime symbols are resolved internally
    # Otherwise ld might resolve them using symbols from the OpenCL ICD
    target_link_libraries(amdocl PRIVATE "-Wl,-Bsymbolic")
    target_link_libraries(amdocl PRIVATE "-Wl,--version-script=${CMAKE_CURRENT_LIST_DIR}/amdocl.map")
    set_target_properties(amdocl PROPERTIES LINK_DEPENDS "${CMAKE_CURRENT_LIST_DIR}/amdocl.map")
  endif()
endif()

if(WIN32)
  configure_file(amdocl.rc.in amdocl_info.rc @ONLY)
  target_sources(amdocl PRIVATE amdocl_info.rc)
endif()

target_link_libraries(amdocl PUBLIC rocclr)

INSTALL(TARGETS amdocl
  COMPONENT MAIN
  RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
  LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
  ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR})
