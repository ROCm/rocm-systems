# Copyright (c) 2025 Advanced Micro Devices, Inc. All rights reserved.
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

function(hip_compute_version)
  find_package(Git QUIET)
  if(NOT GIT_FOUND)
    return()
  endif()

  execute_process(
    COMMAND ${GIT_EXECUTABLE} describe --tags --match "hip-version_*" --abbrev=0
    WORKING_DIRECTORY ${CMAKE_CURRENT_LIST_DIR}
    RESULT_VARIABLE _tag_res
    OUTPUT_VARIABLE _tag
    OUTPUT_STRIP_TRAILING_WHITESPACE
  )

  if(_tag_res EQUAL 0 AND _tag)
    string(REGEX MATCH "hip-version_([0-9]+)\\.([0-9]+)\\.([0-9]+)" _m "${_tag}")
    if(_m)
      set(HIP_VERSION_MAJOR "${CMAKE_MATCH_1}" PARENT_SCOPE)
      set(HIP_VERSION_MINOR "${CMAKE_MATCH_2}" PARENT_SCOPE)
      set(HIP_VERSION_PATCH "${CMAKE_MATCH_3}" PARENT_SCOPE)
      message(STATUS "Computed HIP version: ${CMAKE_MATCH_1}.${CMAKE_MATCH_2}.${CMAKE_MATCH_3}")
      return()
    endif()
  endif()

  message(WARNING "HIP version not found!")
endfunction()
