## Copyright (c) 2026 Advanced Micro Devices, Inc.
##
## Permission is hereby granted, free of charge, to any person obtaining a copy
## of this software and associated documentation files (the "Software"), to
## deal in the Software without restriction, including without limitation the
## rights to use, copy, modify, merge, publish, distribute, sublicense, and/or
## sell copies of the Software, and to permit persons to whom the Software is
## furnished to do so, subject to the following conditions:
##
## The above copyright notice and this permission notice shall be included in
## all copies or substantial portions of the Software.
##
## THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
## IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
## FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
## AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
## LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
## FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
## IN THE SOFTWARE.

# Helper functions for declaring rocdbgapi tests.
#
# These wrap add_executable / add_test so each test gets a consistent
# set of properties (C++ standard, GoogleTest link, CTest labels,
# working directory) without each CMakeLists.txt repeating the same
# boilerplate.
#
# Public API:
#   add_dbgapi_unit_test    (name SOURCES ... [LABELS ...] [LIBS ...])
#   add_dbgapi_feature_test (name SOURCES ... [LABELS ...] [LIBS ...]
#                            [REQUIRES_DRIVER] [REQUIRES_GPU [arch ...]])
#
# All tests get the "dbgapi" label automatically.  Unit tests also get
# "unit"; feature tests get "feature" plus any tier labels implied by
# REQUIRES_DRIVER / REQUIRES_GPU.

include_guard(GLOBAL)

# Single source of truth for OS labels so they don't drift between
# helpers.
if(CMAKE_SYSTEM_NAME STREQUAL "Linux")
  set(_DBGAPI_OS_LABEL "os-linux")
elseif(CMAKE_SYSTEM_NAME STREQUAL "Windows")
  set(_DBGAPI_OS_LABEL "os-windows")
else()
  set(_DBGAPI_OS_LABEL "")
endif()

# Internal: configure common properties shared by all test executables.
function(_dbgapi_apply_common_test_properties target)
  set_target_properties(${target} PROPERTIES
    CXX_STANDARD 17
    CXX_STANDARD_REQUIRED ON
    CXX_EXTENSIONS ON)

  if(MSVC)
    # rocdbgapi disables RTTI in the main library but test code needs
    # RTTI for GMock and gtest's death-test machinery.  Don't propagate
    # the library's /GR- here; the default /GR is what we want.
    target_compile_options(${target} PRIVATE /permissive-)
  else()
    target_compile_options(${target} PRIVATE
      -Wall -Wextra -Wshadow -Wno-attributes)
  endif()
endfunction()

# add_dbgapi_unit_test(name
#                     SOURCES src1.cpp [src2.cpp ...]
#                     [LABELS extra-label ...]
#                     [LIBS  extra::link ...])
function(add_dbgapi_unit_test target_name)
  cmake_parse_arguments(ARG "" "" "SOURCES;LABELS;LIBS" ${ARGN})

  if(NOT ARG_SOURCES)
    message(FATAL_ERROR "add_dbgapi_unit_test(${target_name}): SOURCES is required")
  endif()

  add_executable(${target_name} ${ARG_SOURCES})
  _dbgapi_apply_common_test_properties(${target_name})

  target_link_libraries(${target_name} PRIVATE
    GTest::gtest_main
    GTest::gmock
    ${ARG_LIBS})

  set(labels "dbgapi" "unit")
  if(_DBGAPI_OS_LABEL)
    list(APPEND labels ${_DBGAPI_OS_LABEL})
  endif()
  if(ARG_LABELS)
    list(APPEND labels ${ARG_LABELS})
  endif()
  list(REMOVE_DUPLICATES labels)

  add_test(NAME ${target_name} COMMAND ${target_name})
  set_tests_properties(${target_name} PROPERTIES
    LABELS "${labels}"
    WORKING_DIRECTORY $<TARGET_FILE_DIR:${target_name}>)
endfunction()

# add_dbgapi_feature_test(name
#                        SOURCES src1.cpp [src2.cpp ...]
#                        [LABELS extra-label ...]
#                        [LIBS  extra::link ...]
#                        [REQUIRES_DRIVER]
#                        [REQUIRES_GPU [gfx_target ...]])
#
# REQUIRES_DRIVER  - test needs the kernel driver at runtime; adds the
#                    "feature-driver" label plus the OS-specific
#                    "feature-kfd"/"feature-kmd" label.  Tests still
#                    decide at runtime whether to GTEST_SKIP if the
#                    driver isn't present (probes added in Commit 17).
#
# REQUIRES_GPU     - test needs a real GPU.  Without arguments adds
#                    "feature-gpu".  With architecture names (e.g.
#                    gfx942) adds "feature-gpu-<arch>" for each one so
#                    CI can route the test to the right runner.
function(add_dbgapi_feature_test target_name)
  # REQUIRES_GPU is declared as a multi-value keyword so callers can
  # optionally append architecture names (e.g. REQUIRES_GPU gfx942).
  # When the keyword is used with zero arguments,
  # cmake_parse_arguments leaves ARG_REQUIRES_GPU undefined and lists
  # the keyword in ARG_KEYWORDS_MISSING_VALUES instead — without
  # checking that list we'd silently drop the feature-gpu label for
  # every test that just wants "some GPU", which is the common case.
  cmake_parse_arguments(ARG "REQUIRES_DRIVER" "" "SOURCES;LABELS;LIBS;REQUIRES_GPU" ${ARGN})

  if(NOT ARG_SOURCES)
    message(FATAL_ERROR "add_dbgapi_feature_test(${target_name}): SOURCES is required")
  endif()

  add_executable(${target_name} ${ARG_SOURCES})
  _dbgapi_apply_common_test_properties(${target_name})

  target_link_libraries(${target_name} PRIVATE
    GTest::gtest_main
    GTest::gmock
    ${ARG_LIBS})

  set(labels "dbgapi" "feature")
  if(_DBGAPI_OS_LABEL)
    list(APPEND labels ${_DBGAPI_OS_LABEL})
  endif()

  if(ARG_REQUIRES_DRIVER)
    list(APPEND labels "feature-driver")
    if(CMAKE_SYSTEM_NAME STREQUAL "Linux")
      list(APPEND labels "feature-kfd")
    elseif(CMAKE_SYSTEM_NAME STREQUAL "Windows")
      list(APPEND labels "feature-kmd")
    endif()
  endif()

  if(DEFINED ARG_REQUIRES_GPU OR "REQUIRES_GPU" IN_LIST ARG_KEYWORDS_MISSING_VALUES)
    # The keyword was specified.  Even if no architectures were named,
    # the test still needs *some* GPU, so always emit "feature-gpu".
    list(APPEND labels "feature-gpu")
    foreach(arch IN LISTS ARG_REQUIRES_GPU)
      list(APPEND labels "feature-gpu-${arch}")
    endforeach()
  endif()

  if(ARG_LABELS)
    list(APPEND labels ${ARG_LABELS})
  endif()
  list(REMOVE_DUPLICATES labels)

  add_test(NAME ${target_name} COMMAND ${target_name})
  set_tests_properties(${target_name} PROPERTIES
    LABELS "${labels}"
    WORKING_DIRECTORY $<TARGET_FILE_DIR:${target_name}>)
endfunction()
