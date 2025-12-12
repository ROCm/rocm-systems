# Copyright (c) 2022 Advanced Micro Devices, Inc. All rights reserved.
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

# Attempt to collect the latest git hash
execute_process(COMMAND git log --pretty=format:'%h' -n 1
                OUTPUT_VARIABLE GIT_REV
                ERROR_QUIET)

# Check if git information was found
if ("${GIT_REV}" STREQUAL "")
  set(CURR_GIT_VERSION "const char *rcclTestsGitHash =\"Unknown \";")
else()
  # Check for changes (denote with a '+') after hash
  execute_process(
    COMMAND bash -c "git diff --quiet --exit-code || echo +"
    OUTPUT_VARIABLE GIT_DIFF)
  # Collect branch information
  execute_process(
    COMMAND git rev-parse --abbrev-ref HEAD
    OUTPUT_VARIABLE GIT_BRANCH)

  string(STRIP "${GIT_REV}" GIT_REV)
  string(SUBSTRING "${GIT_REV}" 1 7 GIT_REV)
  string(STRIP "${GIT_DIFF}" GIT_DIFF)
  string(STRIP "${GIT_BRANCH}" GIT_BRANCH)

  set(CURR_GIT_VERSION "const char *rcclTestsGitHash =\"${GIT_BRANCH}:${GIT_REV}${GIT_DIFF}\";")
endif()

# Compare file with older git version file (git_version.cpp)
if (EXISTS ${CMAKE_CURRENT_BINARY_DIR}/git_version.cpp)
  #MESSAGE(STATUS "Found ${CMAKE_CURRENT_BINARY_DIR}/git_version.cpp")
  file(READ ${CMAKE_CURRENT_BINARY_DIR}/git_version.cpp PREV_GIT_VERSION)
  #message(STATUS "CURR GIT version: ${CURR_GIT_VERSION}")
  #message(STATUS "PREV GIT version: ${PREV_GIT_VERSION}")
  if (NOT "${CURR_GIT_VERSION}" STREQUAL "${PREV_GIT_VERSION}")
    message(STATUS "Updating git_version.cpp")
    file(WRITE ${CMAKE_CURRENT_BINARY_DIR}/git_version.cpp "${CURR_GIT_VERSION}")
  else()
    message(STATUS "No changes to git_version.cpp required")
  endif()
else()
  # Create git_version.cpp if it doesn't exist yet
  file(WRITE ${CMAKE_CURRENT_BINARY_DIR}/git_version.cpp "${CURR_GIT_VERSION}")
endif()
