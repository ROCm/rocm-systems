# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
#
# SPDX-License-Identifier: MIT

set(OCLTST_DIR ${CMAKE_CURRENT_SOURCE_DIR})
set(OCLTST_INSTALL_DIR ${CMAKE_INSTALL_BINDIR})

find_package(OpenGL)
find_package(GLEW)

add_subdirectory(module/common)
add_subdirectory(env)
if(OPENGL_FOUND AND GLEW_FOUND)
    add_subdirectory(module/gl)
endif()
add_subdirectory(module/perf)
add_subdirectory(module/runtime)
