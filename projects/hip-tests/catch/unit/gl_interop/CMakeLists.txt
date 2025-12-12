# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
#
# SPDX-License-Identifier: MIT

set(TEST_SRC
    hipGLGetDevices.cc
    hipGLContextSwitch.cc
    hipGLImageDataVerification.cc
    hipGraphicsGLRegisterBuffer.cc
    hipGraphicsGLRegisterImage.cc
    hipGraphicsMapResources.cc
    hipGraphicsSubResourceGetMappedArray.cc
    hipGraphicsResourceGetMappedPointer.cc
    hipGraphicsUnmapResources.cc
    hipGraphicsUnregisterResource.cc
)

# Find OpenGL with optional EGL support
find_package(OpenGL COMPONENTS OpenGL OPTIONAL_COMPONENTS EGL)
message(STATUS "OpenGL_FOUND: ${OpenGL_FOUND}")

if(NOT OpenGL_FOUND OR NOT TARGET OpenGL::GL)
  message(STATUS "OpenGL not found or OpenGL::GL target unavailable, "
                 "OpenGL interop tests not enabled.")
  return()
endif()

if (OpenGL_EGL_FOUND)
  message(STATUS "OpenGL::EGL: Found")
endif()

# Handle GLUT - env vars first, then find_package
set(USE_ENV_GLUT FALSE)
if (DEFINED ENV{GLUT_glut_LIBRARY} AND
    DEFINED ENV{GLUT_INCLUDE_DIR})
  if (NOT "$ENV{GLUT_glut_LIBRARY}" STREQUAL "" AND
      NOT "$ENV{GLUT_INCLUDE_DIR}" STREQUAL "")
    if (EXISTS "$ENV{GLUT_glut_LIBRARY}" AND
        EXISTS "$ENV{GLUT_INCLUDE_DIR}")
      set(USE_ENV_GLUT TRUE)
    else()
      message(WARNING "GLUT env vars invalid, using find_package")
    endif()
  endif()
endif()

if (USE_ENV_GLUT)
  message(STATUS "Using GLUT from environment variables")
  message(STATUS "  GLUT_glut_LIBRARY: $ENV{GLUT_glut_LIBRARY}")
  message(STATUS "  GLUT_INCLUDE_DIR: $ENV{GLUT_INCLUDE_DIR}")

  set(GLUT_glut_LIBRARY "$ENV{GLUT_glut_LIBRARY}"
      CACHE FILEPATH "Path to GLUT library")
  set(GLUT_INCLUDE_DIR "$ENV{GLUT_INCLUDE_DIR}"
      CACHE PATH "Path to GLUT include directory")

  add_library(GLUT::GLUT UNKNOWN IMPORTED)
  set_target_properties(GLUT::GLUT PROPERTIES
      IMPORTED_LOCATION "${GLUT_glut_LIBRARY}"
      INTERFACE_INCLUDE_DIRECTORIES "${GLUT_INCLUDE_DIR}"
  )
else()
  message(STATUS "Searching for GLUT via find_package...")
  find_package(GLUT)
  if (NOT GLUT_FOUND)
    message(STATUS "GLUT not found, OpenGL interop tests not enabled.")
    return()
  endif()
  message(STATUS "GLUT_FOUND: ${GLUT_FOUND}")
endif()

# GLEW is required on Windows for OpenGL extension access
# On Linux, extensions are available via native headers
# Handle GLEW - env vars first, then find_package
set(USE_ENV_GLEW FALSE)
if (DEFINED ENV{GLEW_SHARED_LIBRARY_RELEASE} AND
    DEFINED ENV{GLEW_INCLUDE_DIR})
  if (NOT "$ENV{GLEW_SHARED_LIBRARY_RELEASE}" STREQUAL "" AND
      NOT "$ENV{GLEW_INCLUDE_DIR}" STREQUAL "")
    if (EXISTS "$ENV{GLEW_SHARED_LIBRARY_RELEASE}" AND
        EXISTS "$ENV{GLEW_INCLUDE_DIR}")
      set(USE_ENV_GLEW TRUE)
    else()
      message(WARNING "GLEW env vars invalid, using find_package")
    endif()
  endif()
endif()

if (USE_ENV_GLEW)
  message(STATUS "Using GLEW from environment variables")
  message(STATUS "  GLEW_SHARED_LIBRARY_RELEASE: $ENV{GLEW_SHARED_LIBRARY_RELEASE}")
  message(STATUS "  GLEW_INCLUDE_DIR: $ENV{GLEW_INCLUDE_DIR}")

  set(GLEW_SHARED_LIBRARY_RELEASE "$ENV{GLEW_SHARED_LIBRARY_RELEASE}"
      CACHE FILEPATH "Path to GLEW library")
  set(GLEW_INCLUDE_DIR "$ENV{GLEW_INCLUDE_DIR}"
      CACHE PATH "Path to GLEW include directory")
  set(GLEW_FOUND TRUE)
  add_library(GLEW::GLEW UNKNOWN IMPORTED)
  set_target_properties(GLEW::GLEW PROPERTIES
      IMPORTED_LOCATION "${GLEW_SHARED_LIBRARY_RELEASE}"
      INTERFACE_INCLUDE_DIRECTORIES "${GLEW_INCLUDE_DIR}"
  )
else()
  message(STATUS "Searching for GLEW via find_package...")
  find_package(GLEW)
  message(STATUS "GLEW_FOUND: ${GLEW_FOUND}")
  if (NOT GLEW_FOUND)
    message(STATUS "GLEW not found.")
    message(STATUS "Do you add -DGLEW_INCLUDE_DIR=path-to-include -DGLEW_SHARED_LIBRARY_RELEASE=path-to-x64/glew32.lib to CMake?")
  endif()
endif()

if(WIN32 AND NOT GLEW_FOUND)
  message(STATUS "GLEW required on Windows but not found, "
                 "GL interop tests not enabled.")
  return()
endif()

# Create test executable
hip_add_exe_to_target(NAME GLInteropTest
                      TEST_SRC ${TEST_SRC}
                      TEST_TARGET_NAME build_tests)

# Link dependencies
target_link_libraries(GLInteropTest OpenGL::GL GLUT::GLUT)

if (OpenGL_EGL_FOUND)
  target_link_libraries(GLInteropTest OpenGL::EGL)
  target_compile_definitions(GLInteropTest PRIVATE USE_EGL)
endif()

if (GLEW_FOUND)
  target_link_libraries(GLInteropTest GLEW::GLEW)
  target_compile_definitions(GLInteropTest PRIVATE USE_GLEW)
endif()
