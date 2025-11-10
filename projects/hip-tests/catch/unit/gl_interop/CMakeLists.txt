set(TEST_SRC
    hipGLGetDevices.cc
    hipGraphicsGLRegisterBuffer.cc
    hipGraphicsGLRegisterImage.cc
    hipGraphicsMapResources.cc
    hipGraphicsSubResourceGetMappedArray.cc
    hipGraphicsResourceGetMappedPointer.cc
    hipGraphicsUnmapResources.cc
    hipGraphicsUnregisterResource.cc
)

find_package(OpenGL COMPONENTS OpenGL)
message(STATUS "OpenGL_FOUND: ${OpenGL_FOUND}")
if(NOT OpenGL_FOUND)
  message(STATUS "OpenGL not found, OpenGL interop tests not enabled.")
  return()
endif()

find_package(OpenGL OPTIONAL_COMPONENTS EGL)
if (OpenGL_EGL_FOUND)
  message(STATUS "OpenGL::EGL: Found")
endif()

if (NOT DEFINED ENV{GLUT_glut_LIBRARY} OR "$ENV{GLUT_glut_LIBRARY}" STREQUAL ""
    OR NOT DEFINED ENV{GLUT_INCLUDE_DIR} OR "$ENV{GLUT_INCLUDE_DIR}" STREQUAL "")
  message(STATUS "GLUT_glut_LIBRARY or GLUT_INCLUDE_DIR environment variable not set, searching for GLUT...")
  find_package(GLUT)
  if (NOT GLUT_FOUND)
    message(STATUS "GLUT not found, OpenGL interop tests not enabled.")
    return()
  endif()
  message(STATUS "GLUT_FOUND: ${GLUT_FOUND}")
else()
  message(STATUS "Using GLUT paths from environment variables.")
  message(STATUS "GLUT_glut_LIBRARY: $ENV{GLUT_glut_LIBRARY}")
  message(STATUS "GLUT_INCLUDE_DIR: $ENV{GLUT_INCLUDE_DIR}")

  if (NOT EXISTS "$ENV{GLUT_glut_LIBRARY}")
    message(FATAL_ERROR "GLUT_glut_LIBRARY path '$ENV{GLUT_glut_LIBRARY}' does not exist!")
  endif()
  if (NOT EXISTS "$ENV{GLUT_INCLUDE_DIR}")
    message(FATAL_ERROR "GLUT_INCLUDE_DIR path '$ENV{GLUT_INCLUDE_DIR}' does not exist!")
  endif()

  add_library(GLUT::GLUT UNKNOWN IMPORTED)
  set_target_properties(GLUT::GLUT PROPERTIES
      IMPORTED_LOCATION "$ENV{GLUT_glut_LIBRARY}"
      INTERFACE_INCLUDE_DIRECTORIES "$ENV{GLUT_INCLUDE_DIR}"
  )
endif()

find_package(GLEW)

hip_add_exe_to_target(NAME GLInteropTest
                      TEST_SRC ${TEST_SRC}
                      TEST_TARGET_NAME build_tests
                      COMPILE_OPTIONS -std=c++17)

target_link_libraries(GLInteropTest OpenGL::GL GLUT::GLUT)

if (OpenGL_EGL_FOUND)
  target_link_libraries(GLInteropTest OpenGL::EGL)
endif()

if (GLEW_FOUND)
  target_link_libraries(GLInteropTest GLEW::GLEW)
  target_compile_definitions(GLInteropTest PRIVATE USE_GLEW)
endif()
