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
find_package(GLEW)
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
