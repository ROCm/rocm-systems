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

find_package(OpenGL COMPONENTS OpenGL EGL)
message(STATUS "OpenGL_FOUND: ${OpenGL_FOUND}")
if(NOT OpenGL_FOUND)
  message(STATUS "OpenGL not found, OpenGL interop tests not enabled.")
  return()
endif()

find_package(GLUT)
message(STATUS "GLUT_FOUND: ${GLUT_FOUND}")
if(NOT GLUT_FOUND)
  message(STATUS "GLUT not found, OpenGL interop tests not enabled.")
  return()
endif()

hip_add_exe_to_target(NAME GLInteropTest
                      TEST_SRC ${TEST_SRC}
                      TEST_TARGET_NAME build_tests
                      COMPILE_OPTIONS -std=c++17)
target_link_libraries(GLInteropTest OpenGL::GL OpenGL::EGL GLUT::GLUT)