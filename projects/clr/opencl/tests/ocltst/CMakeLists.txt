set(OCLTST_DIR ${CMAKE_CURRENT_SOURCE_DIR})

if (WIN32)
    set(OCLTST_INSTALL_DIR "tests/ocltst")
else()
    set(OCLTST_INSTALL_DIR "share/opencl/ocltst")
endif()

find_package(OpenGL)
find_package(GLEW)

add_subdirectory(module/common)
add_subdirectory(env)
if(OPENGL_FOUND AND GLEW_FOUND)
    add_subdirectory(module/gl)
endif()
add_subdirectory(module/perf)
add_subdirectory(module/runtime)
