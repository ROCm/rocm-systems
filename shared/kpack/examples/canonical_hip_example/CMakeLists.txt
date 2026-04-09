cmake_minimum_required(VERSION 3.21)
project(CanonicalHIPExample LANGUAGES CXX)

# Enable HIP language early
enable_language(HIP)

# Find HIP
find_package(hip REQUIRED)

# Set GPU architectures (can be overridden via -DGPU_TARGETS)
if(NOT DEFINED GPU_TARGETS)
    set(GPU_TARGETS "gfx1100;gfx1201" CACHE STRING "GPU architectures to build for")
endif()

message(STATUS "Building for GPU targets: ${GPU_TARGETS}")

# (a) Standalone executable with embedded kernel
add_executable(standalone standalone.hip)
target_link_libraries(standalone hip::host)
set_target_properties(standalone PROPERTIES
    HIP_ARCHITECTURES "${GPU_TARGETS}"
)

# (b) Shared library with host API that runs a kernel
add_library(vector_lib SHARED vector_lib.hip)
target_link_libraries(vector_lib PUBLIC hip::host)
set_target_properties(vector_lib PROPERTIES
    HIP_ARCHITECTURES "${GPU_TARGETS}"
    PUBLIC_HEADER vector_lib.h
)

# (c) Client executable that invokes the shared library API
add_executable(client client.cpp)
target_link_libraries(client vector_lib)
target_include_directories(client PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})

# Installation rules
install(TARGETS standalone client vector_lib
    RUNTIME DESTINATION bin
    LIBRARY DESTINATION lib
    PUBLIC_HEADER DESTINATION include
)
