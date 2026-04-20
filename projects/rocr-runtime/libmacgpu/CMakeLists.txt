################################################################################
# libmacgpu — macOS userspace GPU driver shim.
#
# Built only on Darwin. Linked statically into libhsa-runtime64.dylib by the
# hsa-runtime top-level CMakeLists; nothing else consumes it. The public
# header (include/macgpu.h) is added to the hsa-runtime target's include path.
################################################################################

cmake_minimum_required(VERSION 3.16)
project(macgpu CXX)

if(NOT APPLE)
    message(FATAL_ERROR "libmacgpu is only built on Darwin")
endif()

add_library(macgpu STATIC
    src/macgpu_client.cpp)

target_include_directories(macgpu
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>)

target_compile_features(macgpu PRIVATE cxx_std_17)
set_target_properties(macgpu PROPERTIES POSITION_INDEPENDENT_CODE ON)

# IOKit.framework carries IOServiceOpen, IOConnectCallMethod,
# IOConnectMapMemory64, etc. No linker flags needed beyond -framework.
find_library(IOKIT_FRAMEWORK IOKit)
if(NOT IOKIT_FRAMEWORK)
    message(FATAL_ERROR "IOKit.framework not found — libmacgpu requires Darwin")
endif()
target_link_libraries(macgpu PUBLIC ${IOKIT_FRAMEWORK})
