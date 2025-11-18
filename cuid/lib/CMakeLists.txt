cmake_minimum_required(VERSION 3.10)
project(cuid_lib)

set(CMAKE_CXX_STANDARD 14)

find_package(OpenSSL REQUIRED)

set(CUID_SOURCES
    cuid.cc
    cuid_util.cc
    cuid_device.cc
    cuid_device_manager.cc
    cuid_cpu.cc
    cuid_gpu.cc
    hmac.cc
    cuid_lib_loader.cc
    pci_util.cc
)

set(CUID_HEADERS
    cuid.h
    cuid_util.h
    cuid_device.h
    cuid_device_manager.h
    cuid_cpu.h
    cuid_gpu.h
    cuid_nic.h
    cuid_platform.h
    hmac.h
    cuid_lib_loader.h
    pci_util.h
)

add_library(cuid_static STATIC ${CUID_SOURCES} ${CUID_HEADERS})
add_library(cuid_shared SHARED ${CUID_SOURCES} ${CUID_HEADERS})

target_include_directories(cuid_static PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_include_directories(cuid_shared PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
