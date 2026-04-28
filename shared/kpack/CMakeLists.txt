cmake_minimum_required(VERSION 3.20)
project(rocm-kpack VERSION 0.1.0 LANGUAGES CXX)

# Options
option(BUILD_RUNTIME "Build C++ runtime library" ON)
option(BUILD_TESTING "Build tests" ON)

# C++17 standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Find dependencies
if(BUILD_RUNTIME)
    find_package(msgpack-cxx REQUIRED)
    find_package(zstd REQUIRED)
    if(TARGET zstd::libzstd_shared)
        set(zstd_libzstd zstd::libzstd_shared)
    elseif(TARGET zstd::libzstd_static)
        set(zstd_libzstd zstd::libzstd_static)
    else()
        message(
            FATAL_ERROR
            "find_package(zstd) succeeded but no known target found"
        )
    endif()
endif()

# Enable testing
if(BUILD_TESTING)
    enable_testing()
    find_package(GTest REQUIRED)
endif()

# Add runtime subdirectory
if(BUILD_RUNTIME)
    add_subdirectory(runtime)
endif()
