##
 #######################################################################################################################
 #
 #  Modifications Copyright© 2019 Advanced Micro Devices, Inc. All rights reserved.
 #
 #######################################################################################################################

cmake_minimum_required(VERSION 3.1...3.21)

project(MetroHash VERSION 1.0.0 LANGUAGES CXX)

option(METROHASH_ENABLE_WERROR "Build with -Werror enabled" OFF)

add_library(metrohash STATIC "")

target_include_directories(metrohash PUBLIC src)

target_sources(metrohash PRIVATE src/metrohash64.cpp
                                 src/metrohash128.cpp)


set_target_properties(metrohash PROPERTIES CXX_STANDARD              11
                                           CXX_STANDARD_REQUIRED     ON
                                           CXX_EXTENSIONS            OFF
                                           POSITION_INDEPENDENT_CODE ON)


if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")

    if(METROHASH_ENABLE_WERROR)
        target_compile_options(metrohash PRIVATE -Werror)
    endif()

    # [GCC] Exceptions
    #   https://gcc.gnu.org/onlinedocs/libstdc++/manual/using_exceptions.html
    #
    # [GCC] Options Controlling C++ Dialect
    #   https://gcc.gnu.org/onlinedocs/gcc-8.1.0/gcc/C_002b_002b-Dialect-Options.html
    target_compile_options(metrohash PRIVATE
        -fno-exceptions  # Disable exception handling support.
        -fno-rtti)       # Disable run-time type information support.

    # [GCC] Options to Request or Suppress Warnings
    #   https://gcc.gnu.org/onlinedocs/gcc-8.1.0/gcc/Warning-Options.html
    target_compile_options(metrohash PRIVATE
        -Wall
        -Wextra
        -Wpedantic)

elseif(CMAKE_CXX_COMPILER_ID STREQUAL "MSVC")

    # [MSVC] Exception Handling Model
    #
    # [MSVC] Enable Run-Time Type Information
    #
    # [MSVC] Buffer Security Check
    target_compile_options(metrohash PRIVATE
        /EHsc # Catches only C++ exceptions and assumes
              # functions declared as extern "C" never throw a C++ exception.
        /GR-  # Disables run-time type information.
        /GS-) # Disables detection of buffer overruns.

    # [MSVC] Warning Level
    target_compile_options(metrohash PRIVATE
        /W4  # Enable warning level 4.
        /WX) # Treat warnings as errors.

else()
    message(FATAL_ERROR "Compiler ${CMAKE_CXX_COMPILER_ID} is not supported!")
endif()
