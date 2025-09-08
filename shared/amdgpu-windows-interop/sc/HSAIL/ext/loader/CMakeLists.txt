#-----------------------------------------------------------------------------
# Copyright (c) 2015 - 2023 Advanced Micro Devices, Inc.  All rights reserved.
#-----------------------------------------------------------------------------

# loader library
#
# This file is expected to be included from top-level CMakeLists.txt.
#
# Dependencies:
# - Compiler definitions
# - amdhsacode library
#
# Defines:
# - amdhsaloader library and target include directories

file(GLOB sources *.cpp *.hpp)
add_library(amdhsaloader STATIC ${sources})
set_target_properties(amdhsaloader PROPERTIES
  MSVC_RUNTIME_LIBRARY          "MultiThreaded$<$<CONFIG:Debug>:Debug>"
  POSITION_INDEPENDENT_CODE     ON
)

if(CMAKE_CXX_COMPILER_ID MATCHES "^(GNU|(Apple)?Clang)$")
  target_compile_options(amdhsaloader PRIVATE
    -Werror
    -Wno-inconsistent-missing-override
  )
endif()

target_include_directories(amdhsaloader PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(amdhsaloader amdhsacode)
