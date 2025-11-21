#-----------------------------------------------------------------------------
# Copyright (c) 2015 - 2023 Advanced Micro Devices, Inc.  All rights reserved.
#-----------------------------------------------------------------------------

# amdhsacode library
#
# This file is expected to be included from top-level CMakeLists.txt.
#
# Dependencies:
# - Compiler definitions
# - elf library
# - hsail library
# - sp3 library
#
# Defines:
# - amdhsacode library and target include directories

set(USE_AMD_LIBELF "no" CACHE STRING "Do not use AMD LIBELF by default")
set(NO_SI_SP3      "no" CACHE STRING "Disable using SP3")

file(GLOB sources *.cpp *.hpp)

add_library(amdhsacode STATIC ${sources})
set_target_properties(amdhsacode PROPERTIES
  MSVC_RUNTIME_LIBRARY          "MultiThreaded$<$<CONFIG:Debug>:Debug>"
  POSITION_INDEPENDENT_CODE     ON
)

if(CMAKE_CXX_COMPILER_ID MATCHES "^(GNU|(Apple)?Clang)$")
  target_compile_options(amdhsacode PRIVATE
    -Werror
    -Wno-inconsistent-missing-override
  )
endif()

set(LIBELF_LIB)
if(${USE_AMD_LIBELF} STREQUAL "yes")
  target_compile_definitions(amdhsacode PRIVATE AMD_LIBELF)
  target_include_directories(amdhsacode PUBLIC ${HSAIL_ELFTOOLCHAIN_DIR}/common)
  if(WIN32)
    target_include_directories(amdhsacode PUBLIC ${HSAIL_ELFTOOLCHAIN_DIR}/common/win32)
  endif()
  target_include_directories(amdhsacode PUBLIC ${HSAIL_ELFTOOLCHAIN_DIR}/libelf)
  set(LIBELF_LIB oclelf)
  message(STATUS "Using AMD LIBELF")
else()
  find_package(LibElf REQUIRED)
  set(LIBELF_LIB elf)
  message(STATUS "Using SYSTEM LIBELF")
endif()

target_include_directories(amdhsacode PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_include_directories(amdhsacode PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/../../include)
target_include_directories(amdhsacode PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/../../hsail-tools/libHSAIL)
if(${NO_SI_SP3} STREQUAL "no")
  target_include_directories(amdhsacode PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../../../Chip/sp3)
endif()
#if defined(GFX102_BUILD)
target_compile_definitions(amdhsacode PRIVATE GFX102_BUILD)
#endif
#if defined(GFX11_BUILD)
target_compile_definitions(amdhsacode PRIVATE GFX11_BUILD)
#endif
#if defined(GFX115_BUILD)
target_compile_definitions(amdhsacode PRIVATE GFX115_BUILD)
#endif
#if defined(GFX12_BUILD)
target_compile_definitions(amdhsacode PRIVATE GFX12_BUILD)
#endif

if(${NO_SI_SP3} STREQUAL "yes")
  target_compile_definitions(amdhsacode PRIVATE NO_SI_SP3)
endif()

target_link_libraries(amdhsacode PRIVATE ${LIBELF_LIB})

if(${NO_SI_SP3} STREQUAL "no")
  set(SC_BUILD_SP3 ON)
  if(SC_BUILD_SP3)
    target_link_libraries(amdhsacode PUBLIC sp3)
  endif()
endif()
