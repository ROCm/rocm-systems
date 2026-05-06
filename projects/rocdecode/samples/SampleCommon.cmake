# Common CMake configuration for rocdecode samples
# Include this file in sample CMakeLists.txt to get platform-specific settings

# Platform detection
if(WIN32)
  set(IS_WINDOWS TRUE)
  set(IS_LINUX FALSE)
else()
  set(IS_WINDOWS FALSE)
  set(IS_LINUX TRUE)
endif()

# ROCM Path
if(DEFINED ENV{ROCM_PATH})
  set(ROCM_PATH $ENV{ROCM_PATH} CACHE PATH "Default ROCm installation path")
elseif(ROCM_PATH)
  message("-- INFO:ROCM_PATH Set -- ${ROCM_PATH}")
else()
  if(IS_WINDOWS)
    set(ROCM_PATH c:/opt/rocm CACHE PATH "Default ROCm installation path")
  else()
    set(ROCM_PATH /opt/rocm CACHE PATH "Default ROCm installation path")
  endif()
  message("-- INFO: Using default ROCM_PATH -- ${ROCM_PATH}")
endif()

# Set AMD Clang as compiler (MUST be before project() command)
if (NOT DEFINED CMAKE_CXX_COMPILER)
  if(IS_WINDOWS)
    set(CMAKE_C_COMPILER ${ROCM_PATH}/lib/llvm/bin/amdclang.exe)
    set(CMAKE_CXX_COMPILER ${ROCM_PATH}/lib/llvm/bin/amdclang++.exe)
  else()
    set(CMAKE_C_COMPILER ${ROCM_PATH}/lib/llvm/bin/amdclang)
    set(CMAKE_CXX_COMPILER ${ROCM_PATH}/lib/llvm/bin/amdclang++)
  endif()
endif()

# C++ Standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED On)
set(CMAKE_CXX_EXTENSIONS ON)

# Build type setup
set(DEFAULT_BUILD_TYPE "Release")
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE "${DEFAULT_BUILD_TYPE}" CACHE STRING "Default Build Type" FORCE)
  set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS "Debug" "Release")
endif()

# Compiler flags
if(MSVC)
  # MSVC-specific flags
  if(CMAKE_BUILD_TYPE MATCHES Debug)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /Od /Zi")
  else()
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /O2 /DNDEBUG")
  endif()
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /W3")
  # Enable correct __cplusplus macro value for C++17 detection
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /Zc:__cplusplus")
else()
  # Clang/GCC flags
  if(CMAKE_BUILD_TYPE MATCHES Debug)
    if(IS_LINUX)
      set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O0 -gdwarf-4")
    else()
      set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O0 -g")
    endif()
  else()
    if(IS_LINUX)
      set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -DNDEBUG -fPIC")
    else()
      set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -DNDEBUG")
    endif()
  endif()
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall")
endif()

# CMake module and prefix paths
list(APPEND CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/../../cmake)
list(APPEND CMAKE_PREFIX_PATH ${ROCM_PATH}/lib/cmake)
