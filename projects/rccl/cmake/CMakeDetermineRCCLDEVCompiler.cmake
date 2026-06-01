# CMakeDetermineRCCLDEVCompiler.cmake
# Locate the rccl-device-compile driver for the RCCLDEV custom language.

if(NOT CMAKE_RCCLDEV_COMPILER)
  find_program(CMAKE_RCCLDEV_COMPILER
    NAMES rccl-device-compile
    HINTS "${CMAKE_CURRENT_LIST_DIR}/../tools"
    DOC "RCCL device compilation driver"
  )
endif()

if(NOT CMAKE_RCCLDEV_COMPILER)
  message(FATAL_ERROR "Cannot find rccl-device-compile driver. "
    "Set CMAKE_RCCLDEV_COMPILER or ensure it is in tools/.")
endif()

mark_as_advanced(CMAKE_RCCLDEV_COMPILER)

set(CMAKE_RCCLDEV_COMPILER_ID "RCCLDEV")
set(CMAKE_RCCLDEV_COMPILER_LOADED 1)
set(CMAKE_RCCLDEV_COMPILER_ENV_VAR "RCCLDEV")

# RCCLDEV wraps amdclang++ (the CXX compiler), so it shares the same implicit
# include directories.  Declaring them here lets CMake automatically suppress
# paths like /usr/include from -isystem flags, avoiding #include_next breakage
# in the GCC C++ standard library headers.
set(CMAKE_RCCLDEV_IMPLICIT_INCLUDE_DIRECTORIES "${CMAKE_CXX_IMPLICIT_INCLUDE_DIRECTORIES}")

# Write the compiler identification file for subsequent cmake runs.
file(WRITE "${CMAKE_PLATFORM_INFO_DIR}/CMakeRCCLDEVCompiler.cmake"
  "set(CMAKE_RCCLDEV_COMPILER \"${CMAKE_RCCLDEV_COMPILER}\")\n"
  "set(CMAKE_RCCLDEV_COMPILER_LOADED 1)\n"
  "set(CMAKE_RCCLDEV_COMPILER_ID \"RCCLDEV\")\n"
  "set(CMAKE_RCCLDEV_COMPILER_ENV_VAR \"RCCLDEV\")\n"
  "set(CMAKE_RCCLDEV_IMPLICIT_INCLUDE_DIRECTORIES \"${CMAKE_CXX_IMPLICIT_INCLUDE_DIRECTORIES}\")\n"
)
