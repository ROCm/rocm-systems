# AegisBit tests

#===----------------------------------------------------------------------===//
# GoogleTest Setup
#===----------------------------------------------------------------------===//

# Try to find system GoogleTest first
find_package(GTest QUIET)

if(NOT GTEST_FOUND)
  message(STATUS "System GTest not found, fetching from GitHub...")
  include(FetchContent)
  FetchContent_Declare(
    googletest
    GIT_REPOSITORY https://github.com/google/googletest.git
    GIT_TAG v1.14.0
  )
  # Prevent GoogleTest from overriding compiler/linker options
  set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
  set(INSTALL_GTEST OFF CACHE BOOL "" FORCE)
  FetchContent_MakeAvailable(googletest)
  set(GTEST_FOUND TRUE)
endif()

message(STATUS "GoogleTest enabled for testing")

#===----------------------------------------------------------------------===//
# Test Subdirectories
#===----------------------------------------------------------------------===//

# Unit tests
add_subdirectory(unit)

# Integration tests (require GPU or GPU-specific encoding)
add_subdirectory(integration)

# Fuzz tests (require Clang and -DENABLE_FUZZ_TESTS=ON)
add_subdirectory(fuzz)
