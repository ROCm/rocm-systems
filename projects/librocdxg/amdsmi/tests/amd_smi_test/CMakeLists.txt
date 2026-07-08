option(INSTALL_GTEST "Install GTest" OFF)

# Help tests find libraries at runtime
set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -Wl,--enable-new-dtags")
set(CMAKE_INSTALL_RPATH "\$ORIGIN:\$ORIGIN/../../../lib"
    CACHE STRING "RUNPATH for tests. Helps find libgtest.so and libamd_smi.so")

if(POLICY CMP0135)
    cmake_policy(SET CMP0135 NEW)
endif()

# Always fetch and build googletest as static to avoid runtime dependency on libgtest.so.
include(FetchContent)
FetchContent_Declare(
    googletest
    # Originally mirrored from: https://github.com/google/googletest/releases/download/v1.16.0/googletest-1.16.0.tar.gz
    URL https://rocm-third-party-deps.s3.us-east-2.amazonaws.com/googletest-1.16.0.tar.gz
    URL_HASH SHA256=78c676fc63881529bf97bf9d45948d905a66833fbfa5318ea2cd7478cb98f399
)
set(_saved_BUILD_SHARED_LIBS "${BUILD_SHARED_LIBS}")
set(BUILD_SHARED_LIBS OFF)
FetchContent_MakeAvailable(googletest)
set(BUILD_SHARED_LIBS "${_saved_BUILD_SHARED_LIBS}")

enable_testing()

if(WIN32)
    message("amd_smi library test suite is not supported on Windows platform")
    return()
endif()

#
# Print out the build configuration being used:
#
#   Build Src directory
#   Build Binary directory
#   Build Type: Debug Vs Release
#   Compiler Version, etc
#
message("")
message("Build Configuration:")
message("-----------BuildType: " ${CMAKE_BUILD_TYPE})
message("------------Compiler: " ${CMAKE_CXX_COMPILER})
message("-------------Version: " ${CMAKE_CXX_COMPILER_VERSION})
message("--------Proj Src Dir: " ${PROJECT_SOURCE_DIR})
message("--------Proj Bld Dir: " ${PROJECT_BINARY_DIR})
message("--------Proj Lib Dir: " ${PROJECT_BINARY_DIR}/lib)
message("--------Proj Exe Dir: " ${PROJECT_BINARY_DIR}/bin)
message("")

# Other source directories
aux_source_directory(${CMAKE_CURRENT_SOURCE_DIR}/functional functionalSources)
list(REMOVE_ITEM functionalSources
    "${CMAKE_CURRENT_SOURCE_DIR}/functional/mutual_exclusion.cc"
    "${CMAKE_CURRENT_SOURCE_DIR}/functional/dynamic_metrics_test.cc"
    "${CMAKE_CURRENT_SOURCE_DIR}/functional/init_shutdown_refcount.cc"
)

set(TEST "amdsmitst")

# Source files
aux_source_directory(${CMAKE_CURRENT_SOURCE_DIR} tstSources)
# Header file include path
include_directories(${TEST} 
${CMAKE_CURRENT_SOURCE_DIR}/.. 
${PROJECT_SOURCE_DIR}/include 
)

# Build rules
add_executable(${TEST} ${tstSources} ${functionalSources})
target_link_libraries(${TEST} ${AMD_SMI} GTest::gtest c stdc++ pthread ${AMDSMINIC_PATH} ${FILESYSTEM_LIB})

# Install tests
install(
    TARGETS ${TEST}
    DESTINATION ${SHARE_INSTALL_PREFIX}/tests
    COMPONENT ${TESTS_COMPONENT})

install(
    FILES amdsmitst.exclude
    DESTINATION ${SHARE_INSTALL_PREFIX}/tests
    COMPONENT ${TESTS_COMPONENT})
