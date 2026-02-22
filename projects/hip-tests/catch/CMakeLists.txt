cmake_minimum_required(VERSION 3.20)

project(hiptests LANGUAGES C CXX HIP)

option(ENABLE_ADDRESS_SANITIZER "Option to enable ASAN build" OFF)
option(BUILD_SHARED_LIBS "Option for testing shared libraries" ON)
option(BUILD_UNIT_TESTS "Build unit tests" ON)
option(BUILD_PERF_TESTS "Build perf tests" OFF)
option(BUILD_STRESS_TESTS "Build stress tests" OFF)
option(TEST_CLOCK_CYCLE "Option to use clock64" OFF)
option(RTC_TESTING "Run tests using HIP RTC to compile the kernels" OFF)
option(ENABLE_SPIRV "Build hip-tests for SPIRV" OFF)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE "Release")
endif()

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_HIP_STANDARD 17)
set(CMAKE_HIP_STANDARD_REQUIRED ON)
set(CMAKE_INSTALL_RPATH "$ORIGIN;$ORIGIN/../../lib")
set(CMAKE_INSTALL_RPATH_USE_LINK_PATH TRUE)

find_package(hip REQUIRED)
find_package(hiprtc)
find_package(Git)
find_package(Python 3 COMPONENTS Interpreter REQUIRED)
find_program(HIPCONFIG_EXEC hipconfig REQUIRED)

# hip root dir is 3 levels down the hip-config.cmake file path
set(HIP_PATH ${hip_DIR}/../../../)
if(NOT DEFINED ROCM_PATH)
    set(ROCM_PATH "/opt/rocm")
    if(NOT EXISTS "${ROCM_PATH}")
        set(ROCM_PATH ${HIP_PATH})
    endif()
endif()

if(UNIX)
    if(NOT ENABLE_SPIRV)
        find_program(ROCM_AGENT_ENUMERATOR_EXEC rocm_agent_enumerator REQUIRED)
    endif()
    find_program(GCC_EXEC gcc)
    find_program(GXX_EXEC g++)
else() # Win32
    find_program(LLVM_RC_EXEC NAMES llvm-rc llvm-rc.exe
                 PATHS ${HIP_PATH} ${HIP_PATH}../lc/bin ${ROCM_PATH})
endif()

find_program(CLANG_CPP_EXEC NAMES clang-cpp clang-cpp.exe
             PATHS ${HIP_PATH} ${HIP_PATH}../lc/bin ${ROCM_PATH}/llvm/bin)
message(STATUS "HIP Lib path: ${HIP_LIB_INSTALL_DIR}")

# used by clang to find hip
set(HIP_PATH_OPT "--hip-path=${HIP_PATH}")

if(TEST_CLOCK_CYCLE)
    add_definitions(-DTEST_CLOCK_CYCLE)
endif()

# flag to generate standalone exe per src file.
message(STATUS "STANDALONE_TESTS : ${STANDALONE_TESTS}")

if(NOT WIN32)
    set(CPACK_SET_DESTDIR ON CACHE BOOL "Installer package will install hip catch to CMAKE_INSTALL_PREFIX instead of CPACK_PACKAGING_INSTALL_PREFIX")
endif()

# Check if platform is set
if(NOT HIP_PLATFORM STREQUAL "amd" AND NOT HIP_PLATFORM STREQUAL "nvidia")
    message(FATAL_ERROR "Unexpected HIP_PLATFORM: " ${HIP_PLATFORM})
endif()

execute_process(COMMAND ${HIPCONFIG_EXEC} --version
                OUTPUT_VARIABLE HIP_VERSION
                RESULT_VARIABLE result
                OUTPUT_STRIP_TRAILING_WHITESPACE)

if(result AND NOT result EQUAL 0)
  message(FATAL_ERROR "Failure trying to obtain HIP version via command: ${HIPCONFIG_EXEC} --version: ${result}")
endif()

if(NOT WIN32)
  if(NOT BUILD_SHARED_LIBS)
    list(APPEND CMAKE_PREFIX_PATH ${ROCM_PATH} ${ROCM_PATH}/llvm)
    find_package(amd_comgr)
    find_package(hsa-runtime64)
    link_libraries(hsa-runtime64::hsa-runtime64)
    link_libraries(amd_comgr)
    link_libraries(hiprtc-builtins)
  endif() # end BUILD_SHARED_LIBS
endif() # end win32

# Address sanitizer options
if(NOT DEFINED THEROCK_SANITIZER)
  set(THEROCK_SANITIZER)
endif()

if(ENABLE_ADDRESS_SANITIZER OR THEROCK_SANITIZER STREQUAL "ASAN" OR THEROCK_SANITIZER STREQUAL "HOST_ASAN")
    message(STATUS "Building catch tests with Address Sanitizer options")
    # If user is building hip-tests standalone (without TheRock) and wants to turn on sanitizer
    # We append the flags manually, otherwise we use TheRock's CXX flags which already should have
    # `-fsanitize=address` in it.
    if(ENABLE_ADDRESS_SANITIZER)
        # We support clang for this
        if (NOT CMAKE_CXX_COMPILER_ID STREQUAL "Clang" OR NOT CMAKE_HIP_COMPILER_ID STREQUAL "Clang")
            message(WARNING "CXX and HIP compiler are not Clang, sanitizer build might fail")
        endif()

        set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -fsanitize=address -fno-omit-frame-pointer -shared-libasan -g")
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fsanitize=address -fno-omit-frame-pointer -shared-libasan -g")
    endif()
    set(CMAKE_HIP_FLAGS "${CMAKE_HIP_FLAGS} -fsanitize=address -fno-omit-frame-pointer -shared-libasan -g")
    set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -fsanitize=address -shared-libasan")
endif()

string(REPLACE "." ";" VERSION_LIST ${HIP_VERSION})
list(GET VERSION_LIST 0 HIP_VERSION_MAJOR)
list(GET VERSION_LIST 1 HIP_VERSION_MINOR)
list(GET VERSION_LIST 2 HIP_VERSION_PATCH_GITHASH)
string(REPLACE "-" ";" VERSION_LIST ${HIP_VERSION_PATCH_GITHASH})
list(GET VERSION_LIST 0 HIP_VERSION_PATCH)
list(GET VERSION_LIST 1 HIP_VERSION_GITHASH)

if(DEFINED ENV{ROCM_LIBPATCH_VERSION})
   set(HIP_PACKAGING_VERSION_PATCH ${HIP_VERSION_PATCH}.$ENV{ROCM_LIBPATCH_VERSION})
else()
   set(HIP_PACKAGING_VERSION_PATCH ${HIP_VERSION_PATCH}-${HIP_VERSION_GITHASH})
endif()

find_package(Catch2 3.8.1 QUIET)
if(Catch2_FOUND)
    message(STATUS "Catch2 v3.8.1 found: ${Catch2_DIR}")
    # for packaging
    set(CATCH_ADD_TESTS_SCRIPT ${Catch2_DIR}/CatchAddTests.cmake)
    file(GLOB CATCH_LIBS "${Catch2_DIR}/../../*Catch2*")
else()
    # On Windows, ensure Catch2 uses the same MSVC runtime library and build configuration as our tests
    if(WIN32)
        set(BUILD_SHARED_LIBS OFF CACHE BOOL "Build static libraries" FORCE)
        set(CMAKE_MSVC_RUNTIME_LIBRARY "MultiThreaded$<$<CONFIG:Debug>:Debug>" CACHE STRING "MSVC runtime library" FORCE)
        if(CMAKE_BUILD_TYPE STREQUAL "Debug")
            set(win_disable_warnings -fms-runtime-lib=static_dbg)
        endif()
    endif()
    # Setup catch2 manually
    include(FetchContent)
    FetchContent_Declare(
        Catch2
        DOWNLOAD_EXTRACT_TIMESTAMP OFF
        URL "https://rocm-third-party-deps.s3.us-east-2.amazonaws.com/Catch2-3.8.1.tar.gz"
        URL_HASH SHA256=18b3f70ac80fccc340d8c6ff0f339b2ae64944782f8d2fca2bd705cf47cadb79
    )
    FetchContent_MakeAvailable(Catch2)
    # for packaging
    set(CATCH_ADD_TESTS_SCRIPT ${Catch2_SOURCE_DIR}/extras/CatchAddTests.cmake)
    set(CATCH_LIBS $<TARGET_FILE:Catch2>)
endif()

include(cmake/hip-tests.cmake)
include(CTest)

include_directories(
    "./include"
    "./kernels"
    "${CMAKE_CURRENT_LIST_DIR}/external/picojson"
)

if (RTC_TESTING)
    add_definitions(-DRTC_TESTING=ON)
endif()
add_definitions(-DKERNELS_PATH="${CMAKE_CURRENT_SOURCE_DIR}/kernels/")

set(CATCH_BUILD_DIR catch_tests)
set(HIP_TEST_CONFIG_BINARY_DIR ${CMAKE_CURRENT_BINARY_DIR}/${CATCH_BUILD_DIR}/config)
file(MAKE_DIRECTORY ${HIP_TEST_CONFIG_BINARY_DIR})
file(GLOB JSON_FILES "./hipTestMain/config/*.json")
foreach(json IN LISTS JSON_FILES)
    file(COPY ${json}
         DESTINATION ${HIP_TEST_CONFIG_BINARY_DIR})
endforeach()

if (WIN32)
  configure_file(catchProp_in_rc.in ${CMAKE_CURRENT_BINARY_DIR}/catchProp.rc @ONLY)

  # generates the .res files to be used by executables to populate the properties
  # expects LC folder with clang, llvm-rc to be present one level up of HIP
  execute_process(COMMAND ${LLVM_RC_EXEC} ${CMAKE_CURRENT_BINARY_DIR}/catchProp.rc
                  OUTPUT_VARIABLE RC_OUTPUT)
  set(PROP_RC ${CMAKE_CURRENT_BINARY_DIR})
  # When args to linker exceeds max chars.
  # msbuild writes args to a rsp file.
  # This is used to reference the obj file correctly
  SET(CMAKE_C_RESPONSE_FILE_LINK_FLAG "")
  SET(CMAKE_CXX_RESPONSE_FILE_LINK_FLAG "")
endif()

if(HIP_PLATFORM STREQUAL "amd")
    if(WIN32)
        set(win_disable_warnings ${win_disable_warnings} -Wno-ignored-attributes -Wno-microsoft-cast)
        # Windows for some reason does not honor the `CMAKE_CXX_STANDARD` we set initially in the file
        # Which results in errors about missing headers etc
        set(win_cxx_standard -std=c++17)
    endif()
    add_compile_options(${HIP_PATH_OPT} ${win_cxx_standard} -x hip -Wall -Wextra -Wvla -Wno-deprecated -Wno-option-ignored -Wno-unused-parameter ${win_disable_warnings})
    if(CMAKE_BUILD_TYPE STREQUAL "Release")
        add_compile_options(-O3)
    endif()
endif()

# Turn off CMAKE_HIP_ARCHITECTURES Feature if cmake version is 3.21+
if(CMAKE_VERSION VERSION_GREATER_EQUAL 3.21.0)
    set(CMAKE_HIP_ARCHITECTURES OFF)
endif()
message(STATUS "CMAKE HIP ARCHITECTURES: ${CMAKE_HIP_ARCHITECTURES}")

# Note to pass arch use format like -DOFFLOAD_ARCH_STR="--offload-arch=gfx900  --offload-arch=gfx906"
# having space at the start/end of OFFLOAD_ARCH_STR can cause build failures
# Identify the GPU Targets.
# This is done due to limitation of rocm_agent_enumerator
# While building test parallelly, rocm_agent_enumerator can fail and give out an empty target
# That results in hipcc building the test for gfx803 (the default target)
# preference to pass arch -
# OFFLOAD_ARCH_STR
# rocm_agent_enumerator
if(ENABLE_SPIRV)
    set(OFFLOAD_ARCH_STR "--offload-arch=amdgcnspirv")
elseif(DEFINED OFFLOAD_ARCH_STR)
    string(REPLACE "--offload-arch=" "" HIP_GPU_ARCH_LIST ${OFFLOAD_ARCH_STR})
elseif(DEFINED GPU_TARGETS)
    foreach(_hip_gpu_arch ${GPU_TARGETS})
        set(OFFLOAD_ARCH_STR "--offload-arch=${_hip_gpu_arch} ${OFFLOAD_ARCH_STR}")
    endforeach()
elseif(NOT DEFINED OFFLOAD_ARCH_STR
   AND EXISTS "${ROCM_AGENT_ENUMERATOR_EXEC}"
   AND HIP_PLATFORM STREQUAL "amd" AND UNIX)
    execute_process(COMMAND "${ROCM_PATH}/bin/rocm_agent_enumerator"
         OUTPUT_VARIABLE HIP_GPU_ARCH
         RESULT_VARIABLE ROCM_AGENT_ENUM_RESULT
         OUTPUT_STRIP_TRAILING_WHITESPACE)
    # Trim out gfx000
    string(REPLACE "gfx000\n" "" HIP_GPU_ARCH ${HIP_GPU_ARCH})
    if (NOT HIP_GPU_ARCH STREQUAL "")
        string(REGEX REPLACE "\n" ";" HIP_GPU_ARCH_LIST "${HIP_GPU_ARCH}")
        list(REMOVE_DUPLICATES  HIP_GPU_ARCH_LIST)
        list(LENGTH HIP_GPU_ARCH_LIST HIP_GPU_ARCH_LEN)
        set(OFFLOAD_ARCH_STR "")
        foreach(_hip_gpu_arch ${HIP_GPU_ARCH_LIST})
            set(OFFLOAD_ARCH_STR "--offload-arch=${_hip_gpu_arch} ${OFFLOAD_ARCH_STR}")
        endforeach()
    else()
        message(STATUS "ROCm Agent Enumerator found no valid architectures")
    endif()
endif()

# Trim any trailing spaces, this trips up the cxx function call
string(STRIP "${OFFLOAD_ARCH_STR}" OFFLOAD_ARCH_STR)

message(STATUS "Using offload arch string: ${OFFLOAD_ARCH_STR}")
if(DEFINED OFFLOAD_ARCH_STR)
    set(CMAKE_HIP_FLAGS "${CMAKE_HIP_FLAGS} ${OFFLOAD_ARCH_STR}")
endif()

# get hip-tests commit short hash
execute_process(COMMAND ${GIT_EXECUTABLE} rev-parse --short HEAD
  WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
  RESULT_VARIABLE git_result
  OUTPUT_VARIABLE git_output
  OUTPUT_STRIP_TRAILING_WHITESPACE)
if(git_result EQUAL 0)
  set(HIP_TESTS_GITHASH ${git_output})
endif()

set(OFFLOAD_ARCH_LIST ${OFFLOAD_ARCH_STR})
separate_arguments(OFFLOAD_ARCH_LIST)
list(REMOVE_DUPLICATES  OFFLOAD_ARCH_LIST)
set(hip_gpu_arch_list "")
foreach(__offload_arch ${OFFLOAD_ARCH_LIST})
    string(REPLACE "--offload-arch=" "" targetid ${__offload_arch})
    string(REPLACE ":" ";" target_id_list ${targetid})
    list(GET target_id_list 0 arch)
    string(APPEND hip_gpu_arch_list "${arch};")
    list(REMOVE_DUPLICATES  hip_gpu_arch_list)
endforeach()

if(WIN32)
    set(configToUse "config_amd_windows")
    set(config_file ${CMAKE_SOURCE_DIR}/hipTestMain/config/${configToUse})
    set(json_file ${HIP_TEST_CONFIG_BINARY_DIR}/${configToUse}.json)

    set(cmd "${CLANG_CPP_EXEC} -P -DGITHASH=\"${HIP_VERSION_GITHASH}\" ${config_file}>${json_file}")
    message(${cmd})
    execute_process(COMMAND cmd.exe /C ${cmd}
    RESULT_VARIABLE json_result)
    message(STATUS "${configToUse}.json file generation result: ${json_result}")
else()
    set(configToUse "config_amd_linux")
    foreach(arch ${hip_gpu_arch_list})
        set(config_file ${CMAKE_SOURCE_DIR}/hipTestMain/config/${configToUse})
        set(json_file ${HIP_TEST_CONFIG_BINARY_DIR}/${configToUse}_${arch}.json)
        set(cmd "${CLANG_CPP_EXEC} -P -D${arch} -DGITHASH=\\\"${HIP_VERSION_GITHASH}\\\" ${config_file}>${json_file}")
        message(${cmd})
        execute_process(COMMAND bash -c ${cmd}
        RESULT_VARIABLE json_result)
        message(STATUS "${configToUse}_${arch}.json file generation result: ${json_result}")
    endforeach()
endif()
# prints the catch info to a file
string(TIMESTAMP _timestamp UTC)
set(_autogen "# Auto-generated by cmake on ${_timestamp} UTC\n")
set(_catchInfo ${_autogen} "HIP_VERSION=${HIP_VERSION}\n")
set(_catchInfo ${_catchInfo} "HIP_PLATFORM=${HIP_PLATFORM}\n")
set(_catchInfo ${_catchInfo} "HIP_TESTS_GITHASH=${HIP_TESTS_GITHASH}\n")
set(_catchInfo ${_catchInfo} "ARCHS=${HIP_GPU_ARCH_LIST}\n")
file(WRITE ${CMAKE_CURRENT_BINARY_DIR}/${CATCH_BUILD_DIR}/catchInfo.txt ${_catchInfo})
# allows user to run ctest from catch_tests level
set(_subdirs ${_autogen} "subdirs(..)\n")
file(WRITE ${CMAKE_CURRENT_BINARY_DIR}/${CATCH_BUILD_DIR}/CTestTestfile.cmake ${_subdirs})

set(CATCH_SCRIPT_BINARY_DIR ${CMAKE_CURRENT_BINARY_DIR}/${CATCH_BUILD_DIR}/script)
file(COPY ${CMAKE_CURRENT_SOURCE_DIR}/cmake/hip-tests.cmake
     DESTINATION ${CATCH_SCRIPT_BINARY_DIR})
file(COPY ${CATCH_ADD_TESTS_SCRIPT}
         DESTINATION ${CATCH_SCRIPT_BINARY_DIR})
set_property(GLOBAL APPEND PROPERTY G_INSTALL_SCRIPT_TARGETS ${CATCH_ADD_TESTS_SCRIPT})

# copy python script and headers to catch test package
set(CATCH_INCLUDE_DIR include)
execute_process(COMMAND ${CMAKE_COMMAND} -E
                make_directory ${CMAKE_CURRENT_BINARY_DIR}/${CATCH_BUILD_DIR}/${CATCH_INCLUDE_DIR})

file(COPY ./unit/compileAndCaptureOutput.py
            DESTINATION  ${CMAKE_CURRENT_BINARY_DIR}/${CATCH_BUILD_DIR}/unit )

file(COPY ./include/hip_test_common.hh DESTINATION ${CMAKE_CURRENT_BINARY_DIR}/${CATCH_BUILD_DIR}/${CATCH_INCLUDE_DIR})
file(COPY ./include/hip_test_context.hh DESTINATION ${CMAKE_CURRENT_BINARY_DIR}/${CATCH_BUILD_DIR}/${CATCH_INCLUDE_DIR})

# Enable device lambda on nvidia platforms
if(HIP_PLATFORM STREQUAL "nvidia")
    # SPIRV is not supported on it
    if(ENABLE_SPIRV)
        message(FATAL_ERROR "SPIRV is not supported on this platform")
    endif()

    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} --extended-lambda")
    add_compile_options(-Xcompiler=-Wno-deprecated-declarations)
endif()

add_custom_target(build_tests ALL)

# Tests folder
if(BUILD_UNIT_TESTS)
    add_subdirectory(unit ${CATCH_BUILD_DIR}/unit)
    add_subdirectory(ABM ${CATCH_BUILD_DIR}/ABM)
    add_subdirectory(kernels ${CATCH_BUILD_DIR}/kernels)
    add_subdirectory(hipTestMain ${CATCH_BUILD_DIR}/hipTestMain)
    add_subdirectory(TypeQualifiers ${CATCH_BUILD_DIR}/TypeQualifiers)
    if(UNIX)
      add_subdirectory(multiproc ${CATCH_BUILD_DIR}/multiproc)
    endif()
endif()
if(BUILD_PERF_TESTS)
    add_subdirectory(perftests ${CATCH_BUILD_DIR}/perftests)
    add_subdirectory(performance ${CATCH_BUILD_DIR}/performance)
endif()
if(BUILD_STRESS_TESTS)
    add_subdirectory(stress ${CATCH_BUILD_DIR}/stress)
endif()

add_custom_target(gen_coverage
    COMMAND ${CMAKE_COMMAND} -E make_directory ${CMAKE_CURRENT_BINARY_DIR}/utils/coverage
    COMMAND ${CMAKE_COMMAND} -B ${CMAKE_CURRENT_BINARY_DIR}/utils/coverage -S ${CMAKE_CURRENT_SOURCE_DIR}/../utils/coverage
    COMMAND ${CMAKE_COMMAND} --build ${CMAKE_CURRENT_BINARY_DIR}/utils/coverage --target clean-cov-report
    COMMAND ${CMAKE_COMMAND} --build ${CMAKE_CURRENT_BINARY_DIR}/utils/coverage
    COMMAND ${CMAKE_CURRENT_BINARY_DIR}/utils/coverage/generateHipAPICoverage ${HIP_PATH}/include ${CMAKE_CURRENT_SOURCE_DIR} ${CMAKE_CURRENT_BINARY_DIR}/utils/coverage/
    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
    COMMENT "Generating Test Coverage Report"
    USES_TERMINAL
    VERBATIM
)
set_property(TARGET gen_coverage PROPERTY EXCLUDE_FROM_ALL TRUE)

# packaging the tests
add_subdirectory(packaging)

# Doxygen documentation
# check if doxygen is installed
find_package(Doxygen)
if(DOXYGEN_FOUND)
    # Configure Doxyfile with proper paths
    set(DOXYGEN_IN ${CMAKE_CURRENT_SOURCE_DIR}/DoxyfileTests)
    set(DOXYGEN_OUT ${CMAKE_CURRENT_BINARY_DIR}/DoxyfileTests.out)
    configure_file(${DOXYGEN_IN} ${DOXYGEN_OUT} @ONLY)

    add_custom_target(build_docs
        COMMAND ${DOXYGEN_EXECUTABLE} ${DOXYGEN_OUT}
        COMMENT "Generating test plan documentation with Doxygen")
else()
    message(STATUS "Doxygen was not found. Building test plan documentation will not be available")
endif()
