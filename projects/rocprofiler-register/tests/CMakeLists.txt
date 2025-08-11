cmake_minimum_required(VERSION 3.16 FATAL_ERROR)

project(rocprofiler-register-tests LANGUAGES C CXX)
set(CMAKE_INSTALL_DEFAULT_COMPONENT_NAME tests)

include(CMakeParseArguments)
include(GNUInstallDirs)

if("${CMAKE_PROJECT_NAME}" STREQUAL "${PROJECT_NAME}")
    find_package(rocprofiler-register REQUIRED)
    set(CMAKE_CXX_STANDARD 17)
    set(CMAKE_CXX_EXTENSIONS OFF)
    set(CMAKE_CXX_STANDARD_REQUIRED ON)
    set(CMAKE_C_VISIBILITY_PRESET "hidden")
    set(CMAKE_CXX_VISIBILITY_PRESET "hidden")
    set(CMAKE_VISIBILITY_INLINES_HIDDEN OFF)
    set(CMAKE_POSITION_INDEPENDENT_CODE ON)
    set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY "${PROJECT_BINARY_DIR}/${CMAKE_INSTALL_LIBDIR}")
    set(CMAKE_LIBRARY_OUTPUT_DIRECTORY "${PROJECT_BINARY_DIR}/${CMAKE_INSTALL_LIBDIR}")
    set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${PROJECT_BINARY_DIR}/${CMAKE_INSTALL_BINDIR}")
    enable_testing()
    include(CTest)
endif()

function(rocp_register_strip_target)
    if(CMAKE_BUILD_TYPE MATCHES "(DEBUG|Debug)")
        return()
    endif()

    cmake_parse_arguments(STRIP "" "" "ARGS" ${ARGN})

    list(LENGTH STRIP_UNPARSED_ARGUMENTS NUM_UNPARSED)

    if(NUM_UNPARSED EQUAL 1)
        set(_TARGET "${STRIP_UNPARSED_ARGUMENTS}")
    else()
        message(
            FATAL_ERROR "rocp_register_strip_target cannot deduce target from \"${ARGN}\""
            )
    endif()

    if(NOT TARGET "${_TARGET}")
        message(
            FATAL_ERROR
                "rocp_register_strip_target not provided valid target: \"${_TARGET}\"")
    endif()

    if(CMAKE_STRIP)
        add_custom_command(
            TARGET ${_TARGET}
            POST_BUILD
            COMMAND ${CMAKE_STRIP} ${STRIP_ARGS} $<TARGET_FILE:${_TARGET}>
            WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
            COMMENT "Stripping ${_TARGET}...")
    else()
        message(WARNING "CMAKE_STRIP not set... Not stripping ${_TARGET}")
    endif()
endfunction()

if(NOT CMAKE_DL_LIBS)
    find_library(DL_LIBRARY NAMES dl)
    if(NOT DL_LIBRARY)
        set(DL_LIBRARY dl)
    endif()
    set(CMAKE_DL_LIBS ${DL_LIBRARY})
endif()

#
# mock libraries
#
add_subdirectory(hsa-runtime)
add_subdirectory(amdhip)
add_subdirectory(roctx)
add_subdirectory(rccl)
add_subdirectory(rocdecode)
add_subdirectory(rocjpeg)
add_subdirectory(rocprofiler)

#
# mock tool library
#
add_subdirectory(generic-tool)

#
# test app
#
add_subdirectory(bin)

#
# common to multiple tests
#
find_package(Threads REQUIRED)

add_library(rocprofiler-register-tests-common INTERFACE)
target_include_directories(rocprofiler-register-tests-common
                           INTERFACE $<BUILD_INTERFACE:${CMAKE_CURRENT_LIST_DIR}>)
target_link_libraries(rocprofiler-register-tests-common INTERFACE Threads::Threads stdc++
                                                                  ${CMAKE_DL_LIBS})

add_library(rocprofiler-register-tests-strong INTERFACE)
target_compile_definitions(rocprofiler-register-tests-strong
                           INTERFACE ROCP_REG_TEST_WEAK=0)

add_library(rocprofiler-register-tests-weak INTERFACE)
target_compile_definitions(rocprofiler-register-tests-weak INTERFACE ROCP_REG_TEST_WEAK=1)

add_library(rocprofiler-register-tests-tool INTERFACE)
target_link_options(rocprofiler-register-tests-tool INTERFACE -Wl,--no-as-needed)
target_link_libraries(rocprofiler-register-tests-tool
                      INTERFACE generic-tool::generic-tool)

add_library(rocprofiler-register-tests-rocp INTERFACE)
target_link_options(rocprofiler-register-tests-rocp INTERFACE -Wl,--no-as-needed)
target_link_libraries(rocprofiler-register-tests-rocp
                      INTERFACE rocprofiler::rocprofiler generic-tool::generic-tool)

add_library(rocprofiler-register::tests-common ALIAS rocprofiler-register-tests-common)
add_library(rocprofiler-register::tests-strong ALIAS rocprofiler-register-tests-strong)
add_library(rocprofiler-register::tests-weak ALIAS rocprofiler-register-tests-weak)
add_library(rocprofiler-register::tests-tool ALIAS rocprofiler-register-tests-tool)
add_library(rocprofiler-register::tests-rocp ALIAS rocprofiler-register-tests-rocp)

#
# test scenarios
#
function(rocp_register_test_executable _NAME)
    set(_SV_ARGS)
    foreach(_TYPE "CORE" "PRELOAD" "WRAP" "PRELOAD_WRAP")
        list(APPEND _SV_ARGS "${_TYPE}_PASS_REGEX" "${_TYPE}_FAIL_REGEX")
    endforeach()
    cmake_parse_arguments(
        RRTE
        "" # options
        "SECURE;${_SV_ARGS}" # single value args
        "SOURCES;LIBRARIES" # multiple value args
        ${ARGN})

    add_executable(${_NAME})
    target_sources(${_NAME} PRIVATE ${RRTE_SOURCES})
    target_link_libraries(${_NAME} PRIVATE rocprofiler-register::tests-common
                                           ${RRTE_LIBRARIES})

    function(rocp_register_add_test _TEST_NAME _TARGET _ENV _PASS _FAIL)
        if(RRTE_SECURE)
            list(APPEND _ENV "ROCPROFILER_REGISTER_SECURE=true")
            set(_LABELS "secure")
        else()
            set(_LABELS "insecure")
        endif()
        add_test(
            NAME ${_TEST_NAME}
            COMMAND $<TARGET_FILE:${_TARGET}>
            WORKING_DIRECTORY ${PROJECT_BINARY_DIR})
        set_tests_properties(
            ${_TEST_NAME}
            PROPERTIES ENVIRONMENT
                       "LD_LIBRARY_PATH=${CMAKE_LIBRARY_OUTPUT_DIRECTORY};${_ENV}"
                       PASS_REGULAR_EXPRESSION
                       "${_PASS}"
                       FAIL_REGULAR_EXPRESSION
                       "${_FAIL}"
                       LABELS
                       "${_LABELS}"
                       ${ARGN})
    endfunction()

    rocp_register_add_test(
        ${_NAME} ${_NAME} "${ROCPROFILER_REGISTER_MEMCHECK_PRELOAD_ENV}"
        "${RRTE_CORE_PASS_REGEX}" "${RRTE_CORE_FAIL_REGEX}")

    rocp_register_add_test(
        ${_NAME}-preload ${_NAME}
        "LD_PRELOAD=${ROCPROFILER_REGISTER_MEMCHECK_PRELOAD_LIBRARY}:libgeneric-tool.so"
        "${RRTE_PRELOAD_PASS_REGEX}" "${RRTE_PRELOAD_FAIL_REGEX}")

    rocp_register_add_test(
        ${_NAME}-env
        ${_NAME}
        "ROCP_TOOL_LIBRARIES=libgeneric-tool.so;${ROCPROFILER_REGISTER_MEMCHECK_PRELOAD_ENV}"
        "${RRTE_PRELOAD_PASS_REGEX}"
        "${RRTE_PRELOAD_FAIL_REGEX}")

    rocp_register_add_test(
        ${_NAME}-wrap ${_NAME}
        "ROCP_REG_TEST_WRAP=1;${ROCPROFILER_REGISTER_MEMCHECK_PRELOAD_ENV}"
        "${RRTE_WRAP_PASS_REGEX}" "${RRTE_WRAP_FAIL_REGEX}")

    rocp_register_add_test(
        ${_NAME}-preload-wrap
        ${_NAME}
        "LD_PRELOAD=${ROCPROFILER_REGISTER_MEMCHECK_PRELOAD_LIBRARY}:libgeneric-tool.so;ROCP_REG_TEST_WRAP=1"
        "${RRTE_PRELOAD_WRAP_PASS_REGEX}"
        "${RRTE_PRELOAD_WRAP_FAIL_REGEX}")

    rocp_register_add_test(
        ${_NAME}-env-wrap
        ${_NAME}
        "ROCP_TOOL_LIBRARIES=libgeneric-tool.so;ROCP_REG_TEST_WRAP=1;${ROCPROFILER_REGISTER_MEMCHECK_PRELOAD_ENV}"
        "${RRTE_PRELOAD_WRAP_PASS_REGEX}"
        "${RRTE_PRELOAD_WRAP_FAIL_REGEX}")

endfunction()

# serial
rocp_register_test_executable(
    test-amdhip-roctx
    SECURE ON
    SOURCES test-amdhip-roctx.cpp
    LIBRARIES rocprofiler-register::tests-strong amdhip::amdhip roctx::roctx
    CORE_PASS_REGEX
        ".hsa-runtime.cpp. hsa_init\n.amdhip.cpp. hip_init\n.roctx.cpp..push. thread-main\n.roctx.cpp..pop. thread-main\n"
    PRELOAD_PASS_REGEX
        ".rocprofiler.cpp. rocp_ctor\n.rocprofiler.cpp. hsa :: 20100 :: 0 :: 1\n.hsa-runtime.cpp. hsa identifier 0\n.hsa-runtime.cpp. hsa_init\n.rocprofiler.cpp. hip :: 60001 :: 0 :: 1\n.amdhip.cpp. hip identifier 8\n.amdhip.cpp. hip_init\n.rocprofiler.cpp. roctx :: 40601 :: 0 :: 1\n.roctx.cpp. roctx identifier 16\n.roctx.cpp..push. thread-main\n.roctx.cpp..pop. thread-main\n"
    WRAP_PASS_REGEX
        ".hsa-runtime.cpp. hsa_init\n.amdhip.cpp. hip_init\n.roctx.cpp..push. thread-main\n.roctx.cpp..pop. thread-main\n"
    PRELOAD_WRAP_PASS_REGEX
        ".rocprofiler.cpp. rocp_ctor\n.rocprofiler.cpp. hsa :: 20100 :: 0 :: 1\n.hsa-runtime.cpp. hsa identifier 0\n.rocprofiler.cpp. hsa_init\n.rocprofiler.cpp. hip :: 60001 :: 0 :: 1\n.amdhip.cpp. hip identifier 8\n.rocprofiler.cpp. hip_init\n.rocprofiler.cpp. roctx :: 40601 :: 0 :: 1\n.roctx.cpp. roctx identifier 16\n.rocprofiler.cpp..push. thread-main\n.rocprofiler.cpp..pop. thread-main\n"
    )

rocp_register_test_executable(
    test-amdhip-roctx-weak
    SOURCES test-amdhip-roctx.cpp
    LIBRARIES rocprofiler-register::tests-weak
    CORE_PASS_REGEX
        ".fwd.hpp. dlopen libamdhip64.so, 1\n.fwd.hpp. dlopen libroctx64.so, 1\n.hsa-runtime.cpp. hsa_init\n.amdhip.cpp. hip_init\n.roctx.cpp..push. thread-main\n.roctx.cpp..pop. thread-main\n"
    PRELOAD_PASS_REGEX
        ".fwd.hpp. dlopen libamdhip64.so, 1\n.fwd.hpp. dlopen libroctx64.so, 1\n.rocprofiler.cpp. rocp_ctor\n.rocprofiler.cpp. hsa :: 20100 :: 0 :: 1\n.hsa-runtime.cpp. hsa identifier 0\n.hsa-runtime.cpp. hsa_init\n.rocprofiler.cpp. hip :: 60001 :: 0 :: 1\n.amdhip.cpp. hip identifier 8\n.amdhip.cpp. hip_init\n.rocprofiler.cpp. roctx :: 40601 :: 0 :: 1\n.roctx.cpp. roctx identifier 16\n.roctx.cpp..push. thread-main\n.roctx.cpp..pop. thread-main\n"
    WRAP_PASS_REGEX
        ".fwd.hpp. dlopen libamdhip64.so, 1\n.fwd.hpp. dlopen libroctx64.so, 1\n.hsa-runtime.cpp. hsa_init\n.amdhip.cpp. hip_init\n.roctx.cpp..push. thread-main\n.roctx.cpp..pop. thread-main\n"
    PRELOAD_WRAP_PASS_REGEX
        ".fwd.hpp. dlopen libamdhip64.so, 1\n.fwd.hpp. dlopen libroctx64.so, 1\n.rocprofiler.cpp. rocp_ctor\n.rocprofiler.cpp. hsa :: 20100 :: 0 :: 1\n.hsa-runtime.cpp. hsa identifier 0\n.rocprofiler.cpp. hsa_init\n.rocprofiler.cpp. hip :: 60001 :: 0 :: 1\n.amdhip.cpp. hip identifier 8\n.rocprofiler.cpp. hip_init\n.rocprofiler.cpp. roctx :: 40601 :: 0 :: 1\n.roctx.cpp. roctx identifier 16\n.rocprofiler.cpp..push. thread-main\n.rocprofiler.cpp..pop. thread-main\n"
    )

rocp_register_test_executable(
    test-amdhip-roctx-tool
    SOURCES test-amdhip-roctx.cpp
    LIBRARIES rocprofiler-register::tests-strong amdhip::amdhip roctx::roctx
              rocprofiler-register::tests-tool
    CORE_PASS_REGEX
        ".rocprofiler.cpp. rocp_ctor\n.rocprofiler.cpp. hsa :: 20100 :: 0 :: 1\n.hsa-runtime.cpp. hsa identifier 0\n.hsa-runtime.cpp. hsa_init\n.rocprofiler.cpp. hip :: 60001 :: 0 :: 1\n.amdhip.cpp. hip identifier 8\n.amdhip.cpp. hip_init\n.rocprofiler.cpp. roctx :: 40601 :: 0 :: 1\n.roctx.cpp. roctx identifier 16\n.roctx.cpp..push. thread-main\n.roctx.cpp..pop. thread-main\n"
    PRELOAD_PASS_REGEX
        ".rocprofiler.cpp. rocp_ctor\n.rocprofiler.cpp. hsa :: 20100 :: 0 :: 1\n.hsa-runtime.cpp. hsa identifier 0\n.hsa-runtime.cpp. hsa_init\n.rocprofiler.cpp. hip :: 60001 :: 0 :: 1\n.amdhip.cpp. hip identifier 8\n.amdhip.cpp. hip_init\n.rocprofiler.cpp. roctx :: 40601 :: 0 :: 1\n.roctx.cpp. roctx identifier 16\n.roctx.cpp..push. thread-main\n.roctx.cpp..pop. thread-main\n"
    WRAP_PASS_REGEX
        ".rocprofiler.cpp. rocp_ctor\n.rocprofiler.cpp. hsa :: 20100 :: 0 :: 1\n.hsa-runtime.cpp. hsa identifier 0\n.rocprofiler.cpp. hsa_init\n.rocprofiler.cpp. hip :: 60001 :: 0 :: 1\n.amdhip.cpp. hip identifier 8\n.rocprofiler.cpp. hip_init\n.rocprofiler.cpp. roctx :: 40601 :: 0 :: 1\n.roctx.cpp. roctx identifier 16\n.rocprofiler.cpp..push. thread-main\n.rocprofiler.cpp..pop. thread-main\n"
    PRELOAD_WRAP_PASS_REGEX
        ".rocprofiler.cpp. rocp_ctor\n.rocprofiler.cpp. hsa :: 20100 :: 0 :: 1\n.hsa-runtime.cpp. hsa identifier 0\n.rocprofiler.cpp. hsa_init\n.rocprofiler.cpp. hip :: 60001 :: 0 :: 1\n.amdhip.cpp. hip identifier 8\n.rocprofiler.cpp. hip_init\n.rocprofiler.cpp. roctx :: 40601 :: 0 :: 1\n.roctx.cpp. roctx identifier 16\n.rocprofiler.cpp..push. thread-main\n.rocprofiler.cpp..pop. thread-main\n"
    )

rocp_register_test_executable(
    test-amdhip-roctx-rocp
    SOURCES test-amdhip-roctx.cpp
    LIBRARIES rocprofiler-register::tests-strong amdhip::amdhip roctx::roctx
              rocprofiler-register::tests-rocp
    CORE_PASS_REGEX
        ".rocprofiler.cpp. rocp_ctor\n.rocprofiler.cpp. hsa :: 20100 :: 0 :: 1\n.hsa-runtime.cpp. hsa identifier 0\n.hsa-runtime.cpp. hsa_init\n.rocprofiler.cpp. hip :: 60001 :: 0 :: 1\n.amdhip.cpp. hip identifier 8\n.amdhip.cpp. hip_init\n.rocprofiler.cpp. roctx :: 40601 :: 0 :: 1\n.roctx.cpp. roctx identifier 16\n.roctx.cpp..push. thread-main\n.roctx.cpp..pop. thread-main\n"
    PRELOAD_PASS_REGEX
        ".rocprofiler.cpp. rocp_ctor\n.rocprofiler.cpp. hsa :: 20100 :: 0 :: 1\n.hsa-runtime.cpp. hsa identifier 0\n.hsa-runtime.cpp. hsa_init\n.rocprofiler.cpp. hip :: 60001 :: 0 :: 1\n.amdhip.cpp. hip identifier 8\n.amdhip.cpp. hip_init\n.rocprofiler.cpp. roctx :: 40601 :: 0 :: 1\n.roctx.cpp. roctx identifier 16\n.roctx.cpp..push. thread-main\n.roctx.cpp..pop. thread-main\n"
    WRAP_PASS_REGEX
        ".rocprofiler.cpp. rocp_ctor\n.rocprofiler.cpp. hsa :: 20100 :: 0 :: 1\n.hsa-runtime.cpp. hsa identifier 0\n.rocprofiler.cpp. hsa_init\n.rocprofiler.cpp. hip :: 60001 :: 0 :: 1\n.amdhip.cpp. hip identifier 8\n.rocprofiler.cpp. hip_init\n.rocprofiler.cpp. roctx :: 40601 :: 0 :: 1\n.roctx.cpp. roctx identifier 16\n.rocprofiler.cpp..push. thread-main\n.rocprofiler.cpp..pop. thread-main\n"
    PRELOAD_WRAP_PASS_REGEX
        ".rocprofiler.cpp. rocp_ctor\n.rocprofiler.cpp. hsa :: 20100 :: 0 :: 1\n.hsa-runtime.cpp. hsa identifier 0\n.rocprofiler.cpp. hsa_init\n.rocprofiler.cpp. hip :: 60001 :: 0 :: 1\n.amdhip.cpp. hip identifier 8\n.rocprofiler.cpp. hip_init\n.rocprofiler.cpp. roctx :: 40601 :: 0 :: 1\n.roctx.cpp. roctx identifier 16\n.rocprofiler.cpp..push. thread-main\n.rocprofiler.cpp..pop. thread-main\n"
    )

rocp_register_test_executable(
    test-amdhip-roctx-weak-tool
    SOURCES test-amdhip-roctx.cpp
    LIBRARIES rocprofiler-register::tests-weak rocprofiler-register::tests-tool
    CORE_PASS_REGEX
        ".fwd.hpp. dlopen libamdhip64.so, 1\n.fwd.hpp. dlopen libroctx64.so, 1\n.rocprofiler.cpp. rocp_ctor\n.rocprofiler.cpp. hsa :: 20100 :: 0 :: 1\n.hsa-runtime.cpp. hsa identifier 0\n.hsa-runtime.cpp. hsa_init\n.rocprofiler.cpp. hip :: 60001 :: 0 :: 1\n.amdhip.cpp. hip identifier 8\n.amdhip.cpp. hip_init\n.rocprofiler.cpp. roctx :: 40601 :: 0 :: 1\n.roctx.cpp. roctx identifier 16\n.roctx.cpp..push. thread-main\n.roctx.cpp..pop. thread-main\n"
    PRELOAD_PASS_REGEX
        ".fwd.hpp. dlopen libamdhip64.so, 1\n.fwd.hpp. dlopen libroctx64.so, 1\n.rocprofiler.cpp. rocp_ctor\n.rocprofiler.cpp. hsa :: 20100 :: 0 :: 1\n.hsa-runtime.cpp. hsa identifier 0\n.hsa-runtime.cpp. hsa_init\n.rocprofiler.cpp. hip :: 60001 :: 0 :: 1\n.amdhip.cpp. hip identifier 8\n.amdhip.cpp. hip_init\n.rocprofiler.cpp. roctx :: 40601 :: 0 :: 1\n.roctx.cpp. roctx identifier 16\n.roctx.cpp..push. thread-main\n.roctx.cpp..pop. thread-main\n"
    WRAP_PASS_REGEX
        ".fwd.hpp. dlopen libamdhip64.so, 1\n.fwd.hpp. dlopen libroctx64.so, 1\n.rocprofiler.cpp. rocp_ctor\n.rocprofiler.cpp. hsa :: 20100 :: 0 :: 1\n.hsa-runtime.cpp. hsa identifier 0\n.rocprofiler.cpp. hsa_init\n.rocprofiler.cpp. hip :: 60001 :: 0 :: 1\n.amdhip.cpp. hip identifier 8\n.rocprofiler.cpp. hip_init\n.rocprofiler.cpp. roctx :: 40601 :: 0 :: 1\n.roctx.cpp. roctx identifier 16\n.rocprofiler.cpp..push. thread-main\n.rocprofiler.cpp..pop. thread-main\n"
    PRELOAD_WRAP_PASS_REGEX
        ".fwd.hpp. dlopen libamdhip64.so, 1\n.fwd.hpp. dlopen libroctx64.so, 1\n.rocprofiler.cpp. rocp_ctor\n.rocprofiler.cpp. hsa :: 20100 :: 0 :: 1\n.hsa-runtime.cpp. hsa identifier 0\n.rocprofiler.cpp. hsa_init\n.rocprofiler.cpp. hip :: 60001 :: 0 :: 1\n.amdhip.cpp. hip identifier 8\n.rocprofiler.cpp. hip_init\n.rocprofiler.cpp. roctx :: 40601 :: 0 :: 1\n.roctx.cpp. roctx identifier 16\n.rocprofiler.cpp..push. thread-main\n.rocprofiler.cpp..pop. thread-main\n"
    )

rocp_register_test_executable(
    test-amdhip-roctx-weak-rocp
    SOURCES test-amdhip-roctx.cpp
    LIBRARIES rocprofiler-register::tests-weak rocprofiler-register::tests-rocp
    CORE_PASS_REGEX
        ".fwd.hpp. dlopen libamdhip64.so, 1\n.fwd.hpp. dlopen libroctx64.so, 1\n.rocprofiler.cpp. rocp_ctor\n.rocprofiler.cpp. hsa :: 20100 :: 0 :: 1\n.hsa-runtime.cpp. hsa identifier 0\n.hsa-runtime.cpp. hsa_init\n.rocprofiler.cpp. hip :: 60001 :: 0 :: 1\n.amdhip.cpp. hip identifier 8\n.amdhip.cpp. hip_init\n.rocprofiler.cpp. roctx :: 40601 :: 0 :: 1\n.roctx.cpp. roctx identifier 16\n.roctx.cpp..push. thread-main\n.roctx.cpp..pop. thread-main\n"
    PRELOAD_PASS_REGEX
        ".fwd.hpp. dlopen libamdhip64.so, 1\n.fwd.hpp. dlopen libroctx64.so, 1\n.rocprofiler.cpp. rocp_ctor\n.rocprofiler.cpp. hsa :: 20100 :: 0 :: 1\n.hsa-runtime.cpp. hsa identifier 0\n.hsa-runtime.cpp. hsa_init\n.rocprofiler.cpp. hip :: 60001 :: 0 :: 1\n.amdhip.cpp. hip identifier 8\n.amdhip.cpp. hip_init\n.rocprofiler.cpp. roctx :: 40601 :: 0 :: 1\n.roctx.cpp. roctx identifier 16\n.roctx.cpp..push. thread-main\n.roctx.cpp..pop. thread-main\n"
    WRAP_PASS_REGEX
        ".fwd.hpp. dlopen libamdhip64.so, 1\n.fwd.hpp. dlopen libroctx64.so, 1\n.rocprofiler.cpp. rocp_ctor\n.rocprofiler.cpp. hsa :: 20100 :: 0 :: 1\n.hsa-runtime.cpp. hsa identifier 0\n.rocprofiler.cpp. hsa_init\n.rocprofiler.cpp. hip :: 60001 :: 0 :: 1\n.amdhip.cpp. hip identifier 8\n.rocprofiler.cpp. hip_init\n.rocprofiler.cpp. roctx :: 40601 :: 0 :: 1\n.roctx.cpp. roctx identifier 16\n.rocprofiler.cpp..push. thread-main\n.rocprofiler.cpp..pop. thread-main\n"
    PRELOAD_WRAP_PASS_REGEX
        ".fwd.hpp. dlopen libamdhip64.so, 1\n.fwd.hpp. dlopen libroctx64.so, 1\n.rocprofiler.cpp. rocp_ctor\n.rocprofiler.cpp. hsa :: 20100 :: 0 :: 1\n.hsa-runtime.cpp. hsa identifier 0\n.rocprofiler.cpp. hsa_init\n.rocprofiler.cpp. hip :: 60001 :: 0 :: 1\n.amdhip.cpp. hip identifier 8\n.rocprofiler.cpp. hip_init\n.rocprofiler.cpp. roctx :: 40601 :: 0 :: 1\n.roctx.cpp. roctx identifier 16\n.rocprofiler.cpp..push. thread-main\n.rocprofiler.cpp..pop. thread-main\n"
    )

# multithreaded
rocp_register_test_executable(
    test-amdhip-roctx-mt
    SOURCES test-amdhip-roctx-mt.cpp
    LIBRARIES rocprofiler-register::tests-strong amdhip::amdhip roctx::roctx
    CORE_PASS_REGEX
        ".hsa-runtime.cpp. hsa_init\n.amdhip.cpp. hip_init\n.amdhip.cpp. hip_init\n.amdhip.cpp. hip_init\n.amdhip.cpp. hip_init\n.roctx.cpp..push. thread-[0-3]\n.roctx.cpp..push. thread-[0-3]\n.roctx.cpp..push. thread-[0-3]\n.roctx.cpp..push. thread-[0-3]\n.roctx.cpp..pop. thread-[0-3]\n.roctx.cpp..pop. thread-[0-3]\n.roctx.cpp..pop. thread-[0-3]\n.roctx.cpp..pop. thread-[0-3]\n"
    PRELOAD_PASS_REGEX
        ".rocprofiler.cpp. rocp_ctor\n.rocprofiler.cpp. hsa :: 20100 :: 0 :: 1\n.hsa-runtime.cpp. hsa identifier 0\n.hsa-runtime.cpp. hsa_init\n.rocprofiler.cpp. hip :: 60001 :: 0 :: 1\n.amdhip.cpp. hip identifier 8\n.amdhip.cpp. hip_init\n.amdhip.cpp. hip_init\n.amdhip.cpp. hip_init\n.amdhip.cpp. hip_init\n.rocprofiler.cpp. roctx :: 40601 :: 0 :: 1\n.roctx.cpp. roctx identifier 16\n.roctx.cpp..push. thread-[0-3]\n.roctx.cpp..push. thread-[0-3]\n.roctx.cpp..push. thread-[0-3]\n.roctx.cpp..push. thread-[0-3]\n.roctx.cpp..pop. thread-[0-3]\n.roctx.cpp..pop. thread-[0-3]\n.roctx.cpp..pop. thread-[0-3]\n.roctx.cpp..pop. thread-[0-3]\n"
    WRAP_PASS_REGEX
        ".hsa-runtime.cpp. hsa_init\n.amdhip.cpp. hip_init\n.amdhip.cpp. hip_init\n.amdhip.cpp. hip_init\n.amdhip.cpp. hip_init\n.roctx.cpp..push. thread-[0-3]\n.roctx.cpp..push. thread-[0-3]\n.roctx.cpp..push. thread-[0-3]\n.roctx.cpp..push. thread-[0-3]\n.roctx.cpp..pop. thread-[0-3]\n.roctx.cpp..pop. thread-[0-3]\n.roctx.cpp..pop. thread-[0-3]\n.roctx.cpp..pop. thread-[0-3]\n"
    PRELOAD_WRAP_PASS_REGEX
        ".rocprofiler.cpp. rocp_ctor\n.rocprofiler.cpp. hsa :: 20100 :: 0 :: 1\n.hsa-runtime.cpp. hsa identifier 0\n.rocprofiler.cpp. hsa_init\n.rocprofiler.cpp. hip :: 60001 :: 0 :: 1\n.amdhip.cpp. hip identifier 8\n.rocprofiler.cpp. hip_init\n.rocprofiler.cpp. hip_init\n.rocprofiler.cpp. hip_init\n.rocprofiler.cpp. hip_init\n.rocprofiler.cpp. roctx :: 40601 :: 0 :: 1\n.roctx.cpp. roctx identifier 16\n.rocprofiler.cpp..push. thread-[0-3]\n.rocprofiler.cpp..push. thread-[0-3]\n.rocprofiler.cpp..push. thread-[0-3]\n.rocprofiler.cpp..push. thread-[0-3]\n.rocprofiler.cpp..pop. thread-[0-3]\n.rocprofiler.cpp..pop. thread-[0-3]\n.rocprofiler.cpp..pop. thread-[0-3]\n.rocprofiler.cpp..pop. thread-[0-3]\n"
    )

rocp_register_test_executable(
    test-amdhip-roctx-mt-weak
    SOURCES test-amdhip-roctx-mt.cpp
    LIBRARIES rocprofiler-register::tests-weak
    CORE_PASS_REGEX
        ".fwd.hpp. dlopen libamdhip64.so, 1\n.fwd.hpp. dlopen libroctx64.so, 1\n.hsa-runtime.cpp. hsa_init\n.amdhip.cpp. hip_init\n.amdhip.cpp. hip_init\n.amdhip.cpp. hip_init\n.amdhip.cpp. hip_init\n.roctx.cpp..push. thread-[0-3]\n.roctx.cpp..push. thread-[0-3]\n.roctx.cpp..push. thread-[0-3]\n.roctx.cpp..push. thread-[0-3]\n.roctx.cpp..pop. thread-[0-3]\n.roctx.cpp..pop. thread-[0-3]\n.roctx.cpp..pop. thread-[0-3]\n.roctx.cpp..pop. thread-[0-3]\n"
    PRELOAD_PASS_REGEX
        ".fwd.hpp. dlopen libamdhip64.so, 1\n.fwd.hpp. dlopen libroctx64.so, 1\n.rocprofiler.cpp. rocp_ctor\n.rocprofiler.cpp. hsa :: 20100 :: 0 :: 1\n.hsa-runtime.cpp. hsa identifier 0\n.hsa-runtime.cpp. hsa_init\n.rocprofiler.cpp. hip :: 60001 :: 0 :: 1\n.amdhip.cpp. hip identifier 8\n.amdhip.cpp. hip_init\n.amdhip.cpp. hip_init\n.amdhip.cpp. hip_init\n.amdhip.cpp. hip_init\n.rocprofiler.cpp. roctx :: 40601 :: 0 :: 1\n.roctx.cpp. roctx identifier 16\n.roctx.cpp..push. thread-[0-3]\n.roctx.cpp..push. thread-[0-3]\n.roctx.cpp..push. thread-[0-3]\n.roctx.cpp..push. thread-[0-3]\n.roctx.cpp..pop. thread-[0-3]\n.roctx.cpp..pop. thread-[0-3]\n.roctx.cpp..pop. thread-[0-3]\n.roctx.cpp..pop. thread-[0-3]\n"
    WRAP_PASS_REGEX
        ".fwd.hpp. dlopen libamdhip64.so, 1\n.fwd.hpp. dlopen libroctx64.so, 1\n.hsa-runtime.cpp. hsa_init\n.amdhip.cpp. hip_init\n.amdhip.cpp. hip_init\n.amdhip.cpp. hip_init\n.amdhip.cpp. hip_init\n.roctx.cpp..push. thread-[0-3]\n.roctx.cpp..push. thread-[0-3]\n.roctx.cpp..push. thread-[0-3]\n.roctx.cpp..push. thread-[0-3]\n.roctx.cpp..pop. thread-[0-3]\n.roctx.cpp..pop. thread-[0-3]\n.roctx.cpp..pop. thread-[0-3]\n.roctx.cpp..pop. thread-[0-3]\n"
    PRELOAD_WRAP_PASS_REGEX
        ".fwd.hpp. dlopen libamdhip64.so, 1\n.fwd.hpp. dlopen libroctx64.so, 1\n.rocprofiler.cpp. rocp_ctor\n.rocprofiler.cpp. hsa :: 20100 :: 0 :: 1\n.hsa-runtime.cpp. hsa identifier 0\n.rocprofiler.cpp. hsa_init\n.rocprofiler.cpp. hip :: 60001 :: 0 :: 1\n.amdhip.cpp. hip identifier 8\n.rocprofiler.cpp. hip_init\n.rocprofiler.cpp. hip_init\n.rocprofiler.cpp. hip_init\n.rocprofiler.cpp. hip_init\n.rocprofiler.cpp. roctx :: 40601 :: 0 :: 1\n.roctx.cpp. roctx identifier 16\n.rocprofiler.cpp..push. thread-[0-3]\n.rocprofiler.cpp..push. thread-[0-3]\n.rocprofiler.cpp..push. thread-[0-3]\n.rocprofiler.cpp..push. thread-[0-3]\n.rocprofiler.cpp..pop. thread-[0-3]\n.rocprofiler.cpp..pop. thread-[0-3]\n.rocprofiler.cpp..pop. thread-[0-3]\n.rocprofiler.cpp..pop. thread-[0-3]\n"
    )

rocp_register_test_executable(
    test-amdhip-roctx-mt-rocp
    SOURCES test-amdhip-roctx-mt.cpp
    LIBRARIES rocprofiler-register::tests-strong amdhip::amdhip roctx::roctx
              rocprofiler-register::tests-rocp
    CORE_PASS_REGEX
        ".rocprofiler.cpp. rocp_ctor\n.rocprofiler.cpp. hsa :: 20100 :: 0 :: 1\n.hsa-runtime.cpp. hsa identifier 0\n.hsa-runtime.cpp. hsa_init\n.rocprofiler.cpp. hip :: 60001 :: 0 :: 1\n.amdhip.cpp. hip identifier 8\n.amdhip.cpp. hip_init\n.amdhip.cpp. hip_init\n.amdhip.cpp. hip_init\n.amdhip.cpp. hip_init\n.rocprofiler.cpp. roctx :: 40601 :: 0 :: 1\n.roctx.cpp. roctx identifier 16\n.roctx.cpp..push. thread-[0-3]\n.roctx.cpp..push. thread-[0-3]\n.roctx.cpp..push. thread-[0-3]\n.roctx.cpp..push. thread-[0-3]\n.roctx.cpp..pop. thread-[0-3]\n.roctx.cpp..pop. thread-[0-3]\n.roctx.cpp..pop. thread-[0-3]\n.roctx.cpp..pop. thread-[0-3]\n"
    PRELOAD_PASS_REGEX
        ".rocprofiler.cpp. rocp_ctor\n.rocprofiler.cpp. hsa :: 20100 :: 0 :: 1\n.hsa-runtime.cpp. hsa identifier 0\n.hsa-runtime.cpp. hsa_init\n.rocprofiler.cpp. hip :: 60001 :: 0 :: 1\n.amdhip.cpp. hip identifier 8\n.amdhip.cpp. hip_init\n.amdhip.cpp. hip_init\n.amdhip.cpp. hip_init\n.amdhip.cpp. hip_init\n.rocprofiler.cpp. roctx :: 40601 :: 0 :: 1\n.roctx.cpp. roctx identifier 16\n.roctx.cpp..push. thread-[0-3]\n.roctx.cpp..push. thread-[0-3]\n.roctx.cpp..push. thread-[0-3]\n.roctx.cpp..push. thread-[0-3]\n.roctx.cpp..pop. thread-[0-3]\n.roctx.cpp..pop. thread-[0-3]\n.roctx.cpp..pop. thread-[0-3]\n.roctx.cpp..pop. thread-[0-3]\n"
    WRAP_PASS_REGEX
        ".rocprofiler.cpp. rocp_ctor\n.rocprofiler.cpp. hsa :: 20100 :: 0 :: 1\n.hsa-runtime.cpp. hsa identifier 0\n.rocprofiler.cpp. hsa_init\n.rocprofiler.cpp. hip :: 60001 :: 0 :: 1\n.amdhip.cpp. hip identifier 8\n.rocprofiler.cpp. hip_init\n.rocprofiler.cpp. hip_init\n.rocprofiler.cpp. hip_init\n.rocprofiler.cpp. hip_init\n.rocprofiler.cpp. roctx :: 40601 :: 0 :: 1\n.roctx.cpp. roctx identifier 16\n.rocprofiler.cpp..push. thread-[0-3]\n.rocprofiler.cpp..push. thread-[0-3]\n.rocprofiler.cpp..push. thread-[0-3]\n.rocprofiler.cpp..push. thread-[0-3]\n.rocprofiler.cpp..pop. thread-[0-3]\n.rocprofiler.cpp..pop. thread-[0-3]\n.rocprofiler.cpp..pop. thread-[0-3]\n.rocprofiler.cpp..pop. thread-[0-3]\n"
    PRELOAD_WRAP_PASS_REGEX
        ".rocprofiler.cpp. rocp_ctor\n.rocprofiler.cpp. hsa :: 20100 :: 0 :: 1\n.hsa-runtime.cpp. hsa identifier 0\n.rocprofiler.cpp. hsa_init\n.rocprofiler.cpp. hip :: 60001 :: 0 :: 1\n.amdhip.cpp. hip identifier 8\n.rocprofiler.cpp. hip_init\n.rocprofiler.cpp. hip_init\n.rocprofiler.cpp. hip_init\n.rocprofiler.cpp. hip_init\n.rocprofiler.cpp. roctx :: 40601 :: 0 :: 1\n.roctx.cpp. roctx identifier 16\n.rocprofiler.cpp..push. thread-[0-3]\n.rocprofiler.cpp..push. thread-[0-3]\n.rocprofiler.cpp..push. thread-[0-3]\n.rocprofiler.cpp..push. thread-[0-3]\n.rocprofiler.cpp..pop. thread-[0-3]\n.rocprofiler.cpp..pop. thread-[0-3]\n.rocprofiler.cpp..pop. thread-[0-3]\n.rocprofiler.cpp..pop. thread-[0-3]\n"
    )

# 29
rocp_register_test_executable(
    test-amdhip-roctx-mt-weak-rocp
    SOURCES test-amdhip-roctx-mt.cpp
    LIBRARIES rocprofiler-register::tests-weak rocprofiler-register::tests-rocp
    CORE_PASS_REGEX
        ".fwd.hpp. dlopen libamdhip64.so, 1\n.fwd.hpp. dlopen libroctx64.so, 1\n.rocprofiler.cpp. rocp_ctor\n.rocprofiler.cpp. hsa :: 20100 :: 0 :: 1\n.hsa-runtime.cpp. hsa identifier 0\n.hsa-runtime.cpp. hsa_init\n.rocprofiler.cpp. hip :: 60001 :: 0 :: 1\n.amdhip.cpp. hip identifier 8\n.amdhip.cpp. hip_init\n.amdhip.cpp. hip_init\n.amdhip.cpp. hip_init\n.amdhip.cpp. hip_init\n.rocprofiler.cpp. roctx :: 40601 :: 0 :: 1\n.roctx.cpp. roctx identifier 16\n.roctx.cpp..push. thread-[0-3]\n.roctx.cpp..push. thread-[0-3]\n.roctx.cpp..push. thread-[0-3]\n.roctx.cpp..push. thread-[0-3]\n.roctx.cpp..pop. thread-[0-3]\n.roctx.cpp..pop. thread-[0-3]\n.roctx.cpp..pop. thread-[0-3]\n.roctx.cpp..pop. thread-[0-3]\n"
    PRELOAD_PASS_REGEX
        ".fwd.hpp. dlopen libamdhip64.so, 1\n.fwd.hpp. dlopen libroctx64.so, 1\n.rocprofiler.cpp. rocp_ctor\n.rocprofiler.cpp. hsa :: 20100 :: 0 :: 1\n.hsa-runtime.cpp. hsa identifier 0\n.hsa-runtime.cpp. hsa_init\n.rocprofiler.cpp. hip :: 60001 :: 0 :: 1\n.amdhip.cpp. hip identifier 8\n.amdhip.cpp. hip_init\n.amdhip.cpp. hip_init\n.amdhip.cpp. hip_init\n.amdhip.cpp. hip_init\n.rocprofiler.cpp. roctx :: 40601 :: 0 :: 1\n.roctx.cpp. roctx identifier 16\n.roctx.cpp..push. thread-[0-3]\n.roctx.cpp..push. thread-[0-3]\n.roctx.cpp..push. thread-[0-3]\n.roctx.cpp..push. thread-[0-3]\n.roctx.cpp..pop. thread-[0-3]\n.roctx.cpp..pop. thread-[0-3]\n.roctx.cpp..pop. thread-[0-3]\n.roctx.cpp..pop. thread-[0-3]\n"
    WRAP_PASS_REGEX
        ".fwd.hpp. dlopen libamdhip64.so, 1\n.fwd.hpp. dlopen libroctx64.so, 1\n.rocprofiler.cpp. rocp_ctor\n.rocprofiler.cpp. hsa :: 20100 :: 0 :: 1\n.hsa-runtime.cpp. hsa identifier 0\n.rocprofiler.cpp. hsa_init\n.rocprofiler.cpp. hip :: 60001 :: 0 :: 1\n.amdhip.cpp. hip identifier 8\n.rocprofiler.cpp. hip_init\n.rocprofiler.cpp. hip_init\n.rocprofiler.cpp. hip_init\n.rocprofiler.cpp. hip_init\n.rocprofiler.cpp. roctx :: 40601 :: 0 :: 1\n.roctx.cpp. roctx identifier 16\n.rocprofiler.cpp..push. thread-[0-3]\n.rocprofiler.cpp..push. thread-[0-3]\n.rocprofiler.cpp..push. thread-[0-3]\n.rocprofiler.cpp..push. thread-[0-3]\n.rocprofiler.cpp..pop. thread-[0-3]\n.rocprofiler.cpp..pop. thread-[0-3]\n.rocprofiler.cpp..pop. thread-[0-3]\n.rocprofiler.cpp..pop. thread-[0-3]\n"
    PRELOAD_WRAP_PASS_REGEX
        ".fwd.hpp. dlopen libamdhip64.so, 1\n.fwd.hpp. dlopen libroctx64.so, 1\n.rocprofiler.cpp. rocp_ctor\n.rocprofiler.cpp. hsa :: 20100 :: 0 :: 1\n.hsa-runtime.cpp. hsa identifier 0\n.rocprofiler.cpp. hsa_init\n.rocprofiler.cpp. hip :: 60001 :: 0 :: 1\n.amdhip.cpp. hip identifier 8\n.rocprofiler.cpp. hip_init\n.rocprofiler.cpp. hip_init\n.rocprofiler.cpp. hip_init\n.rocprofiler.cpp. hip_init\n.rocprofiler.cpp. roctx :: 40601 :: 0 :: 1\n.roctx.cpp. roctx identifier 16\n.rocprofiler.cpp..push. thread-[0-3]\n.rocprofiler.cpp..push. thread-[0-3]\n.rocprofiler.cpp..push. thread-[0-3]\n.rocprofiler.cpp..push. thread-[0-3]\n.rocprofiler.cpp..pop. thread-[0-3]\n.rocprofiler.cpp..pop. thread-[0-3]\n.rocprofiler.cpp..pop. thread-[0-3]\n.rocprofiler.cpp..pop. thread-[0-3]\n"
    )

# constructor
rocp_register_test_executable(
    test-amdhip-ctor
    SOURCES test-amdhip-ctor.cpp
    LIBRARIES rocprofiler-register::tests-strong amdhip::amdhip
    CORE_PASS_REGEX
        ".hsa-runtime.cpp. hsa_init\n.hsa-runtime.cpp. hsa_init\n.amdhip.cpp. hip_init\n"
    PRELOAD_PASS_REGEX
        ".rocprofiler.cpp. rocp_ctor\n.rocprofiler.cpp. hsa :: 20100 :: 0 :: 1\n.hsa-runtime.cpp. hsa identifier 0\n.hsa-runtime.cpp. hsa_init\n.hsa-runtime.cpp. hsa_init\n.rocprofiler.cpp. hip :: 60001 :: 0 :: 1\n.amdhip.cpp. hip identifier 8\n.amdhip.cpp. hip_init\n"
    WRAP_PASS_REGEX
        ".hsa-runtime.cpp. hsa_init\n.hsa-runtime.cpp. hsa_init\n.amdhip.cpp. hip_init\n"
    PRELOAD_WRAP_PASS_REGEX
        ".rocprofiler.cpp. rocp_ctor\n.rocprofiler.cpp. hsa :: 20100 :: 0 :: 1\n.hsa-runtime.cpp. hsa identifier 0\n.rocprofiler.cpp. hsa_init\n.rocprofiler.cpp. hsa_init\n.rocprofiler.cpp. hip :: 60001 :: 0 :: 1\n.amdhip.cpp. hip identifier 8\n.rocprofiler.cpp. hip_init\n"
    )

rocp_register_test_executable(
    test-amdhip-ctor-weak
    SOURCES test-amdhip-ctor.cpp
    LIBRARIES rocprofiler-register::tests-weak
    CORE_PASS_REGEX
        ".fwd.hpp. dlopen libamdhip64.so, 1\n.fwd.hpp. dlopen libhsa-runtime64.so, 1\n.hsa-runtime.cpp. hsa_init\n.hsa-runtime.cpp. hsa_init\n.amdhip.cpp. hip_init\n"
    PRELOAD_PASS_REGEX
        ".fwd.hpp. dlopen libamdhip64.so, 1\n.fwd.hpp. dlopen libhsa-runtime64.so, 1\n.rocprofiler.cpp. rocp_ctor\n.rocprofiler.cpp. hsa :: 20100 :: 0 :: 1\n.hsa-runtime.cpp. hsa identifier 0\n.hsa-runtime.cpp. hsa_init\n.hsa-runtime.cpp. hsa_init\n.rocprofiler.cpp. hip :: 60001 :: 0 :: 1\n.amdhip.cpp. hip identifier 8\n.amdhip.cpp. hip_init\n"
    WRAP_PASS_REGEX
        ".fwd.hpp. dlopen libamdhip64.so, 1\n.fwd.hpp. dlopen libhsa-runtime64.so, 1\n.hsa-runtime.cpp. hsa_init\n.hsa-runtime.cpp. hsa_init\n.amdhip.cpp. hip_init\n"
    PRELOAD_WRAP_PASS_REGEX
        ".fwd.hpp. dlopen libamdhip64.so, 1\n.fwd.hpp. dlopen libhsa-runtime64.so, 1\n.rocprofiler.cpp. rocp_ctor\n.rocprofiler.cpp. hsa :: 20100 :: 0 :: 1\n.hsa-runtime.cpp. hsa identifier 0\n.rocprofiler.cpp. hsa_init\n.rocprofiler.cpp. hsa_init\n.rocprofiler.cpp. hip :: 60001 :: 0 :: 1\n.amdhip.cpp. hip identifier 8\n.rocprofiler.cpp. hip_init\n"
    )

# 41
rocp_register_test_executable(
    test-amdhip-ctor-rocp
    SOURCES test-amdhip-ctor.cpp
    LIBRARIES rocprofiler-register::tests-strong amdhip::amdhip
              rocprofiler-register::tests-rocp
    CORE_PASS_REGEX
        ".rocprofiler.cpp. rocp_ctor\n.rocprofiler.cpp. hsa :: 20100 :: 0 :: 1\n.hsa-runtime.cpp. hsa identifier 0\n.hsa-runtime.cpp. hsa_init\n.hsa-runtime.cpp. hsa_init\n.rocprofiler.cpp. hip :: 60001 :: 0 :: 1\n.amdhip.cpp. hip identifier 8\n.amdhip.cpp. hip_init\n"
    PRELOAD_PASS_REGEX
        ".rocprofiler.cpp. rocp_ctor\n.rocprofiler.cpp. hsa :: 20100 :: 0 :: 1\n.hsa-runtime.cpp. hsa identifier 0\n.hsa-runtime.cpp. hsa_init\n.hsa-runtime.cpp. hsa_init\n.rocprofiler.cpp. hip :: 60001 :: 0 :: 1\n.amdhip.cpp. hip identifier 8\n.amdhip.cpp. hip_init\n"
    WRAP_PASS_REGEX
        ".rocprofiler.cpp. rocp_ctor\n.rocprofiler.cpp. hsa :: 20100 :: 0 :: 1\n.hsa-runtime.cpp. hsa identifier 0\n.rocprofiler.cpp. hsa_init\n.rocprofiler.cpp. hsa_init\n.rocprofiler.cpp. hip :: 60001 :: 0 :: 1\n.amdhip.cpp. hip identifier 8\n.rocprofiler.cpp. hip_init\n"
    PRELOAD_WRAP_PASS_REGEX
        ".rocprofiler.cpp. rocp_ctor\n.rocprofiler.cpp. hsa :: 20100 :: 0 :: 1\n.hsa-runtime.cpp. hsa identifier 0\n.rocprofiler.cpp. hsa_init\n.rocprofiler.cpp. hsa_init\n.rocprofiler.cpp. hip :: 60001 :: 0 :: 1\n.amdhip.cpp. hip identifier 8\n.rocprofiler.cpp. hip_init\n"
    )

rocp_register_test_executable(
    test-amdhip-ctor-weak-rocp
    SOURCES test-amdhip-ctor.cpp
    LIBRARIES rocprofiler-register::tests-weak rocprofiler-register::tests-rocp
    CORE_PASS_REGEX
        ".fwd.hpp. dlopen libamdhip64.so, 1\n.fwd.hpp. dlopen libhsa-runtime64.so, 1\n.rocprofiler.cpp. rocp_ctor\n.rocprofiler.cpp. hsa :: 20100 :: 0 :: 1\n.hsa-runtime.cpp. hsa identifier 0\n.hsa-runtime.cpp. hsa_init\n.hsa-runtime.cpp. hsa_init\n.rocprofiler.cpp. hip :: 60001 :: 0 :: 1\n.amdhip.cpp. hip identifier 8\n.amdhip.cpp. hip_init\n"
    PRELOAD_PASS_REGEX
        ".fwd.hpp. dlopen libamdhip64.so, 1\n.fwd.hpp. dlopen libhsa-runtime64.so, 1\n.rocprofiler.cpp. rocp_ctor\n.rocprofiler.cpp. hsa :: 20100 :: 0 :: 1\n.hsa-runtime.cpp. hsa identifier 0\n.hsa-runtime.cpp. hsa_init\n.hsa-runtime.cpp. hsa_init\n.rocprofiler.cpp. hip :: 60001 :: 0 :: 1\n.amdhip.cpp. hip identifier 8\n.amdhip.cpp. hip_init\n"
    WRAP_PASS_REGEX
        ".fwd.hpp. dlopen libamdhip64.so, 1\n.fwd.hpp. dlopen libhsa-runtime64.so, 1\n.rocprofiler.cpp. rocp_ctor\n.rocprofiler.cpp. hsa :: 20100 :: 0 :: 1\n.hsa-runtime.cpp. hsa identifier 0\n.rocprofiler.cpp. hsa_init\n.rocprofiler.cpp. hsa_init\n.rocprofiler.cpp. hip :: 60001 :: 0 :: 1\n.amdhip.cpp. hip identifier 8\n.rocprofiler.cpp. hip_init\n"
    PRELOAD_WRAP_PASS_REGEX
        ".fwd.hpp. dlopen libamdhip64.so, 1\n.fwd.hpp. dlopen libhsa-runtime64.so, 1\n.rocprofiler.cpp. rocp_ctor\n.rocprofiler.cpp. hsa :: 20100 :: 0 :: 1\n.hsa-runtime.cpp. hsa identifier 0\n.rocprofiler.cpp. hsa_init\n.rocprofiler.cpp. hsa_init\n.rocprofiler.cpp. hip :: 60001 :: 0 :: 1\n.amdhip.cpp. hip identifier 8\n.rocprofiler.cpp. hip_init\n"
    )

# constructor + mt
rocp_register_test_executable(
    test-amdhip-ctor-mt
    SOURCES test-amdhip-ctor-mt.cpp
    LIBRARIES rocprofiler-register::tests-strong amdhip::amdhip
    CORE_PASS_REGEX
        ".hsa-runtime.cpp. hsa_init\n.amdhip.cpp. hip_init\n.amdhip.cpp. hip_init\n.amdhip.cpp. hip_init\n.amdhip.cpp. hip_init\n"
    PRELOAD_PASS_REGEX
        ".rocprofiler.cpp. rocp_ctor\n.rocprofiler.cpp. hsa :: 20100 :: 0 :: 1\n.hsa-runtime.cpp. hsa identifier 0\n.hsa-runtime.cpp. hsa_init\n.rocprofiler.cpp. hip :: 60001 :: 0 :: 1\n.amdhip.cpp. hip identifier 8\n.amdhip.cpp. hip_init\n.amdhip.cpp. hip_init\n.amdhip.cpp. hip_init\n.amdhip.cpp. hip_init\n"
    WRAP_PASS_REGEX
        ".hsa-runtime.cpp. hsa_init\n.amdhip.cpp. hip_init\n.amdhip.cpp. hip_init\n.amdhip.cpp. hip_init\n.amdhip.cpp. hip_init\n"
    PRELOAD_WRAP_PASS_REGEX
        ".rocprofiler.cpp. rocp_ctor\n.rocprofiler.cpp. hsa :: 20100 :: 0 :: 1\n.hsa-runtime.cpp. hsa identifier 0\n.rocprofiler.cpp. hsa_init\n.rocprofiler.cpp. hip :: 60001 :: 0 :: 1\n.amdhip.cpp. hip identifier 8\n.rocprofiler.cpp. hip_init\n.rocprofiler.cpp. hip_init\n.rocprofiler.cpp. hip_init\n.rocprofiler.cpp. hip_init\n"
    )

rocp_register_test_executable(
    test-amdhip-ctor-mt-weak
    SOURCES test-amdhip-ctor-mt.cpp
    LIBRARIES rocprofiler-register::tests-weak
    CORE_PASS_REGEX
        ".fwd.hpp. dlopen libamdhip64.so, 1\n.hsa-runtime.cpp. hsa_init\n.amdhip.cpp. hip_init\n.amdhip.cpp. hip_init\n.amdhip.cpp. hip_init\n.amdhip.cpp. hip_init\n"
    PRELOAD_PASS_REGEX
        ".fwd.hpp. dlopen libamdhip64.so, 1\n.rocprofiler.cpp. rocp_ctor\n.rocprofiler.cpp. hsa :: 20100 :: 0 :: 1\n.hsa-runtime.cpp. hsa identifier 0\n.hsa-runtime.cpp. hsa_init\n.rocprofiler.cpp. hip :: 60001 :: 0 :: 1\n.amdhip.cpp. hip identifier 8\n.amdhip.cpp. hip_init\n.amdhip.cpp. hip_init\n.amdhip.cpp. hip_init\n.amdhip.cpp. hip_init\n"
    WRAP_PASS_REGEX
        ".fwd.hpp. dlopen libamdhip64.so, 1\n.hsa-runtime.cpp. hsa_init\n.amdhip.cpp. hip_init\n.amdhip.cpp. hip_init\n.amdhip.cpp. hip_init\n.amdhip.cpp. hip_init\n"
    PRELOAD_WRAP_PASS_REGEX
        ".fwd.hpp. dlopen libamdhip64.so, 1\n.rocprofiler.cpp. rocp_ctor\n.rocprofiler.cpp. hsa :: 20100 :: 0 :: 1\n.hsa-runtime.cpp. hsa identifier 0\n.rocprofiler.cpp. hsa_init\n.rocprofiler.cpp. hip :: 60001 :: 0 :: 1\n.amdhip.cpp. hip identifier 8\n.rocprofiler.cpp. hip_init\n.rocprofiler.cpp. hip_init\n.rocprofiler.cpp. hip_init\n.rocprofiler.cpp. hip_init\n"
    )

rocp_register_test_executable(
    test-amdhip-ctor-mt-rocp
    SOURCES test-amdhip-ctor-mt.cpp
    LIBRARIES rocprofiler-register::tests-strong amdhip::amdhip
              rocprofiler-register::tests-rocp
    CORE_PASS_REGEX
        ".rocprofiler.cpp. rocp_ctor\n.rocprofiler.cpp. hsa :: 20100 :: 0 :: 1\n.hsa-runtime.cpp. hsa identifier 0\n.hsa-runtime.cpp. hsa_init\n.rocprofiler.cpp. hip :: 60001 :: 0 :: 1\n.amdhip.cpp. hip identifier 8\n.amdhip.cpp. hip_init\n.amdhip.cpp. hip_init\n.amdhip.cpp. hip_init\n.amdhip.cpp. hip_init\n"
    PRELOAD_PASS_REGEX
        ".rocprofiler.cpp. rocp_ctor\n.rocprofiler.cpp. hsa :: 20100 :: 0 :: 1\n.hsa-runtime.cpp. hsa identifier 0\n.hsa-runtime.cpp. hsa_init\n.rocprofiler.cpp. hip :: 60001 :: 0 :: 1\n.amdhip.cpp. hip identifier 8\n.amdhip.cpp. hip_init\n.amdhip.cpp. hip_init\n.amdhip.cpp. hip_init\n.amdhip.cpp. hip_init\n"
    WRAP_PASS_REGEX
        ".rocprofiler.cpp. rocp_ctor\n.rocprofiler.cpp. hsa :: 20100 :: 0 :: 1\n.hsa-runtime.cpp. hsa identifier 0\n.rocprofiler.cpp. hsa_init\n.rocprofiler.cpp. hip :: 60001 :: 0 :: 1\n.amdhip.cpp. hip identifier 8\n.rocprofiler.cpp. hip_init\n.rocprofiler.cpp. hip_init\n.rocprofiler.cpp. hip_init\n.rocprofiler.cpp. hip_init\n"
    PRELOAD_WRAP_PASS_REGEX
        ".rocprofiler.cpp. rocp_ctor\n.rocprofiler.cpp. hsa :: 20100 :: 0 :: 1\n.hsa-runtime.cpp. hsa identifier 0\n.rocprofiler.cpp. hsa_init\n.rocprofiler.cpp. hip :: 60001 :: 0 :: 1\n.amdhip.cpp. hip identifier 8\n.rocprofiler.cpp. hip_init\n.rocprofiler.cpp. hip_init\n.rocprofiler.cpp. hip_init\n.rocprofiler.cpp. hip_init\n"
    )

rocp_register_test_executable(
    test-amdhip-ctor-mt-weak-rocp
    SOURCES test-amdhip-ctor-mt.cpp
    LIBRARIES rocprofiler-register::tests-weak rocprofiler-register::tests-rocp
    CORE_PASS_REGEX
        ".fwd.hpp. dlopen libamdhip64.so, 1\n.rocprofiler.cpp. rocp_ctor\n.rocprofiler.cpp. hsa :: 20100 :: 0 :: 1\n.hsa-runtime.cpp. hsa identifier 0\n.hsa-runtime.cpp. hsa_init\n.rocprofiler.cpp. hip :: 60001 :: 0 :: 1\n.amdhip.cpp. hip identifier 8\n.amdhip.cpp. hip_init\n.amdhip.cpp. hip_init\n.amdhip.cpp. hip_init\n.amdhip.cpp. hip_init\n"
    PRELOAD_PASS_REGEX
        ".fwd.hpp. dlopen libamdhip64.so, 1\n.rocprofiler.cpp. rocp_ctor\n.rocprofiler.cpp. hsa :: 20100 :: 0 :: 1\n.hsa-runtime.cpp. hsa identifier 0\n.hsa-runtime.cpp. hsa_init\n.rocprofiler.cpp. hip :: 60001 :: 0 :: 1\n.amdhip.cpp. hip identifier 8\n.amdhip.cpp. hip_init\n.amdhip.cpp. hip_init\n.amdhip.cpp. hip_init\n.amdhip.cpp. hip_init\n"
    WRAP_PASS_REGEX
        ".fwd.hpp. dlopen libamdhip64.so, 1\n.rocprofiler.cpp. rocp_ctor\n.rocprofiler.cpp. hsa :: 20100 :: 0 :: 1\n.hsa-runtime.cpp. hsa identifier 0\n.rocprofiler.cpp. hsa_init\n.rocprofiler.cpp. hip :: 60001 :: 0 :: 1\n.amdhip.cpp. hip identifier 8\n.rocprofiler.cpp. hip_init\n.rocprofiler.cpp. hip_init\n.rocprofiler.cpp. hip_init\n.rocprofiler.cpp. hip_init\n"
    PRELOAD_WRAP_PASS_REGEX
        ".fwd.hpp. dlopen libamdhip64.so, 1\n.rocprofiler.cpp. rocp_ctor\n.rocprofiler.cpp. hsa :: 20100 :: 0 :: 1\n.hsa-runtime.cpp. hsa identifier 0\n.rocprofiler.cpp. hsa_init\n.rocprofiler.cpp. hip :: 60001 :: 0 :: 1\n.amdhip.cpp. hip identifier 8\n.rocprofiler.cpp. hip_init\n.rocprofiler.cpp. hip_init\n.rocprofiler.cpp. hip_init\n.rocprofiler.cpp. hip_init\n"
    )

add_executable(test-invalid-hsa-runtime)
target_sources(test-invalid-hsa-runtime PRIVATE test-amdhip-roctx.cpp)
target_link_libraries(
    test-invalid-hsa-runtime PRIVATE rocprofiler-register::tests-common roctx::roctx
                                     amdhip::amdhip-invalid)

add_test(
    NAME test-secure-invalid-hsa-runtime
    COMMAND $<TARGET_FILE:test-invalid-hsa-runtime>
    WORKING_DIRECTORY ${PROJECT_BINARY_DIR})

set_tests_properties(
    test-secure-invalid-hsa-runtime
    PROPERTIES
        ENVIRONMENT
        "LD_LIBRARY_PATH=${CMAKE_LIBRARY_OUTPUT_DIRECTORY}/invalid:${CMAKE_LIBRARY_OUTPUT_DIRECTORY};ROCPROFILER_REGISTER_SECURE=1;${ROCPROFILER_REGISTER_MEMCHECK_PRELOAD_ENV}"
        LABELS
        "secure"
        WILL_FAIL
        ON)

add_test(
    NAME test-insecure-invalid-hsa-runtime
    COMMAND $<TARGET_FILE:test-invalid-hsa-runtime>
    WORKING_DIRECTORY ${PROJECT_BINARY_DIR})

set_tests_properties(
    test-insecure-invalid-hsa-runtime
    PROPERTIES
        ENVIRONMENT
        "LD_LIBRARY_PATH=${CMAKE_LIBRARY_OUTPUT_DIRECTORY}/invalid:${CMAKE_LIBRARY_OUTPUT_DIRECTORY};ROCPROFILER_REGISTER_SECURE=0;${ROCPROFILER_REGISTER_MEMCHECK_PRELOAD_ENV}"
        LABELS
        "insecure")

add_executable(test-amdhip-hsart-roctx)
target_sources(test-amdhip-hsart-roctx PRIVATE test-amdhip-hsart-roctx.cpp)
target_link_libraries(
    test-amdhip-hsart-roctx PRIVATE rocprofiler-register::tests-common
                                    hsa-runtime::hsa-runtime amdhip::amdhip roctx::roctx)

add_test(
    NAME test-force-rocprofiler-dlopen
    COMMAND $<TARGET_FILE:test-amdhip-hsart-roctx>
    WORKING_DIRECTORY ${PROJECT_BINARY_DIR})

set_tests_properties(
    test-force-rocprofiler-dlopen
    PROPERTIES
        ENVIRONMENT
        "LD_LIBRARY_PATH=${CMAKE_LIBRARY_OUTPUT_DIRECTORY};ROCPROFILER_REGISTER_FORCE_LOAD=1;ROCPROFILER_REGISTER_SECURE=yes;ROCPROFILER_REGISTER_VERBOSE=3;${ROCPROFILER_REGISTER_MEMCHECK_PRELOAD_ENV}"
        LABELS
        "dlopen;secure")

add_test(
    NAME test-rocprofiler-register-library-base-path
    COMMAND $<TARGET_FILE:test-amdhip-hsart-roctx>
    WORKING_DIRECTORY ${PROJECT_BINARY_DIR})

set_tests_properties(
    test-rocprofiler-register-library-base-path
    PROPERTIES
        ENVIRONMENT
        "LD_LIBRARY_PATH=${CMAKE_LIBRARY_OUTPUT_DIRECTORY};ROCPROFILER_REGISTER_LIBRARY=librocprofiler-sdk.so.0;ROCPROFILER_REGISTER_SECURE=yes;ROCPROFILER_REGISTER_VERBOSE=3;ROCPROFILER_REGISTER_MONOCHROME=true;${ROCPROFILER_REGISTER_MEMCHECK_PRELOAD_ENV}"
        LABELS
        "dlopen;secure")

add_test(
    NAME test-rocprofiler-register-library-absolute-path
    COMMAND $<TARGET_FILE:test-amdhip-hsart-roctx>
    WORKING_DIRECTORY ${PROJECT_BINARY_DIR})

set_tests_properties(
    test-rocprofiler-register-library-absolute-path
    PROPERTIES
        ENVIRONMENT
        "LD_LIBRARY_PATH=${CMAKE_LIBRARY_OUTPUT_DIRECTORY};ROCPROFILER_REGISTER_LIBRARY=${CMAKE_LIBRARY_OUTPUT_DIRECTORY}/librocprofiler-sdk.so.0;ROCPROFILER_REGISTER_SECURE=yes;ROCPROFILER_REGISTER_VERBOSE=3;ROCPROFILER_REGISTER_MONOCHROME=true;${ROCPROFILER_REGISTER_MEMCHECK_PRELOAD_ENV}"
        LABELS
        "dlopen;secure")
