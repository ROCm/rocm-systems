cmake_minimum_required(VERSION 3.16 FATAL_ERROR)

project(rocprofiler-register-tests LANGUAGES C CXX)
set(CMAKE_INSTALL_DEFAULT_COMPONENT_NAME tests)

include(CMakeParseArguments)
include(GNUInstallDirs)

# WINDOWS-DIVERGENCE: the integration-test suite in this directory exercises
# the rocprofiler-register library through Linux-specific runtime mechanisms:
#   1) LD_PRELOAD-injected librocprofiler-register.so + librocprofiler-sdk.so
#      mocks. PE/COFF has no userland equivalent without a heavyweight
#      injection mechanism such as Microsoft Detours or AppInit_DLLs;
#   2) #pragma weak symbol resolution in tests/common/fwd.h. PE/COFF has no
#      weak-symbol concept, so the "weak" scenario maps onto LoadLibrary at
#      runtime instead -- only partial parity is achievable (see fwd.hpp).
#
# The mock libraries (hsa-runtime, amdhip, roctx, rccl, rocdecode, rocjpeg,
# rocprofiler, generic-tool) and the test-binary CMake targets have been made
# Windows-buildable in their respective CMakeLists.txt (SOVERSION guards,
# RUNTIME_OUTPUT_DIRECTORY for invalid variants). The Phase-10 hardening pass
# also ported the C++ integration-test sources off <dlfcn.h>/<pthread.h>:
# unused dlfcn includes are now Linux-only; pthread_barrier_t was replaced
# with a portable std::condition_variable barrier shim in common/fwd.hpp;
# rocprofiler.cpp's dlopen/dlsym/__attribute__((constructor)) were replaced
# with LoadLibraryA/GetProcAddress/static-initializer.
#
# Building integration tests on Windows is therefore opt-in pending an
# injection-mechanism plumbing decision (Detours / AppInit_DLLs) for the
# PRELOAD/PRELOAD_WRAP/ENV scenarios. CORE and WRAP scenarios run without
# preload and are enabled when ROCPROFILER_REGISTER_BUILD_INTEGRATION_TESTS=ON
# is passed. The unit tests under source/lib/rocprofiler-register/details/tests/
# cover the critical Windows code paths (pe_parser, loader, environment,
# logging) and remain enabled by default.
if(WIN32 AND NOT ROCPROFILER_REGISTER_BUILD_INTEGRATION_TESTS)
    message(
        STATUS
            "[${PROJECT_NAME}] integration tests skipped on Windows (set ROCPROFILER_REGISTER_BUILD_INTEGRATION_TESTS=ON to override)"
        )
    return()
endif()

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

if(NOT WIN32)
    if(NOT CMAKE_DL_LIBS)
        find_library(DL_LIBRARY NAMES dl)
        if(NOT DL_LIBRARY)
            set(DL_LIBRARY dl)
        endif()
        set(CMAKE_DL_LIBS ${DL_LIBRARY})
    endif()
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
# WINDOWS-DIVERGENCE: MSVC's CRT supplies the equivalents of -lstdc++ and -ldl
# implicitly; explicit linkage to those library names fails. CMAKE_DL_LIBS is
# empty on Windows so CMake naturally drops it; stdc++ must be guarded.
if(WIN32)
    target_link_libraries(rocprofiler-register-tests-common INTERFACE Threads::Threads)
else()
    target_link_libraries(rocprofiler-register-tests-common
                          INTERFACE Threads::Threads stdc++ ${CMAKE_DL_LIBS})
endif()

add_library(rocprofiler-register-tests-strong INTERFACE)
target_compile_definitions(rocprofiler-register-tests-strong
                           INTERFACE ROCP_REG_TEST_WEAK=0)

add_library(rocprofiler-register-tests-weak INTERFACE)
target_compile_definitions(rocprofiler-register-tests-weak INTERFACE ROCP_REG_TEST_WEAK=1)

add_library(rocprofiler-register-tests-tool INTERFACE)
# WINDOWS-DIVERGENCE: ld --no-as-needed forces the linker to keep DT_NEEDED
# entries even when no symbol from the library is referenced; this is what
# pulls the generic-tool DSO into the test binary so its constructors run.
# Windows PE has no equivalent (the linker always emits an IDT entry for any
# library on the link line, and you cannot reference a DLL with no imported
# symbols). On Windows we instead rely on the test binary explicitly linking
# generic-tool, which forces an import dependency by way of any exported
# symbol the test references. If the linker still elides the DLL because no
# symbol is referenced, the user must add a /INCLUDE: directive (handled in
# the per-test CMake when needed).
if(NOT WIN32)
    target_link_options(rocprofiler-register-tests-tool INTERFACE -Wl,--no-as-needed)
endif()
target_link_libraries(rocprofiler-register-tests-tool
                      INTERFACE generic-tool::generic-tool)

add_library(rocprofiler-register-tests-rocp INTERFACE)
if(NOT WIN32)
    target_link_options(rocprofiler-register-tests-rocp INTERFACE -Wl,--no-as-needed)
endif()
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

    # WINDOWS-DIVERGENCE: Windows uses PATH (not LD_LIBRARY_PATH) for DLL
    # search at LoadLibrary time. There is no LD_PRELOAD analogue without an
    # injection mechanism (Detours / AppInit_DLLs / SetWindowsHookEx) which is
    # out of scope for this phase. PRELOAD/PRELOAD_WRAP variants are skipped
    # on Windows; CORE/WRAP run with PATH-based DLL discovery instead.
    if(WIN32)
        set(_TEST_LIB_PATH_VAR "PATH")
        set(_TEST_LIB_PATH_SEP "\\;")
    else()
        set(_TEST_LIB_PATH_VAR "LD_LIBRARY_PATH")
        set(_TEST_LIB_PATH_SEP ":")
    endif()

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
                       "${_TEST_LIB_PATH_VAR}=${CMAKE_LIBRARY_OUTPUT_DIRECTORY};${_ENV}"
                       PASS_REGULAR_EXPRESSION
                       "${_PASS}"
                       FAIL_REGULAR_EXPRESSION
                       "${_FAIL}"
                       LABELS
                       "${_LABELS}"
                       ${ARGN})
    endfunction()

    if(WIN32)
        # WINDOWS-DIVERGENCE: PE/COFF link.exe strips a DLL from the Import
        # Directory Table when zero symbols are referenced from it. The Linux
        # test harness keeps rocprofiler-register.so + generic-tool.so loaded
        # via LD_PRELOAD (and the Linux link line uses --no-as-needed for the
        # tests-rocp / tests-tool INTERFACE targets). Neither mechanism exists
        # on Windows, so test-*-rocp/-tool executables end up importing only
        # amdhip64.dll / hsa-runtime64.dll directly. amdhip64.dll does pull
        # rocprofiler-register.dll in transitively, but generic-tool.dll never
        # loads, so rocprofiler_configure is never visible to the module-walk
        # in rocp_reg_scan_for_tools(); _found_tool stays false and the mock
        # rocprofiler-sdk.dll is never opened.
        #
        # Mirror the Linux contract by setting ROCP_TOOL_LIBRARIES, which
        # rocprofiler-register's rocp_reg_scan_for_tools() already honors as
        # an alternative discovery path: a non-empty ROCP_TOOL_LIBRARIES makes
        # _force_tool default to true (see rocprofiler_register.cpp ~L450),
        # which in turn makes _found_tool=true and triggers the SDK load
        # (defaults to "rocprofiler-sdk.dll" on Windows -- the mock). The
        # mock's rocprofiler_set_api_table() then LoadLibraries the named
        # tool DLL, mirroring what LD_PRELOAD=...:libgeneric-tool.so achieves
        # on Linux for the -preload / -env variants. No separate
        # ROCPROFILER_REGISTER_FORCE_LOAD=1 needed -- it's implicit.
        # This mirrors the production discovery path (ROCm cooperative-runtime
        # model) and the hip-trace-poc finding that ROCP_TOOL_LIBRARIES is
        # sufficient for ROCm on Windows; no Detours-style injection needed.
        set(_WIN_ROCP_TOOL_ENV "")
        if("${_NAME}" MATCHES "-(rocp|tool)$")
            set(_WIN_ROCP_TOOL_ENV "ROCP_TOOL_LIBRARIES=generic-tool.dll")
        endif()

        # CORE scenario only: the test binary is linked directly against the
        # mock libraries it needs; no preload/inject step is performed.
        rocp_register_add_test(
            ${_NAME} ${_NAME} "${_WIN_ROCP_TOOL_ENV}"
            "${RRTE_CORE_PASS_REGEX}" "${RRTE_CORE_FAIL_REGEX}")

        # WRAP scenario: behavior selected at run-time via env var. No
        # preload required; safe to enable on Windows.
        set(_WIN_WRAP_ENV "ROCP_REG_TEST_WRAP=1")
        if(_WIN_ROCP_TOOL_ENV)
            set(_WIN_WRAP_ENV "${_WIN_WRAP_ENV};${_WIN_ROCP_TOOL_ENV}")
        endif()
        rocp_register_add_test(
            ${_NAME}-wrap ${_NAME}
            "${_WIN_WRAP_ENV}"
            "${RRTE_WRAP_PASS_REGEX}" "${RRTE_WRAP_FAIL_REGEX}")
    else()
        # Build LD_PRELOAD for base test
        set(_BASE_PRELOAD "LD_PRELOAD=librocprofiler-register.so")
        if(ROCPROFILER_REGISTER_MEMCHECK_PRELOAD_ENV)
            set(_BASE_PRELOAD "${ROCPROFILER_REGISTER_MEMCHECK_PRELOAD_ENV}")
        endif()

        rocp_register_add_test(
            ${_NAME} ${_NAME} "${_BASE_PRELOAD}"
            "${RRTE_CORE_PASS_REGEX}" "${RRTE_CORE_FAIL_REGEX}")

        # Build LD_PRELOAD list for -preload tests
        set(_PRELOAD_LIBS "librocprofiler-register.so" "libgeneric-tool.so")
        if(ROCPROFILER_REGISTER_MEMCHECK_PRELOAD_LIBRARY)
            list(PREPEND _PRELOAD_LIBS "${ROCPROFILER_REGISTER_MEMCHECK_PRELOAD_LIBRARY}")
        endif()
        list(JOIN _PRELOAD_LIBS ":" _PRELOAD_STRING)

        rocp_register_add_test(
            ${_NAME}-preload ${_NAME}
            "LD_PRELOAD=${_PRELOAD_STRING}"
            "${RRTE_PRELOAD_PASS_REGEX}" "${RRTE_PRELOAD_FAIL_REGEX}")

        rocp_register_add_test(
            ${_NAME}-env
            ${_NAME}
            "LD_PRELOAD=librocprofiler-register.so;ROCP_TOOL_LIBRARIES=libgeneric-tool.so;${ROCPROFILER_REGISTER_MEMCHECK_PRELOAD_ENV}"
            "${RRTE_PRELOAD_PASS_REGEX}"
            "${RRTE_PRELOAD_FAIL_REGEX}")

        rocp_register_add_test(
            ${_NAME}-wrap ${_NAME}
            "LD_PRELOAD=librocprofiler-register.so;ROCP_REG_TEST_WRAP=1;${ROCPROFILER_REGISTER_MEMCHECK_PRELOAD_ENV}"
            "${RRTE_WRAP_PASS_REGEX}" "${RRTE_WRAP_FAIL_REGEX}")

        rocp_register_add_test(
            ${_NAME}-preload-wrap
            ${_NAME}
            "LD_PRELOAD=${_PRELOAD_STRING};ROCP_REG_TEST_WRAP=1"
            "${RRTE_PRELOAD_WRAP_PASS_REGEX}"
            "${RRTE_PRELOAD_WRAP_FAIL_REGEX}")

        rocp_register_add_test(
            ${_NAME}-env-wrap
            ${_NAME}
            "LD_PRELOAD=librocprofiler-register.so;ROCP_TOOL_LIBRARIES=libgeneric-tool.so;ROCP_REG_TEST_WRAP=1;${ROCPROFILER_REGISTER_MEMCHECK_PRELOAD_ENV}"
            "${RRTE_PRELOAD_WRAP_PASS_REGEX}"
            "${RRTE_PRELOAD_WRAP_FAIL_REGEX}")
    endif()

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
        ".fwd.hpp. dlopen (lib)?amdhip64\.(so|dll), [01]\n.fwd.hpp. dlopen (lib)?roctx64\.(so|dll), [01]\n.hsa-runtime.cpp. hsa_init\n.amdhip.cpp. hip_init\n.roctx.cpp..push. thread-main\n.roctx.cpp..pop. thread-main\n"
    PRELOAD_PASS_REGEX
        ".fwd.hpp. dlopen (lib)?amdhip64\.(so|dll), [01]\n.fwd.hpp. dlopen (lib)?roctx64\.(so|dll), [01]\n.rocprofiler.cpp. rocp_ctor\n.rocprofiler.cpp. hsa :: 20100 :: 0 :: 1\n.hsa-runtime.cpp. hsa identifier 0\n.hsa-runtime.cpp. hsa_init\n.rocprofiler.cpp. hip :: 60001 :: 0 :: 1\n.amdhip.cpp. hip identifier 8\n.amdhip.cpp. hip_init\n.rocprofiler.cpp. roctx :: 40601 :: 0 :: 1\n.roctx.cpp. roctx identifier 16\n.roctx.cpp..push. thread-main\n.roctx.cpp..pop. thread-main\n"
    WRAP_PASS_REGEX
        ".fwd.hpp. dlopen (lib)?amdhip64\.(so|dll), [01]\n.fwd.hpp. dlopen (lib)?roctx64\.(so|dll), [01]\n.hsa-runtime.cpp. hsa_init\n.amdhip.cpp. hip_init\n.roctx.cpp..push. thread-main\n.roctx.cpp..pop. thread-main\n"
    PRELOAD_WRAP_PASS_REGEX
        ".fwd.hpp. dlopen (lib)?amdhip64\.(so|dll), [01]\n.fwd.hpp. dlopen (lib)?roctx64\.(so|dll), [01]\n.rocprofiler.cpp. rocp_ctor\n.rocprofiler.cpp. hsa :: 20100 :: 0 :: 1\n.hsa-runtime.cpp. hsa identifier 0\n.rocprofiler.cpp. hsa_init\n.rocprofiler.cpp. hip :: 60001 :: 0 :: 1\n.amdhip.cpp. hip identifier 8\n.rocprofiler.cpp. hip_init\n.rocprofiler.cpp. roctx :: 40601 :: 0 :: 1\n.roctx.cpp. roctx identifier 16\n.rocprofiler.cpp..push. thread-main\n.rocprofiler.cpp..pop. thread-main\n"
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
        ".fwd.hpp. dlopen (lib)?amdhip64\.(so|dll), [01]\n.fwd.hpp. dlopen (lib)?roctx64\.(so|dll), [01]\n.rocprofiler.cpp. rocp_ctor\n.rocprofiler.cpp. hsa :: 20100 :: 0 :: 1\n.hsa-runtime.cpp. hsa identifier 0\n.hsa-runtime.cpp. hsa_init\n.rocprofiler.cpp. hip :: 60001 :: 0 :: 1\n.amdhip.cpp. hip identifier 8\n.amdhip.cpp. hip_init\n.rocprofiler.cpp. roctx :: 40601 :: 0 :: 1\n.roctx.cpp. roctx identifier 16\n.roctx.cpp..push. thread-main\n.roctx.cpp..pop. thread-main\n"
    PRELOAD_PASS_REGEX
        ".fwd.hpp. dlopen (lib)?amdhip64\.(so|dll), [01]\n.fwd.hpp. dlopen (lib)?roctx64\.(so|dll), [01]\n.rocprofiler.cpp. rocp_ctor\n.rocprofiler.cpp. hsa :: 20100 :: 0 :: 1\n.hsa-runtime.cpp. hsa identifier 0\n.hsa-runtime.cpp. hsa_init\n.rocprofiler.cpp. hip :: 60001 :: 0 :: 1\n.amdhip.cpp. hip identifier 8\n.amdhip.cpp. hip_init\n.rocprofiler.cpp. roctx :: 40601 :: 0 :: 1\n.roctx.cpp. roctx identifier 16\n.roctx.cpp..push. thread-main\n.roctx.cpp..pop. thread-main\n"
    WRAP_PASS_REGEX
        ".fwd.hpp. dlopen (lib)?amdhip64\.(so|dll), [01]\n.fwd.hpp. dlopen (lib)?roctx64\.(so|dll), [01]\n.rocprofiler.cpp. rocp_ctor\n.rocprofiler.cpp. hsa :: 20100 :: 0 :: 1\n.hsa-runtime.cpp. hsa identifier 0\n.rocprofiler.cpp. hsa_init\n.rocprofiler.cpp. hip :: 60001 :: 0 :: 1\n.amdhip.cpp. hip identifier 8\n.rocprofiler.cpp. hip_init\n.rocprofiler.cpp. roctx :: 40601 :: 0 :: 1\n.roctx.cpp. roctx identifier 16\n.rocprofiler.cpp..push. thread-main\n.rocprofiler.cpp..pop. thread-main\n"
    PRELOAD_WRAP_PASS_REGEX
        ".fwd.hpp. dlopen (lib)?amdhip64\.(so|dll), [01]\n.fwd.hpp. dlopen (lib)?roctx64\.(so|dll), [01]\n.rocprofiler.cpp. rocp_ctor\n.rocprofiler.cpp. hsa :: 20100 :: 0 :: 1\n.hsa-runtime.cpp. hsa identifier 0\n.rocprofiler.cpp. hsa_init\n.rocprofiler.cpp. hip :: 60001 :: 0 :: 1\n.amdhip.cpp. hip identifier 8\n.rocprofiler.cpp. hip_init\n.rocprofiler.cpp. roctx :: 40601 :: 0 :: 1\n.roctx.cpp. roctx identifier 16\n.rocprofiler.cpp..push. thread-main\n.rocprofiler.cpp..pop. thread-main\n"
    )

rocp_register_test_executable(
    test-amdhip-roctx-weak-rocp
    SOURCES test-amdhip-roctx.cpp
    LIBRARIES rocprofiler-register::tests-weak rocprofiler-register::tests-rocp
    CORE_PASS_REGEX
        ".fwd.hpp. dlopen (lib)?amdhip64\.(so|dll), [01]\n.fwd.hpp. dlopen (lib)?roctx64\.(so|dll), [01]\n.rocprofiler.cpp. rocp_ctor\n.rocprofiler.cpp. hsa :: 20100 :: 0 :: 1\n.hsa-runtime.cpp. hsa identifier 0\n.hsa-runtime.cpp. hsa_init\n.rocprofiler.cpp. hip :: 60001 :: 0 :: 1\n.amdhip.cpp. hip identifier 8\n.amdhip.cpp. hip_init\n.rocprofiler.cpp. roctx :: 40601 :: 0 :: 1\n.roctx.cpp. roctx identifier 16\n.roctx.cpp..push. thread-main\n.roctx.cpp..pop. thread-main\n"
    PRELOAD_PASS_REGEX
        ".fwd.hpp. dlopen (lib)?amdhip64\.(so|dll), [01]\n.fwd.hpp. dlopen (lib)?roctx64\.(so|dll), [01]\n.rocprofiler.cpp. rocp_ctor\n.rocprofiler.cpp. hsa :: 20100 :: 0 :: 1\n.hsa-runtime.cpp. hsa identifier 0\n.hsa-runtime.cpp. hsa_init\n.rocprofiler.cpp. hip :: 60001 :: 0 :: 1\n.amdhip.cpp. hip identifier 8\n.amdhip.cpp. hip_init\n.rocprofiler.cpp. roctx :: 40601 :: 0 :: 1\n.roctx.cpp. roctx identifier 16\n.roctx.cpp..push. thread-main\n.roctx.cpp..pop. thread-main\n"
    WRAP_PASS_REGEX
        ".fwd.hpp. dlopen (lib)?amdhip64\.(so|dll), [01]\n.fwd.hpp. dlopen (lib)?roctx64\.(so|dll), [01]\n.rocprofiler.cpp. rocp_ctor\n.rocprofiler.cpp. hsa :: 20100 :: 0 :: 1\n.hsa-runtime.cpp. hsa identifier 0\n.rocprofiler.cpp. hsa_init\n.rocprofiler.cpp. hip :: 60001 :: 0 :: 1\n.amdhip.cpp. hip identifier 8\n.rocprofiler.cpp. hip_init\n.rocprofiler.cpp. roctx :: 40601 :: 0 :: 1\n.roctx.cpp. roctx identifier 16\n.rocprofiler.cpp..push. thread-main\n.rocprofiler.cpp..pop. thread-main\n"
    PRELOAD_WRAP_PASS_REGEX
        ".fwd.hpp. dlopen (lib)?amdhip64\.(so|dll), [01]\n.fwd.hpp. dlopen (lib)?roctx64\.(so|dll), [01]\n.rocprofiler.cpp. rocp_ctor\n.rocprofiler.cpp. hsa :: 20100 :: 0 :: 1\n.hsa-runtime.cpp. hsa identifier 0\n.rocprofiler.cpp. hsa_init\n.rocprofiler.cpp. hip :: 60001 :: 0 :: 1\n.amdhip.cpp. hip identifier 8\n.rocprofiler.cpp. hip_init\n.rocprofiler.cpp. roctx :: 40601 :: 0 :: 1\n.roctx.cpp. roctx identifier 16\n.rocprofiler.cpp..push. thread-main\n.rocprofiler.cpp..pop. thread-main\n"
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
        ".fwd.hpp. dlopen (lib)?amdhip64\.(so|dll), [01]\n.fwd.hpp. dlopen (lib)?roctx64\.(so|dll), [01]\n.hsa-runtime.cpp. hsa_init\n.amdhip.cpp. hip_init\n.amdhip.cpp. hip_init\n.amdhip.cpp. hip_init\n.amdhip.cpp. hip_init\n.roctx.cpp..push. thread-[0-3]\n.roctx.cpp..push. thread-[0-3]\n.roctx.cpp..push. thread-[0-3]\n.roctx.cpp..push. thread-[0-3]\n.roctx.cpp..pop. thread-[0-3]\n.roctx.cpp..pop. thread-[0-3]\n.roctx.cpp..pop. thread-[0-3]\n.roctx.cpp..pop. thread-[0-3]\n"
    PRELOAD_PASS_REGEX
        ".fwd.hpp. dlopen (lib)?amdhip64\.(so|dll), [01]\n.fwd.hpp. dlopen (lib)?roctx64\.(so|dll), [01]\n.rocprofiler.cpp. rocp_ctor\n.rocprofiler.cpp. hsa :: 20100 :: 0 :: 1\n.hsa-runtime.cpp. hsa identifier 0\n.hsa-runtime.cpp. hsa_init\n.rocprofiler.cpp. hip :: 60001 :: 0 :: 1\n.amdhip.cpp. hip identifier 8\n.amdhip.cpp. hip_init\n.amdhip.cpp. hip_init\n.amdhip.cpp. hip_init\n.amdhip.cpp. hip_init\n.rocprofiler.cpp. roctx :: 40601 :: 0 :: 1\n.roctx.cpp. roctx identifier 16\n.roctx.cpp..push. thread-[0-3]\n.roctx.cpp..push. thread-[0-3]\n.roctx.cpp..push. thread-[0-3]\n.roctx.cpp..push. thread-[0-3]\n.roctx.cpp..pop. thread-[0-3]\n.roctx.cpp..pop. thread-[0-3]\n.roctx.cpp..pop. thread-[0-3]\n.roctx.cpp..pop. thread-[0-3]\n"
    WRAP_PASS_REGEX
        ".fwd.hpp. dlopen (lib)?amdhip64\.(so|dll), [01]\n.fwd.hpp. dlopen (lib)?roctx64\.(so|dll), [01]\n.hsa-runtime.cpp. hsa_init\n.amdhip.cpp. hip_init\n.amdhip.cpp. hip_init\n.amdhip.cpp. hip_init\n.amdhip.cpp. hip_init\n.roctx.cpp..push. thread-[0-3]\n.roctx.cpp..push. thread-[0-3]\n.roctx.cpp..push. thread-[0-3]\n.roctx.cpp..push. thread-[0-3]\n.roctx.cpp..pop. thread-[0-3]\n.roctx.cpp..pop. thread-[0-3]\n.roctx.cpp..pop. thread-[0-3]\n.roctx.cpp..pop. thread-[0-3]\n"
    PRELOAD_WRAP_PASS_REGEX
        ".fwd.hpp. dlopen (lib)?amdhip64\.(so|dll), [01]\n.fwd.hpp. dlopen (lib)?roctx64\.(so|dll), [01]\n.rocprofiler.cpp. rocp_ctor\n.rocprofiler.cpp. hsa :: 20100 :: 0 :: 1\n.hsa-runtime.cpp. hsa identifier 0\n.rocprofiler.cpp. hsa_init\n.rocprofiler.cpp. hip :: 60001 :: 0 :: 1\n.amdhip.cpp. hip identifier 8\n.rocprofiler.cpp. hip_init\n.rocprofiler.cpp. hip_init\n.rocprofiler.cpp. hip_init\n.rocprofiler.cpp. hip_init\n.rocprofiler.cpp. roctx :: 40601 :: 0 :: 1\n.roctx.cpp. roctx identifier 16\n.rocprofiler.cpp..push. thread-[0-3]\n.rocprofiler.cpp..push. thread-[0-3]\n.rocprofiler.cpp..push. thread-[0-3]\n.rocprofiler.cpp..push. thread-[0-3]\n.rocprofiler.cpp..pop. thread-[0-3]\n.rocprofiler.cpp..pop. thread-[0-3]\n.rocprofiler.cpp..pop. thread-[0-3]\n.rocprofiler.cpp..pop. thread-[0-3]\n"
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
        ".fwd.hpp. dlopen (lib)?amdhip64\.(so|dll), [01]\n.fwd.hpp. dlopen (lib)?roctx64\.(so|dll), [01]\n.rocprofiler.cpp. rocp_ctor\n.rocprofiler.cpp. hsa :: 20100 :: 0 :: 1\n.hsa-runtime.cpp. hsa identifier 0\n.hsa-runtime.cpp. hsa_init\n.rocprofiler.cpp. hip :: 60001 :: 0 :: 1\n.amdhip.cpp. hip identifier 8\n.amdhip.cpp. hip_init\n.amdhip.cpp. hip_init\n.amdhip.cpp. hip_init\n.amdhip.cpp. hip_init\n.rocprofiler.cpp. roctx :: 40601 :: 0 :: 1\n.roctx.cpp. roctx identifier 16\n.roctx.cpp..push. thread-[0-3]\n.roctx.cpp..push. thread-[0-3]\n.roctx.cpp..push. thread-[0-3]\n.roctx.cpp..push. thread-[0-3]\n.roctx.cpp..pop. thread-[0-3]\n.roctx.cpp..pop. thread-[0-3]\n.roctx.cpp..pop. thread-[0-3]\n.roctx.cpp..pop. thread-[0-3]\n"
    PRELOAD_PASS_REGEX
        ".fwd.hpp. dlopen (lib)?amdhip64\.(so|dll), [01]\n.fwd.hpp. dlopen (lib)?roctx64\.(so|dll), [01]\n.rocprofiler.cpp. rocp_ctor\n.rocprofiler.cpp. hsa :: 20100 :: 0 :: 1\n.hsa-runtime.cpp. hsa identifier 0\n.hsa-runtime.cpp. hsa_init\n.rocprofiler.cpp. hip :: 60001 :: 0 :: 1\n.amdhip.cpp. hip identifier 8\n.amdhip.cpp. hip_init\n.amdhip.cpp. hip_init\n.amdhip.cpp. hip_init\n.amdhip.cpp. hip_init\n.rocprofiler.cpp. roctx :: 40601 :: 0 :: 1\n.roctx.cpp. roctx identifier 16\n.roctx.cpp..push. thread-[0-3]\n.roctx.cpp..push. thread-[0-3]\n.roctx.cpp..push. thread-[0-3]\n.roctx.cpp..push. thread-[0-3]\n.roctx.cpp..pop. thread-[0-3]\n.roctx.cpp..pop. thread-[0-3]\n.roctx.cpp..pop. thread-[0-3]\n.roctx.cpp..pop. thread-[0-3]\n"
    WRAP_PASS_REGEX
        ".fwd.hpp. dlopen (lib)?amdhip64\.(so|dll), [01]\n.fwd.hpp. dlopen (lib)?roctx64\.(so|dll), [01]\n.rocprofiler.cpp. rocp_ctor\n.rocprofiler.cpp. hsa :: 20100 :: 0 :: 1\n.hsa-runtime.cpp. hsa identifier 0\n.rocprofiler.cpp. hsa_init\n.rocprofiler.cpp. hip :: 60001 :: 0 :: 1\n.amdhip.cpp. hip identifier 8\n.rocprofiler.cpp. hip_init\n.rocprofiler.cpp. hip_init\n.rocprofiler.cpp. hip_init\n.rocprofiler.cpp. hip_init\n.rocprofiler.cpp. roctx :: 40601 :: 0 :: 1\n.roctx.cpp. roctx identifier 16\n.rocprofiler.cpp..push. thread-[0-3]\n.rocprofiler.cpp..push. thread-[0-3]\n.rocprofiler.cpp..push. thread-[0-3]\n.rocprofiler.cpp..push. thread-[0-3]\n.rocprofiler.cpp..pop. thread-[0-3]\n.rocprofiler.cpp..pop. thread-[0-3]\n.rocprofiler.cpp..pop. thread-[0-3]\n.rocprofiler.cpp..pop. thread-[0-3]\n"
    PRELOAD_WRAP_PASS_REGEX
        ".fwd.hpp. dlopen (lib)?amdhip64\.(so|dll), [01]\n.fwd.hpp. dlopen (lib)?roctx64\.(so|dll), [01]\n.rocprofiler.cpp. rocp_ctor\n.rocprofiler.cpp. hsa :: 20100 :: 0 :: 1\n.hsa-runtime.cpp. hsa identifier 0\n.rocprofiler.cpp. hsa_init\n.rocprofiler.cpp. hip :: 60001 :: 0 :: 1\n.amdhip.cpp. hip identifier 8\n.rocprofiler.cpp. hip_init\n.rocprofiler.cpp. hip_init\n.rocprofiler.cpp. hip_init\n.rocprofiler.cpp. hip_init\n.rocprofiler.cpp. roctx :: 40601 :: 0 :: 1\n.roctx.cpp. roctx identifier 16\n.rocprofiler.cpp..push. thread-[0-3]\n.rocprofiler.cpp..push. thread-[0-3]\n.rocprofiler.cpp..push. thread-[0-3]\n.rocprofiler.cpp..push. thread-[0-3]\n.rocprofiler.cpp..pop. thread-[0-3]\n.rocprofiler.cpp..pop. thread-[0-3]\n.rocprofiler.cpp..pop. thread-[0-3]\n.rocprofiler.cpp..pop. thread-[0-3]\n"
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
        ".fwd.hpp. dlopen (lib)?amdhip64\.(so|dll), [01]\n.fwd.hpp. dlopen (lib)?hsa-runtime64\.(so|dll), [01]\n.hsa-runtime.cpp. hsa_init\n.hsa-runtime.cpp. hsa_init\n.amdhip.cpp. hip_init\n"
    PRELOAD_PASS_REGEX
        ".fwd.hpp. dlopen (lib)?amdhip64\.(so|dll), [01]\n.fwd.hpp. dlopen (lib)?hsa-runtime64\.(so|dll), [01]\n.rocprofiler.cpp. rocp_ctor\n.rocprofiler.cpp. hsa :: 20100 :: 0 :: 1\n.hsa-runtime.cpp. hsa identifier 0\n.hsa-runtime.cpp. hsa_init\n.hsa-runtime.cpp. hsa_init\n.rocprofiler.cpp. hip :: 60001 :: 0 :: 1\n.amdhip.cpp. hip identifier 8\n.amdhip.cpp. hip_init\n"
    WRAP_PASS_REGEX
        ".fwd.hpp. dlopen (lib)?amdhip64\.(so|dll), [01]\n.fwd.hpp. dlopen (lib)?hsa-runtime64\.(so|dll), [01]\n.hsa-runtime.cpp. hsa_init\n.hsa-runtime.cpp. hsa_init\n.amdhip.cpp. hip_init\n"
    PRELOAD_WRAP_PASS_REGEX
        ".fwd.hpp. dlopen (lib)?amdhip64\.(so|dll), [01]\n.fwd.hpp. dlopen (lib)?hsa-runtime64\.(so|dll), [01]\n.rocprofiler.cpp. rocp_ctor\n.rocprofiler.cpp. hsa :: 20100 :: 0 :: 1\n.hsa-runtime.cpp. hsa identifier 0\n.rocprofiler.cpp. hsa_init\n.rocprofiler.cpp. hsa_init\n.rocprofiler.cpp. hip :: 60001 :: 0 :: 1\n.amdhip.cpp. hip identifier 8\n.rocprofiler.cpp. hip_init\n"
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
        ".fwd.hpp. dlopen (lib)?amdhip64\.(so|dll), [01]\n.fwd.hpp. dlopen (lib)?hsa-runtime64\.(so|dll), [01]\n.rocprofiler.cpp. rocp_ctor\n.rocprofiler.cpp. hsa :: 20100 :: 0 :: 1\n.hsa-runtime.cpp. hsa identifier 0\n.hsa-runtime.cpp. hsa_init\n.hsa-runtime.cpp. hsa_init\n.rocprofiler.cpp. hip :: 60001 :: 0 :: 1\n.amdhip.cpp. hip identifier 8\n.amdhip.cpp. hip_init\n"
    PRELOAD_PASS_REGEX
        ".fwd.hpp. dlopen (lib)?amdhip64\.(so|dll), [01]\n.fwd.hpp. dlopen (lib)?hsa-runtime64\.(so|dll), [01]\n.rocprofiler.cpp. rocp_ctor\n.rocprofiler.cpp. hsa :: 20100 :: 0 :: 1\n.hsa-runtime.cpp. hsa identifier 0\n.hsa-runtime.cpp. hsa_init\n.hsa-runtime.cpp. hsa_init\n.rocprofiler.cpp. hip :: 60001 :: 0 :: 1\n.amdhip.cpp. hip identifier 8\n.amdhip.cpp. hip_init\n"
    WRAP_PASS_REGEX
        ".fwd.hpp. dlopen (lib)?amdhip64\.(so|dll), [01]\n.fwd.hpp. dlopen (lib)?hsa-runtime64\.(so|dll), [01]\n.rocprofiler.cpp. rocp_ctor\n.rocprofiler.cpp. hsa :: 20100 :: 0 :: 1\n.hsa-runtime.cpp. hsa identifier 0\n.rocprofiler.cpp. hsa_init\n.rocprofiler.cpp. hsa_init\n.rocprofiler.cpp. hip :: 60001 :: 0 :: 1\n.amdhip.cpp. hip identifier 8\n.rocprofiler.cpp. hip_init\n"
    PRELOAD_WRAP_PASS_REGEX
        ".fwd.hpp. dlopen (lib)?amdhip64\.(so|dll), [01]\n.fwd.hpp. dlopen (lib)?hsa-runtime64\.(so|dll), [01]\n.rocprofiler.cpp. rocp_ctor\n.rocprofiler.cpp. hsa :: 20100 :: 0 :: 1\n.hsa-runtime.cpp. hsa identifier 0\n.rocprofiler.cpp. hsa_init\n.rocprofiler.cpp. hsa_init\n.rocprofiler.cpp. hip :: 60001 :: 0 :: 1\n.amdhip.cpp. hip identifier 8\n.rocprofiler.cpp. hip_init\n"
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
        ".fwd.hpp. dlopen (lib)?amdhip64\.(so|dll), [01]\n.hsa-runtime.cpp. hsa_init\n.amdhip.cpp. hip_init\n.amdhip.cpp. hip_init\n.amdhip.cpp. hip_init\n.amdhip.cpp. hip_init\n"
    PRELOAD_PASS_REGEX
        ".fwd.hpp. dlopen (lib)?amdhip64\.(so|dll), [01]\n.rocprofiler.cpp. rocp_ctor\n.rocprofiler.cpp. hsa :: 20100 :: 0 :: 1\n.hsa-runtime.cpp. hsa identifier 0\n.hsa-runtime.cpp. hsa_init\n.rocprofiler.cpp. hip :: 60001 :: 0 :: 1\n.amdhip.cpp. hip identifier 8\n.amdhip.cpp. hip_init\n.amdhip.cpp. hip_init\n.amdhip.cpp. hip_init\n.amdhip.cpp. hip_init\n"
    WRAP_PASS_REGEX
        ".fwd.hpp. dlopen (lib)?amdhip64\.(so|dll), [01]\n.hsa-runtime.cpp. hsa_init\n.amdhip.cpp. hip_init\n.amdhip.cpp. hip_init\n.amdhip.cpp. hip_init\n.amdhip.cpp. hip_init\n"
    PRELOAD_WRAP_PASS_REGEX
        ".fwd.hpp. dlopen (lib)?amdhip64\.(so|dll), [01]\n.rocprofiler.cpp. rocp_ctor\n.rocprofiler.cpp. hsa :: 20100 :: 0 :: 1\n.hsa-runtime.cpp. hsa identifier 0\n.rocprofiler.cpp. hsa_init\n.rocprofiler.cpp. hip :: 60001 :: 0 :: 1\n.amdhip.cpp. hip identifier 8\n.rocprofiler.cpp. hip_init\n.rocprofiler.cpp. hip_init\n.rocprofiler.cpp. hip_init\n.rocprofiler.cpp. hip_init\n"
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
        ".fwd.hpp. dlopen (lib)?amdhip64\.(so|dll), [01]\n.rocprofiler.cpp. rocp_ctor\n.rocprofiler.cpp. hsa :: 20100 :: 0 :: 1\n.hsa-runtime.cpp. hsa identifier 0\n.hsa-runtime.cpp. hsa_init\n.rocprofiler.cpp. hip :: 60001 :: 0 :: 1\n.amdhip.cpp. hip identifier 8\n.amdhip.cpp. hip_init\n.amdhip.cpp. hip_init\n.amdhip.cpp. hip_init\n.amdhip.cpp. hip_init\n"
    PRELOAD_PASS_REGEX
        ".fwd.hpp. dlopen (lib)?amdhip64\.(so|dll), [01]\n.rocprofiler.cpp. rocp_ctor\n.rocprofiler.cpp. hsa :: 20100 :: 0 :: 1\n.hsa-runtime.cpp. hsa identifier 0\n.hsa-runtime.cpp. hsa_init\n.rocprofiler.cpp. hip :: 60001 :: 0 :: 1\n.amdhip.cpp. hip identifier 8\n.amdhip.cpp. hip_init\n.amdhip.cpp. hip_init\n.amdhip.cpp. hip_init\n.amdhip.cpp. hip_init\n"
    WRAP_PASS_REGEX
        ".fwd.hpp. dlopen (lib)?amdhip64\.(so|dll), [01]\n.rocprofiler.cpp. rocp_ctor\n.rocprofiler.cpp. hsa :: 20100 :: 0 :: 1\n.hsa-runtime.cpp. hsa identifier 0\n.rocprofiler.cpp. hsa_init\n.rocprofiler.cpp. hip :: 60001 :: 0 :: 1\n.amdhip.cpp. hip identifier 8\n.rocprofiler.cpp. hip_init\n.rocprofiler.cpp. hip_init\n.rocprofiler.cpp. hip_init\n.rocprofiler.cpp. hip_init\n"
    PRELOAD_WRAP_PASS_REGEX
        ".fwd.hpp. dlopen (lib)?amdhip64\.(so|dll), [01]\n.rocprofiler.cpp. rocp_ctor\n.rocprofiler.cpp. hsa :: 20100 :: 0 :: 1\n.hsa-runtime.cpp. hsa identifier 0\n.rocprofiler.cpp. hsa_init\n.rocprofiler.cpp. hip :: 60001 :: 0 :: 1\n.amdhip.cpp. hip identifier 8\n.rocprofiler.cpp. hip_init\n.rocprofiler.cpp. hip_init\n.rocprofiler.cpp. hip_init\n.rocprofiler.cpp. hip_init\n"
    )

add_executable(test-invalid-hsa-runtime)
target_sources(test-invalid-hsa-runtime PRIVATE test-amdhip-roctx.cpp)
target_link_libraries(
    test-invalid-hsa-runtime PRIVATE rocprofiler-register::tests-common roctx::roctx
                                     amdhip::amdhip-invalid)

add_executable(test-amdhip-hsart-roctx)
target_sources(test-amdhip-hsart-roctx PRIVATE test-amdhip-hsart-roctx.cpp)
target_link_libraries(
    test-amdhip-hsart-roctx PRIVATE rocprofiler-register::tests-common
                                    hsa-runtime::hsa-runtime amdhip::amdhip roctx::roctx)

# WINDOWS-DIVERGENCE: the following tests pin ROCPROFILER_REGISTER_LIBRARY,
# ROCPROFILER_REGISTER_FORCE_LOAD, and the secure/insecure/invalid-hsa
# scenarios via LD_PRELOAD. They depend on the loader semantics that
# LD_PRELOAD provides (forcing librocprofiler-register.so into the address
# space before rocprofiler-register's own dlopen runs). PE/COFF has no
# equivalent without an injection mechanism, so these scenarios are
# Linux-only in this phase. The CMake targets (test-invalid-hsa-runtime,
# test-amdhip-hsart-roctx) still build on Windows so the executables are
# verifiable by hand-running with PATH set appropriately.
if(NOT WIN32)
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

    add_test(
        NAME test-force-rocprofiler-dlopen
        COMMAND $<TARGET_FILE:test-amdhip-hsart-roctx>
        WORKING_DIRECTORY ${PROJECT_BINARY_DIR})

    set_tests_properties(
        test-force-rocprofiler-dlopen
        PROPERTIES
            ENVIRONMENT
            "LD_LIBRARY_PATH=${CMAKE_LIBRARY_OUTPUT_DIRECTORY};LD_PRELOAD=librocprofiler-register.so;ROCPROFILER_REGISTER_FORCE_LOAD=1;ROCPROFILER_REGISTER_SECURE=yes;ROCPROFILER_REGISTER_VERBOSE=3;${ROCPROFILER_REGISTER_MEMCHECK_PRELOAD_ENV}"
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
            "LD_LIBRARY_PATH=${CMAKE_LIBRARY_OUTPUT_DIRECTORY};LD_PRELOAD=librocprofiler-register.so;ROCPROFILER_REGISTER_LIBRARY=librocprofiler-sdk.so.0;ROCPROFILER_REGISTER_SECURE=yes;ROCPROFILER_REGISTER_VERBOSE=3;ROCPROFILER_REGISTER_MONOCHROME=true;${ROCPROFILER_REGISTER_MEMCHECK_PRELOAD_ENV}"
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
            "LD_LIBRARY_PATH=${CMAKE_LIBRARY_OUTPUT_DIRECTORY};LD_PRELOAD=librocprofiler-register.so;ROCPROFILER_REGISTER_LIBRARY=${CMAKE_LIBRARY_OUTPUT_DIRECTORY}/librocprofiler-sdk.so.0;ROCPROFILER_REGISTER_SECURE=yes;ROCPROFILER_REGISTER_VERBOSE=3;ROCPROFILER_REGISTER_MONOCHROME=true;${ROCPROFILER_REGISTER_MEMCHECK_PRELOAD_ENV}"
            LABELS
            "dlopen;secure")
endif()
