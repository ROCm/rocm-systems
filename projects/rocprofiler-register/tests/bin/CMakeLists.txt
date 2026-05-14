#
#
#

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(Threads REQUIRED)

# WINDOWS-DIVERGENCE: a real HIP runtime is not available on Windows in the
# rocprofiler-register CI footprint, and the simple-hip preload-style
# integration tests below depend on LD_PRELOAD anyway. Skip the find_package
# entirely and disable the preload tests on Windows.
if(NOT WIN32)
    find_package(
        hip
        HINTS
        ${ROCM_PATH}
        ENV
        ROCM_PATH
        /opt/rocm
        PATHS
        ${ROCM_PATH}
        ENV
        ROCM_PATH
        /opt/rocm)
endif()

# WINDOWS-DIVERGENCE: -W -Wall -Wextra -Werror are GCC/Clang flags. MSVC
# uses /W4 /WX; clang-cl accepts both. Use a generator expression so the
# compile options are correct under both toolchains.
set(_simple_hip_warn_opts
    "$<$<NOT:$<CXX_COMPILER_ID:MSVC>>:-W;-Wall;-Wextra;-Werror>"
    "$<$<CXX_COMPILER_ID:MSVC>:/W4;/WX>")

if(hip_FOUND)
    add_executable(simple-hip)
    target_sources(simple-hip PRIVATE simple-hip.cpp)
    target_compile_options(simple-hip PRIVATE ${_simple_hip_warn_opts})
    target_link_libraries(simple-hip PRIVATE Threads::Threads hip::host)
    set(PRELOAD_TESTS_DISABLED OFF)
    install(
        TARGETS simple-hip
        DESTINATION ${CMAKE_INSTALL_DATADIR}/${PROJECT_NAME}/tests/bin
        COMPONENT tests)
else()
    add_executable(simple-hip EXCLUDE_FROM_ALL)
    target_sources(simple-hip PRIVATE simple-hip.cpp)
    target_link_libraries(simple-hip PRIVATE Threads::Threads)
    set(PRELOAD_TESTS_DISABLED ON)
endif()

# WINDOWS-DIVERGENCE: the simple-hip integration tests below run the real HIP
# runtime under LD_PRELOAD-based instrumentation. Both prerequisites (HIP +
# LD_PRELOAD) are absent in the Windows port footprint, so these test cases
# are Linux-only. The simple-hip executable target itself remains available
# for manual smoke-runs on Windows once a Win32-equivalent injection
# mechanism (Detours / AppInit_DLLs) is plumbed in a later phase.
if(NOT WIN32)
    #
    # INSTALL
    #
    set(TEST_DESTDIR "${CMAKE_BINARY_DIR}/tests/install")
    set(simple-hip-install-env "DESTDIR=${TEST_DESTDIR}")

    add_test(
        NAME test-simple-hip-install
        COMMAND ${CMAKE_COMMAND} --build ${CMAKE_BINARY_DIR} --target install --parallel
                4
        WORKING_DIRECTORY ${CMAKE_BINARY_DIR})

    set_tests_properties(
        test-simple-hip-install
        PROPERTIES TIMEOUT
                   120
                   LABELS
                   "integration-test"
                   ENVIRONMENT
                   "${simple-hip-install-env}"
                   DISABLED
                   ${PRELOAD_TESTS_DISABLED}
                   FIXTURES_SETUP
                   "simple-hip-install")

    #
    # NORMAL (no profiling library)
    #
    string(REPLACE "//" "/" TEST_INSTALL_PREFIX
                   "${TEST_DESTDIR}/${CMAKE_INSTALL_PREFIX}")
    set(simple-hip-normal-env
        "LD_LIBRARY_PATH=${TEST_INSTALL_PREFIX}/${CMAKE_INSTALL_LIBDIR}:${ROCM_PATH}/${CMAKE_INSTALL_LIBDIR}"
        "LD_PRELOAD=${ROCPROFILER_REGISTER_MEMCHECK_PRELOAD_LIBRARY}")

    add_test(NAME test-simple-hip-normal COMMAND $<TARGET_FILE:simple-hip>)

    set_tests_properties(
        test-simple-hip-normal
        PROPERTIES TIMEOUT
                   120
                   LABELS
                   "integration-test"
                   ENVIRONMENT
                   "${simple-hip-normal-env}"
                   DEPENDS
                   test-simple-hip-install
                   DISABLED
                   ${PRELOAD_TESTS_DISABLED}
                   FIXTURES_SETUP
                   "simple-hip-works"
                   FIXTURES_REQUIRED
                   "simple-hip-install")

    #
    # PRELOAD
    #
    set(simple-hip-preload-env
        ${simple-hip-normal-env}
        "LD_PRELOAD=${ROCPROFILER_REGISTER_MEMCHECK_PRELOAD_LIBRARY}:$<TARGET_FILE:generic-tool::generic-tool>"
        )

    add_test(NAME test-simple-hip-preload COMMAND $<TARGET_FILE:simple-hip>)

    set_tests_properties(
        test-simple-hip-preload
        PROPERTIES TIMEOUT
                   120
                   LABELS
                   "integration-test"
                   ENVIRONMENT
                   "${simple-hip-preload-env}"
                   DEPENDS
                   test-simple-hip-install
                   DISABLED
                   ${PRELOAD_TESTS_DISABLED}
                   FIXTURES_REQUIRED
                   "simple-hip-install;simple-hip-works")
endif()
