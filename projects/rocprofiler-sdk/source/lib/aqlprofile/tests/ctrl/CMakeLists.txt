#
# aqlprofile ctrl test (migrated from projects/aqlprofile/test).
#
# Builds the legacy env-var-driven `ctrl` executable and registers one CTest
# entry per exercised mode (dry run / PMC / SQTT / PCSMP), mirroring run.sh.
# run.sh / run_priv.sh are still deployed next to the binary for manual driving.
#
rocprofiler_deactivate_clang_tidy()

# Compile the simple_convolution OpenCL kernel into per-arch code objects placed next to
# the test binary, where TestHsa loads "<gfxip>_simple_convolution.hsaco".
find_program(
    amdclangpp_EXECUTABLE REQUIRED
    NAMES amdclang++
    HINTS ${ROCM_PATH} ENV ROCM_PATH /opt/rocm
    PATHS ${ROCM_PATH} ENV ROCM_PATH /opt/rocm
    PATH_SUFFIXES bin llvm/bin NO_CACHE)

function(generate_hsaco TARGET_ID INPUT_FILE OUTPUT_FILE)
    separate_arguments(
        CLANG_ARG_LIST
        UNIX_COMMAND
        "-O2 -x cl -Xclang -finclude-default-header -cl-denorms-are-zero -cl-std=CL2.0 -Wl,--build-id=sha1
    -target amdgcn-amd-amdhsa -mcpu=${TARGET_ID} -o ${OUTPUT_FILE} ${INPUT_FILE}")
    add_custom_command(
        OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/${OUTPUT_FILE}
        COMMAND ${amdclangpp_EXECUTABLE} ${CLANG_ARG_LIST}
        DEPENDS ${INPUT_FILE}
        WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
        COMMENT "Building ${OUTPUT_FILE}...")
    set(HSACO_TARGET_LIST
        ${HSACO_TARGET_LIST} ${CMAKE_CURRENT_BINARY_DIR}/${OUTPUT_FILE}
        PARENT_SCOPE)
endfunction(generate_hsaco)

foreach(target_id ${GPU_TARGETS})
    generate_hsaco(
        ${target_id} ${CMAKE_CURRENT_SOURCE_DIR}/simple_convolution/simple_convolution.cl
        ${target_id}_simple_convolution.hsaco)
endforeach()

add_custom_target(aqlprofile-ctrl-test-hsaco DEPENDS ${HSACO_TARGET_LIST})

# hsa_rsrc_factory.cpp under util/ is a symlink to the migrated factory that is already
# compiled into rocprofiler-sdk-aqlprofile, so it is NOT listed here.
set(AQLPROFILE_CTRL_TEST_SOURCES
    app/test.cpp ctrl/test_hsa.cpp pgen/test_pmgr.cpp
    simple_convolution/simple_convolution.cpp util/perf_timer.cpp)

add_executable(aqlprofile-ctrl-test)
target_sources(aqlprofile-ctrl-test PRIVATE ${AQLPROFILE_CTRL_TEST_SOURCES})
set_target_properties(
    aqlprofile-ctrl-test PROPERTIES OUTPUT_NAME ctrl RUNTIME_OUTPUT_DIRECTORY
                                                     ${CMAKE_CURRENT_BINARY_DIR})

target_include_directories(aqlprofile-ctrl-test PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})

target_compile_options(
    aqlprofile-ctrl-test PRIVATE -Wno-shadow -Wno-unused-parameter -Wno-unused-variable
                                 -Wno-format -Wno-unused-but-set-variable)

add_dependencies(aqlprofile-ctrl-test aqlprofile-ctrl-test-hsaco)

target_link_libraries(
    aqlprofile-ctrl-test
    PRIVATE -Wl,--start-group
            rocprofiler-sdk::rocprofiler-sdk-aqlprofile
            rocprofiler-sdk::rocprofiler-sdk-static-library
            rocprofiler-sdk::rocprofiler-sdk-common-library
            -Wl,--end-group
            rocprofiler-sdk::rocprofiler-sdk-hsa-runtime
            rocprofiler-sdk::rocprofiler-sdk-amd-comgr
            rocprofiler-sdk::rocprofiler-sdk-drm
            pthread
            dl)

# Deploy the manual run scripts next to the binary (for hand-driven runs).
foreach(_script run.sh run_priv.sh)
    configure_file(${CMAKE_CURRENT_SOURCE_DIR}/${_script}
                   ${CMAKE_CURRENT_BINARY_DIR}/${_script} COPYONLY)
endforeach()

set(_CTRL_LD_LIBRARY_PATH
    "${PROJECT_BINARY_DIR}/${CMAKE_INSTALL_LIBDIR}:${CMAKE_LIBRARY_OUTPUT_DIRECTORY}:${CMAKE_INSTALL_PREFIX}/lib:${ROCM_PATH}/lib:${ROCM_PATH}/llvm/lib:$ENV{LD_LIBRARY_PATH}"
    )

# Register one CTest entry per exercised mode (matches run.sh).
function(add_ctrl_mode_test NAME MODE_ENV)
    add_test(NAME aqlprofile-ctrl-${NAME} COMMAND $<TARGET_FILE:aqlprofile-ctrl-test>)
    set(_env "LD_LIBRARY_PATH=${_CTRL_LD_LIBRARY_PATH}")
    if(NOT MODE_ENV STREQUAL "")
        list(APPEND _env "${MODE_ENV}")
    endif()
    set_tests_properties(
        aqlprofile-ctrl-${NAME}
        PROPERTIES ENVIRONMENT
                   "${_env}"
                   WORKING_DIRECTORY
                   ${CMAKE_CURRENT_BINARY_DIR}
                   LABELS
                   "integration-tests;aqlprofile"
                   TIMEOUT
                   60
                   FAIL_REGULAR_EXPRESSION
                   "${ROCPROFILER_DEFAULT_FAIL_REGEX}")
endfunction()

add_ctrl_mode_test(dry "")
add_ctrl_mode_test(pmc "AQLPROFILE_PMC=1")
add_ctrl_mode_test(sqtt "AQLPROFILE_SQTT=1")
add_ctrl_mode_test(pcsmp "AQLPROFILE_PCSMP=1")
