cmake_minimum_required(VERSION 3.21 FATAL_ERROR)

#Add a name for the rocm-systems location
set(ROCM_SYSTEMS_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/../..")

# Set cmake_prefix_path for searching, ROCM_PATH if avail otherwise default to general rocm install path
set(CMAKE_PREFIX_PATH $ENV{ROCM_PATH} "/opt/rocm/")
message(STATUS "ROCM_PATH: $ENV{ROCM_PATH}")

if(
    CMAKE_SOURCE_DIR STREQUAL CMAKE_BINARY_DIR
    AND CMAKE_CURRENT_SOURCE_DIR STREQUAL CMAKE_SOURCE_DIR
)
    set(MSG "")
    message(STATUS "Warning! Building from the source directory is not recommended")
    message(STATUS "If unintended, please remove 'CMakeCache.txt' and 'CMakeFiles'")
    message(STATUS "and build from a separate directory")
    message(FATAL_ERROR "In-source build")
endif()

# System info
cmake_host_system_information(RESULT LOCALHOST QUERY FQDN)
message(STATUS "Hostname: ${LOCALHOST}")

# Versioning info derived from file
file(READ "${CMAKE_CURRENT_SOURCE_DIR}/VERSION" FULL_VERSION_STRING LIMIT_COUNT 1)
string(REGEX REPLACE "(\n|\r)" "" FULL_VERSION_STRING "${FULL_VERSION_STRING}")
set(ROCPROFCOMPUTE_FULL_VERSION "${FULL_VERSION_STRING}")
string(
    REGEX REPLACE
    "([0-9]+)\.([0-9]+)\.([0-9]+)(.*)"
    "\\1.\\2.\\3"
    ROCPROFCOMPUTE_VERSION
    "${FULL_VERSION_STRING}"
)

# string(REGEX REPLACE "(${ROCPROFCOMPUTE_VERSION})(.*)" "\\2"
# ROCPROFCOMPUTE_VERSION_TWEAK
# "${FULL_VERSION_STRING}")
# string(REGEX REPLACE "^\\." "" ROCPROFCOMPUTE_VERSION_TWEAK
# "${ROCPROFCOMPUTE_VERSION_TWEAK}")

project(
    rocprofiler-compute
    VERSION ${ROCPROFCOMPUTE_VERSION}
    LANGUAGES CXX
    DESCRIPTION
        "A kernel-level profiling tool for machine learning/HPC workloads running on AMD MI GPUs"
    HOMEPAGE_URL
        "https://github.com/ROCm/rocm-systems/tree/develop/projects/rocprofiler-compute"
)

set(PACKAGE_NAME "rocprofiler-compute")
set(PACKAGE_NAME_UNDERSCORE "rocprofiler_compute")
set(EXECUTABLE_NAME "rocprof-compute")

include(ExternalProject)
include(GNUInstallDirs)
include(utils.cmake)
# Must run before asan_build_dependencies.cmake, whose pthread guard reads the
# ENABLE_SANITIZER that this module resolves.
include(cmake/sanitizers.cmake)
include(${CMAKE_CURRENT_LIST_DIR}/cmake/asan_build_dependencies.cmake)

set(BUILD_ENABLE_LINTIAN_OVERRIDES ON CACHE BOOL "Enable/Disable Lintian Overrides")
set(BUILD_DEBIAN_PKGING_FLAG
    ON
    CACHE BOOL
    "Internal Status Flag to indicate Debian Packaging Build"
)
set(COMP_TYPE "main")

# version control info
execute_process(
    COMMAND git log --pretty=format:%h -n 1
    WORKING_DIRECTORY ${PROJECT_SOURCE_DIR}
    OUTPUT_VARIABLE ROCPROFCOMPUTE_GIT_REV
    OUTPUT_STRIP_TRAILING_WHITESPACE
)
message(STATUS "Git revision: ${ROCPROFCOMPUTE_GIT_REV}")
configure_file(
    ${PROJECT_SOURCE_DIR}/cmake/VERSION.sha.in
    ${PROJECT_SOURCE_DIR}/VERSION.sha
    @ONLY
)

set(CMAKE_BUILD_TYPE "Release")
if(CMAKE_INSTALL_PREFIX_INITIALIZED_TO_DEFAULT)
    set(CMAKE_INSTALL_PREFIX "/opt/rocm" CACHE PATH "default install path" FORCE)
endif()
message(STATUS "Installation path: ${CMAKE_INSTALL_PREFIX}")

# Profile mode requires only Python 3.8+ standard library (no external dependencies)
# Analyze mode dependencies are checked at runtime via verify_deps()

# Only find Python interpreter - no dependency checking
message(STATUS "Detecting Python interpreter...")
find_package(Python3 3.8 COMPONENTS Interpreter REQUIRED)
message(STATUS "Python ${Python3_VERSION} found")
message(
    STATUS
    "Note: Profile mode uses stdlib only. Analyze mode deps checked at runtime."
)

# ----------------------
# Portable Python command for tests
# ----------------------
# For build-time checks, use ${Python3_EXECUTABLE} (absolute path from find_package)
# For runtime tests (add_test), use portable command that resolves from PATH
# This allows tests to work when build/install folders are moved to different machines
set(PYTHON_TEST_COMMAND
    "python3"
    CACHE STRING
    "Python command for running tests (portable)"
)
enable_sanitizer_python_launcher(PYTHON_TEST_COMMAND)
message(STATUS "Python test command: ${PYTHON_TEST_COMMAND}")

# ----------------------
# modulefile creation
# ----------------------

set(MOD_INSTALL_PATH
    "${CMAKE_INSTALL_DATADIR}/${PROJECT_NAME}/modulefiles/${PROJECT_NAME}"
    CACHE STRING
    "Install path for modulefile"
)
message(STATUS "Modulefile install path: ${MOD_INSTALL_PATH}")

set(moduleFileTemplate "rocprofcompute.lua.in")

configure_file(
    ${PROJECT_SOURCE_DIR}/cmake/${moduleFileTemplate}
    ${PROJECT_BINARY_DIR}/${MOD_INSTALL_PATH}/${ROCPROFCOMPUTE_FULL_VERSION}.lua
    @ONLY
)

# Thera mods
if(LOCALHOST MATCHES "TheraS01|.*\.thera\.amd\.com|thera-hn")
    list(APPEND CMAKE_MESSAGE_INDENT "  ")
    message(STATUS "Using thera-specific modulefile modification")
    file(READ ${PROJECT_SOURCE_DIR}/cmake/modfile.thera.mod mod_additions)
    file(
        APPEND
        ${PROJECT_BINARY_DIR}/${MOD_INSTALL_PATH}/${ROCPROFCOMPUTE_FULL_VERSION}.lua
        ${mod_additions}
    )
    list(POP_BACK CMAKE_MESSAGE_INDENT)
endif()

# Setup testing collateral

option(INSTALL_TESTS "Build test suite" OFF)
option(ENABLE_TESTS "Enable compilation of testing collateral" OFF)
set(CMAKE_HIP_FLAGS_RELEASE "-O2")
if(${ENABLE_TESTS})
    enable_language("C" "HIP")
    add_subdirectory(tests)
endif()
message(STATUS "Enable tests compilation: ${ENABLE_TESTS}")

enable_testing()
include(CTest)

option(ENABLE_COVERAGE "Enable code coverage" OFF)
set(COV_OPTION "")
if(${ENABLE_COVERAGE})
    # Set coverage source path based on test mode
    if(TEST_FROM_INSTALL)
        set(COV_SRC ".")
    else()
        set(COV_SRC "src")
    endif()

    set(COV_OPTION
        "--cov=${COV_SRC}"
        "--cov-append"
        "--cov-report=term-missing"
        "--cov-report=lcov:tests/coverage.info"
    )

    set(CTEST_COVERAGE_COMMAND "python3")
endif()
message(STATUS "Code coverage: ${ENABLE_COVERAGE}")

# CPU threads available for testing
set(PYTEST_NUMPROCS
    "1"
    CACHE STRING
    "Number of parallel threads to use with CPU-oriented tests"
)
message(STATUS "Pytest CPU threadcount: ${PYTEST_NUMPROCS}")

# 2 CPU threads available for testing(test-analyze-commands)
set(PYTEST_NUMPROCS_ANALYSIS
    "4"
    CACHE STRING
    "Number of parallel threads to use with CPU-oriented tests"
)
message(STATUS "Pytest CPU threadcount: ${PYTEST_NUMPROCS_ANALYSIS}")

# ---------------------------
# add tests
# ---------------------------

find_package(MPI COMPONENTS CXX)
if(MPI_CXX_COMPILER)
    message(STATUS "MPI CXX compiler found: ${MPI_CXX_COMPILER}")
else()
    message(WARNING "MPI CXX compiler not found. MPI tests will be skipped.")
endif()

option(
    TEST_FROM_INSTALL
    "Test from install directory rather than from source directory"
    OFF
)

if(TEST_FROM_INSTALL)
    set(WORKING_DIR_OPTION "")
else()
    set(WORKING_DIR_OPTION WORKING_DIRECTORY ${PROJECT_SOURCE_DIR})
endif()

if(TEST_FROM_INSTALL)
    # When testing from install, write coverage to install directory
    set(COVERAGE_XML_PATH
        "${CMAKE_INSTALL_PREFIX}/${CMAKE_INSTALL_LIBEXECDIR}/${PACKAGE_NAME}/coverage.xml"
    )
    set(COVERAGE_DATA_PATH
        "${CMAKE_INSTALL_PREFIX}/${CMAKE_INSTALL_LIBEXECDIR}/${PACKAGE_NAME}/.coverage"
    )
else()
    # When testing from build, write coverage XML to build directory
    # and coverage data to source directory
    set(COVERAGE_XML_PATH "${CMAKE_BINARY_DIR}/coverage.xml")
    set(COVERAGE_DATA_PATH "${PROJECT_SOURCE_DIR}/.coverage")
endif()

# -------------------
# Setup tool library
# -------------------
add_subdirectory(src/lib)

# ---------------------------
# Profile mode import guard
# ---------------------------

add_test(
    NAME test_conftest
    COMMAND
        ${PYTHON_TEST_COMMAND} -m pytest --junitxml=tests/test_conftest.xml ${COV_OPTION}
        tests/unit/test_conftest.py ${WORKING_DIR_OPTION}
)

# ---------------------------
# profile mode tests
# ---------------------------

add_test(
    NAME test_profile_kernel_execution
    COMMAND
        ${PYTHON_TEST_COMMAND} -m pytest
        --junitxml=tests/test_profile_kernel_execution.xml ${COV_OPTION}
        tests/integration/test_profile_kernel_execution.py ${WORKING_DIR_OPTION}
)

add_test(
    NAME test_profile_dispatch
    COMMAND
        ${PYTHON_TEST_COMMAND} -m pytest --junitxml=tests/test_profile_dispatch.xml
        ${COV_OPTION} tests/integration/test_profile_dispatch.py ${WORKING_DIR_OPTION}
)

add_test(
    NAME test_profile_sort
    COMMAND
        ${PYTHON_TEST_COMMAND} -m pytest --junitxml=tests/test_profile_sort.xml
        ${COV_OPTION} tests/integration/test_profile_sort.py ${WORKING_DIR_OPTION}
)

add_test(
    NAME test_profile_misc
    COMMAND
        ${PYTHON_TEST_COMMAND} -m pytest -m misc --junitxml=tests/test_profile_misc.xml
        ${COV_OPTION} tests/integration/test_profile_misc.py ${WORKING_DIR_OPTION}
)

add_test(
    NAME test_profile_path
    COMMAND
        ${PYTHON_TEST_COMMAND} -m pytest --junitxml=tests/test_profile_path.xml
        ${COV_OPTION} tests/integration/test_profile_path.py ${WORKING_DIR_OPTION}
)

add_test(
    NAME test_profile_roofline_dir
    COMMAND
        ${PYTHON_TEST_COMMAND} -m pytest -m roofline_dir
        --junitxml=tests/test_profile_roofline_dir.xml ${COV_OPTION}
        tests/integration/test_profile_roofline.py ${WORKING_DIR_OPTION}
)

add_test(
    NAME test_profile_roofline_extra_options
    COMMAND
        ${PYTHON_TEST_COMMAND} -m pytest -m roofline_extra_options
        --junitxml=tests/test_profile_roofline_extra_options.xml ${COV_OPTION}
        tests/integration/test_profile_roofline.py ${WORKING_DIR_OPTION}
)

add_test(
    NAME test_profile_roofline_bench
    COMMAND
        ${PYTHON_TEST_COMMAND} -m pytest -m roofline_bench
        --junitxml=tests/test_profile_roofline_bench.xml ${COV_OPTION}
        tests/integration/test_profile_roofline.py ${WORKING_DIR_OPTION}
)

add_test(
    NAME test_profile_roofline_plot
    COMMAND
        ${PYTHON_TEST_COMMAND} -m pytest -m roofline_plot
        --junitxml=tests/test_profile_roofline_plot.xml ${COV_OPTION}
        tests/integration/test_profile_roofline.py ${WORKING_DIR_OPTION}
)

add_test(
    NAME test_profile_roofline_validation
    COMMAND
        ${PYTHON_TEST_COMMAND} -m pytest -m roofline_validation
        --junitxml=tests/test_profile_roofline_validation.xml ${COV_OPTION}
        tests/integration/test_profile_roofline.py ${WORKING_DIR_OPTION}
)

add_test(
    NAME test_profile_section
    COMMAND
        ${PYTHON_TEST_COMMAND} -m pytest --junitxml=tests/test_profile_section.xml
        ${COV_OPTION} tests/integration/test_profile_section.py ${WORKING_DIR_OPTION}
)

add_test(
    NAME test_profile_sets_func
    COMMAND
        ${PYTHON_TEST_COMMAND} -m pytest --junitxml=tests/test_profile_sets_func.xml
        ${COV_OPTION} tests/integration/test_profile_sets_func.py ${WORKING_DIR_OPTION}
)

add_test(
    NAME test_profile_live_attach_detach
    COMMAND
        ${PYTHON_TEST_COMMAND} -m pytest -s
        --junitxml=tests/test_profile_live_attach_detach.xml ${COV_OPTION}
        tests/integration/test_profile_live_attach_detach.py ${WORKING_DIR_OPTION}
)
set_tests_properties(
    test_profile_live_attach_detach
    PROPERTIES TIMEOUT 300 LABELS "profile" RESOURCE_GROUPS gpus:1
)

add_test(
    NAME test_profile_iteration_multiplexing_1
    COMMAND
        ${PYTHON_TEST_COMMAND} -m pytest -s -m iteration_multiplexing_1
        --junitxml=tests/test_profile_iteration_multiplexing_1.xml ${COV_OPTION}
        tests/integration/test_profile_iteration_multiplexing.py ${WORKING_DIR_OPTION}
)

add_test(
    NAME test_profile_iteration_multiplexing_2
    COMMAND
        ${PYTHON_TEST_COMMAND} -m pytest -s -m iteration_multiplexing_2
        --junitxml=tests/test_profile_iteration_multiplexing_2.xml ${COV_OPTION}
        tests/integration/test_profile_iteration_multiplexing.py ${WORKING_DIR_OPTION}
)

add_test(
    NAME test_profile_iteration_multiplexing_stochastic
    COMMAND
        ${PYTHON_TEST_COMMAND} -m pytest -s -m iteration_multiplexing_stochastic
        --junitxml=tests/test_profile_iteration_multiplexing_stochastic.xml ${COV_OPTION}
        tests/integration/test_profile_iteration_multiplexing.py ${WORKING_DIR_OPTION}
)

if(MPI_CXX_COMPILER)
    add_test(
        NAME test_profile_multi_rank
        COMMAND
            ${PYTHON_TEST_COMMAND} -m pytest -s
            --junitxml=tests/test_profile_multi_rank.xml ${COV_OPTION}
            tests/integration/test_profile_multi_rank.py ${WORKING_DIR_OPTION}
    )
    set_tests_properties(
        test_profile_multi_rank
        PROPERTIES LABELS "profile" RESOURCE_GROUPS gpus:2 TIMEOUT 1800
    )
endif()

add_test(
    NAME test_metric_validation
    COMMAND
        ${PYTHON_TEST_COMMAND} -m pytest --junitxml=tests/test_metric_validation.xml
        ${COV_OPTION} tests/integration/test_metric_validation.py ${WORKING_DIR_OPTION}
)

add_test(
    NAME test_native_tool_finder
    COMMAND
        ${PYTHON_TEST_COMMAND} -m pytest --junitxml=tests/test_native_tool_finder.xml
        ${COV_OPTION} tests/unit/utils/test_native_tool_finder.py ${WORKING_DIR_OPTION}
)

add_test(
    NAME test_profile_torch_trace
    COMMAND
        ${PYTHON_TEST_COMMAND} -m pytest -m torch_trace
        --junitxml=tests/test_profile_torch_trace.xml ${COV_OPTION}
        tests/integration/test_profile_torch_trace.py ${WORKING_DIR_OPTION}
)

add_test(
    NAME test_profile_triton_trace
    COMMAND
        ${PYTHON_TEST_COMMAND} -m pytest --junitxml=tests/test_profile_triton_trace.xml
        ${COV_OPTION} tests/integration/test_profile_triton_trace.py ${WORKING_DIR_OPTION}
)

set_tests_properties(
    test_profile_kernel_execution
    test_profile_dispatch
    test_profile_sort
    test_profile_misc
    test_profile_path
    test_profile_roofline_dir
    test_profile_roofline_extra_options
    test_profile_roofline_bench
    test_profile_roofline_plot
    test_profile_roofline_validation
    test_profile_section
    test_profile_sets_func
    test_profile_iteration_multiplexing_1
    test_profile_iteration_multiplexing_2
    test_profile_iteration_multiplexing_stochastic
    test_profile_torch_trace
    test_profile_triton_trace
    PROPERTIES LABELS "profile" RESOURCE_GROUPS gpus:1 TIMEOUT 1800
)

# ---------------------------
# analysis command tests
# ---------------------------

add_test(
    NAME test_analyze_commands
    COMMAND
        ${PYTHON_TEST_COMMAND} -m pytest -n ${PYTEST_NUMPROCS_ANALYSIS} --verbose
        --junitxml=tests/test_analyze_commands.xml ${COV_OPTION}
        tests/integration/test_analyze_commands.py ${WORKING_DIR_OPTION}
)

# ---------------------------
# analyze workloads tests
# ---------------------------

add_test(
    NAME test_analyze_workloads
    COMMAND
        ${PYTHON_TEST_COMMAND} -m pytest -n ${PYTEST_NUMPROCS}
        --junitxml=tests/test_analyze_workloads.xml ${COV_OPTION}
        tests/integration/test_analyze_workloads.py ${WORKING_DIR_OPTION}
)

# ---------------------------
# TCP counter tests
# ---------------------------

add_test(
    NAME test_L1_cache_counters
    COMMAND
        ${PYTHON_TEST_COMMAND} -m pytest --junitxml=tests/test_L1_cache_counters.xml
        ${COV_OPTION} tests/integration/test_l1_cache_counters.py ${WORKING_DIR_OPTION}
)

# ---------------------------
# Spec tests
# ---------------------------

add_test(
    NAME test_num_xcds_spec_class
    COMMAND
        ${PYTHON_TEST_COMMAND} -m pytest -m num_xcds_spec_class
        --junitxml=tests/test_num_xcds_spec_class.xml ${COV_OPTION}
        tests/integration/test_num_xcds.py ${WORKING_DIR_OPTION}
)

add_test(
    NAME test_num_xcds_cli_output
    COMMAND
        ${PYTHON_TEST_COMMAND} -m pytest -m num_xcds_cli_output
        --junitxml=tests/test_num_xcds_cli_output.xml ${COV_OPTION}
        tests/integration/test_num_xcds.py ${WORKING_DIR_OPTION}
)

# Machine spec parsing and cache sizing — no GPU required
add_test(
    NAME test_specs
    COMMAND
        ${PYTHON_TEST_COMMAND} -m pytest --junitxml=tests/test_specs.xml ${COV_OPTION}
        tests/unit/utils/test_specs.py ${WORKING_DIR_OPTION}
)

add_test(
    NAME test_mi_gpu_spec
    COMMAND
        ${PYTHON_TEST_COMMAND} -m pytest --junitxml=tests/test_mi_gpu_spec.xml
        ${COV_OPTION} tests/unit/utils/test_mi_gpu_spec.py ${WORKING_DIR_OPTION}
)

# ---------------------------
# Utils tests
# ---------------------------

add_test(
    NAME test_utils_common
    COMMAND
        ${PYTHON_TEST_COMMAND} -m pytest --junitxml=tests/test_utils_common.xml
        ${COV_OPTION} tests/unit/utils/test_utils_common.py ${WORKING_DIR_OPTION}
)

add_test(
    NAME test_utils_analysis
    COMMAND
        ${PYTHON_TEST_COMMAND} -m pytest --junitxml=tests/test_utils_analysis.xml
        ${COV_OPTION} tests/unit/utils/test_utils_analysis.py ${WORKING_DIR_OPTION}
)

add_test(
    NAME test_utils_profile
    COMMAND
        ${PYTHON_TEST_COMMAND} -m pytest --junitxml=tests/test_utils_profile.xml
        ${COV_OPTION} tests/unit/utils/test_utils_profile.py ${WORKING_DIR_OPTION}
)

add_test(
    NAME test_common
    COMMAND
        ${PYTHON_TEST_COMMAND} -m pytest --junitxml=tests/test_common.xml ${COV_OPTION}
        tests/unit/test_common.py ${WORKING_DIR_OPTION}
)

add_test(
    NAME test_amdsmi_interface
    COMMAND
        ${PYTHON_TEST_COMMAND} -m pytest --junitxml=tests/test_amdsmi_interface.xml
        ${COV_OPTION} tests/unit/utils/test_amdsmi_interface.py ${WORKING_DIR_OPTION}
)

add_test(
    NAME test_noise_clamper
    COMMAND
        ${PYTHON_TEST_COMMAND} -m pytest --junitxml=tests/test_noise_clamper.xml
        ${COV_OPTION} tests/unit/utils/metrics/test_noise_clamper.py ${WORKING_DIR_OPTION}
)

add_test(
    NAME test_benchmark_base
    COMMAND
        ${PYTHON_TEST_COMMAND} -m pytest --junitxml=tests/test_benchmark_base.xml
        ${COV_OPTION} tests/unit/roofline/benchmark/test_benchmark_base.py
        ${WORKING_DIR_OPTION}
)

add_test(
    NAME test_analysis_cli
    COMMAND
        ${PYTHON_TEST_COMMAND} -m pytest --junitxml=tests/test_analysis_cli.xml
        ${COV_OPTION} tests/unit/rocprof_compute_analyze/test_analysis_cli.py
        ${WORKING_DIR_OPTION}
)

add_test(
    NAME test_pattern_matching
    COMMAND
        ${PYTHON_TEST_COMMAND} -m pytest --junitxml=tests/test_pattern_matching.xml
        ${COV_OPTION} tests/unit/utils/test_pattern_matching.py ${WORKING_DIR_OPTION}
)

add_test(
    NAME test_list_metrics_cli
    COMMAND
        ${PYTHON_TEST_COMMAND} -m pytest --junitxml=tests/test_list_metrics_cli.xml
        ${COV_OPTION} tests/integration/test_list_metrics_cli.py ${WORKING_DIR_OPTION}
)

# rocprofv3 avail library interface — no GPU required
add_test(
    NAME test_rocprofv3_avail_interface
    COMMAND
        ${PYTHON_TEST_COMMAND} -m pytest
        --junitxml=tests/test_rocprofv3_avail_interface.xml ${COV_OPTION}
        tests/unit/utils/test_rocprofv3_avail_interface.py ${WORKING_DIR_OPTION}
)

# Memory chart tests — no GPU required
add_test(
    NAME test_mem_chart_gfx11
    COMMAND
        ${PYTHON_TEST_COMMAND} -m pytest --junitxml=tests/test_mem_chart_gfx11.xml
        ${COV_OPTION} tests/unit/utils/test_mem_chart_gfx11.py ${WORKING_DIR_OPTION}
)

add_test(
    NAME test_mem_chart_gfx9
    COMMAND
        ${PYTHON_TEST_COMMAND} -m pytest --junitxml=tests/test_mem_chart_gfx9.xml
        ${COV_OPTION} tests/unit/utils/test_mem_chart_gfx9.py ${WORKING_DIR_OPTION}
)

add_test(
    NAME test_roofline_calc
    COMMAND
        ${PYTHON_TEST_COMMAND} -m pytest --junitxml=tests/test_roofline_calc.xml
        ${COV_OPTION} tests/unit/utils/test_roofline_calc.py ${WORKING_DIR_OPTION}
)

add_test(
    NAME test_roofline_analyze
    COMMAND
        ${PYTHON_TEST_COMMAND} -m pytest --junitxml=tests/test_roofline_analyze.xml
        ${COV_OPTION} tests/integration/test_roofline_analyze.py ${WORKING_DIR_OPTION}
)

add_test(
    NAME test_roofline_main
    COMMAND
        ${PYTHON_TEST_COMMAND} -m pytest --junitxml=tests/test_roofline_main.xml
        ${COV_OPTION} tests/unit/roofline/test_roofline_main.py ${WORKING_DIR_OPTION}
)

# -----------------------------------
# Data imputation tests
# -----------------------------------

# -----------------------------------
# Analysis database tests
# -----------------------------------

add_test(
    NAME test_analysis_db
    COMMAND
        ${PYTHON_TEST_COMMAND} -m pytest --junitxml=tests/test_analysis_db.xml
        ${COV_OPTION} tests/unit/rocprof_compute_analyze/test_analysis_db.py
        ${WORKING_DIR_OPTION}
)

add_test(
    NAME test_analysis_base
    COMMAND
        ${PYTHON_TEST_COMMAND} -m pytest --junitxml=tests/test_analysis_base.xml
        ${COV_OPTION} tests/unit/rocprof_compute_analyze/test_analysis_base.py
        ${WORKING_DIR_OPTION}
)

add_test(
    NAME test_analysis_orm
    COMMAND
        ${PYTHON_TEST_COMMAND} -m pytest --junitxml=tests/test_analysis_orm.xml
        ${COV_OPTION} tests/unit/utils/test_analysis_orm.py ${WORKING_DIR_OPTION}
)

add_test(
    NAME test_code_object_analysis
    COMMAND
        ${PYTHON_TEST_COMMAND} -m pytest --junitxml=tests/test_code_object_analysis.xml
        ${COV_OPTION} tests/unit/pc_sampling/test_code_object_analysis.py
        ${WORKING_DIR_OPTION}
)

add_test(
    NAME test_source_snapshot_analysis
    COMMAND
        ${PYTHON_TEST_COMMAND} -m pytest
        --junitxml=tests/test_source_snapshot_analysis.xml ${COV_OPTION}
        tests/unit/pc_sampling/test_source_snapshot_analysis.py ${WORKING_DIR_OPTION}
)

add_test(
    NAME test_per_kernel_isa_export
    COMMAND
        ${PYTHON_TEST_COMMAND} -m pytest --junitxml=tests/test_per_kernel_isa_export.xml
        ${COV_OPTION} tests/unit/pc_sampling/test_per_kernel_isa_export.py
        ${WORKING_DIR_OPTION}
)

# -----------------------------------
# Metrics unit tests
# -----------------------------------

add_test(
    NAME test_aggregation
    COMMAND
        ${PYTHON_TEST_COMMAND} -m pytest --junitxml=tests/test_aggregation.xml
        ${COV_OPTION} tests/unit/utils/metrics/test_aggregation.py ${WORKING_DIR_OPTION}
)

add_test(
    NAME test_expression
    COMMAND
        ${PYTHON_TEST_COMMAND} -m pytest --junitxml=tests/test_expression.xml
        ${COV_OPTION} tests/unit/utils/metrics/test_expression.py ${WORKING_DIR_OPTION}
)

add_test(
    NAME test_evaluation_pipeline
    COMMAND
        ${PYTHON_TEST_COMMAND} -m pytest --junitxml=tests/test_evaluation_pipeline.xml
        ${COV_OPTION} tests/unit/utils/metrics/test_evaluation_pipeline.py
        ${WORKING_DIR_OPTION}
)

add_test(
    NAME test_metric_evaluator
    COMMAND
        ${PYTHON_TEST_COMMAND} -m pytest --junitxml=tests/test_metric_evaluator.xml
        ${COV_OPTION} tests/unit/utils/metrics/test_metric_evaluator.py
        ${WORKING_DIR_OPTION}
)

add_test(
    NAME test_utils_counter_defs
    COMMAND
        ${PYTHON_TEST_COMMAND} -m pytest --junitxml=tests/test_utils_counter_defs.xml
        ${COV_OPTION} tests/unit/utils/test_utils_counter_defs.py ${WORKING_DIR_OPTION}
)

add_test(
    NAME test_parser
    COMMAND
        ${PYTHON_TEST_COMMAND} -m pytest --junitxml=tests/test_parser.xml ${COV_OPTION}
        tests/unit/utils/test_parser.py ${WORKING_DIR_OPTION}
)

add_test(
    NAME test_metrics_common
    COMMAND
        ${PYTHON_TEST_COMMAND} -m pytest --junitxml=tests/test_metrics_common.xml
        ${COV_OPTION} tests/unit/utils/metrics/test_metrics_common.py
        ${WORKING_DIR_OPTION}
)

# -----------------------------------
# CLI options unit tests
# -----------------------------------

add_test(
    NAME test_argparser
    COMMAND
        ${PYTHON_TEST_COMMAND} -m pytest --junitxml=tests/test_argparser.xml ${COV_OPTION}
        tests/unit/test_argparser.py ${WORKING_DIR_OPTION}
)

# ---------------------------
# PC sampling tests
# ---------------------------

add_test(
    NAME test_pc_sampling
    COMMAND
        ${PYTHON_TEST_COMMAND} -m pytest -s --junitxml=tests/test_pc_sampling.xml
        ${COV_OPTION} tests/integration/test_pc_sampling.py ${WORKING_DIR_OPTION}
)

add_test(
    NAME test_pc_sampling_analysis
    COMMAND
        ${PYTHON_TEST_COMMAND} -m pytest -s --junitxml=tests/test_pc_sampling_analysis.xml
        ${COV_OPTION} tests/unit/pc_sampling/test_pc_sampling_analysis.py
        ${WORKING_DIR_OPTION}
)

add_test(
    NAME test_pc_sampling_analyze_cli
    COMMAND
        ${PYTHON_TEST_COMMAND} -m pytest --junitxml=tests/test_pc_sampling_analyze_cli.xml
        ${COV_OPTION} tests/integration/test_pc_sampling_analyze_cli.py
        ${WORKING_DIR_OPTION}
)

# -----------------------------------
# TUI component tests
# -----------------------------------

add_test(
    NAME test_instant_button
    COMMAND
        ${PYTHON_TEST_COMMAND} -m pytest -m tui --junitxml=tests/test_instant_button.xml
        ${COV_OPTION} tests/unit/rocprof_compute_tui/widgets/test_instant_button.py
        ${WORKING_DIR_OPTION}
)

add_test(
    NAME test_menu_bar
    COMMAND
        ${PYTHON_TEST_COMMAND} -m pytest -m tui --junitxml=tests/test_menu_bar.xml
        ${COV_OPTION} tests/unit/rocprof_compute_tui/widgets/menu_bar/test_menu_bar.py
        ${WORKING_DIR_OPTION}
)

add_test(
    NAME test_tui_utils
    COMMAND
        ${PYTHON_TEST_COMMAND} -m pytest -m tui --junitxml=tests/test_tui_utils.xml
        ${COV_OPTION} tests/unit/rocprof_compute_tui/utils/test_tui_utils.py
        ${WORKING_DIR_OPTION}
)

add_test(
    NAME test_collapsibles
    COMMAND
        ${PYTHON_TEST_COMMAND} -m pytest -m tui --junitxml=tests/test_collapsibles.xml
        ${COV_OPTION} tests/unit/rocprof_compute_tui/widgets/test_collapsibles.py
        ${WORKING_DIR_OPTION}
)

# -----------------------------------
# Torch trace tests
# -----------------------------------

add_test(
    NAME test_rocpd_data
    COMMAND
        ${PYTHON_TEST_COMMAND} -m pytest --junitxml=tests/test_rocpd_data.xml
        ${COV_OPTION} tests/unit/utils/test_rocpd_data.py ${WORKING_DIR_OPTION}
)

add_test(
    NAME test_torch_trace_coverage
    COMMAND
        ${PYTHON_TEST_COMMAND} -m pytest --junitxml=tests/test_torch_trace_coverage.xml
        ${COV_OPTION} tests/integration/test_torch_trace_coverage.py ${WORKING_DIR_OPTION}
)

set_tests_properties(
    test_torch_trace_coverage
    PROPERTIES TIMEOUT 1800 LABELS "profile" RESOURCE_GROUPS gpus:1
)

add_test(
    NAME test_torch_cpp_loader
    COMMAND
        ${PYTHON_TEST_COMMAND} -m pytest --junitxml=tests/test_torch_cpp_loader.xml
        ${COV_OPTION} tests/unit/utils/inject_roctx/_backends/test_torch_cpp_loader.py
        ${WORKING_DIR_OPTION}
)

add_test(
    NAME test_inject_roctx_package
    COMMAND
        ${PYTHON_TEST_COMMAND} -m pytest --junitxml=tests/test_inject_roctx_package.xml
        ${COV_OPTION} tests/unit/utils/inject_roctx/test_inject_roctx_package.py
        ${WORKING_DIR_OPTION}
)

# ---------------------------------------
# Pure csv compatibility layer for pandas
# ---------------------------------------

add_test(
    NAME test_file_io
    COMMAND
        ${PYTHON_TEST_COMMAND} -m pytest --junitxml=tests/test_file_io.xml ${COV_OPTION}
        tests/unit/utils/test_file_io.py ${WORKING_DIR_OPTION}
)

add_test(
    NAME test_utils_profile_csv
    COMMAND
        ${PYTHON_TEST_COMMAND} -m pytest --junitxml=tests/test_utils_profile_csv.xml
        ${COV_OPTION} tests/unit/utils/test_utils_profile_csv.py ${WORKING_DIR_OPTION}
)

# ---------------------------------------
# Compression boundary for csv artifacts
# ---------------------------------------

add_test(
    NAME test_csv_compression
    COMMAND
        ${PYTHON_TEST_COMMAND} -m pytest --junitxml=tests/test_csv_compression.xml
        ${COV_OPTION} tests/unit/utils/test_csv_compression.py ${WORKING_DIR_OPTION}
)

# -----------------------------------
# Profiler base unit tests
# -----------------------------------

add_test(
    NAME test_profiler_base
    COMMAND
        ${PYTHON_TEST_COMMAND} -m pytest --junitxml=tests/test_profiler_base.xml
        ${COV_OPTION} tests/unit/rocprof_compute_profile/test_profiler_base.py
        ${WORKING_DIR_OPTION}
)

# -----------------------------------
# PC sampling profiler unit tests
# -----------------------------------

add_test(
    NAME test_pc_sampling_profile
    COMMAND
        ${PYTHON_TEST_COMMAND} -m pytest --junitxml=tests/test_pc_sampling_profile.xml
        ${COV_OPTION} tests/unit/pc_sampling/test_pc_sampling_profile.py
        ${WORKING_DIR_OPTION}
)

# -----------------------------------
# SoC base counter allocation tests
# -----------------------------------

add_test(
    NAME test_soc_base
    COMMAND
        ${PYTHON_TEST_COMMAND} -m pytest --junitxml=tests/test_soc_base.xml ${COV_OPTION}
        tests/unit/rocprof_compute_soc/test_soc_base.py ${WORKING_DIR_OPTION}
)

# -----------------------------------
# TTY table rendering unit tests
# -----------------------------------

add_test(
    NAME test_tty
    COMMAND
        ${PYTHON_TEST_COMMAND} -m pytest --junitxml=tests/test_tty.xml ${COV_OPTION}
        tests/unit/utils/test_tty.py ${WORKING_DIR_OPTION}
)

if(TARGET test-rocprofiler-compute-tool)
    if(TEST_FROM_INSTALL)
        set(_tool_test_cmd tests/test-rocprofiler-compute-tool)
    else()
        set(_tool_test_cmd $<TARGET_FILE:test-rocprofiler-compute-tool>)
    endif()
    add_test(NAME test-rocprofiler-compute-tool COMMAND ${_tool_test_cmd})
    sanitizer_runtime_env(_san_env OFF)
    if(_san_env)
        set_tests_properties(
            test-rocprofiler-compute-tool
            PROPERTIES ENVIRONMENT "${_san_env}"
        )
    endif()
endif()

if(TARGET test-pc-sampling-collector)
    if(TEST_FROM_INSTALL)
        set(_pcs_test_cmd tests/test-pc-sampling-collector)
    else()
        set(_pcs_test_cmd $<TARGET_FILE:test-pc-sampling-collector>)
    endif()
    add_test(NAME test-pc-sampling-collector COMMAND ${_pcs_test_cmd})
    sanitizer_runtime_env(_san_env OFF)
    if(_san_env)
        set_tests_properties(
            test-pc-sampling-collector
            PROPERTIES ENVIRONMENT "${_san_env}"
        )
    endif()
endif()

# Registered here rather than in src/lib/torch_trace_collector/tests so the YAML
# category labels, applied in this directory scope, can reach it. The target
# only exists when torch and the Python development files were found.
if(TARGET test-torch-trace-collector)
    if(TEST_FROM_INSTALL)
        set(_ttc_launcher tests/torch_trace_collector_launcher.py)
        set(_ttc_test_cmd tests/test-torch-trace-collector)
    else()
        set(_ttc_launcher ${PROJECT_SOURCE_DIR}/tests/torch_trace_collector_launcher.py)
        set(_ttc_test_cmd $<TARGET_FILE:test-torch-trace-collector>)
    endif()
    add_test(
        NAME test-torch-trace-collector
        COMMAND ${PYTHON_TEST_COMMAND} ${_ttc_launcher} ${_ttc_test_cmd}
    )
    set_tests_properties(test-torch-trace-collector PROPERTIES SKIP_RETURN_CODE 125)
endif()

# -----------------------------------
# CSV compression Tests
# -----------------------------------

if(TARGET test-compression)
    if(TEST_FROM_INSTALL)
        set(_compression_test_cmd tests/test-compression)
    else()
        set(_compression_test_cmd $<TARGET_FILE:test-compression>)
    endif()
    add_test(NAME test-compression COMMAND ${_compression_test_cmd})
    sanitizer_runtime_env(_san_env OFF)
    if(_san_env)
        set_tests_properties(test-compression PROPERTIES ENVIRONMENT "${_san_env}")
    endif()
endif()

# -----------------------------------
# Generate coverage tests
# -----------------------------------

if(${ENABLE_COVERAGE})
    set(ALL_TEST_NAMES
        test_profile_kernel_execution
        test_profile_dispatch
        test_profile_sort
        test_profile_misc
        test_profile_path
        test_profile_roofline_dir
        test_profile_roofline_extra_options
        test_profile_roofline_bench
        test_profile_roofline_plot
        test_profile_roofline_validation
        test_profile_section
        test_profile_sets_func
        test_pc_sampling
        test_analyze_commands
        test_analyze_workloads
        test_aggregation
        test_expression
        test_evaluation_pipeline
        test_metric_evaluator
        test_utils_counter_defs
        test_parser
        test_argparser
        test_L1_cache_counters
        test_num_xcds_spec_class
        test_num_xcds_cli_output
        test_specs
        test_mi_gpu_spec
        test_utils_common
        test_utils_analysis
        test_utils_profile
        test_common
        test_amdsmi_interface
        test_noise_clamper
        test_benchmark_base
        test_analysis_cli
        test_pattern_matching
        test_list_metrics_cli
        test_mem_chart_gfx9
        test_mem_chart_gfx11
        test_roofline_calc
        test_roofline_analyze
        test_roofline_main
        test_instant_button
        test_menu_bar
        test_tui_utils
        test_collapsibles
        test_rocpd_data
        test_torch_trace_coverage
        test_profile_torch_trace
        test_profile_triton_trace
        test_torch_cpp_loader
        test_inject_roctx_package
        test_analysis_db
        test_analysis_base
        test_analysis_orm
        test_code_object_analysis
        test_source_snapshot_analysis
        test_pc_sampling_analysis
        test_pc_sampling_analyze_cli
        test_pc_sampling_profile
        test_metrics_common
        test_metric_validation
        test_native_tool_finder
        test_profiler_base
        test_rocprofv3_avail_interface
        test_soc_base
        test_tty
        test_file_io
        test_utils_profile_csv
        test_csv_compression
        test_conftest
        test_profile_iteration_multiplexing_1
        test_profile_iteration_multiplexing_2
        test_profile_iteration_multiplexing_stochastic
        test_profile_live_attach_detach
        test_profile_multi_rank
    )

    string(REPLACE ";" ";" ALL_TESTS_STRING "${ALL_TEST_NAMES}")

    add_test(
        NAME generate_coverage_report
        COMMAND
            ${PYTHON_TEST_COMMAND} -m coverage xml -o ${COVERAGE_XML_PATH}
            ${WORKING_DIR_OPTION}
    )

    set_tests_properties(
        generate_coverage_report
        PROPERTIES
            DEPENDS "${ALL_TESTS_STRING}"
            LABELS "coverage"
            ENVIRONMENT "COVERAGE_FILE=${COVERAGE_DATA_PATH}"
            TIMEOUT 600
    )
endif()

# -------------------
# Test categorization
# -------------------
set(_test_categories_yaml "${CMAKE_CURRENT_SOURCE_DIR}/tests/test_categories.yaml")
include(${ROCM_SYSTEMS_ROOT}/shared/ctest/TestCategories.cmake OPTIONAL)

if(COMMAND apply_ctest_category_labels)
    if(EXISTS "${_test_categories_yaml}")
        message(STATUS "YAML-based test categorization")
        apply_ctest_category_labels("${_test_categories_yaml}")
    else()
        message(STATUS "No test categorization (YAML missing)")
    endif()
else()
    message(STATUS "Test categorization not available (TestCategories.cmake not found)")
endif()

# ---------
# Install
# ---------

# top-level rocprofiler-compute utility
install(
    PROGRAMS src/${EXECUTABLE_NAME}
    DESTINATION ${CMAKE_INSTALL_LIBEXECDIR}/${PROJECT_NAME}
    COMPONENT main
)
# python dependency requirements
install(
    FILES requirements.txt
    DESTINATION ${CMAKE_INSTALL_LIBEXECDIR}/${PROJECT_NAME}
    COMPONENT main
)
# support files and version info
install(
    FILES src/argparser.py src/config.py src/rocprof_compute_base.py VERSION VERSION.sha
    DESTINATION ${CMAKE_INSTALL_LIBEXECDIR}/${PROJECT_NAME}
    COMPONENT main
)
# src/rocprof_compute_analyze
install(
    DIRECTORY src/rocprof_compute_analyze
    DESTINATION ${CMAKE_INSTALL_LIBEXECDIR}/${PROJECT_NAME}
    COMPONENT main
    PATTERN src/rocprof_compute_analyze/tests EXCLUDE
    PATTERN "__pycache__" EXCLUDE
)
# src/utils
install(
    DIRECTORY src/utils
    DESTINATION ${CMAKE_INSTALL_LIBEXECDIR}/${PROJECT_NAME}
    COMPONENT main
    PATTERN "rooflines*" EXCLUDE
    PATTERN "__pycache__" EXCLUDE
)
# src/roofline
install(
    DIRECTORY src/roofline
    DESTINATION ${CMAKE_INSTALL_LIBEXECDIR}/${PROJECT_NAME}
    COMPONENT main
    PATTERN "__pycache__" EXCLUDE
)
# src/rocprof_compute_soc
install(
    DIRECTORY src/rocprof_compute_soc
    DESTINATION ${CMAKE_INSTALL_LIBEXECDIR}/${PROJECT_NAME}
    COMPONENT main
    PATTERN "__pycache__" EXCLUDE
)
# src/rocprof_compute_profile
install(
    DIRECTORY src/rocprof_compute_profile
    DESTINATION ${CMAKE_INSTALL_LIBEXECDIR}/${PROJECT_NAME}
    COMPONENT main
    PATTERN "__pycache__" EXCLUDE
)
# src/pc_sampling
install(
    DIRECTORY src/pc_sampling
    DESTINATION ${CMAKE_INSTALL_LIBEXECDIR}/${PROJECT_NAME}
    COMPONENT main
    PATTERN "__pycache__" EXCLUDE
)
# src/rocprof_compute_tui
install(
    DIRECTORY src/rocprof_compute_tui
    DESTINATION ${CMAKE_INSTALL_LIBEXECDIR}/${PROJECT_NAME}
    COMPONENT main
    PATTERN "__pycache__" EXCLUDE
)

# ==============================================================================
# Vendored Dependencies Installation
# ==============================================================================
# Vendored packages are included as git submodules and installed as pure Python.
# Each package has: conditional submodule init + unconditional install.
# See src/vendored/README.md and CONTRIBUTING.md for vendoring guidelines.

find_package(Git QUIET)

# Vendored package wrapper
install(
    FILES src/vendored/__init__.py
    DESTINATION ${CMAKE_INSTALL_LIBEXECDIR}/${PROJECT_NAME}/vendored
    COMPONENT main
)

# PyYAML
if(NOT EXISTS "${PROJECT_SOURCE_DIR}/src/vendored/pyyaml/lib/yaml")
    if(Git_FOUND)
        message(STATUS "Initializing vendored/pyyaml submodule...")
        execute_process(
            COMMAND
                ${GIT_EXECUTABLE} submodule update --init --recursive src/vendored/pyyaml
            WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
            RESULT_VARIABLE GIT_SUBMOD_RESULT
        )
        if(NOT GIT_SUBMOD_RESULT EQUAL "0")
            message(FATAL_ERROR "git submodule update failed for src/vendored/pyyaml")
        endif()
    else()
        message(FATAL_ERROR "vendored/pyyaml not found and git not available")
    endif()
else()
    message(STATUS "Vendored/pyyaml: OK")
endif()
install(
    DIRECTORY src/vendored/pyyaml/lib/yaml
    DESTINATION ${CMAKE_INSTALL_LIBEXECDIR}/${PROJECT_NAME}/vendored/pyyaml/lib
    COMPONENT main
    FILES_MATCHING
    PATTERN "*.py"
    PATTERN "__pycache__" EXCLUDE
    PATTERN "*.pyc" EXCLUDE
)

# samples
install(
    DIRECTORY sample
    DESTINATION ${CMAKE_INSTALL_DATADIR}/${PROJECT_NAME}
    COMPONENT main
    FILES_MATCHING
    PATTERN "*.hip"
    PATTERN "*.h"
    PATTERN "*.hpp"
    PATTERN "*.cpp"
    PATTERN "*.py"
    PATTERN "Makefile"
    PATTERN "workloads" EXCLUDE
)
# modulefile
install(
    FILES ${PROJECT_BINARY_DIR}/${MOD_INSTALL_PATH}/${ROCPROFCOMPUTE_FULL_VERSION}.lua
    DESTINATION ${MOD_INSTALL_PATH}
    COMPONENT main
)

# install librocprofiler-compute-tool.so and
# torch_trace_collector-<torch-version>.so
install(
    TARGETS rocprofiler-compute-tool
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}/rocprofiler-compute COMPONENT main
)
# The collector is skipped when PyTorch is absent or an unsupported version.
# The loader searches lib/rocprofiler-compute/ for the versioned artifact.
if(TARGET torch-trace-collector-module)
    install(
        TARGETS torch-trace-collector-module
        LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}/rocprofiler-compute COMPONENT main
    )
endif()

# top-level bin/rocprof-compute launcher (re-execs into libexec entrypoint)
configure_file(
    ${PROJECT_SOURCE_DIR}/cmake/${EXECUTABLE_NAME}.in
    ${PROJECT_BINARY_DIR}/${EXECUTABLE_NAME}
    @ONLY
)
install(
    PROGRAMS ${PROJECT_BINARY_DIR}/${EXECUTABLE_NAME}
    DESTINATION ${CMAKE_INSTALL_BINDIR}
    COMPONENT main
)

install(
    FILES ${PROJECT_SOURCE_DIR}/LICENSE.md
    DESTINATION ${CMAKE_INSTALL_DOCDIR}
    COMPONENT main
)

# TEST collateral
if(INSTALL_TESTS)
    install(
        DIRECTORY tests
        DESTINATION ${CMAKE_INSTALL_LIBEXECDIR}/${PROJECT_NAME}
        COMPONENT tests
        FILES_MATCHING
        PATTERN "*.py"
        PATTERN "__pycache__" EXCLUDE
    )
    # The conjugate_gradient driver resolves its kernel libraries relative to
    # its own path, so the whole directory installs as a unit.
    install(
        DIRECTORY tests/conjugate_gradient
        DESTINATION ${CMAKE_INSTALL_LIBEXECDIR}/${PROJECT_NAME}/tests
        COMPONENT tests
        USE_SOURCE_PERMISSIONS
    )
    install(
        PROGRAMS therock_test_runner.py
        DESTINATION ${CMAKE_INSTALL_LIBEXECDIR}/${PROJECT_NAME}
        COMPONENT tests
    )
    install(
        DIRECTORY tests/workloads
        DESTINATION ${CMAKE_INSTALL_LIBEXECDIR}/${PROJECT_NAME}/tests
        COMPONENT tests
        PATTERN "__pycache__" EXCLUDE
    )
    install(
        PROGRAMS
            tests/vcopy
            tests/vmem
            tests/vrandom_access
            tests/vsequential_access
            tests/occupancy
            tests/hip_dynamic_shared
            tests/laplace_eqn
            tests/mat_mul_max
            tests/rocflop
            tests/memcopy
        DESTINATION ${CMAKE_INSTALL_LIBEXECDIR}/${PROJECT_NAME}/tests
        COMPONENT tests
    )
    if(MPI_CXX_COMPILER)
        install(
            PROGRAMS tests/mpi_aware_laplace_eqn
            DESTINATION ${CMAKE_INSTALL_LIBEXECDIR}/${PROJECT_NAME}/tests
            COMPONENT tests
        )
    endif()
    install(
        FILES requirements-test.txt pyproject.toml
        DESTINATION ${CMAKE_INSTALL_LIBEXECDIR}/${PROJECT_NAME}
        COMPONENT tests
    )
    install(
        FILES ${CMAKE_CURRENT_BINARY_DIR}/CTestTestfile.cmake
        COMPONENT tests
        DESTINATION ${CMAKE_INSTALL_LIBEXECDIR}/${PROJECT_NAME}
    )
endif()
message(STATUS "Install tests: ${INSTALL_TESTS}")

# ----------
# Packaging
# ----------

set(PKG_MAINTAINER_NM "rocprofiler compute support")
set(PKG_MAINTAINER_EMAIL "dl.omnitools@amd.com")
message(STATUS "Packaging config...")
set(CPACK_GENERATOR "DEB" "RPM" CACHE STRING "")
set(CPACK_PACKAGE_NAME "${PROJECT_NAME}" CACHE STRING "")
set(CPACK_PACKAGE_CONTACT
    "https://github.com/ROCm/rocm-systems/tree/develop/projects/rocprofiler-compute"
)
set(CPACK_PACKAGE_DESCRIPTION_SUMMARY
    "ROCm Compute Profiler: tool for GPU performance profiling"
)
set(CPACK_RPM_PACKAGE_DESCRIPTION
    "ROCm Compute Profiler is a performance analysis tool for profiling
machine learning/HPC workloads running on AMD GPUs."
)
set(CPACK_PACKAGE_VENDOR "Advanced Micro Devices, Inc.")

# Package versioning
set(CPACK_PACKAGE_VERSION_MAJOR ${PROJECT_VERSION_MAJOR})
set(CPACK_PACKAGE_VERSION_MINOR ${PROJECT_VERSION_MINOR})
set(CPACK_PACKAGE_VERSION_PATCH ${PROJECT_VERSION_PATCH})
set(CPACK_PACKAGE_VERSION
    "${CPACK_PACKAGE_VERSION_MAJOR}.${CPACK_PACKAGE_VERSION_MINOR}.${CPACK_PACKAGE_VERSION_PATCH}"
)

# RPM package specific variables
set(CPACK_RPM_PACKAGE_LICENSE "MIT")
set(CPACK_RPM_COMPONENT_INSTALL ON)
set(CPACK_RPM_PACKAGE_RELEASE_DIST ON)
set(CPACK_RPM_FILE_NAME "RPM-DEFAULT")
set(CPACK_RPM_SPEC_MORE_DEFINE "%undefine __brp_mangle_shebangs")

if(DEFINED CPACK_PACKAGING_INSTALL_PREFIX)
    set(CPACK_RPM_EXCLUDE_FROM_AUTO_FILELIST_ADDITION "${CPACK_PACKAGING_INSTALL_PREFIX}")
endif()

# Debian package specific variables
set(CPACK_DEBIAN_PACKAGE_LICENSE "MIT")
set(CPACK_DEB_COMPONENT_INSTALL ON)
set(CPACK_DEBIAN_FILE_NAME "DEB-DEFAULT")

# Dependencies
set(PACKAGE_REQUIRES "rocprofiler" CACHE STRING "Package dependencies")
set(CPACK_RPM_PACKAGE_REQUIRES "${PACKAGE_REQUIRES}, zlib")
set(CPACK_DEBIAN_PACKAGE_DEPENDS "${PACKAGE_REQUIRES}, zlib1g")

# Handle the project rebranding from omniperf to rocprofiler-compute
set(OMNIPERF_PACKAGE_NAME "omniperf")
set(CPACK_RPM_PACKAGE_PROVIDES ${OMNIPERF_PACKAGE_NAME})
set(CPACK_RPM_PACKAGE_OBSOLETES "${OMNIPERF_PACKAGE_NAME} < 3.0.0")
set(CPACK_RPM_PACKAGE_CONFLICTS ${OMNIPERF_PACKAGE_NAME})

set(CPACK_DEBIAN_PACKAGE_PROVIDES ${OMNIPERF_PACKAGE_NAME})
set(CPACK_DEBIAN_PACKAGE_REPLACES ${OMNIPERF_PACKAGE_NAME})
set(CPACK_DEBIAN_PACKAGE_BREAKS ${OMNIPERF_PACKAGE_NAME})

# Disable automatic dependency generation
set(CPACK_RPM_PACKAGE_AUTOREQPROV OFF)
set(CPACK_RPM_PACKAGE_AUTOREQ OFF)
set(CPACK_RPM_PACKAGE_AUTOPROV OFF)

if(INSTALL_TESTS)
    set(CPACK_RPM_TESTS_PACKAGE_REQUIRES ${CPACK_PACKAGE_NAME})
    set(CPACK_DEBIAN_TESTS_PACKAGE_DEPENDS ${CPACK_PACKAGE_NAME})
endif()

# ----- Check for packaging override -----
if(DEFINED ENV{ROCM_LIBPATCH_VERSION})
    set(CPACK_PACKAGE_VERSION "${CPACK_PACKAGE_VERSION}.$ENV{ROCM_LIBPATCH_VERSION}")
endif()

if(DEFINED ENV{CPACK_RPM_PACKAGE_RELEASE})
    set(CPACK_RPM_PACKAGE_RELEASE $ENV{CPACK_RPM_PACKAGE_RELEASE})
else()
    set(CPACK_RPM_PACKAGE_RELEASE "local")
endif()

if(DEFINED ENV{CPACK_DEBIAN_PACKAGE_RELEASE})
    set(CPACK_DEBIAN_PACKAGE_RELEASE $ENV{CPACK_DEBIAN_PACKAGE_RELEASE})
else()
    set(CPACK_DEBIAN_PACKAGE_RELEASE "local")
endif()

# Log package info
message(STATUS "  Package Name: ${CPACK_PACKAGE_NAME}")
message(STATUS "  Package Version: ${CPACK_PACKAGE_VERSION}")
message(STATUS "  RPM Package Release: ${CPACK_RPM_PACKAGE_RELEASE}")
message(STATUS "  Debian Package Release: ${CPACK_DEBIAN_PACKAGE_RELEASE}")
message(STATUS "  Packaging Install Prefix: ${CPACK_PACKAGING_INSTALL_PREFIX}")
message(STATUS "  Install Tests: ${INSTALL_TESTS}")
message(STATUS "  Package Dependencies: ${PACKAGE_REQUIRES}")
message(STATUS "  CPack Generator: ${CPACK_GENERATOR}")

configure_pkg(${PACKAGE_NAME} ${COMP_TYPE} ${CPACK_PACKAGE_VERSION} ${PKG_MAINTAINER_NM} ${PKG_MAINTAINER_EMAIL})

include(CPack)
