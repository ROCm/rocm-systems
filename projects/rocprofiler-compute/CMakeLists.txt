cmake_minimum_required(VERSION 3.19 FATAL_ERROR)

if(CMAKE_SOURCE_DIR STREQUAL CMAKE_BINARY_DIR AND CMAKE_CURRENT_SOURCE_DIR STREQUAL
                                                  CMAKE_SOURCE_DIR)
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
set(OMNIPERF_FULL_VERSION "${FULL_VERSION_STRING}")
string(REGEX REPLACE "([0-9]+)\.([0-9]+)\.([0-9]+)(.*)" "\\1.\\2.\\3" OMNIPERF_VERSION
                     "${FULL_VERSION_STRING}")

# string(REGEX REPLACE "(${OMNIPERF_VERSION})(.*)" "\\2" OMNIPERF_VERSION_TWEAK
# "${FULL_VERSION_STRING}")
# string(REGEX REPLACE "^\\." "" OMNIPERF_VERSION_TWEAK "${OMNIPERF_VERSION_TWEAK}")

project(
    omniperf
    VERSION ${OMNIPERF_VERSION}
    LANGUAGES C
    DESCRIPTION "OmniPerf"
    HOMEPAGE_URL "https://github.com/AMDResearch/omniperf")

include(ExternalProject)
include(GNUInstallDirs)

# version control info
find_package(Git)
if(Git_FOUND AND EXISTS "${PROJECT_SOURCE_DIR}/.git")
    execute_process(
        COMMAND git log --pretty=format:%h -n 1
        OUTPUT_VARIABLE OMNIPERF_GIT_REV
        OUTPUT_STRIP_TRAILING_WHITESPACE)
    message(STATUS "Git revision: ${OMNIPERF_GIT_REV}")
    set(GIT_CLONE TRUE)
else()
    set(GIT_CLONE FALSER)
endif()

set(CMAKE_BUILD_TYPE "Release")
if(CMAKE_INSTALL_PREFIX_INITIALIZED_TO_DEFAULT)
    set(CMAKE_INSTALL_PREFIX
        "/opt/apps/omniperf"
        CACHE PATH "default install path" FORCE)
endif()
message(STATUS "Installation path: ${CMAKE_INSTALL_PREFIX}")

# Python 3 is required
message(STATUS "Detecting Python interpreter...")
find_package(
    Python3 3.8
    COMPONENTS Interpreter
    REQUIRED)

# Allow user-provided python search path
if(DEFINED PYTHON_DEPS)
    set(ENV{PYTHONPATH} "${PYTHON_DEPS}")
    message(STATUS "Optional PYTHON_DEPS provided:")
    list(APPEND CMAKE_MESSAGE_INDENT "  ")
    message(STATUS "including ${PYTHON_DEPS} in search path")
    list(POP_BACK CMAKE_MESSAGE_INDENT)
endif()

# Check required Python packages
file(STRINGS "${CMAKE_CURRENT_SOURCE_DIR}/requirements.txt" pythonDeps)

message(STATUS "Checking for required Python package dependencies...")
set_property(GLOBAL PROPERTY pythonDepsFlag "groovy")

function(checkPythonPackage [package])
    # mapping for non-default package names
    set(PACKAGE ${ARGV0})
    if(${ARGV0} STREQUAL "pyyaml")
        set(PACKAGE "yaml")
    endif()
    execute_process(
        COMMAND ${Python3_EXECUTABLE} -c "import ${PACKAGE}"
        OUTPUT_QUIET ERROR_QUIET
        RESULT_VARIABLE EXIT_CODE)
    if(${EXIT_CODE} EQUAL 0)
        message(STATUS "${ARGV0} = yes")
    else()
        message(STATUS "${ARGV0} = missing")
        set_property(GLOBAL PROPERTY pythonDepsFlag "missing")
    endif()
endfunction()

list(APPEND CMAKE_MESSAGE_INDENT "  ")
foreach(package IN LISTS pythonDeps)
    # Filter out any version requirements from requirements.txt
    string(REGEX REPLACE "[><=].*" "" package "${package}")
    string(REPLACE "-" "_" package "${package}")
    checkpythonpackage(${package})
endforeach()
list(POP_BACK CMAKE_MESSAGE_INDENT)

get_property(pythonDepsInstalled GLOBAL PROPERTY pythonDepsFlag)
if(${pythonDepsInstalled} STREQUAL "groovy")
    message(STATUS "OK: Python dependencies available in current environment.")
else()
    message(
        FATAL_ERROR
            "\nNecessary Python package dependencies not found. Please install required dependencies "
            "above using your favorite package manager. If using pip, consider running:\n"
            "python3 -m pip install -r requirements.txt\n"
            "at the top-level of this repository. If preparing a shared installation for "
            "multiple users, consider adding the -t <target-dir> option to install necessary dependencies "
            "into a shared directory, e.g.\n"
            "python3 -m pip install -t <shared-install-path> -r requirements.txt\n"
            "Note that the -DPYTHON_DEPS=<shared-install-path> can be used to provide an "
            "additional search path to cmake for python packages.")
endif()

# ----------------------
# modulefile creation
# ----------------------

set(MOD_INSTALL_PATH
    "${CMAKE_INSTALL_PREFIX}/${CMAKE_INSTALL_DATAROOTDIR}/modulefiles"
    CACHE STRING "Install path for modulefile")
message(STATUS "Modulefile install path: ${MOD_INSTALL_PATH}")

set(moduleFileTemplate "omniperf.lua.in")

configure_file(
    ${PROJECT_SOURCE_DIR}/cmake/${moduleFileTemplate}
    ${PROJECT_BINARY_DIR}/${CMAKE_INSTALL_DATAROOTDIR}/modulefiles/${PROJECT_NAME}/${OMNIPERF_FULL_VERSION}.lua
    @ONLY)

# Crusher mods
if(LOCALHOST MATCHES ".*\.crusher\.olcf\.ornl\.gov")
    list(APPEND CMAKE_MESSAGE_INDENT "  ")
    message(STATUS "Using crusher-specific modulefile modification")
    file(READ ${PROJECT_SOURCE_DIR}/cmake/modfile.crusher.mod mod_additions)
    file(
        APPEND
        ${PROJECT_BINARY_DIR}/${CMAKE_INSTALL_DATAROOTDIR}/modulefiles/${PROJECT_NAME}/${OMNIPERF_FULL_VERSION}.lua
        ${mod_additions})
    list(POP_BACK CMAKE_MESSAGE_INDENT)
endif()

# Thera mods
if(LOCALHOST MATCHES "TheraS01|.*\.thera\.amd\.com|thera-hn")
    list(APPEND CMAKE_MESSAGE_INDENT "  ")
    message(STATUS "Using thera-specific modulefile modification")
    file(READ ${PROJECT_SOURCE_DIR}/cmake/modfile.thera.mod mod_additions)
    file(
        APPEND
        ${PROJECT_BINARY_DIR}/${CMAKE_INSTALL_DATAROOTDIR}/modulefiles/${PROJECT_NAME}/${OMNIPERF_FULL_VERSION}.lua
        ${mod_additions})
    list(POP_BACK CMAKE_MESSAGE_INDENT)
endif()

# git versioning file
if(${GIT_CLONE})
    configure_file(${PROJECT_SOURCE_DIR}/cmake/VERSION.sha.in
                   ${PROJECT_SOURCE_DIR}/VERSION.sha @ONLY)
endif()

# Setup testing collateral
enable_testing()

option(ENABLE_COVERAGE "Enable code coverage" OFF)
set(COV_OPTION "")
if(${ENABLE_COVERAGE})
    set(COV_OPTION "--cov=src" "--cov-append" "--cov-report=term-missing"
                   "--cov-report=lcov:tests/coverage.info")
    # "--cov-report=term-missing" "--cov-report=xml:tests/coverage.xml")
endif()
message(STATUS "Code coverage: ${ENABLE_COVERAGE}")

# CPU threads available for testing
set(PYTEST_NUMPROCS
    "1"
    CACHE STRING "Number of parallel threads to use with CPU-oriented tests")
message(STATUS "Pytest CPU threadcount: ${PYTEST_NUMPROCS}")

# ---------------------------
# profile mode tests
# ---------------------------

add_test(
    NAME test_profile_kernel_execution
    COMMAND pytest -m kernel_execution --junitxml=tests/test_profile_kernel_execution.xml
            ${COV_OPTION} ${PROJECT_SOURCE_DIR}/tests/test_profile_general.py
    WORKING_DIRECTORY ${PROJECT_SOURCE_DIR})

add_test(
    NAME test_profile_ipblocks
    COMMAND pytest -m ipblocks --junitxml=tests/test_profile_ipblocks.xml ${COV_OPTION}
            ${PROJECT_SOURCE_DIR}/tests/test_profile_general.py
    WORKING_DIRECTORY ${PROJECT_SOURCE_DIR})
set_property(TEST test_profile_ipblocks PROPERTY COST 11)

add_test(
    NAME test_profile_dispatch
    COMMAND pytest -m dispatch --junitxml=tests/test_profile_dispatch.xml ${COV_OPTION}
            ${PROJECT_SOURCE_DIR}/tests/test_profile_general.py
    WORKING_DIRECTORY ${PROJECT_SOURCE_DIR})
set_property(TEST test_profile_ipblocks PROPERTY COST 5)

add_test(
    NAME test_profile_verbosity
    COMMAND pytest -m verbosity --junitxml=tests/test_profile_verbosity.xml ${COV_OPTION}
            ${PROJECT_SOURCE_DIR}/tests/test_profile_general.py
    WORKING_DIRECTORY ${PROJECT_SOURCE_DIR})
set_property(TEST test_profile_ipblocks PROPERTY COST 10)

add_test(
    NAME test_profile_mem
    COMMAND pytest -m mem --junitxml=tests/test_profile_mem.xml ${COV_OPTION}
            ${PROJECT_SOURCE_DIR}/tests/test_profile_general.py
    WORKING_DIRECTORY ${PROJECT_SOURCE_DIR})

add_test(
    NAME test_profile_join
    COMMAND pytest -m join --junitxml=tests/test_profile_join.xml ${COV_OPTION}
            ${PROJECT_SOURCE_DIR}/tests/test_profile_general.py
    WORKING_DIRECTORY ${PROJECT_SOURCE_DIR})

add_test(
    NAME test_profile_sort
    COMMAND pytest -m sort --junitxml=tests/test_profile_sort.xml ${COV_OPTION}
            ${PROJECT_SOURCE_DIR}/tests/test_profile_general.py
    WORKING_DIRECTORY ${PROJECT_SOURCE_DIR})

add_test(
    NAME test_profile_misc
    COMMAND pytest -m misc --junitxml=tests/test_profile_misc.xml ${COV_OPTION}
            ${PROJECT_SOURCE_DIR}/tests/test_profile_general.py
    WORKING_DIRECTORY ${PROJECT_SOURCE_DIR})

set_tests_properties(
    test_profile_kernel_execution
    test_profile_ipblocks
    test_profile_dispatch
    test_profile_verbosity
    test_profile_mem
    test_profile_join
    test_profile_sort
    test_profile_misc
    PROPERTIES LABELS "profile" RESOURCE_GROUPS gpus:1)

# original add_test( NAME test_profiler_general COMMAND pytest ${COV_OPTION}
# ${PROJECT_SOURCE_DIR}/tests/test_profile_general.py WORKING_DIRECTORY
# ${PROJECT_SOURCE_DIR})

# ---------------------------
# analysis command tests
# ---------------------------

add_test(
    NAME test_analyze_commands
    COMMAND pytest -n ${PYTEST_NUMPROCS} --junitxml=tests/test_analyze_commands.xml
            ${COV_OPTION} ${PROJECT_SOURCE_DIR}/tests/test_analyze_commands.py
    WORKING_DIRECTORY ${PROJECT_SOURCE_DIR})

# add_test( NAME test_analyze_commands_misc COMMAND pytest -m misc ${COV_OPTION}
# ${PROJECT_SOURCE_DIR}/tests/test_analyze_commands.py WORKING_DIRECTORY
# ${PROJECT_SOURCE_DIR})
#
# add_test( NAME test_analyze_commands_list_metrics COMMAND pytest -m list_metrics
# ${COV_OPTION} ${PROJECT_SOURCE_DIR}/tests/test_analyze_commands.py WORKING_DIRECTORY
# ${PROJECT_SOURCE_DIR})
#
# add_test( NAME test_analyze_commands_filter_metrics COMMAND pytest -m filter_metrics
# ${COV_OPTION} ${PROJECT_SOURCE_DIR}/tests/test_analyze_commands.py WORKING_DIRECTORY
# ${PROJECT_SOURCE_DIR})
#
# add_test( NAME test_analyze_commands_filter_kernel COMMAND pytest -m filter_kernel
# ${COV_OPTION} ${PROJECT_SOURCE_DIR}/tests/test_analyze_commands.py WORKING_DIRECTORY
# ${PROJECT_SOURCE_DIR})
#
# add_test( NAME test_analyze_commands_dispatch COMMAND pytest -m dispatch ${COV_OPTION}
# ${PROJECT_SOURCE_DIR}/tests/test_analyze_commands.py WORKING_DIRECTORY
# ${PROJECT_SOURCE_DIR})
#
# add_test( NAME test_analyze_commands_normal_unit COMMAND pytest -m normal_unit
# ${COV_OPTION} ${PROJECT_SOURCE_DIR}/tests/test_analyze_commands.py WORKING_DIRECTORY
# ${PROJECT_SOURCE_DIR})
#
# add_test( NAME test_analyze_commands_max_kernel COMMAND pytest -m max_kernel
# ${COV_OPTION} ${PROJECT_SOURCE_DIR}/tests/test_analyze_commands.py WORKING_DIRECTORY
# ${PROJECT_SOURCE_DIR})
#
# add_test( NAME test_analyze_commands_time_unit COMMAND pytest -m time_unit ${COV_OPTION}
# ${PROJECT_SOURCE_DIR}/tests/test_analyze_commands.py WORKING_DIRECTORY
# ${PROJECT_SOURCE_DIR})
#
# add_test( NAME test_analyze_commands_decimal COMMAND pytest -m decimal ${COV_OPTION}
# ${PROJECT_SOURCE_DIR}/tests/test_analyze_commands.py WORKING_DIRECTORY
# ${PROJECT_SOURCE_DIR})
#
# add_test( NAME test_analyze_commands_col COMMAND pytest -m col ${COV_OPTION}
# ${PROJECT_SOURCE_DIR}/tests/test_analyze_commands.py WORKING_DIRECTORY
# ${PROJECT_SOURCE_DIR})
#
# add_test( NAME test_analyze_commands_kernel_verbose COMMAND pytest -m kernel_verbose
# ${COV_OPTION} ${PROJECT_SOURCE_DIR}/tests/test_analyze_commands.py WORKING_DIRECTORY
# ${PROJECT_SOURCE_DIR})
#
# set_tests_properties( test_analyze_commands_misc test_analyze_commands_list_metrics
# test_analyze_commands_filter_metrics test_analyze_commands_filter_kernel
# test_analyze_commands_dispatch test_analyze_commands_normal_unit
# test_analyze_commands_max_kernel test_analyze_commands_time_unit
# test_analyze_commands_decimal test_analyze_commands_col
# test_analyze_commands_kernel_verbose PROPERTIES LABELS "analyze_commands")

# ---------------------------
# analyze workloads tests
# ---------------------------

add_test(
    NAME test_analyze_workloads
    COMMAND pytest -n ${PYTEST_NUMPROCS} --junitxml=tests/test_analyze_workloads.xml
            ${COV_OPTION} ${PROJECT_SOURCE_DIR}/tests/test_analyze_workloads.py
    WORKING_DIRECTORY ${PROJECT_SOURCE_DIR})

# ---------------------------
# saved analysis tests
# ---------------------------

add_test(
    NAME test_saved_analysis
    COMMAND pytest ${COV_OPTION} ${PROJECT_SOURCE_DIR}/tests/test_saved_analysis.py
    WORKING_DIRECTORY ${PROJECT_SOURCE_DIR})

# ---------
# Install
# ---------

# top-level driver and associated .py files
install(PROGRAMS src/omniperf TYPE BIN)
install(FILES src/argparser.py TYPE BIN)
install(FILES src/config.py TYPE BIN)
install(FILES src/omniperf_base.py TYPE BIN)
install(FILES src/roofline.py TYPE BIN)

install(FILES VERSION VERSION.sha DESTINATION ${CMAKE_INSTALL_PREFIX})
# src/omniperf_analyze
install(
    DIRECTORY src/omniperf_analyze
    TYPE BIN
    PATTERN src/omniperf_analyze/tests EXCLUDE
    PATTERN "__pycache__" EXCLUDE)
# src/utils
install(
    DIRECTORY src/utils
    TYPE BIN
    PATTERN "rooflines*" EXCLUDE)
# src/utils/rooflines
file(GLOB rooflinebins src/utils/rooflines/roofline-*)
install(PROGRAMS ${rooflinebins} DESTINATION ${CMAKE_INSTALL_BINDIR}/utils/rooflines)
# src/omniperf_soc
install(
    DIRECTORY src/omniperf_soc
    TYPE BIN
    PATTERN "__pycache__" EXCLUDE)
# src/omniperf_profile
install(
    DIRECTORY src/omniperf_profile
    TYPE BIN
    PATTERN "__pycache__" EXCLUDE)
# samples
install(DIRECTORY sample TYPE DATA)
# modulefile
install(
    FILES
        ${PROJECT_BINARY_DIR}/${CMAKE_INSTALL_DATAROOTDIR}/modulefiles/${PROJECT_NAME}/${OMNIPERF_FULL_VERSION}.lua
    DESTINATION ${MOD_INSTALL_PATH}/${PROJECT_NAME})

# License header update(s)
add_custom_target(
    license
    COMMAND
        ${PROJECT_SOURCE_DIR}/utils/update_license.py --source ${PROJECT_SOURCE_DIR}/src
        --license ${PROJECT_SOURCE_DIR}/LICENSE --extension '.py'
    COMMAND
        ${PROJECT_SOURCE_DIR}/utils/update_license.py --source ${PROJECT_SOURCE_DIR}
        --license ${PROJECT_SOURCE_DIR}/LICENSE --file
        "src/omniperf,cmake/Dockerfile,cmake/rocm_install.sh,docker/docker-entrypoint.sh,src/omniperf_analyze/convertor/mongodb/convert"
    )

# Source tarball
set(CPACK_SOURCE_GENERATOR "TGZ")
set(CPACK_SOURCE_PACKAGE_FILE_NAME ${CMAKE_PROJECT_NAME}-${FULL_VERSION_STRING})
set(CPACK_SOURCE_IGNORE_FILES
    ".*~$"
    \.git/
    \.github
    \.gitmodules
    \.gitignore
    /tests
    /build)
include(CPack)
