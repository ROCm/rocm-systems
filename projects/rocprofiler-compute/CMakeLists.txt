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
string(REGEX REPLACE "([0-9]+)\.([0-9]+)\.([0-9]+)(.*)" "\\1.\\2.\\3" OMNIPERF_VERSION
                     "${FULL_VERSION_STRING}")

string(REGEX REPLACE "(${OMNIPERF_VERSION})(.*)" "\\2" OMNIPERF_VERSION_TWEAK
                     "${FULL_VERSION_STRING}")
string(REGEX REPLACE "^\\." "" OMNIPERF_VERSION_TWEAK "${OMNIPERF_VERSION_TWEAK}")

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
    Python3 3.7
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
set(pythonDeps
    "astunparse"
    "colorlover"
    "dash"
    "matplotlib"
    "numpy"
    "pandas"
    "pymongo"
    "yaml"
    "tabulate"
    "tqdm"
    "dash_svg"
    "dash_bootstrap_components")

message(STATUS "Checking for required Python package dependencies...")
set_property(GLOBAL PROPERTY pythonDepsFlag "groovy")

function(checkPythonPackage [package])
    execute_process(
        COMMAND ${Python3_EXECUTABLE} -c "import ${ARGV0}"
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
    ${PROJECT_BINARY_DIR}/${CMAKE_INSTALL_DATAROOTDIR}/modulefiles/${PROJECT_NAME}/${PROJECT_VERSION}.lua
    @ONLY)

# Crusher mods
if(LOCALHOST MATCHES ".*\.crusher\.olcf\.ornl\.gov")
    list(APPEND CMAKE_MESSAGE_INDENT "  ")
    message(STATUS "Using crusher-specific modulefile modification")
    file(READ ${PROJECT_SOURCE_DIR}/cmake/modfile.crusher.mod mod_additions)
    file(
        APPEND
        ${PROJECT_BINARY_DIR}/${CMAKE_INSTALL_DATAROOTDIR}/modulefiles/${PROJECT_NAME}/${PROJECT_VERSION}.lua
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
        ${PROJECT_BINARY_DIR}/${CMAKE_INSTALL_DATAROOTDIR}/modulefiles/${PROJECT_NAME}/${PROJECT_VERSION}.lua
        ${mod_additions})
    list(POP_BACK CMAKE_MESSAGE_INDENT)
endif()

# git versioning file
if(${GIT_CLONE})
    configure_file(${PROJECT_SOURCE_DIR}/cmake/VERSION.sha.in
                   ${PROJECT_SOURCE_DIR}/VERSION.sha @ONLY)
endif()

enable_testing()

add_test(
    NAME test_analyze_commands
    COMMAND ${Python3_EXECUTABLE} -m pytest
            ${PROJECT_SOURCE_DIR}/tests/test_analyze_commands.py
    WORKING_DIRECTORY ${PROJECT_SOURCE_DIR})

add_test(
    NAME test_analyze_workloads
    COMMAND ${Python3_EXECUTABLE} -m pytest
            ${PROJECT_SOURCE_DIR}/tests/test_analyze_workloads.py
    WORKING_DIRECTORY ${PROJECT_SOURCE_DIR})

add_test(
    NAME test_saved_analysis
    COMMAND ${Python3_EXECUTABLE} -m pytest
            ${PROJECT_SOURCE_DIR}/tests/test_saved_analysis.py
    WORKING_DIRECTORY ${PROJECT_SOURCE_DIR})

# if(EXISTS ${PROJECT_SOURCE_DIR}/src/mibench/roofline/roofline.cpp) message(STATUS
# "Enabling optional roofline binaries..") foreach(ROCM_VERSION ${ROCM_VERSIONS})
# externalproject_add( roofline-rocm-${ROCM_VERSION} PREFIX
# ${PROJECT_BINARY_DIR}/rocm-${ROCM_VERSION}/roofline SOURCE_DIR
# ${PROJECT_SOURCE_DIR}/src/mibench/roofline BINARY_DIR
# ${PROJECT_BINARY_DIR}/rocm-${ROCM_VERSION}/roofline/build BUILD_BYPRODUCTS
# <BINARY_DIR>/roofline CONFIGURE_COMMAND ${CMAKE_COMMAND} -E env
# HIP_PATH=/opt/rocm-${ROCM_VERSION} ${CMAKE_COMMAND}
# -DCMAKE_CXX_COMPILER=/opt/rocm-${ROCM_VERSION}/hip/bin/hipcc
# -DCMAKE_BUILD_TYPE=${CMAKE_BUILD_TYPE} -B <BINARY_DIR> <SOURCE_DIR> BUILD_COMMAND
# ${CMAKE_COMMAND} --build <BINARY_DIR> --target all INSTALL_COMMAND "") #
# add_dependencies(roofline-rocm-${ROCM_VERSION} run-pyinstaller) string(REGEX REPLACE
# "([0-9]+)\.([0-9]+)\.([0-9]+)(.*)" "\\1.\\2" ROCM_BASIC_VERSION "${ROCM_VERSION}")
# install( PROGRAMS ${PROJECT_BINARY_DIR}/rocm-${ROCM_VERSION}/roofline/build/roofline
# DESTINATION ${CMAKE_INSTALL_BINDIR}/utils RENAME roofline-${ROCM_BASIC_VERSION})
# endforeach() else() message(STATUS "Optional roofline package not found") endif()

# ---------
# Install
# ---------

# top-level driver and associated .py files
install(PROGRAMS src/omniperf TYPE BIN)
install(FILES src/parser.py TYPE BIN)
install(FILES src/common.py TYPE BIN)
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
install(
    PROGRAMS src/utils/rooflines/roofline-rhel8-mi200-rocm5
             src/utils/rooflines/roofline-sle15sp3-mi200-rocm5
             src/utils/rooflines/roofline-ubuntu20_04-mi200-rocm5
    DESTINATION ${CMAKE_INSTALL_BINDIR}/utils/rooflines)
# src/perfmon_pub
install(
    DIRECTORY src/perfmon_pub
    TYPE BIN
    PATTERN "__pycache__" EXCLUDE)

# src/soc_params
install(
    DIRECTORY src/soc_params
    TYPE BIN
    PATTERN "__pycache__" EXCLUDE)

# samples
install(DIRECTORY sample TYPE DATA)

# modulefile
install(
    FILES
        ${PROJECT_BINARY_DIR}/${CMAKE_INSTALL_DATAROOTDIR}/modulefiles/${PROJECT_NAME}/${PROJECT_VERSION}.lua
    DESTINATION ${MOD_INSTALL_PATH}/${PROJECT_NAME})

# install(PROGRAMS src/omniperf.py DESTINATION ${OMNIPERF_PYINSTALLER_PATH})
# set(OMNIPERF_BUILD_ARGS "-y --workpath=${PROJECT_BINARY_DIR}/pyinstaller
# --distpath=${PROJECT_BINARY_DIR}/dist" CACHE STRING "Arguments for build.sh" FORCE)
#
# string(REPLACE " " ";" _OMNIPERF_BUILD_ARGS "${OMNIPERF_BUILD_ARGS}")
#
# add_custom_target( run-pyinstaller COMMAND ${PROJECT_SOURCE_DIR}/build.sh
# ${_OMNIPERF_BUILD_ARGS} WORKING_DIRECTORY ${PROJECT_SOURCE_DIR} BYPRODUCTS
# ${PROJECT_BINARY_DIR}/dist COMMENT "Running pyinstaller")
#
# set(OMNIPERF_PYINSTALLER_PATH
# "${CMAKE_INSTALL_LIBDIR}/python${PYTHON_VERSION}/site-packages/omniperf" CACHE STRING
# "Relative installation directory for omniperf")
#
# install( DIRECTORY ${PROJECT_BINARY_DIR}/dist/omniperf/ DESTINATION
# ${OMNIPERF_PYINSTALLER_PATH} USE_SOURCE_PERMISSIONS COMPONENT pyinstaller PATTERN ".a"
# EXCLUDE PATTERN ".git" EXCLUDE PATTERN "utils/roofline" EXCLUDE)
#
# install(FILES ${PROJECT_SOURCE_DIR}/VERSION DESTINATION
# ${OMNIPERF_PYINSTALLER_PATH}/../)
#
# file(RELATIVE_PATH OMNIPERF_RELATIVE_PATH
# "${CMAKE_INSTALL_PREFIX}/${CMAKE_INSTALL_BINDIR}"
# "${CMAKE_INSTALL_PREFIX}/${OMNIPERF_PYINSTALLER_PATH}")
#
# configure_file(${PROJECT_SOURCE_DIR}/cmake/omniperf.sh.in
# ${PROJECT_BINARY_DIR}/omniperf-sh @ONLY) install( PROGRAMS
# ${PROJECT_BINARY_DIR}/omniperf-sh DESTINATION ${CMAKE_INSTALL_BINDIR} RENAME omniperf)

# Packaging directives set(CPACK_GENERATOR "DEB" "RPM" "TGZ") set(CPACK_PACKAGE_NAME
# "omniperf") set(CPACK_PACKAGE_VENDOR "AMD") set(CPACK_PACKAGE_VERSION_MAJOR
# ${PROJECT_VERSION_MAJOR}) set(CPACK_PACKAGE_VERSION_MINOR ${PROJECT_VERSION_MINOR})
# set(CPACK_PACKAGE_VERSION_PATCH ${PROJECT_VERSION_PATCH}${OMNIPERF_VERSION_TWEAK})
# set(CPACK_PACKAGE_CONTACT "Advanced Micro Devices Inc.")
# set(CPACK_PACKAGE_DESCRIPTION_SUMMARY "omniperf") set(CPACK_RESOURCE_FILE_LICENSE
# "${CMAKE_CURRENT_SOURCE_DIR}/LICENSE")

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
