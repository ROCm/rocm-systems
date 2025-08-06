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
set(ROCPROFCOMPUTE_FULL_VERSION "${FULL_VERSION_STRING}")
string(REGEX REPLACE "([0-9]+)\.([0-9]+)\.([0-9]+)(.*)" "\\1.\\2.\\3"
                     ROCPROFCOMPUTE_VERSION "${FULL_VERSION_STRING}")

# string(REGEX REPLACE "(${ROCPROFCOMPUTE_VERSION})(.*)" "\\2"
# ROCPROFCOMPUTE_VERSION_TWEAK
# "${FULL_VERSION_STRING}")
# string(REGEX REPLACE "^\\." "" ROCPROFCOMPUTE_VERSION_TWEAK
# "${ROCPROFCOMPUTE_VERSION_TWEAK}")

project(
    rocprofiler-compute
    VERSION ${ROCPROFCOMPUTE_VERSION}
    LANGUAGES C
    DESCRIPTION
        "A kernel-level profiling tool for machine learning/HPC workloads running on AMD MI GPUs"
    HOMEPAGE_URL "https://github.com/ROCm/rocprofiler-compute")

set(PACKAGE_NAME "rocprofiler-compute")
set(PACKAGE_NAME_UNDERSCORE "rocprofiler_compute")
set(EXECUTABLE_NAME "rocprof-compute")

include(ExternalProject)
include(GNUInstallDirs)

# version control info
find_package(Git)
if(Git_FOUND AND EXISTS "${PROJECT_SOURCE_DIR}/.git")
    execute_process(
        COMMAND git log --pretty=format:%h -n 1
        OUTPUT_VARIABLE ROCPROFCOMPUTE_GIT_REV
        OUTPUT_STRIP_TRAILING_WHITESPACE)
    message(STATUS "Git revision: ${ROCPROFCOMPUTE_GIT_REV}")
    set(GIT_CLONE TRUE)
else()
    set(GIT_CLONE FALSER)
endif()

set(CMAKE_BUILD_TYPE "Release")
if(CMAKE_INSTALL_PREFIX_INITIALIZED_TO_DEFAULT)
    set(CMAKE_INSTALL_PREFIX
        "/opt/rocm"
        CACHE PATH "default install path" FORCE)
endif()
message(STATUS "Installation path: ${CMAKE_INSTALL_PREFIX}")

option(CHECK_PYTHON_DEPS "Verify necessary python dependencies" ON)
if(CHECK_PYTHON_DEPS)
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
        # Skip check for textual-fspicker
        if(${package} STREQUAL "textual-fspicker")
            message(STATUS "Skipping check for textual-fspicker")
            return()
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
endif()

# ----------------------
# modulefile creation
# ----------------------

set(MOD_INSTALL_PATH
    "${CMAKE_INSTALL_DATADIR}/${PROJECT_NAME}/modulefiles/${PROJECT_NAME}"
    CACHE STRING "Install path for modulefile")
message(STATUS "Modulefile install path: ${MOD_INSTALL_PATH}")

set(moduleFileTemplate "rocprofcompute.lua.in")

configure_file(
    ${PROJECT_SOURCE_DIR}/cmake/${moduleFileTemplate}
    ${PROJECT_BINARY_DIR}/${MOD_INSTALL_PATH}/${ROCPROFCOMPUTE_FULL_VERSION}.lua @ONLY)

# Thera mods
if(LOCALHOST MATCHES "TheraS01|.*\.thera\.amd\.com|thera-hn")
    list(APPEND CMAKE_MESSAGE_INDENT "  ")
    message(STATUS "Using thera-specific modulefile modification")
    file(READ ${PROJECT_SOURCE_DIR}/cmake/modfile.thera.mod mod_additions)
    file(APPEND
         ${PROJECT_BINARY_DIR}/${MOD_INSTALL_PATH}/${ROCPROFCOMPUTE_FULL_VERSION}.lua
         ${mod_additions})
    list(POP_BACK CMAKE_MESSAGE_INDENT)
endif()

# git versioning file
if(${GIT_CLONE})
    configure_file(${PROJECT_SOURCE_DIR}/cmake/VERSION.sha.in
                   ${PROJECT_SOURCE_DIR}/VERSION.sha @ONLY)
endif()

# Setup testing collateral

option(ENABLE_TESTS "Enable compilation of testing collateral" OFF)
set(CMAKE_HIP_FLAGS_RELEASE "-O2")
if(${ENABLE_TESTS})
    enable_language("C" "HIP")
    add_subdirectory(tests)

endif()
message(STATUS "Enable tests compilation: ${ENABLE_TESTS}")

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

# 2 CPU threads available for testing(test-analyze-commands)
set(PYTEST_NUMPROCS_ANALYSIS
    "4"
    CACHE STRING "Number of parallel threads to use with CPU-oriented tests")
message(STATUS "Pytest CPU threadcount: ${PYTEST_NUMPROCS_ANALYSIS}")

# ---------------------------
# profile mode tests
# ---------------------------

add_test(
    NAME test_profile_kernel_execution
    COMMAND
        ${Python3_EXECUTABLE} -m pytest -m kernel_execution
        --junitxml=tests/test_profile_kernel_execution.xml ${COV_OPTION}
        ${PROJECT_SOURCE_DIR}/tests/test_profile_general.py
    WORKING_DIRECTORY ${PROJECT_SOURCE_DIR})

add_test(
    NAME test_profile_dispatch
    COMMAND
        ${Python3_EXECUTABLE} -m pytest -m dispatch
        --junitxml=tests/test_profile_dispatch.xml ${COV_OPTION}
        ${PROJECT_SOURCE_DIR}/tests/test_profile_general.py
    WORKING_DIRECTORY ${PROJECT_SOURCE_DIR})

add_test(
    NAME test_profile_mem
    COMMAND ${Python3_EXECUTABLE} -m pytest -m mem --junitxml=tests/test_profile_mem.xml
            ${COV_OPTION} ${PROJECT_SOURCE_DIR}/tests/test_profile_general.py
    WORKING_DIRECTORY ${PROJECT_SOURCE_DIR})

add_test(
    NAME test_profile_join
    COMMAND ${Python3_EXECUTABLE} -m pytest -m join --junitxml=tests/test_profile_join.xml
            ${COV_OPTION} ${PROJECT_SOURCE_DIR}/tests/test_profile_general.py
    WORKING_DIRECTORY ${PROJECT_SOURCE_DIR})

add_test(
    NAME test_profile_sort
    COMMAND ${Python3_EXECUTABLE} -m pytest -m sort --junitxml=tests/test_profile_sort.xml
            ${COV_OPTION} ${PROJECT_SOURCE_DIR}/tests/test_profile_general.py
    WORKING_DIRECTORY ${PROJECT_SOURCE_DIR})

add_test(
    NAME test_profile_misc
    COMMAND ${Python3_EXECUTABLE} -m pytest -m misc --junitxml=tests/test_profile_misc.xml
            ${COV_OPTION} ${PROJECT_SOURCE_DIR}/tests/test_profile_general.py
    WORKING_DIRECTORY ${PROJECT_SOURCE_DIR})

add_test(
    NAME test_profile_section
    COMMAND
        ${Python3_EXECUTABLE} -m pytest -m section
        --junitxml=tests/test_profile_section.xml ${COV_OPTION}
        ${PROJECT_SOURCE_DIR}/tests/test_profile_general.py
    WORKING_DIRECTORY ${PROJECT_SOURCE_DIR})

add_test(
    NAME test_profile_sets_func
    COMMAND
        ${Python3_EXECUTABLE} -m pytest -m sets_func
        --junitxml=tests/test_profile_sets_func.xml ${COV_OPTION}
        ${PROJECT_SOURCE_DIR}/tests/test_profile_general.py
    WORKING_DIRECTORY ${PROJECT_SOURCE_DIR})

add_test(
    NAME test_profile_live_attach_detach
    COMMAND ${Python3_EXECUTABLE} -m pytest -m live_attach_detach --junitxml=tests/test_profile_misc.xml
            ${COV_OPTION} ${PROJECT_SOURCE_DIR}/tests/test_profile_general.py
    WORKING_DIRECTORY ${PROJECT_SOURCE_DIR})

set_tests_properties(
    test_profile_kernel_execution test_profile_dispatch test_profile_mem test_profile_join
    test_profile_sort test_profile_misc PROPERTIES LABELS "profile" RESOURCE_GROUPS
                                                   gpus:1)

# ---------------------------
# analysis command tests
# ---------------------------

add_test(
    NAME test_analyze_commands
    COMMAND
        ${Python3_EXECUTABLE} -m pytest -n ${PYTEST_NUMPROCS_ANALYSIS} --verbose
        --junitxml=tests/test_analyze_commands.xml ${COV_OPTION}
        ${PROJECT_SOURCE_DIR}/tests/test_analyze_commands.py
    WORKING_DIRECTORY ${PROJECT_SOURCE_DIR})

# ---------------------------
# analyze workloads tests
# ---------------------------

add_test(
    NAME test_analyze_workloads
    COMMAND
        ${Python3_EXECUTABLE} -m pytest -n ${PYTEST_NUMPROCS}
        --junitxml=tests/test_analyze_workloads.xml ${COV_OPTION}
        ${PROJECT_SOURCE_DIR}/tests/test_analyze_workloads.py
    WORKING_DIRECTORY ${PROJECT_SOURCE_DIR})

# ---------------------------
# TCP counter tests
# ---------------------------

add_test(
    NAME test_L1_cache_counters
    COMMAND
        ${Python3_EXECUTABLE} -m pytest -m L1_cache
        --junitxml=tests/test_L1_cache_counters.xml ${COV_OPTION}
        ${PROJECT_SOURCE_DIR}/tests/test_TCP_counters.py
    WORKING_DIRECTORY ${PROJECT_SOURCE_DIR})

# ---------------------------
# Spec tests
# ---------------------------

add_test(
    NAME test_num_xcds_spec_class
    COMMAND
        ${Python3_EXECUTABLE} -m pytest -m num_xcds_spec_class
        --junitxml=tests/test_num_xcds_spec_class.xml ${COV_OPTION}
        ${PROJECT_SOURCE_DIR}/tests/test_gpu_specs.py
    WORKING_DIRECTORY ${PROJECT_SOURCE_DIR})

add_test(
    NAME test_num_xcds_cli_output
    COMMAND
        ${Python3_EXECUTABLE} -m pytest -m num_xcds_cli_output
        --junitxml=tests/test_num_xcds_cli_output.xml ${COV_OPTION}
        ${PROJECT_SOURCE_DIR}/tests/test_gpu_specs.py
    WORKING_DIRECTORY ${PROJECT_SOURCE_DIR})

# ---------------------------
# DB Connector tests
# ---------------------------

add_test(
    NAME test_db_connector
    COMMAND ${Python3_EXECUTABLE} -m pytest --junitxml=tests/test_db_connector.xml
            ${COV_OPTION} ${PROJECT_SOURCE_DIR}/tests/test_db_connector.py
    WORKING_DIRECTORY ${PROJECT_SOURCE_DIR})

# ---------------------------
# Utils tests
# ---------------------------

add_test(
    NAME test_utils
    COMMAND ${Python3_EXECUTABLE} -m pytest --junitxml=tests/test_utils.xml ${COV_OPTION}
            ${PROJECT_SOURCE_DIR}/tests/test_utils.py
    WORKING_DIRECTORY ${PROJECT_SOURCE_DIR})

# -----------------------------------
# Autogenerated configuration tests
# -----------------------------------

add_test(
    NAME test_autogen_config
    COMMAND ${Python3_EXECUTABLE} -m pytest --junitxml=tests/test_autogen_config.xml
            ${COV_OPTION} ${PROJECT_SOURCE_DIR}/tests/test_autogen_config.py
    WORKING_DIRECTORY ${PROJECT_SOURCE_DIR})

# ---------
# Install
# ---------

# top-level rocprofiler-compute utility
install(
    PROGRAMS src/${EXECUTABLE_NAME}
    DESTINATION ${CMAKE_INSTALL_LIBEXECDIR}/${PROJECT_NAME}
    COMPONENT main)
# python dependency requirements
install(
    FILES requirements.txt
    DESTINATION ${CMAKE_INSTALL_LIBEXECDIR}/${PROJECT_NAME}
    COMPONENT main)
# support files and version info
install(
    FILES src/argparser.py src/config.py src/rocprof_compute_base.py src/roofline.py
          VERSION VERSION.sha
    DESTINATION ${CMAKE_INSTALL_LIBEXECDIR}/${PROJECT_NAME}
    COMPONENT main)
# src/rocprof_compute_analyze
install(
    DIRECTORY src/rocprof_compute_analyze
    DESTINATION ${CMAKE_INSTALL_LIBEXECDIR}/${PROJECT_NAME}
    COMPONENT main
    PATTERN src/rocprof_compute_analyze/tests EXCLUDE
    PATTERN "__pycache__" EXCLUDE)
# src/utils
install(
    DIRECTORY src/utils
    DESTINATION ${CMAKE_INSTALL_LIBEXECDIR}/${PROJECT_NAME}
    COMPONENT main
    PATTERN "rooflines*" EXCLUDE
    PATTERN "__pycache__" EXCLUDE)
# src/utils/rooflines
file(GLOB rooflinebins src/utils/rooflines/roofline-*)
install(
    PROGRAMS ${rooflinebins}
    DESTINATION ${CMAKE_INSTALL_BINDIR}
    COMPONENT main)
# src/rocprof_compute_soc
install(
    DIRECTORY src/rocprof_compute_soc
    DESTINATION ${CMAKE_INSTALL_LIBEXECDIR}/${PROJECT_NAME}
    COMPONENT main
    PATTERN "__pycache__" EXCLUDE)
# src/rocprof_compute_profile
install(
    DIRECTORY src/rocprof_compute_profile
    DESTINATION ${CMAKE_INSTALL_LIBEXECDIR}/${PROJECT_NAME}
    COMPONENT main
    PATTERN "__pycache__" EXCLUDE)
# src/rocprof_compute_tui
install(
    DIRECTORY src/rocprof_compute_tui
    DESTINATION ${CMAKE_INSTALL_LIBEXECDIR}/${PROJECT_NAME}
    COMPONENT main
    PATTERN "__pycache__" EXCLUDE)
# grafana assets
install(
    DIRECTORY grafana
    DESTINATION ${CMAKE_INSTALL_DATADIR}/${PROJECT_NAME}
    COMPONENT main)
# samples
install(
    DIRECTORY sample
    DESTINATION ${CMAKE_INSTALL_DATADIR}/${PROJECT_NAME}
    COMPONENT main
    FILES_MATCHING
    PATTERN "*.hip"
    PATTERN "*.h"
    PATTERN "*.cpp"
    PATTERN "workloads" EXCLUDE)
# modulefile
install(
    FILES ${PROJECT_BINARY_DIR}/${MOD_INSTALL_PATH}/${ROCPROFCOMPUTE_FULL_VERSION}.lua
    DESTINATION ${MOD_INSTALL_PATH}
    COMPONENT main)

# top-level symlink for bin/rocprof-compute
install(
    CODE "execute_process(
    COMMAND bash -c \"set -e
    cd \$ENV{DESTDIR}\${CMAKE_INSTALL_PREFIX}
    ln -sf ../${CMAKE_INSTALL_LIBEXECDIR}/${PROJECT_NAME}/${EXECUTABLE_NAME} ${CMAKE_INSTALL_BINDIR}/${EXECUTABLE_NAME}
    \")"
    COMPONENT main)

# License header update(s)
add_custom_target(
    license
    COMMAND
        ${PROJECT_SOURCE_DIR}/utils/update_license.py --source ${PROJECT_SOURCE_DIR}/src
        --license ${PROJECT_SOURCE_DIR}/LICENSE --extension '.py'
    COMMAND
        ${PROJECT_SOURCE_DIR}/utils/update_license.py --source ${PROJECT_SOURCE_DIR}
        --license ${PROJECT_SOURCE_DIR}/LICENSE --file
        "src/${PACKAGE_NAME},cmake/Dockerfile,cmake/rocm_install.sh,docker/docker-entrypoint.sh,src/rocprof_compute_analyze/convertor/mongodb/convert"
    )

# Standalone binary creation
add_custom_target(
    standalonebinary
    # Change working directory to src
    WORKING_DIRECTORY ${PROJECT_SOURCE_DIR}/src
    # Check nuitka
    COMMAND ${Python3_EXECUTABLE} -m pip list | grep -i nuitka > /dev/null 2>&1
    # Check patchelf
    COMMAND ${Python3_EXECUTABLE} -m pip list | grep -i patchelf > /dev/null 2>&1
    # Create VERSION.sha file
    COMMAND git -C ${PROJECT_SOURCE_DIR} rev-parse HEAD > VERSION.sha
    # Build standalone binary
    # NOTE: --no-deployment-flag=self-execution is used to avoid self-execution and fork
    # bombs as explained in
    # https://nuitka.net/user-documentation/common-issue-solutions.html#fork-bombs-self-execution
    COMMAND
        ${Python3_EXECUTABLE} -m nuitka --mode=onefile --no-deployment-flag=self-execution
        --include-data-files=${PROJECT_SOURCE_DIR}/VERSION*=./ --enable-plugin=no-qt
        --include-package=dash_svg --include-package-data=dash_svg
        --include-package=dash_bootstrap_components
        --include-package-data=dash_bootstrap_components --include-package=plotly
        --include-package-data=plotly --include-package=kaleido
        --include-package-data=kaleido --include-package=rocprof_compute_analyze
        --include-package-data=rocprof_compute_analyze
        --include-package=rocprof_compute_soc --include-package-data=rocprof_compute_soc
        --include-package=utils --include-package-data=utils rocprof-compute
    # Remove library rpath from executable
    COMMAND patchelf --remove-rpath rocprof-compute.bin
    # Move to build directory
    COMMAND mv rocprof-compute.bin ${CMAKE_BINARY_DIR})

install(
    FILES ${PROJECT_SOURCE_DIR}/LICENSE
    DESTINATION ${CMAKE_INSTALL_DOCDIR}
    COMPONENT main)

# TEST collateral
option(INSTALL_TESTS "Build test suite" OFF)
if(INSTALL_TESTS)
    install(
        DIRECTORY tests
        DESTINATION ${CMAKE_INSTALL_LIBEXECDIR}/${PROJECT_NAME}
        COMPONENT tests
        FILES_MATCHING
        PATTERN "*.py"
        PATTERN "__pycache__" EXCLUDE)
    install(
        FILES requirements-test.txt
        DESTINATION ${CMAKE_INSTALL_LIBEXECDIR}/${PROJECT_NAME}
        COMPONENT tests)
    install(
        FILES ${CMAKE_CURRENT_BINARY_DIR}/CTestTestfile.cmake
        COMPONENT tests
        DESTINATION ${CMAKE_INSTALL_LIBEXECDIR}/${PROJECT_NAME})
endif()
message(STATUS "Install tests: ${INSTALL_TESTS}")

# ----------
# Packaging
# ----------

message(STATUS "Packaging config...")
set(CPACK_GENERATOR
    "DEB" "RPM"
    CACHE STRING "")
set(CPACK_PACKAGE_NAME
    "${PROJECT_NAME}"
    CACHE STRING "")
set(CPACK_PACKAGE_CONTACT "https://github.com/ROCm/rocprofiler-compute")
set(CPACK_PACKAGE_DESCRIPTION_SUMMARY
    "ROCm Compute Profiler: tool for GPU performance profiling")
set(CPACK_RPM_PACKAGE_DESCRIPTION
    "ROCm Compute Profiler is a performance analysis tool for profiling
machine learning/HPC workloads running on AMD GPUs.")
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
set(PACKAGE_REQUIRES
    "rocprofiler"
    CACHE STRING "Package dependencies")
set(CPACK_RPM_PACKAGE_REQUIRES ${PACKAGE_REQUIRES})
set(CPACK_DEBIAN_PACKAGE_DEPENDS ${PACKAGE_REQUIRES})

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
