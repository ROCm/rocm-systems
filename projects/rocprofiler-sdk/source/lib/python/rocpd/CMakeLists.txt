#
# CMakeLists.txt for rocpd pure Python implementation
# This creates a version-agnostic Python package that works across all Python 3.6+
#

message(STATUS "")
message(STATUS "===============================================")
message(STATUS "rocpd Pure Python Package Configuration")
message(STATUS "===============================================")

# Set the installation directory - VERSION AGNOSTIC! No more python3.8, python3.9, etc. -
# just one location for all Python versions
set(ROCPD_INSTALL_DIR
    "${CMAKE_INSTALL_LIBDIR}/python/site-packages/rocpd"
    CACHE STRING "Installation directory for rocpd pure Python package")

# Also set the build directory location for testing
set(ROCPD_BUILD_DIR "${CMAKE_BINARY_DIR}/${ROCPD_INSTALL_DIR}")

# Configure the __init__.py file to substitute CMake variables
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/__init__.py
               ${CMAKE_CURRENT_BINARY_DIR}/__init__.py.in @ONLY)

file(
    GENERATE
    OUTPUT ${ROCPD_BUILD_DIR}/__init__.py
    INPUT ${CMAKE_CURRENT_BINARY_DIR}/__init__.py.in)

# Copy all other Python files to build directory for testing
file(
    COPY ${CMAKE_CURRENT_SOURCE_DIR}/libpyrocpd.py
         ${CMAKE_CURRENT_SOURCE_DIR}/importer.py
         ${CMAKE_CURRENT_SOURCE_DIR}/output_config.py
         ${CMAKE_CURRENT_SOURCE_DIR}/schema.py
         ${CMAKE_CURRENT_SOURCE_DIR}/csv.py
         ${CMAKE_CURRENT_SOURCE_DIR}/pftrace.py
         ${CMAKE_CURRENT_SOURCE_DIR}/otf2.py
         ${CMAKE_CURRENT_SOURCE_DIR}/query.py
         ${CMAKE_CURRENT_SOURCE_DIR}/time_window.py
         ${CMAKE_CURRENT_SOURCE_DIR}/summary.py
         ${CMAKE_CURRENT_SOURCE_DIR}/__main__.py
    DESTINATION ${ROCPD_BUILD_DIR})

# List all Python files to install
set(ROCPD_PYTHON_FILES
    ${ROCPD_BUILD_DIR}/__init__.py
    ${CMAKE_CURRENT_SOURCE_DIR}/libpyrocpd.py
    ${CMAKE_CURRENT_SOURCE_DIR}/importer.py
    ${CMAKE_CURRENT_SOURCE_DIR}/output_config.py
    ${CMAKE_CURRENT_SOURCE_DIR}/schema.py
    ${CMAKE_CURRENT_SOURCE_DIR}/csv.py
    ${CMAKE_CURRENT_SOURCE_DIR}/pftrace.py
    ${CMAKE_CURRENT_SOURCE_DIR}/otf2.py
    ${CMAKE_CURRENT_SOURCE_DIR}/query.py
    ${CMAKE_CURRENT_SOURCE_DIR}/time_window.py
    ${CMAKE_CURRENT_SOURCE_DIR}/summary.py
    ${CMAKE_CURRENT_SOURCE_DIR}/__main__.py)

message(STATUS "  Installation directory: ${ROCPD_INSTALL_DIR}")
message(
    STATUS "  Python files to install: ${CMAKE_CURRENT_LIST_FILE}(${ROCPD_PYTHON_FILES})")
message(STATUS "  Python version agnostic: YES")
message(STATUS "  Binding type: Pure Python (no compilation)")
message(STATUS "  Dependencies: sqlite3 (built-in), otf2 (PyPI), perfetto (PyPI)")
message(STATUS "")

# Install all Python files
install(
    FILES ${ROCPD_PYTHON_FILES}
    DESTINATION ${ROCPD_INSTALL_DIR}
    COMPONENT rocpd)

# Create requirements.txt for Python dependencies
file(
    WRITE ${CMAKE_CURRENT_BINARY_DIR}/requirements.txt
    "# rocpd Pure Python Dependencies\n"
    "# Install with: pip install -r requirements.txt\n\n"
    "# OTF2 trace export (optional - only needed for OTF2 export)\n"
    "otf2>=3.0\n\n"
    "# Perfetto trace export (optional - only needed for Perfetto export)\n"
    "perfetto>=0.11.0\n\n"
    "# Note: Both otf2 and perfetto are optional dependencies.\n"
    "# CSV export and all other functionality works without them.\n")

install(
    FILES ${CMAKE_CURRENT_BINARY_DIR}/requirements.txt
    DESTINATION ${ROCPD_INSTALL_DIR}
    COMPONENT rocpd)

# Add test if we have Python for testing
if(ROCPROFILER_BUILD_TESTS)
    # Find any available Python3
    find_package(Python3 COMPONENTS Interpreter)

    if(Python3_FOUND)
        # Create a simple test script
        file(
            WRITE ${CMAKE_CURRENT_BINARY_DIR}/test_rocpd.py
            "#!/usr/bin/env python3
import sys
import os

# Add the build site-packages directory to the path
sys.path.insert(0, '${CMAKE_BINARY_DIR}/lib/python/site-packages')

print('='*70)
print('Testing rocpd pure Python implementation')
print('='*70)

try:
    # Test import
    import rocpd
    print(f'✓ Successfully imported rocpd')
    print(f'  Python version: {sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}')
    print(f'  Binding type: {rocpd.version_info.get(\"binding_type\", \"unknown\")}')
    print()

    # Test sqlite3 availability
    import sqlite3
    conn = sqlite3.connect(':memory:')
    print('✓ sqlite3 is available')
    conn.close()

    # Test basic functionality
    print('\\nTesting basic rocpd functionality...')

    # Test format_path
    path = rocpd.format_path('/tmp/%pid%/output.db')
    print(f'✓ format_path() works: {path}')

    # Test enumerations
    print(f'✓ agent_indexing enum: {rocpd.libpyrocpd.agent_indexing.node}')

    # Test data classes
    config = rocpd.libpyrocpd.output_config()
    print(f'✓ output_config class works')

    print('\\n' + '='*70)
    print('All basic tests passed!')
    print('='*70)
    print()
    print('Note: Full functionality requires:')
    print('  - pip install otf2 (for OTF2 export)')
    print('  - pip install perfetto (for Perfetto export)')
    sys.exit(0)

except Exception as e:
    print(f'✗ FAIL: {e}', file=sys.stderr)
    import traceback
    traceback.print_exc()
    sys.exit(1)
")

        # Add a test target
        add_test(
            NAME rocpd_basic_test
            COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_BINARY_DIR}/test_rocpd.py
            WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR})

        set_tests_properties(
            rocpd_basic_test
            PROPERTIES ENVIRONMENT
                       "PYTHONPATH=${CMAKE_BINARY_DIR}/lib/python/site-packages" LABELS
                       "python;rocpd;pure_python")
    endif()
endif()
