## SPDX-License-Identifier: MIT
## SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.

# Test suite entry point.  Sub-directories are added based on the
# AMD_DBGAPI_BUILD_UNIT_TESTS / AMD_DBGAPI_BUILD_FEATURE_TESTS options
# set in the parent CMakeLists.txt.  For more information, see README.md.

# Helper functions add_dbgapi_unit_test() / add_dbgapi_feature_test()
# (defined in cmake/AddDbgapiTest.cmake) apply these labels automatically.

# Make GoogleTest + GMock available to all test sub-directories.
include(${CMAKE_CURRENT_SOURCE_DIR}/../cmake/DbgapiGoogleTest.cmake)

# Bring in the add_dbgapi_*_test() helper functions.
include(${CMAKE_CURRENT_SOURCE_DIR}/../cmake/AddDbgapiTest.cmake)

if(AMD_DBGAPI_BUILD_UNIT_TESTS)
  add_subdirectory(unit)
endif()

if(AMD_DBGAPI_BUILD_FEATURE_TESTS)
  add_subdirectory(feature)
endif()
