# Copyright (c) Advanced Micro Devices, Inc. All rights reserved.
#
# SPDX-License-Identifier: MIT

include_guard(GLOBAL)

#-----------------------------------------------------------------------------
# Option to build the hipFile API documentation
#-----------------------------------------------------------------------------
option(HIPFILE_BUILD_DOCS "Build hipFile API docs (requires Doxygen)" OFF)

if(HIPFILE_BUILD_DOCS)
    find_package(Doxygen REQUIRED)

    # Set Doxygen input (pasted into Doxyfile.in)
    set(HIPFILE_DOXYFILE_INPUT "${HIPFILE_ROOT_PATH}/include")

    # Set the path to the documentation
    set(HIPFILE_DOC_PATH "${CMAKE_CURRENT_BINARY_DIR}/docs")

    # Set the Doxyfile install location
    set(HIPFILE_DOXYFILE ${HIPFILE_DOC_PATH}/Doxyfile)

    # Create the Doxyfile from the input file
    configure_file("docs/doxygen/Doxyfile.in" ${HIPFILE_DOXYFILE})

    # Set the output directory
    set(DOXYGEN_OUT ${HIPFILE_DOC_PATH})

    # Configure the documentation build
    add_custom_target("doc"
        COMMAND ${DOXYGEN_EXECUTABLE} ${HIPFILE_DOXYFILE}
        WORKING_DIRECTORY ${HIPFILE_DOC_PATH}
        COMMENT "Generating hipFile API documentation with Doxygen"
        VERBATIM
    )

endif()
