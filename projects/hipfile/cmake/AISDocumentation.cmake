# Copyright (c) Advanced Micro Devices, Inc. All rights reserved.
#
# SPDX-License-Identifier: MIT

#-----------------------------------------------------------------------------
# Option to build the hipFile documentation
#
# TODO: Consider turning this into two options, one of which just requires
#       Doxygen and produces API docs, the other produces Sphinx/Breathe
#       output
#-----------------------------------------------------------------------------
option(AIS_BUILD_DOCS "Build the hipFile docs (requires Doxygen, Sphinx, and Breathe)" OFF)

if(AIS_BUILD_DOCS)
    find_package(Doxygen REQUIRED)

    # Set Doxygen input (pasted into Doxyfile.in)
    set(AIS_DOXYFILE_INPUT "${HIPFILE_ROOT_PATH}/include")

    # Set the path to the documentation
    set(AIS_DOC_PATH "${CMAKE_CURRENT_BINARY_DIR}/docs")

    # Set the Doxyfile install location
    set(AIS_DOXYFILE ${AIS_DOC_PATH}/Doxyfile)

    # Create the Doxyfile from the input file
    configure_file("docs/Doxyfile.in" ${AIS_DOXYFILE})

    # Set the output directory
    set(DOXYGEN_OUT ${AIS_DOC_PATH})

    # Configure the documentation build
    add_custom_target("doc"
        COMMAND ${DOXYGEN_EXECUTABLE} ${AIS_DOXYFILE}
        WORKING_DIRECTORY ${AIS_DOC_PATH}
        COMMENT "Generating hipFile API documentation with Doxygen"
        VERBATIM
    )

endif()
