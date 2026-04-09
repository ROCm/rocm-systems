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
    find_package(Python3 REQUIRED COMPONENTS Interpreter)

    # Verify Sphinx + rocm_docs are available
    execute_process(
        COMMAND "${Python3_EXECUTABLE}" -c "import rocm_docs"
        RESULT_VARIABLE _rocm_docs_found
        OUTPUT_QUIET ERROR_QUIET
    )
    if(NOT _rocm_docs_found EQUAL 0)
        message(FATAL_ERROR "rocm_docs Python package not found. Install it with: pip install rocm-docs-core")
    endif()

    # Set Doxygen input (pasted into Doxyfile.in)
    set(AIS_DOXYFILE_INPUT "${HIPFILE_ROOT_PATH}/include")

    # Set the path to the documentation
    set(AIS_DOC_PATH "${CMAKE_CURRENT_BINARY_DIR}/docs")

    ###########
    # Doxygen #
    ###########

    # The Doxygen HTML, XML, etc. goes in docs/doxygen/

    # Set the Doxyfile install location
    set(AIS_DOXYFILE "${AIS_DOC_PATH}/doxygen/Doxyfile")

    # Doxygen XML output dir (consumed by Breathe inside Sphinx)
    set(AIS_DOXYGEN_XML_DIR "${AIS_DOC_PATH}/doxygen/xml")

    # Create the Doxyfile from the input file
    configure_file("docs/doxygen/Doxyfile.in" ${AIS_DOXYFILE})

    ##########
    # Sphinx #
    ##########

    # The Sphinx HTML, XML, etc. goes in docs/sphinx/

    # Set the target for the config file and toc file
    set(AIS_SPHINX_CONF_FILE "${AIS_DOC_PATH}/sphinx/conf.py")
    set(AIS_SPHINX_TOC_FILE "${AIS_DOC_PATH}/sphinx/_toc.yml.in")

    # Sphinx HTML output dir
    set(AIS_SPHINX_BUILD_DIR "${AIS_DOC_PATH}/sphinx/html")

    # Copy conf.py and the table of contents file
    # Need to do this so the toc file doesn't dirty the repo
    # _toc.yml.in is transformed by rocm_docs, not CMake
    configure_file("docs/sphinx/conf.py" ${AIS_SPHINX_CONF_FILE} COPYONLY)
    configure_file("docs/sphinx/_toc.yml.in" ${AIS_SPHINX_TOC_FILE} COPYONLY)

    # Build docs: Doxygen first, then Sphinx (which pulls in rocm_docs via conf.py)
    add_custom_target(doc
        # Step 1: Run Doxygen to produce XML for Breathe
        COMMAND "${DOXYGEN_EXECUTABLE}" "${AIS_DOXYFILE}"
        # Step 2: Run Sphinx (html), injecting the Doxygen paths via environment
        COMMAND
            ${CMAKE_COMMAND} -E env
                "DOXYFILE_PATH=${AIS_DOXYFILE}"
                "DOXYGEN_ROOT=${AIS_DOC_PATH}/doxygen"
                "DOXYGEN_XML_DIR=${AIS_DOXYGEN_XML_DIR}"
            "${Python3_EXECUTABLE}" -m sphinx
                -b html
                -c "${AIS_DOC_PATH}/sphinx"
                "${HIPFILE_ROOT_PATH}/docs"
                "${AIS_SPHINX_BUILD_DIR}"
                -v
        # Step 3: Run Sphinx (LaTeX), injecting the Doxygen paths via environment
        COMMAND
            ${CMAKE_COMMAND} -E env
                "DOXYFILE_PATH=${AIS_DOXYFILE}"
                "DOXYGEN_ROOT=${AIS_DOC_PATH}/doxygen"
                "DOXYGEN_XML_DIR=${AIS_DOXYGEN_XML_DIR}"
            "${Python3_EXECUTABLE}" -m sphinx
                -b latex
                -c "${AIS_DOC_PATH}/sphinx"
                "${HIPFILE_ROOT_PATH}/docs"
                "${AIS_SPHINX_BUILD_DIR}"
                -v
        WORKING_DIRECTORY "${AIS_DOC_PATH}"
        COMMENT "Generating hipFile API documentation with Doxygen + Sphinx (rocm_docs)"
        VERBATIM
    )

endif()
