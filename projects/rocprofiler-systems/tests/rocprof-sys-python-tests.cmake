# MIT License
#
# Copyright (c) 2025 Advanced Micro Devices, Inc. All rights reserved.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.

# -------------------------------------------------------------------------------------- #
#
# python tests
#
# -------------------------------------------------------------------------------------- #

set(_INDEX 0)

foreach(_VERSION ${ROCPROFSYS_PYTHON_VERSIONS})
    if(NOT ROCPROFSYS_USE_PYTHON)
        continue()
    endif()

    list(GET ROCPROFSYS_PYTHON_ROOT_DIRS ${_INDEX} _PYTHON_ROOT_DIR)

    rocprofiler_systems_find_python(
        _PYTHON
        ROOT_DIR "${_PYTHON_ROOT_DIR}"
        COMPONENTS Interpreter
    )

    # ---------------------------------------------------------------------------------- #
    # python tests
    # ---------------------------------------------------------------------------------- #
    rocprofiler_systems_add_python_test(
        NAME python-external
        PYTHON_EXECUTABLE ${_PYTHON_EXECUTABLE}
        PYTHON_VERSION ${_VERSION}
        FILE ${CMAKE_SOURCE_DIR}/examples/python/external.py
        PROFILE_ARGS "--label" "file"
        RUN_ARGS -v 10 -n 5
        ENVIRONMENT "${_python_environment}"
    )

    rocprofiler_systems_add_python_test(
        NAME python-external-exclude-inefficient
        PYTHON_EXECUTABLE ${_PYTHON_EXECUTABLE}
        PYTHON_VERSION ${_VERSION}
        FILE ${CMAKE_SOURCE_DIR}/examples/python/external.py
        PROFILE_ARGS -E "^inefficient$"
        RUN_ARGS -v 10 -n 5
        ENVIRONMENT "${_python_environment}"
    )

    rocprofiler_systems_add_python_test(
        NAME python-builtin
        PYTHON_EXECUTABLE ${_PYTHON_EXECUTABLE}
        PYTHON_VERSION ${_VERSION}
        FILE ${CMAKE_SOURCE_DIR}/examples/python/builtin.py
        PROFILE_ARGS "-b" "--label" "file" "line"
        RUN_ARGS -v 10 -n 5
        ENVIRONMENT "${_python_environment}"
    )

    rocprofiler_systems_add_python_test(
        NAME python-builtin-noprofile
        PYTHON_EXECUTABLE ${_PYTHON_EXECUTABLE}
        PYTHON_VERSION ${_VERSION}
        FILE ${CMAKE_SOURCE_DIR}/examples/python/noprofile.py
        PROFILE_ARGS "-b" "--label" "file"
        RUN_ARGS -v 15 -n 5
        ENVIRONMENT "${_python_environment}"
    )

    rocprofiler_systems_add_python_test(
        STANDALONE
        NAME python-source
        PYTHON_EXECUTABLE ${_PYTHON_EXECUTABLE}
        PYTHON_VERSION ${_VERSION}
        FILE ${CMAKE_SOURCE_DIR}/examples/python/source.py
        RUN_ARGS -v 5 -n 5 -s 3
        ENVIRONMENT "${_python_environment}"
    )

    rocprofiler_systems_add_python_test(
        STANDALONE
        NAME python-code-coverage
        PYTHON_EXECUTABLE ${_PYTHON_EXECUTABLE}
        PYTHON_VERSION ${_VERSION}
        FILE ${CMAKE_SOURCE_DIR}/examples/code-coverage/code-coverage.py
        RUN_ARGS
            -i
            ${PROJECT_BINARY_DIR}/rocprof-sys-tests-output/code-coverage-basic-blocks-binary-rewrite/coverage.json
            ${PROJECT_BINARY_DIR}/rocprof-sys-tests-output/code-coverage-basic-blocks-hybrid-runtime-instrument/coverage.json
            -o
            ${PROJECT_BINARY_DIR}/rocprof-sys-tests-output/code-coverage-basic-blocks-summary/coverage.json
        DEPENDS code-coverage-basic-blocks-binary-rewrite
                code-coverage-basic-blocks-binary-rewrite-run
                code-coverage-basic-blocks-hybrid-runtime-instrument
        LABELS "code-coverage"
        ENVIRONMENT "${_python_environment}"
    )

    # ---------------------------------------------------------------------------------- #
    # python output tests
    # ---------------------------------------------------------------------------------- #
    if(CMAKE_VERSION VERSION_LESS "3.18.0")
        find_program(ROCPROFSYS_CAT_EXE NAMES cat PATH_SUFFIXES bin)

        if(ROCPROFSYS_CAT_EXE)
            set(ROCPROFSYS_CAT_COMMAND ${ROCPROFSYS_CAT_EXE})
        endif()
    else()
        set(ROCPROFSYS_CAT_COMMAND ${CMAKE_COMMAND} -E cat)
    endif()

    if(ROCPROFSYS_CAT_COMMAND)
        rocprofiler_systems_add_python_test(
            NAME python-external-check
            COMMAND ${ROCPROFSYS_CAT_COMMAND}
            PYTHON_VERSION ${_VERSION}
            FILE rocprof-sys-tests-output/python-external/${_VERSION}/trip_count.txt
            PASS_REGEX
                "(\\\[compile\\\]).*(\\\| \\\|0>>> \\\[run\\\]\\\[external.py\\\]).*(\\\| \\\|0>>> \\\|_\\\[fib\\\]\\\[external.py\\\]).*(\\\| \\\|0>>> \\\|_\\\[inefficient\\\]\\\[external.py\\\])"
            DEPENDS python-external-${_VERSION}
            ENVIRONMENT "${_python_environment}"
        )

        rocprofiler_systems_add_python_test(
            NAME python-external-exclude-inefficient-check
            COMMAND ${ROCPROFSYS_CAT_COMMAND}
            PYTHON_VERSION ${_VERSION}
            FILE rocprof-sys-tests-output/python-external-exclude-inefficient/${_VERSION}/trip_count.txt
            FAIL_REGEX "(\\\|_inefficient).*(\\\|_sum)|ROCPROFSYS_ABORT_FAIL_REGEX"
            DEPENDS python-external-exclude-inefficient-${_VERSION}
            ENVIRONMENT "${_python_environment}"
        )

        rocprofiler_systems_add_python_test(
            NAME python-builtin-check
            COMMAND ${ROCPROFSYS_CAT_COMMAND}
            PYTHON_VERSION ${_VERSION}
            FILE rocprof-sys-tests-output/python-builtin/${_VERSION}/trip_count.txt
            PASS_REGEX "\\\[inefficient\\\]\\\[builtin.py:14\\\]"
            DEPENDS python-builtin-${_VERSION}
            ENVIRONMENT "${_python_environment}"
        )

        rocprofiler_systems_add_python_test(
            NAME python-builtin-noprofile-check
            COMMAND ${ROCPROFSYS_CAT_COMMAND}
            PYTHON_VERSION ${_VERSION}
            FILE rocprof-sys-tests-output/python-builtin-noprofile/${_VERSION}/trip_count.txt
            PASS_REGEX ".(run)..(noprofile.py)."
            FAIL_REGEX ".(fib|inefficient)..(noprofile.py).|ROCPROFSYS_ABORT_FAIL_REGEX"
            DEPENDS python-builtin-noprofile-${_VERSION}
            ENVIRONMENT "${_python_environment}"
        )
    else()
        rocprofiler_systems_message(
            WARNING
            "Neither 'cat' nor 'cmake -E cat' are available. Python source checks are disabled"
        )
    endif()

    function(ROCPROFILER_SYSTEMS_ADD_PYTHON_VALIDATION_TEST)
        cmake_parse_arguments(
            TEST
            ""
            "NAME;TIMEMORY_METRIC;TIMEMORY_FILE;PERFETTO_FILE"
            "ARGS;PERFETTO_METRIC"
            ${ARGN}
        )

        rocprofiler_systems_add_python_test(
            NAME ${TEST_NAME}-validate-timemory
            COMMAND
                ${_PYTHON_EXECUTABLE} ${CMAKE_CURRENT_LIST_DIR}/validate-timemory-json.py
                -m ${TEST_TIMEMORY_METRIC} ${TEST_ARGS} -i
            PYTHON_VERSION ${_VERSION}
            FILE rocprof-sys-tests-output/${TEST_NAME}/${_VERSION}/${TEST_TIMEMORY_FILE}
            DEPENDS ${TEST_NAME}-${_VERSION}
            PASS_REGEX
                "rocprof-sys-tests-output/${TEST_NAME}/${_VERSION}/${TEST_TIMEMORY_FILE} validated"
            ENVIRONMENT "${_python_environment}"
        )

        rocprofiler_systems_add_python_test(
            NAME ${TEST_NAME}-validate-perfetto
            COMMAND
                ${_PYTHON_EXECUTABLE} ${CMAKE_CURRENT_LIST_DIR}/validate-perfetto-proto.py
                -m ${TEST_PERFETTO_METRIC} ${TEST_ARGS} -p -t
                /opt/trace_processor/bin/trace_processor_shell -i
            PYTHON_VERSION ${_VERSION}
            FILE rocprof-sys-tests-output/${TEST_NAME}/${_VERSION}/${TEST_PERFETTO_FILE}
            DEPENDS ${TEST_NAME}-${_VERSION}
            PASS_REGEX
                "rocprof-sys-tests-output/${TEST_NAME}/${_VERSION}/${TEST_PERFETTO_FILE} validated"
            ENVIRONMENT "${_python_environment}"
        )
    endfunction()

    set(python_source_labels
        main_loop
        run
        fib
        fib
        fib
        fib
        fib
        inefficient
        _sum
    )
    set(python_source_count
        5
        3
        3
        6
        12
        18
        6
        3
        3
    )
    set(python_source_depth
        0
        1
        2
        3
        4
        5
        6
        2
        3
    )

    set(python_source_categories python user)

    rocprofiler_systems_add_python_validation_test(
        NAME python-source
        TIMEMORY_METRIC "trip_count"
        TIMEMORY_FILE "trip_count.json"
        PERFETTO_FILE "perfetto-trace.proto"
        PERFETTO_METRIC ${python_source_categories}
        ARGS -l ${python_source_labels} -c ${python_source_count} -d
             ${python_source_depth}
    )

    set(python_builtin_labels
        [run][builtin.py:28]
        [fib][builtin.py:10]
        [fib][builtin.py:10]
        [fib][builtin.py:10]
        [fib][builtin.py:10]
        [fib][builtin.py:10]
        [fib][builtin.py:10]
        [fib][builtin.py:10]
        [fib][builtin.py:10]
        [fib][builtin.py:10]
        [fib][builtin.py:10]
        [inefficient][builtin.py:14]
    )
    set(python_builtin_count
        5
        5
        10
        20
        40
        80
        160
        260
        220
        80
        10
        5
    )
    set(python_builtin_depth
        0
        1
        2
        3
        4
        5
        6
        7
        8
        9
        10
        1
    )

    rocprofiler_systems_add_python_validation_test(
        NAME python-builtin
        TIMEMORY_METRIC "trip_count"
        TIMEMORY_FILE "trip_count.json"
        PERFETTO_METRIC "python"
        PERFETTO_FILE "perfetto-trace.proto"
        ARGS -l ${python_builtin_labels} -c ${python_builtin_count} -d
             ${python_builtin_depth}
    )
    math(EXPR _INDEX "${_INDEX} + 1")
endforeach()
