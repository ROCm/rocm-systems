#
function(ROCPROFILER_SYSTEMS_FIND_PYTHON _VAR)
    set(options REQUIRED QUIET)
    set(args VERSION ROOT_DIR)
    set(kwargs COMPONENTS)
    cmake_parse_arguments(ARG "${options}" "${args}" "${kwargs}" ${ARGN})

    if(ARG_QUIET)
        set(_QUIET "QUIET")
    endif()

    if(ARG_VERSION)
        set(_EXACT "EXACT")
    endif()

    if(ARG_REQUIRED)
        set(_FIND_REQUIREMENT "REQUIRED")
    endif()

    if(NOT ARG_COMPONENTS)
        set(ARG_COMPONENTS Interpreter Development)
    endif()

    set(Python3_ROOT_DIR "${ARG_ROOT_DIR}")
    set(Python3_FIND_STRATEGY "LOCATION")
    set(Python3_FIND_VIRTUALENV "FIRST")
    set(Python3_ARTIFACTS_INTERACTIVE OFF)

    find_package(
        Python3
        ${ARG_VERSION}
        ${_EXACT}
        ${_QUIET}
        MODULE
        ${_FIND_REQUIREMENT}
        COMPONENTS ${ARG_COMPONENTS}
    )

    set(${_VAR}_FOUND "${Python3_FOUND}" PARENT_SCOPE)
    if(NOT Python3_FOUND)
        set(${_VAR}_EXECUTABLE "" PARENT_SCOPE)
        set(${_VAR}_ROOT_DIR "" PARENT_SCOPE)
        set(${_VAR}_VERSION "" PARENT_SCOPE)
        return()
    else()
        set(${_VAR}_EXECUTABLE "${Python3_EXECUTABLE}" PARENT_SCOPE)
        execute_process(
            COMMAND
                "${Python3_EXECUTABLE}" "-c"
                "import sys; print('.'.join(str(v) for v in [sys.version_info[0], sys.version_info[1]])); print(sys.prefix);"
            RESULT_VARIABLE _PYTHON_SUCCESS
            OUTPUT_VARIABLE _PYTHON_VALUES
            ERROR_VARIABLE _PYTHON_ERROR_VALUE
        )

        if(_PYTHON_SUCCESS MATCHES 0)
            # Convert the process output into a list
            string(REGEX REPLACE ";" "\\\\;" _PYTHON_VALUES ${_PYTHON_VALUES})
            string(REGEX REPLACE "\n" ";" _PYTHON_VALUES ${_PYTHON_VALUES})
            list(GET _PYTHON_VALUES 0 _PYTHON_VERSION_LIST)
            list(GET _PYTHON_VALUES 1 _PYTHON_PREFIX)
            set(${_VAR}_ROOT_DIR "${_PYTHON_PREFIX}" PARENT_SCOPE)
            set(${_VAR}_VERSION "${_PYTHON_VERSION_LIST}" PARENT_SCOPE)
        else()
            rocprofiler_systems_message(WARNING "${_PYTHON_ERROR_VALUE}")
        endif()
    endif()
endfunction()
#
# Internal: find the appropriate link time optimization flags for this compiler
function(_ROCPROFSYS_PYBIND11_ADD_LTO_FLAGS target_name prefer_thin_lto)
    # Checks whether the given CXX/linker flags can compile and link a cxx file.  cxxflags
    # and linkerflags are lists of flags to use.  The result variable is a unique variable
    # name for each set of flags: the compilation result will be cached base on the result
    # variable.  If the flags work, sets them in cxxflags_out/linkerflags_out internal
    # cache variables (in addition to ${result}).
    function(
        _PYBIND11_RETURN_IF_CXX_AND_LINKER_FLAGS_WORK
        result
        cxxflags
        linkerflags
        cxxflags_out
        linkerflags_out
    )
        include(CheckCXXCompilerFlag)
        set(CMAKE_REQUIRED_LIBRARIES ${linkerflags})
        check_cxx_compiler_flag("${cxxflags}" ${result})
        if(${result})
            set(${cxxflags_out} "${cxxflags}" CACHE INTERNAL "" FORCE)
            set(${linkerflags_out} "${linkerflags}" CACHE INTERNAL "" FORCE)
        endif()
    endfunction()

    if(NOT DEFINED PYBIND11_LTO_CXX_FLAGS)
        set(PYBIND11_LTO_CXX_FLAGS "" CACHE INTERNAL "")
        set(PYBIND11_LTO_LINKER_FLAGS "" CACHE INTERNAL "")

        if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
            set(cxx_append "")
            set(linker_append "")
            if(CMAKE_CXX_COMPILER_ID MATCHES "Clang" AND NOT APPLE)
                # Clang Gold plugin does not support -Os; append -O3 to MinSizeRel builds
                # to override it
                set(linker_append ";$<$<CONFIG:MinSizeRel>:-O3>")
            elseif(CMAKE_CXX_COMPILER_ID MATCHES "GNU")
                set(cxx_append ";-fno-fat-lto-objects")
            endif()

            if(CMAKE_CXX_COMPILER_ID MATCHES "Clang" AND prefer_thin_lto)
                _pybind11_return_if_cxx_and_linker_flags_work(
                    HAS_FLTO_THIN "-flto=thin${cxx_append}" "-flto=thin${linker_append}"
                    PYBIND11_LTO_CXX_FLAGS PYBIND11_LTO_LINKER_FLAGS
                )
            endif()

            if(NOT HAS_FLTO_THIN)
                _pybind11_return_if_cxx_and_linker_flags_work(
                    HAS_FLTO "-flto${cxx_append}" "-flto${linker_append}"
                    PYBIND11_LTO_CXX_FLAGS PYBIND11_LTO_LINKER_FLAGS
                )
            endif()
        elseif(CMAKE_CXX_COMPILER_ID MATCHES "Intel")
            # Intel equivalent to LTO is called IPO
            _pybind11_return_if_cxx_and_linker_flags_work(
                HAS_INTEL_IPO "-ipo" "-ipo" PYBIND11_LTO_CXX_FLAGS
                PYBIND11_LTO_LINKER_FLAGS
            )
        elseif(MSVC)
            # cmake only interprets libraries as linker flags when they start with a -
            # (otherwise it converts /LTCG to \LTCG as if it was a Windows path).  Luckily
            # MSVC supports passing flags with - instead of /, even if it is a bit
            # non-standard:
            _pybind11_return_if_cxx_and_linker_flags_work(
                HAS_MSVC_GL_LTCG "/GL" "-LTCG" PYBIND11_LTO_CXX_FLAGS
                PYBIND11_LTO_LINKER_FLAGS
            )
        endif()

        if(PYBIND11_LTO_CXX_FLAGS)
            rocprofiler_systems_message(STATUS "${target_name} :: LTO enabled")
        else()
            rocprofiler_systems_message(
                STATUS
                "${target_name} :: LTO disabled (not supported by the compiler and/or linker)"
            )
        endif()
    endif()

    # Enable LTO flags if found, except for Debug builds
    if(PYBIND11_LTO_CXX_FLAGS)
        target_compile_options(
            ${target_name}
            PRIVATE "$<$<NOT:$<CONFIG:Debug>>:${PYBIND11_LTO_CXX_FLAGS}>"
        )
    endif()
    if(PYBIND11_LTO_LINKER_FLAGS)
        target_link_libraries(
            ${target_name}
            PRIVATE "$<$<NOT:$<CONFIG:Debug>>:${PYBIND11_LTO_LINKER_FLAGS}>"
        )
    endif()
endfunction()
#
function(ROCPROFILER_SYSTEMS_PYBIND11_ADD_MODULE target_name)
    set(options
        MODULE
        SHARED
        EXCLUDE_FROM_ALL
        NO_EXTRAS
        SYSTEM
        THIN_LTO
        LTO
    )
    set(args PYTHON_VERSION VISIBILITY CXX_STANDARD)
    set(kwargs)
    cmake_parse_arguments(ARG "${options}" "${args}" "${kwargs}" ${ARGN})

    if(ARG_MODULE AND ARG_SHARED)
        rocprofiler_systems_message(FATAL_ERROR "Can't be both MODULE and SHARED")
    elseif(ARG_SHARED)
        set(lib_type SHARED)
    else()
        set(lib_type MODULE)
    endif()
    if(NOT ARG_VISIBILITY)
        set(ARG_VISIBILITY "hidden")
    endif()
    if(ARG_PYTHON_VERSION)
        set(PythonLibsNew_FIND_REQUIRED ON)
    endif()
    if(NOT ARG_CXX_STANDARD AND CMAKE_CXX_STANDARD)
        set(ARG_CXX_STANDARD ${CMAKE_CXX_STANDARD})
    elseif(NOT ARG_CXX_STANDARD)
        set(ARG_CXX_STANDARD 11)
    endif()
    if(ARG_EXCLUDE_FROM_ALL)
        set(exclude_from_all EXCLUDE_FROM_ALL)
    endif()

    add_library(${target_name} ${lib_type} ${exclude_from_all} ${ARG_UNPARSED_ARGUMENTS})

    target_link_libraries(${target_name} PRIVATE pybind11::module)

    if(ARG_SYSTEM)
        set(inc_isystem SYSTEM)
    endif()

    list(INSERT CMAKE_MODULE_PATH 0 "${PROJECT_SOURCE_DIR}/source/python/cmake/Modules")
    find_package(PyBind11Python ${ARG_PYTHON_VERSION})

    target_include_directories(
        ${target_name}
        ${inc_isystem}
        PRIVATE
            ${PYBIND11_INCLUDE_DIR} # from project CMakeLists.txt
        PRIVATE
            ${pybind11_INCLUDE_DIR} # from pybind11Config
        PRIVATE ${Python3_INCLUDE_DIRS}
    )

    # Python debug libraries expose slightly different objects
    # https://docs.python.org/3.6/c-api/intro.html#debugging-builds
    # https://stackoverflow.com/questions/39161202/how-to-work-around-missing-pymodule-create2-in-amd64-win-python35-d-lib
    if(PYTHON_IS_DEBUG)
        target_compile_definitions(${target_name} PRIVATE Py_DEBUG)
    endif()

    # The prefix and extension are provided by FindPythonLibsNew.cmake
    set_target_properties(${target_name} PROPERTIES PREFIX "${PYTHON_MODULE_PREFIX}")
    set_target_properties(${target_name} PROPERTIES SUFFIX "${PYTHON_MODULE_EXTENSION}")

    # -fvisibility=hidden is required to allow multiple modules compiled against different
    # pybind versions to work properly, and for some features (e.g. py::module_local).  We
    # force it on everything inside the `pybind11` namespace; also turning it on for a
    # pybind module compilation here avoids potential warnings or issues from having mixed
    # hidden/non-hidden types.
    set_target_properties(
        ${target_name}
        PROPERTIES CXX_VISIBILITY_PRESET "${ARG_VISIBILITY}"
    )
    set_target_properties(
        ${target_name}
        PROPERTIES CUDA_VISIBILITY_PRESET "${ARG_VISIBILITY}"
    )

    if(WIN32 OR CYGWIN)
        # Link against the Python shared library on Windows
        target_link_libraries(${target_name} PRIVATE ${Python3_LIBRARIES})
    elseif(APPLE)
        # It's quite common to have multiple copies of the same Python version installed
        # on one's system. E.g.: one copy from the OS and another copy that's statically
        # linked into an application like Blender or Maya. If we link our plugin library
        # against the OS Python here and import it into Blender or Maya later on, this
        # will cause segfaults when multiple conflicting Python instances are active at
        # the same time (even when they are of the same version).

        # Windows is not affected by this issue since it handles DLL imports differently.
        # The solution for Linux and Mac OS is simple: we just don't link against the
        # Python library. The resulting shared library will have missing symbols, but
        # that's perfectly fine -- they will be resolved at import time.

        target_link_libraries(${target_name} PRIVATE "-undefined dynamic_lookup")

        if(ARG_SHARED)
            # Suppress CMake >= 3.0 warning for shared libraries
            set_target_properties(${target_name} PROPERTIES MACOSX_RPATH ON)
        endif()
    endif()

    # Make sure C++11/14 are enabled
    set_target_properties(
        ${target_name}
        PROPERTIES CXX_STANDARD ${ARG_CXX_STANDARD} CXX_STANDARD_REQUIRED ON
    )

    if(ARG_NO_EXTRAS)
        return()
    endif()

    if(ARG_LTO OR ARG_THIN_LTO)
        _rocprofiler_systems_pybind11_add_lto_flags(${target_name} ${ARG_THIN_LTO})
    endif()

    if(NOT MSVC AND NOT ${CMAKE_BUILD_TYPE} MATCHES Debug|RelWithDebInfo)
        # Strip unnecessary sections of the binary on Linux/Mac OS
        if(CMAKE_STRIP)
            if(APPLE)
                add_custom_command(
                    TARGET ${target_name}
                    POST_BUILD
                    COMMAND ${CMAKE_STRIP} -x $<TARGET_FILE:${target_name}>
                )
            else()
                add_custom_command(
                    TARGET ${target_name}
                    POST_BUILD
                    COMMAND ${CMAKE_STRIP} $<TARGET_FILE:${target_name}>
                )
            endif()
        endif()
    endif()

    if(MSVC)
        # /MP enables multithreaded builds (relevant when there are many files), /bigobj
        # is needed for bigger binding projects due to the limit to 64k addressable
        # sections
        target_compile_options(${target_name} PRIVATE /bigobj)
        if(CMAKE_VERSION VERSION_LESS 3.11)
            target_compile_options(${target_name} PRIVATE $<$<NOT:$<CONFIG:Debug>>:/MP>)
        else()
            # Only set these options for C++ files.  This is important so that, for
            # instance, projects that include other types of source files like CUDA .cu
            # files don't get these options propagated to nvcc since that would cause the
            # build to fail.
            target_compile_options(
                ${target_name}
                PRIVATE $<$<NOT:$<CONFIG:Debug>>:$<$<COMPILE_LANGUAGE:CXX>:/MP>>
            )
        endif()
    endif()
endfunction()
