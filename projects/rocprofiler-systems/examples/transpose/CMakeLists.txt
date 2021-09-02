
cmake_minimum_required(VERSION 3.13 FATAL_ERROR)

# uses make to avoid CMAKE_CXX_COMPILER issues
project(transpose
    LANGUAGES CXX)

find_program(MAKE_EXECUTABLE NAMES make gmake)
find_program(HIPCC_EXECUTABLE NAMES hipcc)

if(NOT MAKE_EXECUTABLE)
    message(AUTHOR_WARNING "make/gmake could not be found. Cannot build transpose target")
    return()
endif()

if(NOT HIPCC_EXECUTABLE)
    message(AUTHOR_WARNING "hipcc could not be found. Cannot build transpose target")
    return()
endif()

string(TOUPPER "${CMAKE_BUILD_TYPE}" BUILD_TYPE)
set(transpose_CXX_FLAGS "-W -Wall ${CMAKE_CXX_FLAGS_${BUILD_TYPE}} ${CMAKE_CXX_FLAGS}")
set(transpose_LINK_FLAGS "")

if(CMAKE_CXX_COMPILER_IS_CLANG AND TARGET hosttrace::hosttrace-compile-options)
    set(transpose_CXX_FLAGS "${transpose_CXX_FLAGS} ${hosttrace_CXX_FLAGS}")
    get_target_property(COMPILE_OPTS hosttrace::hosttrace-compile-options INTERFACE_COMPILE_OPTIONS)
    set(transpose_CXX_FLAGS "${transpose_CXX_FLAGS} ${COMPILE_OPTS}")
endif()

option(TRANSPOSE_USE_MPI "Enable MPI support in transpose exe" ${TIMEMORY_USE_MPI})

if(TRANSPOSE_USE_MPI)
    find_package(MPI REQUIRED)
endif()

if(TARGET MPI::MPI_C)
    set(transpose_CXX_FLAGS "${transpose_CXX_FLAGS} -DUSE_MPI")
    get_target_property(COMPILE_OPTS MPI::MPI_C INTERFACE_COMPILE_OPTIONS)
    foreach(_COPT ${COMPILE_OPTS})
        if(NOT "${_COPT}" MATCHES "COMPILE_LANG")
            set(transpose_CXX_FLAGS "${transpose_CXX_FLAGS} ${_COPT}")
        endif()
    endforeach()
    get_target_property(INCLUDE_DIRS MPI::MPI_C INTERFACE_INCLUDE_DIRECTORIES)
    foreach(_IDIR ${INCLUDE_DIRS})
        set(transpose_CXX_FLAGS "${transpose_CXX_FLAGS} -I${_IDIR}")
    endforeach()
    if(MPI_C_LINK_FLAGS)
        set(transpose_LINK_FLAGS "${transpose_LINK_FLAGS} ${MPI_C_LINK_FLAGS}")
    endif()
    set(_LINK_LIBS "")
    foreach(_LIB ${MPI_C_LIB_NAMES})
        string(APPEND _LINK_LIBS "-l${_LIB} ")
    endforeach()
    foreach(_IDIR ${INCLUDE_DIRS} ${MPI_mpich_LIBRARY} ${MPI_mpi_LIBRARY} ${MPI_LIBRARY_DIRS})
        get_filename_component(_LIBDIR "${_IDIR}" DIRECTORY)
        if(EXISTS "${_IDIR}/libmpi${CMAKE_SHARED_LIBRARY_SUFFIX}")
            set(transpose_LINK_FLAGS "${transpose_LINK_FLAGS} -L${_IDIR} ${_LINK_LIBS}")
        endif()
        if(EXISTS "${_LIBDIR}/libmpi${CMAKE_SHARED_LIBRARY_SUFFIX}")
            set(transpose_LINK_FLAGS "${transpose_LINK_FLAGS} -L${_LIBDIR} ${_LINK_LIBS}")
        endif()
        foreach(_LDIR lib lib64)
            set(_LIBDIR_SAVE "${_LIBDIR}")
            if(NOT EXISTS "${_LIBDIR}/${_LDIR}")
                get_filename_component(_LIBDIR "${_LIBDIR}" DIRECTORY)
            endif()
            if(EXISTS "${_LIBDIR}/${_LDIR}")
                set(transpose_LINK_FLAGS "${transpose_LINK_FLAGS} -L${_LIBDIR}/${_LDIR} ${_LINK_LIBS}")
            endif()
            set(_LIBDIR "${_LIBDIR_SAVE}")
        endforeach()
    endforeach()
endif()

# remove generator expressions
string(REPLACE "$<" "" transpose_CXX_FLAGS "${transpose_CXX_FLAGS}")
string(REPLACE ">:" ":" transpose_CXX_FLAGS "${transpose_CXX_FLAGS}")
string(REPLACE ">" "" transpose_CXX_FLAGS "${transpose_CXX_FLAGS}")
string(REPLACE "COMPILE_LANGUAGE:CXX:" "" transpose_CXX_FLAGS "${transpose_CXX_FLAGS}")
# remove double spaces
string(REPLACE "  " " " transpose_CXX_FLAGS "${transpose_CXX_FLAGS}")
# convert to list
string(REPLACE " " ";" transpose_CXX_FLAGS "${transpose_CXX_FLAGS}")
# remove duplicate args
list(REMOVE_DUPLICATES transpose_CXX_FLAGS)
# make single string
string(REPLACE ";" " " transpose_CXX_FLAGS "${transpose_CXX_FLAGS}")

# configure files in build directory
configure_file(${PROJECT_SOURCE_DIR}/transpose.cpp
    ${CMAKE_BINARY_DIR}/transpose/transpose.cpp COPYONLY)

configure_file(${PROJECT_SOURCE_DIR}/Makefile.in
    ${CMAKE_BINARY_DIR}/transpose/Makefile @ONLY)

# touch the file when cmake re-runs to ensure rebuilt
execute_process(COMMAND ${CMAKE_COMMAND} -E touch ${CMAKE_BINARY_DIR}/transpose/transpose.cpp)

# target for adding a dependency to libpytimemory target
add_custom_target(transpose ALL
  COMMAND ${MAKE_EXECUTABLE}
  WORKING_DIRECTORY ${CMAKE_BINARY_DIR}/transpose
  BYPRODUCTS ${CMAKE_BINARY_DIR}/transpose/transpose
  SOURCES ${CMAKE_BINARY_DIR}/transpose/transpose.cpp)
