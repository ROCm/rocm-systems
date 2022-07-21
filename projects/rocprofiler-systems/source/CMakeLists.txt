if(NOT DEFINED CMAKE_INTERPROCEDURAL_OPTIMIZATION)
    set(CMAKE_INTERPROCEDURAL_OPTIMIZATION OFF)
endif()

if(OMNITRACE_BUILD_LTO)
    set(CMAKE_INTERPROCEDURAL_OPTIMIZATION ON)
endif()

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${PROJECT_BINARY_DIR}/${CMAKE_INSTALL_BINDIR})
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${PROJECT_BINARY_DIR}/${CMAKE_INSTALL_LIBDIR})
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${PROJECT_BINARY_DIR}/${CMAKE_INSTALL_LIBDIR})

add_subdirectory(lib)
add_subdirectory(bin)

if(OMNITRACE_USE_PYTHON)
    add_subdirectory(python)
endif()

if(OMNITRACE_BUILD_DEVELOPER)
    add_custom_target(omnitrace-precommit)
    foreach(_TARGET format-omnitrace-source format-omnitrace-cmake format-timemory-source
                    format-timemory-cmake)
        if(TARGET ${_TARGET})
            add_dependencies(omnitrace-precommit ${_TARGET})
        endif()
    endforeach()
endif()
