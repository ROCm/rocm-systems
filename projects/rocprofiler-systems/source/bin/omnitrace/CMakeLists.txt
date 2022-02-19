# ------------------------------------------------------------------------------#
#
# omnitrace-exe target
#
# ------------------------------------------------------------------------------#

add_executable(
    omnitrace-exe
    ${_EXCLUDE} ${CMAKE_CURRENT_LIST_DIR}/omnitrace.cpp
    ${CMAKE_CURRENT_LIST_DIR}/omnitrace.hpp ${CMAKE_CURRENT_LIST_DIR}/details.cpp)

target_link_libraries(
    omnitrace-exe
    PRIVATE omnitrace::omnitrace-headers
            omnitrace::omnitrace-dyninst
            omnitrace::omnitrace-compile-options
            $<BUILD_INTERFACE:timemory::timemory-headers>
            $<IF:$<BOOL:${OMNITRACE_USE_SANITIZER}>,omnitrace::omnitrace-sanitizer,>)

set_target_properties(
    omnitrace-exe
    PROPERTIES
        OUTPUT_NAME omnitrace
        INSTALL_RPATH_USE_LINK_PATH ON
        INSTALL_RPATH
        "\$ORIGIN/../${CMAKE_INSTALL_LIBDIR}:\$ORIGIN/../${CMAKE_INSTALL_LIBDIR}/timemory/libunwind:\$ORIGIN/../${CMAKE_INSTALL_LIBDIR}/dyninst-tpls/lib"
    )

if(CMAKE_BUILD_TYPE MATCHES "^(DEBUG|Debug)")
    string(REPLACE " " ";" _FLAGS "${CMAKE_CXX_FLAGS_RELWITHDEBINFO}")
    target_compile_options(omnitrace-exe PRIVATE ${_FLAGS})
endif()

install(
    TARGETS omnitrace-exe
    DESTINATION ${CMAKE_INSTALL_BINDIR}
    OPTIONAL)
