#
# code object
#
if(WIN32)
    # Identifying a loaded code object means parsing ELF, which has no Windows counterpart
    # here; windows_stubs.cpp supplies the stand-in name table.
    set(ROCPROFILER_LIB_CODE_OBJECT_SOURCES windows_stubs.cpp)
else()
    set(ROCPROFILER_LIB_CODE_OBJECT_SOURCES code_object.cpp)
endif()
set(ROCPROFILER_LIB_CODE_OBJECT_HEADERS code_object.hpp)

target_sources(
    rocprofiler-sdk-object-library PRIVATE ${ROCPROFILER_LIB_CODE_OBJECT_SOURCES}
                                           ${ROCPROFILER_LIB_CODE_OBJECT_HEADERS})

if(NOT WIN32)
    # Both hold the parsed representation of a loaded code object, which nothing produces
    # on Windows.
    add_subdirectory(hsa)
    add_subdirectory(hip)
endif()
