#
# ATT decoder wrapper library for use by the rocprofv3 tool
#
find_library(COMGR REQUIRED NAMES amd_comgr HINTS /opt/rocm/lib)

set(ATT_TOOL_SOURCE_FILES
    waitcnt/analysis.cpp
    waitcnt/gfx9.cpp
    waitcnt/gfx10.cpp
    waitcnt/gfx12.cpp
    att_lib_wrapper.cpp
    wave.cpp
    code.cpp
    filenames.cpp
    occupancy.cpp
    perfcounter.cpp
    profile_interface.cpp
    standalone_tool_main.cpp)

set(DECODER_SRC ${CMAKE_CURRENT_SOURCE_DIR}/../../source/)

set(INCLUDE_DIRS ${CMAKE_CURRENT_SOURCE_DIR}/../../include/ ${DECODER_SRC} ${CMAKE_CURRENT_SOURCE_DIR} ${json_SOURCE_DIR}/include /opt/rocm/include)
set(LINK_LIBS ${COMGR} pthread)

# Sources re-globbed only for sanitizer builds
file(GLOB ATT_DECODER_V3_FILES
  ${DECODER_SRC}/*.cpp
  ${DECODER_SRC}/trie.cpp
  ${DECODER_SRC}/gfx9/*.cpp
  ${DECODER_SRC}/gfx10/*.cpp
  ${DECODER_SRC}/gfx11/*.cpp
  ${DECODER_SRC}/gfx12/*.cpp
  ${DECODER_SRC}/stitch/stitch.cpp
  ${DECODER_SRC}/stitch/pctranslator.cpp
)

# Regular build: link against static library
add_executable(tool)
target_sources(tool PRIVATE ${ATT_TOOL_SOURCE_FILES})
target_include_directories(tool PRIVATE ${INCLUDE_DIRS})
target_link_libraries(tool PRIVATE rocprof-trace-decoder-static ${LINK_LIBS})
target_compile_options(tool PUBLIC -g -O3)

# Sanitizer build: recompile library sources with sanitizer flags
add_executable(sanitized)
target_sources(sanitized PRIVATE ${ATT_TOOL_SOURCE_FILES} ${ATT_DECODER_V3_FILES})
target_include_directories(sanitized PRIVATE ${INCLUDE_DIRS})
target_link_libraries(sanitized PRIVATE ${LINK_LIBS})

target_compile_options(sanitized PUBLIC -fsanitize=address,undefined -g)
target_link_options(sanitized PUBLIC -fsanitize=address,undefined)
