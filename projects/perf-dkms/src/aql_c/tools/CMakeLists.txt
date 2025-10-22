# Userspace Tools for Packet Generation and Decoding

# Common include directories
set(AQL_INCLUDE_DIRS
    ${CMAKE_CURRENT_SOURCE_DIR}/..
    ${CMAKE_CURRENT_SOURCE_DIR}/../..
)

# Common compile options for tools
set(TOOLS_COMPILE_OPTIONS -Wall -Wextra -O2)

#================================================================================
# Packet Generation Tool
#================================================================================

set(PACKET_GEN_SOURCES
    packet_gen_tool.c
    ../packet_generation.c
    ../arch_creator.c
    ../gfx12_creator.c
    ../pm4_packets.c
    ../counter_registry.c
    ../gfx12_events.c
)

add_executable(packet_gen_tool ${PACKET_GEN_SOURCES})

target_include_directories(packet_gen_tool PRIVATE ${AQL_INCLUDE_DIRS})
target_compile_options(packet_gen_tool PRIVATE ${TOOLS_COMPILE_OPTIONS})
target_link_libraries(packet_gen_tool PRIVATE m)

#================================================================================
# PM4 Decoder Tool
#================================================================================

set(PM4_DECODER_SOURCES
    pm4_decoder.c
)

add_executable(pm4_decoder ${PM4_DECODER_SOURCES})

target_include_directories(pm4_decoder PRIVATE ${AQL_INCLUDE_DIRS})
target_compile_options(pm4_decoder PRIVATE ${TOOLS_COMPILE_OPTIONS})

#================================================================================
# Installation
#================================================================================

install(TARGETS packet_gen_tool pm4_decoder
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
)

#================================================================================
# Help target
#================================================================================

add_custom_target(tools-help
    COMMAND ${CMAKE_COMMAND} -E echo "Available tools:"
    COMMAND ${CMAKE_COMMAND} -E echo "  packet_gen_tool - Packet generation tool"
    COMMAND ${CMAKE_COMMAND} -E echo "  pm4_decoder     - PM4 packet decoder"
    COMMAND ${CMAKE_COMMAND} -E echo ""
    COMMAND ${CMAKE_COMMAND} -E echo "Usage examples:"
    COMMAND ${CMAKE_COMMAND} -E echo "  ./packet_gen_tool gfx12 0x1000000 SQ:0:0x42 SQ:1:0x43"
    COMMAND ${CMAKE_COMMAND} -E echo "  echo 'c001790000000200e0000000' | ./pm4_decoder"
    COMMENT "Displaying tools help"
)
