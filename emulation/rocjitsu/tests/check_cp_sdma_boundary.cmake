# Copyright (c) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

if(NOT DEFINED ROCJITSU_SOURCE_DIR)
    message(FATAL_ERROR "ROCJITSU_SOURCE_DIR is required")
endif()

set(CP_SOURCES
    lib/rocjitsu/src/rocjitsu/vm/amdgpu/command_processor.h
    lib/rocjitsu/src/rocjitsu/vm/amdgpu/command_processor.cpp
    lib/rocjitsu/src/rocjitsu/vm/amdgpu/aql_packet_types.h
    lib/rocjitsu/src/rocjitsu/vm/amdgpu/aql_packet_processor.h
    lib/rocjitsu/src/rocjitsu/vm/amdgpu/aql_packet_processor.cpp
    lib/rocjitsu/src/rocjitsu/vm/amdgpu/aql_queue_binding_factory.h
    lib/rocjitsu/src/rocjitsu/vm/amdgpu/aql_queue_binding_factory.cpp
    lib/rocjitsu/src/rocjitsu/vm/amdgpu/dispatch_entry.h
    lib/rocjitsu/src/rocjitsu/vm/amdgpu/completion_tracker.h
    lib/rocjitsu/src/rocjitsu/vm/amdgpu/completion_tracker.cpp
    lib/rocjitsu/src/rocjitsu/vm/amdgpu/pm4_ring_consumer.h
    lib/rocjitsu/src/rocjitsu/vm/amdgpu/pm4_ring_consumer.cpp
    lib/rocjitsu/src/rocjitsu/vm/amdgpu/pm4_packet_processor.h
    lib/rocjitsu/src/rocjitsu/vm/amdgpu/pm4_packet_processor.cpp
    lib/rocjitsu/src/rocjitsu/vm/amdgpu/pm4_queue_binding_factory.h
    lib/rocjitsu/src/rocjitsu/vm/amdgpu/pm4_queue_binding_factory.cpp
    lib/rocjitsu/src/rocjitsu/vm/amdgpu/pm4_queue_controller.h
    lib/rocjitsu/src/rocjitsu/vm/amdgpu/pm4_queue_controller.cpp
)

foreach(SOURCE_FILE IN LISTS CP_SOURCES)
    file(READ "${ROCJITSU_SOURCE_DIR}/${SOURCE_FILE}" SOURCE_CONTENT)
    string(TOLOWER "${SOURCE_CONTENT}" LOWER_SOURCE_CONTENT)
    string(FIND "${LOWER_SOURCE_CONTENT}" "sdma" SDMA_POSITION)
    if(NOT SDMA_POSITION EQUAL -1)
        message(FATAL_ERROR "${SOURCE_FILE} contains an SDMA dependency")
    endif()
endforeach()

set(PACKET_PROCESSOR_SOURCES
    lib/rocjitsu/src/rocjitsu/vm/amdgpu/packet_processor.h
    lib/rocjitsu/src/rocjitsu/vm/amdgpu/aql_packet_types.h
    lib/rocjitsu/src/rocjitsu/vm/amdgpu/aql_packet_processor.h
    lib/rocjitsu/src/rocjitsu/vm/amdgpu/aql_packet_processor.cpp
    lib/rocjitsu/src/rocjitsu/vm/amdgpu/pm4_packet_processor.h
    lib/rocjitsu/src/rocjitsu/vm/amdgpu/pm4_packet_processor.cpp
    lib/rocjitsu/src/rocjitsu/vm/amdgpu/sdma_packet_processor.h
    lib/rocjitsu/src/rocjitsu/vm/amdgpu/sdma_packet_processor.cpp
)

foreach(SOURCE_FILE IN LISTS PACKET_PROCESSOR_SOURCES)
    file(READ "${ROCJITSU_SOURCE_DIR}/${SOURCE_FILE}" SOURCE_CONTENT)
    string(TOLOWER "${SOURCE_CONTENT}" LOWER_SOURCE_CONTENT)
    foreach(
        FORBIDDEN_TOKEN
        IN
        ITEMS
            "gpu_memory"
            "/pci/"
            "vfio"
            "vfu"
            "mes_engine"
            "gpu_queue_registry"
            "ring_consumer"
            "doorbell"
    )
        string(
            FIND "${LOWER_SOURCE_CONTENT}"
            "${FORBIDDEN_TOKEN}"
            TOKEN_POSITION
        )
        if(NOT TOKEN_POSITION EQUAL -1)
            message(
                FATAL_ERROR
                "${SOURCE_FILE} depends on queue, transport, or backing-store ownership"
            )
        endif()
    endforeach()
endforeach()

foreach(SOURCE_FILE IN LISTS CP_SOURCES)
    file(READ "${ROCJITSU_SOURCE_DIR}/${SOURCE_FILE}" SOURCE_CONTENT)
    foreach(FORBIDDEN_TOKEN IN ITEMS "gpu_memory.h" "GpuMemory")
        string(FIND "${SOURCE_CONTENT}" "${FORBIDDEN_TOKEN}" TOKEN_POSITION)
        if(NOT TOKEN_POSITION EQUAL -1)
            message(
                FATAL_ERROR
                "${SOURCE_FILE} bypasses GpuVm and depends directly on GpuMemory"
            )
        endif()
    endforeach()
endforeach()

set(PCI_FRONTEND_SOURCES
    lib/rocjitsu/src/rocjitsu/vm/amdgpu/pci/gpu_pci_device.h
    lib/rocjitsu/src/rocjitsu/vm/amdgpu/pci/gpu_pci_device.cpp
)

foreach(SOURCE_FILE IN LISTS PCI_FRONTEND_SOURCES)
    file(READ "${ROCJITSU_SOURCE_DIR}/${SOURCE_FILE}" SOURCE_CONTENT)
    foreach(FORBIDDEN_TOKEN IN ITEMS "command_processor.h" "CommandProcessor")
        string(FIND "${SOURCE_CONTENT}" "${FORBIDDEN_TOKEN}" TOKEN_POSITION)
        if(NOT TOKEN_POSITION EQUAL -1)
            message(
                FATAL_ERROR
                "${SOURCE_FILE} reaches directly into the command processor"
            )
        endif()
    endforeach()
endforeach()

set(QUEUE_FRONTEND_SOURCES
    ${PCI_FRONTEND_SOURCES}
    lib/rocjitsu/src/rocjitsu/kmd/linux/simulated_kfd.h
    lib/rocjitsu/src/rocjitsu/kmd/linux/simulated_kfd.cpp
)

foreach(SOURCE_FILE IN LISTS QUEUE_FRONTEND_SOURCES)
    file(READ "${ROCJITSU_SOURCE_DIR}/${SOURCE_FILE}" SOURCE_CONTENT)
    foreach(
        FORBIDDEN_TOKEN
        IN
        ITEMS
            "AqlPacketProcessor"
            "Pm4PacketProcessor"
            "SdmaPacketProcessor"
            "PacketProcessResult"
            "Pm4RingConsumer"
            "SdmaRingConsumer"
            "CircularRingReader"
            "ConsumerCursorJournal"
            "Pm4QueueController"
            "SdmaExecutor"
            "SdmaQueueRunner"
            "SdmaQueueBackend"
            "CommandProcessorQueueBackend"
    )
        string(FIND "${SOURCE_CONTENT}" "${FORBIDDEN_TOKEN}" TOKEN_POSITION)
        if(NOT TOKEN_POSITION EQUAL -1)
            message(
                FATAL_ERROR
                "${SOURCE_FILE} contains packet or ring processing semantics"
            )
        endif()
    endforeach()
endforeach()
