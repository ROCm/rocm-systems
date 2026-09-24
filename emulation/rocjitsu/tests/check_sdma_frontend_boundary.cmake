# Copyright (c) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

if(NOT DEFINED ROCJITSU_SOURCE_DIR)
    message(FATAL_ERROR "ROCJITSU_SOURCE_DIR is required")
endif()

set(SDMA_CORE_SOURCES
    lib/rocjitsu/src/rocjitsu/vm/amdgpu/sdma_queue_scheduler.h
    lib/rocjitsu/src/rocjitsu/vm/amdgpu/sdma_queue_scheduler.cpp
    lib/rocjitsu/src/rocjitsu/vm/amdgpu/sdma_packet_processor.h
    lib/rocjitsu/src/rocjitsu/vm/amdgpu/sdma_packet_processor.cpp
    lib/rocjitsu/src/rocjitsu/vm/amdgpu/sdma_ring_consumer.h
    lib/rocjitsu/src/rocjitsu/vm/amdgpu/sdma_ring_consumer.cpp
    lib/rocjitsu/src/rocjitsu/vm/amdgpu/sdma_queue_binding_factory.h
    lib/rocjitsu/src/rocjitsu/vm/amdgpu/sdma_queue_binding_factory.cpp
)

foreach(SOURCE_FILE IN LISTS SDMA_CORE_SOURCES)
    file(READ "${ROCJITSU_SOURCE_DIR}/${SOURCE_FILE}" SOURCE_CONTENT)
    foreach(
        FORBIDDEN_TOKEN
        IN
        ITEMS
            "rocjitsu/vm/amdgpu/pci/"
            "GpuPciDevice"
            "SdmaBlockModel"
            "PciTransportSession"
            "command_processor.h"
            "CommandProcessor"
            "aql_packet_processor"
            "AqlPacketProcessor"
            "aql_queue_binding_factory"
            "AqlQueueBindingFactory"
            "dispatch_entry.h"
            "DispatchEntry"
            "pm4_"
            "Pm4"
    )
        string(FIND "${SOURCE_CONTENT}" "${FORBIDDEN_TOKEN}" TOKEN_POSITION)
        if(NOT TOKEN_POSITION EQUAL -1)
            message(
                FATAL_ERROR
                "${SOURCE_FILE} depends on the PCI/MMIO frontend"
            )
        endif()
    endforeach()
endforeach()

foreach(
    SOURCE_FILE
    IN
    ITEMS
        lib/rocjitsu/src/rocjitsu/vm/amdgpu/sdma_queue_scheduler.h
        lib/rocjitsu/src/rocjitsu/vm/amdgpu/sdma_queue_scheduler.cpp
)
    file(READ "${ROCJITSU_SOURCE_DIR}/${SOURCE_FILE}" SOURCE_CONTENT)
    foreach(
        FORBIDDEN_TOKEN
        IN
        ITEMS "QueueRegistrationRequest" "QueueType" "QueuePacketFormat"
    )
        string(FIND "${SOURCE_CONTENT}" "${FORBIDDEN_TOKEN}" TOKEN_POSITION)
        if(NOT TOKEN_POSITION EQUAL -1)
            message(
                FATAL_ERROR
                "${SOURCE_FILE} consumes generic frontend queue registration state"
            )
        endif()
    endforeach()
endforeach()

set(SDMA_FRONTEND_SOURCES
    lib/rocjitsu/src/rocjitsu/vm/amdgpu/pci/gpu_pci_device.h
    lib/rocjitsu/src/rocjitsu/vm/amdgpu/pci/gpu_pci_device.cpp
    lib/rocjitsu/src/rocjitsu/vm/amdgpu/pci/sdma_block_model.h
    lib/rocjitsu/src/rocjitsu/vm/amdgpu/pci/sdma_block_model.cpp
    lib/rocjitsu/src/rocjitsu/kmd/linux/simulated_kfd.h
    lib/rocjitsu/src/rocjitsu/kmd/linux/simulated_kfd.cpp
    lib/rocjitsu/src/rocjitsu/vm/amdgpu/mes_engine.h
    lib/rocjitsu/src/rocjitsu/vm/amdgpu/mes_engine.cpp
    lib/rocjitsu/src/rocjitsu/vm/amdgpu/pci/mes_block_model.h
    lib/rocjitsu/src/rocjitsu/vm/amdgpu/pci/mes_block_model.cpp
)

foreach(SOURCE_FILE IN LISTS SDMA_FRONTEND_SOURCES)
    file(READ "${ROCJITSU_SOURCE_DIR}/${SOURCE_FILE}" SOURCE_CONTENT)
    foreach(
        FORBIDDEN_TOKEN
        IN
        ITEMS
            "PollRegister"
            "poll_register"
            "SdmaRingConsumer"
            "SdmaPacketProcessor"
            "PacketProcessResult"
            "CircularRingReader"
            "ConsumerCursorJournal"
            "SdmaExecutor"
            "SdmaQueueRunner"
            "SdmaQueueBackend"
    )
        string(FIND "${SOURCE_CONTENT}" "${FORBIDDEN_TOKEN}" TOKEN_POSITION)
        if(NOT TOKEN_POSITION EQUAL -1)
            message(FATAL_ERROR "${SOURCE_FILE} contains SDMA packet semantics")
        endif()
    endforeach()
endforeach()

file(
    READ
        "${ROCJITSU_SOURCE_DIR}/lib/rocjitsu/src/rocjitsu/vm/amdgpu/pci/gpu_pci_device.cpp"
    GPU_PCI_DEVICE_CONTENT
)
string(
    FIND "${GPU_PCI_DEVICE_CONTENT}"
    "callbacks.read_register"
    RAW_READ_POSITION
)
if(RAW_READ_POSITION EQUAL -1)
    message(
        FATAL_ERROR
        "the PCI SDMA adapter does not expose a raw register-read callback"
    )
endif()
