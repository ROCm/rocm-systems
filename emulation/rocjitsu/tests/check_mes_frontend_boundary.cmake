# Copyright (c) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

if(NOT DEFINED ROCJITSU_SOURCE_DIR)
    message(FATAL_ERROR "ROCJITSU_SOURCE_DIR is required")
endif()

set(MES_ENGINE_SOURCES
    lib/rocjitsu/src/rocjitsu/vm/amdgpu/mes_engine.h
    lib/rocjitsu/src/rocjitsu/vm/amdgpu/mes_engine.cpp
)
foreach(SOURCE_FILE IN LISTS MES_ENGINE_SOURCES)
    file(READ "${ROCJITSU_SOURCE_DIR}/${SOURCE_FILE}" SOURCE_CONTENT)
    foreach(
        FORBIDDEN_TOKEN
        IN
        ITEMS "pci/" "PciMemoryAccess" "PciTransportSession"
    )
        string(FIND "${SOURCE_CONTENT}" "${FORBIDDEN_TOKEN}" TOKEN_POSITION)
        if(NOT TOKEN_POSITION EQUAL -1)
            message(
                FATAL_ERROR
                "${SOURCE_FILE} contains frontend dependency ${FORBIDDEN_TOKEN}"
            )
        endif()
    endforeach()
endforeach()

set(MES_FRONTEND_SOURCES
    lib/rocjitsu/src/rocjitsu/vm/amdgpu/pci/mes_block_model.h
    lib/rocjitsu/src/rocjitsu/vm/amdgpu/pci/mes_block_model.cpp
)
foreach(SOURCE_FILE IN LISTS MES_FRONTEND_SOURCES)
    file(READ "${ROCJITSU_SOURCE_DIR}/${SOURCE_FILE}" SOURCE_CONTENT)
    foreach(
        FORBIDDEN_TOKEN
        IN
        ITEMS
            "GpuQueueRegistry"
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
            "register_gfx12_address_space"
            "replace_gfx12_address_space_root"
            "pending_mes_frames_"
            "pending_compute_runs_"
    )
        string(FIND "${SOURCE_CONTENT}" "${FORBIDDEN_TOKEN}" TOKEN_POSITION)
        if(NOT TOKEN_POSITION EQUAL -1)
            message(
                FATAL_ERROR
                "${SOURCE_FILE} contains core MES semantic ${FORBIDDEN_TOKEN}"
            )
        endif()
    endforeach()
endforeach()
