/*************************************************************************
 * Copyright (c) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#ifndef RCCL_ROCTX_H
#define RCCL_ROCTX_H

#include <iostream>
#include <string.h>
#include <map>

#ifdef ROCTX_ENABLE
#include <roctracer/roctx.h>
#endif
#include "nvtx3/nvtx3.hpp"
#include "device.h"

#define MAX_MESSAGE_LENGTH 1024
#define NVTX_PAYLOAD_ENTRY_TYPE_REDOP 11 + NVTX_PAYLOAD_ENTRY_TYPE_SCHEMA_ID_STATIC_START

/**
 * \brief Equivalent of nvtx types for roctx.
*/
enum roctxPayloadEntryType {
  /**
   * Only include the required/used types by rccl,
   * and needs to be updated in case of new type
   * tracing.
  */
  ROCTX_PAYLOAD_ENTRY_TYPE_INT,
  ROCTX_PAYLOAD_ENTRY_TYPE_SIZE,
  ROCTX_PAYLOAD_ENTRY_TYPE_REDOP,
  ROCTX_PAYLOAD_ENTRY_TYPE_DATATYPE,
  ROCTX_PAYLOAD_NUM_ENTRY_TYPES
};

/**
 * \brief Stores the contents of the message that will be used by roctx.
*/
struct roctxPayloadSchemaEntryInfo {
  /**
   * Description of the data.
  */
  const char* name;

  /**
   * Type of the data.
  */
  roctxPayloadEntryType type;

  /**
   * Union of possible payload types.
   * 
   * Should be in sync with roctxPayloadEntryType.
  */
  union {
    int typeInt;
    size_t typeSize;
    ncclDevRedOp_t typeRedOp;
    ncclDataType_t typeDataType;
  } payload;
};

struct roctxPayloadInfo {
  /**
   * Payload name. Usually the name of the function/API 
   * being called from.
  */
  const char* id;

  /**
   * Number of paylod entries
  */
  size_t numEntries;
  
  /**
   * Pointer to roctxPayloadSchemaEntryInfo elements in memory.
  */
  roctxPayloadSchemaEntryInfo* payloadEntries = nullptr;

  /**
   * Message that will be used by roctx
  */
  char* message = nullptr;
};

typedef roctxPayloadInfo* roctxPayloadInfo_t;

extern const char* roctxEntryTypeStr[ROCTX_PAYLOAD_NUM_ENTRY_TYPES];
extern const char* ncclRedOpStr[ncclNumDevRedOps];
extern const char* ncclDataTypeStr[ncclNumTypes];

/**
 * \brief Maps nvtx types to roctx types.
*/
extern std::map<uint64_t, roctxPayloadEntryType> nvtxToRoctx;

/**
 * \brief Allocate required memory for roctx
*/
void roctxAlloc(roctxPayloadInfo_t payloadInfo, const size_t numEntries);

/**
 * \brief Free all the resources used by roctx
*/
void roctxFree(roctxPayloadInfo_t payloadInfo);

/**
 * \brief Extracts payload schema entry info from nvtxPayloadSchemaEntry_t and,
 * nvtxPayloadData_t and stores in an array.
*/
void extractPayloadInfo(const nvtxPayloadSchemaEntry_t* schema, const nvtxPayloadData_t* data, const size_t numEntries, 
                        const char* schemaName, roctxPayloadInfo_t payloadInfo);

/**
 * \brief Stringify roctxPayloadInfo_t struct. Used as roctx message.
*/
void stringify(roctxPayloadInfo_t payloadInfo);

/**
 * \brief Class to make roctx calls scoped.
*/
class roctx_scoped_range_in {
public:
  /**
   * Construct a 'roctx_scoped_range_in' with specified NVTX params,
   * 'numEntries', and 'schemaName'
  */
  explicit roctx_scoped_range_in(const nvtxPayloadSchemaEntry_t* schema, const nvtxPayloadData_t* data, 
                                const size_t numEntries, const char* schemaName) noexcept;

  /**
   * Construct a 'roctx_scoped_range_in' with the specified 'message'
  */
  explicit roctx_scoped_range_in(const char* message) noexcept;

  /**
   * Default constructor 'roctx_scoped_range_in'
  */
  roctx_scoped_range_in() noexcept;

  /**
   * Destroy the roctx_scoped_range_in, ending the ROCTX range event.
   */
  ~roctx_scoped_range_in() noexcept;

private:
  roctxPayloadInfo payloadInfo;
};

#endif // RCCL_ROCTX_H