// MIT License
//
// Copyright (c) 2017-2025 Advanced Micro Devices, Inc.
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
// THE SOFTWARE.


#pragma once

#include <hsa/hsa.h>
#include <hsa/hsa_ven_amd_aqlprofile.h>

#include "aqlprofile-sdk/version.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
  uint64_t handle;
} aqlprofile_handle_t;

typedef enum {
  AQLPROFILE_MEMORY_HINT_NONE = 0,
  AQLPROFILE_MEMORY_HINT_HOST = 1,
  AQLPROFILE_MEMORY_HINT_DEVICE_UNCACHED = 2,
  AQLPROFILE_MEMORY_HINT_DEVICE_COHERENT = 3,
  AQLPROFILE_MEMORY_HINT_DEVICE_NONCOHERENT = 4,
  AQLPROFILE_MEMORY_HINT_LAST
} aqlprofile_memory_hint_t;

typedef enum {
  AQLPROFILE_AGENT_VERSION_NONE = 0,
  AQLPROFILE_AGENT_VERSION_V0 = 1,
  AQLPROFILE_AGENT_VERSION_V1 = 2,
  AQLPROFILE_AGENT_VERSION_LAST
} aqlprofile_agent_version_t;

/**
 * @brief Enums for counter blocks.
 * AQLPROFILE_BLOCK_NAME_RESERVED_X are blocks reserved for npi. Reserving them here can maintain
 * enum consistency between mainline and npi.
 * TODO: Move all counter blocks here from hsa_ven_amd_aqlprofile.h
 */
typedef enum {
  // Blocks reserved for NPI support
  AQLPROFILE_BLOCK_NAME_RESERVED_0 = HSA_VEN_AMD_AQLPROFILE_BLOCKS_NUMBER,
  AQLPROFILE_BLOCK_NAME_RESERVED_1,
  AQLPROFILE_BLOCK_NAME_RESERVED_2,
  AQLPROFILE_BLOCK_NAME_RESERVED_3,
  AQLPROFILE_BLOCK_NAME_RESERVED_4,
  AQLPROFILE_BLOCK_NAME_RESERVED_5,

  // Blocks available for most ASICs, but not currently in use
  AQLPROFILE_BLOCK_NAME_CPG,
  AQLPROFILE_BLOCK_NAME_RLC,

  // New blocks for gc_12_0_x
  AQLPROFILE_BLOCK_NAME_CHA,
  AQLPROFILE_BLOCK_NAME_CHC,
  AQLPROFILE_BLOCK_NAME_GC_CANE,
  AQLPROFILE_BLOCK_NAME_GC_FFBM,
  AQLPROFILE_BLOCK_NAME_GC_L2TLB,
  AQLPROFILE_BLOCK_NAME_GC_UTCL1,
  AQLPROFILE_BLOCK_NAME_GC_UTCL2,
  AQLPROFILE_BLOCK_NAME_GC_VML2,
  AQLPROFILE_BLOCK_NAME_GC_VML2_SPM,
  AQLPROFILE_BLOCK_NAME_GCEA_SE,
  AQLPROFILE_BLOCK_NAME_GRBMH,
  AQLPROFILE_BLOCK_NAME_SQG,

  // Blocks reserved for NPI support
  AQLPROFILE_BLOCK_NAME_RESERVED_6,
  AQLPROFILE_BLOCK_NAME_RESERVED_7,
  AQLPROFILE_BLOCK_NAME_RESERVED_8,
  AQLPROFILE_BLOCK_NAME_RESERVED_9,

  // Add new blocks above
  AQLPROFILE_BLOCKS_NUMBER
} aqlprofile_block_name_t;

/**
 * @brief Flags to describe which agents can access given buffer.
 */
typedef union {
  uint32_t raw;
  struct {
    uint32_t device_access : 1;
    uint32_t host_access : 1;
    uint32_t memory_hint : 6;  // One of aqlprofile_memory_hint_t
    uint32_t _reserved : 24;
  };
} aqlprofile_buffer_desc_flags_t;

/**
 * @brief Callback to request a memory buffer, which will be tied to a profile.
 * The user is responsible for clearing up memory after the profile is no longer needed.
 * @param[out] ptr The pointer containing memory.
 * @param[in] size Minimum requested buffer size.
 * @param[in] flags Access flags, requesting which agents need to read/write to the buffer.
 * @param[in] userdata Data to be passed back to user.
 * @retval HSA_STATUS_SUCCESS if successful
 * @retval HSA_STATUS_ERROR if memory could not be allocated
 */
typedef hsa_status_t (*aqlprofile_memory_alloc_callback_t)(void** ptr, uint64_t size,
                                                           aqlprofile_buffer_desc_flags_t flags,
                                                           void* userdata);

/**
 * @brief Callback to dealloc memory requested via aqlprofile_memory_alloc_callback_t
 * @param[in] ptr The pointer containing memory.
 * @param[in] userdata Data to be passed back to user.
 * @retval HSA_STATUS_SUCCESS if successful
 * @retval HSA_STATUS_ERROR if memory could not be allocated
 */
typedef void (*aqlprofile_memory_dealloc_callback_t)(void* ptr, void* userdata);

typedef enum {
  AQLPROFILE_ACCUMULATION_NONE = 0, /** Do not accumulate event */
  AQLPROFILE_ACCUMULATION_LO_RES,   /**< The event should be integrated over quad-cycles */
  AQLPROFILE_ACCUMULATION_HI_RES,   /**< The event should be integrated every cycle */
  AQLPROFILE_ACCUMULATION_LAST,
} aqlprofile_accumulation_type_t;

typedef enum
{
    AQLPROFILE_SPM_DEPTH_NONE,
    AQLPROFILE_SPM_DEPTH_16_BITS,
    AQLPROFILE_SPM_DEPTH_32_BITS,
    AQLPROFILE_SPM_DEPTH_64_BITS
} aqlprofile_spm_depth_t;

/**
 * @brief Special flags indicating additional properties to a counter. E.g. Accumulation metrics
 */
typedef union
{
    uint32_t raw;
    struct
    {
        uint32_t accum     : 3; /**< One of aqlprofile_accumulation_type_t */
        uint32_t _reserved : 25;
        uint32_t depth     : 4; /**< One of aqlprofile_spm_depth_t */
    } sq_flags;
    struct
    {
        uint32_t _reserved : 28;
        uint32_t depth     : 4; /**< One of aqlprofile_spm_depth_t */
    } spm_flags;
} aqlprofile_pmc_event_flags_t;

/**
 * @brief Struct containing all necessary information of an event (counter).
 */
typedef struct {
  uint32_t block_index;                           /**< Block channel. */
  uint32_t event_id;                              /**< Event ID as fined by XML */
  aqlprofile_pmc_event_flags_t flags;             /**< Special event flags e.g. accumulation */
  hsa_ven_amd_aqlprofile_block_name_t block_name; /**< Block name as defined by block indexes */
} aqlprofile_pmc_event_t;

/**
 * @brief Struct containing information about the agent. User code sets these values
 * to the describe the agent to profile. Information can be obtained either from HSA
 * (if loaded) or the KFD topology.
 */
typedef struct {
  const char* agent_gfxip; /**< Agent GFXIP (HSA_AGENT_INFO_NAME or KFD.product_name) */
  uint32_t xcc_num;        /**< XCC's on the agent (HSA_AMD_AGENT_INFO_NUM_XCC or KFD.num_xcc) */
  uint32_t se_num;         /**< SE's on the agent (HSA_AMD_AGENT_INFO_NUM_SHADER_ENGINES or
                              KFD.num_shader_banks) */
  uint32_t cu_num; /**< CU's on the agent (HSA_AMD_AGENT_INFO_COMPUTE_UNIT_COUNT or KFD.cu_count) */
  uint32_t shader_arrays_per_se; /**< Shader arrays per SE of agent
                                    (HSA_AMD_AGENT_INFO_NUM_SHADER_ARRAYS_PER_SE or
                                    KFD.simd_arrays_per_engine)*/
} aqlprofile_agent_info_t;

/**
 * @brief Struct containing information about the agent. User code sets these values
 * to the describe the agent to profile. Information can be obtained either from HSA
 * (if loaded) or the KFD topology.
 */
typedef struct {
  const char* agent_gfxip; /**< Agent GFXIP (HSA_AGENT_INFO_NAME or KFD.product_name) */
  uint32_t xcc_num;        /**< XCC's on the agent (HSA_AMD_AGENT_INFO_NUM_XCC or KFD.num_xcc) */
  uint32_t se_num;         /**< SE's on the agent (HSA_AMD_AGENT_INFO_NUM_SHADER_ENGINES or
                              KFD.num_shader_banks) */
  uint32_t cu_num; /**< CU's on the agent (HSA_AMD_AGENT_INFO_COMPUTE_UNIT_COUNT or KFD.cu_count) */
  uint32_t shader_arrays_per_se; /**< Shader arrays per SE of agent
                                    (HSA_AMD_AGENT_INFO_NUM_SHADER_ARRAYS_PER_SE or
                                    KFD.simd_arrays_per_engine)*/
  uint32_t domain; /**< PCI domain of the GPU agent (HSA_AMD_AGENT_INFO_DOMAIN or KFD.domain) */
  uint32_t location_id; /**< BDF (Bus/Device/function number) of the GPU agent
                           (HSA_AMD_AGENT_INFO_BDFID or KFD.location_id)*/
} aqlprofile_agent_info_v1_t;

/**
 * @brief Struct containing a handle to a registered agent
 *
 */
typedef struct {
  uint64_t handle;
} aqlprofile_agent_handle_t;

/**
 * @brief Versioning info.
 */
typedef struct aqlprofile_version_t {
  uint32_t major;
  uint32_t minor;
  uint32_t patch;
} aqlprofile_version_t;

/**
 * @brief Query the version of aqlprofile library.
 * @param[out] version aqlprofile version info is stored if non-NULL
 * @retval HSA_STATUS_SUCCESS returned when version is a valid pointer
 * @retval HSA_STATUS_ERROR_INVALID_ARGUMENT if version is a null
 */
hsa_status_t aqlprofile_get_version(aqlprofile_version_t* version);

/**
 * @brief Registers an agent to be used with AQL profile.
 * @param[out] agent_id Handle to newly registered agent
 * @param[in] agent_info Info to register a new agent with AQL Profiler
 * @retval HSA_STATUS_SUCCESS registration ok
 * @retval HSA_STATUS_ERROR registration failed
 */
hsa_status_t aqlprofile_register_agent(aqlprofile_agent_handle_t* agent_id,
                                       const aqlprofile_agent_info_t* agent_info);

/**
 * @brief Registers an agent to be used with AQL profile.
 * @param[out] agent_id Handle to newly registered agent
 * @param[in] agent_info Info to register a new agent with AQL Profiler
 * @param[in] version Version of the agent info structure
 * @retval HSA_STATUS_SUCCESS registration ok
 * @retval HSA_STATUS_ERROR registration failed
 */
hsa_status_t aqlprofile_register_agent_info(aqlprofile_agent_handle_t* agent_id,
                                            const void* agent_info,
                                            aqlprofile_agent_version_t version);
/**
 * @brief AQLprofile struct containing information for perfmon events
 */
typedef struct {
  aqlprofile_agent_handle_t agent;
  const aqlprofile_pmc_event_t* events;
  uint32_t event_count;
} aqlprofile_pmc_profile_t;

// Profile attributes
typedef enum {
  AQLPROFILE_INFO_COMMAND_BUFFER_SIZE = 0,  // get_info returns uint32_t value
  AQLPROFILE_INFO_PMC_DATA_SIZE = 1,        // get_info returns uint32_t value
  AQLPROFILE_INFO_PMC_DATA = 2,             // get_info returns PMC uint64_t value
                                            // in info_data object
  AQLPROFILE_INFO_BLOCK_COUNTERS = 4,       // get_info returns number of block counter
  AQLPROFILE_INFO_BLOCK_ID = 5,             // get_info returns block id, instances
                                            // by name string using _id_query_t
  AQLPROFILE_INFO_ENABLE_CMD = 6,           // get_info returns size/pointer for
                                            // counters enable command buffer
  AQLPROFILE_INFO_DISABLE_CMD = 7,          // get_info returns size/pointer for
                                            // counters disable command buffer
} aqlprofile_pmc_info_type_t;

hsa_status_t aqlprofile_get_pmc_info(const aqlprofile_pmc_profile_t* profile,
                                     aqlprofile_pmc_info_type_t attribute, void* value);

typedef enum aqlprofile_att_parameter_rt_timestamp_t
{
  AQLPROFILE_ATT_PARAMETER_RT_TIMESTAMP_DEFAULT = 0,
  AQLPROFILE_ATT_PARAMETER_RT_TIMESTAMP_ENABLE,
  AQLPROFILE_ATT_PARAMETER_RT_TIMESTAMP_DISABLE
} aqlprofile_att_parameter_rt_timestamp_t;

typedef enum aqlprofile_att_parameter_name_ext_t
{
  /**
   * HSA_VEN_AMD_AQLPROFILE_PARAMETER_NAME_ATT_BUFFER_SIZE + 1
   */
  AQLPROFILE_ATT_PARAMETER_NAME_BUFFER_SIZE_HIGH = 11,
  AQLPROFILE_ATT_PARAMETER_NAME_RT_TIMESTAMP,  // one of aqlprofile_att_parameter_rt_timestamp_t
  AQLPROFILE_ATT_PARAMETER_NAME_NUM_BUFFERS
} aqlprofile_att_parameter_name_ext_t;

// Profile parameter object
typedef struct {
  hsa_ven_amd_aqlprofile_parameter_name_t parameter_name;  // Or aqlprofile_att_parameter_name_ext_t
  union {
    uint32_t value;
    struct {
      uint32_t counter_id : 28;
      uint32_t simd_mask : 4;
    };
  };
} aqlprofile_att_parameter_t;

/**
 * @brief AQLprofile struct containing information for Advanced Thread Trace
 */
typedef struct {
  hsa_agent_t agent;
  const aqlprofile_att_parameter_t* parameters;
  uint32_t parameter_count;
} aqlprofile_att_profile_t;

/**
 * @brief Data callback for perfmon events. Each event will call this once per coordinate
 * @param[in] event The event information passed in from aqlprofile_pmc_profile_t
 * @param[in] counter_id Internal ID of the counter
 * @param[in] counter_value The event value, as incremented from start() to stop()
 * @param[in] userdata Data returned to user
 * @retval HSA_STATUS_SUCCESS to continue iteration
 * @retval HSA_STATUS_ERROR to stop callback iteration
 */
typedef hsa_status_t (*aqlprofile_pmc_data_callback_t)(aqlprofile_pmc_event_t event,
                                                       uint64_t counter_id, uint64_t counter_value,
                                                       void* userdata);

/**
 * @brief Data callback for thread trace. This will be called at least once per shader engine
 * @param[in] shader Shader Engine ID
 * @param[in] buffer Pointer containing the data
 * @param[in] size Amount of bytes used by thread trace
 * @param[in] callback_data Data returned to user
 * @retval HSA_STATUS_SUCCESS to continue iteration
 * @retval HSA_STATUS_ERROR to stop callback iteration
 */
typedef hsa_status_t (*aqlprofile_att_data_callback_t)(uint32_t shader, void* buffer, uint64_t size,
                                                       void* callback_data);

/**
 * @brief Memory copy fn for aqlprofile to copy data.
 * @param[in] dst Destination pointer to copy data to.
 * @param[in] src Source pointer where data is to be copied from.
 * @param[in] size Amount of bytes to be copied.
 * @param[in] userdata Data returned to user
 * @retval HSA_STATUS_SUCCESS on success
 * @retval HSA_STATUS_ERROR on failure
 */
typedef hsa_status_t (*aqlprofile_memory_copy_t)(void* dst, const void* src, size_t size,
                                                 void* userdata);

/**
 * @brief Validates the event for the agent.
 * @param[in] agent The agent to validate the event for.
 * @param[in] event The event to validate.
 * @param[out] result True if the event is valid for the agent, false otherwise.
 * @retval HSA_STATUS_SUCCESS if the event was validated.
 * @retval HSA_STATUS_ERROR if the event was not validated.
 */
hsa_status_t aqlprofile_validate_pmc_event(aqlprofile_agent_handle_t agent,
                                           const aqlprofile_pmc_event_t* event, bool* result);

/**
 * @brief Iterate_data() will parse the event data and call @callback with the resulting event data
 * @param[in] handle The handle returned from aqlprofile_pmc_create_packets()
 * @param[in] callback CB where the resulting event values are going to be returned
 * @param[in] userdata Data sent back to user
 * @retval HSA_STATUS_SUCCESS all operations exited succesfully
 * @retval HSA_STATUS_ERROR if some callback returns an error
 * @retval HSA_STATUS_ERROR_INVALID_ARGUMENT if invalid handle is given
 */
hsa_status_t aqlprofile_pmc_iterate_data(aqlprofile_handle_t handle,
                                         aqlprofile_pmc_data_callback_t callback, void* userdata);

/**
 * @brief Struct to be returned by aqlprofile_pmc_create_packets
 */
typedef struct {
  hsa_ext_amd_aql_pm4_packet_t start_packet; /**< Reset counters and start incrementing */
  hsa_ext_amd_aql_pm4_packet_t stop_packet;  /**< Pause counters from incrementing */
  hsa_ext_amd_aql_pm4_packet_t read_packet;  /**< Retrieve results from device */
} aqlprofile_pmc_aql_packets_t;

/**
 * @brief Function to create AQL packets to be inserted into the queue.
 * @param[out] handle To be passed to iterate_data()
 * @param[out] packets Pointer to where the start, stop and read packets will be written to
 * @param[in] profile Agent and events information
 * @param[in] alloc_cb Memory allocation, which may request cpu or gpu memory for internal use
 * @param[in] dealloc_cb Function to free memory allocated by alloc_cb
 * @param[in] userdata Data passed back to user via memory alloc callback
 */
hsa_status_t aqlprofile_pmc_create_packets(aqlprofile_handle_t* handle,
                                           aqlprofile_pmc_aql_packets_t* packets,
                                           aqlprofile_pmc_profile_t profile,
                                           aqlprofile_memory_alloc_callback_t alloc_cb,
                                           aqlprofile_memory_dealloc_callback_t dealloc_cb,
                                           aqlprofile_memory_copy_t memcpy_cb, void* userdata);

/**
 * @brief Function to delete AQL packets after creation by aqlprofile_pmc_create_packets
 * @param[in] handle Returned by aqlprofile_pmc_create_packets()
 */
void aqlprofile_pmc_delete_packets(aqlprofile_handle_t handle);

/**
 * @brief Iterates over thread trace data and the data to user
 * @param[in] handle The handle returned from aqlprofile_att_create_packets()
 * @param[in] callback CB where the resulting data is going to be returned
 * @param[in] userdata Data sent back to user
 * @retval HSA_STATUS_SUCCESS all operations exited succesfully
 * @retval HSA_STATUS_ERROR if some callback returns an error
 * @retval HSA_STATUS_ERROR_INVALID_ARGUMENT if invalid handle is given
 */
hsa_status_t aqlprofile_att_iterate_data(aqlprofile_handle_t handle,
                                         aqlprofile_att_data_callback_t callback, void* userdata);

/**
 * @brief Struct containing AQLpackets to start and stop thread trace
 */
typedef struct {
  hsa_ext_amd_aql_pm4_packet_t start_packet; /**< Packet to start thread trace */
  hsa_ext_amd_aql_pm4_packet_t stop_packet;  /**< Packet to stop thread trace and flush data */
} aqlprofile_att_control_aql_packets_t;

/**
 * @brief Fn to create start and stop thread trace packets
 * @param[out] handle To be passed to iterate_data()
 * @param[out] packets Packets returned by this function to start and stop thread trace
 * @param[in] profile Agent information and extra parameters for thread trace
 * @param[in] callback Memory allocation fn which may request cpu or gpu memory
 * @retval HSA_STATUS_SUCCESS if all packets created succesfully
 * @retval HSA_STATUS_ERROR otherwise
 */
hsa_status_t aqlprofile_att_create_packets(aqlprofile_handle_t* handle,
                                           aqlprofile_att_control_aql_packets_t* packets,
                                           aqlprofile_att_profile_t profile,
                                           aqlprofile_memory_alloc_callback_t alloc_cb,
                                           aqlprofile_memory_dealloc_callback_t dealloc_cb,
                                           aqlprofile_memory_copy_t memcpy_cb, void* userdata);

void aqlprofile_att_delete_packets(aqlprofile_handle_t handle);

/**
 * @brief Fn to create query and swap packets in case of double buffering.
 * The caller must pool information by sending a query_status packet, followed by a call
 * to aqlprofile_att_get_buffer_status(). If aqlprofile_att_buffer_status_t.is_full, then
 * a buffer_swap packet must be inserted into the queue.
 * 
 * @param[out] header If not zero, must be inserted as first 8 bytes.
 * @param[out] query_status To be inserted before calls to aqlprofile_att_get_buffer_status
 * @param[out] buffer_swap array of AQLPROFILE_ATT_PARAMETER_NAME_NUM_BUFFERS transition packets
 * @param[inout] num_buffer_swap In: # of packets in number buffer_swap. Out: # of buffers used.
 * @param[in] handle Created in aqlprofile_att_create_packets()
 * @param[in] shader_engine_id Shader engine to get packets from
 * @param[in] flags Must be zero
 * @retval HSA_STATUS_SUCCESS if all packets created succesfully
 * @retval HSA_STATUS_ERROR otherwise
 */
hsa_status_t aqlprofile_att_get_buffer_packets(uint64_t* header,
                                               hsa_ext_amd_aql_pm4_packet_t* query_status,
                                               hsa_ext_amd_aql_pm4_packet_t** buffer_swap,
                                               uint64_t* num_buffer_swap,
                                               aqlprofile_handle_t handle,
                                               int shader_engine_id,
                                               int flags);

struct aqlprofile_att_buffer_status_t
{
  uint64_t _size;       // sizeof(aqlprofile_att_buffer_status_t)
  void*    data;        // Read data from, if is full
  uint64_t read_size;   // Number of bytes to read, if is full
  uint64_t num_swaps;   // For verification purposes. Number of swaps previously executed.
  bool     needs_swap;  // If buffer requires swap
  bool     is_too_late;
  bool     error;
};

/**
 * @brief Fn to retrieve buffer status.
 * Must be called at least once with has_buffer_swapped=true for every swap packet inserted.
 * @param[out] out Query result
 * @param[in] handle What was passed to aqlprofile_att_get_buffer_packets
 * @param[in] shader_engine_id Shader engine (SE) ID
 * @param[in] flags Must be zero
 * @retval HSA_STATUS_SUCCESS if all packets created succesfully
 * @retval HSA_STATUS_ERROR otherwise
 */
hsa_status_t aqlprofile_att_update_buffer_status(aqlprofile_att_buffer_status_t* out,
                                                 aqlprofile_handle_t handle,
                                                 int shader_engine_id,
                                                 int flags);

/**
 * @brief Callback for iteration of all possible event coordinate IDs and coordinate names.
 * @param [in] id Integer identifying the dimension.
 * @param [in] name Name of the dimension
 * @param [in] data User data supplied to @ref aqlprofile_iterate_event_ids
 * @retval HSA_STATUS_SUCCESS Continues iteration
 * @retval OTHERS Any other HSA return values stops iteration, passing back this value through
 *         @ref aqlprofile_iterate_event_ids
 */
typedef hsa_status_t (*aqlprofile_eventname_callback_t)(int id, const char* name, void* data);

/**
 * @brief Iterate over all possible event coordinate IDs and their names.
 * @param [in] callback Callback to use for iteration of dimensions
 * @param [in] user_data Data to supply to callback @ref aqlprofile_eventname_callback_t
 * @retval HSA_STATUS_SUCCESS if successful
 * @retval HSA_STATUS_ERROR if error on interation
 * @retval OTHERS If @ref aqlprofile_eventname_callback_t returns non-HSA_STATUS_SUCCESS,
 *         that value is returned.
 */
hsa_status_t aqlprofile_iterate_event_ids(aqlprofile_eventname_callback_t callback,
                                          void* user_data);

/**
 * @brief Iterate over all event coordinates for a given agent_t and event_t.
 * @param position A counting sequence indicating callback number.
 * @param id Coordinate ID as in _iterate_event_ids.
 * @param extent Coordinate extent indicating maximum allowed instances.
 * @param coordinate The coordinate, in the range [0,extent-1].
 * @param name Coordinate name as in _iterate_event_ids.
 * @param userdata Userdata returned from _iterate_event_coord function.
 */
typedef hsa_status_t (*aqlprofile_coordinate_callback_t)(int position, int id, int extent,
                                                         int coordinate, const char* name,
                                                         void* userdata);

/**
 * @brief Iterate over all event coordinates for a given agent_t and event_t.
 * @param[in] agent HSA agent.
 * @param[in] event The event ID and block ID to iterate for.
 * @param[in] sample_id aqlprofile_info_data_t.sample_id returned from _aqlprofile_iterate_data.
 * @param[in] callback Callback function to return the coordinates.
 * @param[in] userdata Arbitrary data pointer to be sent back to the user via callback.
 */
hsa_status_t aqlprofile_iterate_event_coord(aqlprofile_agent_handle_t agent,
                                            aqlprofile_pmc_event_t event, uint64_t sample_id,
                                            aqlprofile_coordinate_callback_t callback,
                                            void* userdata);

typedef struct {
  uint64_t id;
  uint64_t addr;
  uint64_t size;
  hsa_agent_t agent;
  uint32_t isUnload : 1;
  uint32_t fromStart : 1;
} aqlprofile_att_codeobj_data_t;

/**
 * @brief Creates an AQL packet for marking code objects
 * @param[out] packet Returned packet
 * @param[out] handle The handle created for these packets
 * @param[in] data Code object information
 * @param[in] alloc_cb Callback to return both CPU and GPU accessible memory on demand
 * @param[in] dealloc_cb Callback to free data allocated by alloc_cb()
 * @param[in] userdata Userdata to be passed back to memory callbacks
 */
hsa_status_t aqlprofile_att_codeobj_marker(hsa_ext_amd_aql_pm4_packet_t* packet,
                                           aqlprofile_handle_t* handle,
                                           aqlprofile_att_codeobj_data_t data,
                                           aqlprofile_memory_alloc_callback_t alloc_cb,
                                           aqlprofile_memory_dealloc_callback_t dealloc_cb,
                                           void* userdata);

/**
 * @brief Struct to be returned by aqlprofile_spm_create_packets
 */
typedef struct
{
    hsa_ext_amd_aql_pm4_packet_t start_packet;
    hsa_ext_amd_aql_pm4_packet_t stop_packet;
} aqlprofile_spm_aql_packets_t;

typedef struct
{
    void*  data;  // Valid until delete_packets() is scalled. Caller must save contents otherwise.
    size_t size;  // Size of "data"
} aqlprofile_spm_buffer_desc_t;

typedef enum
{
    AQLPROFILE_SPM_PARAMETER_TYPE_BUFFER_SIZE = 0,
    AQLPROFILE_SPM_PARAMETER_TYPE_SAMPLE_INTERVAL,
    AQLPROFILE_SPM_PARAMETER_TYPE_TIMEOUT,
    AQLPROFILE_SPM_PARAMETER_TYPE_SAMPLE_MODE,
    AQLPROFILE_SPM_PARAMETER_TYPE_LAST,
} aqlprofile_spm_parameter_type_t;

typedef enum
{
    AQLPROFILE_SPM_PARAMETER_SAMPLE_MODE_SCLK = 0,
    AQLPROFILE_SPM_PARAMETER_SAMPLE_MODE_REFCLK
} aqlprofile_spm_parameter_interval_mode_t;

typedef struct
{
    aqlprofile_spm_parameter_type_t type;
    uint64_t                        value;
} aqlprofile_spm_parameter_t;

/**
 * @brief AQLprofile struct containing information for SPM counter events
 */
typedef struct
{
    aqlprofile_agent_handle_t     aql_agent;
    hsa_agent_t                   hsa_agent;
    const aqlprofile_pmc_event_t* events;
    size_t                        event_count;
    aqlprofile_spm_parameter_t*   parameters;
    size_t                        parameter_count;
    size_t                        reserved;  // For future use

    aqlprofile_memory_alloc_callback_t alloc_cb;  // Memory allocation, usually a wrapper for hsa_amd_memory_pool_allocate
    aqlprofile_memory_dealloc_callback_t dealloc_cb;  // Frees memory allocated by alloc_cb
    aqlprofile_memory_copy_t memcpy_cb;  // Copy memory in and out of GPU memory allocated by alloc_cb
    void* userdata;   // Passed back to user in the memory callbacks
} aqlprofile_spm_profile_t;

/**
 * @brief Function to create control SPM packets
 * @param[out] handle    To be passed to iterate_data()
 * @param[out] desc      Used to decode SPM buffer contents
 * @param[out] packets   Start/Stop AQL packets to be inserted in the queue
 * @param[in] profile    Agent and events information
 * @param[in] data_cb    Callback to retrieve SPM data when available
 * @param[in] flags      Reserved. Must be zero.
 * @param[in] userdata   Passed back to user
 * @retval HSA_STATUS_SUCCESS on success
 * @retval HSA_STATUS_ERROR   on generic error
 * @retval HSA_STATUS_ERROR_OUT_OF_RESOURCES if memory allocation unsuccessful
 * @retval HSA_STATUS_ERROR_INVALID_ARGUMENT for invalid parameter or event
 * @retval HSA_STATUS_ERROR_INVALID_AGENT    for invalid agent handle
 */
hsa_status_t
aqlprofile_spm_create_packets(aqlprofile_handle_t*          handle,
                              aqlprofile_spm_buffer_desc_t* desc,
                              aqlprofile_spm_aql_packets_t* packets,
                              aqlprofile_spm_profile_t      profile,
                              size_t                        flags);

/**
 * @brief Destroys resources allocated by aqlprofile_spm_create_packets()
 * Implicitly calls aqlprofile_spm_stop. The descriptor pointer is invalid after this call.
 * @param[in] handle Handle
 */
void
aqlprofile_spm_delete_packets(aqlprofile_handle_t handle);

typedef size_t aqlprofile_spm_buffer_handle_t;

typedef enum
{
    AQLPROFILE_SPM_DATA_FLAGS_DATA_LOSS = 0,
} aqlprofile_spm_data_flags_t;

/**
 * @brief Data callback for SPM events.
 * @param[in] handle   Handle to be passed to aqlprofile_spm_decode_data_callback_t
 * @param[in] spm_data SPM raw data. Can be decoded via aqlprofile_spm_decode()
 * @param[in] size     Size of "spm_data"
 * @param[in] flags    Bitwise combination of aqlprofile_spm_data_flags_t
 * @param[in] userdata Data returned to user
 */
typedef void (*aqlprofile_spm_data_callback_t)(aqlprofile_spm_buffer_handle_t handle,
                                               void*                          spm_data,
                                               size_t                         size,
                                               int                            flags,
                                               void*                          userdata);

/**
 * @brief Starts processing of SPM buffer
 * @param[in] handle   Handle
 * @param[in] data_cb  Callback to retrieve SPM data when available
 * @param[in] userdata Passed back to user
 * @retval HSA_STATUS_SUCCESS on success
 * @retval HSA_STATUS_ERROR generic error
 * @retval HSA_STATUS_ERROR_NOT_INITIALIZED for invalid handle
 */
hsa_status_t
aqlprofile_spm_start(aqlprofile_handle_t            handle,
                     aqlprofile_spm_data_callback_t data_cb,
                     void*                          userdata);

/**
 * @brief Flushes remaining SPM data and stops processing of SPM buffer
 * @param[in] handle Handle
 * @retval HSA_STATUS_SUCCESS on success
 * @retval HSA_STATUS_ERROR generic error
 * @retval HSA_STATUS_ERROR_NOT_INITIALIZED for invalid handle
 */
hsa_status_t
aqlprofile_spm_stop(aqlprofile_handle_t handle);

typedef void (*aqlprofile_spm_decode_callback_v1_t)(uint64_t timestamp,
                                                    uint64_t value,
                                                    uint64_t index,
                                                    int      shader_engine,
                                                    void*    userdata);

/**
 * @brief Decodes a raw buffer returned by aqlprofile_spm_data_callback_t.
 * Returns results accumulated per event_id requested.
 * @param[in] desc Descriptor returned in create_packets()
 * @param[in] decode_cb  Callback where decoded SPM data will be returned to
 * @param[in] data       Raw SPM data returned in aqlprofile_spm_data_callback_t
 * @param[in] size       Raw data size
 * @param[in] userdata   Passed back to user
 * @retval HSA_STATUS_SUCCESS if decode successful
 * @retval HSA_STATUS_ERROR   for generic error
 */
hsa_status_t
aqlprofile_spm_decode_stream_v1(aqlprofile_spm_buffer_desc_t        desc,
                                aqlprofile_spm_decode_callback_v1_t decode_cb,
                                void*                               data,
                                size_t                              size,
                                void*                               userdata);

enum aqlprofile_spm_decode_query_t
{
    AQLPROFILE_SPM_DECODE_QUERY_SEG_SIZE = 0,
    AQLPROFILE_SPM_DECODE_QUERY_NUM_XCC,
    AQLPROFILE_SPM_DECODE_QUERY_EVENT_COUNT,
    AQLPROFILE_SPM_DECODE_QUERY_COUNTER_MAP_BYTE_OFFSET,
    AQLPROFILE_SPM_DECODE_QUERY_LAST
};

hsa_status_t
aqlprofile_spm_decode_query(aqlprofile_spm_buffer_desc_t  desc,
                            aqlprofile_spm_decode_query_t query,
                            uint64_t*                     param_out);

bool
aqlprofile_spm_is_event_supported(aqlprofile_agent_handle_t agent, aqlprofile_pmc_event_t event);

#ifdef __cplusplus
}
#endif
