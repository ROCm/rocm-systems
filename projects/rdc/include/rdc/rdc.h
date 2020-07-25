/*
Copyright (c) 2020 - present Advanced Micro Devices, Inc. All rights reserved.

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
*/

#ifndef INCLUDE_RDC_RDC_H_
#define INCLUDE_RDC_RDC_H_

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

#include <cstdint>

/** \file rdc.h
 *  Main header file for the ROCm RDC library.
 *  All required function, structure, enum, etc. definitions should be defined
 *  in this file.
 *
 *  @brief The rocm_rdc library api is new, and therefore subject to change
 *  either at the ABI or API level. Instead of marking every function prototype
 *  as "unstable", we are instead saying the API is unstable (i.e., changes
 *  are possible) while the major version remains 0. This means that if the
 *  API/ABI changes, we will not increment the major version to 1. Once the
 *  ABI stabilizes, we will increment the major version to 1, and thereafter
 *  increment it on all ABI breaks.
 */

/**
 * @brief Error codes returned by rocm_rdc_lib functions
 */
typedef enum {
     RDC_ST_OK                  = 0,  //!< Success
     RDC_ST_NOT_SUPPORTED,            //!< Not supported feature
     RDC_ST_MSI_ERROR,                //!< The MSI library error
     RDC_ST_FAIL_LOAD_MODULE,         //!< Fail to load the library
     RDC_ST_INVALID_HANDLER,          //!< Invalid handler
     RDC_ST_BAD_PARAMETER,            //!< A parameter is invalid
     RDC_ST_NOT_FOUND,                //!< Cannot find the value
     RDC_ST_CONFLICT,                 //!< Conflict with current state
     RDC_ST_CLIENT_ERROR,             //!< The RDC client error
     RDC_ST_ALREADY_EXIST,            //!< The item already exists
     RDC_ST_MAX_LIMIT                 //!< Max limit recording for the object
} rdc_status_t;

/**
 * @brief rdc operation mode
 * rdc can run in auto mode where background threads will collect metrics.
 * When run in manual mode, the user needs to periodically call
 * rdc_field_update_all for data collection.
 */
typedef enum {
    RDC_OPERATION_MODE_AUTO   = 0,
    RDC_OPERATION_MODE_MANUAL
} rdc_operation_mode_t;

/**
 * @brief type of GPU group
 */
typedef enum {
    RDC_GROUP_DEFAULT = 0,      //!< All GPUs on the Node
    RDC_GROUP_EMPTY             //!< Empty group
} rdc_group_type_t;

/**
 * @brief the type stored in the filed value
 */
typedef enum {
     INTEGER = 0,
     DOUBLE,
     STRING,
     BLOB
} rdc_field_type_t;


//! ID used to represent an invalid GPU
#define GPU_ID_INVALID          -1
//! Used to specify all GPUs
#define RDC_GROUP_ALL_GPUS      -1000
//! Used to specify all stats fields
#define RDC_JOB_STATS_FIELDS    -1000

/**
 * @brief The max rdc field string length
 */
#define RDC_MAX_STR_LENGTH      256

/**
 * @brief The max entities in a group
 */
#define RDC_GROUP_MAX_ENTITIES  64

/**
 * @brief Max number of GPUs supported by RDC
 */
#define RDC_MAX_NUM_DEVICES     16

/**
 * @brief The max fields in a field group
 */
#define RDC_MAX_FIELD_IDS_PER_FIELD_GROUP 128

/**
 * @brief The max number of groups
 */
#define RDC_MAX_NUM_GROUPS      64

/**
 * @brief The max number of the field groups
 */
#define RDC_MAX_NUM_FIELD_GROUPS 64

/**
 * These enums are used to specify a particular field to be retrieved.
 */
typedef enum {
  RDC_FI_INVALID = 0,                //!< Invalid field value
  //!< @brief Identifier fields
  RDC_FI_GPU_COUNT = 1,              //!< GPU count in the system
  RDC_FI_DEV_NAME,                   //!< Name of the device

  /*
   * @brief Frequency related fields
   */
  RDC_FI_GPU_CLOCK = 100,           //!< The current clock for the GPU
  RDC_FI_MEM_CLOCK,                 //!< Clock for the memory

  /*
   * @brief Physical monitor fields
   */
  RDC_FI_MEMORY_TEMP = 200,          //!< Memory temperature for the device
  RDC_FI_GPU_TEMP,                   //!< Current temperature for the device
  RDC_FI_POWER_USAGE = 300,          //!< Power usage for the device

  /*
   * @brief PCIe related fields
   */
  RDC_FI_PCIE_TX = 400,              //!< PCIe Tx utilization information
  RDC_FI_PCIE_RX,                    //!< PCIe Rx utilization information

  /*
   * @brief GPU usage related fields
   */
  RDC_FI_GPU_UTIL = 500,             //!< GPU Utilization
  RDC_FI_GPU_MEMORY_USAGE,           //!< Memory usage of the GPU instance
  RDC_FI_GPU_MEMORY_TOTAL,           //!< Total memory of the GPU instance

  /**
   * @brief ECC related fields
   */
  RDC_FI_ECC_CORRECT_TOTAL = 600,    //!< Accumulated correctable ECC errors
  RDC_FI_ECC_UNCORRECT_TOTAL,        //!< Accumulated uncorrectable ECC errors
} rdc_field_t;

/**
 * @brief handlers used in various rdc calls
 */
typedef void *rdc_handle_t;           //!< Handle used for an RDC session
typedef uint32_t rdc_gpu_group_t;     //!< GPU Group ID type
typedef uint32_t rdc_field_grp_t;     //!< Field group ID type

/**
 * @brief Represents attributes corresponding to a device
 */
typedef struct {
    char  device_name[RDC_MAX_STR_LENGTH];     //!< Name of the device.
} rdc_device_attributes_t;

/**
 * @brief The structure to store the group info
 */
typedef struct {
    unsigned int count;     //!< count of GPUs in the group
    char group_name[RDC_MAX_STR_LENGTH];     //!< group name
    /**
    * The list of entities in the group
    */
    uint32_t entity_ids[RDC_GROUP_MAX_ENTITIES];
} rdc_group_info_t;

/**
 * @brief The structure to store summary of data
 */
typedef struct {
    uint64_t max_value;   //!< Maximum value measured
    uint64_t min_value;   //!< Minimum value measured
    uint64_t average;     //!< Average value measured
    double standard_deviation;   //!< The standard deviation
} rdc_stats_summary_t;

/**
 * @brief The structure to hold the GPU usage information
 */
typedef struct {
    uint32_t gpu_id;        //!< GPU_ID_INVALID for summary information
    uint64_t start_time;    //!< The time to start the watching
    uint64_t end_time;      //!< The time to stop the watching

    uint64_t energy_consumed;               //!< GPU Energy consumed
    uint64_t ecc_correct;                   //!< Correctable errors
    uint64_t ecc_uncorrect;                 //!< Uncorrtable errors
    rdc_stats_summary_t pcie_tx;            //!< Bytes sent over PCIe stats
    rdc_stats_summary_t pcie_rx;            //!< Bytes received over PCIe stats
    rdc_stats_summary_t power_usage;        //!< GPU Power usage stats
    rdc_stats_summary_t gpu_clock;          //!< GPU Clock speed stats
    rdc_stats_summary_t memory_clock;       //!< Mem. Clock speed stats
    rdc_stats_summary_t gpu_utilization;    //!< GPU Utilization stats
    rdc_stats_summary_t gpu_temperature;    //!< GPU temperature stats

    uint64_t max_gpu_memory_used;            //!< Maximum GPU memory used
    rdc_stats_summary_t memory_utilization;  //!< Memory Utilization statistics
} rdc_gpu_usage_info_t;       //!< GPU usage statistics

/**
 * @brief The structure to hold the job stats
 */
typedef struct {
     uint32_t num_gpus;              //!< Number of GPUs used by job
     rdc_gpu_usage_info_t summary;   //!< Job usage summary statistics
                                     //!< (overall)
     rdc_gpu_usage_info_t gpus[16];  //!< Job usage summary staticstics by GPU
} rdc_job_info_t;

/**
 * @brief The structure to store the field value
 */
typedef struct {
    rdc_field_t  field_id;      //!< The field id of the value
    int     status;             //!< RDC_ST_OK or error status
    uint64_t ts;                //!< Timestamp in usec since 1970
    rdc_field_type_t type;      //!< The field type
    union {
        int64_t l_int;
        double  dbl;
        char str[RDC_MAX_STR_LENGTH];
    } value;                     //!< Value of the field. Value type
                                 //!< depends on the field type.
} rdc_field_value;

/**
 * @brief The structure to store the field group info
 */
typedef struct {
    uint32_t count;                      //!< count of fields in the group
    char group_name[RDC_MAX_STR_LENGTH];     //!< field group name
    /**
     * The list of fields in the group
     */
    rdc_field_t field_ids[RDC_MAX_FIELD_IDS_PER_FIELD_GROUP];
} rdc_field_group_info_t;

/**
 * @brief The structure to store the job info
 */
typedef struct {
    char job_id[RDC_MAX_STR_LENGTH];            //!< job id
    rdc_gpu_group_t group_id;                   //!< group name
    uint64_t start_time;                         //!< job start time
    uint64_t stop_time;                          //!< job stop time
} rdc_job_group_info_t;


/**
 *  @brief Initialize ROCm RDC.
 *
 *  @details When called, this initializes internal data structures,
 *  including those corresponding to sources of information that RDC provides.
 *  This must be called before rdc_start_embedded() or rdc_connect()
 *
 *  @param[in] init_flags init_flags Bit flags that tell RDC how to initialize.
 *
 *  @retval ::RDC_ST_OK is returned upon successful call.
 */
rdc_status_t rdc_init(uint64_t init_flags);

/**
 *  @brief Shutdown ROCm RDC.
 *
 *  @details Do any necessary clean up.
 */
rdc_status_t rdc_shutdown();

/**
 *  @brief Start embedded RDC agent within this process.
 *
 *  @details The RDC is loaded as library so that it does not require rdcd
 *  daemon. In this mode, the user has to periodically call
 *  rdc_field_update_all() when op_mode is RDC_OPERATION_MODE_MANUAL, which
 *  tells RDC to collect the stats.
 *
 *  @param[in] op_mode Operation modes. When RDC_OPERATION_MODE_AUTO, RDC schedules
 *  background task to collect the stats. When RDC_OPERATION_MODE_MANUAL, the user
 *  needs to call rdc_field_update_all() periodically.
 *
 *  @param[inout] p_rdc_handle Caller provided pointer to rdc_handle_t. Upon
 *  successful call, the value will contain the handler for following API calls.
 *
 *  @retval ::RDC_ST_OK is returned upon successful call.
 */
rdc_status_t rdc_start_embedded(rdc_operation_mode_t op_mode,
                                    rdc_handle_t* p_rdc_handle);

/**
 *  @brief Stop embedded RDC agent.
 *
 *  @details Stop the embedded RDC agent, and p_rdc_handle becomes
 *  invalid after this call.
 *
 *  @param[in] p_rdc_handle The RDC handler that come from
 *  rdc_start_embedded().
 *  @retval ::RDC_ST_OK is returned upon successful call.
 */
rdc_status_t rdc_stop_embedded(rdc_handle_t p_rdc_handle);

/**
 *  @brief Connect to rdcd daemon
 *
 *  @details This method is used to connect to a remote stand-alone
 *  rdcd daemon.
 *
 *  @param[in] ipAndPort The IP and port of the remote rdcd. The ipAndPort
 *  can be specified in this x.x.x.x:yyyy format, where x.x.x.x is the
 *  IP address and yyyy is the port.
 *
 *  @param[inout] p_rdc_handle Caller provided pointer to rdc_handle_t. Upon
 *  successful call, the value will contain the handler
 *  for following API calls.
 *
 *  @param [in] root_ca The root CA stored in the string in pem format. Set it
 *  as nullptr if the communication is not encrypted.
 *
 *  @param [in] client_cert The client certificate stored in the string in pem
 *  format. Set it as nullptr if the communication is not encrypted.
 *
 *  @param [in] client_key The client key stored in the string in pem format.
 *  Set it as nullptr if the communication is not encrypted.
 *
 *  @retval ::RDC_ST_OK is returned upon successful call.
 */
rdc_status_t rdc_connect(const char *ipAndPort, rdc_handle_t* p_rdc_handle,
        const char* root_ca, const char* client_cert, const char* client_key);

/**
 *  @brief Disconnect from rdcd daemon.
 *
 *  @details Disconnect from rdcd daemon, and p_rdc_handle becomes invalid
 *  after this call.
 *
 *  @param[in] p_rdc_handle The RDC handler that come from rdc_connect().
 *
 *  @retval ::RDC_ST_OK is returned upon successful call.
 */
rdc_status_t rdc_disconnect(rdc_handle_t p_rdc_handle);

/**
 *  @brief Request the RDC to watch the job stats
 *
 *  @details This should be executed as part of job prologue. The summary
 *  job stats can be retrieved using rdc_job_get_stats().
 *  In RDC_OPERATION_MODE_MANUAL, user must call rdc_field_update_all(1)
 *  at least once, before call rdc_job_get_stats()
 *
 *  @param[in] p_rdc_handle The RDC handler.
 *
 *  @param[in] group_id The group of GPUs to be watched.
 *
 *  @param[in] job_id The name of the job.
 *
 *  @param[in] update_freq  How often to update this field in usec.
 *
 *  @retval ::RDC_ST_OK is returned upon successful call.
 */
rdc_status_t rdc_job_start_stats(rdc_handle_t p_rdc_handle,
    rdc_gpu_group_t group_id, const char job_id[64], uint64_t update_freq);

/**
 *  @brief Get the stats of the job using the job id.
 *
 *  @details The stats can be retrieved at any point when the job is in
 *  process.
 *
 *  @param[in] p_rdc_handle The RDC handler.
 *
 *  @param[in] job_id The name of the job.
 *
 *  @param[inout] p_job_info Caller provided pointer to rdc_job_info_t. Upon
 *  successful call, the value will contain the stats of the job.
 *
 *  @retval ::RDC_ST_OK is returned upon successful call.
 */
rdc_status_t rdc_job_get_stats(rdc_handle_t p_rdc_handle,
                           const char job_id[64], rdc_job_info_t* p_job_info);

/**
 *  @brief Request RDC to stop watching the stats of the job
 *
 *  @details This should be execute as part of job epilogue. The job Id
 *  remains available to view the stats at any point. You must call
 *  rdc_watch_job_fields() before this call.
 *
 *  @param[in] p_rdc_handle The RDC handler.
 *
 *  @param[in] job_id The name of the job.
 *
 *  @retval ::RDC_ST_OK is returned upon successful call.
 */
rdc_status_t rdc_job_stop_stats(rdc_handle_t p_rdc_handle,
                                                       const char job_id[64]);

/**
 *  @brief Request RDC to stop tracking the job given by job_id
 *
 *  @details After this call, you will no longer be able to call
 *  rdc_job_get_stats() on this job_id. But you will be able to reuse
 *  the job_id after this call.
 *
 *  @param[in] p_rdc_handle The RDC handler.
 *
 *  @param[in] job_id The name of the job.
 *
 *  @retval ::RDC_ST_OK is returned upon successful call.
 */
rdc_status_t rdc_job_remove(rdc_handle_t p_rdc_handle, const char job_id[64]);

/**
 *  @brief Request RDC to stop tracking all the jobs
 *
 *  @details After this call, you will no longer be able to call
 *  rdc_job_get_stats() on any job id. But you will be able to reuse
 *  the any previous used job id after this call.
 *
 *  @param[in] p_rdc_handle The RDC handler.
 *
 *  @retval ::RDC_ST_OK is returned upon successful call.
 */
rdc_status_t rdc_job_remove_all(rdc_handle_t p_rdc_handle);

/**
 *  @brief Request RDC to update all fields to be watched.
 *
 *  @details In RDC_OPERATION_MODE_MANUAL, the user must call this method
 *  periodically.
 *
 *  @param[in] p_rdc_handle The RDC handler.
 *
 *  @param[in] wait_for_update Whether or not to wait for the update loop to
 *  complete before returning to the caller 1=wait. 0=do not wait.
 *
 *  @retval ::RDC_ST_OK is returned upon successful call.
 */
rdc_status_t rdc_field_update_all(rdc_handle_t p_rdc_handle,
                    uint32_t wait_for_update);

/**
 *  @brief Get indexes corresponding to all the devices on the system.
 *
 *  @details Indexes represents RDC GPU Id corresponding to each GPU on the
 * system and is immutable during the lifespan of the engine. The list
 * should be queried again if the engine is restarted.
 *
 *  @param[in] p_rdc_handle The RDC handler.
 *
 *  @param[out] gpu_index_list Array reference to fill GPU indexes present on
 *  the system.
 *
 *  @param[out] count Number of GPUs returned in gpu_index_list.
 *
 *  @retval ::RDC_ST_OK is returned upon successful call.
 */
rdc_status_t rdc_device_get_all(rdc_handle_t p_rdc_handle,
            uint32_t gpu_index_list[RDC_MAX_NUM_DEVICES], uint32_t* count);

/**
 *  @brief Gets device attributes corresponding to the gpu_index.
 *
 *  @details Fetch the attributes, such as device name, of a GPU.
 *
 *  @param[in] p_rdc_handle The RDC handler.
 *
 *  @param[in] gpu_index GPU index corresponding to which the attributes
 *  should be fetched
 *
 *  @param[out] p_rdc_attr GPU attribute corresponding to the gpu_index.
 *
 *  @retval ::RDC_ST_OK is returned upon successful call.
 */
rdc_status_t rdc_device_get_attributes(rdc_handle_t p_rdc_handle,
            uint32_t gpu_index, rdc_device_attributes_t* p_rdc_attr);

/**
 *  @brief Create a group contains multiple GPUs
 *
 *  @details This method can create a group contains multiple GPUs. Instead of
 *  executing an operation separately for each GPU, the RDC group enables
 *  the user to execute same operation on all the GPUs present in the group as
 *  a single API call.
 *
 *  @param[in] p_rdc_handle The RDC handler.
 *
 *  @param[in] type The type of the group. RDC_GROUP_DEFAULT includes all the
 *  GPUs on the node, and RDC_GROUP_EMPTY creates an empty group.
 *
 *  @param[in] group_name The group name specified as NULL terminated C String
 *
 *  @param[inout] p_rdc_group_id Caller provided pointer to rdc_gpu_group_t.
 *  Upon successful call, the value will contain the group id for following
 *  group API calls.
 *
 *  @retval ::RDC_ST_OK is returned upon successful call.
 */
rdc_status_t rdc_group_gpu_create(rdc_handle_t p_rdc_handle,
    rdc_group_type_t type, const char* group_name,
    rdc_gpu_group_t* p_rdc_group_id);

/**
 *  @brief Add a GPU to the group
 *
 *  @details This method can add a GPU to the group
 *
 *  @param[in] p_rdc_handle The RDC handler.
 *
 *  @param[in] group_id The group id to which the GPU will be added.
 *
 *  @param[in] gpu_index The GPU index to be added to the group.
 *
 *  @retval ::RDC_ST_OK is returned upon successful call.
 */
rdc_status_t rdc_group_gpu_add(rdc_handle_t p_rdc_handle,
    rdc_gpu_group_t group_id, uint32_t gpu_index);

/**
 *  @brief Get information about a GPU group
 *
 *  @details Get detail information about a GPU group created by
 *  rdc_group_gpu_create
 *
 *  @param[in] p_rdc_handle The RDC handler.
 *
 *  @param[in] p_rdc_group_id The GPU group handler created by
 *  rdc_group_gpu_create
 *
 *  @param[out] p_rdc_group_info The information of the GPU
 *  group p_rdc_group_id.
 *
 *  @retval ::RDC_ST_OK is returned upon successful call.
 */
rdc_status_t rdc_group_gpu_get_info(rdc_handle_t p_rdc_handle,
            rdc_gpu_group_t p_rdc_group_id, rdc_group_info_t* p_rdc_group_info);

/**
 *  @brief Used to get information about all GPU groups in the system.
 *
 *  @details Get the list of GPU group ids in the system.
 *
 *  @param[in] p_rdc_handle The RDC handler.
 *
 *  @param[out] group_id_list Array reference to fill GPU group
 *  ids in the system.
 *
 *  @param[out] count Number of GPU group returned in group_id_list.
 *
 *  @retval ::RDC_ST_OK is returned upon successful call.
 */
rdc_status_t rdc_group_get_all_ids(rdc_handle_t p_rdc_handle,
            rdc_gpu_group_t group_id_list[], uint32_t* count);

/**
 *  @brief Destroy GPU group represented by p_rdc_group_id
 *
 *  @details Delete the logic group represented by p_rdc_group_id
 *
 *  @param[in] p_rdc_handle The RDC handler.
 *
 *  @param[in] p_rdc_group_id The group id
 *
 *  @retval ::RDC_ST_OK is returned upon successful call.
 */
rdc_status_t rdc_group_gpu_destroy(rdc_handle_t p_rdc_handle,
        rdc_gpu_group_t p_rdc_group_id);

/**
 *  @brief create a group of fields
 *
 *  @details The user can create a group of fields and perform an operation
 *  on a group of fields at once.
 *
 *  @param[in] p_rdc_handle The RDC handler.
 *
 *  @param[in] num_field_ids Number of field IDs that are being provided
 *  in field_ids.
 *
 *  @param[in] field_ids Field IDs to be added to the newly-created
 *  field group.
 *
 *  @param[in] field_group_name Unique name for this group of fields.
 *
 *  @param[out] rdc_field_group_id Handle to the newly-created field group
 *
 *  @retval ::RDC_ST_OK is returned upon successful call.
 */
rdc_status_t rdc_group_field_create(rdc_handle_t p_rdc_handle,
            uint32_t num_field_ids, rdc_field_t* field_ids,
            const char* field_group_name, rdc_field_grp_t* rdc_field_group_id);

/**
 *  @brief Get information about a field group
 *
 *  @details Get detail information about a field group created by
 *  rdc_group_field_create
 *
 *  @param[in] p_rdc_handle The RDC handler.
 *
 *  @param[in] rdc_field_group_id The field group handler created by
 *  rdc_group_field_create
 *
 *  @param[out] field_group_info The information of the field group
 *  rdc_field_group_id.
 *
 *  @retval ::RDC_ST_OK is returned upon successful call.
 */
rdc_status_t rdc_group_field_get_info(rdc_handle_t p_rdc_handle,
            rdc_field_grp_t rdc_field_group_id,
            rdc_field_group_info_t* field_group_info);

/**
 *  @brief Used to get information about all field groups in the system.
 *
 *  @details Get the list of field group ids in the system.
 *
 *  @param[in] p_rdc_handle The RDC handler.
 *
 *  @param[out] field_group_id_list Array reference to fill field group
 *  ids in the system.
 *
 *  @param[out] count Number of field group returned in field_group_id_list.
 *
 *  @retval ::RDC_ST_OK is returned upon successful call.
 */
rdc_status_t rdc_group_field_get_all_ids(rdc_handle_t p_rdc_handle,
            rdc_field_grp_t field_group_id_list[], uint32_t* count);

/**
 *  @brief Destroy field group represented by rdc_field_group_id
 *
 *  @details Delete the logic group represented by rdc_field_group_id
 *
 *  @param[in] p_rdc_handle The RDC handler.
 *
 *  @param[in] rdc_field_group_id The field group id
 *
 *  @retval ::RDC_ST_OK is returned upon successful call.
 */
rdc_status_t rdc_group_field_destroy(rdc_handle_t p_rdc_handle,
        rdc_field_grp_t rdc_field_group_id);

/**
 *  @brief Request the RDC start recording updates for a given field
 *  collection.
 *
 *  @details Note that the first update of the field will not occur
 *  until the next field update cycle. To force a field update cycle,
 *  user must call rdc_field_update_all(1)
 *
 *  @param[in] p_rdc_handle The RDC handler.
 *
 *  @param[in] group_id The group of GPUs to be watched.
 *
 *  @param[in] field_group_id  The collection of fields to record
 *
 *  @param[in] update_freq  How often to update fields in usec.
 *
 *  @param[in] max_keep_age How long to keep data for fields in seconds.
 *
 *  @param[in] max_keep_samples Maximum number of samples to keep. 0=no limit.
 *
 *  @retval ::RDC_ST_OK is returned upon successful call.
 */
rdc_status_t rdc_field_watch(rdc_handle_t p_rdc_handle,
        rdc_gpu_group_t group_id, rdc_field_grp_t field_group_id,
        uint64_t update_freq, double max_keep_age, uint32_t max_keep_samples);

/**
 *  @brief Request a latest cached field of a GPU
 *
 *  @details Note that the field can be cached after called rdc_field_watch
 *
 *  @param[in] p_rdc_handle The RDC handler.
 *
 *  @param[in] gpu_index The GPU index.
 *
 *  @param[in] field  The field id
 *
 *  @param[out] value  The field value got from cache.
 *
 *  @retval ::RDC_ST_OK is returned upon successful call.
 */
rdc_status_t rdc_field_get_latest_value(rdc_handle_t p_rdc_handle,
        uint32_t gpu_index, rdc_field_t field, rdc_field_value* value);

/**
 *  @brief Request a history cached field of a GPU
 *
 *  @details Note that the field can be cached after called rdc_field_watch
 *
 *  @param[in] p_rdc_handle The RDC handler.
 *
 *  @param[in] gpu_index The GPU index.
 *
 *  @param[in] field  The field id
 *
 *  @param[in] since_time_stamp  Timestamp to request values since in
 *  usec since 1970.
 *
 *  @param[out] next_since_time_stamp Timestamp to use for sinceTimestamp
 *  on next call to this function
 *
 *  @param[out] value  The field value got from cache.
 *
 *  @retval ::RDC_ST_OK is returned upon successful call.
 */
rdc_status_t rdc_field_get_value_since(rdc_handle_t p_rdc_handle,
        uint32_t gpu_index, rdc_field_t field, uint64_t since_time_stamp,
        uint64_t *next_since_time_stamp, rdc_field_value* value);

/**
 *  @brief Stop record updates for a given field collection.
 *
 *  @details The cache of those fields will not be updated after this call
 *
 *  @param[in] p_rdc_handle The RDC handler.
 *
 *  @param[in] group_id The GPU group id.
 *
 *  @param[in] field_group_id  The field group id.
 *
 *  @retval ::RDC_ST_OK is returned upon successful call.
 */
rdc_status_t rdc_field_unwatch(rdc_handle_t p_rdc_handle,
        rdc_gpu_group_t group_id, rdc_field_grp_t field_group_id);

/**
 *  @brief Get a description of a provided RDC error status
 *
 *  @details return the string in human readable format.
 *
 *  @param[in] status The RDC status.
 *
 *  @retval The string to describe the RDC status.
 */
const char* rdc_status_string(rdc_status_t status);

/**
 *  @brief Get the name of a field
 *
 *  @details return the string in human readable format.
 *
 *  @param[in] field_id The field id.
 *
 *  @retval The string to describe the field.
 */
const char* field_id_string(rdc_field_t field_id);

#ifdef __cplusplus
}
#endif  // __cplusplus

#endif  // INCLUDE_RDC_RDC_H_
