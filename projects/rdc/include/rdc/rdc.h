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

#ifndef RDC_RDC_H_
#define RDC_RDC_H_

#include <cstdint>

/** \file rdc_lib.h
 *  Main header file for the ROCm RDC library.
 *  All required function, structure, enum, etc. definitions should be defined
 *  in this file.
 *
 *  @brief The rocm_rdc library api is new, and therefore subject to change
 *  either at the ABI or API level. Instead of marking every function prototype as "unstable",
 *  we areinstead saying the API is unstable (i.e., changes are possible) while the
 *  major version remains 0. This means that if the API/ABI changes, we will
 *  not increment the major version to 1. Once the ABI stabilizes, we will
 *  increment the major version to 1, and thereafter increment it on all ABI
 *  breaks.
 */

/**
 * @brief Error codes retured by rocm_rdc_lib functions
 */
typedef enum {
     RDC_ST_OK                  = 0,
     RDC_ST_NOT_SUPPORTED,           //!< Not supported feature
     RDC_ST_MSI_ERROR,               //!< The MSI library error
     RDC_ST_FAIL_LOAD_MODULE,        //!< Fail to load the library
     RDC_ST_INVALID_HANDLER          //!< Fail to load the library
} rdc_status_t;

/**
 * @brief rdc operation mode
 * rdc can run in auto mode where background threads will collect metrics.
 * When run in manual mode, the user needs to periodically call rdc_update_all_fields
 * for data collection.
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


#define GPU_ID_INVALID          -1
#define RDC_GROUP_ALL_GPUS      -1000
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
 * @brief The max fields in a field group
 */
#define RDC_MAX_FIELD_IDS_PER_FIELD_GROUP 128


/**
 * @biref The fields
 */

/**
 * Memory usage of the GPU instance
 */
#define RDC_FI_GPU_MEMORY_USAGE             525

/**
 * Total memory of the GPU instance
 */
#define RDC_FI_GPU_MEMORY_TOTAL             580

/**
 * Power usage for the device
 */
#define RDC_FI_POWER_USAGE                  155

/**
 * SM clock for the GPU
 */
#define RDC_FI_GPU_SM_CLOCK                 100

/**
 * GPU Utilization
 */
#define RDC_FI_GPU_UTIL                     203

/**
 * Current temperature for the device
 */
#define RDC_FI_GPU_TEMP                     150

/**
 * @brief handlers used in various rdc calls
 */
typedef void *rdc_handle_t;
typedef uint32_t rdc_gpu_group_t;
typedef uint32_t rdc_field_grp_t;

/**
 * @brief The structure to store summary of data
 */
typedef struct {
    uint64_t max_value;
    uint64_t min_value;
    uint64_t average;
} rdc_stats_summary_t;

/**
 * @brief The structure to hold the GPU usage information
 */
typedef struct {
    uint32_t gpu_id;            //!< GPU_ID_INVALID for summary information
    uint64_t start_time;    //!< The time to start the watching
    uint64_t end_time;      //!< The time to stop the watching

    uint64_t energy_consumed;
    rdc_stats_summary_t power_usage;
    rdc_stats_summary_t gpu_clock;
    rdc_stats_summary_t gpu_utilization;

    uint64_t max_gpu_memory_used;
    rdc_stats_summary_t memory_utilization;
} rdc_gpu_usage_info_t;

/**
 * @brief The structure to hold the job stats
 */
typedef struct {
     uint32_t num_gpus;
     rdc_gpu_usage_info_t summary;
     rdc_gpu_usage_info_t gpus[16];
} rdc_job_info_t;

/**
 * @brief The structure to store the field value
 */
typedef struct {
    uint16_t     field_id;      //!< The field id of the value
    int     status;             //!< RDC_ST_OK or error status
    uint64_t ts;                 //!< Timestamp in usec since 1970
    rdc_field_type_t type;      //!< The field type
    union {
        int64_t l_int;
        double  dbl;
        char str[RDC_MAX_STR_LENGTH];
    } value;
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
    uint32_t field_ids[RDC_MAX_FIELD_IDS_PER_FIELD_GROUP];
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
 *  daemon. In this mode, the user has to periodically call rdc_update_all_fields()
 *  when op_mode is RDC_OPERATION_MODE_MANUAL, which tells RDC to collect
 *  the stats. This function is not thread safe.
 *
 *  @param[in] op_mode Operation modes. When RDC_OPERATION_MODE_AUTO, RDC schedules
 *  background task to collect the stats. When RDC_OPERATION_MODE_MANUAL, the user
 *  needs to call rdc_update_all_fields() periodically.
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
 *  @details Stop the embedded RDC agent, and p_rdc_handle becomes invalid after
 *  this call. This function is not thread safe.
 *
 *  @param[in] p_rdc_handle The RDC handler that come from rdc_start_embedded().
 *  @retval ::RDC_ST_OK is returned upon successful call.
 */
rdc_status_t rdc_stop_embedded(rdc_handle_t p_rdc_handle);

/**
 *  @brief Connect to rdcd daemon
 *
 *  @details This method is used to connect to a remote stand-alone rdcd daemon.
 *  This function is not thread safe.
 *
 *  @param[in] ipAndPort The IP and port of the remote rdcd. The ipAndPort can be
 *  specified in this x.x.x.x:yyyy format, where x.x.x.x is the IP address and
 *  yyyy is the port.
 *
 *  @param[inout] p_rdc_handle Caller provided pointer to rdc_handle_t. Upon
 *  successful call, the value will contain the handler for following API calls.
 *
 *  @retval ::RDC_ST_OK is returned upon successful call.
 */
rdc_status_t rdc_connect(const char *ipAndPort, rdc_handle_t* p_rdc_handle);

/**
 *  @brief Disconnect from rdcd daemon.
 *
 *  @details Disconnect from rdcd daemon, and p_rdc_handle becomes invalid after
 *  this call. This function is not thread safe.
 *
 *  @param[in] p_rdc_handle The RDC handler that come from rdc_connect().
 *
 *  @retval ::RDC_ST_OK is returned upon successful call.
 */
rdc_status_t rdc_disconnect(rdc_handle_t p_rdc_handle);

/**
 *  @brief Create a group contains multiple GPUs
 *
 *  @details This method can create a group contains multiple GPUs. Instead of
 *  executing an operation separately for each GPU, the RDC group enables
 *  the user to execute same operation on all the GPUs present in the group as a
 *  single API call.
 *
 *  @param[in] p_rdc_handle The RDC handler.
 *
 *  @param[in] type The type of the group. RDC_GROUP_DEFAULT includes all the GPUs
 *  on the node, and RDC_GROUP_EMPTY creates an empty group.
 *
 *  @param[in] group_name The group name specified as NULL terminated C String
 *
 *  @param[inout] p_rdc_group_id Caller provided pointer to rdc_gpu_group_t. Upon
 *  successful call, the value will contain the group id for following group API calls.
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
 *  @param[in] p_rdc_handle The The RDC handler.
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
 *  @brief Request the RDC to watch the job stats
 *
 *  @details The summary job stats can be retrieved using rdc_job_get_stats()
 *  In RDC_OPERATION_MODE_MANUAL, user must call rdc_update_all_fields(1)
 *  at least once, before call rdc_job_get_stats()
 *
 *  @param[in] p_rdc_handle The The RDC handler.
 *
 *  @param[in] group_id The group of GPUs to be watched.
 *
 *  @param[in] update_freq  How often to update this field in usec.
 *
 *  @param[in] max_keep_age How long to keep data for this field in seconds.
 *
 *  @param[in] max_keep_samples Maximum number of samples to keep. 0=no limit.
 *
 *  @retval ::RDC_ST_OK is returned upon successful call.
 */
rdc_status_t rdc_watch_job_fields(rdc_handle_t p_rdc_handle,
            rdc_gpu_group_t group_id,    uint64_t update_freq,
            double  max_keep_age, uint32_t  max_keep_samples);

/**
 *  @brief Request RDC a job to be started
 *
 *  @details This should be execute as part of job prologue
 *
 *  @param[in] p_rdc_handle The The RDC handler.
 *
 *  @param[in] job_id The name of the job.
 *
 *  @retval ::RDC_ST_OK is returned upon successful call.
 */
rdc_status_t rdc_job_start_stats(rdc_handle_t p_rdc_handle,
    rdc_gpu_group_t group_id, char  job_id[64]);

/**
 *  @brief Get the stats of the job using the job id.
 *
 *  @details The stats can be retrieved at any point when the job is in process.
 *
 *  @param[in] p_rdc_handle The The RDC handler.
 *
 *  @param[in] job_id The name of the job.
 *
 *  @param[inout] p_job_info Caller provided pointer to rdc_job_info_t. Upon
 *  successful call, the value will contain the stats of the job.
 *
 *  @retval ::RDC_ST_OK is returned upon successful call.
 */
rdc_status_t rdc_job_get_stats(rdc_handle_t p_rdc_handle, char  job_id[64],
                rdc_job_info_t* p_job_info);

/**
 *  @brief Request RDC to stop watching the stats of the job
 *
 *  @details This should be execute as part of job epilogue. The job Id remains
 *  available to view the stats at any point. You must call rdc_watch_job_fields()
 *  before this call.
 *
 *  @param[in] p_rdc_handle The The RDC handler.
 *
 *  @param[in] job_id The name of the job.
 *
 *  @retval ::RDC_ST_OK is returned upon successful call.
 */
rdc_status_t rdc_job_stop_stats(rdc_handle_t p_rdc_handle,
            char  job_id[64]);

/**
 *  @brief Request RDC to update all fields to be watched.
 *
 *  @details In RDC_OPERATION_MODE_MANUAL, the user must call this method periodically.
 *
 *  @param[in] p_rdc_handle The The RDC handler.
 *
 *  @param[in] wait_for_update Whether or not to wait for the update loop to
 *  complete before returning to the caller 1=wait. 0=do not wait.
 *
 *  @retval ::RDC_ST_OK is returned upon successful call.
 */
rdc_status_t rdc_update_all_fields(rdc_handle_t p_rdc_handle,
                    uint32_t wait_for_update);

/**
 *  @brief Get a description of a provided RDC error status
 *
 *  @details return the string in human readable format.
 *
 *  @param[in] status The The RDC status.
 *
 *  @retval The string to describe the RDC status.
 */
const char* rdc_status_string(rdc_status_t status);

#endif  // RDC_RDC_H_
