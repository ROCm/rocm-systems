
/*
Copyright (c) 2019 - present Advanced Micro Devices, Inc. All rights reserved.

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

#ifndef CLIENT_INCLUDE_RDC_RDC_CLIENT_H_
#define CLIENT_INCLUDE_RDC_RDC_CLIENT_H_

#include <grpcpp/grpcpp.h>

#include <memory>
#include <string>
#include "rocm_smi/rocm_smi.h"

/**
 * @brief Error codes retured by rdc functions
 */
typedef enum {
  RDC_STATUS_SUCCESS = 0x0,             //!< Operation was successful
  RDC_RSMI_STATUS_INVALID_ARGS,         //!< Passed in arguments are not valid
  RDC_RSMI_STATUS_NOT_SUPPORTED,        //!< The requested information or
                                        //!< action is not available for the
                                        //!< given input, on the given system
  RDC_RSMI_STATUS_FILE_ERROR,           //!< Problem accessing a file. This
                                        //!< may because the operation is not
                                        //!< supported by the Linux kernel
                                        //!< version running on the executing
                                        //!< machine
  RDC_RSMI_STATUS_PERMISSION,           //!< Permission denied/EACCESS file
                                        //!< error. Many functions require
                                        //!< root access to run.
  RDC_RSMI_STATUS_OUT_OF_RESOURCES,     //!< Unable to acquire memory or other
                                        //!< resource
  RDC_RSMI_STATUS_INTERNAL_EXCEPTION,   //!< An internal exception was caught
  RDC_RSMI_STATUS_INPUT_OUT_OF_BOUNDS,  //!< The provided input is out of
                                        //!< allowable or safe range
  RDC_RSMI_STATUS_INIT_ERROR,           //!< An error occurred when creating
                                        //!< a communications channel
  RDC_RSMI_STATUS_NOT_YET_IMPLEMENTED,  //!< The requested function has not
                                        //!< yet been implemented in the
                                        //!< current system for the current
                                        //!< devices
  RDC_RSMI_STATUS_NOT_FOUND,            //!< An item was searched for but not
                                        //!< found
  RDC_RSMI_STATUS_INSUFFICIENT_SIZE,    //!< Not enough resources were
                                        //!< available for the operation
  RDC_RSMI_STATUS_INTERRUPT,            //!< An interrupt occurred during
                                        //!< execution of function
  RDC_RSMI_STATUS_UNEXPECTED_SIZE,      //!< An unexpected amount of data
                                        //!< was read
  RDC_RSMI_STATUS_NO_DATA,              //!< No data was found for a given
                                        //!< input
  RDC_RSMI_STATUS_UNKNOWN_ERROR,        //!< An unknown error occurred
  RDC_STATUS_GRPC_ERR_FIRST = 1000,

  /// Not an error; returned on success.
  RDC_STATUS_GRPC_OK = RDC_STATUS_GRPC_ERR_FIRST,

  /// The operation was cancelled (typically by the caller).
  RDC_STATUS_GRPC_CANCELLED,

  /// Unknown error. An example of where this error may be returned is if a
  /// Status value received from another address space belongs to an error-space
  /// that is not known in this address space. Also errors raised by APIs that
  /// do not return enough error information may be converted to this error.
  RDC_STATUS_GRPC_UNKNOWN,

  /// Client specified an invalid argument. Note that this differs from
  /// FAILED_PRECONDITION. INVALID_ARGUMENT indicates arguments that are
  /// problematic regardless of the state of the system (e.g., a malformed file
  /// name).
  RDC_STATUS_GRPC_INVALID_ARG,

  /// Deadline expired before operation could complete. For operations that
  /// change the state of the system, this error may be returned even if the
  /// operation has completed successfully. For example, a successful response
  /// from a server could have been delayed long enough for the deadline to
  /// expire.
  RDC_STATUS_GRPC_DEADLINE_EXCEEDED,

  /// Some requested entity (e.g., file or directory) was not found.
  RDC_STATUS_GRPC_NOT_FOUND,

  /// Some entity that we attempted to create (e.g., file or directory) already
  /// exists.
  RDC_STATUS_GRPC_ALREADY_EXISTS,

  /// The caller does not have permission to execute the specified operation.
  /// PERMISSION_DENIED must not be used for rejections caused by exhausting
  /// some resource (use RESOURCE_EXHAUSTED instead for those errors).
  /// PERMISSION_DENIED must not be used if the caller can not be identified
  /// (use UNAUTHENTICATED instead for those errors).
  RDC_STATUS_GRPC_PERM_DENIED,

  /// The request does not have valid authentication credentials for the
  /// operation.
  RDC_STATUS_GRPC_UNAUTHENTICATED,

  /// Some resource has been exhausted, perhaps a per-user quota, or perhaps the
  /// entire file system is out of space.
  RDC_STATUS_GRPC_RESOURCE_EXHAUSTED,

  /// Operation was rejected because the system is not in a state required for
  /// the operation's execution. For example, directory to be deleted may be
  /// non-empty, an rmdir operation is applied to a non-directory, etc.
  ///
  /// A litmus test that may help a service implementor in deciding
  /// between FAILED_PRECONDITION, ABORTED, and UNAVAILABLE:
  ///  (a) Use UNAVAILABLE if the client can retry just the failing call.
  ///  (b) Use ABORTED if the client should retry at a higher-level
  ///      (e.g., restarting a read-modify-write sequence).
  ///  (c) Use FAILED_PRECONDITION if the client should not retry until
  ///      the system state has been explicitly fixed. E.g., if an "rmdir"
  ///      fails because the directory is non-empty, FAILED_PRECONDITION
  ///      should be returned since the client should not retry unless
  ///      they have first fixed up the directory by deleting files from it.
  ///  (d) Use FAILED_PRECONDITION if the client performs conditional
  ///      REST Get/Update/Delete on a resource and the resource on the
  ///      server does not match the condition. E.g., conflicting
  ///      read-modify-write on the same resource.
  RDC_STATUS_GRPC_FAILED_PRECOND,

  /// The operation was aborted, typically due to a concurrency issue like
  /// sequencer check failures, transaction aborts, etc.
  ///
  /// See litmus test above for deciding between FAILED_PRECONDITION, ABORTED,
  /// and UNAVAILABLE.
  RDC_STATUS_GRPC_ABORTED,

  /// Operation was attempted past the valid range. E.g., seeking or reading
  /// past end of file.
  ///
  /// Unlike INVALID_ARGUMENT, this error indicates a problem that may be fixed
  /// if the system state changes. For example, a 32-bit file system will
  /// generate INVALID_ARGUMENT if asked to read at an offset that is not in the
  /// range [0,2^32-1], but it will generate OUT_OF_RANGE if asked to read from
  /// an offset past the current file size.
  ///
  /// There is a fair bit of overlap between FAILED_PRECONDITION and
  /// OUT_OF_RANGE. We recommend using OUT_OF_RANGE (the more specific error)
  /// when it applies so that callers who are iterating through a space can
  /// easily look for an OUT_OF_RANGE error to detect when they are done.
  RDC_STATUS_GRPC_OUT_OF_RANGE,

  /// Operation is not implemented or not supported/enabled in this service.
  RDC_STATUS_GRPC_UNIMPLEMENTED,

  /// Internal errors. Means some invariants expected by underlying System has
  /// been broken. If you see one of these errors, Something is very broken.
  RDC_STATUS_GRPC_INTERNAL,

  /// The service is currently unavailable. This is a most likely a transient
  /// condition and may be corrected by retrying with a backoff.
  ///
  /// \warning Although data MIGHT not have been transmitted when this
  /// status occurs, there is NOT A GUARANTEE that the server has not seen
  /// anything. So in general it is unsafe to retry on this status code
  /// if the call is non-idempotent.
  ///
  /// See litmus test above for deciding between FAILED_PRECONDITION, ABORTED,
  /// and UNAVAILABLE.
  RDC_STATUS_GRPC_UNAVAILABLE,

  /// Unrecoverable data loss or corruption.
  RDC_STATUS_GRPC_DATA_LOSS,

  RDC_STATUS_UNKNOWN_ERROR = 0xFFFFFFFF,  //!< An unknown error occurred
} rdc_status_t;


/**
 * @brief Handle to RDC server channel
 */
typedef uintptr_t rdc_channel_t;

#define RDC_DEFAULT_SERVER_PORT 50051
#define RDC_DEFAULT_SERVER_IP "localhost"

/*****************************************************************************/
/** @defgroup RDCAdmin RDC Administration Functions
 *  These administrative functions are used to monitor and control, for
 *  example RDC connectivity.
 *  @{
 */

/**
 * @brief Check the connection status of a channel
 *
 * @details Given an ::rdc_channel_t @p channel and a boolean @p
 * try_to_connect, this function will return the grpc_connectivity_state for
 * that channel
 *
 * @p channel[in] The channel for which the status will be given
 *
 * @param[in] try_to_connect If the channel is currently IDLE, if the argument
 * is true, transition to CONNECTING.
 *
 * @param[inout] state A pointer to caller provided memory to which an
 * the grpc_connectivity_state will be written. grpc_connectivity_state has
 * the following possible values:
 * GRPC_CHANNEL_IDLE               channel is idle
 * GRPC_CHANNEL_CONNECTING         channel is connecting
 * GRPC_CHANNEL_READY              channel is ready for work
 * GRPC_CHANNEL_TRANSIENT_FAILURE  channel has seen a failure but expects to
 * recover
 * GRPC_CHANNEL_SHUTDOWN           channel has seen a failure that it cannot
 * recover from
 *
 * @retval ::RDC_STATUS_SUCCESS is returned upon successful call.
 *
 */
rdc_status_t
rdc_channel_state_get(rdc_channel_t channel, bool try_to_connect,
                                              grpc_connectivity_state *state);


/**
 * @brief Verify a channel's connection to the server
 *
 * @details Given an ::rdc_channel_t @p channel, this function will send a
 * random number to the server associated with @p channel. The server will send
 * the number back. Upon receiving the returned message from the server, the
 * number sent to the server is compared to the number received from the
 * server. If the 2 numbers are the same, the connection is verified.
 * Otherwise, an appropriate error code is returned.
 *
 * @p channel[in] The channel for which the connection will be verified
 *
 * @retval ::RDC_STATUS_SUCCESS is returned upon successful call.
 *
 */
rdc_status_t
rdc_channel_connection_verify(rdc_channel_t channel);

/** @} */  // end of RDCAdmin

/*****************************************************************************/
/** @defgroup InitShutAdmin Initialization and Shutdown
 *  These functions are used for initialization of RDC and clean up when
 *  done.
 *  @{
 */

/**
 * @brief Create a communications channel to an RDC server
 *
 * @details Given a pointer to an ::rdc_channel_t @p channel, a string 
 * containing the ip address of the server @p ip, a string containing
 * the port number on which the server is listening @p port and a bool
 * indicating whether the channel should use a secure link @p secure,
 * this function will attempt to create a new channel and write its
 * location to address pointed to by @p channel.
 *
 * @p channel[inout] A pointer to caller provided memory to which an
 * ::rdc_channel_t will be written
 *
 * @param[in] ip A pointer to a string containing the address of the server
 *
 * @param[in] port A pointer to string containing the port on which the
 * RDC server is listening
 *
 * @param[in] secure A bool indicating whether SSL should be used for
 * communications (not currently supported)
 *
 * @retval ::RDC_STATUS_SUCCESS is returned upon successful call.
 *
 */
rdc_status_t
rdc_channel_create(rdc_channel_t *channel, const char *ip, const char *port,
                                                                 bool secure);

/**
 * @brief Destroy a communications channel to an RDC server
 *
 * @details Given an ::rdc_channel_t @p channel, this function will free any
 * resources used by @p channel
 *
 * @p channel[inout] An ::rdc_channel_t will be freed
 *
 * @retval ::RDC_STATUS_SUCCESS is returned upon successful call.
 *
 */
rdc_status_t
rdc_channel_destroy(rdc_channel_t channel);

/** @} */  // end of InitShutAdmin

/*****************************************************************************/
/** @defgroup RSMIAccess Remote ROCm SMI Calls 
 *  These functions calls make ROCm SMI function calls on the remote server.
 *  Please refer to the
 *  [ROCm SMI documentation]
 *  (https://github.com/RadeonOpenCompute/rocm_smi_lib/tree/master/docs) for
 *  information about the calls. Here, we will document any additional aspects
 *  of the calls introduced by RDC that are not covered in the ROCm SMI
 *  documentation.
 *  
 *  All of the functions in this section attempt to make an RSMI call on the
 *  server machine, given an ::rdc_channel_t associated with the server, and
 *  all the arguments that are required to make the RSMI call. 
 *  @{
 */

/**
 * @brief Remote call to rsmi_num_monitor_devices()
 *
 */
rdc_status_t
rdc_num_gpus_get(rdc_channel_t channel, uint64_t *num_gpu);

/** @} */  // end of RSMIAccess


/** @defgroup PhysQuer Physical State Queries
 *  These functions provide information about the physical characteristics of
 *  the device.
 *  @{
 */
/**
 * @brief Remote call to rsmi_dev_temp_metric_get()
 *
 */
rdc_status_t
rdc_dev_temp_metric_get(rdc_channel_t channel, uint32_t dv_ind,
                      uint32_t sensor_type, rsmi_temperature_metric_t metric,
                                                         int64_t *temperature);
/** @} */  // end of PhysQuer

/**
 *  @brief Get a description of a provided RDC error status
 *
 *  @details Set the provided pointer to a const char *, @p status_string, to
 *  a string containing a description of the provided error code @p status.
 *
 *  @param[in] status The error status for which a description is desired
 *
 *  @param[inout] status_string A pointer to a const char * which will be made
 *  to point to a description of the provided error code
 *
 *  @retval ::RSMI_STATUS_SUCCESS is returned upon successful call
 *
 */
rdc_status_t
rdc_status_string(rdc_status_t status, const char **status_string);

#endif  // CLIENT_INCLUDE_RDC_RDC_CLIENT_H_
