/*
Copyright (c) 2019 - Advanced Micro Devices, Inc. All rights reserved.

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

#include <grpcpp/grpcpp.h>

#include <time.h>
#include <unistd.h>
#include <iostream>

#include "rdc/rdc_client_main.h"
#include "rdc/rdc_client.h"
#include "common/rdc_utils.h"
#include "rdc/rdc_exception.h"
#include "rdc.grpc.pb.h"  // NOLINT
#include "rocm_smi/rocm_smi.h"

#define CHK_PTR_ARG(PTR) \
  if ((PTR) == nullptr) { \
    return RDC_RSMI_STATUS_INVALID_ARGS; \
  }

#define UINTPTR_TO_RDC_CHAN(UPTR) \
    amd::rdc::RDCChannel *ch = reinterpret_cast<amd::rdc::RDCChannel *>(UPTR); \
    if (ch == nullptr) { \
      return RDC_STATUS_GRPC_INVALID_ARG; \
    } \

static rdc_status_t handleException() {
  try {
    throw;
  } catch (const std::bad_alloc& e) {
    debug_print("RDC exception: BadAlloc\n");
    return RDC_RSMI_STATUS_OUT_OF_RESOURCES;
  } catch (const amd::rdc::rdc_exception& e) {
    debug_print("Exception caught: %s.\n", e.what());
    return e.error_code();
    return RDC_RSMI_STATUS_INTERNAL_EXCEPTION;
  } catch (const std::exception& e) {
    debug_print("Unhandled exception: %s\n", e.what());
    assert(false && "Unhandled exception.");
    return RDC_RSMI_STATUS_INTERNAL_EXCEPTION;
  } catch (const std::nested_exception& e) {
    debug_print("Callback threw, forwarding.\n");
    e.rethrow_nested();
    return RDC_RSMI_STATUS_INTERNAL_EXCEPTION;
  } catch (...) {
    assert(false && "Unhandled exception.");
    abort();
    return RDC_RSMI_STATUS_INTERNAL_EXCEPTION;
  }
}

#define TRY try {
#define CATCH } catch (...) {return handleException();}

rdc_status_t
rdc_channel_create(rdc_channel_t *channel, const char *ip,
                                              const char *port, bool secure) {
  TRY
  std::string server_str;
  std::string port_str;

  if (channel == nullptr) {
    return RDC_STATUS_GRPC_INVALID_ARG;
  }
  if (ip != nullptr) {
    server_str = ip;
  } else {
    server_str = RDC_DEFAULT_SERVER_IP;
  }

  if (port != nullptr) {
    port_str = port;
  } else {
    port_str = std::to_string(RDC_DEFAULT_SERVER_PORT);
  }

  amd::rdc::RDCChannel *ch =
                       new amd::rdc::RDCChannel(server_str, port_str, secure);

  if (ch == nullptr) {
    return RDC_STATUS_GRPC_RESOURCE_EXHAUSTED;
  }

  rdc_status_t ret = ch->Initialize();

  if (ret != 0) {
    delete ch;
    return ret;
  }

  *channel = reinterpret_cast<rdc_channel_t>(ch);

  return RDC_STATUS_SUCCESS;

  CATCH
}
rdc_status_t
rdc_channel_state_get(rdc_channel_t channel, bool try_to_connect,
                                             grpc_connectivity_state *state) {
  TRY
  CHK_PTR_ARG(state)
  UINTPTR_TO_RDC_CHAN(channel)

  *state = ch->channel()->GetState(try_to_connect);
  return RDC_STATUS_SUCCESS;

  CATCH
}

rdc_status_t
rdc_channel_connection_verify(rdc_channel_t channel) {
  TRY
  UINTPTR_TO_RDC_CHAN(channel)

  ::rdc::VerifyConnectionResponse resp;
  ::rdc::VerifyConnectionRequest req;
  ::grpc::ClientContext context;
  unsigned int seed = time(NULL);

  req.set_magic_num(static_cast<uint64_t>(rand_r(&seed)));
  ::grpc::Status status =
                 ch->rdc_admin_stub()->VerifyConnection(&context, req, &resp);

  if (!status.ok()) {
    return amd::rdc::GrpcErrorToRdcError(status.error_code());
  }

  if (resp.echo_magic_num() != req.magic_num()) {
    return RDC_STATUS_GRPC_DATA_LOSS;
  }

  return RDC_STATUS_SUCCESS;

  CATCH
}

rdc_status_t
rdc_channel_destroy(rdc_channel_t channel) {
  TRY
  UINTPTR_TO_RDC_CHAN(channel)

  delete ch;

  return RDC_STATUS_SUCCESS;

  CATCH
}

rdc_status_t
rdc_num_gpus_get(rdc_channel_t channel, uint64_t *num_gpu) {
  TRY
  CHK_PTR_ARG(num_gpu)
  UINTPTR_TO_RDC_CHAN(channel)

  ::rdc::GetNumDevicesResponse resp;
  ::rdc::GetNumDevicesRequest empty;
  ::grpc::ClientContext context;
  ::grpc::Status status =
                       ch->rsmi_stub()->GetNumDevices(&context, empty, &resp);

  if (!status.ok()) {
    return amd::rdc::GrpcErrorToRdcError(status.error_code());
  }

  *num_gpu = resp.val();
  return static_cast<rdc_status_t>(resp.ret_val());

  CATCH
}

// rsmi and rdc currently happen to have a 1-to-1 mapping, but
// have this function in case that changes
static ::rdc::GetTemperatureRequest_TemperatureMetric
    rsmi_temp2rdc_temp(rsmi_temperature_metric_t rsmi_temp) {
  return
       static_cast<::rdc::GetTemperatureRequest_TemperatureMetric>(rsmi_temp);
}

rdc_status_t
rdc_dev_temp_metric_get(rdc_channel_t channel, uint32_t dv_ind,
                  uint32_t sensor_type, rsmi_temperature_metric_t metric,
                                                       int64_t *temperature) {
  TRY
  CHK_PTR_ARG(temperature)
  UINTPTR_TO_RDC_CHAN(channel)

  ::rdc::GetTemperatureResponse resp;
  ::rdc::GetTemperatureRequest in_args;
  ::grpc::ClientContext context;

  in_args.set_metric(rsmi_temp2rdc_temp(metric));
  in_args.set_dv_ind(dv_ind);
  in_args.set_sensor_type(sensor_type);

  ::grpc::Status status =
                    ch->rsmi_stub()->GetTemperature(&context, in_args, &resp);

  if (!status.ok()) {
    return ::amd::rdc::GrpcErrorToRdcError(status.error_code());
  }

  *temperature = resp.temperature();

  return static_cast<rdc_status_t>(resp.ret_val());
  CATCH
}


rdc_status_t
rdc_status_string(rdc_status_t status, const char **status_string) {
  TRY
  if (status_string == nullptr) {
    return RDC_RSMI_STATUS_INVALID_ARGS;
  }

  const size_t status_u = static_cast<size_t>(status);
  switch (status_u) {
    case RDC_STATUS_SUCCESS:
      *status_string = "RDC_STATUS_SUCCESS: The function has been executed"
                          " successfully.";
      break;

    case RDC_RSMI_STATUS_INVALID_ARGS:
      *status_string =
          "RDC_RSMI_STATUS_INVALID_ARGS: The provided arguments do not"
                " meet the preconditions required for calling this function.";
      break;

    case RDC_RSMI_STATUS_NOT_SUPPORTED:
      *status_string = "RDC_RSMI_STATUS_NOT_SUPPORTED: This function is not"
                                     " supported in the current environment.";
      break;

    case RDC_RSMI_STATUS_FILE_ERROR:
      *status_string =
        "RDC_RSMI_STATUS_FILE_ERROR: There was an error in finding or"
        " opening a file or directory. The operation may not be supported by "
        "this Linux kernel version.";
      break;

    case RDC_RSMI_STATUS_PERMISSION:
      *status_string = "RDC_RSMI_STATUS_PERMISSION: The user ID of the calling"
        " process does not have sufficient permission to execute a command."
        "  Often this is fixed by running as root (sudo).";
      break;

    case RDC_RSMI_STATUS_OUT_OF_RESOURCES:
      *status_string = "RDC_RSMI_STATUS_OUT_OF_RESOURCES: Unable to acquire "
          "memory or other resource";
      break;

    case RDC_RSMI_STATUS_INTERNAL_EXCEPTION:
      *status_string = "RDC_RSMI_STATUS_INTERNAL_EXCEPTION: An internal "
          "exception was caught";
      break;

    case RDC_RSMI_STATUS_INPUT_OUT_OF_BOUNDS:
      *status_string = "RDC_RSMI_STATUS_INPUT_OUT_OF_BOUNDS: The provided "
          "input is out of allowable or safe range";
      break;

    case RDC_RSMI_STATUS_INIT_ERROR:
      *status_string = "RDC_RSMI_STATUS_INIT_ERROR: An error occurred during "
          "initialization, during "
       "monitor discovery or when when initializing internal data structures";
      break;

    case RDC_RSMI_STATUS_NOT_YET_IMPLEMENTED:
      *status_string = "RDC_RSMI_STATUS_NOT_YET_IMPLEMENTED: The called "
          "function has not been implemented in this "
        "system for this device type";
      break;

    case RDC_RSMI_STATUS_NOT_FOUND:
      *status_string = "RDC_RSMI_STATUS_NOT_FOUND: An item required to "
          "complete the call was not found";
      break;

    case RDC_RSMI_STATUS_INSUFFICIENT_SIZE:
      *status_string = "RDC_RSMI_STATUS_INSUFFICIENT_SIZE: Not enough "
          "resources were available to fully execute"
                             " the call";
      break;

    case RDC_RSMI_STATUS_UNKNOWN_ERROR:
      *status_string = "An unknown error prevented the call from completing"
                          " successfully";
      break;

    case RDC_RSMI_STATUS_INTERRUPT:
      *status_string = "RDC_RSMI_STATUS_INTERRUPT An interrupt occurred while "
          "executing the function";
      break;

    case RDC_STATUS_GRPC_CANCELLED:
      *status_string =
          "RDC_STATUS_GRPC_CANCELLED The operation was cancelled (typically by "
          "the caller).";
      break;

    case RDC_STATUS_GRPC_UNKNOWN:
      *status_string =
        "RDC_STATUS_GRPC_UNKNOWN Unknown error. An example of where this error"
        " may be returned is if a"
        "Status value received from another address space belongs to an error-"
        "space that is not known in this address space. Also errors raised by "
        "APIs that do not return enough error information may be converted to "
        "this error.";
      break;

    case RDC_STATUS_GRPC_INVALID_ARG:
      *status_string =
        "RDC_STATUS_GRPC_INVALID_ARG Client specified an invalid argument. "
        "Note that this differs from"
        "FAILED_PRECONDITION. INVALID_ARGUMENT indicates arguments that are "
        "problematic regardless of the state of the system (e.g., a malformed "
        "file name).";
      break;

    case RDC_STATUS_GRPC_DEADLINE_EXCEEDED:
      *status_string =
        "RDC_STATUS_GRPC_DEADLINE_EXCEEDED Deadline expired before operation "
        "could complete. For operations that"
        "change the state of the system, this error may be returned even if "
        "the operation has completed successfully. For example, a successful "
        "response from a server could have been delayed long enough for the "
        "deadline to expire.";
      break;

    case RDC_STATUS_GRPC_NOT_FOUND:
      *status_string =
          "RDC_STATUS_GRPC_NOT_FOUND Some requested entity (e.g., file or "
          "directory) was not found.";
      break;

    case RDC_STATUS_GRPC_ALREADY_EXISTS:
      *status_string =
      "RDC_STATUS_GRPC_ALREADY_EXISTS Some entity that we attempted to create "
      "(e.g., file or directory) already exists.";
      break;

    case RDC_STATUS_GRPC_PERM_DENIED:
      *status_string =
        "RDC_STATUS_GRPC_PERM_DENIED The caller does not have permission to "
        "execute the specified operation."
        "PERMISSION_DENIED must not be used for rejections caused by "
        "exhausting some resource (use RESOURCE_EXHAUSTED instead for those "
        "errors). PERMISSION_DENIED must not be used if the caller can not "
        " be identified (use UNAUTHENTICATED instead for those errors).";
      break;

    case RDC_STATUS_GRPC_UNAUTHENTICATED:
      *status_string =
        "RDC_STATUS_GRPC_UNAUTHENTICATED The request does not have valid "
        "authentication credentials for the operation.";
      break;

    case RDC_STATUS_GRPC_RESOURCE_EXHAUSTED:
      *status_string =
        "RDC_STATUS_GRPC_RESOURCE_EXHAUSTED Some resource has been exhausted, "
        "perhaps a per-user quota, or perhaps the "
        "entire file system is out of space.";
      break;

    case RDC_STATUS_GRPC_FAILED_PRECOND:
      *status_string =
        "RDC_STATUS_GRPC_FAILED_PRECOND Operation was rejected because the "
        "system is not in a state required for "
        "the operation's execution. For example, directory to be deleted may "
        "be non-empty, an rmdir operation is applied to a non-directory, etc.\n"
        "A litmus test that may help a service implementor in deciding "
        "between FAILED_PRECONDITION, ABORTED, and UNAVAILABLE:\n"
        " (a) Use UNAVAILABLE if the client can retry just the failing call.\n"
        " (b) Use ABORTED if the client should retry at a higher-level "
        " (e.g., restarting a read-modify-write sequence).\n"
        " (c) Use FAILED_PRECONDITION if the client should not retry until"
        " the system state has been explicitly fixed. E.g., if an \"rmdir\""
        " fails because the directory is non-empty, FAILED_PRECONDITION"
        " should be returned since the client should not retry unless"
        " they have first fixed up the directory by deleting files from it.\n"
        " (d) Use FAILED_PRECONDITION if the client performs conditional"
        " REST Get/Update/Delete on a resource and the resource on the"
        " server does not match the condition. E.g., conflicting"
        " read-modify-write on the same resource.";
      break;

    case RDC_STATUS_GRPC_ABORTED:
      *status_string =
        "RDC_STATUS_GRPC_ABORTED The operation was aborted, "
        "typically due to a concurrency issue like "
        "sequencer check failures, transaction aborts, etc.\n"
        "See litmus test above for deciding between "
        "FAILED_PRECONDITION, ABORTED, "
        "and UNAVAILABLE.";
      break;

    case RDC_STATUS_GRPC_OUT_OF_RANGE:
      *status_string =
        "RDC_STATUS_GRPC_OUT_OF_RANGE Operation was attempted "
        "past the valid range. E.g., seeking or reading "
        "past end of file.\n"
        "Unlike INVALID_ARGUMENT, this error indicates a "
        "problem that may be fixed "
        "if the system state changes. For example, a 32-bit file system will "
        "generate INVALID_ARGUMENT if asked to read "
        "at an offset that is not in the "
        "range [0,2^32-1], but it will generate "
        "OUT_OF_RANGE if asked to read from "
        "an offset past the current file size.\n"
        "There is a fair bit of overlap between FAILED_PRECONDITION and "
        "OUT_OF_RANGE. We recommend using OUT_OF_RANGE "
        "(the more specific error) "
        "when it applies so that callers who are "
        "iterating through a space can "
        "easily look for an OUT_OF_RANGE error to detect when they are done.";
      break;

    case RDC_STATUS_GRPC_UNIMPLEMENTED:
      *status_string =
        "RDC_STATUS_GRPC_UNIMPLEMENTED Operation is not "
        "implemented or not supported/enabled in this service.";
      break;

    case RDC_STATUS_GRPC_INTERNAL:
      *status_string =
        "RDC_STATUS_GRPC_INTERNAL Internal errors. This means "
        "some invariants expected by underlying System has "
        "been broken. If you see one of these errors.";
      break;

    case RDC_STATUS_GRPC_UNAVAILABLE:
      *status_string =
        "RDC_STATUS_GRPC_UNAVAILABLE The service is currently unavailable. "
        "This is a most likely a transient "
        "condition and may be corrected by retrying with a backoff.\n"
        "Warning: Although data MIGHT not have been transmitted when this "
        "status occurs, there is NOT A GUARANTEE that the server has not seen "
        "anything. So in general it is unsafe to retry on this status code "
        "if the call is non-idempotent. "
        "See litmus test above for deciding between "
        "FAILED_PRECONDITION, ABORTED,"
        "and UNAVAILABLE.";
      break;

    case RDC_STATUS_GRPC_DATA_LOSS:
      *status_string =
        "RDC_STATUS_GRPC_DATA_LOSS Unrecoverable data loss or corruption.";
      break;

    case RDC_STATUS_UNKNOWN_ERROR:
      *status_string =
        "RDC_STATUS_UNKNOWN_ERROR An unknown RDC error occurred.";
      break;

    default:
      *status_string = "RDC_RSMI_STATUS_UNKNOWN_ERROR An "
          "unknown error occurred";
      return RDC_RSMI_STATUS_UNKNOWN_ERROR;
  }
  return RDC_STATUS_SUCCESS;
  CATCH
}
