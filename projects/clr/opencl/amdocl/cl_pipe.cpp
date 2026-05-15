/* Copyright (c) 2013 - 2021 Advanced Micro Devices, Inc.

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
 FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
 THE SOFTWARE. */

#include "cl_common.hpp"
#include "platform/memory.hpp"
#include "platform/context.hpp"
#include "platform/command.hpp"

/*! \addtogroup API
 *  @{
 *
 *  \addtogroup CL_Pipes
 *  @{
 */

/*! \brief creates a pipe object.
 *
 * \param context is a valid OpenCL context used to create the pipe object.
 *
 * \param flags is a bit-field that is used to specify allocation and usage
 * information such as the memory arena that should be used to allocate the pipe
 * object and how it will be used. Only CL_MEM_READ_WRITE and CL_MEM_HOST_NO_ACCESS
 * can be specified when creating a pipe object. If value specified for flags is 0,
 * the default is used which is CL_MEM_READ_WRITE | CL_MEM_HOST_NO_ACCESS.
 *
 * \param pipe_packet_size is the size in bytes of a pipe packet.
 *
 * \param pipe_max_packets specifies the pipe capacity by specifying the maximum
 * number of packets the pipe can hold.
 *
 * \param properties specifies a list of properties for the pipe and their
 * corresponding values. Each property name is immediately followed by the
 * corresponding desired value. The list is terminated with 0.
 *
 * In OpenCL 2.0, properties must be NULL.
 *
 * \param errcode_ret will return an appropriate error code.
 * If \a errcode_ret is NULL, no error code is returned.
 *
 * \return a valid non-zero pipe object and \a errcode_ret is set to CL_SUCCESS
 * if the pipe object is created successfully. Otherwise, it returns a NULL
 * value with one of the following error values returned in errcode_ret:
 * - CL_INVALID_CONTEXT if context is not a valid context.
 * - CL_INVALID_VALUE if values specified in flags are not as defined above.
 * - CL_INVALID_VALUE if properties is not NULL.
 * - CL_INVALID_PIPE_SIZE if pipe_packet_size is 0 or the pipe_packet_size
 *   exceeds CL_DEVICE_PIPE_MAX_PACKET_SIZE value for all devices in context
 *   or if pipe_max_packets is 0.
 * - CL_MEM_OBJECT_ALLOCATION_FAILURE if there is a failure to allocate memory
 *   for the pipe object.
 * - CL_OUT_OF_RESOURCES if there is a failure to allocate resources required
 *   by the OpenCL implementation on the device.
 * - CL_OUT_OF_HOST_MEMORY if there is a failure to allocate resources required
 *   by the OpenCL implementation on the host.
 *
 * \version 2.0r29
 */
RUNTIME_ENTRY_RET(cl_mem, clCreatePipe,
                  (cl_context context, cl_mem_flags flags, cl_uint pipe_packet_size,
                   cl_uint pipe_max_packets, const cl_pipe_properties* properties,
                   cl_int* errcode_ret)) {
  if (!is_valid(context)) {
    *not_null(errcode_ret) = CL_INVALID_CONTEXT;
    return nullptr;
  }

  const cl_mem_flags validFlags = (CL_MEM_READ_WRITE | CL_MEM_HOST_NO_ACCESS);

  if (flags == 0) {
    flags = validFlags;
  }

  // check flags for validity
  cl_bitfield temp = flags & ~validFlags;

  if (temp) {
    *not_null(errcode_ret) = CL_INVALID_VALUE;
    LogWarning("invalid parameter \"flags\"");
    return nullptr;
  }

  size_t size = sizeof(struct clk_pipe_t) + pipe_packet_size * pipe_max_packets;

  const std::vector<amd::Device*>& devices = as_amd(context)->devices();
  bool sizePass = false;
  for (const auto& it : devices) {
    if (it->info().maxMemAllocSize_ >= size) {
      sizePass = true;
      break;
    }
  }

  // check size
  if (pipe_packet_size == 0 || pipe_max_packets == 0 || !sizePass) {
    *not_null(errcode_ret) = CL_INVALID_PIPE_SIZE;
    LogWarning("invalid parameter \"size = 0 or size > CL_DEVICE_PIPE_MAX_PACKET_SIZE\"");
    return nullptr;
  }

  amd::Context& amdContext = *as_amd(context);
  amd::Memory* mem = new (amdContext)
      amd::Pipe(amdContext, flags, size, (size_t)pipe_packet_size, (size_t)pipe_max_packets);
  if (mem == NULL) {
    *not_null(errcode_ret) = CL_OUT_OF_HOST_MEMORY;
    return nullptr;
  }

  if (!mem->create()) {
    *not_null(errcode_ret) = CL_MEM_OBJECT_ALLOCATION_FAILURE;
    mem->release();
    return nullptr;
  }

  *not_null(errcode_ret) = CL_SUCCESS;
  return as_cl(mem);
}
RUNTIME_EXIT

/*! \brief Get information specific to a pipe object created with clCreatePipe.
 *
 * \param param_name specifies the information to query.
 *
 * \param param_value is a pointer to memory where the appropriate result being
 * queried is returned. If param_value is NULL, it is ignored.
 *
 * \param param_value_size is used to specify the size in bytes of memory
 * pointed to by param_value. This size must be >= size of return type.
 *
 * \param param_value_size_ret returns the actual size in bytes of data being
 * queried by param_value. If param_value_size_ret is NULL, it is ignored.
 *
 * \return CL_SUCCESS if the function is executed successfully. Otherwise, it
 * returns one of the following errors:
 * - CL_INVALID_VALUE if param_name is not valid, or if size in bytes specified
 *   by param_value_size is < size of return type and param_value is not NULL.
 * - CL_INVALID_MEM_OBJECT if pipe is a not a valid pipe object.
 * - CL_OUT_OF_RESOURCES if there is a failure to allocate resources required
 *   by the OpenCL implementation on the device.
 * - CL_OUT_OF_HOST_MEMORY if there is a failure to allocate resources required
 *   by the OpenCL implementation on the host.
 *
 * \version 2.0r19
 */
RUNTIME_ENTRY(cl_int, clGetPipeInfo,
              (cl_mem memobj, cl_image_info param_name, size_t param_value_size, void* param_value,
               size_t* param_value_size_ret)) {
  if (!is_valid(memobj)) {
    return CL_INVALID_MEM_OBJECT;
  }

  amd::Pipe* pipe = as_amd(memobj)->asPipe();
  if (pipe == NULL) {
    return CL_INVALID_MEM_OBJECT;
  }

  switch (param_name) {
    case CL_PIPE_PACKET_SIZE: {
      cl_uint packetSize = pipe->getPacketSize();
      return amd::clGetInfo(packetSize, param_value_size, param_value, param_value_size_ret);
    }
    case CL_PIPE_MAX_PACKETS: {
      cl_uint count = pipe->getMaxNumPackets();
      return amd::clGetInfo(count, param_value_size, param_value, param_value_size_ret);
    }
    default:
      break;
  }

  return CL_INVALID_VALUE;
}
RUNTIME_EXIT

/*! @}
 *  @}
 */
