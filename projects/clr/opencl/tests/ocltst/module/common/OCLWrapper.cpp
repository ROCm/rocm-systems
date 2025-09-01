/* Copyright (c) 2010 - 2021 Advanced Micro Devices, Inc.

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

#include "OCLWrapper.h"

OCLWrapper::OCLWrapper(cl_platform_id platform) {
  clEnqueueWaitSignalAMD_ptr = (clEnqueueWaitSignalAMD_fn)clGetExtensionFunctionAddressForPlatform(
      platform, "clEnqueueWaitSignalAMD");
  clEnqueueWriteSignalAMD_ptr =
      (clEnqueueWriteSignalAMD_fn)clGetExtensionFunctionAddressForPlatform(
          platform, "clEnqueueWriteSignalAMD");
  clEnqueueMakeBuffersResidentAMD_ptr =
      (clEnqueueMakeBuffersResidentAMD_fn)clGetExtensionFunctionAddressForPlatform(
          platform, "clEnqueueMakeBuffersResidentAMD");

  clUnloadPlatformAMD_ptr = (clUnloadPlatformAMD_fn)clGetExtensionFunctionAddressForPlatform(
      platform, "clUnloadPlatformAMD");

  // CL-GL function pointers
  clGetGLContextInfoKHR_ptr =
      (clGetGLContextInfoKHR_fn)clGetExtensionFunctionAddress("clGetGLContextInfoKHR");
  clCreateFromGLBuffer_ptr = (clCreateFromGLBuffer_fn)clGetExtensionFunctionAddressForPlatform(
      platform, "clCreateFromGLBuffer");
  clCreateFromGLTexture_ptr = (clCreateFromGLTexture_fn)clGetExtensionFunctionAddressForPlatform(
      platform, "clCreateFromGLTexture");
  clCreateFromGLTexture2D_ptr =
      (clCreateFromGLTexture2D_fn)clGetExtensionFunctionAddressForPlatform(
          platform, "clCreateFromGLTexture2D");
  clCreateFromGLRenderbuffer_ptr =
      (clCreateFromGLRenderbuffer_fn)clGetExtensionFunctionAddressForPlatform(
          platform, "clCreateFromGLRenderbuffer");
  clGetGLObjectInfo_ptr =
      (clGetGLObjectInfo_fn)clGetExtensionFunctionAddressForPlatform(platform, "clGetGLObjectInfo");
  clGetGLTextureInfo_ptr = (clGetGLTextureInfo_fn)clGetExtensionFunctionAddressForPlatform(
      platform, "clGetGLTextureInfo");
  clEnqueueAcquireGLObjects_ptr =
      (clEnqueueAcquireGLObjects_fn)clGetExtensionFunctionAddressForPlatform(
          platform, "clEnqueueAcquireGLObjects");
  clEnqueueReleaseGLObjects_ptr =
      (clEnqueueReleaseGLObjects_fn)clGetExtensionFunctionAddressForPlatform(
          platform, "clEnqueueReleaseGLObjects");

  // Performance counter function pointers
  clCreatePerfCounterAMD_ptr = (clCreatePerfCounterAMD_fn)clGetExtensionFunctionAddressForPlatform(
      platform, "clCreatePerfCounterAMD");
  clEnqueueBeginPerfCounterAMD_ptr =
      (clEnqueueBeginPerfCounterAMD_fn)clGetExtensionFunctionAddressForPlatform(
          platform, "clEnqueueBeginPerfCounterAMD");
  clEnqueueEndPerfCounterAMD_ptr =
      (clEnqueueEndPerfCounterAMD_fn)clGetExtensionFunctionAddressForPlatform(
          platform, "clEnqueueEndPerfCounterAMD");
  clGetPerfCounterInfoAMD_ptr =
      (clGetPerfCounterInfoAMD_fn)clGetExtensionFunctionAddressForPlatform(
          platform, "clGetPerfCounterInfoAMD");
  clReleasePerfCounterAMD_ptr =
      (clReleasePerfCounterAMD_fn)clGetExtensionFunctionAddressForPlatform(
          platform, "clReleasePerfCounterAMD");
  clRetainPerfCounterAMD_ptr = (clRetainPerfCounterAMD_fn)clGetExtensionFunctionAddressForPlatform(
      platform, "clRetainPerfCounterAMD");
  clSetDeviceClockModeAMD_ptr =
      (clSetDeviceClockModeAMD_fn)clGetExtensionFunctionAddressForPlatform(
          platform, "clSetDeviceClockModeAMD");
}

cl_int OCLWrapper::clGetPlatformIDs(cl_uint num_entries, cl_platform_id* platforms,
                                    cl_uint* num_platforms) {
  return ::clGetPlatformIDs(num_entries, platforms, num_platforms);
}

cl_int OCLWrapper::clGetPlatformInfo(cl_platform_id platform, cl_platform_info param_name,
                                     size_t param_value_size, void* param_value,
                                     size_t* param_value_size_ret) {
  return ::clGetPlatformInfo(platform, param_name, param_value_size, param_value,
                             param_value_size_ret);
}

cl_int OCLWrapper::clGetDeviceIDs(cl_platform_id platform, cl_device_type device_type,
                                  cl_uint num_entries, cl_device_id* devices,
                                  cl_uint* num_devices) {
  return ::clGetDeviceIDs(platform, device_type, num_entries, devices, num_devices);
}

cl_int OCLWrapper::clGetDeviceInfo(cl_device_id device, cl_device_info param_name,
                                   size_t param_value_size, void* param_value,
                                   size_t* param_value_size_ret) {
  return ::clGetDeviceInfo(device, param_name, param_value_size, param_value, param_value_size_ret);
}

cl_context OCLWrapper::clCreateContext(cl_context_properties* properties, cl_uint num_devices,
                                       const cl_device_id* devices,
                                       void(CL_CALLBACK* pfn_notify)(const char*, const void*,
                                                                     size_t, void*),
                                       void* user_data, cl_int* errcode_ret) {
  return ::clCreateContext(properties, num_devices, devices, pfn_notify, user_data, errcode_ret);
}

cl_context OCLWrapper::clCreateContextFromType(
    cl_context_properties* properties, cl_device_type device_type,
    void(CL_CALLBACK* pfn_notify)(const char*, const void*, size_t, void*), void* user_data,
    cl_int* errcode_ret) {
  return ::clCreateContextFromType(properties, device_type, pfn_notify, user_data, errcode_ret);
}

cl_int OCLWrapper::clRetainContext(cl_context context) { return ::clRetainContext(context); }

cl_int OCLWrapper::clReleaseContext(cl_context context) { return ::clReleaseContext(context); }

cl_int OCLWrapper::clGetContextInfo(cl_context context, cl_context_info param_name,
                                    size_t param_value_size, void* param_value,
                                    size_t* param_value_size_ret) {
  return ::clGetContextInfo(context, param_name, param_value_size, param_value,
                            param_value_size_ret);
}

cl_command_queue OCLWrapper::clCreateCommandQueue(cl_context context, cl_device_id device,
                                                  cl_command_queue_properties properties,
                                                  cl_int* errcode_ret) {
#if defined(CL_VERSION_2_0)
  cl_int err;
  cl_platform_id pid;
  bool version20 = true;
  err = ::clGetDeviceInfo(device, CL_DEVICE_PLATFORM, sizeof(cl_platform_id), &pid, NULL);
  if (err == CL_SUCCESS) {
    size_t size;
    char* ver;
    err = ::clGetPlatformInfo(pid, CL_PLATFORM_VERSION, 0, NULL, &size);
    if (err == CL_SUCCESS) {
      ver = new char[size];
      if (ver) {
        err = ::clGetPlatformInfo(pid, CL_PLATFORM_VERSION, size, ver, NULL);
        if (err == CL_SUCCESS) {
          if (ver[8] == '1') {
            version20 = false;
          }
        }
        delete[] ver;
      }
    }
  }
  if (version20) {
    const cl_queue_properties cprops[] = {CL_QUEUE_PROPERTIES,
                                          static_cast<cl_queue_properties>(properties), 0};
    return ::clCreateCommandQueueWithProperties(context, device, properties ? cprops : NULL,
                                                errcode_ret);
  } else {
    return ::clCreateCommandQueue(context, device, properties, errcode_ret);
  }
#else
  return ::clCreateCommandQueue(context, device, properties, errcode_ret);
#endif
}

cl_int OCLWrapper::clRetainCommandQueue(cl_command_queue command_queue) {
  return ::clRetainCommandQueue(command_queue);
}

cl_int OCLWrapper::clReleaseCommandQueue(cl_command_queue command_queue) {
  return ::clReleaseCommandQueue(command_queue);
}

cl_int OCLWrapper::clGetCommandQueueInfo(cl_command_queue command_queue,
                                         cl_command_queue_info param_name, size_t param_value_size,
                                         void* param_value, size_t* param_value_size_ret) {
  return ::clGetCommandQueueInfo(command_queue, param_name, param_value_size, param_value,
                                 param_value_size_ret);
}

cl_mem OCLWrapper::clCreateBuffer(cl_context context, cl_mem_flags flags, size_t size,
                                  void* host_ptr, cl_int* errcode_ret) {
  return ::clCreateBuffer(context, flags, size, host_ptr, errcode_ret);
}

cl_mem OCLWrapper::clCreateImage2D(cl_context context, cl_mem_flags flags,
                                   const cl_image_format* image_format, size_t image_width,
                                   size_t image_height, size_t image_row_pitch, void* host_ptr,
                                   cl_int* errcode_ret) {
  return ::clCreateImage2D(context, flags, image_format, image_width, image_height, image_row_pitch,
                           host_ptr, errcode_ret);
}

cl_mem OCLWrapper::clCreateImage3D(cl_context context, cl_mem_flags flags,
                                   const cl_image_format* image_format, size_t image_width,
                                   size_t image_height, size_t image_depth, size_t image_row_pitch,
                                   size_t image_slice_pitch, void* host_ptr, cl_int* errcode_ret) {
  return ::clCreateImage3D(context, flags, image_format, image_width, image_height, image_depth,
                           image_row_pitch, image_slice_pitch, host_ptr, errcode_ret);
}

cl_int OCLWrapper::clRetainMemObject(cl_mem memobj) { return ::clRetainMemObject(memobj); }

cl_int OCLWrapper::clReleaseMemObject(cl_mem memobj) { return ::clReleaseMemObject(memobj); }

cl_int OCLWrapper::clGetSupportedImageFormats(cl_context context, cl_mem_flags flags,
                                              cl_mem_object_type image_type, cl_uint num_entries,
                                              cl_image_format* image_formats,
                                              cl_uint* num_image_formats) {
  return ::clGetSupportedImageFormats(context, flags, image_type, num_entries, image_formats,
                                      num_image_formats);
}

cl_int OCLWrapper::clGetMemObjectInfo(cl_mem memobj, cl_mem_info param_name,
                                      size_t param_value_size, void* param_value,
                                      size_t* param_value_size_ret) {
  return ::clGetMemObjectInfo(memobj, param_name, param_value_size, param_value,
                              param_value_size_ret);
}

cl_int OCLWrapper::clGetImageInfo(cl_mem image, cl_image_info param_name, size_t param_value_size,
                                  void* param_value, size_t* param_value_size_ret) {
  return ::clGetImageInfo(image, param_name, param_value_size, param_value, param_value_size_ret);
}

cl_sampler OCLWrapper::clCreateSampler(cl_context context, cl_bool normalized_coords,
                                       cl_addressing_mode addressing_mode,
                                       cl_filter_mode filter_mode, cl_int* errcode_ret) {
#ifdef CL_VERSION_2_0
  const cl_sampler_properties sprops[] = {CL_SAMPLER_NORMALIZED_COORDS,
                                          static_cast<cl_sampler_properties>(normalized_coords),
                                          CL_SAMPLER_ADDRESSING_MODE,
                                          static_cast<cl_sampler_properties>(addressing_mode),
                                          CL_SAMPLER_FILTER_MODE,
                                          static_cast<cl_sampler_properties>(filter_mode),
                                          0};
  return ::clCreateSamplerWithProperties(context, sprops, errcode_ret);
#else
  return ::clCreateSampler(context, normalized_coords, addressing_mode, filter_mode, errcode_ret);
#endif
}

cl_int OCLWrapper::clRetainSampler(cl_sampler sampler) { return ::clRetainSampler(sampler); }

cl_int OCLWrapper::clReleaseSampler(cl_sampler sampler) { return ::clReleaseSampler(sampler); }

cl_int OCLWrapper::clGetSamplerInfo(cl_sampler sampler, cl_sampler_info param_name,
                                    size_t param_value_size, void* param_value,
                                    size_t* param_value_size_ret) {
  return ::clGetSamplerInfo(sampler, param_name, param_value_size, param_value,
                            param_value_size_ret);
}

cl_program OCLWrapper::clCreateProgramWithSource(cl_context context, cl_uint count,
                                                 const char** strings, const size_t* lengths,
                                                 cl_int* errcode_ret) {
  return ::clCreateProgramWithSource(context, count, strings, lengths, errcode_ret);
}

cl_program OCLWrapper::clCreateProgramWithBinary(cl_context context, cl_uint num_devices,
                                                 const cl_device_id* device_list,
                                                 const size_t* lengths,
                                                 const unsigned char** binaries,
                                                 cl_int* binary_status, cl_int* errcode_ret) {
  return ::clCreateProgramWithBinary(context, num_devices, device_list, lengths, binaries,
                                     binary_status, errcode_ret);
}

cl_int OCLWrapper::clRetainProgram(cl_program program) { return ::clRetainProgram(program); }

cl_int OCLWrapper::clReleaseProgram(cl_program program) { return ::clReleaseProgram(program); }

cl_int OCLWrapper::clBuildProgram(
    cl_program program, cl_uint num_devices, const cl_device_id* device_list, const char* options,
    void(CL_CALLBACK* pfn_notify)(cl_program program, void* user_data), void* user_data) {
  return ::clBuildProgram(program, num_devices, device_list, options, pfn_notify, user_data);
}

cl_int OCLWrapper::clCompileProgram(
    cl_program program, cl_uint num_devices, const cl_device_id* device_list, const char* options,
    cl_uint num_input_headers, const cl_program* input_headers, const char** header_include_names,
    void(CL_CALLBACK* pfn_notify)(cl_program program, void* user_data), void* user_data) {
  return ::clCompileProgram(program, num_devices, device_list, options, num_input_headers,
                            input_headers, header_include_names, pfn_notify, user_data);
}

cl_program OCLWrapper::clLinkProgram(cl_context context, cl_uint num_devices,
                                     const cl_device_id* device_list, const char* options,
                                     cl_uint num_input_programs, const cl_program* input_programs,
                                     void(CL_CALLBACK* pfn_notify)(cl_program program,
                                                                   void* user_data),
                                     void* user_data, cl_int* errcode_ret) {
  return ::clLinkProgram(context, num_devices, device_list, options, num_input_programs,
                         input_programs, pfn_notify, user_data, errcode_ret);
}

cl_int OCLWrapper::clUnloadCompiler(void) { return ::clUnloadCompiler(); }

cl_int OCLWrapper::clGetProgramInfo(cl_program program, cl_program_info param_name,
                                    size_t param_value_size, void* param_value,
                                    size_t* param_value_size_ret) {
  return ::clGetProgramInfo(program, param_name, param_value_size, param_value,
                            param_value_size_ret);
}

cl_int OCLWrapper::clGetProgramBuildInfo(cl_program program, cl_device_id device,
                                         cl_program_build_info param_name, size_t param_value_size,
                                         void* param_value, size_t* param_value_size_ret) {
  return ::clGetProgramBuildInfo(program, device, param_name, param_value_size, param_value,
                                 param_value_size_ret);
}

cl_kernel OCLWrapper::clCreateKernel(cl_program program, const char* kernel_name,
                                     cl_int* errcode_ret) {
  return ::clCreateKernel(program, kernel_name, errcode_ret);
}

cl_int OCLWrapper::clCreateKernelsInProgram(cl_program program, cl_uint num_kernels,
                                            cl_kernel* kernels, cl_uint* num_kernels_ret) {
  return ::clCreateKernelsInProgram(program, num_kernels, kernels, num_kernels_ret);
}

cl_int OCLWrapper::clRetainKernel(cl_kernel kernel) { return ::clRetainKernel(kernel); }

cl_int OCLWrapper::clReleaseKernel(cl_kernel kernel) { return ::clReleaseKernel(kernel); }

cl_int OCLWrapper::clSetKernelArg(cl_kernel kernel, cl_uint arg_index, size_t arg_size,
                                  const void* arg_value) {
  return ::clSetKernelArg(kernel, arg_index, arg_size, arg_value);
}

cl_int OCLWrapper::clGetKernelInfo(cl_kernel kernel, cl_kernel_info param_name,
                                   size_t param_value_size, void* param_value,
                                   size_t* param_value_size_ret) {
  return ::clGetKernelInfo(kernel, param_name, param_value_size, param_value, param_value_size_ret);
}

cl_int OCLWrapper::clGetKernelWorkGroupInfo(cl_kernel kernel, cl_device_id device,
                                            cl_kernel_work_group_info param_name,
                                            size_t param_value_size, void* param_value,
                                            size_t* param_value_size_ret) {
  return ::clGetKernelWorkGroupInfo(kernel, device, param_name, param_value_size, param_value,
                                    param_value_size_ret);
}

cl_int OCLWrapper::clWaitForEvents(cl_uint num_events, const cl_event* event_list) {
  return ::clWaitForEvents(num_events, event_list);
}

cl_int OCLWrapper::clGetEventInfo(cl_event evnt, cl_event_info param_name, size_t param_value_size,
                                  void* param_value, size_t* param_value_size_ret) {
  return ::clGetEventInfo(evnt, param_name, param_value_size, param_value, param_value_size_ret);
}

cl_int OCLWrapper::clRetainEvent(cl_event evnt) { return ::clRetainEvent(evnt); }

cl_int OCLWrapper::clReleaseEvent(cl_event evnt) { return ::clReleaseEvent(evnt); }

cl_int OCLWrapper::clGetEventProfilingInfo(cl_event evnt, cl_profiling_info param_name,
                                           size_t param_value_size, void* param_value,
                                           size_t* param_value_size_ret) {
  return ::clGetEventProfilingInfo(evnt, param_name, param_value_size, param_value,
                                   param_value_size_ret);
}

cl_int OCLWrapper::clFlush(cl_command_queue command_queue) { return ::clFlush(command_queue); }

cl_int OCLWrapper::clFinish(cl_command_queue command_queue) { return ::clFinish(command_queue); }

cl_int OCLWrapper::clEnqueueReadBuffer(cl_command_queue command_queue, cl_mem buffer,
                                       cl_bool blocking_read, size_t offset, size_t cb, void* ptr,
                                       cl_uint num_events_in_wait_list,
                                       const cl_event* event_wait_list, cl_event* evnt) {
  return ::clEnqueueReadBuffer(command_queue, buffer, blocking_read, offset, cb, ptr,
                               num_events_in_wait_list, event_wait_list, evnt);
}

cl_int OCLWrapper::clEnqueueWriteBuffer(cl_command_queue command_queue, cl_mem buffer,
                                        cl_bool blocking_write, size_t offset, size_t cb,
                                        const void* ptr, cl_uint num_events_in_wait_list,
                                        const cl_event* event_wait_list, cl_event* evnt) {
  return ::clEnqueueWriteBuffer(command_queue, buffer, blocking_write, offset, cb, ptr,
                                num_events_in_wait_list, event_wait_list, evnt);
}

cl_int OCLWrapper::clEnqueueCopyBuffer(cl_command_queue command_queue, cl_mem src_buffer,
                                       cl_mem dst_buffer, size_t src_offset, size_t dst_offset,
                                       size_t cb, cl_uint num_events_in_wait_list,
                                       const cl_event* event_wait_list, cl_event* evnt) {
  return ::clEnqueueCopyBuffer(command_queue, src_buffer, dst_buffer, src_offset, dst_offset, cb,
                               num_events_in_wait_list, event_wait_list, evnt);
}

cl_int OCLWrapper::clEnqueueReadBufferRect(cl_command_queue command_queue, cl_mem buffer,
                                           cl_bool blocking_read, const size_t* buffer_origin,
                                           const size_t* host_origin, const size_t* region,
                                           size_t buffer_row_pitch, size_t buffer_slice_pitch,
                                           size_t host_row_pitch, size_t host_slice_pitch,
                                           void* ptr, cl_uint num_events_in_wait_list,
                                           const cl_event* event_wait_list, cl_event* evnt) {
  return ::clEnqueueReadBufferRect(command_queue, buffer, blocking_read, buffer_origin, host_origin,
                                   region, buffer_row_pitch, buffer_slice_pitch, host_row_pitch,
                                   host_slice_pitch, ptr, num_events_in_wait_list, event_wait_list,
                                   evnt);
}

cl_int OCLWrapper::clEnqueueWriteBufferRect(cl_command_queue command_queue, cl_mem buffer,
                                            cl_bool blocking_write, const size_t* buffer_origin,
                                            const size_t* host_origin, const size_t* region,
                                            size_t buffer_row_pitch, size_t buffer_slice_pitch,
                                            size_t host_row_pitch, size_t host_slice_pitch,
                                            const void* ptr, cl_uint num_events_in_wait_list,
                                            const cl_event* event_wait_list, cl_event* evnt) {
  return ::clEnqueueWriteBufferRect(command_queue, buffer, blocking_write, buffer_origin,
                                    host_origin, region, buffer_row_pitch, buffer_slice_pitch,
                                    host_row_pitch, host_slice_pitch, ptr, num_events_in_wait_list,
                                    event_wait_list, evnt);
}

cl_int OCLWrapper::clEnqueueCopyBufferRect(cl_command_queue command_queue, cl_mem src_buffer,
                                           cl_mem dst_buffer, const size_t* src_origin,
                                           const size_t* dst_origin, const size_t* region,
                                           size_t src_row_pitch, size_t src_slice_pitch,
                                           size_t dst_row_pitch, size_t dst_slice_pitch,
                                           cl_uint num_events_in_wait_list,
                                           const cl_event* event_wait_list, cl_event* evnt) {
  return ::clEnqueueCopyBufferRect(command_queue, src_buffer, dst_buffer, src_origin, dst_origin,
                                   region, src_row_pitch, src_slice_pitch, dst_row_pitch,
                                   dst_slice_pitch, num_events_in_wait_list, event_wait_list, evnt);
}

cl_int OCLWrapper::clEnqueueReadImage(cl_command_queue command_queue, cl_mem image,
                                      cl_bool blocking_read, const size_t* origin,
                                      const size_t* region, size_t row_pitch, size_t slice_pitch,
                                      void* ptr, cl_uint num_events_in_wait_list,
                                      const cl_event* event_wait_list, cl_event* evnt) {
  return ::clEnqueueReadImage(command_queue, image, blocking_read, origin, region, row_pitch,
                              slice_pitch, ptr, num_events_in_wait_list, event_wait_list, evnt);
}

cl_int OCLWrapper::clEnqueueWriteImage(cl_command_queue command_queue, cl_mem image,
                                       cl_bool blocking_write, const size_t* origin,
                                       const size_t* region, size_t input_row_pitch,
                                       size_t input_slice_pitch, const void* ptr,
                                       cl_uint num_events_in_wait_list,
                                       const cl_event* event_wait_list, cl_event* evnt) {
  return ::clEnqueueWriteImage(command_queue, image, blocking_write, origin, region,
                               input_row_pitch, input_slice_pitch, ptr, num_events_in_wait_list,
                               event_wait_list, evnt);
}

cl_int OCLWrapper::clEnqueueCopyImage(cl_command_queue command_queue, cl_mem src_image,
                                      cl_mem dst_image, const size_t* src_origin,
                                      const size_t* dst_origin, const size_t* region,
                                      cl_uint num_events_in_wait_list,
                                      const cl_event* event_wait_list, cl_event* evnt) {
  return ::clEnqueueCopyImage(command_queue, src_image, dst_image, src_origin, dst_origin, region,
                              num_events_in_wait_list, event_wait_list, evnt);
}

cl_int OCLWrapper::clEnqueueCopyImageToBuffer(cl_command_queue command_queue, cl_mem src_image,
                                              cl_mem dst_buffer, const size_t* src_origin,
                                              const size_t* region, size_t dst_offset,
                                              cl_uint num_events_in_wait_list,
                                              const cl_event* event_wait_list, cl_event* evnt) {
  return ::clEnqueueCopyImageToBuffer(command_queue, src_image, dst_buffer, src_origin, region,
                                      dst_offset, num_events_in_wait_list, event_wait_list, evnt);
}

cl_int OCLWrapper::clEnqueueCopyBufferToImage(cl_command_queue command_queue, cl_mem src_buffer,
                                              cl_mem dst_image, size_t src_offset,
                                              const size_t* dst_origin, const size_t* region,
                                              cl_uint num_events_in_wait_list,
                                              const cl_event* event_wait_list, cl_event* evnt) {
  return ::clEnqueueCopyBufferToImage(command_queue, src_buffer, dst_image, src_offset, dst_origin,
                                      region, num_events_in_wait_list, event_wait_list, evnt);
}

void* OCLWrapper::clEnqueueMapBuffer(cl_command_queue command_queue, cl_mem buffer,
                                     cl_bool blocking_map, cl_map_flags map_flags, size_t offset,
                                     size_t cb, cl_uint num_events_in_wait_list,
                                     const cl_event* event_wait_list, cl_event* evnt,
                                     cl_int* errcode_ret) {
  return ::clEnqueueMapBuffer(command_queue, buffer, blocking_map, map_flags, offset, cb,
                              num_events_in_wait_list, event_wait_list, evnt, errcode_ret);
}

void* OCLWrapper::clEnqueueMapImage(cl_command_queue command_queue, cl_mem image,
                                    cl_bool blocking_map, cl_map_flags map_flags,
                                    const size_t* origin, const size_t* region,
                                    size_t* image_row_pitch, size_t* image_slice_pitch,
                                    cl_uint num_events_in_wait_list,
                                    const cl_event* event_wait_list, cl_event* evnt,
                                    cl_int* errcode_ret) {
  return ::clEnqueueMapImage(command_queue, image, blocking_map, map_flags, origin, region,
                             image_row_pitch, image_slice_pitch, num_events_in_wait_list,
                             event_wait_list, evnt, errcode_ret);
}

cl_int OCLWrapper::clEnqueueUnmapMemObject(cl_command_queue command_queue, cl_mem memobj,
                                           void* mapped_ptr, cl_uint num_events_in_wait_list,
                                           const cl_event* event_wait_list, cl_event* evnt) {
  return ::clEnqueueUnmapMemObject(command_queue, memobj, mapped_ptr, num_events_in_wait_list,
                                   event_wait_list, evnt);
}

cl_int OCLWrapper::clEnqueueNDRangeKernel(cl_command_queue command_queue, cl_kernel kernel,
                                          cl_uint work_dim, const size_t* global_work_offset,
                                          const size_t* global_work_size,
                                          const size_t* local_work_size,
                                          cl_uint num_events_in_wait_list,
                                          const cl_event* event_wait_list, cl_event* evnt) {
  return ::clEnqueueNDRangeKernel(command_queue, kernel, work_dim, global_work_offset,
                                  global_work_size, local_work_size, num_events_in_wait_list,
                                  event_wait_list, evnt);
}

cl_int OCLWrapper::clEnqueueTask(cl_command_queue command_queue, cl_kernel kernel,
                                 cl_uint num_events_in_wait_list, const cl_event* event_wait_list,
                                 cl_event* evnt) {
#if defined(CL_VERSION_2_0)
  static size_t const globalWorkSize[3] = {1, 0, 0};
  static size_t const localWorkSize[3] = {1, 0, 0};

  return ::clEnqueueNDRangeKernel(command_queue, kernel, 1, NULL, globalWorkSize, localWorkSize,
                                  num_events_in_wait_list, event_wait_list, evnt);
#else
  return ::clEnqueueTask(command_queue, kernel, num_events_in_wait_list, event_wait_list, evnt);
#endif
}

cl_int OCLWrapper::clEnqueueNativeKernel(cl_command_queue command_queue,
                                         void(CL_CALLBACK* user_func)(void*), void* args,
                                         size_t cb_args, cl_uint num_mem_objects,
                                         const cl_mem* mem_list, const void** args_mem_loc,
                                         cl_uint num_events_in_wait_list,
                                         const cl_event* event_wait_list, cl_event* evnt) {
  return ::clEnqueueNativeKernel(command_queue, user_func, args, cb_args, num_mem_objects, mem_list,
                                 args_mem_loc, num_events_in_wait_list, event_wait_list, evnt);
}

cl_int OCLWrapper::clEnqueueMarker(cl_command_queue command_queue, cl_event* evnt) {
  return ::clEnqueueMarker(command_queue, evnt);
}

cl_int OCLWrapper::clEnqueueMarkerWithWaitList(cl_command_queue command_queue,
                                               cl_uint num_events_in_wait_list,
                                               const cl_event* event_wait_list, cl_event* evnt) {
  return ::clEnqueueMarkerWithWaitList(command_queue, num_events_in_wait_list, event_wait_list,
                                       evnt);
}

cl_int OCLWrapper::clEnqueueWaitForEvents(cl_command_queue command_queue, cl_uint num_events,
                                          const cl_event* event_list) {
  return ::clEnqueueWaitForEvents(command_queue, num_events, event_list);
}

cl_int OCLWrapper::clEnqueueBarrier(cl_command_queue command_queue) {
  return ::clEnqueueBarrier(command_queue);
}

void* OCLWrapper::clGetExtensionFunctionAddress(const char* func_name) {
  return ::clGetExtensionFunctionAddress(func_name);
}

cl_mem OCLWrapper::clCreateImage(cl_context context, cl_mem_flags flags,
                                 const cl_image_format* image_format,
                                 const cl_image_desc* image_desc, void* host_ptr,
                                 cl_int* errcode_ret) {
  return ::clCreateImage(context, flags, image_format, image_desc, host_ptr, errcode_ret);
}

cl_mem OCLWrapper::clCreateSubBuffer(cl_mem mem, cl_mem_flags flags,
                                     cl_buffer_create_type buffer_create_type,
                                     const void* buffer_create_info, cl_int* errcode_ret) {
  return ::clCreateSubBuffer(mem, flags, buffer_create_type, buffer_create_info, errcode_ret);
}

cl_int OCLWrapper::clSetEventCallback(
    cl_event event, cl_int command_exec_callback_type,
    void(CL_CALLBACK* pfn_event_notify)(cl_event event, cl_int event_command_exec_status,
                                        void* user_data),
    void* user_data) {
  return ::clSetEventCallback(event, command_exec_callback_type, pfn_event_notify, user_data);
}

cl_int OCLWrapper::clEnqueueFillImage(cl_command_queue command_queue, cl_mem image, void* ptr,
                                      const size_t* origin, const size_t* region,
                                      cl_uint num_events_in_wait_list,
                                      const cl_event* event_wait_list, cl_event* evnt) {
  return ::clEnqueueFillImage(command_queue, image, ptr, origin, region, num_events_in_wait_list,
                              event_wait_list, evnt);
}

cl_int OCLWrapper::clUnloadPlatformAMD(cl_platform_id id) {
  if (clUnloadPlatformAMD_ptr) return clUnloadPlatformAMD_ptr(id);
  return CL_SUCCESS;
}
cl_int OCLWrapper::clEnqueueWaitSignalAMD(cl_command_queue command_queue, cl_mem mem_object,
                                          cl_uint value, cl_uint num_events,
                                          const cl_event* event_wait_list, cl_event* event) {
  return clEnqueueWaitSignalAMD_ptr(command_queue, mem_object, value, num_events, event_wait_list,
                                    event);
}

cl_int OCLWrapper::clEnqueueWriteSignalAMD(cl_command_queue command_queue, cl_mem mem_object,
                                           cl_uint value, cl_ulong offset, cl_uint num_events,
                                           const cl_event* event_list, cl_event* event) {
  return clEnqueueWriteSignalAMD_ptr(command_queue, mem_object, value, offset, num_events,
                                     event_list, event);
}

cl_int OCLWrapper::clEnqueueMakeBuffersResidentAMD(cl_command_queue command_queue,
                                                   cl_uint num_mem_objs, cl_mem* mem_objects,
                                                   cl_bool blocking_make_resident,
                                                   cl_bus_address_amd* bus_addresses,
                                                   cl_uint num_events, const cl_event* event_list,
                                                   cl_event* event) {
  return clEnqueueMakeBuffersResidentAMD_ptr(command_queue, num_mem_objs, mem_objects,
                                             blocking_make_resident, bus_addresses, num_events,
                                             event_list, event);
}

cl_int OCLWrapper::clEnqueueMigrateMemObjects(cl_command_queue command_queue,
                                              cl_uint num_mem_objects, const cl_mem* mem_objects,
                                              cl_mem_migration_flags flags,
                                              cl_uint num_events_in_wait_list,
                                              const cl_event* event_wait_list, cl_event* event) {
  return ::clEnqueueMigrateMemObjects(command_queue, num_mem_objects, mem_objects, flags,
                                      num_events_in_wait_list, event_wait_list, event);
}

cl_int OCLWrapper::clGetGLContextInfoKHR(const cl_context_properties* properties,
                                         cl_gl_context_info param_name, size_t param_value_size,
                                         void* param_value, size_t* param_value_size_ret) {
  return (*clGetGLContextInfoKHR_ptr)(properties, param_name, param_value_size, param_value,
                                      param_value_size_ret);
}

cl_mem OCLWrapper::clCreateFromGLBuffer(cl_context context, cl_mem_flags flags, unsigned int bufobj,
                                        int* errcode_ret) {
  return (*clCreateFromGLBuffer_ptr)(context, flags, bufobj, errcode_ret);
}

cl_mem OCLWrapper::clCreateFromGLTexture(cl_context context, cl_mem_flags flags,
                                         unsigned int texture_target, int miplevel,
                                         unsigned int texture, cl_int* errcode_ret) {
  return (*clCreateFromGLTexture_ptr)(context, flags, texture_target, miplevel, texture,
                                      errcode_ret);
}

cl_mem OCLWrapper::clCreateFromGLTexture2D(cl_context context, cl_mem_flags flags,
                                           unsigned int texture_target, int miplevel,
                                           unsigned int texture, cl_int* errcode_ret) {
  return (*clCreateFromGLTexture2D_ptr)(context, flags, texture_target, miplevel, texture,
                                        errcode_ret);
}

cl_mem OCLWrapper::clCreateFromGLRenderbuffer(cl_context context, cl_mem_flags flags,
                                              unsigned int renderbuffer, cl_int* errcode_ret) {
  return (*clCreateFromGLRenderbuffer_ptr)(context, flags, renderbuffer, errcode_ret);
}

cl_int OCLWrapper::clGetGLObjectInfo(cl_mem memobj, cl_gl_object_type* gl_object_type,
                                     unsigned int* gl_object_name) {
  return (*clGetGLObjectInfo_ptr)(memobj, gl_object_type, gl_object_name);
}

cl_int OCLWrapper::clGetGLTextureInfo(cl_mem memobj, cl_gl_texture_info param_name,
                                      size_t param_value_size, void* param_value,
                                      size_t* param_value_size_ret) {
  return (*clGetGLTextureInfo_ptr)(memobj, param_name, param_value_size, param_value,
                                   param_value_size_ret);
}

cl_int OCLWrapper::clEnqueueAcquireGLObjects(cl_command_queue command_queue, cl_uint num_objects,
                                             const cl_mem* mem_objects,
                                             cl_uint num_events_in_wait_list,
                                             const cl_event* event_wait_list, cl_event* event) {
  return (*clEnqueueAcquireGLObjects_ptr)(command_queue, num_objects, mem_objects,
                                          num_events_in_wait_list, event_wait_list, event);
}

cl_int OCLWrapper::clEnqueueReleaseGLObjects(cl_command_queue command_queue, cl_uint num_objects,
                                             const cl_mem* mem_objects,
                                             cl_uint num_events_in_wait_list,
                                             const cl_event* event_wait_list, cl_event* event) {
  return (*clEnqueueReleaseGLObjects_ptr)(command_queue, num_objects, mem_objects,
                                          num_events_in_wait_list, event_wait_list, event);
}

#if defined(CL_VERSION_2_0)
cl_command_queue OCLWrapper::clCreateCommandQueueWithProperties(
    cl_context context, cl_device_id device, const cl_queue_properties* properties,
    cl_int* errcode_ret) {
  return ::clCreateCommandQueueWithProperties(context, device, properties, errcode_ret);
}

void* OCLWrapper::clSVMAlloc(cl_context context, cl_svm_mem_flags flags, size_t size,
                             cl_uint alignment) {
  return ::clSVMAlloc(context, flags, size, alignment);
}

void OCLWrapper::clSVMFree(cl_context context, void* svm_pointer) {
  return ::clSVMFree(context, svm_pointer);
}

cl_int OCLWrapper::clEnqueueSVMMap(cl_command_queue command_queue, cl_bool blocking_map,
                                   cl_map_flags flags, void* svm_ptr, size_t size,
                                   cl_uint num_events_in_wait_list, const cl_event* event_wait_list,
                                   cl_event* event) {
  return ::clEnqueueSVMMap(command_queue, blocking_map, flags, svm_ptr, size,
                           num_events_in_wait_list, event_wait_list, event);
}

cl_int OCLWrapper::clEnqueueSVMUnmap(cl_command_queue command_queue, void* svm_ptr,
                                     cl_uint num_events_in_wait_list,
                                     const cl_event* event_wait_list, cl_event* event) {
  return ::clEnqueueSVMUnmap(command_queue, svm_ptr, num_events_in_wait_list, event_wait_list,
                             event);
}
cl_int OCLWrapper::clEnqueueSVMMemFill(cl_command_queue command_queue, void* svm_ptr,
                                       const void* pattern, size_t pattern_size, size_t size,
                                       cl_uint num_events_in_wait_list,
                                       const cl_event* event_wait_list, cl_event* event) {
  return ::clEnqueueSVMMemFill(command_queue, svm_ptr, pattern, pattern_size, size,
                               num_events_in_wait_list, event_wait_list, event);
}

cl_int OCLWrapper::clSetKernelArgSVMPointer(cl_kernel kernel, cl_uint arg_index,
                                            const void* arg_value) {
  return ::clSetKernelArgSVMPointer(kernel, arg_index, arg_value);
}

cl_mem OCLWrapper::clCreatePipe(cl_context context, cl_mem_flags flags, cl_uint packet_size,
                                cl_uint pipe_max_packets, const cl_pipe_properties* properties,
                                cl_int* errcode_ret) {
  return ::clCreatePipe(context, flags, packet_size, pipe_max_packets, properties, errcode_ret);
}

cl_int OCLWrapper::clGetPipeInfo(cl_mem pipe, cl_pipe_info param_name, size_t param_value_size,
                                 void* param_value, size_t* param_value_size_ret) {
  return ::clGetPipeInfo(pipe, param_name, param_value_size, param_value, param_value_size_ret);
}

#endif

cl_perfcounter_amd OCLWrapper::clCreatePerfCounterAMD(cl_device_id device,
                                                      cl_perfcounter_property* properties,
                                                      cl_int* errcode_ret) {
  return (*clCreatePerfCounterAMD_ptr)(device, properties, errcode_ret);
}

cl_int OCLWrapper::clEnqueueBeginPerfCounterAMD(cl_command_queue command_queue,
                                                cl_uint num_perf_counters,
                                                cl_perfcounter_amd* perf_counters,
                                                cl_uint num_events_in_wait_list,
                                                const cl_event* event_wait_list, cl_event* event) {
  return (*clEnqueueBeginPerfCounterAMD_ptr)(command_queue, num_perf_counters, perf_counters,
                                             num_events_in_wait_list, event_wait_list, event);
}

cl_int OCLWrapper::clEnqueueEndPerfCounterAMD(cl_command_queue command_queue,
                                              cl_uint num_perf_counters,
                                              cl_perfcounter_amd* perf_counters,
                                              cl_uint num_events_in_wait_list,
                                              const cl_event* event_wait_list, cl_event* event) {
  return (*clEnqueueEndPerfCounterAMD_ptr)(command_queue, num_perf_counters, perf_counters,
                                           num_events_in_wait_list, event_wait_list, event);
}

cl_int OCLWrapper::clGetPerfCounterInfoAMD(cl_perfcounter_amd perf_counter,
                                           cl_perfcounter_info param_name, size_t param_value_size,
                                           void* param_value, size_t* param_value_size_ret) {
  return (*clGetPerfCounterInfoAMD_ptr)(perf_counter, param_name, param_value_size, param_value,
                                        param_value_size_ret);
}

cl_int OCLWrapper::clReleasePerfCounterAMD(cl_perfcounter_amd perf_counter) {
  return (*clReleasePerfCounterAMD_ptr)(perf_counter);
}

cl_int OCLWrapper::clRetainPerfCounterAMD(cl_perfcounter_amd perf_counter) {
  return (*clRetainPerfCounterAMD_ptr)(perf_counter);
}

cl_int OCLWrapper::clSetDeviceClockModeAMD(
    cl_device_id device, cl_set_device_clock_mode_input_amd set_clock_mode_input,
    cl_set_device_clock_mode_output_amd* set_clock_mode_output) {
  return (*clSetDeviceClockModeAMD_ptr)(device, set_clock_mode_input, set_clock_mode_output);
}
