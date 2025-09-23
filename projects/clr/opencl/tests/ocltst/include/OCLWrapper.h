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

#ifndef __OCLWrapper_H
#define __OCLWrapper_H

#define CL_USE_DEPRECATED_OPENCL_1_1_APIS
#define CL_USE_DEPRECATED_OPENCL_2_0_APIS
#include "CL/cl.h"
#include "CL/cl_ext.h"
#include "CL/cl_gl.h"
#include "cl_profile_amd.h"

typedef CL_API_ENTRY cl_int(CL_API_CALL* clUnloadPlatformAMD_fn)(cl_platform_id id);

// Function Pointer Declarations for cl_khr_gl_sharing extension (missing in
// cl_gl.h)
typedef CL_API_ENTRY cl_int(CL_API_CALL* clGetGLContextInfoKHR_fn)(
    const cl_context_properties* properties, cl_gl_context_info param_name, size_t param_value_size,
    void* param_value, size_t* param_value_size_ret);

typedef CL_API_ENTRY cl_mem(CL_API_CALL* clCreateFromGLBuffer_fn)(cl_context context,
                                                                  cl_mem_flags flags,
                                                                  unsigned int bufobj,
                                                                  int* errcode_ret);

typedef CL_API_ENTRY cl_mem(CL_API_CALL* clCreateFromGLTexture_fn)(
    cl_context context, cl_mem_flags flags, unsigned int texture_target, int miplevel,
    unsigned int texture, cl_int* errcode_ret);

typedef CL_API_ENTRY cl_mem(CL_API_CALL* clCreateFromGLTexture2D_fn)(
    cl_context context, cl_mem_flags flags, unsigned int texture_target, int miplevel,
    unsigned int texture, cl_int* errcode_ret);

typedef CL_API_ENTRY cl_mem(CL_API_CALL* clCreateFromGLRenderbuffer_fn)(cl_context context,
                                                                        cl_mem_flags flags,
                                                                        unsigned int renderbuffer,
                                                                        cl_int* errcode_ret);

typedef CL_API_ENTRY cl_int(CL_API_CALL* clGetGLObjectInfo_fn)(cl_mem memobj,
                                                               cl_gl_object_type* gl_object_type,
                                                               unsigned int* gl_object_name);

typedef CL_API_ENTRY cl_int(CL_API_CALL* clGetGLTextureInfo_fn)(cl_mem memobj,
                                                                cl_gl_texture_info param_name,
                                                                size_t param_value_size,
                                                                void* param_value,
                                                                size_t* param_value_size_ret);

typedef CL_API_ENTRY cl_int(CL_API_CALL* clEnqueueAcquireGLObjects_fn)(
    cl_command_queue command_queue, cl_uint num_objects, const cl_mem* mem_objects,
    cl_uint num_events_in_wait_list, const cl_event* event_wait_list, cl_event* event);

typedef CL_API_ENTRY cl_int(CL_API_CALL* clEnqueueReleaseGLObjects_fn)(
    cl_command_queue command_queue, cl_uint num_objects, const cl_mem* mem_objects,
    cl_uint num_events_in_wait_list, const cl_event* event_wait_list, cl_event* event);

// Function Pointer Declarations for performance counters
typedef CL_API_ENTRY cl_perfcounter_amd(CL_API_CALL* clCreatePerfCounterAMD_fn)(
    cl_device_id device, cl_perfcounter_property* properties, cl_int* errcode_ret);

typedef CL_API_ENTRY cl_int(CL_API_CALL* clEnqueueBeginPerfCounterAMD_fn)(
    cl_command_queue command_queue, cl_uint num_perf_counters, cl_perfcounter_amd* perf_counters,
    cl_uint num_events_in_wait_list, const cl_event* event_wait_list, cl_event* event);

typedef CL_API_ENTRY cl_int(CL_API_CALL* clEnqueueEndPerfCounterAMD_fn)(
    cl_command_queue command_queue, cl_uint num_perf_counters, cl_perfcounter_amd* perf_counters,
    cl_uint num_events_in_wait_list, const cl_event* event_wait_list, cl_event* event);

typedef CL_API_ENTRY cl_int(CL_API_CALL* clGetPerfCounterInfoAMD_fn)(
    cl_perfcounter_amd perf_counter, cl_perfcounter_info param_name, size_t param_value_size,
    void* param_value, size_t* param_value_size_ret);

typedef CL_API_ENTRY cl_int(CL_API_CALL* clReleasePerfCounterAMD_fn)(
    cl_perfcounter_amd perf_counter);

typedef CL_API_ENTRY cl_int(CL_API_CALL* clRetainPerfCounterAMD_fn)(
    cl_perfcounter_amd perf_counter);

typedef CL_API_ENTRY cl_int(CL_API_CALL* clSetDeviceClockModeAMD_fn)(
    cl_device_id device, cl_set_device_clock_mode_input_amd set_clock_mode_input,
    cl_set_device_clock_mode_output_amd* set_clock_mode_Output);

class OCLWrapper {
 public:
  OCLWrapper(cl_platform_id);

  ~OCLWrapper() {}

  // All OCL APIs are declared in the order they appear in cl.h

  cl_int clGetPlatformIDs(cl_uint num_entries, cl_platform_id* platforms, cl_uint* num_platforms);

  cl_int clGetPlatformInfo(cl_platform_id platform, cl_platform_info param_name,
                           size_t param_value_size, void* param_value,
                           size_t* param_value_size_ret);

  cl_int clGetDeviceIDs(cl_platform_id platform, cl_device_type device_type, cl_uint num_entries,
                        cl_device_id* devices, cl_uint* num_devices);

  cl_int clGetDeviceInfo(cl_device_id device, cl_device_info param_name, size_t param_value_size,
                         void* param_value, size_t* param_value_size_ret);

  cl_context clCreateContext(cl_context_properties* properties, cl_uint num_devices,
                             const cl_device_id* devices,
                             void(CL_CALLBACK* pfn_notify)(const char*, const void*, size_t, void*),
                             void* user_data, cl_int* errcode_ret);

  cl_context clCreateContextFromType(cl_context_properties* properties, cl_device_type device_type,
                                     void(CL_CALLBACK* pfn_notify)(const char*, const void*, size_t,
                                                                   void*),
                                     void* user_data, cl_int* errcode_ret);

  cl_int clRetainContext(cl_context context);

  cl_int clReleaseContext(cl_context context);

  cl_int clGetContextInfo(cl_context context, cl_context_info param_name, size_t param_value_size,
                          void* param_value, size_t* param_value_size_ret);

  cl_command_queue clCreateCommandQueue(cl_context context, cl_device_id device,
                                        cl_command_queue_properties properties,
                                        cl_int* errcode_ret);

  cl_int clRetainCommandQueue(cl_command_queue command_queue);

  cl_int clReleaseCommandQueue(cl_command_queue command_queue);

  cl_int clGetCommandQueueInfo(cl_command_queue command_queue, cl_command_queue_info param_name,
                               size_t param_value_size, void* param_value,
                               size_t* param_value_size_ret);

  cl_mem clCreateBuffer(cl_context context, cl_mem_flags flags, size_t size, void* host_ptr,
                        cl_int* errcode_ret);

  cl_mem clCreateImage2D(cl_context context, cl_mem_flags flags,
                         const cl_image_format* image_format, size_t image_width,
                         size_t image_height, size_t image_row_pitch, void* host_ptr,
                         cl_int* errcode_ret);

  cl_mem clCreateImage3D(cl_context context, cl_mem_flags flags,
                         const cl_image_format* image_format, size_t image_width,
                         size_t image_height, size_t image_depth, size_t image_row_pitch,
                         size_t image_slice_pitch, void* host_ptr, cl_int* errcode_ret);

  cl_int clRetainMemObject(cl_mem memobj);

  cl_int clReleaseMemObject(cl_mem memobj);

  cl_int clGetSupportedImageFormats(cl_context context, cl_mem_flags flags,
                                    cl_mem_object_type image_type, cl_uint num_entries,
                                    cl_image_format* image_formats, cl_uint* num_image_formats);

  cl_int clGetMemObjectInfo(cl_mem memobj, cl_mem_info param_name, size_t param_value_size,
                            void* param_value, size_t* param_value_size_ret);

  cl_int clGetImageInfo(cl_mem image, cl_image_info param_name, size_t param_value_size,
                        void* param_value, size_t* param_value_size_ret);

  cl_sampler clCreateSampler(cl_context context, cl_bool normalized_coords,
                             cl_addressing_mode addressing_mode, cl_filter_mode filter_mode,
                             cl_int* errcode_ret);

  cl_int clRetainSampler(cl_sampler sampler);

  cl_int clReleaseSampler(cl_sampler sampler);

  cl_int clGetSamplerInfo(cl_sampler sampler, cl_sampler_info param_name, size_t param_value_size,
                          void* param_value, size_t* param_value_size_ret);

  cl_program clCreateProgramWithSource(cl_context context, cl_uint count, const char** strings,
                                       const size_t* lengths, cl_int* errcode_ret);

  cl_program clCreateProgramWithBinary(cl_context context, cl_uint num_devices,
                                       const cl_device_id* device_list, const size_t* lengths,
                                       const unsigned char** binaries, cl_int* binary_status,
                                       cl_int* errcode_ret);

  cl_int clRetainProgram(cl_program program);

  cl_int clReleaseProgram(cl_program program);

  cl_int clBuildProgram(cl_program program, cl_uint num_devices, const cl_device_id* device_list,
                        const char* options,
                        void(CL_CALLBACK* pfn_notify)(cl_program program, void* user_data),
                        void* user_data);

  cl_int clCompileProgram(cl_program program, cl_uint num_devices, const cl_device_id* device_list,
                          const char* options, cl_uint num_input_headers,
                          const cl_program* input_headers, const char** header_include_names,
                          void(CL_CALLBACK* pfn_notify)(cl_program program, void* user_data),
                          void* user_data);

  cl_program clLinkProgram(cl_context context, cl_uint num_devices, const cl_device_id* device_list,
                           const char* options, cl_uint num_input_programs,
                           const cl_program* input_programs,
                           void(CL_CALLBACK* pfn_notify)(cl_program program, void* user_data),
                           void* user_data, cl_int* errcode_ret);

  cl_int clUnloadCompiler(void);

  cl_int clUnloadPlatform(cl_platform_id);

  cl_int clGetProgramInfo(cl_program program, cl_program_info param_name, size_t param_value_size,
                          void* param_value, size_t* param_value_size_ret);

  cl_int clGetProgramBuildInfo(cl_program program, cl_device_id device,
                               cl_program_build_info param_name, size_t param_value_size,
                               void* param_value, size_t* param_value_size_ret);

  cl_kernel clCreateKernel(cl_program program, const char* kernel_name, cl_int* errcode_ret);

  cl_int clCreateKernelsInProgram(cl_program program, cl_uint num_kernels, cl_kernel* kernels,
                                  cl_uint* num_kernels_ret);

  cl_int clRetainKernel(cl_kernel kernel);

  cl_int clReleaseKernel(cl_kernel kernel);

  cl_int clSetKernelArg(cl_kernel kernel, cl_uint arg_index, size_t arg_size,
                        const void* arg_value);

  cl_int clGetKernelInfo(cl_kernel kernel, cl_kernel_info param_name, size_t param_value_size,
                         void* param_value, size_t* param_value_size_ret);

  cl_int clGetKernelWorkGroupInfo(cl_kernel kernel, cl_device_id device,
                                  cl_kernel_work_group_info param_name, size_t param_value_size,
                                  void* param_value, size_t* param_value_size_ret);

  cl_int clWaitForEvents(cl_uint num_events, const cl_event* event_list);

  cl_int clGetEventInfo(cl_event evnt, cl_event_info param_name, size_t param_value_size,
                        void* param_value, size_t* param_value_size_ret);

  cl_int clRetainEvent(cl_event evnt);

  cl_int clReleaseEvent(cl_event evnt);

  cl_int clGetEventProfilingInfo(cl_event evnt, cl_profiling_info param_name,
                                 size_t param_value_size, void* param_value,
                                 size_t* param_value_size_ret);

  cl_int clFlush(cl_command_queue command_queue);

  cl_int clFinish(cl_command_queue command_queue);

  cl_int clEnqueueReadBuffer(cl_command_queue command_queue, cl_mem buffer, cl_bool blocking_read,
                             size_t offset, size_t cb, void* ptr, cl_uint num_events_in_wait_list,
                             const cl_event* event_wait_list, cl_event* evnt);

  cl_int clEnqueueWriteBuffer(cl_command_queue command_queue, cl_mem buffer, cl_bool blocking_write,
                              size_t offset, size_t cb, const void* ptr,
                              cl_uint num_events_in_wait_list, const cl_event* event_wait_list,
                              cl_event* evnt);

  cl_int clEnqueueCopyBuffer(cl_command_queue command_queue, cl_mem src_buffer, cl_mem dst_buffer,
                             size_t src_offset, size_t dst_offset, size_t cb,
                             cl_uint num_events_in_wait_list, const cl_event* event_wait_list,
                             cl_event* evnt);

  cl_int clEnqueueReadImage(cl_command_queue command_queue, cl_mem image, cl_bool blocking_read,
                            const size_t* origin, const size_t* region, size_t row_pitch,
                            size_t slice_pitch, void* ptr, cl_uint num_events_in_wait_list,
                            const cl_event* event_wait_list, cl_event* evnt);

  cl_int clEnqueueWriteImage(cl_command_queue command_queue, cl_mem image, cl_bool blocking_write,
                             const size_t* origin, const size_t* region, size_t input_row_pitch,
                             size_t input_slice_pitch, const void* ptr,
                             cl_uint num_events_in_wait_list, const cl_event* event_wait_list,
                             cl_event* evnt);

  cl_int clEnqueueCopyImage(cl_command_queue command_queue, cl_mem src_image, cl_mem dst_image,
                            const size_t* src_origin, const size_t* dst_origin,
                            const size_t* region, cl_uint num_events_in_wait_list,
                            const cl_event* event_wait_list, cl_event* evnt);

  cl_int clEnqueueCopyImageToBuffer(cl_command_queue command_queue, cl_mem src_image,
                                    cl_mem dst_buffer, const size_t* src_origin,
                                    const size_t* region, size_t dst_offset,
                                    cl_uint num_events_in_wait_list,
                                    const cl_event* event_wait_list, cl_event* evnt);

  cl_int clEnqueueCopyBufferToImage(cl_command_queue command_queue, cl_mem src_buffer,
                                    cl_mem dst_image, size_t src_offset, const size_t* dst_origin,
                                    const size_t* region, cl_uint num_events_in_wait_list,
                                    const cl_event* event_wait_list, cl_event* evnt);

  void* clEnqueueMapBuffer(cl_command_queue command_queue, cl_mem buffer, cl_bool blocking_map,
                           cl_map_flags map_flags, size_t offset, size_t cb,
                           cl_uint num_events_in_wait_list, const cl_event* event_wait_list,
                           cl_event* evnt, cl_int* errcode_ret);

  void* clEnqueueMapImage(cl_command_queue command_queue, cl_mem image, cl_bool blocking_map,
                          cl_map_flags map_flags, const size_t* origin, const size_t* region,
                          size_t* image_row_pitch, size_t* image_slice_pitch,
                          cl_uint num_events_in_wait_list, const cl_event* event_wait_list,
                          cl_event* evnt, cl_int* errcode_ret);

  cl_int clEnqueueUnmapMemObject(cl_command_queue command_queue, cl_mem memobj, void* mapped_ptr,
                                 cl_uint num_events_in_wait_list, const cl_event* event_wait_list,
                                 cl_event* evnt);

  cl_int clEnqueueNDRangeKernel(cl_command_queue command_queue, cl_kernel kernel, cl_uint work_dim,
                                const size_t* global_work_offset, const size_t* global_work_size,
                                const size_t* local_work_size, cl_uint num_events_in_wait_list,
                                const cl_event* event_wait_list, cl_event* evnt);

  cl_int clEnqueueTask(cl_command_queue command_queue, cl_kernel kernel,
                       cl_uint num_events_in_wait_list, const cl_event* event_wait_list,
                       cl_event* evnt);

  cl_int clEnqueueNativeKernel(cl_command_queue command_queue, void(CL_CALLBACK* user_func)(void*),
                               void* args, size_t cb_args, cl_uint num_mem_objects,
                               const cl_mem* mem_list, const void** args_mem_loc,
                               cl_uint num_events_in_wait_list, const cl_event* event_wait_list,
                               cl_event* evnt);

  cl_int clEnqueueMarker(cl_command_queue command_queue, cl_event* evnt);

  cl_int clEnqueueMarkerWithWaitList(cl_command_queue command_queue,
                                     cl_uint num_events_in_wait_list,
                                     const cl_event* event_wait_list, cl_event* evnt);

  cl_int clEnqueueWaitForEvents(cl_command_queue command_queue, cl_uint num_events,
                                const cl_event* event_list);

  cl_int clEnqueueBarrier(cl_command_queue command_queue);

  void* clGetExtensionFunctionAddress(const char* func_name);

  cl_int clEnqueueReadBufferRect(cl_command_queue command_queue, cl_mem buffer,
                                 cl_bool blocking_read, const size_t* buffer_origin,
                                 const size_t* host_origin, const size_t* region,
                                 size_t buffer_row_pitch, size_t buffer_slice_pitch,
                                 size_t host_row_pitch, size_t host_slice_pitch, void* ptr,
                                 cl_uint num_events_in_wait_list, const cl_event* event_wait_list,
                                 cl_event* evnt);

  cl_int clEnqueueWriteBufferRect(cl_command_queue command_queue, cl_mem buffer,
                                  cl_bool blocking_write, const size_t* buffer_origin,
                                  const size_t* host_origin, const size_t* region,
                                  size_t buffer_row_pitch, size_t buffer_slice_pitch,
                                  size_t host_row_pitch, size_t host_slice_pitch, const void* ptr,
                                  cl_uint num_events_in_wait_list, const cl_event* event_wait_list,
                                  cl_event* evnt);

  cl_int clEnqueueCopyBufferRect(cl_command_queue command_queue, cl_mem src_buffer,
                                 cl_mem dst_buffer, const size_t* src_origin,
                                 const size_t* dst_origin, const size_t* region,
                                 size_t src_row_pitch, size_t src_slice_pitch, size_t dst_row_pitch,
                                 size_t dst_slice_pitch, cl_uint num_events_in_wait_list,
                                 const cl_event* event_wait_list, cl_event* evnt);

  cl_mem clCreateImage(cl_context context, cl_mem_flags flags, const cl_image_format* image_format,
                       const cl_image_desc* image_desc, void* host_ptr, cl_int* errcode_ret);

  cl_mem clCreateSubBuffer(cl_mem mem, cl_mem_flags flags, cl_buffer_create_type buffer_create_type,
                           const void* buffer_create_info, cl_int* errcode_ret);

  cl_int clSetEventCallback(cl_event event, cl_int command_exec_callback_type,
                            void(CL_CALLBACK* pfn_event_notify)(cl_event event,
                                                                cl_int event_command_exec_status,
                                                                void* user_data),
                            void* user_data);

  cl_int clEnqueueFillImage(cl_command_queue command_queue, cl_mem image, void* ptr,
                            const size_t* origin, const size_t* region,
                            cl_uint num_events_in_wait_list, const cl_event* event_wait_list,
                            cl_event* evnt);

  cl_int clUnloadPlatformAMD(cl_platform_id id);

  cl_int clEnqueueWaitSignalAMD(cl_command_queue command_queue, cl_mem mem_object, cl_uint value,
                                cl_uint num_events, const cl_event* event_wait_list,
                                cl_event* event);

  cl_int clEnqueueWriteSignalAMD(cl_command_queue command_queue, cl_mem mem_object, cl_uint value,
                                 cl_ulong offset, cl_uint num_events, const cl_event* event_list,
                                 cl_event* event);

  cl_int clEnqueueMakeBuffersResidentAMD(cl_command_queue command_queue, cl_uint num_mem_objs,
                                         cl_mem* mem_objects, cl_bool blocking_make_resident,
                                         cl_bus_address_amd* bus_addresses, cl_uint num_events,
                                         const cl_event* event_list, cl_event* event);

  cl_int clEnqueueMigrateMemObjects(cl_command_queue command_queue, cl_uint num_mem_objects,
                                    const cl_mem* mem_objects, cl_mem_migration_flags flags,
                                    cl_uint num_events_in_wait_list,
                                    const cl_event* event_wait_list, cl_event* event);

  // CL-GL Extension: cl_khr_gl_sharing
  cl_int clGetGLContextInfoKHR(const cl_context_properties* properties,
                               cl_gl_context_info param_name, size_t param_value_size,
                               void* param_value, size_t* param_value_size_ret);

  cl_mem clCreateFromGLBuffer(cl_context context, cl_mem_flags flags, unsigned int bufobj,
                              int* errcode_ret);

  cl_mem clCreateFromGLTexture(cl_context context, cl_mem_flags flags, unsigned int texture_target,
                               int miplevel, unsigned int texture, cl_int* errcode_ret);

  cl_mem clCreateFromGLTexture2D(cl_context context, cl_mem_flags flags,
                                 unsigned int texture_target, int miplevel, unsigned int texture,
                                 cl_int* errcode_ret);

  cl_mem clCreateFromGLRenderbuffer(cl_context context, cl_mem_flags flags,
                                    unsigned int renderbuffer, cl_int* errcode_ret);

  cl_int clGetGLObjectInfo(cl_mem memobj, cl_gl_object_type* gl_object_type,
                           unsigned int* gl_object_name);

  cl_int clGetGLTextureInfo(cl_mem memobj, cl_gl_texture_info param_name, size_t param_value_size,
                            void* param_value, size_t* param_value_size_ret);

  cl_int clEnqueueAcquireGLObjects(cl_command_queue command_queue, cl_uint num_objects,
                                   const cl_mem* mem_objects, cl_uint num_events_in_wait_list,
                                   const cl_event* event_wait_list, cl_event* event);

  cl_int clEnqueueReleaseGLObjects(cl_command_queue command_queue, cl_uint num_objects,
                                   const cl_mem* mem_objects, cl_uint num_events_in_wait_list,
                                   const cl_event* event_wait_list, cl_event* event);

#if defined(CL_VERSION_2_0)
  cl_command_queue clCreateCommandQueueWithProperties(cl_context context, cl_device_id device,
                                                      const cl_queue_properties* properties,
                                                      cl_int* errcode_ret);

  void* clSVMAlloc(cl_context context, cl_svm_mem_flags flags, size_t size, cl_uint alignment);

  void clSVMFree(cl_context context, void* svm_pointer);

  cl_int clEnqueueSVMMap(cl_command_queue command_queue, cl_bool blocking_map, cl_map_flags flags,
                         void* svm_ptr, size_t size, cl_uint num_events_in_wait_list,
                         const cl_event* event_wait_list, cl_event* event);

  cl_int clEnqueueSVMUnmap(cl_command_queue command_queue, void* svm_ptr,
                           cl_uint num_events_in_wait_list, const cl_event* event_wait_list,
                           cl_event* event);

  cl_int clEnqueueSVMMemFill(cl_command_queue command_queue, void* svm_ptr, const void* pattern,
                             size_t pattern_size, size_t size, cl_uint num_events_in_wait_list,
                             const cl_event* event_wait_list, cl_event* event);

  cl_int clSetKernelArgSVMPointer(cl_kernel kernel, cl_uint arg_index, const void* arg_value);

  cl_mem clCreatePipe(cl_context context, cl_mem_flags flags, cl_uint packet_size,
                      cl_uint num_packets, const cl_pipe_properties* properties,
                      cl_int* errcode_ret);

  cl_int clGetPipeInfo(cl_mem pipe, cl_pipe_info param_name, size_t param_value_size,
                       void* param_value, size_t* param_value_size_ret);

#endif

  cl_perfcounter_amd clCreatePerfCounterAMD(cl_device_id device,
                                            cl_perfcounter_property* properties,
                                            cl_int* errcode_ret);

  cl_int clEnqueueBeginPerfCounterAMD(cl_command_queue command_queue, cl_uint num_perf_counters,
                                      cl_perfcounter_amd* perf_counters,
                                      cl_uint num_events_in_wait_list,
                                      const cl_event* event_wait_list, cl_event* event);

  cl_int clEnqueueEndPerfCounterAMD(cl_command_queue command_queue, cl_uint num_perf_counters,
                                    cl_perfcounter_amd* perf_counters,
                                    cl_uint num_events_in_wait_list,
                                    const cl_event* event_wait_list, cl_event* event);

  cl_int clGetPerfCounterInfoAMD(cl_perfcounter_amd perf_counter, cl_perfcounter_info param_name,
                                 size_t param_value_size, void* param_value,
                                 size_t* param_value_size_ret);

  cl_int clReleasePerfCounterAMD(cl_perfcounter_amd perf_counter);

  cl_int clRetainPerfCounterAMD(cl_perfcounter_amd perf_counter);

  cl_int clSetDeviceClockModeAMD(cl_device_id device,
                                 cl_set_device_clock_mode_input_amd set_clock_mode_input,
                                 cl_set_device_clock_mode_output_amd* set_clock_mode_Output);

 private:
  clEnqueueWaitSignalAMD_fn clEnqueueWaitSignalAMD_ptr;
  clEnqueueWriteSignalAMD_fn clEnqueueWriteSignalAMD_ptr;
  clEnqueueMakeBuffersResidentAMD_fn clEnqueueMakeBuffersResidentAMD_ptr;

  // Unload the platform
  clUnloadPlatformAMD_fn clUnloadPlatformAMD_ptr;

  // CL-GL Extension: cl_khr_gl_sharing
  clGetGLContextInfoKHR_fn clGetGLContextInfoKHR_ptr;
  clCreateFromGLBuffer_fn clCreateFromGLBuffer_ptr;
  clCreateFromGLTexture_fn clCreateFromGLTexture_ptr;
  clCreateFromGLTexture2D_fn clCreateFromGLTexture2D_ptr;
  clCreateFromGLRenderbuffer_fn clCreateFromGLRenderbuffer_ptr;
  clGetGLObjectInfo_fn clGetGLObjectInfo_ptr;
  clGetGLTextureInfo_fn clGetGLTextureInfo_ptr;
  clEnqueueAcquireGLObjects_fn clEnqueueAcquireGLObjects_ptr;
  clEnqueueReleaseGLObjects_fn clEnqueueReleaseGLObjects_ptr;

  // Performance counters
  clCreatePerfCounterAMD_fn clCreatePerfCounterAMD_ptr;
  clEnqueueBeginPerfCounterAMD_fn clEnqueueBeginPerfCounterAMD_ptr;
  clEnqueueEndPerfCounterAMD_fn clEnqueueEndPerfCounterAMD_ptr;
  clGetPerfCounterInfoAMD_fn clGetPerfCounterInfoAMD_ptr;
  clReleasePerfCounterAMD_fn clReleasePerfCounterAMD_ptr;
  clRetainPerfCounterAMD_fn clRetainPerfCounterAMD_ptr;
  // Set clockMode
  clSetDeviceClockModeAMD_fn clSetDeviceClockModeAMD_ptr;
};

#endif
