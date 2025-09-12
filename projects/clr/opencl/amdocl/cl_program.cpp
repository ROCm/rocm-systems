/* Copyright (c) 2008 - 2021 Advanced Micro Devices, Inc.

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
#include "vdi_common.hpp"
#include "platform/context.hpp"
#include "platform/program.hpp"
#include "platform/kernel.hpp"
#include "platform/sampler.hpp"
#include "cl_semaphore_amd.h"

#include <vector>

static amd::Program* createProgram(cl_context context, cl_uint num_devices,
                                   const cl_device_id* device_list, cl_int* errcode_ret) {
  // Create the program
  amd::Program* program = new amd::Program(*as_amd(context));
  if (program == NULL) {
    *not_null(errcode_ret) = CL_OUT_OF_HOST_MEMORY;
    return NULL;
  }

  // Add programs for all devices in the context.
  if (device_list == NULL) {
    const std::vector<amd::Device*>& devices = as_amd(context)->devices();
    for (const auto& it : devices) {
      if (program->addDeviceProgram(*it) == CL_OUT_OF_HOST_MEMORY) {
        *not_null(errcode_ret) = CL_OUT_OF_HOST_MEMORY;
        program->release();
        return NULL;
      }
    }
    return program;
  }

  *not_null(errcode_ret) = CL_SUCCESS;
  for (cl_uint i = 0; i < num_devices; ++i) {
    cl_device_id device = device_list[i];

    if (!is_valid(device) || !as_amd(context)->containsDevice(as_amd(device))) {
      *not_null(errcode_ret) = CL_INVALID_DEVICE;
      program->release();
      return NULL;
    }

    cl_int status = program->addDeviceProgram(*as_amd(device));
    if (status == CL_OUT_OF_HOST_MEMORY) {
      *not_null(errcode_ret) = CL_OUT_OF_HOST_MEMORY;
      program->release();
      return NULL;
    }
  }
  return program;
}

/*! \addtogroup API
 *  @{
 *
 *  \addtogroup CL_Programs
 *
 *  An OpenCL program consists of a set of kernels that are identified as
 *  functions declared with the __kernel qualifier in the program source.
 *  OpenCL programs may also contain auxiliary functions and constant data that
 *  can be used by __kernel functions. The program executable can be generated
 *  online or offline by the OpenCL compiler for the appropriate
 *  target device(s).
 *
 *  @{
 *
 *  \addtogroup CL_CreatingPrograms
 *  @{
 */

/*! \brief Create a program object for a context, and loads the source code
 *  specified by the text strings in the strings array into the program object.
 *
 *  \param context must be a valid OpenCL context.
 *
 *  \param count is the number of pointers in \a strings
 *
 *  \param strings is an array of \a count pointers to optionally
 *  null-terminated character strings that make up the source code.
 *
 *  \param lengths is an array with the number of chars in each string (the
 *  string length). If an element in lengths is zero, its accompanying string
 *  is null-terminated. If lengths is NULL, all strings in the strings argument
 *  are considered null-terminated.
 *
 *  \param errcode_ret will return an appropriate error code. If \a errcode_ret
 *  is NULL, no error code is returned.
 *
 *  \return A valid non-zero program object and errcode_ret is set to
 *  \a CL_SUCCESS if the program object is created successfully. It returns a
 *  NULL value with one of the following error values returned in
 *  \a errcode_ret:
 *  - CL_INVALID_CONTEXT if \a context is not a valid context.
 *  - CL_INVALID_VALUE if \a count is zero or if \a strings or any entry in
 *    \a strings is NULL.
 *  - CL_COMPILER_NOT_AVAILABLE if a compiler is not available.
 *  - CL_OUT_OF_HOST_MEMORY if there is a failure to allocate resources required
 *    by the runtime.
 *
 *  \version 1.0r33
 */
RUNTIME_ENTRY_RET(cl_program, clCreateProgramWithSource,
                  (cl_context context, cl_uint count, const char** strings, const size_t* lengths,
                   cl_int* errcode_ret)) {
  if (!is_valid(context)) {
    *not_null(errcode_ret) = CL_INVALID_CONTEXT;
    return (cl_program)0;
  }
  if (count == 0 || strings == NULL) {
    *not_null(errcode_ret) = CL_INVALID_VALUE;
    return (cl_program)0;
  }

  std::string sourceCode;
  for (cl_uint i = 0; i < count; ++i) {
    if (strings[i] == NULL) {
      *not_null(errcode_ret) = CL_INVALID_VALUE;
      return (cl_program)0;
    }
    if (lengths && lengths[i] != 0) {
      sourceCode.append(strings[i], lengths[i]);
    } else {
      sourceCode.append(strings[i]);
    }
  }
  if (sourceCode.empty()) {
    *not_null(errcode_ret) = CL_INVALID_VALUE;
    return (cl_program)0;
  }

  // Create the program
  amd::Program* program = new amd::Program(*as_amd(context), sourceCode, amd::Program::OpenCL_C);
  if (program == NULL) {
    *not_null(errcode_ret) = CL_OUT_OF_HOST_MEMORY;
    return (cl_program)0;
  }

  // Add programs for all devices in the context.
  const std::vector<amd::Device*>& devices = as_amd(context)->devices();
  for (const auto& it : devices) {
    if (program->addDeviceProgram(*it) == CL_OUT_OF_HOST_MEMORY) {
      *not_null(errcode_ret) = CL_OUT_OF_HOST_MEMORY;
      program->release();
      return (cl_program)0;
    }
  }

  *not_null(errcode_ret) = CL_SUCCESS;
  return as_cl(program);
}
RUNTIME_EXIT

/*! \brief Create a program object for a context, and loads the IL into the
 *  program object.
 *
 *  \param context must be a valid OpenCL context.
 *
 *  \param string is a pointer to IL.
 *
 *  \param length is the size in bytes of IL.
 *
 *  \param errcode_ret will return an appropriate error code. If \a errcode_ret
 *  is NULL, no error code is returned.
 *
 *  \return A valid non-zero program object and errcode_ret is set to
 *  \a CL_SUCCESS if the program object is created successfully. It returns a
 *  NULL value with one of the following error values returned in
 *  \a errcode_ret:
 *  - CL_INVALID_CONTEXT if \a context is not a valid context.
 *  - CL_INVALID_VALUE if \a il is NULL or \a length is zero.
 *  - CL_INVALID_VALUE if the \a length-byte memory pointed to by \a il does
 *   not contain well-formed intermediate language input appropriate for the
 *   deployment environment in which the OpenCL platform is running.
 *  - CL_OUT_OF_RESOURCES if there is a failure to allocate resources required
 *   by the OpenCL implementation on the device.
 *  - CL_OUT_OF_HOST_MEMORY if there is a failure to allocate resources
 *   required by the OpenCL implementation on the host.
 *
 *  \version 1.0r33
 */
RUNTIME_ENTRY_RET(cl_program, clCreateProgramWithIL,
                  (cl_context context, const void* il, size_t length, cl_int* errcode_ret)) {
  if (!is_valid(context)) {
    *not_null(errcode_ret) = CL_INVALID_CONTEXT;
    return (cl_program)0;
  }
  if (length == 0 || il == NULL) {
    *not_null(errcode_ret) = CL_INVALID_VALUE;
    return (cl_program)0;
  }

  // Create the program
  amd::Program* program = new amd::Program(*as_amd(context), amd::Program::SPIRV);
  if (program == NULL) {
    *not_null(errcode_ret) = CL_OUT_OF_HOST_MEMORY;
    return (cl_program)0;
  }

  // Add programs for all devices in the context.
  const std::vector<amd::Device*>& devices = as_amd(context)->devices();
  for (const auto& it : devices) {
    if (program->addDeviceProgram(*it, il, length) == CL_OUT_OF_HOST_MEMORY) {
      *not_null(errcode_ret) = CL_OUT_OF_HOST_MEMORY;
      program->release();
      return (cl_program)0;
    }
  }

  *not_null(errcode_ret) = CL_SUCCESS;
  return as_cl(program);
}
RUNTIME_EXIT

/*! \brief Create a program object for a context, and loads the binary images
 *  into the program object.
 *
 *  \param context must be a valid OpenCL context.
 *
 *  \param device_list is a pointer to a list of devices that are in context.
 *  \a device_list must be a non-NULL value. The binaries are loaded for devices
 *  specified in this list.
 *
 *  \param num_devices is the number of devices listed in \a device_list.
 *
 *  \param device_list The devices associated with the program object. The
 *  list of devices specified by \a device_list must be devices associated with
 *  \a context.
 *
 *  \param lengths is an array of the size in bytes of the program binaries to
 *  be loaded for devices specified by \a device_list.
 *
 *  \param binaries is an array of pointers to program binaries to be loaded
 *  for devices specified by \a device_list. For each device given by
 *  \a device_list[i], the pointer to the program binary for that device is
 *  given by \a binaries[i] and the length of this corresponding binary is given
 *  by \a lengths[i]. \a lengths[i] cannot be zero and \a binaries[i] cannot be
 *  a NULL pointer. The program binaries specified by binaries contain the bits
 *  that describe the program executable that will be run on the device(s)
 *  associated with context. The program binary can consist of either or both:
 *  - Device-specific executable(s)
 *  - Implementation specific intermediate representation (IR) which will be
 *    converted to the device-specific executable.
 *
 *  \param binary_status returns whether the program binary for each device
 *  specified in \a device_list was loaded successfully or not. It is an array
 *  of \a num_devices entries and returns CL_SUCCESS in \a binary_status[i] if
 *  binary was successfully loaded for device specified by \a device_list[i];
 *  otherwise returns CL_INVALID_VALUE if \a lengths[i] is zero or if
 *  \a binaries[i] is a NULL value or CL_INVALID_BINARY in \a binary_status[i]
 *  if program binary is not a valid binary for the specified device.
 *  If \a binary_status is NULL, it is ignored.
 *
 *  \param errcode_ret will return an appropriate error code. If \a errcode_ret
 *  is NULL, no error code is returned.
 *
 *  \return A valid non-zero program object and \a errcode_ret is set to
 *  CL_SUCCESS if the program object is created successfully. It returns a NULL
 *  value with one of the following error values returned in \a errcode_ret:
 *  - CL_INVALID_CONTEXT if \a context is not a valid context.
 *  - CL_INVALID_VALUE if \a device_list is NULL or \a num_devices is zero.
 *  - CL_INVALID_DEVICE if OpenCL devices listed in \a device_list are not in
 *    the list of devices associated with \a context
 *  - CL_INVALID_VALUE if \a lengths or \a binaries are NULL or if any entry
 *    in \a lengths[i] is zero or \a binaries[i] is NULL.
 *  - CL_INVALID_BINARY if an invalid program binary was encountered for any
 *    device. \a binary_status will return specific status for each device.
 *  - CL_OUT_OF_HOST_MEMORY if there is a failure to allocate resources required
 *    by the runtime.
 *
 *  \version 1.0r33
 */
RUNTIME_ENTRY_RET(cl_program, clCreateProgramWithBinary,
                  (cl_context context, cl_uint num_devices, const cl_device_id* device_list,
                   const size_t* lengths, const unsigned char** binaries, cl_int* binary_status,
                   cl_int* errcode_ret)) {
  if (!is_valid(context)) {
    *not_null(errcode_ret) = CL_INVALID_CONTEXT;
    return (cl_program)0;
  }
  if (num_devices == 0 || device_list == NULL || binaries == NULL || lengths == NULL) {
    *not_null(errcode_ret) = CL_INVALID_VALUE;
    return (cl_program)0;
  }

  amd::Program* program = new amd::Program(*as_amd(context));
  if (program == NULL) {
    *not_null(errcode_ret) = CL_OUT_OF_HOST_MEMORY;
    return (cl_program)0;
  }

  *not_null(errcode_ret) = CL_SUCCESS;
  for (cl_uint i = 0; i < num_devices; ++i) {
    cl_device_id device = device_list[i];

    if (!is_valid(device) || !as_amd(context)->containsDevice(as_amd(device))) {
      *not_null(errcode_ret) = CL_INVALID_DEVICE;
      program->release();
      return (cl_program)0;
    }
    if (binaries[i] == NULL || lengths[i] == 0) {
      if (binary_status != NULL) {
        binary_status[i] = CL_INVALID_VALUE;
      }
      *not_null(errcode_ret) = CL_INVALID_VALUE;
      continue;
    }

    cl_int status = program->addDeviceProgram(*as_amd(device), binaries[i], lengths[i]);

    *not_null(errcode_ret) = status;

    if (status == CL_OUT_OF_HOST_MEMORY) {
      *not_null(errcode_ret) = CL_OUT_OF_HOST_MEMORY;
      program->release();
      return (cl_program)0;
    }

    if (binary_status != NULL) {
      binary_status[i] = status;
    }
  }
  return as_cl(program);
}
RUNTIME_EXIT

RUNTIME_ENTRY_RET(cl_program, clCreateProgramWithAssemblyAMD,
                  (cl_context context, cl_uint count, const char** strings, const size_t* lengths,
                   cl_int* errcode_ret)) {
  if (!is_valid(context)) {
    *not_null(errcode_ret) = CL_INVALID_CONTEXT;
    return (cl_program)0;
  }
  if (count == 0 || strings == NULL) {
    *not_null(errcode_ret) = CL_INVALID_VALUE;
    return (cl_program)0;
  }

  std::string assembly;
  for (cl_uint i = 0; i < count; ++i) {
    if (strings[i] == NULL) {
      *not_null(errcode_ret) = CL_INVALID_VALUE;
      return (cl_program)0;
    }
    if (lengths && lengths[i] != 0) {
      assembly.append(strings[i], lengths[i]);
    } else {
      assembly.append(strings[i]);
    }
  }
  if (assembly.empty()) {
    *not_null(errcode_ret) = CL_INVALID_VALUE;
    return (cl_program)0;
  }

  // Create the program
  amd::Program* program = new amd::Program(*as_amd(context), assembly, amd::Program::Assembly);
  if (program == NULL) {
    *not_null(errcode_ret) = CL_OUT_OF_HOST_MEMORY;
    return (cl_program)0;
  }

  // Add programs for all devices in the context.
  const std::vector<amd::Device*>& devices = as_amd(context)->devices();
  for (const auto& it : devices) {
    if (program->addDeviceProgram(*it) == CL_OUT_OF_HOST_MEMORY) {
      *not_null(errcode_ret) = CL_OUT_OF_HOST_MEMORY;
      program->release();
      return (cl_program)0;
    }
  }

  *not_null(errcode_ret) = CL_SUCCESS;
  return as_cl(program);
}
RUNTIME_EXIT

/*! \brief Increment the program reference count.
 *
 *  clCreateProgram does an implicit retain.
 *
 *  \return CL_SUCCESS if the function is executed successfully. It returns
 *  CL_INVALID_PROGRAM if \a program is not a valid program object.
 *
 *  \version 1.0r33
 */
RUNTIME_ENTRY(cl_int, clRetainProgram, (cl_program program)) {
  if (!is_valid(program)) {
    return CL_INVALID_PROGRAM;
  }
  as_amd(program)->retain();
  return CL_SUCCESS;
}
RUNTIME_EXIT

/*! \brief Decrement the program reference count.
 *
 *  The program object is deleted after all kernel objects associated with
 *  \a program have been deleted and the program reference count becomes zero.
 *
 *  \return CL_SUCCESS if the function is executed successfully. It returns
 *  CL_INVALID_PROGRAM if \a program is not a valid program object.
 *
 *  \version 1.0r33
 */
RUNTIME_ENTRY(cl_int, clReleaseProgram, (cl_program program)) {
  if (!is_valid(program)) {
    return CL_INVALID_PROGRAM;
  }
  as_amd(program)->release();
  return CL_SUCCESS;
}
RUNTIME_EXIT

/*! @}
 *  \addtogroup CL_Build
 *  @{
 */

/*! \brief Build (compile & link) a program executable from the program source
 *  or binary for all the devices or a specific device(s) in the OpenCL context
 *  associated with program.
 *
 *  OpenCL allows program executables to be built using the sources or binaries.
 *
 *  \param program is the program object.
 *
 *  \param device_list is a pointer to a list of devices associated with
 *  \a program. If \a device_list is a NULL value, the program executable is
 *  built for all devices associated with \a program for which a source or
 *  binary has been loaded. If \a device_list is a non-NULL value, the program
 *  executable is built for devices specified in this list for which a source
 *  or binary has been loaded.
 *
 *  \param num_devices is the number of devices listed in \a device_list.
 *
 *  \param options is a pointer to a string that describes the build options to
 *  be used for building the program executable.
 *
 *  \param pfn_notify is a function pointer to a notification routine. The
 *  notification routine allows an application to register a callback function
 *  which will be called when the program executable has been built
 *  (successfully or unsuccessfully). If \a pfn_notify is not NULL,
 *  clBuildProgram does not need to wait for the build to complete and can
 *  return immediately. If \a pfn_notify is NULL, clBuildProgram does not
 *  return until the build has completed. This callback function may be called
 *  asynchronously by the OpenCL implementation. It is the application's
 *  responsibility to ensure that the callback function is thread-safe.
 *
 *  \param user_data will be passed as the argument when \a pfn_notify is
 *  called. \a user_data can be NULL.
 *
 *  \return One of the following values:
 *  - CL_SUCCESS if the function is executed successfully
 *  - CL_INVALID_PROGRAM if \a program is not a valid program object
 *  - CL_INVALID_VALUE if \a device_list is NULL and \a num_devices is greater
 *    than zero, or if \a device_list is not NULL and \a num_devices is zero,
 *  - CL_INVALID_DEVICE if OpenCL devices listed in \a device_list are not in
 *    the list of devices associated with \a program
 *  - CL_INVALID_BINARY if \a program is created with clCreateWithProgramBinary
 *    and devices listed in \a device_list do not have a valid program binary
 *    loaded
 *  - CL_INVALID_BUILD_OPTIONS if the build options specified by \a options are
 *    invalid
 *  - CL_INVALID_OPERATION if the build of a program executable for any of the
 *    devices listed in \a device_list by a previous call to clBuildProgram for
 *    \a program has not completed
 *  - CL_OUT_OF_HOST_MEMORY if there is a failure to allocate resources required
 *    by the runtime.
 *
 *  \version 1.0r33
 */
RUNTIME_ENTRY(cl_int, clBuildProgram,
              (cl_program program, cl_uint num_devices, const cl_device_id* device_list,
               const char* options,
               void(CL_CALLBACK* pfn_notify)(cl_program program, void* user_data),
               void* user_data)) {
  if (!is_valid(program)) {
    return CL_INVALID_PROGRAM;
  }
  if ((num_devices > 0 && device_list == NULL) || (num_devices == 0 && device_list != NULL)) {
    return CL_INVALID_VALUE;
  }
  if (pfn_notify == nullptr && user_data != nullptr) {
    return CL_INVALID_VALUE;
  }

  amd::Program* amdProgram = as_amd(program);

  if (device_list == NULL) {
    // build for all devices in the context.
    return amdProgram->build(amdProgram->context().devices(), options, pfn_notify, user_data);
  }

  std::vector<amd::Device*> devices(num_devices);
  for (cl_uint i = 0; i < num_devices; ++i) {
    amd::Device* device = as_amd(device_list[i]);
    if (!amdProgram->context().containsDevice(device)) {
      return CL_INVALID_DEVICE;
    }
    devices[i] = device;
  }
  return amdProgram->build(devices, options, pfn_notify, user_data);
}
RUNTIME_EXIT

/*! \brief compiles a program's source for all the devices or a specific
 *  device(s) in the OpenCL context associated with program. The pre-processor
 *  runs before the program sources are compiled.
 *  The compiled binary is built for all devices associated with program or
 *  the list of devices specified. The compiled binary can be queried using
 *  \a clGetProgramInfo(program, CL_PROGRAM_BINARIES, ...) and can be specified
 *  to \a clCreateProgramWithBinary to create a new program object.
 *
 *  \param program is the program object that is the compilation target.
 *
 *  \param device_list is a pointer to a list of devices associated with program.
 *  If device_list is a NULL value, the compile is performed for all devices
 *  associated with program. If device_list is a non-NULL value, the compile is
 *  performed for devices specified in this list.
 *
 *  \param num_devices is the number of devices listed in \a device_list.
 *
 *  \param options is a pointer to a null-terminated string of characters that
 *  describes the compilation options to be used for building the program
 *  executable. The list of supported options is as described in section 5.6.4.
 *
 *  \param num_input_headers specifies the number of programs that describe
 *  headers in the array referenced by input_headers.
 *
 *  \param input_headers is an array of program embedded headers created with
 *  \a clCreateProgramWithSource.
 *
 *  \param header_include_names is an array that has a one to one correspondence
 *  with input_headers.
 *  Each entry in \a header_include_names specifies the include name used by
 *  source in program that comes from an embedded header. The corresponding entry
 *  in input_headers identifies the program object which contains the header
 *  source to be used. The embedded headers are first searched before the headers
 *  in the list of directories specified by the -I compile option (as described in
 *  section 5.6.4.1). If multiple entries in header_include_names refer to the same
 *  header name, the first one encountered will be used.
 *
 *  \param pfn_notify is a function pointer to a notification routine. The
 *  notification routine is a callback function that an application can register
 *  and which will be called when the program executable has been built
 *  (successfully or unsuccessfully). If pfn_notify is not NULL,
 *  \a clCompileProgram does not need to wait for the compiler to complete and can
 *  return immediately. If \a pfn_notify is NULL, \a clCompileProgram does not
 *  return until the compiler has completed. This callback function may be called
 *  asynchronously by the OpenCL implementation. It is the application's
 *  responsibility to ensure that the callback function is thread-safe.
 *
 *  \param user_data will be passed as an argument when pfn_notify is called.
 *  \a user_data can be NULL.
 *
 *  \return CL_SUCCESS if the function is executed successfully. Otherwise, it
 *  returns one of the following errors:
 *  - CL_INVALID_PROGRAM if program is not a valid program object.
 *  - CL_INVALID_VALUE if device_list is NULL and num_devices is greater than
 *    zero, or if \a device_list is not NULL and \a num_devices is zero.
 *  - CL_INVALID_VALUE if num_input_headers is zero and \a header_include_names
 *    or input_headers are not NULL or if num_input_headers is not zero and
 *    \a header_include_names or input_headers are NULL.
 *  - CL_INVALID_VALUE if \a pfn_notify is NULL but \a user_data is not NULL.
 *  - CL_INVALID_DEVICE if OpenCL devices listed in device_list are not in the
 *    list of devices associated with program
 *  - CL_INVALID_COMPILER_OPTIONS if the compiler options specified by options
 *    are invalid.
 *  - CL_INVALID_OPERATION if the compilation or build of a program executable
 *    for any of the devices listed in device_list by a previous call to
 *    \a clCompileProgram or \a clBuildProgram for program has not completed.
 *  - CL_COMPILER_NOT_AVAILABLE if a compiler is not available i.e.
 *  - CL_DEVICE_COMPILER_AVAILABLE specified in table 4.3 is set to CL_FALSE.
 *  - CL_COMPILE_PROGRAM_FAILURE if there is a failure to compile the program
 *    source. This error will be returned if clCompileProgram does not return
 *    until the compile has completed.
 *  - CL_INVALID_OPERATION if there are kernel objects attached to program.
 *  - CL_INVALID_OPERATION if program has no source i.e. it has not been created
 *    with \a clCreateProgramWithSource.
 *  - CL_OUT_OF_RESOURCES if there is a failure to allocate resources required
 *    by the OpenCL implementation on the device.
 *  - CL_OUT_OF_HOST_MEMORY if there is a failure to allocate resources required
 *    by the OpenCL implementation on the host.
 *
 *  \version 1.2r07
 */
RUNTIME_ENTRY(cl_int, clCompileProgram,
              (cl_program program, cl_uint num_devices, const cl_device_id* device_list,
               const char* options, cl_uint num_input_headers, const cl_program* input_headers,
               const char** header_include_names,
               void(CL_CALLBACK* pfn_notify)(cl_program program, void* user_data),
               void* user_data)) {
  if (!is_valid(program)) {
    return CL_INVALID_PROGRAM;
  }
  if ((num_devices > 0 && device_list == NULL) || (num_devices == 0 && device_list != NULL)) {
    return CL_INVALID_VALUE;
  }
  if ((num_input_headers > 0 && (input_headers == NULL || header_include_names == NULL)) ||
      (num_input_headers == 0 && (input_headers != NULL || header_include_names != NULL))) {
    return CL_INVALID_VALUE;
  }
  if (pfn_notify == NULL && user_data != NULL) {
    return CL_INVALID_VALUE;
  }

  amd::Program* amdProgram = as_amd(program);
  if (amdProgram->referenceCount() > 1) {
    return CL_INVALID_OPERATION;
  }

  std::vector<const amd::Program*> headerPrograms(num_input_headers);
  for (cl_uint i = 0; i < num_input_headers; ++i) {
    if (!is_valid(input_headers[i])) {
      return CL_INVALID_OPERATION;
    }
    const amd::Program* headerProgram = as_amd(input_headers[i]);
    headerPrograms[i] = headerProgram;
  }

  if (device_list == NULL) {
    // compile for all devices in the context.
    return amdProgram->compile(amdProgram->context().devices(), num_input_headers, headerPrograms,
                               header_include_names, options, pfn_notify, user_data);
  }

  std::vector<amd::Device*> devices(num_devices);

  for (cl_uint i = 0; i < num_devices; ++i) {
    amd::Device* device = as_amd(device_list[i]);
    if (!amdProgram->context().containsDevice(device)) {
      return CL_INVALID_DEVICE;
    }
    devices[i] = device;
  }

  return amdProgram->compile(devices, num_input_headers, headerPrograms, header_include_names,
                             options, pfn_notify, user_data);
}
RUNTIME_EXIT

/*! \brief links a set of compiled program objects and libraries for all
 *  the devices or a specific device(s) in the OpenCL context and creates
 *  an executable. clLinkProgram creates a new program object which contains
 *  this executable. The executable binary can be queried using
 *  \a clGetProgramInfo(program, CL_PROGRAM_BINARIES, ...) and can be specified
 *  to \a clCreateProgramWithBinary to create a new program object.
 *  The devices associated with the returned program object will be the list
 *  of devices specified by device_list or if device_list is NULL it will be
 *  the list of devices associated with context.
 *
 *  \param context must be a valid OpenCL context.
 *
 *  \param device_list is a pointer to a list of devices that are in context.
 *  If device_list is a NULL value, the link is performed for all devices
 *  associated with context for which a compiled object is available.
 *  If device_list is a non-NULL value, the compile is performed for devices
 *  specified in this list for which a source has been loaded.
 *
 *  \param num_devices is the number of devices listed in device_list.
 *
 *  \param options is a pointer to a null-terminated string of characters
 *  that describes the link options to be used for building the program
 *  executable. The list of supported options is as described in section 5.6.5.
 *
 *  \param num_input_programs specifies the number of programs in array
 *  referenced by input_programs.
 *
 *  \param input_programs is an array of program objects that are compiled
 *  binaries or libraries that are to be linked to create the program executable.
 *  For each device in device_list or if device_list is NULL the list of devices
 *  associated with context, the following cases occur:
 *  All programs specified by input_programs contain a compiled binary or
 *  library for the device. In this case, a link is performed to generate
 *  a program executable for this device. None of the programs contain
 *  a compiled binary or library for that device. In this case, no link is
 *  performed and there will be no program executable generated for this device.
 *  All other cases will return a CL_INVALID_OPERATION error.
 *
 *  \param pfn_notify is a function pointer to a notification routine.
 *  The notification routine is a callback function that an application can
 *  register and which will be called when the program executable has been built
 *  (successfully or unsuccessfully). If \a pfn_notify is not NULL,
 *  \a clLinkProgram does not need to wait for the linker to complete and can
 *  return immediately. Once the linker has completed, the \a pfn_notify
 *  callback function is called with a valid program object (if the link was
 *  successful) or NULL (if the link encountered a failure). This callback
 *  function may be called asynchronously by the OpenCL implementation. It is
 *  the application's responsibility to ensure that the callback function is
 *  thread-safe. If \a pfn_notify is NULL, \a clLinkProgram does not return
 *  until the linker has completed. clLinkProgram returns a valid non-zero
 *  program object (if the link was successful) or NULL (if the link
 *  encountered a failure).
 *
 *  \a user_data will be passed as an argument when \a pfn_notify is called.
 *  user_data can be NULL.
 *
 *  \return a valid non-zero program object and errcode_ret is set to CL_SUCCESS
 *  if the link was successful in generating a program executable for at least
 *  one device and the program object was created successfully. If \a pfn_notify
 *  is not NULL, \a clLinkProgram returns a NULL program object and
 *  \a errcode_ret is set to CL_SUCCESS if the function was executed
 *  successfully. Otherwise, it returns one of the following errors:
 *  - CL_INVALID_CONTEXT if context is not a valid context.
 *  - CL_INVALID_VALUE if device_list is NULL and num_devices is greater than
 *    zero, or if \a device_list is not NULL and \a num_devices is zero.
 *  - CL_INVALID_VALUE if \a num_input_programs is zero and \a input_programs
 *    is NULL or if \a num_input_programs is zero and \a input_programs is not
 *    NULL or if \a num_input_programs is not zero and \a input_programs is NULL.
 *  - CL_INVALID_PROGRAM if programs specified in \a input_programs are not
 *    valid program objects.
 *  - CL_INVALID_VALUE if \a pfn_notify is NULL but \a user_data is not NULL.
 *  - CL_INVALID_DEVICE if OpenCL devices listed in \a device_list are not in
 *    the list of devices associated with context
 *  - CL_INVALID_LINKER_OPTIONS if the linker options specified by options are
 *    invalid.
 *  - CL_INVALID_OPERATION if the compilation or build of a program executable
 *    for any of the devices listed in \a device_list by a previous call to
 *    clCompileProgram or clBuildProgram for program has not completed.
 *  - CL_INVALID_OPERATION if the rules for devices containing compiled binaries
 *    or libraries as described in \a input_programs argument above are
 *    not followed.
 *  - CL_LINKER_NOT_AVAILABLE if a linker is not available i.e.
 *  - CL_DEVICE_LINKER_AVAILABLE specified in table 4.3 is set to CL_FALSE.
 *  - CL_LINK_PROGRAM_FAILURE if there is a failure to link the compiled
 *    binaries and/or libraries.
 *  - CL_OUT_OF_RESOURCES if there is a failure to allocate resources required
 *    by the OpenCL implementation on the device.
 *  - CL_OUT_OF_HOST_MEMORY if there is a failure to allocate resources required
 *    by the OpenCL implementation on the host.
 *
 *  \version 1.2r07
 */
RUNTIME_ENTRY_RET(cl_program, clLinkProgram,
                  (cl_context context, cl_uint num_devices, const cl_device_id* device_list,
                   const char* options, cl_uint num_input_programs,
                   const cl_program* input_programs,
                   void(CL_CALLBACK* pfn_notify)(cl_program program, void* user_data),
                   void* user_data, cl_int* errcode_ret)) {
  if (!is_valid(context)) {
    *not_null(errcode_ret) = CL_INVALID_CONTEXT;
    return (cl_program)0;
  }

  if ((num_devices > 0 && device_list == NULL) || (num_devices == 0 && device_list != NULL)) {
    *not_null(errcode_ret) = CL_INVALID_VALUE;
    return (cl_program)0;
  }

  if (num_input_programs == 0 || input_programs == NULL) {
    *not_null(errcode_ret) = CL_INVALID_VALUE;
    return (cl_program)0;
  }

  if (pfn_notify == NULL && user_data != NULL) {
    *not_null(errcode_ret) = CL_INVALID_VALUE;
    return (cl_program)0;
  }

  std::vector<amd::Program*> inputPrograms(num_input_programs);
  for (cl_uint i = 0; i < num_input_programs; ++i) {
    if (!is_valid(input_programs[i])) {
      *not_null(errcode_ret) = CL_INVALID_PROGRAM;
      return (cl_program)0;
    }
    amd::Program* inputProgram = as_amd(input_programs[i]);
    inputPrograms[i] = inputProgram;
  }

  amd::Program* program = createProgram(context, num_devices, device_list, errcode_ret);
  if (program == NULL) return (cl_program)0;

  *not_null(errcode_ret) = CL_SUCCESS;
  cl_int status;

  if (device_list == NULL) {
    // compile for all devices in the context.
    status = program->link(as_amd(context)->devices(), num_input_programs, inputPrograms, options,
                           pfn_notify, user_data);
  } else {
    std::vector<amd::Device*> devices(num_devices);

    for (cl_uint i = 0; i < num_devices; ++i) {
      amd::Device* device = as_amd(device_list[i]);
      if (!as_amd(context)->containsDevice(device)) {
        program->release();
        *not_null(errcode_ret) = CL_INVALID_DEVICE;
        return (cl_program)0;
      }
      devices[i] = device;
    }

    status =
        program->link(devices, num_input_programs, inputPrograms, options, pfn_notify, user_data);
  }
  *not_null(errcode_ret) = status;
  if (status == CL_SUCCESS) {
    return as_cl(program);
  }

  program->release();
  return (cl_program)0;
}
RUNTIME_EXIT

/*! \brief creates a program object for a context, and loads the information
 *   related to the built-in kernels into a program object.
 *
 *  \param context must be a valid OpenCL context.
 *
 *  \param num_devices is the number of devices listed in device_list.
 *
 *  \param device_list is a pointer to a list of devices that are in context.
 *  \a device_list must be a non-NULL value. The built-in kernels are loaded
 *  for devices specified in this list. The devices associated with the
 *  program object will be the list of devices specified by \a device_list.
 *  The list of devices specified by \a device_list must be devices associated
 *  with context.
 *
 *  \param kernel_names is a semi-colon separated list of built-in kernel names.
 *
 *  \return a valid non-zero program object and \a errcode_ret is set to
 *  CL_SUCCESS if the program object is created successfully. Otherwise, it
 *  returns a NULL value with one of the following error values returned
 *  in errcode_ret:
 *  - CL_INVALID_CONTEXT if context is not a valid context.
 *  - CL_INVALID_VALUE if device_list is NULL or num_devices is zero.
 *  - CL_INVALID_VALUE if kernel_names is NULL or kernel_names contains a kernel
 *    name that is not supported by any of the devices in \a device_list.
 *  - CL_INVALID_DEVICE if devices listed in device_list are not in the list
 *    of devices associated with context.
 *  - CL_OUT_OF_RESOURCES if there is a failure to allocate resources required
 *    by the OpenCL implementation on the device.
 *  - CL_OUT_OF_HOST_MEMORY if there is a failure to allocate resources required
 *    by the OpenCL implementation on the host.
 *
 *  \version 1.2r07
 */
RUNTIME_ENTRY_RET(cl_program, clCreateProgramWithBuiltInKernels,
                  (cl_context context, cl_uint num_devices, const cl_device_id* device_list,
                   const char* kernel_names, cl_int* errcode_ret)) {
  //!@todo Add implementation
  amd::Program* program = NULL;
  Unimplemented();
  return as_cl(program);
}
RUNTIME_EXIT

/*! @}
 *  \addtogroup CL_Unloading
 *  @{
 */

/*! \brief Allows the implementation to release the resources allocated by
 *  the OpenCL compiler for platform. This is a hint from the application
 *  and does not guarantee that the compiler will not be used in the future
 *  or that the compiler will actually be unloaded by the implementation.
 *  Calls to \a clBuildProgram, \a clCompileProgram or \a clLinkProgram after
 *  \a clUnloadPlatformCompiler will reload the compiler,
 *  if necessary, to build the appropriate program executable.
 *
 *  \return CL_SUCCESS if the function is executed successfully.
 *  Otherwise, it returns one of the following errors:
 *  - CL_INVALID_PLATFORM if platform is not a valid platform.
 *
 *  \version 1.2r07
 */
RUNTIME_ENTRY(cl_int, clUnloadPlatformCompiler, (cl_platform_id platform)) {
  if (platform != NULL && platform != AMD_PLATFORM) {
    return CL_INVALID_PLATFORM;
  }

  //! @todo: Implement Compiler::unload()
  return CL_SUCCESS;
}
RUNTIME_EXIT

/*! \brief Allow to runtime to release the resources allocated by the OpenCL
 *  compiler.
 *
 *  This is a hint from the application and does not guarantee that the compiler
 *  will not be used in the future or that the compiler will actually be
 *  unloaded by the implementation.
 *
 *  Calls to clBuildProgram after clUnloadCompiler may reload the compiler,
 *  if necessary, to build the appropriate program executable.
 *
 *  \return This call currently always returns CL_SUCCESS
 *
 *  \version 1.0r33
 */
RUNTIME_ENTRY(cl_int, clUnloadCompiler, (void)) {
  //! @todo: Implement Compiler::unload()
  return CL_SUCCESS;
}
RUNTIME_EXIT

/*! @}
 *  \addtogroup CL_ProgramQueries
 *  @{
 */

/*! \brief Return information about the program object.
 *
 *  \param program specifies the program object being queried.
 *
 *  \param param_name specifies the information to query.
 *
 *  \param param_value is a pointer to memory where the appropriate result
 *  being queried is returned. If \a param_value is NULL, it is ignored.
 *
 *  \param param_value_size is used to specify the size in bytes of memory
 *  pointed to by \a param_value. This size must be >= size of return type.
 *
 *  \param param_value_size_ret returns the actual size in bytes of data copied
 *  to \a param_value. If \a param_value_size_ret is NULL, it is ignored.
 *
 *  \return One of the following values:
 *  - CL_SUCCESS if the function is executed successfully
 *  - CL_INVALID_VALUE if \a param_name is not valid, or if size in bytes
 *    specified by \a param_value_size is < size of return type and
 *    \a param_value is not NULL
 *  - CL_INVALID_PROGRAM_EXECUTABLE if param_name is
 *    CL_PROGRAM_NUM_KERNELS or CL_PROGRAM_KERNEL_NAMES and a successful
 *    program executable has not been built for at least one device in the list
 *    of devices associated with program.
 *  - CL_INVALID_PROGRAM if \a program is a not a valid program object
 *
 *  \version 1.2r07
 */
RUNTIME_ENTRY(cl_int, clGetProgramInfo,
              (cl_program program, cl_program_info param_name, size_t param_value_size,
               void* param_value, size_t* param_value_size_ret)) {
  if (!is_valid(program)) {
    return CL_INVALID_PROGRAM;
  }

  switch (param_name) {
    case CL_PROGRAM_REFERENCE_COUNT: {
      cl_uint count = as_amd(program)->referenceCount();
      return amd::clGetInfo(count, param_value_size, param_value, param_value_size_ret);
    }
    case CL_PROGRAM_CONTEXT: {
      cl_context context = const_cast<cl_context>(as_cl(&as_amd(program)->context()));
      return amd::clGetInfo(context, param_value_size, param_value, param_value_size_ret);
    }
    case CL_PROGRAM_NUM_DEVICES: {
      cl_uint numDevices = (cl_uint)as_amd(program)->deviceList().size();
      return amd::clGetInfo(numDevices, param_value_size, param_value, param_value_size_ret);
    }
    case CL_PROGRAM_DEVICES: {
      const amd::Program::devicelist_t& devices = as_amd(program)->deviceList();
      const size_t numDevices = devices.size();
      const size_t valueSize = numDevices * sizeof(cl_device_id);

      if (param_value != NULL && param_value_size < valueSize) {
        return CL_INVALID_VALUE;
      }
      *not_null(param_value_size_ret) = valueSize;
      if (param_value != NULL) {
        cl_device_id* device_list = (cl_device_id*)param_value;
        for (const auto& it : devices) {
          *device_list++ = const_cast<cl_device_id>(as_cl(it));
        }
        if (param_value_size > valueSize) {
          ::memset(static_cast<address>(param_value) + valueSize, '\0',
                   param_value_size - valueSize);
        }
      }
      return CL_SUCCESS;
    }
    case CL_PROGRAM_SOURCE: {
      const char* source = as_amd(program)->sourceCode().c_str();
      return amd::clGetInfo(source, param_value_size, param_value, param_value_size_ret);
    }
    case CL_PROGRAM_BINARY_SIZES: {
      amd::Program* amdProgram = as_amd(program);
      const amd::Program::devicelist_t& devices = amdProgram->deviceList();
      const size_t numBinaries = devices.size();
      const size_t valueSize = numBinaries * sizeof(size_t);

      if (param_value != NULL && param_value_size < valueSize) {
        return CL_INVALID_VALUE;
      }
      *not_null(param_value_size_ret) = valueSize;
      if (param_value != NULL) {
        size_t* binary_sizes = (size_t*)param_value;
        for (const auto& it : devices) {
          *binary_sizes++ = amdProgram->getDeviceProgram(*it)->binary().second;
        }
        if (param_value_size > valueSize) {
          ::memset(static_cast<address>(param_value) + valueSize, '\0',
                   param_value_size - valueSize);
        }
      }
      return CL_SUCCESS;
    }
    case CL_PROGRAM_BINARIES: {
      amd::Program* amdProgram = as_amd(program);
      const amd::Program::devicelist_t& devices = amdProgram->deviceList();
      const size_t numBinaries = devices.size();
      const size_t valueSize = numBinaries * sizeof(char*);

      if (param_value != NULL && param_value_size < valueSize) {
        return CL_INVALID_VALUE;
      }
      *not_null(param_value_size_ret) = valueSize;
      if (param_value != NULL) {
        char** binaries = (char**)param_value;
        for (const auto& it : devices) {
          const device::Program::binary_t& binary = amdProgram->getDeviceProgram(*it)->binary();
          // If an entry value in the array is NULL,
          // then runtime should skip copying the program binary
          if (*binaries != NULL) {
            ::memcpy(*binaries, binary.first, binary.second);
          }
          binaries++;
        }
        if (param_value_size > valueSize) {
          ::memset(static_cast<address>(param_value) + valueSize, '\0',
                   param_value_size - valueSize);
        }
      }
      return CL_SUCCESS;
    }
    case CL_PROGRAM_NUM_KERNELS: {
      if (as_amd(program)->symbolsPtr() == NULL) {
        return CL_INVALID_PROGRAM_EXECUTABLE;
      }
      size_t numKernels = as_amd(program)->symbols().size();
      return amd::clGetInfo(numKernels, param_value_size, param_value, param_value_size_ret);
    }
    case CL_PROGRAM_KERNEL_NAMES: {
      const char* kernelNames = as_amd(program)->kernelNames().c_str();
      return amd::clGetInfo(kernelNames, param_value_size, param_value, param_value_size_ret);
    }
    default:
      break;
  }

  return CL_INVALID_VALUE;
}
RUNTIME_EXIT

/*! \brief Return build information for each device in the program object.
 *
 *  \param program specifies the program object being queried.
 *
 *  \param device specifies the device for which build information is being
 *  queried. device must be a valid device associated with \a program.
 *
 *  \param param_name specifies the information to query.
 *
 *  \param param_value is a pointer to memory where the appropriate result being
 *  queried is returned. If \a param_value is NULL, it is ignored.
 *
 *  \param param_value_size is used to specify the size in bytes of memory
 *  pointed to by \a param_value. This size must be >= size of return type
 *
 *  \param param_value_size_ret returns the actual size in bytes of data copied
 *  to \a param_value. If \a param_value_size_ret is NULL, it is ignored.
 *
 *  \return One of the following values:
 *  - CL_SUCCESS if the function is executed successfully
 *  - CL_INVALID_DEVICE if \a device is not in the list of devices associated
 *    with \a program
 *  - CL_INVALID_VALUE if \a param_name is not valid, or if size in bytes
 *    specified by \a param_value_size is < size of return type and
 *    \a param_value is not NULL
 *  - CL_INVALID_PROGRAM if \a program is a not a valid program object
 *
 *  \version 1.0r33
 */
RUNTIME_ENTRY(cl_int, clGetProgramBuildInfo,
              (cl_program program, cl_device_id device, cl_program_build_info param_name,
               size_t param_value_size, void* param_value, size_t* param_value_size_ret)) {
  if (!is_valid(program)) {
    return CL_INVALID_PROGRAM;
  }
  if (!is_valid(device)) {
    return CL_INVALID_DEVICE;
  }

  const device::Program* devProgram = as_amd(program)->getDeviceProgram(*as_amd(device));
  if (devProgram == NULL) {
    return CL_INVALID_DEVICE;
  }

  switch (param_name) {
    case CL_PROGRAM_BUILD_STATUS: {
      cl_build_status status = devProgram->buildStatus();
      return amd::clGetInfo(status, param_value_size, param_value, param_value_size_ret);
    }
    case CL_PROGRAM_BUILD_OPTIONS: {
      const std::string optionsStr = devProgram->lastBuildOptionsArg();
      const char* options = optionsStr.c_str();
      return amd::clGetInfo(options, param_value_size, param_value, param_value_size_ret);
    }
    case CL_PROGRAM_BUILD_LOG: {
      const std::string logstr = as_amd(program)->programLog() + devProgram->buildLog().c_str();
      const char* log = logstr.c_str();
      return amd::clGetInfo(log, param_value_size, param_value, param_value_size_ret);
    }
    case CL_PROGRAM_BINARY_TYPE: {
      const device::Program::type_t devProgramType = devProgram->type();
      cl_uint type;
      switch (devProgramType) {
        case device::Program::TYPE_NONE: {
          type = CL_PROGRAM_BINARY_TYPE_NONE;
          break;
        }
        case device::Program::TYPE_COMPILED: {
          type = CL_PROGRAM_BINARY_TYPE_COMPILED_OBJECT;
          break;
        }
        case device::Program::TYPE_LIBRARY: {
          type = CL_PROGRAM_BINARY_TYPE_LIBRARY;
          break;
        }
        case device::Program::TYPE_EXECUTABLE: {
          type = CL_PROGRAM_BINARY_TYPE_EXECUTABLE;
          break;
        }
        case device::Program::TYPE_INTERMEDIATE: {
          type = CL_PROGRAM_BINARY_TYPE_INTERMEDIATE;
          break;
        }
        default:
          return CL_INVALID_VALUE;
      }
      return amd::clGetInfo(type, param_value_size, param_value, param_value_size_ret);
    }
    case CL_PROGRAM_BUILD_GLOBAL_VARIABLE_TOTAL_SIZE: {
      size_t size = devProgram->globalVariableTotalSize();
      return amd::clGetInfo(size, param_value_size, param_value, param_value_size_ret);
    }
    default:
      break;
  }
  return CL_INVALID_VALUE;
}
RUNTIME_EXIT

/*! \brief Sets the values of a SPIR-V specialization constants.
 *
 *  \param program must be a valid OpenCL program created from a SPIR-V module.
 *
 *  \param spec id_ identifies the SPIR-V specialization constant whose value will be set.
 *
 *  \param spec_size specifies the size in bytes of the data pointed to by spec_value. This should
 *  be 1 for boolean constants. For all other constant types this should match the size of the
 *  specialization constant in the SPIR-V module.
 *
 *  \param spec_value is a pointer to the memory location that contains the value of the
 *  specialization constant. The data pointed to by \a spec_value are copied and can be safely
 *  reused by the application after \a clSetProgramSpecializationConstant returns. This
 *  specialization value will be used by subsequent calls to \a clBuildProgram until another call to
 *  \a clSetProgramSpecializationConstant changes it. If a specialization constant is a boolean
 *  constant, _spec value_should be a pointer to a cl_uchar value. A value of zero will set the
 *  specialization constant to false; any other value will set it to true.
 *
 *  Calling this function multiple times for the same specialization constant shall cause the last
 *  provided value to override any previously specified value. The values are used by a subsequent
 *  \a clBuildProgram call for the program.
 *
 *  Application is not required to provide values for every specialization constant contained in
 *  SPIR-V module. SPIR-V provides default values for all specialization constants.
 *
 *  \return One of the following values:
 *  - CL_SUCCESS if the function is executed successfully.
 *  - CL_INVALID_PROGRAM if program is not a valid program object created from a SPIR-V module.
 *  - CL_INVALID_SPEC_ID if spec_id is not a valid specialization constant ID
 *  - CL_INVALID_VALUE if spec_size does not match the size of the specialization constant in the
 *    SPIR-V module, or if spec_value is NULL.
 *  - CL_OUT_OF_RESOURCES if there is a failure to allocate resources required by the OpenCL
 *    implementation on the device.
 *  - CL_OUT_OF_HOST_MEMORY if there is a failure to allocate resources required by the OpenCL
 *    implementation on the host.
 *
 *  \version 2.2-3
 */
RUNTIME_ENTRY(cl_int, clSetProgramSpecializationConstant,
              (cl_program program, cl_uint spec_id, size_t spec_size, const void* spec_value)) {
  if (!is_valid(program)) {
    return CL_INVALID_PROGRAM;
  }
  return CL_INVALID_VALUE;
}
RUNTIME_EXIT

/*! \brief registers a user callback function with a program object. Each call to
 * \a clSetProgramReleaseCallback registers the specified user callback function on a callback stack
 * associated with program. The registered user callback functions are called in the reverse order
 * in which they were registered. The user callback functions are called after destructors (if any)
 * for program scope global variables (if any) are called and before the program is released.
 * This provides a mechanism for the application (and libraries) to be notified when destructors
 * are complete.
 *
 * \param program is a valid program object
 *
 * \param pfn_notify is the callback function that can be registered by the application. This
 * callback function may be called asynchronously by the OpenCL implementation. It is the
 * application's responsibility to ensure that the callback function is thread safe. The parameters
 * to this callback function are:
 * - \a prog is the program object whose destructors are being called. When the user callback is
 *   called by the implementation, this program object is not longer valid. \a prog is only provided
 *   for reference purposes.
 * - \a user_data is a pointer to user supplied data. \a user_data will be passed as the
 *   \a user_data argument when pfn_notify is called. user data can be NULL.
 *
 *  \return One of the following values:
 * - CL_SUCCESS if the function is executed successfully.
 * - CL_INVALID_PROGRAM if program is not a valid program object.
 * - CL_INVALID_VALUE if pfn_notify is NULL.
 * - CL_OUT_OF_RESOURCES if there is a failure to allocate resources required by the OpenCL
 * implementation on the device.
 *
 * \version 2.2-3
 */
RUNTIME_ENTRY(cl_int, clSetProgramReleaseCallback,
              (cl_program program,
               void(CL_CALLBACK* pfn_notify)(cl_program program, void* user_data),
               void* user_data)) {
  if (!is_valid(program)) {
    return CL_INVALID_PROGRAM;
  }
  return CL_INVALID_VALUE;
}
RUNTIME_EXIT

/*! @}
 *  @}
 *
 *  \addtogroup CL_Kernels
 *
 *  A kernel is a function declared in a program. A kernel is identified by the
 *  __kernel qualifier applied to any function in a program. A kernel object
 *  encapsulates the specific __kernel function declared in a program and
 *  the argument values to be used when executing this __kernel function.
 *
 *  @{
 *
 *  \addtogroup CL_CreateKernel
 *  @{
 */

/*! \brief Create a kernel object.
 *
 *  \param program is a program object with a successfully built executable.
 *
 *  \param kernel_name is a function name in the program declared with the
 *  __kernel qualifier.
 *
 *  \param errcode_ret will return an appropriate error code. If \a errcode_ret
 *  is NULL, no error code is returned.
 *
 *  \return A valid non-zero kernel object and \a errcode_ret is set to
 *  CL_SUCCESS if the kernel object is created successfully. It returns a NULL
 *  value with one of the following error values returned in \a errcode_ret:
 *  - CL_INVALID_PROGRAM if \a program is not a valid program object
 *  - CL_INVALID_PROGRAM_EXECUTABLE if there is no successfully built executable
 *    for \a program.
 *  - CL_INVALID_KERNEL_NAME if \a kernel_name is not found in \a program.
 *  - CL_INVALID_KERNEL_DEFINITION if the function definition for __kernel
 *    function given by \a kernel_name such as the number of arguments, the
 *    argument types are not the same for all devices for which the program
 *    executable has been built.
 *  - CL_INVALID_VALUE if \a kernel_name is NULL.
 *  - CL_OUT_OF_HOST_MEMORY if there is a failure to allocate resources required
 *    by the runtime.
 *
 *  \version 1.0r33
 */
RUNTIME_ENTRY_RET(cl_kernel, clCreateKernel,
                  (cl_program program, const char* kernel_name, cl_int* errcode_ret)) {
  if (!is_valid(program)) {
    *not_null(errcode_ret) = CL_INVALID_PROGRAM;
    return (cl_kernel)0;
  }
  if (kernel_name == NULL) {
    *not_null(errcode_ret) = CL_INVALID_VALUE;
    return (cl_kernel)0;
  }
  /* FIXME_lmoriche, FIXME_spec: What are we supposed to do here?
   * if (!as_amd(program)->containsOneSuccesfullyBuiltProgram())
   * {
   *     *NotNull(errcode) = CL_INVALID_PROGRAM_EXECUTABLE;
   *     return (cl_kernel) 0;
   * }
   */
  amd::Program* amd_program = as_amd(program);

  if (!amd_program->load()) {
    *not_null(errcode_ret) = CL_OUT_OF_HOST_MEMORY;
    return (cl_kernel)0;
  }

  const amd::Symbol* symbol = amd_program->findSymbol(kernel_name);
  if (symbol == NULL) {
    *not_null(errcode_ret) = CL_INVALID_KERNEL_NAME;
    return (cl_kernel)0;
  }

  amd::Kernel* kernel = new amd::Kernel(*amd_program, *symbol, kernel_name);
  if (kernel == NULL) {
    *not_null(errcode_ret) = CL_OUT_OF_HOST_MEMORY;
    return (cl_kernel)0;
  }

  *not_null(errcode_ret) = CL_SUCCESS;
  return as_cl(kernel);
}
RUNTIME_EXIT

/*! \brief Create kernel objects for all kernel functions in program.
 *
 *  Kernel objects may not be created for any __kernel functions in program
 *  that do not have the same function definition across all devices for which
 *  a program executable has been successfully built.
 *
 *  \param program is a program object with a successfully built executable.
 *
 *  \param num_kernels is the size of memory pointed to by \a kernels specified
 *  as the number of cl_kernel entries.
 *
 *  \param kernels is the buffer where the kernel objects for kernels in
 *  \a program will be returned. If \a kernels is NULL, it is ignored.
 *  If \a kernels is not NULL, \a num_kernels must be greater than or equal
 *  to the number of kernels in program.
 *
 *  \param num_kernels_ret is the number of kernels in program. If
 *  \a num_kernels_ret is NULL, it is ignored.
 *
 *  \return One of the following values:
 *  - CL_SUCCESS if the kernel objects were successfully allocated
 *  - CL_INVALID_PROGRAM if \a program is not a valid program object
 *  - CL_INVALID_PROGRAM_EXECUTABLE if there is no successfully built executable
 *    for any device in \a program
 *  - CL_INVALID_VALUE if \a kernels is not NULL and \a num_kernels is less
 *    than the number of kernels in program
 *  - CL_OUT_OF_HOST_MEMORY if there is a failure to allocate resources required
 *    by the runtime.
 *
 *  Kernel objects can only be created once you have a program object with a
 *  valid program source or binary loaded into the program object and the
 *  program executable has been successfully built for one or more devices
 *  associated with \a program. No changes to the program executable are
 *  allowed while there are kernel objects associated with a program object.
 *  This means that calls to clBuildProgram return CL_INVALID_OPERATION if there
 *  are kernel objects attached to a program object. The OpenCL context
 *  associated with program will be the context associated with kernel.
 *  Devices associated with a program object for which a valid program
 *  executable has been built can be used to execute kernels declared in the
 *  program object.
 *
 *  \version 1.0r33
 */
RUNTIME_ENTRY(cl_int, clCreateKernelsInProgram,
              (cl_program program, cl_uint num_kernels, cl_kernel* kernels,
               cl_uint* num_kernels_ret)) {
  if (!is_valid(program)) {
    return CL_INVALID_PROGRAM;
  }

  amd::Program* amd_program = as_amd(program);

  if (!amd_program->load()) {
    return CL_OUT_OF_HOST_MEMORY;
  }

  cl_uint numKernels = (cl_uint)amd_program->symbols().size();

  if (kernels != NULL && num_kernels < numKernels) {
    return CL_INVALID_VALUE;
  }
  *not_null(num_kernels_ret) = numKernels;
  if (kernels == NULL) {
    return CL_SUCCESS;
  }

  const amd::Program::symbols_t& symbols = amd_program->symbols();
  cl_kernel* result = kernels;

  for (const auto& it : symbols) {
    amd::Kernel* kernel = new amd::Kernel(*amd_program, it.second, it.first);
    if (kernel == NULL) {
      while (--result >= kernels) {
        as_amd(*result)->release();
      }
      return CL_OUT_OF_HOST_MEMORY;
    }
    *result++ = as_cl(kernel);
  }

  return CL_SUCCESS;
}
RUNTIME_EXIT

/*! \brief Increment the kernel reference count.
 *
 *  \return CL_SUCCESS if the function is executed successfully. It returns
 *  CL_INVALID_KERNEL if \a kernel is not a valid kernel object.
 *
 *  clCreateKernel or clCreateKernelsInProgram do an implicit retain.
 *
 *  \version 1.0r33
 */
RUNTIME_ENTRY(cl_int, clRetainKernel, (cl_kernel kernel)) {
  if (!is_valid(kernel)) {
    return CL_INVALID_KERNEL;
  }
  as_amd(kernel)->retain();
  return CL_SUCCESS;
}
RUNTIME_EXIT

/*! \brief Decrement the kernel reference count.
 *
 *  \return CL_SUCCESS if the function is executed successfully. It returns
 *  CL_INVALID_KERNEL if \a kernel is not a valid kernel object.
 *
 *  The kernel object is deleted once the number of instances that are retained
 *  to \a kernel become zero and after all queued execution instances of
 *  \a kernel have finished.
 *
 *  \version 1.0r33
 */
RUNTIME_ENTRY(cl_int, clReleaseKernel, (cl_kernel kernel)) {
  if (!is_valid(kernel)) {
    return CL_INVALID_KERNEL;
  }
  as_amd(kernel)->release();
  return CL_SUCCESS;
}
RUNTIME_EXIT

/*! \brief Makes a shallow copy of the kernel object, its arguments and any
 *  information passed to the kernel object using \a clSetKernelExecInfo. If
 *  the kernel object was ready to be enqueued before copying it, the clone of
 *  the kernel object is ready to enqueue.
 *
 *  \param source_kernel is a valid cl_kernel object that will be copied.
 *  source_kernel will not be modified in any way by this function.
 *
 *  \param errcode_ret will be assigned an appropriate error code. If
 *  errcode_ret is NULL, no error code is returned.
 *
 *  \return a valid non-zero kernel object and errcode_ret is set to
 *  CL_SUCCESS if the kernel is successfully copied. Otherwise it returns a
 *  NULL value with one of the following error values returned in errcode_ret:
 *  - CL_INVALID_KERNEL if kernel is not a valid kernel object.
 *  - CL_OUT_OF_RESOURCES if there is a failure to allocate resources required
 *    by the OpenCL implementation on the device.
 *  - CL_OUT_OF_HOST_MEMORY if there is a failure to allocate resources
 *    required by the OpenCL implementation on the host.
 *
 *  \version 2.1r01
 */
RUNTIME_ENTRY_RET(cl_kernel, clCloneKernel, (cl_kernel source_kernel, cl_int* errcode_ret)) {
  if (!is_valid(source_kernel)) {
    *not_null(errcode_ret) = CL_INVALID_KERNEL;
    return (cl_kernel)0;
  }

  amd::Kernel* kernel = new amd::Kernel(*as_amd(source_kernel));
  if (kernel == NULL) {
    *not_null(errcode_ret) = CL_OUT_OF_HOST_MEMORY;
    return (cl_kernel)0;
  }

  *not_null(errcode_ret) = CL_SUCCESS;
  return as_cl(kernel);
}
RUNTIME_EXIT

/*! @}
 *  \addtogroup CL_SettingArgs
 *  @{
 */

/*! \brief Set the argument value for a specific argument of a kernel.
 *
 *  \param kernel is a valid kernel object.
 *
 *  \param arg_index is the argument index. Arguments to the kernel are referred
 *  by indices that go from 0 for the leftmost argument to n - 1, where n is the
 *  total number of arguments declared by a kernel.
 *
 *  \param arg_value is a pointer to data that should be used as the argument
 *  value for argument specified by \a arg_index. The argument data pointed to
 *  by \a arg_value is copied and the \a arg_value pointer can therefore be
 *  reused by the application after clSetKernelArg returns. If the argument is
 *  a memory object (buffer or image), the \a arg_value entry will be a pointer
 *  to the appropriate buffer or image object. The memory object must be created
 *  with the context associated with the kernel object. If the argument is
 *  declared with the __local qualifier, the \a arg_value entry must be NULL.
 *  For all other kernel arguments, the \a arg_value entry must be a pointer to
 *  the actual data to be used as argument value. The memory object specified
 *  as argument value must be a buffer object if the argument is declared to be
 *  a pointer of a built-in or user defined type with the __global or __constant
 *  qualifier. If the argument is declared with the __constant qualifier, the
 *  size in bytes of the memory object cannot exceed
 *  CL_DEVICE_MAX_CONSTANT_BUFFER_SIZE and the number of arguments declared
 *  with the __constant qualifier cannot exceed CL_DEVICE_MAX_CONSTANT_ARGS. The
 *  memory object specified as argument value must be a 2D image object if the
 *  argument is declared to be of type image2d_t. The memory object specified as
 *  argument value must be a 3D image object if argument is declared to be of
 *  type image3d_t. If the argument is of type sampler_t, the arg_value entry
 *  must be a pointer to the sampler object.
 *
 *  \param arg_size specifies the size of the argument value. If the argument is
 *  a memory object, the size is the size of the buffer or image object type.
 *  For arguments declared with the __local qualifier, the size specified will
 *  be the size in bytes of the buffer that must be allocated for the __local
 *  argument. If the argument is of type sampler_t, the arg_size value must be
 *  equal to sizeof(cl_sampler). For all other arguments, the size will be the
 *  size of argument type.
 *
 *  \return One of the following values:
 *  - CL_SUCCESS if the function was executed successfully
 *  - CL_INVALID_KERNEL if \a kernel is not a valid kernel object.
 *  - CL_INVALID_ARG_INDEX if \a arg_index is not a valid argument index.
 *  - CL_INVALID_ARG_VALUE if \a arg_value specified is NULL for an argument
 *    that is not declared with the __local qualifier or vice-versa.
 *  - CL_INVALID_MEM_OBJECT for an argument declared to be a memory object but
 *    the specified \a arg_value is not a valid memory object.
 *  - CL_INVALID_SAMPLER for an argument declared to be of type sampler_t but
 *    the specified \a arg_value is not a valid sampler object.
 *  - CL_INVALID_ARG_SIZE if \a arg_size does not match the size of the data
 *    type for an argument that is not a memory object or if the argument is a
 *    memory object and \a arg_size != sizeof(cl_mem) or if \a arg_size is zero
 *    and the argument is declared with the __local qualifier or if the
 *    argument is a sampler and arg_size != sizeof(cl_sampler).
 *
 *  \version 1.0r33
 */
RUNTIME_ENTRY(cl_int, clSetKernelArg,
              (cl_kernel kernel, cl_uint arg_index, size_t arg_size, const void* arg_value)) {
  if (!is_valid(kernel)) {
    return CL_INVALID_KERNEL;
  }

  const amd::KernelSignature& signature = as_amd(kernel)->signature();
  if (arg_index >= signature.numParameters()) {
    return CL_INVALID_ARG_INDEX;
  }

  const amd::KernelParameterDescriptor& desc = signature.at(arg_index);
  const bool is_local = (desc.addressQualifier_ == CL_KERNEL_ARG_ADDRESS_LOCAL);
  if (((arg_value == NULL) && !is_local && (desc.type_ != T_POINTER)) ||
      ((arg_value != NULL) && is_local)) {
    as_amd(kernel)->parameters().reset(static_cast<size_t>(arg_index));
    return CL_INVALID_ARG_VALUE;
  }
  if (!is_local && (desc.type_ == T_POINTER) && (arg_value != NULL)) {
    cl_mem memObj = *static_cast<const cl_mem*>(arg_value);
    amd::RuntimeObject* pObject = as_amd(memObj);
    if (NULL != memObj && amd::RuntimeObject::ObjectTypeMemory != pObject->objectType()) {
      as_amd(kernel)->parameters().reset(static_cast<size_t>(arg_index));
      return CL_INVALID_MEM_OBJECT;
    }
  } else if ((desc.type_ == T_SAMPLER) && !is_valid(*static_cast<const cl_sampler*>(arg_value))) {
    return CL_INVALID_SAMPLER;
  } else if (desc.type_ == T_QUEUE) {
    cl_command_queue queue = *static_cast<const cl_command_queue*>(arg_value);
    if (!is_valid(queue)) {
      as_amd(kernel)->parameters().reset(static_cast<size_t>(arg_index));
      return CL_INVALID_DEVICE_QUEUE;
    }
    if (NULL == as_amd(queue)->asDeviceQueue()) {
      as_amd(kernel)->parameters().reset(static_cast<size_t>(arg_index));
      return CL_INVALID_DEVICE_QUEUE;
    }
  }
  if ((!is_local && (arg_size != desc.size_)) || (is_local && (arg_size == 0))) {
    if (LP64_ONLY(true ||)((desc.type_ != T_POINTER) && (desc.type_ != T_SAMPLER)) ||
        (arg_size != sizeof(void*))) {
      as_amd(kernel)->parameters().reset(static_cast<size_t>(arg_index));
      return CL_INVALID_ARG_SIZE;
    }
  }

  as_amd(kernel)->parameters().set(static_cast<size_t>(arg_index), arg_size, arg_value);
  return CL_SUCCESS;
}
RUNTIME_EXIT

/*! @}
 *  \addtogroup CL_KernelQuery
 *  @{
 */

/*! \brief Return information about the kernel object.
 *
 *  \param kernel specifies the kernel object being queried.
 *
 *  \param param_name specifies the information to query.
 *
 *  \param param_value is a pointer to memory where the appropriate result
 *  being queried is returned. If \a param_value is NULL, it is ignored.
 *
 *  \param param_value_size is used to specify the size in bytes of memory
 *  pointed to by \a param_value. This size must be >= size of return type.
 *
 *  \param param_value_size_ret returns the actual size in bytes of data copied
 *  to \a param_value. If \a param_value_size_ret is NULL, it is ignored.
 *
 *  \return One of the following values:
 *  - CL_SUCCESS if the function is executed successfully
 *  - CL_INVALID_VALUE if \a param_name is not valid, or if size in bytes
 *    specified by \a param_value_size is < size of return type and
 *    \a param_value is not NULL
 *  - CL_INVALID_KERNEL if \a kernel is a not a valid kernel object.
 *
 *  \version 1.0r33
 */
RUNTIME_ENTRY(cl_int, clGetKernelInfo,
              (cl_kernel kernel, cl_kernel_info param_name, size_t param_value_size,
               void* param_value, size_t* param_value_size_ret)) {
  // Check if we have a valid kernel
  if (!is_valid(kernel)) {
    return CL_INVALID_KERNEL;
  }

  const amd::Kernel* amdKernel = as_amd(kernel);

  // Get the corresponded parameters
  switch (param_name) {
    case CL_KERNEL_FUNCTION_NAME: {
      const char* name = amdKernel->name().c_str();
      // Return the kernel's name
      return amd::clGetInfo(name, param_value_size, param_value, param_value_size_ret);
    }
    case CL_KERNEL_NUM_ARGS: {
      cl_uint numParam = static_cast<cl_uint>(amdKernel->signature().numParameters());
      // Return the number of kernel's parameters
      return amd::clGetInfo(numParam, param_value_size, param_value, param_value_size_ret);
    }
    case CL_KERNEL_REFERENCE_COUNT: {
      cl_uint count = amdKernel->referenceCount();
      // Return the reference counter
      return amd::clGetInfo(count, param_value_size, param_value, param_value_size_ret);
    }
    case CL_KERNEL_CONTEXT: {
      cl_context context = const_cast<cl_context>(as_cl(&amdKernel->program().context()));
      // Return the context, associated with the program
      return amd::clGetInfo(context, param_value_size, param_value, param_value_size_ret);
    }
    case CL_KERNEL_PROGRAM: {
      cl_program program = const_cast<cl_program>(as_cl(&amdKernel->program()));
      // Return the program, associated with the kernel
      return amd::clGetInfo(program, param_value_size, param_value, param_value_size_ret);
    }
    case CL_KERNEL_ATTRIBUTES: {
      const char* name = amdKernel->signature().attributes().c_str();
      // Return the kernel attributes
      return amd::clGetInfo(name, param_value_size, param_value, param_value_size_ret);
    }
    default:
      return CL_INVALID_VALUE;
  }

  return CL_SUCCESS;
}
RUNTIME_EXIT

/*! \brief Returns information about the arguments of a kernel. Kernel
 *  argument information is only available if the program object associated
 *  with kernel is created with \a clCreateProgramWithSource and the program
 *  executable is built with the -cl-kernel-arg-info option specified in
 *  options argument to clBuildProgram or clCompileProgram.
 *
 *  \param kernel specifies the kernel object being queried.
 *
 *  \param param_name specifies the information to query.
 *
 *  \param param_value is a pointer to memory where the appropriate result
 *  being queried is returned. If \a param_value is NULL, it is ignored.
 *
 *  \param param_value_size is used to specify the size in bytes of memory
 *  pointed to by \a param_value. This size must be >= size of return type.
 *
 *  \param param_value_size_ret returns the actual size in bytes of data copied
 *  to \a param_value. If \a param_value_size_ret is NULL, it is ignored.
 *
 *  \return One of the following values:
 *  - CL_SUCCESS if the function is executed successfully
 *  - CL_INVALID_VALUE if \a param_name is not valid, or if size in bytes
 *    specified by \a param_value_size is < size of return type and
 *    \a param_value is not NULL
 *  - CL_INVALID_KERNEL if \a kernel is a not a valid kernel object.
 *
 *  \version 1.2r07
 */
RUNTIME_ENTRY(cl_int, clGetKernelArgInfo,
              (cl_kernel kernel, cl_uint arg_indx, cl_kernel_arg_info param_name,
               size_t param_value_size, void* param_value, size_t* param_value_size_ret)) {
  // Check if we have a valid kernel
  if (!is_valid(kernel)) {
    return CL_INVALID_KERNEL;
  }

  amd::Kernel* amdKernel = as_amd(kernel);

  const amd::KernelSignature& signature = amdKernel->signature();
  if (arg_indx >= signature.numParameters()) {
    return CL_INVALID_ARG_INDEX;
  }

  const amd::KernelParameterDescriptor& desc = signature.at(arg_indx);

  // Get the corresponded parameters
  switch (param_name) {
    case CL_KERNEL_ARG_ADDRESS_QUALIFIER: {
      cl_kernel_arg_address_qualifier qualifier = desc.addressQualifier_;
      return amd::clGetInfo(qualifier, param_value_size, param_value, param_value_size_ret);
    }
    case CL_KERNEL_ARG_ACCESS_QUALIFIER: {
      cl_kernel_arg_access_qualifier qualifier = desc.accessQualifier_;
      return amd::clGetInfo(qualifier, param_value_size, param_value, param_value_size_ret);
    }
    case CL_KERNEL_ARG_TYPE_NAME: {
      const char* typeName = desc.typeName_.c_str();
      // Return the argument's type name
      return amd::clGetInfo(typeName, param_value_size, param_value, param_value_size_ret);
    }
    case CL_KERNEL_ARG_TYPE_QUALIFIER: {
      cl_kernel_arg_type_qualifier qualifier = desc.typeQualifier_;
      return amd::clGetInfo(qualifier, param_value_size, param_value, param_value_size_ret);
    }
    case CL_KERNEL_ARG_NAME: {
      const char* name = desc.name_.c_str();
      // Return the argument's name
      return amd::clGetInfo(name, param_value_size, param_value, param_value_size_ret);
    }
    default:
      return CL_INVALID_VALUE;
  }

  return CL_SUCCESS;
}
RUNTIME_EXIT

/*! \brief Return information about the kernel object that may be specific
 *  to a device.
 *
 *  \param kernel specifies the kernel object being queried.
 *
 *  \param device identifies a specific device in the list of devices associated
 *  with \a kernel. The list of devices is the list of devices in the OpenCL
 *  context that is associated with \a kernel. If the list of devices associated
 *  with kernel is a single device, \a device can be a NULL value.
 *
 *  \param param_name specifies the information to query
 *
 *  \param param_value is a pointer to memory where the appropriate result being
 *  queried is returned. If \a param_value is NULL, it is ignored.
 *
 *  \param param_value_size is used to specify the size in bytes of memory
 *  pointed to by \a param_value. This size must be >= size of return type.
 *
 *  \param param_value_size_ret returns the actual size in bytes of data copied
 *  to \a param_value. If \a param_value_size_ret is NULL, it is ignored.
 *
 *  \return One of the following values:
 *  - CL_SUCCESS if the function is executed successfully,
 *  - CL_INVALID_DEVICE if \a device is not in the list of devices associated
 *    with \a kernel or if \a device is NULL but there are more than one
 *    devices in the associated with \a kernel
 *  - CL_INVALID_VALUE if \a param_name is not valid, or if size in bytes
 *    specified by \a param_value_size is < size of return type and
 *    \a param_value is not NULL
 *  - CL_INVALID_KERNEL if \a kernel is a not a valid kernel object.
 *
 *  \version 1.2r15
 */
RUNTIME_ENTRY(cl_int, clGetKernelWorkGroupInfo,
              (cl_kernel kernel, cl_device_id device, cl_kernel_work_group_info param_name,
               size_t param_value_size, void* param_value, size_t* param_value_size_ret)) {
  // Check if we have a valid device
  if (!is_valid(device)) {
    return CL_INVALID_DEVICE;
  }

  // Check if we have a valid kernel
  if (!is_valid(kernel)) {
    return CL_INVALID_KERNEL;
  }


  const amd::Device& amdDevice = *as_amd(device);
  // Find the kernel, associated with the specified device
  const device::Kernel* devKernel = as_amd(kernel)->getDeviceKernel(amdDevice);

  // Make sure we found a valid kernel
  if (devKernel == NULL) {
    return CL_INVALID_KERNEL;
  }

  // Get the corresponded parameters
  switch (param_name) {
    case CL_KERNEL_WORK_GROUP_SIZE: {
      // Return workgroup size
      return amd::clGetInfo(devKernel->workGroupInfo()->size_, param_value_size, param_value,
                            param_value_size_ret);
    }
    case CL_KERNEL_COMPILE_WORK_GROUP_SIZE: {
      // Return the compile workgroup size
      return amd::clGetInfo(devKernel->workGroupInfo()->compileSize_, param_value_size, param_value,
                            param_value_size_ret);
    }
    case CL_KERNEL_LOCAL_MEM_SIZE: {
      // Return the amount of used local memory
      const size_t align = amdDevice.info().minDataTypeAlignSize_;
      cl_ulong memSize = as_amd(kernel)->parameters().localMemSize(align) +
                         amd::alignUp(devKernel->workGroupInfo()->localMemSize_, align);
      return amd::clGetInfo(memSize, param_value_size, param_value, param_value_size_ret);
    }
    case CL_KERNEL_PREFERRED_WORK_GROUP_SIZE_MULTIPLE: {
      // Return the compile workgroup size
      return amd::clGetInfo(devKernel->workGroupInfo()->preferredSizeMultiple_, param_value_size,
                            param_value, param_value_size_ret);
    }
    case CL_KERNEL_PRIVATE_MEM_SIZE: {
      // Return the compile workgroup size
      return amd::clGetInfo(devKernel->workGroupInfo()->privateMemSize_, param_value_size,
                            param_value, param_value_size_ret);
    }
    case CL_KERNEL_GLOBAL_WORK_SIZE: {
      return CL_INVALID_VALUE;
    }
    case CL_KERNEL_MAX_SEMAPHORE_SIZE_AMD: {
      return amd::clGetInfo(amdDevice.info().maxSemaphoreSize_, param_value_size, param_value,
                            param_value_size_ret);
    }
    default:
      return CL_INVALID_VALUE;
  }

  return CL_SUCCESS;
}
RUNTIME_EXIT

/*! \brief Returns information about the kernel object.
 *
 * \param kernel specifies the kernel object being queried.
 *
 * \param device identifies a specific device in the list of devices associated
 * with kernel. The list of devices is the list of devices in the OpenCL context
 * that is associated with kernel. If the list of devices associated with kernel
 * is a single device, device can be a NULL value.
 *
 * \param param_name specifies the information to query. The list of supported
 * param_name types and the information returned in param_value by
 * clGetKernelSubGroupInfo is described in the table below.
 *
 * \param input_value_size is used to specify the size in bytes of memory
 * pointed to by input_value. This size must be == size of input type as
 * described in the table below.
 *
 * \param input_value is a pointer to memory where the appropriate
 * parameterization of the query is passed from. If input_value is NULL, it is
 * ignored.
 *
 * \param param_value is a pointer to memory where the appropriate result being
 * queried is returned. If param_value is NULL, it is ignored.
 *
 * \param param_value_size is used to specify the size in bytes of memory
 * pointed to by param_value. This size must be >= size of return type as
 * described in the table below.
 *
 * \param param_value_size_ret returns the actual size in bytes of data copied
 * to param_value. If param_value_size_ret is NULL, it is ignored.
 *
 * \return CL_SUCCESS if the function is executed successfully.
 * Otherwise, it returns one of the following errors:
 *
 * - CL_INVALID_DEVICE if device is not in the list of devices associated with
 *   kernel or if device is NULL but there is more than one device associated
 *   with kernel.
 * - CL_INVALID_VALUE if param_name is not valid, or if size in bytes specified
 *   by param_value_size is < size of return type as described in the table
 *   above and param_value is not NULL.
 * - CL_INVALID_VALUE if param_name is CL_KERNEL_SUB_GROUP_SIZE_FOR_NDRANGE and
 *   the size in bytes specified by input_value_size is not valid or if
 *   input_value is NULL.
 * - CL_INVALID_KERNEL if kernel is a not a valid kernel object.
 * - CL_OUT_OF_RESOURCES if there is a failure to allocate resources required by
 *   the OpenCL implementation on the device.
 * - CL_OUT_OF_HOST_MEMORY if there is a failure to allocate resources required
 *   by the OpenCL implementation on the host.
 *
 *  \version 2.0r12
 */
RUNTIME_ENTRY(cl_int, clGetKernelSubGroupInfo,
              (cl_kernel kernel, cl_device_id device, cl_kernel_sub_group_info param_name,
               size_t input_value_size, const void* input_value, size_t param_value_size,
               void* param_value, size_t* param_value_size_ret)) {
  // Check if we have a valid device
  if (!is_valid(device)) {
    return CL_INVALID_DEVICE;
  }

  // Check if we have a valid kernel
  if (!is_valid(kernel)) {
    return CL_INVALID_KERNEL;
  }


  const amd::Device& amdDevice = *as_amd(device);
  // Find the kernel, associated with the specified device
  const device::Kernel* devKernel = as_amd(kernel)->getDeviceKernel(amdDevice);

  // Make sure we found a valid kernel
  if (devKernel == NULL) {
    return CL_INVALID_KERNEL;
  }

  // Get the corresponded parameters
  switch (param_name) {
    case CL_KERNEL_MAX_SUB_GROUP_SIZE_FOR_NDRANGE:
    case CL_KERNEL_SUB_GROUP_COUNT_FOR_NDRANGE: {
      // Infer the number of dimensions from 'input_value_size'
      size_t dims = input_value_size / sizeof(size_t);
      if (dims == 0 || dims > 3 || input_value_size != dims * sizeof(size_t)) {
        return CL_INVALID_VALUE;
      }

      // Get the linear workgroup size
      size_t workGroupSize = ((size_t*)input_value)[0];
      for (size_t i = 1; i < dims; ++i) {
        workGroupSize *= ((size_t*)input_value)[i];
      }

      // Get the subgroup size. GPU devices sub-groups are wavefronts.
      size_t subGroupSize = as_amd(device)->info().wavefrontWidth_;

      size_t numSubGroups = (workGroupSize + subGroupSize - 1) / subGroupSize;


      return amd::clGetInfo((param_name == CL_KERNEL_MAX_SUB_GROUP_SIZE_FOR_NDRANGE_KHR)
                                ? subGroupSize
                                : numSubGroups,
                            param_value_size, param_value, param_value_size_ret);
    }
    case CL_KERNEL_COMPILE_NUM_SUB_GROUPS: {
      size_t numSubGroups = 0;
      return amd::clGetInfo(numSubGroups, param_value_size, param_value, param_value_size_ret);
    }
    case CL_KERNEL_MAX_NUM_SUB_GROUPS: {
      size_t waveSize = as_amd(device)->info().wavefrontWidth_;
      size_t numSubGroups = (devKernel->workGroupInfo()->size_ + waveSize - 1) / waveSize;
      return amd::clGetInfo(numSubGroups, param_value_size, param_value, param_value_size_ret);
    }
    case CL_KERNEL_LOCAL_SIZE_FOR_SUB_GROUP_COUNT: {
      if (input_value_size != sizeof(size_t)) {
        return CL_INVALID_VALUE;
      }
      size_t numSubGroups = ((size_t*)input_value)[0];

      // Infer the number of dimensions from 'param_value_size'
      size_t dims = param_value_size / sizeof(size_t);
      if (dims == 0 || dims > 3 || param_value_size != dims * sizeof(size_t)) {
        return CL_INVALID_VALUE;
      }
      *not_null(param_value_size_ret) = param_value_size;

      size_t localSize;
      localSize = numSubGroups * as_amd(device)->info().wavefrontWidth_;
      if (localSize > devKernel->workGroupInfo()->size_) {
        ::memset(param_value, '\0', dims * sizeof(size_t));
        return CL_SUCCESS;
      }

      switch (dims) {
        case 3:
          ((size_t*)param_value)[2] = 1;
        case 2:
          ((size_t*)param_value)[1] = 1;
        case 1:
          ((size_t*)param_value)[0] = localSize;
      }
      return CL_SUCCESS;
    }
    default:
      return CL_INVALID_VALUE;
  }

  return CL_SUCCESS;
}
RUNTIME_EXIT

/*! @}
 *  @}
 *  @}
 */
