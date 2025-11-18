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

#include "platform/context.hpp"
#include "platform/command.hpp"
#include "platform/memory.hpp"
#include <cmath>

#ifdef _WIN32
#include <d3d10_1.h>
#include "cl_d3d9_amd.hpp"
#include "cl_d3d10_amd.hpp"
#include "cl_d3d11_amd.hpp"
#endif  //_WIN32

#include <cstring>

/*! \addtogroup API
 *  @{
 *
 *  \addtogroup CL_MemObjs
 *
 *  Memory objects are categorized into two types: buffer objects, and image
 *  objects. A buffer object stores a one-dimensional collection of elements
 *  whereas an image object is used to store a two- or three- dimensional
 *  texture, frame-buffer or image.
 *
 *  Elements of a buffer object can be a scalar data type (such as an int,
 *  float), vector data type, or a user-defined structure. An image object is
 *  used to represent a buffer that can be used as a texture or a frame-buffer.
 *  The elements of an image object are selected from a list of predefined
 *  image formats. The minimum number of elements in a memory object is one.
 *
 *  @{
 *
 *  \addtogroup CL_CreatingBuffer
 *
 *  @{
 */

/*! \brief Helper function to validate cl_mem_flags
 *
 * chkReadWrite: true: check the flag CL_MEM_KERNEL_READ_AND_WRITE
 *              false: don't check the falg CL_MEM_KERNEL_READ_AND_WRITE
 *  \return true of flags are valid, otherwise - false
 */
static bool validateFlags(cl_mem_flags flags, bool chkReadWrite = false) {
  // check flags for validity
  cl_bitfield temp = flags & (CL_MEM_READ_WRITE | CL_MEM_WRITE_ONLY | CL_MEM_READ_ONLY);
  if (chkReadWrite) {
    temp |= (flags & CL_MEM_KERNEL_READ_AND_WRITE);
  }

  if (temp && !(CL_MEM_READ_WRITE == temp || CL_MEM_WRITE_ONLY == temp ||
                (chkReadWrite && (CL_MEM_KERNEL_READ_AND_WRITE == temp ||
                                  (CL_MEM_KERNEL_READ_AND_WRITE | CL_MEM_READ_WRITE) == temp)) ||
                CL_MEM_READ_ONLY == temp)) {
    return false;
  }

  if ((flags & (CL_MEM_USE_HOST_PTR | CL_MEM_ALLOC_HOST_PTR)) ==
      (CL_MEM_USE_HOST_PTR | CL_MEM_ALLOC_HOST_PTR)) {
    return false;
  }
  if ((flags & (CL_MEM_USE_HOST_PTR | CL_MEM_COPY_HOST_PTR)) ==
      (CL_MEM_USE_HOST_PTR | CL_MEM_COPY_HOST_PTR)) {
    return false;
  }

  if ((flags & CL_MEM_EXTERNAL_PHYSICAL_AMD) &&
      (flags & (CL_MEM_USE_HOST_PTR | CL_MEM_COPY_HOST_PTR | CL_MEM_ALLOC_HOST_PTR |
                CL_MEM_READ_WRITE | CL_MEM_READ_ONLY))) {
    return false;
  }

  if ((flags & CL_MEM_BUS_ADDRESSABLE_AMD) &&
      (flags & (CL_MEM_USE_HOST_PTR | CL_MEM_ALLOC_HOST_PTR))) {
    return false;
  }

  return true;
}

/*! \brief Helper function to validate cl_image_desc
 *
 *  \return true of cl_image_desc parameters are valid, otherwise - false
 *
 *  image_type describes the image type and must be either CL_MEM_OBJECT_IMAGE1D,
 *  CL_MEM_OBJECT_IMAGE1D_BUFFER, CL_MEM_OBJECT_IMAGE1D_ARRAY,
 *  CL_MEM_OBJECT_IMAGE2D, CL_MEM_OBJECT_IMAGE2D_ARRAY or CL_MEM_OBJECT_IMAGE3D.
 *
 *  image_width is the width of the image in pixels. For a 2D image and
 *  image array, the image width must be <= CL_DEVICE_IMAGE2D_MAX_WIDTH.
 *  For a 3D image, the image width must be <= CL_DEVICE_IMAGE3D_MAX_WIDTH.
 *  For a 1D image buffer, the image width must be <= CL_DEVICE_IMAGE_MAX_BUFFER_SIZE.
 *  For a 1D image and 1D image array, the image width must be
 *  <= CL_DEVICE_IMAGE2D_MAX_WIDTH.
 *
 *  image_height is height of the image in pixels. This is only used if
 *  the image is a 2D, 3D or 2D image array. For a 2D image or image array,
 *  the image height must be <= CL_DEVICE_IMAGE2D_MAX_HEIGHT. For a 3D image,
 *  the image height must be <= CL_DEVICE_IMAGE3D_MAX_HEIGHT.
 *
 *  image_depth is the depth of the image in pixels. This is only used if
 *  the image is a 3D image and must be a value > 1 and
 *  <= CL_DEVICE_IMAGE3D_MAX_DEPTH.
 *
 *  image_array_size is the number of images in the image array. This is only
 *  used if the image is a 1D or 2D image array. The values for
 *  image_array_size, if specified, must be between 1 and
 *  CL_DEVICE_IMAGE_MAX_ARRAY_SIZE.
 *
 *  image_row_pitch is the scan-line pitch in bytes. This must be 0 if
 *  host_ptr is NULL and can be either 0 or >= image_width * size of element in
 *  bytes if host_ptr is not NULL. If host_ptr is not NULL and image_row_pitch = 0,
 *  image_row_pitch is calculated as image_width * size of element in bytes.
 *  If image_row_pitch is not 0, it must be a multiple of the image element
 *  size in bytes.
 *
 *  image_slice_pitch is the size in bytes of each 2D slice in the 3D image or
 *  the size in bytes of each image in a 1D or 2D image array. This must be 0
 *  if host_ptr is NULL. If host_ptr is not NULL, image_slice_pitch can be either
 *  0 or >= image_row_pitch * image_height for a 2D image array or 3D image and
 *  can be either 0 or >= image_row_pitch for a 1D image array. If host_ptr is
 *  not NULL and image_slice_pitch = 0, image_slice_pitch is calculated as
 *  image_row_pitch * image_height for a 2D image array or 3D image and
 *  image_row_pitch for a 1D image array. If image_slice_pitch is not 0, it must
 *  be a multiple of the image_row_pitch.
 *
 *  num_mip_levels and num_samples must be 0.
 *
 *  buffer refers to a valid buffer memory object if image_type is
 *  CL_MEM_OBJECT_IMAGE1D_BUFFER. Otherwise it must be NULL. For a 1D image
 *  buffer object, the image pixels are taken from the buffer object’s
 *  data store. When the contents of a buffer object’s data store are modified,
 *  those changes are reflected in the contents of the 1D image buffer object
 *  and vice-versa at corresponding sychronization points. The image_width
 *  size of element in bytes must be <= size of buffer object data store.
 */
static bool validateImageDescriptor(const std::vector<amd::Device*>& devices,
                                    const amd::Image::Format imageFormat, const cl_image_desc* desc,
                                    void* hostPtr, size_t& imageRowPitch, size_t& imageSlicePitch) {
  if (desc == NULL) {
    return false;
  }

  // Check if any device supports mipmaps
  bool mipMapSupport = false;
  for (auto& dev : devices) {
    if (dev->settings().checkExtension(ClKhrMipMapImage)) {
      mipMapSupport = true;
      break;
    }
  }

  // Check if any device can accept mipmaps
  if ((desc->num_mip_levels != 0) && (!mipMapSupport || (hostPtr != NULL))) {
    return false;
  }

  if (desc->num_samples != 0) {
    return false;
  }

  amd::Buffer* buffer = NULL;
  size_t elemSize = imageFormat.getElementSize();
  bool imageBuffer = false;

  if (desc->image_type == CL_MEM_OBJECT_IMAGE1D_BUFFER ||
      (desc->mem_object != NULL && desc->image_type == CL_MEM_OBJECT_IMAGE2D)) {
    if (desc->mem_object == NULL) {
      return false;
    }
    buffer = as_amd(desc->mem_object)->asBuffer();
    if (buffer == NULL) {
      return false;
    }
    if ((desc->image_width * desc->image_height * elemSize) > buffer->getSize()) {
      return false;
    }
    imageBuffer = true;
  } else if (desc->mem_object != NULL) {
    return false;
  }

  imageRowPitch = desc->image_row_pitch;
  imageSlicePitch = desc->image_slice_pitch;

  switch (desc->image_type) {
    case CL_MEM_OBJECT_IMAGE3D:
    case CL_MEM_OBJECT_IMAGE2D_ARRAY:
    case CL_MEM_OBJECT_IMAGE1D_ARRAY:
      // check slice pitch
      if (hostPtr == NULL) {
        if (imageSlicePitch != 0) {
          return false;
        }
      }
    // Fall through to process pitch...
    case CL_MEM_OBJECT_IMAGE2D:
    case CL_MEM_OBJECT_IMAGE1D:
      // check row pitch rules
      if (hostPtr == NULL && !imageBuffer) {
        if (imageRowPitch != 0) {
          return false;
        }
      } else if (imageRowPitch != 0) {
        if ((imageRowPitch < desc->image_width * elemSize) || ((imageRowPitch % elemSize) != 0)) {
          return false;
        }
      }
      if (imageRowPitch == 0) {
        if (desc->mem_object != nullptr) {
          imageRowPitch =
              amd::alignUp(desc->image_width, devices[0]->info().imagePitchAlignment_) * elemSize;
        } else {
          imageRowPitch = desc->image_width * elemSize;
        }
      }
      break;
    case CL_MEM_OBJECT_IMAGE1D_BUFFER:
      break;
    default:
      return false;
      break;
  }

  // Extra slice validation for three dimensional images
  if ((desc->image_type == CL_MEM_OBJECT_IMAGE3D) ||
      (desc->image_type == CL_MEM_OBJECT_IMAGE2D_ARRAY)) {
    if (imageSlicePitch != 0) {
      if ((imageSlicePitch < (imageRowPitch * desc->image_height)) ||
          ((imageSlicePitch % imageRowPitch) != 0)) {
        return false;
      }
    }
    if (imageSlicePitch == 0) {
      imageSlicePitch = imageRowPitch * desc->image_height;
    }
  } else if (desc->image_type == CL_MEM_OBJECT_IMAGE1D_ARRAY) {
    if (imageSlicePitch != 0) {
      if ((imageSlicePitch % imageRowPitch) != 0) {
        return false;
      }
    }
    if (imageSlicePitch == 0) {
      imageSlicePitch = imageRowPitch;
    }
  }

  return true;
}

class ImageViewRef : public amd::EmbeddedObject {
 private:
  amd::Image* ref_;
  // Do not copy image view references.
  ImageViewRef& operator=(const ImageViewRef& sref);

 public:
  explicit ImageViewRef() : ref_(NULL) {}
  ~ImageViewRef() {
    if (ref_ != NULL) {
      ref_->release();
    }
  }

  ImageViewRef& operator=(amd::Image* sref) {
    ref_ = sref;
    return *this;
  }
  amd::Image* operator()() const { return ref_; }
};

/*! \brief Create a buffer object.
 *
 *  \param context is a valid OpenCL context used to create the buffer object.
 *
 *  \param flags is a bit-field that is used to specify allocation and usage
 *  information such as the memory arena that should be used to allocate the
 *  buffer object and how it will be used.
 *
 *  \param size is the size in bytes of the buffer memory object to be
 *  allocated.
 *
 *  \param host_ptr is a pointer to the buffer data that may already be
 *  allocated by the application. The size of the buffer that host_ptr points
 *  to must be >= \a size bytes. Passing in a pointer to an already allocated
 *  buffer on the host and using it as a buffer object allows applications to
 *  share data efficiently with kernels and the host.
 *
 *  \param errcode_ret will return an appropriate error code. If \a errcode_ret
 *  is NULL, no error code is returned.
 *
 *  \return A valid non-zero buffer object and \a errcode_ret is set to
 *  CL_SUCCESS if the buffer object is created successfully or a NULL value
 *  with one of the following error values returned in \a errcode_ret:
 *  - CL_INVALID_CONTEXT if \a context is not a valid context.
 *  - CL_INVALID_VALUE if values specified in \a flags are not valid.
 *  - CL_INVALID_BUFFER_SIZE if \a size is 0 or is greater than
 *    CL_DEVICE_MAX_MEM_ALLOC_SIZE value.
 *  - CL_INVALID_HOST_PTR if host_ptr is NULL and CL_MEM_USE_HOST_PTR or
 *    CL_MEM_COPY_HOST_PTR are set in \a flags or if \a host_ptr is not NULL but
 *    CL_MEM_COPY_HOST_PTR or CL_MEM_USE_HOST_PTR are not set in \a flags.
 *  - CL_MEM_OBJECT_ALLOCATION_FAILURE if there is a failure to allocate memory
 *    for buffer object.
 *  - CL_INVALID_OPERATION if the buffer object cannot be created for all
 *    devices in \a context.
 *  - CL_OUT_OF_HOST_MEMORY if there is a failure to allocate resources required
 *    by the runtime.
 *
 *  \version 1.0r33
 */
RUNTIME_ENTRY_RET(cl_mem, clCreateBuffer,
                  (cl_context context, cl_mem_flags flags, size_t size, void* host_ptr,
                   cl_int* errcode_ret)) {
  if (!is_valid(context)) {
    *not_null(errcode_ret) = CL_INVALID_CONTEXT;
    return NULL;
  }
  // check flags for validity
  if (!validateFlags(flags)) {
    *not_null(errcode_ret) = CL_INVALID_VALUE;
    LogWarning("invalid parameter \"flags\"");
    return (cl_mem)0;
  }
  // check size
  if (size == 0) {
    *not_null(errcode_ret) = CL_INVALID_BUFFER_SIZE;
    LogWarning("invalid parameter \"size = 0\"");
    return (cl_mem)0;
  }
  const std::vector<amd::Device*>& devices = as_amd(context)->devices();
  bool sizePass = false;
  for (auto& dev : devices) {
    if ((dev->info().maxMemAllocSize_ >= size) ||
        (flags & (CL_MEM_USE_HOST_PTR | CL_MEM_ALLOC_HOST_PTR))) {
      sizePass = true;
      break;
    }
  }
  if (!sizePass) {
    *not_null(errcode_ret) = CL_INVALID_BUFFER_SIZE;
    LogWarning("invalid parameter \"size\"");
    return (cl_mem)0;
  }

  // check host_ptr consistency
  if (host_ptr == NULL) {
    if (flags & (CL_MEM_USE_HOST_PTR | CL_MEM_COPY_HOST_PTR | CL_MEM_EXTERNAL_PHYSICAL_AMD)) {
      *not_null(errcode_ret) = CL_INVALID_HOST_PTR;
      LogWarning("invalid parameter \"host_ptr\"");
      return (cl_mem)0;
    }
  } else {
    if (!(flags & (CL_MEM_USE_HOST_PTR | CL_MEM_COPY_HOST_PTR | CL_MEM_EXTERNAL_PHYSICAL_AMD))) {
      *not_null(errcode_ret) = CL_INVALID_HOST_PTR;
      LogWarning("invalid parameter \"host_ptr\"");
      return (cl_mem)0;
    }

    if (flags & CL_MEM_EXTERNAL_PHYSICAL_AMD) {
      flags |= CL_MEM_WRITE_ONLY;

      cl_bus_address_amd* bus_address = reinterpret_cast<cl_bus_address_amd*>(host_ptr);

      if (bus_address->surface_bus_address == 0) {
        *not_null(errcode_ret) = CL_INVALID_HOST_PTR;
        LogWarning("invalid parameter \"surface bus address\"");
        return static_cast<cl_mem>(NULL);
      }

      if (bus_address->surface_bus_address & (amd::Os::pageSize() - 1)) {
        *not_null(errcode_ret) = CL_INVALID_HOST_PTR;
        LogWarning("invalid parameter \"surface bus address\"");
        return static_cast<cl_mem>(NULL);
      }

      if (bus_address->marker_bus_address == 0) {
        *not_null(errcode_ret) = CL_INVALID_HOST_PTR;
        LogWarning("invalid parameter \"marker bus address\"");
        return static_cast<cl_mem>(NULL);
      }

      if (bus_address->marker_bus_address & (amd::Os::pageSize() - 1)) {
        *not_null(errcode_ret) = CL_INVALID_HOST_PTR;
        LogWarning("invalid parameter \"marker bus address\"");
        return static_cast<cl_mem>(NULL);
      }
    }
  }

  // check extensions flag consistency
  if ((flags & CL_MEM_USE_PERSISTENT_MEM_AMD) &&
      (flags & (CL_MEM_USE_HOST_PTR | CL_MEM_ALLOC_HOST_PTR | CL_MEM_EXTERNAL_PHYSICAL_AMD |
                CL_MEM_BUS_ADDRESSABLE_AMD))) {
    *not_null(errcode_ret) = CL_INVALID_VALUE;
    LogWarning("conflicting flags CL_MEM_USE_PERSISTENT_MEM_AMD and host memory specific flags");
    return (cl_mem)0;
  }

  if ((flags & CL_MEM_EXTERNAL_PHYSICAL_AMD) || (flags & CL_MEM_BUS_ADDRESSABLE_AMD)) {
    size = (size + (amd::Os::pageSize() - 1)) & (~(amd::Os::pageSize() - 1));
  }

  amd::Context& amdContext = *as_amd(context);
  amd::Memory* mem = NULL;
  // check if the ptr is in the svm space, if yes, we need return SVM buffer
  amd::Memory* svmMem = amd::MemObjMap::FindMemObj(host_ptr);
  if ((NULL != svmMem) && (flags & CL_MEM_USE_HOST_PTR)) {
    size_t svmSize = svmMem->getSize();
    size_t offset = static_cast<address>(host_ptr) - static_cast<address>(svmMem->getSvmPtr());
    if (size + offset > svmSize) {
      LogWarning("invalid parameter \"size\"");
      *not_null(errcode_ret) = CL_INVALID_BUFFER_SIZE;
      return (cl_mem)0;
    }
    mem = new (amdContext) amd::Buffer(*svmMem, flags, offset, size);
    svmMem->setHostMem(host_ptr);
  } else {
    mem = new (amdContext) amd::Buffer(amdContext, flags, size);
  }

  if (mem == NULL) {
    *not_null(errcode_ret) = CL_OUT_OF_HOST_MEMORY;
    return (cl_mem)0;
  }

  if (!mem->create(host_ptr)) {
    *not_null(errcode_ret) = CL_MEM_OBJECT_ALLOCATION_FAILURE;
    mem->release();
    return NULL;
  }

  *not_null(errcode_ret) = CL_SUCCESS;
  return as_cl(mem);
}
RUNTIME_EXIT

RUNTIME_ENTRY_RET(cl_mem, clCreateSubBuffer,
                  (cl_mem mem, cl_mem_flags flags, cl_buffer_create_type buffer_create_type,
                   const void* buffer_create_info, cl_int* errcode_ret)) {
  if (!is_valid(mem) || as_amd(mem)->asBuffer() == NULL) {
    *not_null(errcode_ret) = CL_INVALID_MEM_OBJECT;
    return NULL;
  }
  amd::Buffer& buffer = *as_amd(mem)->asBuffer();

  // check flags for validity
  if (!validateFlags(flags) || (buffer_create_type != CL_BUFFER_CREATE_TYPE_REGION)) {
    *not_null(errcode_ret) = CL_INVALID_VALUE;
    return NULL;
  }

  if (buffer.getMemFlags() & (CL_MEM_EXTERNAL_PHYSICAL_AMD | CL_MEM_BUS_ADDRESSABLE_AMD)) {
    *not_null(errcode_ret) = CL_INVALID_VALUE;
    return NULL;
  }

  const cl_buffer_region* region = (const cl_buffer_region*)buffer_create_info;

  // Check sub buffer offset alignment
  bool alignmentPass = false;
  const std::vector<amd::Device*>& devices = buffer.getContext().devices();
  for (auto& dev : devices) {
    cl_uint deviceAlignmentBytes = dev->info().memBaseAddrAlign_ >> 3;
    if (region->origin == amd::alignDown(region->origin, deviceAlignmentBytes)) {
      alignmentPass = true;
    }
  }

  // Return an error if the offset is misaligned on all devices
  if (!alignmentPass) {
    *not_null(errcode_ret) = CL_MISALIGNED_SUB_BUFFER_OFFSET;
    return NULL;
  }

  // check size
  if ((region->size == 0) || (region->origin + region->size) > buffer.getSize()) {
    *not_null(errcode_ret) = CL_INVALID_BUFFER_SIZE;
    return NULL;
  }

  amd::Memory* mem = new (buffer.getContext())
      amd::Buffer(buffer, (flags) ? flags : buffer.getMemFlags(), region->origin, region->size);
  if (mem == NULL) {
    *not_null(errcode_ret) = CL_OUT_OF_HOST_MEMORY;
    return NULL;
  }

  if (!mem->create(NULL)) {
    *not_null(errcode_ret) = CL_MEM_OBJECT_ALLOCATION_FAILURE;
    mem->release();
    return NULL;
  }

  *not_null(errcode_ret) = CL_SUCCESS;
  return as_cl(mem);
}
RUNTIME_EXIT

/*! @}
 *  \addtogroup CL_ReadWriteBuffer
 *  @{
 */

/*! \brief Enqueue a command to read from a buffer object to host memory.
 *
 *  \param command_queue refers to the command-queue in which the read / write
 *  command will be queued. \a command_queue and \a buffer must be created with
 *  the same OpenCL context.
 *
 *  \param buffer refers to a valid buffer object.
 *
 *  \param blocking_read indicates if the read operation is blocking or
 *  nonblocking. If \a blocking_read is CL_TRUE i.e. the read command is
 *  blocking, clEnqueueReadBuffer does not return until the buffer data has been
 *  read and copied into memory pointed to by ptr.
 *  If \a blocking_read is CL_FALSE i.e. the read command is non-blocking,
 *  clEnqueueReadBuffer queues a non-blocking read command and returns. The
 *  contents of the buffer that ptr points to cannot be used until the read
 *  command has completed. The \a event argument returns an event object which
 *  can be used to query the execution status of the read command. When the read
 *  command has completed, the contents of the buffer that ptr points to can be
 *  used by the application.
 *
 *  \param offset is the offset in bytes in the buffer object to read from or
 *  write to.
 *
 *  \param cb is the size in bytes of data being read or written.
 *
 *  \param ptr is the pointer to buffer in host memory where data is to be read
 *  into or to be written from.
 *
 *  \param num_events_in_wait_list specifies the number of event objects in
 *  \a event_wait_list.
 *
 *  \param event_wait_list specifies events that need to complete before this
 *  particular command can be executed. If \a event_wait_list is NULL,
 *  then this particular command does not wait on  any event to complete.
 *  If \a event_wait_list is NULL, \a num_events_in_wait_list must be 0.
 *  If \a event_wait_list is not NULL, the list of events pointed to by
 *  \a event_wait_list must be valid and \a num_events_in_wait_list must be
 *  greater than 0. The events specified in \a event_wait_list act as
 *  synchronization points.
 *
 *  \param event returns an event object that identifies this particular read
 *  command and can be used to query or queue a wait for this particular command
 *  to complete. \a event can be NULL in which case it will not be possible for
 *  the application to query the status of this command or queue a wait for this
 *  command to complete.
 *
 *  \return CL_SUCCESS if the function is executed successfully. Otherwise it
 *  returns one of the following errors:
 *  - CL_INVALID_COMMAND_QUEUE if \a command_queue is not a valid command-queue.
 *  - CL_INVALID_CONTEXT if the context associated with \a command_queue and
 *    \a buffer are not the same.
 *  - CL_INVALID_MEM_OBJECT if \a buffer is not a valid buffer object.
 *  - CL_INVALID_VALUE if the region being read or written specified by (offset,
 *    cb) is out of bounds or if \a ptr is a NULL value.
 *  - CL_INVALID_OPERATION if \a clEnqueueReadBuffer is called on buffer which
 *    has been created with CL_MEM_HOST_WRITE_ONLY or CL_MEM_HOST_NO_ACCESS.
 *  - CL_INVALID_EVENT_WAIT_LIST if \a event_wait_list is NULL and \a
 *    num_events_in_wait_list > 0, or \a event_wait_list is not NULL and
 *    \a num_events_in_wait_list is 0, or if event objects in \a event_wait_list
 *    are not valid events.
 *  - CL_OUT_OF_HOST_MEMORY if there is a failure to allocate resources required
 *    by the runtime.
 *
 *  \version 1.2r07
 */
RUNTIME_ENTRY(cl_int, clEnqueueReadBuffer,
              (cl_command_queue command_queue, cl_mem buffer, cl_bool blocking_read, size_t offset,
               size_t cb, void* ptr, cl_uint num_events_in_wait_list,
               const cl_event* event_wait_list, cl_event* event)) {
  if (!is_valid(command_queue)) {
    return CL_INVALID_COMMAND_QUEUE;
  }

  if (!is_valid(buffer)) {
    return CL_INVALID_MEM_OBJECT;
  }
  amd::Buffer* srcBuffer = as_amd(buffer)->asBuffer();
  if (srcBuffer == NULL) {
    return CL_INVALID_MEM_OBJECT;
  }

  if (srcBuffer->getMemFlags() & (CL_MEM_HOST_WRITE_ONLY | CL_MEM_HOST_NO_ACCESS)) {
    return CL_INVALID_OPERATION;
  }

  amd::HostQueue* queue = as_amd(command_queue)->asHostQueue();
  if (NULL == queue) {
    return CL_INVALID_COMMAND_QUEUE;
  }
  amd::HostQueue& hostQueue = *queue;

  if (hostQueue.context() != srcBuffer->getContext()) {
    return CL_INVALID_CONTEXT;
  }

  if (ptr == NULL) {
    return CL_INVALID_VALUE;
  }

  amd::Coord3D srcOffset(offset, 0, 0);
  amd::Coord3D srcSize(cb, 1, 1);

  if (!srcBuffer->validateRegion(srcOffset, srcSize)) {
    return CL_INVALID_VALUE;
  }

  amd::Command::EventWaitList eventWaitList;
  cl_int err =
      amd::clSetEventWaitList(eventWaitList, hostQueue, num_events_in_wait_list, event_wait_list);
  if (err != CL_SUCCESS) {
    return err;
  }

  amd::CopyMetadata copyMetadata(!blocking_read, amd::CopyMetadata::CopyEnginePreference::SDMA);
  amd::ReadMemoryCommand* command =
      new amd::ReadMemoryCommand(hostQueue, CL_COMMAND_READ_BUFFER, eventWaitList, *srcBuffer,
                                 srcOffset, srcSize, ptr, 0, 0, copyMetadata);

  if (command == NULL) {
    return CL_OUT_OF_HOST_MEMORY;
  }

  // Make sure we have memory for the command execution
  if (!command->validateMemory()) {
    delete command;
    return CL_MEM_OBJECT_ALLOCATION_FAILURE;
  }

  command->enqueue();
  if (blocking_read) {
    command->awaitCompletion();
  }

  *not_null(event) = as_cl(&command->event());
  if (event == NULL) {
    command->release();
  }
  return CL_SUCCESS;
}
RUNTIME_EXIT

/*! \brief Enqueue a command to write to  a  buffer  object  from  host memory.
 *
 *  \param command_queue refers to the command-queue in which the  read / write
 *  command will be queued. \a command_queue and \a buffer must be created with
 *  the same OpenCL context.
 *
 *  \param buffer refers to a valid buffer object.
 *
 *  \param blocking_write indicates if  the  write  operation  is  blocking  or
 *  non-blocking. If \a blocking_write is CL_TRUE,  the  OpenCL  implementation
 *  copies the data referred to by \a ptr and enqueues the write  operation  in
 *  the command-queue. The memory pointed to by \a ptr can  be  reused  by  the
 *  application after the clEnqueueWriteBuffer call returns. If
 *  \a blocking_write is CL_FALSE, the OpenCL implementation will use \a ptr to
 *  perform a nonblocking write. As the write is non-blocking the implementation
 *  can return immediately. The memory pointed to by \a ptr cannot be reused by
 *  the application after the call returns. The \a event  argument  returns  an
 *  event object which can be used to query the execution status of  the  write
 *  command. When the write command has completed, the  memory  pointed  to  by
 *  \a ptr can then be reused by the application
 *
 *  \param offset is the offset in bytes in the buffer object to read  from  or
 *  write to.
 *
 *  \param cb is the size in bytes of data being read or written.
 *
 *  \param ptr is the pointer to buffer in host memory where data is to be read
 *  into or to be written from.
 *
 *  \param num_events_in_wait_list specifies the number  of  event  objects  in
 *  \a event_wait_list.
 *
 *  \param event_wait_list specifies events that need to complete  before  this
 *  particular command can be executed.      If  \a  event_wait_list  is  NULL,
 *  then this particular command does  not  wait  on  any  event  to  complete.
 *  If \a event_wait_list is NULL, \a num_events_in_wait_list must be 0.
 *  If \a event_wait_list is not NULL, the list of events pointed to by
 *  \a event_wait_list must be valid and \a num_events_in_wait_list must be
 *  greater than 0. The events specified in \a event_wait_list act as
 *  synchronization points.
 *
 *  \param event returns an event object that identifies this particular write
 *  command and can be used to query or queue a wait for this particular command
 *  to complete. \a event can be NULL in which case it will not be possible for
 *  the application to query the status of this command or queue a wait for this
 *  command to complete.
 *
 *  \return CL_SUCCESS if the function is executed successfully. Otherwise it
 *  returns one of the following errors:
 *  - CL_INVALID_COMMAND_QUEUE if \a command_queue is not a valid command-queue.
 *  - CL_INVALID_CONTEXT if the context associated with \a command_queue and
 *    \a buffer are not the same.
 *  - CL_INVALID_MEM_OBJECT if \a buffer is not a valid buffer object.
 *  - CL_INVALID_VALUE if the region being read or written specified by (offset,
 *    cb) is out of bounds or if \a ptr is a NULL value.
 *  - CL_INVALID_OPERATION if \a clEnqueueWriteBuffer is called on buffer which
 *    has been created with CL_MEM_HOST_READ_ONLY or CL_MEM_HOST_NO_ACCESS.
 *  - CL_INVALID_EVENT_WAIT_LIST if \a event_wait_list is NULL and \a
 *    num_events_in_wait_list > 0, or \a event_wait_list is not NULL and
 *    \a num_events_in_wait_list is 0, or if event objects in \a event_wait_list
 *    are not valid events.
 *  - CL_OUT_OF_HOST_MEMORY if there is a failure to allocate resources required
 *    by the runtime.
 *
 *  \version 1.0r33
 */
RUNTIME_ENTRY(cl_int, clEnqueueWriteBuffer,
              (cl_command_queue command_queue, cl_mem buffer, cl_bool blocking_write, size_t offset,
               size_t cb, const void* ptr, cl_uint num_events_in_wait_list,
               const cl_event* event_wait_list, cl_event* event)) {
  if (!is_valid(command_queue)) {
    return CL_INVALID_COMMAND_QUEUE;
  }

  if (!is_valid(buffer)) {
    return CL_INVALID_MEM_OBJECT;
  }
  amd::Buffer* dstBuffer = as_amd(buffer)->asBuffer();
  if (dstBuffer == NULL) {
    return CL_INVALID_MEM_OBJECT;
  }

  if (dstBuffer->getMemFlags() & (CL_MEM_HOST_READ_ONLY | CL_MEM_HOST_NO_ACCESS)) {
    return CL_INVALID_OPERATION;
  }

  amd::HostQueue* queue = as_amd(command_queue)->asHostQueue();
  if (NULL == queue) {
    return CL_INVALID_COMMAND_QUEUE;
  }
  amd::HostQueue& hostQueue = *queue;

  if (hostQueue.context() != dstBuffer->getContext()) {
    return CL_INVALID_CONTEXT;
  }

  if (ptr == NULL) {
    return CL_INVALID_VALUE;
  }

  amd::Coord3D dstOffset(offset, 0, 0);
  amd::Coord3D dstSize(cb, 1, 1);

  if (!dstBuffer->validateRegion(dstOffset, dstSize)) {
    return CL_INVALID_VALUE;
  }

  amd::Command::EventWaitList eventWaitList;
  cl_int err =
      amd::clSetEventWaitList(eventWaitList, hostQueue, num_events_in_wait_list, event_wait_list);
  if (err != CL_SUCCESS) {
    return err;
  }

  amd::CopyMetadata copyMetadata(!blocking_write, amd::CopyMetadata::CopyEnginePreference::SDMA);
  amd::WriteMemoryCommand* command =
      new amd::WriteMemoryCommand(hostQueue, CL_COMMAND_WRITE_BUFFER, eventWaitList, *dstBuffer,
                                  dstOffset, dstSize, ptr, 0, 0, copyMetadata);

  if (command == NULL) {
    return CL_OUT_OF_HOST_MEMORY;
  }

  // Make sure we have memory for the command execution
  if (!command->validateMemory()) {
    delete command;
    return CL_MEM_OBJECT_ALLOCATION_FAILURE;
  }

  command->enqueue();
  if (blocking_write) {
    command->awaitCompletion();
  }

  *not_null(event) = as_cl(&command->event());
  if (event == NULL) {
    command->release();
  }
  return CL_SUCCESS;
}
RUNTIME_EXIT

/*! \brief Enqueues a command to copy a buffer object to another
 *
 *  \param command_queue refers to the command-queue in which the copy command
 *  will be queued. The OpenCL context associated with \a command_queue,
 *  \a src_buffer and \a dst_buffer must be the same.
 *
 *  \param src_buffer is the source buffer object.
 *
 *  \param dst_buffer is the destination buffer object.
 *
 *  \param src_offset refers to the offset where to begin reading data in
 *  \a src_buffer.
 *
 *  \param dst_offset refers to the offset where to begin copying data in
 *  \a dst_buffer.
 *
 *  \param cb refers to the size in bytes to copy.
 *
 *  \param num_events_in_wait_list specifies the number of event objects in
 *  \a event_wait_list.
 *
 *  \param event_wait_list specifies events that need to complete before this
 *  particular command can be executed. If \a event_wait_list is NULL,
 *  then this particular command does not wait on  any event to complete.
 *  If \a event_wait_list is NULL, \a num_events_in_wait_list must be 0.
 *  If \a event_wait_list is not NULL, the list of events pointed to by
 *  \a event_wait_list must be valid and \a num_events_in_wait_list must be
 *  greater than 0. The events specified in \a event_wait_list act as
 *  synchronization points.
 *
 *  \param event returns an event object that identifies this particular copy
 *  command and can be used to query or queue a wait for this particular command
 *  to complete. \a event can be NULL in which case it will not be possible for
 *  the application to query the status of this command or queue and wait for
 *  this command to complete. clEnqueueBarrier can be used instead.
 *
 *  \return CL_SUCCESS if the function is executed successfully. Otherwise it
 *  returns one of the following errors:
 *  - CL_INVALID_COMMAND_QUEUE if \a command_queue is not a valid command-queue.
 *  - CL_INVALID_CONTEXT if the context associated with \a command_queue,
 *    \a src_buffer and \a dst_buffer are not the same.
 *  - CL_INVALID_MEM_OBJECT if \a src_buffer and \a dst_buffer are not valid
 *    buffer objects.
 *  - CL_INVALID_VALUE if \a src_offset, \a dst_offset, \a cb, \a src_offset +
 *    \a cb or \a dst_offset + \a cb require accessing elements outside the
 *    buffer memory objects.
 *  - CL_INVALID_EVENT_WAIT_LIST if \a event_wait_list is NULL and
 *    \a num_events_in_wait_list > 0, or \a event_wait_list is not NULL and
 *    \a num_events_in_wait_list is 0, or if event objects in \a event_wait_list
 *    are not valid events.
 *  - CL_OUT_OF_HOST_MEMORY if there is a failure to allocate resources required
 *    by the runtime
 *
 *  \version 1.0r33
 */
RUNTIME_ENTRY(cl_int, clEnqueueCopyBuffer,
              (cl_command_queue command_queue, cl_mem src_buffer, cl_mem dst_buffer,
               size_t src_offset, size_t dst_offset, size_t cb, cl_uint num_events_in_wait_list,
               const cl_event* event_wait_list, cl_event* event)) {
  if (!is_valid(command_queue)) {
    return CL_INVALID_COMMAND_QUEUE;
  }

  if (!is_valid(src_buffer) || !is_valid(dst_buffer)) {
    return CL_INVALID_MEM_OBJECT;
  }
  amd::Buffer* srcBuffer = as_amd(src_buffer)->asBuffer();
  amd::Buffer* dstBuffer = as_amd(dst_buffer)->asBuffer();
  if (srcBuffer == NULL || dstBuffer == NULL) {
    return CL_INVALID_MEM_OBJECT;
  }

  amd::HostQueue* queue = as_amd(command_queue)->asHostQueue();
  if (NULL == queue) {
    return CL_INVALID_COMMAND_QUEUE;
  }
  amd::HostQueue& hostQueue = *queue;

  if (hostQueue.context() != srcBuffer->getContext() ||
      hostQueue.context() != dstBuffer->getContext()) {
    return CL_INVALID_CONTEXT;
  }

  amd::Coord3D srcOffset(src_offset, 0, 0);
  amd::Coord3D dstOffset(dst_offset, 0, 0);
  amd::Coord3D size(cb, 1, 1);

  if (!srcBuffer->validateRegion(srcOffset, size) || !dstBuffer->validateRegion(dstOffset, size)) {
    return CL_INVALID_VALUE;
  }

  if (srcBuffer == dstBuffer && ((src_offset <= dst_offset && dst_offset < src_offset + cb) ||
                                 (dst_offset <= src_offset && src_offset < dst_offset + cb))) {
    return CL_MEM_COPY_OVERLAP;
  }

  amd::Command::EventWaitList eventWaitList;
  cl_int err =
      amd::clSetEventWaitList(eventWaitList, hostQueue, num_events_in_wait_list, event_wait_list);
  if (err != CL_SUCCESS) {
    return err;
  }

  amd::CopyMemoryCommand* command =
      new amd::CopyMemoryCommand(hostQueue, CL_COMMAND_COPY_BUFFER, eventWaitList, *srcBuffer,
                                 *dstBuffer, srcOffset, dstOffset, size);

  if (command == NULL) {
    return CL_OUT_OF_HOST_MEMORY;
  }

  // Make sure we have memory for the command execution
  if (!command->validateMemory()) {
    delete command;
    return CL_MEM_OBJECT_ALLOCATION_FAILURE;
  }

  command->enqueue();

  *not_null(event) = as_cl(&command->event());
  if (event == NULL) {
    command->release();
  }
  return CL_SUCCESS;
}
RUNTIME_EXIT

/*! \brief clEnqueueReadBufferRect enqueues commands to read a 2D or 3D rectangular
 *  region from a buffer object to host memory.
 *
 *  \param command_queue refers to the command-queue in which the read / write
 *  command will be queued. command_queue and buffer must be created with the same
 *  OpenCL context. buffer refers to a valid buffer object.
 *
 *  \param blocking_read indicates if the read operations are blocking or
 *  nonblocking.
 *  If \a blocking_read is CL_TRUE i.e. the read command is blocking,
 *  clEnqueueReadBufferRect does not return until the buffer data has been read
 *  and copied into memory pointed to by ptr.
 *  If blocking_read is CL_FALSE i.e. the read command is non-blocking,
 *  clEnqueueReadBufferRect queues a non-blocking read command and returns.
 *  The contents of the buffer that ptr points to cannot be used until
 *  the read command has completed. The event argument returns an event object
 *  which can be used to query the execution status of the read command.
 *  When the read command has completed, the contents of the buffer that
 *  ptr points to can be used by the application.
 *
 *  \buffer_origin defines the (x, y, z) offset in the memory region associated
 *  with buffer. For a 2D rectangle region, the z value given by buffer_origin[2]
 *  should be 0. The offset in bytes is computed as
 *  buffer_origin[2] * buffer_slice_pitch + buffer_origin[1] * buffer_row_pitch +
 *  buffer_origin[0].
 *
 *  \host_origin defines the (x, y, z) offset in the memory region pointed to
 *  by ptr. For a 2D rectangle region, the z value given by host_origin[2]
 *  should be 0. The offset in bytes is computed as
 *  host_origin[2] * host_slice_pitch + host_origin[1] * host_row_pitch +
 *  host_origin[0].
 *
 *  \param region defines the (width, height, depth) in bytes of the 2D or 3D
 *  rectangle being read or written.
 *  For a 2D rectangle copy, the depth value given by region[2] should be 1.
 *
 *  \param buffer_row_pitch is the length of each row in bytes to be used for
 *  the memory region associated with buffer. If \a buffer_row_pitch is 0,
 *  \a buffer_row_pitch is computed as region[0].
 *
 *  \param buffer_slice_pitch is the length of each 2D slice in bytes to be used
 *  for the memory region associated with buffer. If \a buffer_slice_pitch is 0,
 *  \a buffer_slice_pitch is computed as region[1] * \a buffer_row_pitch.
 *
 *  \param host_row_pitch is the length of each row in bytes to be used for
 *  the memory region pointed to by ptr. If \a host_row_pitch is 0, \a host_row_pitch
 *  is computed as region[0].
 *
 *  \param host_slice_pitch is the length of each 2D slice in bytes to be used
 *  for the memory region pointed to by ptr. If \a host_slice_pitch is 0,
 *  \a host_slice_pitch is computed as region[1] * \a host_row_pitch.
 *  ptr is the pointer to buffer in host memory where data is to be read into
 *  or to be written from.
 *
 *  \param event_wait_list and \a num_events_in_wait_list specify events that
 *  need to complete before this particular command can be executed.
 *  If \a event_wait_list is NULL, then this particular command does not wait on any
 *  event to complete. If \a event_wait_list is NULL, \a num_events_in_wait_list
 *  must be 0. If \a event_wait_list is not NULL, the list of events pointed to
 *  by \a event_wait_list must be valid and \a num_events_in_wait_list
 *  must be greater than 0. The events specified in \a event_wait_list act as
 *  synchronization points. The context associated with events in
 *  \a event_wait_list and \a command_queue must be the same.
 *
 *  \param event returns an event object that identifies this particular
 *  read / write command and can be used to query or queue a wait for this
 *  particular command to complete. event can be NULL in which case it will not
 *  be possible for the application to query the status of this command or queue a
 *  wait for this command to complete.
 *
 *  \return CL_SUCCESS if the function is executed successfully. Otherwise,
 *  it returns one of the following errors:
 *   - CL_INVALID_COMMAND_QUEUE if command_queue is not a valid command-queue.
 *   - CL_INVALID_CONTEXT if the context associated with command_queue and
 *     buffer are not the same or if the context associated with \a command_queue
 *      and events in event_wait_list are not the same.
 *   - CL_INVALID_MEM_OBJECT if buffer is not a valid buffer object.
 *   - CL_INVALID_VALUE if the region being read or written specified by
 *     (buffer_origin, region) is out of bounds.
 *   - CL_INVALID_VALUE if ptr is a NULL value.
 *   - CL_INVALID_OPERATION if \a clEnqueueReadBufferRect is called on buffer which
 *     has been created with CL_MEM_HOST_WRITE_ONLY or CL_MEM_HOST_NO_ACCESS.
 *   - CL_INVALID_EVENT_WAIT_LIST if event_wait_list is NULL and
 *     \a num_events_in_wait_list > 0, or event_wait_list is not NULL and
 *     \a num_events_in_wait_list is 0, or if event objects in \a event_wait_list
 *     are not valid events.
 *   - CL_MISALIGNED_SUB_BUFFER_OFFSET if buffer is a sub-buffer object and offset
 *     specified when the sub-buffer object is created is not aligned to
 *   - CL_DEVICE_MEM_BASE_ADDR_ALIGN value for device associated with queue.
 *   - CL_MEM_OBJECT_ALLOCATION_FAILURE if there is a failure to allocate memory
 *     for data store associated with buffer.
 *   - CL_OUT_OF_RESOURCES if there is a failure to allocate resources required
 *     by the OpenCL implementation on the device.
 *   - CL_OUT_OF_HOST_MEMORY if there is a failure to allocate resources
 *     required by the OpenCL implementation on the host.
 *
 *  \version 1.2r07
 */
RUNTIME_ENTRY(cl_int, clEnqueueReadBufferRect,
              (cl_command_queue command_queue, cl_mem buffer, cl_bool blocking_read,
               const size_t* buffer_origin, const size_t* host_origin, const size_t* region,
               size_t buffer_row_pitch, size_t buffer_slice_pitch, size_t host_row_pitch,
               size_t host_slice_pitch, void* ptr, cl_uint num_events_in_wait_list,
               const cl_event* event_wait_list, cl_event* event)) {
  // Validate command queue
  if (!is_valid(command_queue)) {
    return CL_INVALID_COMMAND_QUEUE;
  }

  // Validate opencl buffer
  if (!is_valid(buffer)) {
    return CL_INVALID_MEM_OBJECT;
  }
  amd::Buffer* srcBuffer = as_amd(buffer)->asBuffer();
  if (srcBuffer == NULL) {
    return CL_INVALID_MEM_OBJECT;
  }

  if (srcBuffer->getMemFlags() & (CL_MEM_HOST_WRITE_ONLY | CL_MEM_HOST_NO_ACCESS)) {
    return CL_INVALID_OPERATION;
  }

  amd::HostQueue* queue = as_amd(command_queue)->asHostQueue();
  if (NULL == queue) {
    return CL_INVALID_COMMAND_QUEUE;
  }
  amd::HostQueue& hostQueue = *queue;

  if (hostQueue.context() != srcBuffer->getContext()) {
    return CL_INVALID_CONTEXT;
  }
  // Make sure we have a valid system memory pointer
  if (ptr == NULL) {
    return CL_INVALID_VALUE;
  }

  // Create buffer rectangle info structure
  amd::BufferRect bufRect;
  amd::BufferRect hostRect;

  if (!bufRect.create(buffer_origin, region, buffer_row_pitch, buffer_slice_pitch) ||
      !hostRect.create(host_origin, region, host_row_pitch, host_slice_pitch)) {
    return CL_INVALID_VALUE;
  }

  amd::Coord3D srcStart(bufRect.start_, 0, 0);
  amd::Coord3D srcEnd(bufRect.end_, 1, 1);

  if (!srcBuffer->validateRegion(srcStart, srcEnd)) {
    return CL_INVALID_VALUE;
  }

  amd::Command::EventWaitList eventWaitList;
  cl_int err =
      amd::clSetEventWaitList(eventWaitList, hostQueue, num_events_in_wait_list, event_wait_list);
  if (err != CL_SUCCESS) {
    return err;
  }

  amd::Coord3D size(region[0], region[1], region[2]);
  amd::CopyMetadata copyMetadata(!blocking_read, amd::CopyMetadata::CopyEnginePreference::SDMA);
  amd::ReadMemoryCommand* command =
      new amd::ReadMemoryCommand(hostQueue, CL_COMMAND_READ_BUFFER_RECT, eventWaitList, *srcBuffer,
                                 srcStart, size, ptr, bufRect, hostRect, copyMetadata);
  if (command == NULL) {
    return CL_OUT_OF_HOST_MEMORY;
  }

  // Make sure we have memory for the command execution
  if (!command->validateMemory()) {
    delete command;
    return CL_MEM_OBJECT_ALLOCATION_FAILURE;
  }

  command->enqueue();
  if (blocking_read) {
    command->awaitCompletion();
  }

  *not_null(event) = as_cl(&command->event());
  if (event == NULL) {
    command->release();
  }
  return CL_SUCCESS;
}
RUNTIME_EXIT

/*! \brief clEnqueueWriteBufferRect enqueues commands to write a 2D or 3D
 *  rectangular region to a buffer object from host memory.
 *
 *  \param command_queue refers to the command-queue in which the read / write
 *  command will be queued. command_queue and buffer must be created with the same
 *  OpenCL context. buffer refers to a valid buffer object.
 *
 *  \param blocking_write indicates if the write operations are blocking or
 *  nonblocking.
 *  If \a blocking_write is CL_TRUE, the OpenCL implementation copies the data
 *  referred to by ptr and enqueues the write operation in the command-queue.
 *  The memory pointed to by ptr can be reused by the application after
 *  the clEnqueueWriteBufferRect call returns.
 *  If \a blocking_write is CL_FALSE, the OpenCL implementation will use ptr to
 *  perform a nonblocking write. As the write is non-blocking the implementation
 *  can return immediately. The memory pointed to by ptr cannot be reused by
 *  the application after the call returns. The event argument returns
 *  an event object which can be used to query the execution status of the write
 *  command. When the write command has completed, the memory pointed to by ptr
 *  can then be reused by the application.
 *
 *  \buffer_origin defines the (x, y, z) offset in the memory region associated
 *  with buffer. For a 2D rectangle region, the z value given by buffer_origin[2]
 *  should be 0. The offset in bytes is computed as
 *  buffer_origin[2] * buffer_slice_pitch + buffer_origin[1] * buffer_row_pitch +
 *  buffer_origin[0].
 *
 *  \host_origin defines the (x, y, z) offset in the memory region pointed to
 *  by ptr. For a 2D rectangle region, the z value given by host_origin[2]
 *  should be 0. The offset in bytes is computed as
 *  host_origin[2] * host_slice_pitch + host_origin[1] * host_row_pitch +
 *  host_origin[0].
 *
 *  \param region defines the (width, height, depth) in bytes of the 2D or 3D
 *  rectangle being read or written.
 *  For a 2D rectangle copy, the depth value given by region[2] should be 1.
 *
 *  \param buffer_row_pitch is the length of each row in bytes to be used for
 *  the memory region associated with buffer. If \a buffer_row_pitch is 0,
 *  \a buffer_row_pitch is computed as region[0].
 *
 *  \param buffer_slice_pitch is the length of each 2D slice in bytes to be used
 *  for the memory region associated with buffer. If \a buffer_slice_pitch is 0,
 *  \a buffer_slice_pitch is computed as region[1] * \a buffer_row_pitch.
 *
 *  \param host_row_pitch is the length of each row in bytes to be used for
 *  the memory region pointed to by ptr. If \a host_row_pitch is 0, \a host_row_pitch
 *  is computed as region[0].
 *
 *  \param host_slice_pitch is the length of each 2D slice in bytes to be used
 *  for the memory region pointed to by ptr. If \a host_slice_pitch is 0,
 *  \a host_slice_pitch is computed as region[1] * \a host_row_pitch.
 *  ptr is the pointer to buffer in host memory where data is to be read into
 *  or to be written from.
 *
 *  \param event_wait_list and \a num_events_in_wait_list specify events that
 *  need to complete before this particular command can be executed.
 *  If \a event_wait_list is NULL, then this particular command does not wait on any
 *  event to complete. If \a event_wait_list is NULL, \a num_events_in_wait_list
 *  must be 0. If \a event_wait_list is not NULL, the list of events pointed to
 *  by \a event_wait_list must be valid and \a num_events_in_wait_list
 *  must be greater than 0. The events specified in \a event_wait_list act as
 *  synchronization points. The context associated with events in
 *  \a event_wait_list and \a command_queue must be the same.
 *
 *  \param event returns an event object that identifies this particular
 *  read / write command and can be used to query or queue a wait for this
 *  particular command to complete. event can be NULL in which case it will not
 *  be possible for the application to query the status of this command or queue a
 *  wait for this command to complete.
 *
 *  clEnqueueReadBufferRect and clEnqueueWriteBufferRect
 *  \return CL_SUCCESS if the function is executed successfully. Otherwise,
 *  it returns one of the following errors:
 *   - CL_INVALID_COMMAND_QUEUE if command_queue is not a valid command-queue.
 *   - CL_INVALID_CONTEXT if the context associated with command_queue and
 *     buffer are not the same or if the context associated with \a command_queue
 *      and events in event_wait_list are not the same.
 *   - CL_INVALID_MEM_OBJECT if buffer is not a valid buffer object.
 *   - CL_INVALID_VALUE if the region being read or written specified by
 *     (buffer_origin, region) is out of bounds.
 *   - CL_INVALID_VALUE if ptr is a NULL value.
 *   - CL_INVALID_OPERATION if \a clEnqueueWriteBufferRect is called on buffer
 *     which has been created with CL_MEM_HOST_READ_ONLY or CL_MEM_HOST_NO_ACCESS.
 *   - CL_INVALID_EVENT_WAIT_LIST if event_wait_list is NULL and
 *     \a num_events_in_wait_list > 0, or event_wait_list is not NULL and
 *     \a num_events_in_wait_list is 0, or if event objects in \a event_wait_list
 *     are not valid events.
 *   - CL_MISALIGNED_SUB_BUFFER_OFFSET if buffer is a sub-buffer object and offset
 *     specified when the sub-buffer object is created is not aligned to
 *   - CL_DEVICE_MEM_BASE_ADDR_ALIGN value for device associated with queue.
 *   - CL_MEM_OBJECT_ALLOCATION_FAILURE if there is a failure to allocate memory
 *     for data store associated with buffer.
 *   - CL_OUT_OF_RESOURCES if there is a failure to allocate resources required
 *     by the OpenCL implementation on the device.
 *   - CL_OUT_OF_HOST_MEMORY if there is a failure to allocate resources
 *     required by the OpenCL implementation on the host.
 */
RUNTIME_ENTRY(cl_int, clEnqueueWriteBufferRect,
              (cl_command_queue command_queue, cl_mem buffer, cl_bool blocking_write,
               const size_t* buffer_origin, const size_t* host_origin, const size_t* region,
               size_t buffer_row_pitch, size_t buffer_slice_pitch, size_t host_row_pitch,
               size_t host_slice_pitch, const void* ptr, cl_uint num_events_in_wait_list,
               const cl_event* event_wait_list, cl_event* event)) {
  if (!is_valid(command_queue)) {
    return CL_INVALID_COMMAND_QUEUE;
  }

  if (!is_valid(buffer)) {
    return CL_INVALID_MEM_OBJECT;
  }
  amd::Buffer* dstBuffer = as_amd(buffer)->asBuffer();
  if (dstBuffer == NULL) {
    return CL_INVALID_MEM_OBJECT;
  }

  if (dstBuffer->getMemFlags() & (CL_MEM_HOST_READ_ONLY | CL_MEM_HOST_NO_ACCESS)) {
    return CL_INVALID_OPERATION;
  }

  amd::HostQueue* queue = as_amd(command_queue)->asHostQueue();
  if (NULL == queue) {
    return CL_INVALID_COMMAND_QUEUE;
  }
  amd::HostQueue& hostQueue = *queue;

  if (hostQueue.context() != dstBuffer->getContext()) {
    return CL_INVALID_CONTEXT;
  }

  if (ptr == NULL) {
    return CL_INVALID_VALUE;
  }

  // Create buffer rectangle info structure
  amd::BufferRect bufRect;
  amd::BufferRect hostRect;

  if (!bufRect.create(buffer_origin, region, buffer_row_pitch, buffer_slice_pitch) ||
      !hostRect.create(host_origin, region, host_row_pitch, host_slice_pitch)) {
    return CL_INVALID_VALUE;
  }

  amd::Coord3D dstStart(bufRect.start_, 0, 0);
  amd::Coord3D dstEnd(bufRect.end_, 1, 1);

  if (!dstBuffer->validateRegion(dstStart, dstEnd)) {
    return CL_INVALID_VALUE;
  }

  amd::Command::EventWaitList eventWaitList;
  cl_int err =
      amd::clSetEventWaitList(eventWaitList, hostQueue, num_events_in_wait_list, event_wait_list);
  if (err != CL_SUCCESS) {
    return err;
  }

  amd::Coord3D size(region[0], region[1], region[2]);
  amd::CopyMetadata copyMetadata(!blocking_write, amd::CopyMetadata::CopyEnginePreference::SDMA);
  amd::WriteMemoryCommand* command =
      new amd::WriteMemoryCommand(hostQueue, CL_COMMAND_WRITE_BUFFER_RECT, eventWaitList,
                                  *dstBuffer, dstStart, size, ptr, bufRect, hostRect, copyMetadata);
  if (command == NULL) {
    return CL_OUT_OF_HOST_MEMORY;
  }

  // Make sure we have memory for the command execution
  if (!command->validateMemory()) {
    delete command;
    return CL_MEM_OBJECT_ALLOCATION_FAILURE;
  }

  command->enqueue();
  if (blocking_write) {
    command->awaitCompletion();
  }

  *not_null(event) = as_cl(&command->event());
  if (event == NULL) {
    command->release();
  }
  return CL_SUCCESS;
}
RUNTIME_EXIT

/*! \brief Enqueues a command to copy a 2D or 3D rectangular region from
 *  the buffer object identified by \a src_buffer to a 2D or 3D region
 *  in the buffer object identified by \a dst_buffer.
 *
 *  \param command_queue refers to the command-queue in which the copy command
 *  will be queued. The OpenCL context associated with command_queue,
 *  \a src_buffer and \a dst_buffer must be the same.
 *
 *  \param src_origin defines the (x, y, z) offset in the memory region
 *  associated with \a src_buffer. For a 2D rectangle region, the z value given
 *  by src_origin[2] should be 0. The offset in bytes is computed as
 *  src_origin[2] * src_slice_pitch + src_origin[1] * src_row_pitch + src_origin[0].
 *
 *  \param dst_origin defines the (x, y, z) offset in the memory region
 *  associated with \a dst_buffer. For a 2D rectangle region, the z value given
 *  by dst_origin[2] should be 0. The offset in bytes is computed as
 *  dst_origin[2] * dst_slice_pitch + dst_origin[1] * dst_row_pitch + dst_origin[0].
 *
 *  \param region defines the (width, height, depth) in bytes of the 2D or 3D
 *  rectangle being copied. For a 2D rectangle, the depth value given by
 *  region[2] should be 1.
 *
 *  \param pasrc_row_pitch is the length of each row in bytes to be used for
 *  the memory region associated with src_buffer. If src_row_pitch is 0,
 *  src_row_pitch is computed as region[0].
 *
 *  \param src_slice_pitch is the length of each 2D slice in bytes to be used
 *  for the memory region associated with src_buffer. If src_slice_pitch is 0,
 *  src_slice_pitch is computed as region[1] * src_row_pitch.
 *
 *  \param dst_row_pitch is the length of each row in bytes to be used for
 *  the memory region associated with dst_buffer. If dst_row_pitch is 0,
 *  dst_row_pitch is computed as region[0].
 *
 *  \param dst_slice_pitch is the length of each 2D slice in bytes to be used
 *  for the memory region associated with dst_buffer. If dst_slice_pitch is 0,
 *  dst_slice_pitch is computed as region[1] * dst_row_pitch.
 *
 *  \param event_wait_list and num_events_in_wait_list specify events that
 *  need to complete before this particular command can be executed.
 *  If event_wait_list is NULL, then this particular command does not wait on
 *  any event to complete. If event_wait_list is NULL, num_events_in_wait_list
 *  must be 0. If event_wait_list is not NULL, the list of events pointed to by
 *  event_wait_list must be valid and num_events_in_wait_list must be greater
 *  than 0. The events specified in event_wait_list act as synchronization
 *  points. The context associated with events in event_wait_list and
 *  command_queue must be the same.
 *
 *  \param event returns an event object that identifies this particular copy
 *  command and can be used to query or queue a wait for this particular
 *  command to complete. event can be NULL in which case it will not be
 *  possible for the application to query the status of this command or queue
 *  a wait for this command to complete. clEnqueueBarrier can be used instead.
 *
 *  \return CL_SUCCESS if the function is executed successfully. Otherwise,
 *  it returns one of the following errors:
 *   - CL_INVALID_COMMAND_QUEUE if command_queue is not a valid command-queue.
 *   - CL_INVALID_CONTEXT if the context associated with command_queue,
 *     \a src_buffer and \a dst_buffer are not the same or if the context
 *     associated with \a command_queue and in \a event_wait_list are not the same.
 *   - CL_INVALID_MEM_OBJECT if \a src_buffer and \a dst_buffer are not valid
 *     buffer objects.
 *   - CL_INVALID_VALUE if (\a src_offset, \a region) or (\a dst_offset,
 *     \a region) require accessing elements outside the \a src_buffer and
 *     \a dst_buffer buffer objects respectively.
 *   - CL_INVALID_EVENT_WAIT_LIST if \a event_wait_list is NULL and
 *     \a num_events_in_wait_list > 0, or \a event_wait_list is not NULL and
 *     \a num_events_in_wait_list is 0, or if event objects in
 *     \a event_wait_list are not valid events.
 *   - CL_MEM_COPY_OVERLAP if \a src_buffer and \a dst_buffer are the same
 *     buffer object and the source and destination regions overlap.
 *   - CL_MISALIGNED_SUB_BUFFER_OFFSET if \a src_buffer is a sub-buffer object
 *     and offset specified when the sub-buffer object is created is
 *     not aligned to CL_DEVICE_MEM_BASE_ADDR_ALIGN value for device
 *     associated with queue.
 *   - CL_MISALIGNED_SUB_BUFFER_OFFSET if dst_buffer is a sub-buffer object
 *     and offset specified when the sub-buffer object is created is not
 *     aligned to CL_DEVICE_MEM_BASE_ADDR_ALIGN value for device associated
 *     with queue.
 *   - CL_MEM_OBJECT_ALLOCATION_FAILURE if there is a failure to allocate
 *     memory for data store associated with src_buffer or dst_buffer.
 *   - CL_OUT_OF_RESOURCES if there is a failure to allocate resources
 *     required by the OpenCL implementation on the device.
 *   - CL_OUT_OF_HOST_MEMORY if there is a failure to allocate resources
 *     required by the OpenCL implementation on the host
 *
 */
RUNTIME_ENTRY(cl_int, clEnqueueCopyBufferRect,
              (cl_command_queue command_queue, cl_mem src_buffer, cl_mem dst_buffer,
               const size_t* src_origin, const size_t* dst_origin, const size_t* region,
               size_t src_row_pitch, size_t src_slice_pitch, size_t dst_row_pitch,
               size_t dst_slice_pitch, cl_uint num_events_in_wait_list,
               const cl_event* event_wait_list, cl_event* event)) {
  if (!is_valid(command_queue)) {
    return CL_INVALID_COMMAND_QUEUE;
  }

  if (!is_valid(src_buffer) || !is_valid(dst_buffer)) {
    return CL_INVALID_MEM_OBJECT;
  }
  amd::Buffer* srcBuffer = as_amd(src_buffer)->asBuffer();
  amd::Buffer* dstBuffer = as_amd(dst_buffer)->asBuffer();
  if (srcBuffer == NULL || dstBuffer == NULL) {
    return CL_INVALID_MEM_OBJECT;
  }

  amd::HostQueue* queue = as_amd(command_queue)->asHostQueue();
  if (NULL == queue) {
    return CL_INVALID_COMMAND_QUEUE;
  }
  amd::HostQueue& hostQueue = *queue;

  if (hostQueue.context() != srcBuffer->getContext() ||
      hostQueue.context() != dstBuffer->getContext()) {
    return CL_INVALID_CONTEXT;
  }

  // Create buffer rectangle info structure
  amd::BufferRect srcRect;
  amd::BufferRect dstRect;

  if (!srcRect.create(src_origin, region, src_row_pitch, src_slice_pitch) ||
      !dstRect.create(dst_origin, region, dst_row_pitch, dst_slice_pitch)) {
    return CL_INVALID_VALUE;
  }

  amd::Coord3D srcStart(srcRect.start_, 0, 0);
  amd::Coord3D dstStart(dstRect.start_, 0, 0);
  amd::Coord3D srcEnd(srcRect.end_, 1, 1);
  amd::Coord3D dstEnd(dstRect.end_, 1, 1);

  if (!srcBuffer->validateRegion(srcStart, srcEnd) ||
      !dstBuffer->validateRegion(dstStart, dstEnd)) {
    return CL_INVALID_VALUE;
  }

  // Check if regions overlap each other
  if ((srcBuffer == dstBuffer) &&
      (std::abs(static_cast<long>(src_origin[0]) - static_cast<long>(dst_origin[0])) <
       static_cast<long>(region[0])) &&
      (std::abs(static_cast<long>(src_origin[1]) - static_cast<long>(dst_origin[1])) <
       static_cast<long>(region[1])) &&
      (std::abs(static_cast<long>(src_origin[2]) - static_cast<long>(dst_origin[2])) <
       static_cast<long>(region[2]))) {
    return CL_MEM_COPY_OVERLAP;
  }

  amd::Command::EventWaitList eventWaitList;
  cl_int err =
      amd::clSetEventWaitList(eventWaitList, hostQueue, num_events_in_wait_list, event_wait_list);
  if (err != CL_SUCCESS) {
    return err;
  }

  amd::Coord3D size(region[0], region[1], region[2]);
  amd::CopyMemoryCommand* command =
      new amd::CopyMemoryCommand(hostQueue, CL_COMMAND_COPY_BUFFER_RECT, eventWaitList, *srcBuffer,
                                 *dstBuffer, srcStart, dstStart, size, srcRect, dstRect);

  if (command == NULL) {
    return CL_OUT_OF_HOST_MEMORY;
  }

  // Make sure we have memory for the command execution
  if (!command->validateMemory()) {
    delete command;
    return CL_MEM_OBJECT_ALLOCATION_FAILURE;
  }

  command->enqueue();

  *not_null(event) = as_cl(&command->event());
  if (event == NULL) {
    command->release();
  }
  return CL_SUCCESS;
}
RUNTIME_EXIT

/*! @}
 *  \addtogroup CL_MemoryCallback
 *  @{
 */

/*! \brief Registers a user callback function that will be called when the
 *   memory object is deleted and its resources freed.
 *
 * Each call to clSetMemObjectDestructorCallback registers the specified user
 * callback function on a callback stack associated with memobj. The registered
 * user callback functions are called in the reverse order in which they were
 * registered. The user callback functions are called and then the memory
 * object’s resources are freed and the memory object is deleted.
 * This provides a mechanism for the application (and libraries) using memobj
 * to be notified when the memory referenced by host_ptr, specified when
 * the memory object is created and used as the storage bits for the memory
 * object, can be reused or freed.
 *
 * \a memobj is a valid memory object.
 * \a pfn_notify is the callback function that can be registered by the
 *    application. This callback function may be called asynchronously by the
 *    OpenCL implementation. It is the application’s responsibility to ensure
 *    that the callback function is thread-safe. The parameters to this callback
 *    function are:
 *      - memobj is the memory object being deleted.
 *      - user_data is a pointer to user supplied data.
 *    If pfn_notify is NULL, no callback function is registered for memobj.
 * \a user_data will be passed as the user_data argument when pfn_notify is
 *    called. user_data can be NULL.
 *
 * \return CL_SUCCESS if the function is executed successfully. Otherwise it
 * returns one of the following errors:
 *   - CL_INVALID_MEM_OBJECT if memobj is not a valid memory object.
 *   - CL_OUT_OF_HOST_MEMORY if there is a failure to allocate resources
 *     required by the OpenCL implementation on the host.
 *
 * NOTE: When the user callback function is called by the implementation, the
 * contents of the memory region pointed to by host_ptr (if the memory object is
 * created with CL_MEM_USE_HOST_PTR) are undefined. The callback function is
 * typically used by the application to either free or reuse the memory region
 * pointed to by host_ptr. The behavior of calling expensive system routines,
 * OpenCL API calls to create contexts or command-queues, or blocking OpenCL
 * operations from the following list below, in a callback is undefined.
 *
 *  \version 1.1r17
 */
RUNTIME_ENTRY(cl_int, clSetMemObjectDestructorCallback,
              (cl_mem memobj, void(CL_CALLBACK* pfn_notify)(cl_mem memobj, void* user_data),
               void* user_data)) {
  if (!is_valid(memobj)) {
    return CL_INVALID_MEM_OBJECT;
  }

  if (pfn_notify == NULL) {
    return CL_INVALID_VALUE;
  }

  if (!as_amd(memobj)->setDestructorCallback(pfn_notify, user_data)) {
    return CL_OUT_OF_HOST_MEMORY;
  }

  return CL_SUCCESS;
}
RUNTIME_EXIT

/*! @}
 *  \addtogroup CL_RetRelMemory
 *  @{
 */

/*! \brief Increment the \a memobj reference count.
 *
 *  \return CL_SUCCESS if the function is executed successfully or
 *  CL_INVALID_MEM_OBJECT if \a memobj is not a valid memory object.
 *
 *  clCreateBuffer and clCreateImage{2D|3D} perform an implicit retain.
 *
 *  \version 1.0r33
 */
RUNTIME_ENTRY(cl_int, clRetainMemObject, (cl_mem memobj)) {
  if (!is_valid(memobj)) {
    return CL_INVALID_MEM_OBJECT;
  }
  as_amd(memobj)->retain();
  return CL_SUCCESS;
}
RUNTIME_EXIT

/*! \brief Decrement the \a memobj reference count.
 *
 *  After the \a memobj reference count becomes zero and commands queued for
 *  execution on a command-queue(s) that use \a memobj have finished, the
 *  memory object is deleted.
 *
 *  \return CL_SUCCESS if the function is executed successfully or
 *  CL_INVALID_MEM_OBJECT if \a memobj is not a valid memory object.
 *
 *  \version 1.0r33
 */
RUNTIME_ENTRY(cl_int, clReleaseMemObject, (cl_mem memobj)) {
  if (!is_valid(memobj)) {
    return CL_INVALID_MEM_OBJECT;
  }
  as_amd(memobj)->release();
  return CL_SUCCESS;
}
RUNTIME_EXIT

/*! @}
 *  \addtogroup CL_CreatingImage
 *  @{
 */

/*! \brief Create a (1D, or 2D) image object.
 *
 *  \param context is a valid OpenCL context on which the image object is to be
 *  created.
 *
 *  \param flags is a bit-field that is used to specify allocation and usage
 *  information about the image memory object being created.
 *
 *  \param image_format is a pointer to a structure that describes format
 *  properties of the image to be allocated.
 *
 *  \param image_width is the width of the image in pixels. Must be greater
 *  than or equal to 1.
 *
 *  \param image_height is the height of the image in pixels. Must be greater
 *  than or equal to 1.
 *
 *  \param image_row_pitch is the scan-line pitch in bytes. This must be 0 if
 *  \a host_ptr is NULL and can be either 0 or >= \a image_width * size of
 *  element in bytes if \a host_ptr is not NULL. If \a host_ptr is not NULL and
 *  \a image_row_pitch = 0, \a image_row_pitch is calculated as
 *  \a image_width * size of element in bytes.
 *
 *  \param host_ptr is a pointer to the image data that may already be allocated
 *  by the application. The size of the buffer that \a host_ptr points to must
 *  be >= \a image_row_pitch * \a image_height. The size of each element in
 *  bytes must be a power of 2. Passing in a pointer to an already allocated
 *  buffer on the host and using it as a memory object allows applications to
 *  share data efficiently with kernels and the host.
 *
 *  \param errcode_ret will return an appropriate error code. If \a errcode_ret
 *  is NULL, no error code is returned.
 *
 *  \return A valid non-zero image object and errcode_ret is set to CL_SUCCESS
 *  if the image object is created successfully. It returns a NULL value with
 *  one of the following error values returned in \a errcode_ret:
 *  - CL_INVALID_CONTEXT if \a context is not a valid context.
 *  - CL_INVALID_VALUE if values specified in \a flags are not valid.
 *  - CL_INVALID_IMAGE_FORMAT_DESCRIPTOR if values specified in \a image_format
 *    are not valid or if \a image_format is NULL.
 *  - CL_INVALID_IMAGE_SIZE if \a image_width or \a image_height are 0 or if
 *    they exceed values specified in CL_DEVICE_IMAGE2D_MAX_WIDTH or
 *    CL_DEVICE_IMAGE2D_MAX_HEIGHT respectively or if values specified by
 *    \a image_row_pitch do not follow rules described in the argument
 *    description above.
 *  - CL_INVALID_HOST_PTR if \a host_ptr is NULL and CL_MEM_USE_HOST_PTR or
 *    CL_MEM_COPY_HOST_PTR are set in \a flags or if \a host_ptr is not NULL
 *    but CL_MEM_COPY_HOST_PTR or CL_MEM_USE_HOST_PTR are not set in \a flags.
 *  - CL_IMAGE_FORMAT_NOT_SUPPORTED if the \a image_format is not supported.
 *  - CL_MEM_OBJECT_ALLOCATION_FAILURE if there is a failure to allocate memory
 *    for image object.
 *  - CL_INVALID_OPERATION if the image object as specified by the
 *    \a image_format, \a flags and dimensions cannot be created for all devices
 *    in context that support images or if there are no devices in context that
 *    support images.
 *  - CL_OUT_OF_HOST_MEMORY if there is a failure to allocate resources required
 *    by the runtime.
 *
 *  \version 1.0r33
 */
RUNTIME_ENTRY_RET(cl_mem, clCreateImage2D,
                  (cl_context context, cl_mem_flags flags, const cl_image_format* image_format,
                   size_t image_width, size_t image_height, size_t image_row_pitch, void* host_ptr,
                   cl_int* errcode_ret)) {
  if (!is_valid(context)) {
    *not_null(errcode_ret) = CL_INVALID_CONTEXT;
    LogWarning("invalid parameter \"context\"");
    return (cl_mem)0;
  }
  // check flags for validity
  if (!validateFlags(flags)) {
    *not_null(errcode_ret) = CL_INVALID_VALUE;
    LogWarning("invalid parameter \"flags\"");
    return (cl_mem)0;
  }
  // check format
  if (image_format == NULL) {
    *not_null(errcode_ret) = CL_INVALID_IMAGE_FORMAT_DESCRIPTOR;
    LogWarning("invalid parameter \"image_format\"");
    return (cl_mem)0;
  }

  const amd::Image::Format imageFormat(*image_format);
  if (!imageFormat.isValid()) {
    *not_null(errcode_ret) = CL_INVALID_IMAGE_FORMAT_DESCRIPTOR;
    LogWarning("invalid parameter \"image_format\"");
    return (cl_mem)0;
  }

  amd::Context& amdContext = *as_amd(context);
  if (!imageFormat.isSupported(amdContext)) {
    *not_null(errcode_ret) = CL_IMAGE_FORMAT_NOT_SUPPORTED;
    LogWarning("invalid parameter \"image_format\"");
    return (cl_mem)0;
  }
  // check size parameters
  if (image_width == 0 || image_height == 0) {
    *not_null(errcode_ret) = CL_INVALID_IMAGE_SIZE;
    LogWarning("invalid parameter \"image_width\" or \"image_height\"");
    return (cl_mem)0;
  }
  const std::vector<amd::Device*>& devices = as_amd(context)->devices();
  bool supportPass = false;
  bool sizePass = false;
  for (auto& dev : devices) {
    if (dev->info().imageSupport_) {
      supportPass = true;
      if (dev->info().image2DMaxWidth_ >= image_width &&
          dev->info().image2DMaxHeight_ >= image_height) {
        sizePass = true;
        break;
      }
    }
  }
  if (!supportPass) {
    *not_null(errcode_ret) = CL_INVALID_OPERATION;
    LogWarning("there are no devices in context to support images");
    return (cl_mem)0;
  }
  if (!sizePass) {
    *not_null(errcode_ret) = CL_INVALID_IMAGE_SIZE;
    LogWarning("invalid parameter \"image_width\" or \"image_height\"");
    return (cl_mem)0;
  }
  // check row pitch rules
  if (host_ptr == NULL) {
    if (image_row_pitch) {
      *not_null(errcode_ret) = CL_INVALID_IMAGE_SIZE;
      LogWarning("invalid parameter \"image_row_pitch\"");
      return (cl_mem)0;
    }
  } else if (image_row_pitch) {
    size_t elemSize = imageFormat.getElementSize();
    if ((image_row_pitch < image_width * elemSize) || (image_row_pitch % elemSize)) {
      *not_null(errcode_ret) = CL_INVALID_IMAGE_SIZE;
      LogWarning("invalid parameter \"image_row_pitch\"");
      return (cl_mem)0;
    }
  }
  // check host_ptr consistency
  if (host_ptr == NULL) {
    if (flags & (CL_MEM_USE_HOST_PTR | CL_MEM_COPY_HOST_PTR)) {
      *not_null(errcode_ret) = CL_INVALID_HOST_PTR;
      LogWarning("invalid parameter \"host_ptr\"");
      return (cl_mem)0;
    }
  } else {
    if (!(flags & (CL_MEM_USE_HOST_PTR | CL_MEM_COPY_HOST_PTR))) {
      *not_null(errcode_ret) = CL_INVALID_HOST_PTR;
      LogWarning("invalid parameter \"host_ptr\"");
      return (cl_mem)0;
    }
  }

  // CL_IMAGE_FORMAT_NOT_SUPPORTED ???

  if (image_row_pitch == 0) {
    image_row_pitch = image_width * imageFormat.getElementSize();
  }

  amd::Image* image =
      new (amdContext) amd::Image(amdContext, CL_MEM_OBJECT_IMAGE2D, flags, imageFormat,
                                  image_width, image_height, 1, image_row_pitch, 0);
  if (image == NULL) {
    *not_null(errcode_ret) = CL_OUT_OF_HOST_MEMORY;
    LogWarning("cannot allocate resources");
    return (cl_mem)0;
  }

  // CL_MEM_OBJECT_ALLOCATION_FAILURE
  if (!image->create(host_ptr)) {
    *not_null(errcode_ret) = CL_MEM_OBJECT_ALLOCATION_FAILURE;
    image->release();
    return (cl_mem)0;
  }

  *not_null(errcode_ret) = CL_SUCCESS;
  return (cl_mem)as_cl<amd::Memory>(image);
}
RUNTIME_EXIT

/*! \brief Create a 3D image object.
 *
 *  \param context is a valid OpenCL context on which the image object is to be
 *  created.
 *
 *  \param flags is a bit-field that is used to specify allocation and usage
 *  information about the image memory object being created.
 *
 *  \param image_format is a pointer to a structure that describes format
 *  properties of the image to be allocated.
 *
 *  \param image_width is the width of the image in pixels. Must be greater
 *  than or equal to 1.
 *
 *  \param image_height is the height of the image in pixels. Must be greater
 *  than or equal to 1.
 *
 *  \param image_depth is the depth of the image in pixels. This must be a
 *  value > 1.
 *
 *  \param image_row_pitch is the scan-line pitch in bytes. This must be 0 if
 *  \a host_ptr is NULL and can be either 0 or >= \a image_width * size of
 *  element in bytes if \a host_ptr is not NULL. If \a host_ptr is not NULL and
 *  \a image_row_pitch = 0, \a image_row_pitch is calculated as
 *  \a image_width * size of element in bytes.
 *
 *  \param image_slice_pitch is the size in bytes of each 2D slice in the 3D
 *  image. This must be 0 if \a host_ptr is NULL and can be either 0 or >=
 *  \a image_row_pitch * \a image_height if \a host_ptr is not NULL.
 *  If \a host_ptr is not NULL and \a image_slice_pitch = 0,
 *  \a image_slice_pitch is calculated as \a image_row_pitch * \a image_height.
 *
 *  \param host_ptr is a pointer to the image data that may already be allocated
 *  by the application. The size of the buffer that \a host_ptr points to must
 *  be >= \a image_row_pitch * \a image_height * \a image_depth. The size of
 *  each element in bytes must be a power of 2. Passing in a pointer to an
 *  already allocated buffer on the host and using it as a memory object allows
 *  applications to share data efficiently with kernels and the host.
 *
 *  \param errcode_ret will return an appropriate error code. If \a errcode_ret
 *  is NULL, no error code is returned.
 *
 *  \return valid non-zero image object created and the \a errcode_ret is set to
 *  CL_SUCCESS if the image object is created successfully. It returns a NULL
 *  value with one of the following error values returned in \a errcode_ret:
 *  - CL_INVALID_CONTEXT if \a context is not a valid context.
 *  - CL_INVALID_VALUE if values specified in \a flags are not valid.
 *  - CL_INVALID_IMAGE_FORMAT_DESCRIPTOR if values specified in \a image_format
 *    are not valid or if \a image_format is NULL.
 *  - CL_INVALID_IMAGE_SIZE if \a image_width, \a image_height or \a image_depth
 *    are 0 or if they exceed values specified in CL_DEVICE_IMAGE3D_MAX_WIDTH,
 *    CL_DEVICE_IMAGE3D_MAX_HEIGHT or CL_DEVICE_IMAGE3D_MAX_DEPTH respectively
 *    or if values specified by \a image_row_pitch and \a image_slice_pitch do
 *    not follow rules described in the argument description above.
 *  - CL_INVALID_HOST_PTR if \a host_ptr is NULL and CL_MEM_USE_HOST_PTR or
 *    CL_MEM_COPY_HOST_PTR are set in \a flags or if \a host_ptr is not NULL but
 *    CL_MEM_COPY_HOST_PTR or CL_MEM_USE_HOST_PTR are not set in \a flags.
 *  - CL_IMAGE_FORMAT_NOT_SUPPORTED if the \a image_format is not supported.
 *  - CL_MEM_OBJECT_ALLOCATION_FAILURE if there is a failure to allocate memory
 *    for image object.
 *  - CL_INVALID_OPERATION if the image object as specified by the
 *    \a image_format, \a flags and dimensions cannot be created for all devices
 *    in context that support images, or if there are no devices in context that
 *    support images.
 *  - CL_OUT_OF_HOST_MEMORY if there is a failure to allocate resources required
 *    by the runtime.
 *
 *  \version 1.0r33
 */
RUNTIME_ENTRY_RET(cl_mem, clCreateImage3D,
                  (cl_context context, cl_mem_flags flags, const cl_image_format* image_format,
                   size_t image_width, size_t image_height, size_t image_depth,
                   size_t image_row_pitch, size_t image_slice_pitch, void* host_ptr,
                   cl_int* errcode_ret)) {
  if (!is_valid(context)) {
    *not_null(errcode_ret) = CL_INVALID_CONTEXT;
    LogWarning("invalid parameter \"context\"");
    return (cl_mem)0;
  }
  // check flags for validity
  if (!validateFlags(flags)) {
    *not_null(errcode_ret) = CL_INVALID_VALUE;
    LogWarning("invalid parameter \"flags\"");
    return (cl_mem)0;
  }
  // check format
  if (image_format == NULL) {
    *not_null(errcode_ret) = CL_INVALID_IMAGE_FORMAT_DESCRIPTOR;
    LogWarning("invalid parameter \"image_format\"");
    return (cl_mem)0;
  }
  amd::Image::Format imageFormat(*image_format);

  if (!imageFormat.isValid()) {
    *not_null(errcode_ret) = CL_INVALID_IMAGE_FORMAT_DESCRIPTOR;
    LogWarning("invalid parameter \"image_format\"");
    return (cl_mem)0;
  }

  amd::Context& amdContext = *as_amd(context);
  if (!imageFormat.isSupported(amdContext)) {
    *not_null(errcode_ret) = CL_IMAGE_FORMAT_NOT_SUPPORTED;
    LogWarning("invalid parameter \"image_format\"");
    return (cl_mem)0;
  }
  // check size parameters
  if (image_width == 0 || image_height == 0 || image_depth <= 1) {
    *not_null(errcode_ret) = CL_INVALID_IMAGE_SIZE;
    LogWarning("invalid size parameter(s)");
    return (cl_mem)0;
  }
  const std::vector<amd::Device*>& devices = as_amd(context)->devices();
  bool supportPass = false;
  bool sizePass = false;
  for (auto& dev : devices) {
    if (dev->info().imageSupport_) {
      supportPass = true;
      if ((dev->info().image3DMaxWidth_ >= image_width) &&
          (dev->info().image3DMaxHeight_ >= image_height) &&
          (dev->info().image3DMaxDepth_ >= image_depth)) {
        sizePass = true;
        break;
      }
    }
  }
  if (!supportPass) {
    *not_null(errcode_ret) = CL_INVALID_OPERATION;
    LogWarning("there are no devices in context to support images");
    return (cl_mem)0;
  }
  if (!sizePass) {
    *not_null(errcode_ret) = CL_INVALID_IMAGE_SIZE;
    LogWarning("invalid size parameter(s)");
    return (cl_mem)0;
  }
  // check row pitch rules
  if (host_ptr == NULL) {
    if (image_row_pitch) {
      *not_null(errcode_ret) = CL_INVALID_IMAGE_SIZE;
      LogWarning("invalid parameter \"image_row_pitch\"");
      return (cl_mem)0;
    }
  } else if (image_row_pitch) {
    size_t elemSize = imageFormat.getElementSize();
    if ((image_row_pitch < image_width * elemSize) || (image_row_pitch % elemSize)) {
      *not_null(errcode_ret) = CL_INVALID_IMAGE_SIZE;
      LogWarning("invalid parameter \"image_row_pitch\"");
      return (cl_mem)0;
    }
  }
  // check slice pitch
  if (host_ptr == NULL) {
    if (image_slice_pitch) {
      *not_null(errcode_ret) = CL_INVALID_IMAGE_SIZE;
      LogWarning("invalid parameter \"image_row_pitch\"");
      return (cl_mem)0;
    }
  } else if (image_slice_pitch) {
    size_t elemSize = imageFormat.getElementSize();
    if ((image_slice_pitch < image_row_pitch * image_height) ||
        (image_slice_pitch % image_row_pitch)) {
      *not_null(errcode_ret) = CL_INVALID_IMAGE_SIZE;
      LogWarning("invalid parameter \"image_row_pitch\"");
      return (cl_mem)0;
    }
  }
  // check host_ptr consistency
  if (host_ptr == NULL) {
    if (flags & (CL_MEM_USE_HOST_PTR | CL_MEM_COPY_HOST_PTR)) {
      *not_null(errcode_ret) = CL_INVALID_HOST_PTR;
      LogWarning("invalid parameter \"host_ptr\"");
      return (cl_mem)0;
    }
  } else {
    if (!(flags & (CL_MEM_USE_HOST_PTR | CL_MEM_COPY_HOST_PTR))) {
      *not_null(errcode_ret) = CL_INVALID_HOST_PTR;
      LogWarning("invalid parameter \"host_ptr\"");
      return (cl_mem)0;
    }
  }

  // CL_IMAGE_FORMAT_NOT_SUPPORTED ???

  if (image_row_pitch == 0) {
    image_row_pitch = image_width * imageFormat.getElementSize();
  }
  if (image_slice_pitch == 0) {
    image_slice_pitch = image_row_pitch * image_height;
  }

  amd::Image* image = new (amdContext)
      amd::Image(amdContext, CL_MEM_OBJECT_IMAGE3D, flags, imageFormat, image_width, image_height,
                 image_depth, image_row_pitch, image_slice_pitch);
  if (image == NULL) {
    *not_null(errcode_ret) = CL_OUT_OF_HOST_MEMORY;
    LogWarning("cannot allocate resources");
    return (cl_mem)0;
  }

  // CL_MEM_OBJECT_ALLOCATION_FAILURE
  if (!image->create(host_ptr)) {
    *not_null(errcode_ret) = CL_MEM_OBJECT_ALLOCATION_FAILURE;
    image->release();
    return (cl_mem)0;
  }

  *not_null(errcode_ret) = CL_SUCCESS;
  return (cl_mem)as_cl<amd::Memory>(image);
}
RUNTIME_EXIT

/*! @}
 *  \addtogroup CL_QueryImageFormat
 *  @{
 */

/*! \brief Get the list of supported image formats.
 *
 *  \param context is a valid OpenCL context on which the image object(s) will
 *  be created.
 *
 *  \param flags is a bit-field that is used to specify allocation and usage
 *  information about the image memory object being created.
 *
 *  \param image_type describes the image type and must be either
 *  CL_MEM_OBJECT_IMAGE1D, CL_MEM_OBJECT_IMAGE1D_BUFFER, CL_MEM_OBJECT_IMAGE2D,
 *  CL_MEM_OBJECT_IMAGE3D, CL_MEM_OBJECT_IMAGE1D_ARRAY or
 *  CL_MEM_OBJECT_IMAGE2D_ARRAY.
 *
 *  \param num_entries specifies the number of entries that can be returned in
 *  the memory location given by \a image_formats.
 *
 *  \param image_formats is a pointer to a memory location where the list of
 *  supported image formats are returned. Each entry describes a cl_image_format
 *  structure supported by the runtime. If \a image_formats is NULL, it is
 *  ignored.
 *
 *  \param num_image_formats is the actual number of supported image formats for
 *  a specific context and values specified by \a flags. If \a num_image_formats
 *  is NULL, it is ignored.
 *
 *  \return One of the following values:
 *  - CL_SUCCESS if the function is executed successfully
 *  - CL_INVALID_CONTEXT if \a context is not a valid context
 *  - CL_INVALID_VALUE if \a flags or \a image_type are not valid, or if
 *    \a num_entries is 0 and \a image_formats is not NULL
 *
 *  \version 1.2r08
 */
RUNTIME_ENTRY(cl_int, clGetSupportedImageFormats,
              (cl_context context, cl_mem_flags flags, cl_mem_object_type image_type,
               cl_uint num_entries, cl_image_format* image_formats, cl_uint* num_image_formats)) {
  if (!is_valid(context)) {
    LogWarning("invalid parameter \"context\"");
    return CL_INVALID_CONTEXT;
  }
  // check flags for validity
  if (!validateFlags(flags, true)) {
    LogWarning("invalid parameter \"flags\"");
    return CL_INVALID_VALUE;
  }
  // chack image_type
  switch (image_type) {
    case CL_MEM_OBJECT_IMAGE1D_BUFFER:
    case CL_MEM_OBJECT_IMAGE1D:
    case CL_MEM_OBJECT_IMAGE1D_ARRAY:
    case CL_MEM_OBJECT_IMAGE2D:
    case CL_MEM_OBJECT_IMAGE2D_ARRAY:
    case CL_MEM_OBJECT_IMAGE3D:
      break;

    default:
      LogWarning("invalid parameter \"image_type\"");
      return CL_INVALID_VALUE;
  }
  if (num_entries == 0 && image_formats != NULL) {
    LogWarning("invalid parameter \"num_entries\"");
    return CL_INVALID_VALUE;
  }

  const amd::Context& amdContext = *as_amd(context);

  if (image_formats != NULL) {
    amd::Image::getSupportedFormats(amdContext, image_type, num_entries, image_formats, flags);
  }
  if (num_image_formats != NULL) {
    *num_image_formats = amd::Image::numSupportedFormats(amdContext, image_type, flags);
  }

  return CL_SUCCESS;
}
RUNTIME_EXIT


/*! @}
 *  \addtogroup CL_ReadWriteImage
 *  @{
 */

/*! \brief Enqueue a command to read from a 2D or 3D image object to host memory
 *
 *  \param command_queue refers to the command-queue in which the read
 *  command will be queued. \a command_queue and \a image must be created with
 *  the same OpenCL context.
 *
 *  \param image refers to a valid 2D or 3D image object.
 *
 *  \param blocking_read indicates if the read is blocking or nonblocking. If
 *  \a blocking_read is CL_TRUE i.e. the read command is blocking,
 *  clEnqueueReadImage does not return until the buffer data has been read and
 *  copied into memory pointed to by \a ptr. If \a blocking_read is CL_FALSE
 *  i.e. the read command is non-blocking, clEnqueueReadImage queues a
 *  non-blocking read command and returns. The contents of the buffer that
 *  \a ptr points to cannot be used until the read command has completed.
 *  The \a event argument returns an event object which can be used to query the
 *  execution status of the read command. When the read command has completed,
 *  the contents of the buffer that ptr points to can be used by the application
 *
 *  \param origin defines the (x, y, z) offset in the image from where to read
 *  or write. If image is a 2D image object, the z value given by origin[2] must
 *  be 0.
 *
 *  \param region defines the (width, height, depth) of the 2D or 3D rectangle
 *  being read or written. If image is a 2D image object, the depth value given
 *  by region[2] must be 1.
 *
 *  \param row_pitch in clEnqueueReadImage is the length of each row in bytes.
 *  This value must be greater than or equal to the element size in bytes
 *  width. If \a row_pitch is set to 0, the appropriate row pitch is calculated
 *  based on the size of each element in bytes multiplied by width.
 *
 *  \param slice_pitch in clEnqueueReadImage clEnqueueWriteImage is the size
 *  in bytes of the 2D slice of the 3D region of a 3D image being read or
 *  written respectively. This must be 0 if image is a 2D image. This value
 *  must be greater than or equal to row_pitch * height. If \a slice_pitch is
 *  set to 0, the appropriate slice pitch is calculated based on the
 *  \a row_pitch * \a height.
 *
 *  \param ptr is the pointer to a buffer in host memory where image data is
 *  to be read from.
 *
 *  \param num_events_in_wait_list specifies the number of event objects in
 *  \a event_wait_list.
 *
 *  \param event_wait_list specifies events that need to complete before this
 *  particular command can be executed. If \a event_wait_list is NULL, then this
 *  particular command does not wait on any event to complete. If
 *  \a event_wait_list is NULL, \a num_events_in_wait_list must be 0.
 *  If \a event_wait_list is not NULL, the list of events pointed to by
 *  \a event_wait_list must be valid and \a num_events_in_wait_list must be
 *  greater than 0. The events specified in \a event_wait_list act as
 *  synchronization points.
 *
 *  \param event returns an event object that identifies this particular read
 *  command and can be used to query or queue a wait for this particular command
 *  to complete. \a event can be NULL in which case it will not be possible for
 *  the application to query the status of this command or queue a wait for this
 *  command to complete.
 *
 *  \return CL_SUCCESS if the function is executed successfully. Otherwise it
 *  returns one of the following errors:
 *  - CL_INVALID_COMMAND_QUEUE if \a command_queue is not a valid command-queue.
 *  - CL_INVALID_CONTEXT if the context associated with \a command_queue and
 *    \a image are not the same.
 *  - CL_INVALID_MEM_OBJECT if \a image is not a valid image object.
 *  - CL_INVALID_VALUE if the region being read specified by \a origin and
 *    \a region is out of bounds or if \a ptr is a NULL value.
 *  - CL_INVALID_VALUE if \a image is a 2D image object and \a origin[2] is not
 *    equal to 0 or \a region[2] is not equal to 1 or \a slice_pitch is not
 *    equal to 0.
 *  - CL_INVALID_OPERATION if \a clEnqueueReadImage is called on image which
 *    has been created with CL_MEM_HOST_WRITE_ONLY or CL_MEM_HOST_NO_ACCESS.
 *  - CL_INVALID_EVENT_WAIT_LIST if \a event_wait_list is NULL and
 *    \a num_events_in_wait_list > 0, or \a event_wait_list is not NULL and
 *    \a num_events_in_wait_list is 0, or if event objects in \a event_wait_list
 *    are not valid events.
 *  - CL_INVALID_VALUE if blocking_read is CL_FALSE and \a event is NULL.
 *  - CL_OUT_OF_HOST_MEMORY if there is a failure to allocate resources required
 *    by the runtime.
 *
 *  \version 1.2r07
 */
RUNTIME_ENTRY(cl_int, clEnqueueReadImage,
              (cl_command_queue command_queue, cl_mem image, cl_bool blocking_read,
               const size_t* origin, const size_t* region, size_t row_pitch, size_t slice_pitch,
               void* ptr, cl_uint num_events_in_wait_list, const cl_event* event_wait_list,
               cl_event* event)) {
  if (!is_valid(command_queue)) {
    return CL_INVALID_COMMAND_QUEUE;
  }

  if (!is_valid(image)) {
    return CL_INVALID_MEM_OBJECT;
  }
  amd::Image* srcImage = as_amd(image)->asImage();
  if (srcImage == NULL) {
    return CL_INVALID_MEM_OBJECT;
  }

  if (srcImage->getMemFlags() & (CL_MEM_HOST_WRITE_ONLY | CL_MEM_HOST_NO_ACCESS)) {
    return CL_INVALID_OPERATION;
  }

  if (srcImage->getImageFormat().image_channel_order == CL_DEPTH_STENCIL) {
    return CL_INVALID_OPERATION;
  }

  amd::HostQueue* queue = as_amd(command_queue)->asHostQueue();
  if (NULL == queue) {
    return CL_INVALID_COMMAND_QUEUE;
  }
  amd::HostQueue& hostQueue = *queue;

  if (hostQueue.context() != srcImage->getContext()) {
    return CL_INVALID_CONTEXT;
  }

  if (ptr == NULL) {
    return CL_INVALID_VALUE;
  }

  amd::Coord3D srcOrigin(origin[0], origin[1], origin[2]);
  amd::Coord3D srcRegion(region[0], region[1], region[2]);

  ImageViewRef mip;
  if (srcImage->getMipLevels() > 1) {
    // Create a view for the specified mip level
    mip = srcImage->createView(srcImage->getContext(), srcImage->getImageFormat(), NULL,
                               origin[srcImage->getDims()]);
    if (mip() == NULL) {
      return CL_OUT_OF_HOST_MEMORY;
    }
    // Reset the mip level value to 0, since a view was created
    if (srcImage->getDims() < 3) {
      srcOrigin.c[srcImage->getDims()] = 0;
    }
    srcImage = mip();
  }

  if (!srcImage->validateRegion(srcOrigin, srcRegion) ||
      !srcImage->isRowSliceValid(row_pitch, slice_pitch, region[0], region[1])) {
    return CL_INVALID_VALUE;
  }

  amd::Command::EventWaitList eventWaitList;
  cl_int err =
      amd::clSetEventWaitList(eventWaitList, hostQueue, num_events_in_wait_list, event_wait_list);
  if (err != CL_SUCCESS) {
    return err;
  }

  amd::CopyMetadata copyMetadata(!blocking_read, amd::CopyMetadata::CopyEnginePreference::SDMA);
  amd::ReadMemoryCommand* command =
      new amd::ReadMemoryCommand(hostQueue, CL_COMMAND_READ_IMAGE, eventWaitList, *srcImage,
                                 srcOrigin, srcRegion, ptr, row_pitch, slice_pitch, copyMetadata);

  if (command == NULL) {
    return CL_OUT_OF_HOST_MEMORY;
  }

  // Make sure we have memory for the command execution
  if (!command->validateMemory()) {
    delete command;
    return CL_MEM_OBJECT_ALLOCATION_FAILURE;
  }

  command->enqueue();
  if (blocking_read) {
    command->awaitCompletion();
  }

  *not_null(event) = as_cl(&command->event());
  if (event == NULL) {
    command->release();
  }

  return CL_SUCCESS;
}
RUNTIME_EXIT

/*! \brief Enqueue a command to write to a 2D or 3D image object from host
 *  memory
 *
 *  \param command_queue refers to the command-queue in which the write
 *  command will be queued. \a command_queue and \a image must be created with
 *  the same OpenCL context.
 *
 *  \param image refers to a valid 2D or 3D image object.
 *
 *  \param blocking_write indicates if the write operation is blocking or
 *  nonblocking. If blocking_write is CL_TRUE, the OpenCL implementation copies
 *  the data referred to by \a ptr and enqueues the write command in the
 *  command-queue. The memory pointed to by ptr can be reused by the application
 *  after the clEnqueueWriteImage call returns. If blocking_write is CL_FALSE,
 *  the OpenCL implementation will use ptr to perform a nonblocking write. As
 *  the write is non-blocking the implementation can return immediately. The
 *  memory pointed to by ptr cannot be reused by the application after the call
 *  returns. The event argument returns an event object which can be used to
 *  query the execution status of the write command. When the write command has
 *  completed, the memory pointed to by ptr can then be reused by the
 *  application.
 *
 *  \param origin defines the (x, y, z) offset in the image from where to read
 *  or write. If image is a 2D image object, the z value given by origin[2] must
 *  be 0.
 *
 *  \param region defines the (width, height, depth) of the 2D or 3D rectangle
 *  being read or written. If image is a 2D image object, the depth value given
 *  by region[2] must be 1.
 *
 *  \param input_row_pitch in is the length of each row in bytes.
 *  This value must be greater than or equal to the element size in bytes
 *  width. If \a input_row_pitch is set to 0, the appropriate row pitch is
 *  calculated based on the size of each element in bytes multiplied by width.
 *
 *  \param input_slice_pitch is the size
 *  in bytes of the 2D slice of the 3D region of a 3D image being read or
 *  written respectively. This must be 0 if image is a 2D image. This value
 *  must be greater than or equal to input_row_pitch * height. If
 *  \a input_slice_pitch is  set to 0, the appropriate slice pitch is calculated
 *  based on the  \a input_row_pitch * \a height.
 *
 *  \param ptr is the pointer to a buffer in host memory where image data is
 *  to be written to.
 *
 *  \param num_events_in_wait_list specifies the number of event objects in
 *  \a event_wait_list.
 *
 *  \param event_wait_list specifies events that need to complete before this
 *  particular command can be executed. If \a event_wait_list is NULL, then this
 *  particular command does not wait on any event to complete. If
 *  \a event_wait_list is NULL, \a num_events_in_wait_list must be 0.
 *  If \a event_wait_list is not NULL, the list of events pointed to by
 *  \a event_wait_list must be valid and \a num_events_in_wait_list must be
 *  greater than 0. The events specified in \a event_wait_list act as
 *  synchronization points.
 *
 *  \param event returns an event object that identifies this particular write
 *  command and can be used to query or queue a wait for this particular command
 *  to complete. \a event can be NULL in which case it will not be possible for
 *  the application to query the status of this command or queue a wait for this
 *  command to complete.
 *
 *  \return CL_SUCCESS if the function is executed successfully. Otherwise it
 *  returns one of the following errors:
 *  - CL_INVALID_COMMAND_QUEUE if \a command_queue is not a valid command-queue.
 *  - CL_INVALID_CONTEXT if the context associated with \a command_queue and
 *    \a image are not the same.
 *  - CL_INVALID_MEM_OBJECT if \a image is not a valid image object.
 *  - CL_INVALID_VALUE if the region being written specified by \a origin and
 *    \a region is out of bounds or if \a ptr is a NULL value.
 *  - CL_INVALID_VALUE if \a image is a 2D image object and \a origin[2] is not
 *    equal to 0 or \a region[2] is not equal to 1 or \a slice_pitch is not
 *    equal to 0.
 *  - CL_INVALID_OPERATION if \a clEnqueueWriteImage is called on image which
 *    has been created with CL_MEM_HOST_READ_ONLY or CL_MEM_HOST_NO_ACCESS.
 *  - CL_INVALID_EVENT_WAIT_LIST if \a event_wait_list is NULL and
 *    \a num_events_in_wait_list > 0, or \a event_wait_list is not NULL and
 *    \a num_events_in_wait_list is 0, or if event objects in \a event_wait_list
 *    are not valid events.
 *  - CL_INVALID_VALUE if blocking_write is CL_FALSE and \a event is NULL.
 *  - CL_OUT_OF_HOST_MEMORY if there is a failure to allocate resources required
 *    by the runtime.
 *
 *  \version 1.0r33
 */
RUNTIME_ENTRY(cl_int, clEnqueueWriteImage,
              (cl_command_queue command_queue, cl_mem image, cl_bool blocking_write,
               const size_t* origin, const size_t* region, size_t input_row_pitch,
               size_t input_slice_pitch, const void* ptr, cl_uint num_events_in_wait_list,
               const cl_event* event_wait_list, cl_event* event)) {
  if (!is_valid(command_queue)) {
    return CL_INVALID_COMMAND_QUEUE;
  }

  if (!is_valid(image)) {
    return CL_INVALID_MEM_OBJECT;
  }
  amd::Image* dstImage = as_amd(image)->asImage();
  if (dstImage == NULL) {
    return CL_INVALID_MEM_OBJECT;
  }

  if (dstImage->getMemFlags() & (CL_MEM_HOST_READ_ONLY | CL_MEM_HOST_NO_ACCESS)) {
    return CL_INVALID_OPERATION;
  }

  if (dstImage->getImageFormat().image_channel_order == CL_DEPTH_STENCIL) {
    return CL_INVALID_OPERATION;
  }

  if (dstImage->getDims() == 2 && origin[2] != 0 && dstImage->getMipLevels() == 0) {
    return CL_INVALID_VALUE;
  }

  amd::HostQueue* queue = as_amd(command_queue)->asHostQueue();
  if (NULL == queue) {
    return CL_INVALID_COMMAND_QUEUE;
  }
  amd::HostQueue& hostQueue = *queue;

  if (hostQueue.context() != dstImage->getContext()) {
    return CL_INVALID_CONTEXT;
  }

  if (ptr == NULL) {
    return CL_INVALID_VALUE;
  }

  amd::Coord3D dstOrigin(origin[0], origin[1], origin[2]);
  amd::Coord3D dstRegion(region[0], region[1], region[2]);
  ImageViewRef mip;
  if (dstImage->getMipLevels() > 1) {
    // Create a view for the specified mip level
    mip = dstImage->createView(dstImage->getContext(), dstImage->getImageFormat(), NULL,
                               origin[dstImage->getDims()]);
    if (mip() == NULL) {
      return CL_OUT_OF_HOST_MEMORY;
    }
    // Reset the mip level value to 0, since a view was created
    if (dstImage->getDims() < 3) {
      dstOrigin.c[dstImage->getDims()] = 0;
    }
    dstImage = mip();
  }

  if (!dstImage->validateRegion(dstOrigin, dstRegion) ||
      !dstImage->isRowSliceValid(input_row_pitch, input_slice_pitch, region[0], region[1])) {
    return CL_INVALID_VALUE;
  }

  amd::Command::EventWaitList eventWaitList;
  cl_int err =
      amd::clSetEventWaitList(eventWaitList, hostQueue, num_events_in_wait_list, event_wait_list);
  if (err != CL_SUCCESS) {
    return err;
  }

  amd::CopyMetadata copyMetadata(!blocking_write, amd::CopyMetadata::CopyEnginePreference::SDMA);
  amd::WriteMemoryCommand* command = new amd::WriteMemoryCommand(
      hostQueue, CL_COMMAND_WRITE_IMAGE, eventWaitList, *dstImage, dstOrigin, dstRegion, ptr,
      input_row_pitch, input_slice_pitch, copyMetadata);

  if (command == NULL) {
    return CL_OUT_OF_HOST_MEMORY;
  }

  // Make sure we have memory for the command execution
  if (!command->validateMemory()) {
    delete command;
    return CL_MEM_OBJECT_ALLOCATION_FAILURE;
  }

  command->enqueue();
  if (blocking_write) {
    command->awaitCompletion();
  }

  *not_null(event) = as_cl(&command->event());
  if (event == NULL) {
    command->release();
  }
  return CL_SUCCESS;
}
RUNTIME_EXIT

/*! \brief Enqueue a command to copy image objects.
 *
 *  \param command_queue refers to the command-queue in which the copy command
 *  will be queued. The OpenCL context associated with \a command_queue,
 *  \a src_image and \a dst_image must be the same.
 *
 *  \param src_image is the source image object.
 *
 *  \param dst_image is the destination image object.
 *
 *  \param src_origin defines the starting (x, y, z) location in \a src_image
 *  from where to start the data copy.  If \a src_image is a 2D image object,
 *  the z value given by \a src_origin[2] must be 0.
 *
 *  \param dst_origin defines the starting (x, y, z) location in \a dst_image
 *  from where to start the data copy. If \a dst_image is a 2D image object,
 *  the z value given by \a dst_origin[2] must be 0.
 *
 *  \param region defines the (width, height, depth) of the 2D or 3D rectangle
 *  to copy. If \a src_image or \a dst_image is a 2D image object, the depth
 *  value given by \a region[2] must be 1.
 *
 *  \param num_events_in_wait_list specifies the number of event objects in
 *  \a event_wait_list.
 *
 *  \param event_wait_list specifies events that need to complete before this
 *  particular command can be executed. If \a event_wait_list is NULL, then
 *  this particular command does not wait on any event to complete. If
 *  \a event_wait_list is NULL, \a num_events_in_wait_list must be 0. If
 *  \a event_wait_list is not NULL, the list of events pointed to by
 *  \a event_wait_list must be valid and \a num_events_in_wait_list must be
 *  greater than 0. The events specified in \a event_wait_list act as
 *  synchronization points.
 *
 *  \param event returns an event object that identifies this particular copy
 *  command and can be used to query or queue a wait for this particular
 *  command to complete. \a event can be NULL in which case it will not be
 *  possible for the application to query the status of this command or queue
 *  a wait for this command to complete. clEnqueueBarrier can be used instead.
 *  It is currently a requirement that the \a src_image and \a dst_image image
 *  memory objects for clEnqueueCopyImage must have the exact image format
 *  (i.e. channel order and channel data type must match).
 *
 *  \return CL_SUCCESS if the function is executed successfully. Otherwise it
 *  returns one of the following errors:
 *  - CL_INVALID_COMMAND_QUEUE if \a command_queue is not a valid command-queue.
 *  - CL_INVALID_CONTEXT if the context associated with \a command_queue,
 *    \a src_image and \a dst_image are not the same.
 *  - CL_INVALID_MEM_OBJECT if \a src_image and \a dst_image are not valid image
 *    objects.
 *  - CL_IMAGE_FORMAT_MISMATCH if src_image and dst_image do not use the same
 *    image format.
 *  - CL_INVALID_VALUE if the 2D or 3D rectangular region specified by
 *    \a src_origin and \a src_origin + \a region refers to a region outside
 *    \a src_image, or if the 2D or 3D rectangular region specified by
 *    \a dst_origin and \a dst_origin + \a region refers to a region outside
 *    \a dst_image.
 *  - CL_INVALID_VALUE if \a src_image is a 2D image object and \a origin[2] is
 *    not equal to 0 or \a region[2] is not equal to 1.
 *  - CL_INVALID_VALUE if \a dst_image is a 2D image object and \a dst_origin[2]
 *    is not equal to 0 or \a region[2] is not equal to 1.
 *  - CL_INVALID_EVENT_WAIT_LIST if \a event_wait_list is NULL and
 *    \a num_events_in_wait_list > 0, or \a event_wait_list is not NULL and
 *    \a num_events_in_wait_list is 0, or if event objects in \a event_wait_list
 *    are not valid events.
 *  - CL_OUT_OF_HOST_MEMORY if there is a failure to allocate resources required
 *    by the runtime.
 *
 *  \version 1.0r33
 */
RUNTIME_ENTRY(cl_int, clEnqueueCopyImage,
              (cl_command_queue command_queue, cl_mem src_image, cl_mem dst_image,
               const size_t* src_origin, const size_t* dst_origin, const size_t* region,
               cl_uint num_events_in_wait_list, const cl_event* event_wait_list, cl_event* event)) {
  if (!is_valid(command_queue)) {
    return CL_INVALID_COMMAND_QUEUE;
  }

  if (!is_valid(src_image) || !is_valid(dst_image)) {
    return CL_INVALID_MEM_OBJECT;
  }
  amd::Image* srcImage = as_amd(src_image)->asImage();
  amd::Image* dstImage = as_amd(dst_image)->asImage();

  amd::HostQueue* queue = as_amd(command_queue)->asHostQueue();
  if (NULL == queue) {
    return CL_INVALID_COMMAND_QUEUE;
  }
  amd::HostQueue& hostQueue = *queue;

  if (hostQueue.context() != srcImage->getContext() ||
      hostQueue.context() != dstImage->getContext()) {
    return CL_INVALID_CONTEXT;
  }

  if (srcImage->getImageFormat() != dstImage->getImageFormat()) {
    return CL_IMAGE_FORMAT_MISMATCH;
  }

  if (srcImage->getImageFormat().image_channel_order == CL_DEPTH_STENCIL) {
    return CL_INVALID_OPERATION;
  }

  amd::Coord3D srcOrigin(src_origin[0], src_origin[1], src_origin[2]);
  amd::Coord3D dstOrigin(dst_origin[0], dst_origin[1], dst_origin[2]);
  amd::Coord3D copyRegion(region[0], region[1], region[2]);

  ImageViewRef srcMip;
  if (srcImage->getMipLevels() > 1) {
    // Create a view for the specified mip level
    srcMip = srcImage->createView(srcImage->getContext(), srcImage->getImageFormat(), NULL,
                                  src_origin[srcImage->getDims()]);
    if (srcMip() == NULL) {
      return CL_OUT_OF_HOST_MEMORY;
    }
    // Reset the mip level value to 0, since a view was created
    if (srcImage->getDims() < 3) {
      srcOrigin.c[srcImage->getDims()] = 0;
    }
    srcImage = srcMip();
  }

  if (!srcImage->validateRegion(srcOrigin, copyRegion)) {
    return CL_INVALID_VALUE;
  }

  ImageViewRef dstMip;
  if (dstImage->getMipLevels() > 1) {
    // Create a view for the specified mip level
    dstMip = dstImage->createView(dstImage->getContext(), dstImage->getImageFormat(), NULL,
                                  dst_origin[dstImage->getDims()]);
    if (dstMip() == NULL) {
      return CL_OUT_OF_HOST_MEMORY;
    }
    // Reset the mip level value to 0, since a view was created
    if (dstImage->getDims() < 3) {
      dstOrigin.c[dstImage->getDims()] = 0;
    }
    dstImage = dstMip();
  }

  if (!dstImage->validateRegion(dstOrigin, copyRegion)) {
    return CL_INVALID_VALUE;
  }

  amd::Command::EventWaitList eventWaitList;
  cl_int err =
      amd::clSetEventWaitList(eventWaitList, hostQueue, num_events_in_wait_list, event_wait_list);
  if (err != CL_SUCCESS) {
    return err;
  }

  if (src_image == dst_image) {
    if ((src_origin[0] <= dst_origin[0] && dst_origin[0] < src_origin[0] + region[0]) ||
        (dst_origin[0] <= src_origin[0] && src_origin[0] < dst_origin[0] + region[0]) ||
        (src_origin[1] <= dst_origin[1] && dst_origin[1] < src_origin[1] + region[1]) ||
        (dst_origin[1] <= src_origin[1] && src_origin[1] < dst_origin[1] + region[1])) {
      return CL_MEM_COPY_OVERLAP;
    }
    if (srcImage->getDims() > 2) {
      if ((src_origin[2] <= dst_origin[2] && dst_origin[2] < src_origin[2] + region[2]) ||
          (dst_origin[2] <= src_origin[2] && src_origin[2] < dst_origin[2] + region[2])) {
        return CL_MEM_COPY_OVERLAP;
      }
    }
  }

  amd::CopyMemoryCommand* command =
      new amd::CopyMemoryCommand(hostQueue, CL_COMMAND_COPY_IMAGE, eventWaitList, *srcImage,
                                 *dstImage, srcOrigin, dstOrigin, copyRegion);

  if (command == NULL) {
    return CL_OUT_OF_HOST_MEMORY;
  }

  // Make sure we have memory for the command execution
  if (!command->validateMemory()) {
    delete command;
    return CL_MEM_OBJECT_ALLOCATION_FAILURE;
  }

  command->enqueue();

  *not_null(event) = as_cl(&command->event());
  if (event == NULL) {
    command->release();
  }
  return CL_SUCCESS;
}
RUNTIME_EXIT

/*! @}
 *  \addtogroup CL_CopyingImageBuffer
 *  @{
 */

/*! \brief Enqueue a command to copy an image object to a buffer object.
 *
 *  \param command_queue must be a valid command-queue. The OpenCL context
 *  associated with \a command_queue, \a src_image and \a dst_buffer must be
 *  the same.
 *
 *  \param src_image is a valid image object.
 *
 *  \param dst_buffer is a valid buffer object.
 *
 *  \param src_origin defines the (x, y, z) offset in the image from where to
 *  copy. If \a src_image is a 2D image object, the z value given by
 *  \a src_origin[2] must be 0.
 *
 *  \param region defines the (width, height, depth) of the 2D or 3D rectangle
 *  to copy. If \a src_image is a 2D image object, the depth value given by
 *  \a region[2] must be 1.
 *
 *  \param dst_offset refers to the offset where to begin copying data in
 *  \a dst_buffer. The size in bytes of the region to be copied referred to as
 *  \a dst_cb is computed as width * height * depth * bytes/image element if
 *  \a src_image is a 3D image object and is computed as
 *  width * height * bytes/image element if \a src_image is a 2D image object.
 *
 *  \param num_events_in_wait_list specifies the number of event objects in
 *  \a event_wait_list.
 *
 *  \param event_wait_list specifies events that need to complete before this
 *  particular command can be executed. If \a event_wait_list is NULL, then this
 *  particular command does not wait on any event to complete. If
 *  \a event_wait_list is NULL, \a num_events_in_wait_list must be 0.
 *  If \a event_wait_list is not NULL, the list of events pointed to by
 *  \a event_wait_list must be valid and \a num_events_in_wait_list must be
 *  greater than 0. The events specified in \a event_wait_list act as
 *  synchronization points.
 *
 *  \param event returns an event object that identifies this particular copy
 *  command and can be used to query or queue a wait for this particular
 *  command to complete. \a event can be NULL in which case it will not be
 *  possible for the application to query the status of this command or queue a
 *  wait for this command to complete. clEnqueueBarrier can be used instead.
 *
 *  \return CL_SUCCESS if the function is executed successfully. Otherwise it
 *  returns one of the following errors:
 *  - CL_INVALID_COMMAND_QUEUE if \a command_queue is not a valid command-queue.
 *  - CL_INVALID_CONTEXT if the context associated with \a command_queue,
 *    \a src_image and \a dst_buffer are not the same.
 *  - CL_INVALID_MEM_OBJECT if \a src_image is not a valid image object or
 *    \a dst_buffer is not a valid buffer object.
 *  - CL_INVALID_VALUE if the 2D or 3D rectangular region specified by
 *    \a src_origin and \a src_origin + \a region refers to a region outside
 *    \a src_image, or if the region specified by \a dst_offset and
 *    \a dst_offset + \a dst_cb to a region outside \a dst_buffer.
 *  - CL_INVALID_VALUE if \a src_image is a 2D image object and \a src_origin[2]
 *    is not equal to 0 or \a region[2] is not equal to 1.
 *  - CL_INVALID_EVENT_WAIT_LIST if \a event_wait_list is NULL and
 *    \a num_events_in_wait_list > 0, or \a event_wait_list is not NULL and
 *    \a num_events_in_wait_list is 0, or if event objects in \a event_wait_list
 *    are not valid events.
 *  - CL_OUT_OF_HOST_MEMORY if there is a failure to allocate resources required
 *    by the runtime.
 *
 *  \version 1.0r33
 */
RUNTIME_ENTRY(cl_int, clEnqueueCopyImageToBuffer,
              (cl_command_queue command_queue, cl_mem src_image, cl_mem dst_buffer,
               const size_t* src_origin, const size_t* region, size_t dst_offset,
               cl_uint num_events_in_wait_list, const cl_event* event_wait_list, cl_event* event)) {
  if (!is_valid(command_queue)) {
    return CL_INVALID_COMMAND_QUEUE;
  }

  if (!is_valid(src_image) || !is_valid(dst_buffer)) {
    return CL_INVALID_MEM_OBJECT;
  }

  amd::Image* srcImage = as_amd(src_image)->asImage();
  amd::Buffer* dstBuffer = as_amd(dst_buffer)->asBuffer();
  if (srcImage == NULL || dstBuffer == NULL) {
    return CL_INVALID_MEM_OBJECT;
  }

  amd::HostQueue* queue = as_amd(command_queue)->asHostQueue();
  if (NULL == queue) {
    return CL_INVALID_COMMAND_QUEUE;
  }
  amd::HostQueue& hostQueue = *queue;

  if (hostQueue.context() != srcImage->getContext() ||
      hostQueue.context() != dstBuffer->getContext()) {
    return CL_INVALID_CONTEXT;
  }

  if (srcImage->getImageFormat().image_channel_order == CL_DEPTH_STENCIL) {
    return CL_INVALID_OPERATION;
  }

  amd::Coord3D srcOrigin(src_origin[0], src_origin[1], src_origin[2]);
  amd::Coord3D dstOffset(dst_offset, 0, 0);
  amd::Coord3D srcRegion(region[0], region[1], region[2]);
  amd::Coord3D copySize(
      region[0] * region[1] * region[2] * srcImage->getImageFormat().getElementSize(), 0, 0);

  ImageViewRef mip;
  if (srcImage->getMipLevels() > 1) {
    // Create a view for the specified mip level
    mip = srcImage->createView(srcImage->getContext(), srcImage->getImageFormat(), NULL,
                               src_origin[srcImage->getDims()]);
    if (mip() == NULL) {
      return CL_OUT_OF_HOST_MEMORY;
    }
    // Reset the mip level value to 0, since a view was created
    if (srcImage->getDims() < 3) {
      srcOrigin.c[srcImage->getDims()] = 0;
    }
    srcImage = mip();
  }

  if (!srcImage->validateRegion(srcOrigin, srcRegion) ||
      !dstBuffer->validateRegion(dstOffset, copySize)) {
    return CL_INVALID_VALUE;
  }

  amd::Command::EventWaitList eventWaitList;
  cl_int err =
      amd::clSetEventWaitList(eventWaitList, hostQueue, num_events_in_wait_list, event_wait_list);
  if (err != CL_SUCCESS) {
    return err;
  }

  amd::CopyMemoryCommand* command =
      new amd::CopyMemoryCommand(hostQueue, CL_COMMAND_COPY_IMAGE_TO_BUFFER, eventWaitList,
                                 *srcImage, *dstBuffer, srcOrigin, dstOffset, srcRegion);

  if (command == NULL) {
    return CL_OUT_OF_HOST_MEMORY;
  }

  // Make sure we have memory for the command execution
  if (!command->validateMemory()) {
    delete command;
    return CL_MEM_OBJECT_ALLOCATION_FAILURE;
  }

  command->enqueue();

  *not_null(event) = as_cl(&command->event());
  if (event == NULL) {
    command->release();
  }
  return CL_SUCCESS;
}
RUNTIME_EXIT

/*! \brief Enqueue a command to copy a buffer object to an image object.
 *
 *  \param command_queue must be a valid command-queue. The OpenCL context
 *  associated with \a command_queue, \a src_buffer and \a dst_image must be
 *  the same.
 *
 *  \param src_buffer is a valid buffer object.
 *
 *  \param dst_image is a valid image object.
 *
 *  \param src_offset refers to the offset where to begin copying data in
 *  \a src_buffer.
 *
 *  \param dst_origin defines the (x, y, z) offset in the image from where to
 *  copy. If \a dst_image is a 2D image object, the z value given by
 *  \a dst_origin[2] must be 0.
 *
 *  \param region defines the (width, height, depth) of the 2D or 3D rectangle
 *  to copy. If dst_image is a 2D image object, the depth value given by
 *  \a region[2] must be 1. The size in bytes of the region to be copied from
 *  \a src_buffer referred to as \a src_cb is computed as
 *  width * height * depth * bytes/image element if \a dst_image is a 3D image
 *  object and is computed as width * height * bytes/image element if
 *  \a dst_image is a 2D image object.
 *
 *  \param num_events_in_wait_list specifies the number of event objects in
 *  \a event_wait_list.
 *
 *  \param event_wait_list specifies events that need to complete before this
 *  particular command can be executed. If \a event_wait_list is NULL, then
 *  this particular command does not wait on any event to complete. If
 *  \a event_wait_list is NULL, \a num_events_in_wait_list must be 0.
 *  If \a event_wait_list is not NULL, the list of events pointed to by
 *  \a event_wait_list must be valid and \a num_events_in_wait_list must be
 *  greater than 0. The events specified in \a event_wait_list act as
 *  synchronization points.
 *
 *  \param event returns an event object that identifies this particular copy
 *  command and can be used to query or queue a wait for this particular command
 *  to complete. \a event can be NULL in which case it will not be possible for
 *  the application to query the status of this command or queue a wait for
 *  this command to complete. clEnqueueBarrier can be used instead.
 *
 *  \return CL_SUCCESS if the function is executed successfully. Otherwise it
 *  returns one of the following errors:
 *  - CL_INVALID_COMMAND_QUEUE if \a command_queue is not a valid command-queue.
 *  - CL_INVALID_CONTEXT if the context associated with \a command_queue,
 *    \a src_buffer and \a dst_image are not the same.
 *  - CL_INVALID_MEM_OBJECT if \a src_buffer is not a valid buffer object or
 *    \a dst_image is not a valid image object.
 *  - CL_INVALID_VALUE if the 2D or 3D rectangular region specified by
 *    \a dst_origin and \a dst_origin + \a region refers to a region outside
 *    \a dst_image, or if the region specified by \a src_offset and
 *    \a src_offset + \a src_cb to a region outside \a src_buffer.
 *  - CL_INVALID_VALUE if \a dst_image is a 2D image object and \a dst_origin[2]
 *    is not equal to 0 or \a region[2] is not equal to 1.
 *  - CL_INVALID_EVENT_WAIT_LIST if \a event_wait_list is NULL and
 *    \a num_events_in_wait_list > 0, or \a event_wait_list is not NULL and
 *    \a num_events_in_wait_list is 0, or if event objects in
 *    \a event_wait_list are not valid events.
 *  - CL_OUT_OF_HOST_MEMORY if there is a failure to allocate resources required
 *    by the runtime
 *
 *  \version 1.0r33
 */
RUNTIME_ENTRY(cl_int, clEnqueueCopyBufferToImage,
              (cl_command_queue command_queue, cl_mem src_buffer, cl_mem dst_image,
               size_t src_offset, const size_t* dst_origin, const size_t* region,
               cl_uint num_events_in_wait_list, const cl_event* event_wait_list, cl_event* event)) {
  if (!is_valid(command_queue)) {
    return CL_INVALID_COMMAND_QUEUE;
  }

  if (!is_valid(src_buffer) || !is_valid(dst_image)) {
    return CL_INVALID_MEM_OBJECT;
  }
  amd::Buffer* srcBuffer = as_amd(src_buffer)->asBuffer();
  amd::Image* dstImage = as_amd(dst_image)->asImage();
  if (srcBuffer == NULL || dstImage == NULL) {
    return CL_INVALID_MEM_OBJECT;
  }

  amd::HostQueue* queue = as_amd(command_queue)->asHostQueue();
  if (NULL == queue) {
    return CL_INVALID_COMMAND_QUEUE;
  }
  amd::HostQueue& hostQueue = *queue;

  if (hostQueue.context() != srcBuffer->getContext() ||
      hostQueue.context() != dstImage->getContext()) {
    return CL_INVALID_CONTEXT;
  }

  if (dstImage->getImageFormat().image_channel_order == CL_DEPTH_STENCIL) {
    return CL_INVALID_OPERATION;
  }

  amd::Coord3D dstOrigin(dst_origin[0], dst_origin[1], dst_origin[2]);
  amd::Coord3D srcOffset(src_offset, 0, 0);
  amd::Coord3D dstRegion(region[0], region[1], region[2]);
  amd::Coord3D copySize(
      region[0] * region[1] * region[2] * dstImage->getImageFormat().getElementSize(), 0, 0);

  ImageViewRef mip;
  if (dstImage->getMipLevels() > 1) {
    // Create a view for the specified mip level
    mip = dstImage->createView(dstImage->getContext(), dstImage->getImageFormat(), NULL,
                               dst_origin[dstImage->getDims()]);
    if (mip() == NULL) {
      return CL_OUT_OF_HOST_MEMORY;
    }
    // Reset the mip level value to 0, since a view was created
    if (dstImage->getDims() < 3) {
      dstOrigin.c[dstImage->getDims()] = 0;
    }
    dstImage = mip();
  }

  if (!srcBuffer->validateRegion(srcOffset, copySize) ||
      !dstImage->validateRegion(dstOrigin, dstRegion)) {
    return CL_INVALID_VALUE;
  }

  amd::Command::EventWaitList eventWaitList;
  cl_int err =
      amd::clSetEventWaitList(eventWaitList, hostQueue, num_events_in_wait_list, event_wait_list);
  if (err != CL_SUCCESS) {
    return err;
  }

  amd::CopyMemoryCommand* command =
      new amd::CopyMemoryCommand(hostQueue, CL_COMMAND_COPY_BUFFER_TO_IMAGE, eventWaitList,
                                 *srcBuffer, *dstImage, srcOffset, dstOrigin, dstRegion);

  if (command == NULL) {
    return CL_OUT_OF_HOST_MEMORY;
  }

  // Make sure we have memory for the command execution
  if (!command->validateMemory()) {
    delete command;
    return CL_MEM_OBJECT_ALLOCATION_FAILURE;
  }

  command->enqueue();

  *not_null(event) = as_cl(&command->event());
  if (event == NULL) {
    command->release();
  }
  return CL_SUCCESS;
}
RUNTIME_EXIT

/*! @}
 *  \addtogroup CL_MapUnmap
 *  @{
 */

/*! \brief Enqueue a command to map a region of a buffer object into the
 *  host address.
 *
 *  \param command_queue must be a valid command-queue.
 *
 *  \param blocking_map indicates if the map operation is blocking or
 *  non-blocking. If \a blocking_map is CL_TRUE, clEnqueueMapBuffer does not
 *  return until the specified region in \a buffer can be mapped. If
 *  \a blocking_map is CL_FALSE i.e. map operation is non-blocking, the pointer
 *  to the mapped region returned by clEnqueueMapBuffer cannot be used until the
 *  map command has completed. The event argument returns an event object which
 *  can be used to query the execution status of the map command. When the map
 *  command is completed, the application can access the contents of the mapped
 *  region using the pointer returned by clEnqueueMapBuffer.
 *
 *  \param map_flags is a bit-field and can be set to CL_MAP_READ to indicate
 *  that the region specified by (\a offset, \a cb) in the buffer object is
 *  being mapped for reading, and/or CL_MAP_WRITE to indicate that the region
 *  specified by (\a offset, \a cb) in the buffer object is being mapped for
 *  writing.
 *
 *  \param buffer is a valid buffer object. The OpenCL context associated with
 *  \a command_queue and \a buffer must be the same.
 *
 *  \param offset is the offset in bytes of the region in the buffer object
 *  that is being mapped
 *
 *  \param cb is the size in bytes of the region in the buffer object that
 *  is being mapped.
 *
 *  \param num_events_in_wait_list specifies the number of event objects in
 *  \a event_wait_list.
 *
 *  \param event_wait_list specifies events that need to complete before this
 *  particular command can be executed. If \a event_wait_list is NULL, then
 *  this particular command does not wait on any event to complete. If
 *  \a event_wait_list is NULL, \a num_events_in_wait_list must be 0. If
 *  \a event_wait_list is not NULL, the list of events pointed to by
 *  \a event_wait_list must be valid and \a num_events_in_wait_list must be
 *  greater than 0. The events specified in \a event_wait_list act as
 *  synchronization points.
 *
 *  \param event returns an event object that identifies this particular
 *  command and can be used to query or queue a wait for this particular
 *  command to complete. \a event can be NULL in which case it will not be
 *  possible for the application to query the status of this command or queue
 *  a wait for this command to complete.
 *
 *  \param errcode_ret will return an appropriate error code. If \a errcode_ret
 *  is NULL, no error code is returned.
 *
 *  \return A pointer to the mapped region if  buffer  is  a memory object
 *  created  with  clCreateBuffer  and the region specified by (offset , cb)
 *  is a valid region in the buffer  object  and is successfully mapped into the
 *  host address space .  The  \a errcode_ret  is set to CL_SUCCESS.
 *  A NULL pointer is returned otherwise with one of the following error values
 *  returned in \a errcode_ret:
 *  - CL_INVALID_COMMAND_QUEUE if \a command_queue is not a valid command-queue.
 *  - CL_INVALID_CONTEXT if context associated with \a command_queue and
 *    \a buffer are not the same.
 *  - CL_INVALID_MEM_OBJECT if \a buffer is not a valid buffer object.
 *  - CL_INVALID_OPERATION if buffer has been created with
 *    CL_MEM_HOST_WRITE_ONLY or CL_MEM_HOST_NO_ACCESS and CL_MAP_READ
 *    is set in map_flags or if buffer has been created with
 *    CL_MEM_HOST_READ_ONLY or CL_MEM_HOST_NO_ACCESS and CL_MAP_WRITE or
 *    CL_MAP_WRITE_INVALIDATE_REGION is set in map_flags.
 *  - CL_INVALID_VALUE if region being mapped given by (\a offset, \a cb) is out
 *    of bounds or if values specified in \a map_flags are not valid.
 *  - CL_INVALID_EVENT_WAIT_LIST if \a event_wait_list is NULL and
 *    \a num_events_in_wait_list > 0, or \a event_wait_list is not NULL and
 *    \a num_events_in_wait_list is 0, or if event objects in
 *    \a event_wait_list are not valid events.
 *  - CL_MEM_O BJECT_MAP_FAILURE  if there is a failure to map  the specified
 *    region  in the host address space.
 *  - CL_OUT_OF_HOST_MEMORY if there is a failure to allocate resources required
 *    by the runtime.
 *
 *  The pointer returned maps a region starting at \a offset and is atleast
 *  \a cb bytes in size. The result of a memory access outside this region is
 *  undefined.
 *
 *  \version 1.2r07
 */
RUNTIME_ENTRY_RET(void*, clEnqueueMapBuffer,
                  (cl_command_queue command_queue, cl_mem buffer, cl_bool blocking_map,
                   cl_map_flags map_flags, size_t offset, size_t cb,
                   cl_uint num_events_in_wait_list, const cl_event* event_wait_list,
                   cl_event* event, cl_int* errcode_ret)) {
  if (!is_valid(command_queue)) {
    *not_null(errcode_ret) = CL_INVALID_COMMAND_QUEUE;
    return NULL;
  }

  if (!is_valid(buffer)) {
    *not_null(errcode_ret) = CL_INVALID_MEM_OBJECT;
    return NULL;
  }
  amd::Buffer* srcBuffer = as_amd(buffer)->asBuffer();
  if (srcBuffer == NULL) {
    *not_null(errcode_ret) = CL_INVALID_MEM_OBJECT;
    return NULL;
  }

  amd::HostQueue* queue = as_amd(command_queue)->asHostQueue();
  if (NULL == queue) {
    *not_null(errcode_ret) = CL_INVALID_COMMAND_QUEUE;
  }
  amd::HostQueue& hostQueue = *queue;

  if (hostQueue.context() != srcBuffer->getContext()) {
    *not_null(errcode_ret) = CL_INVALID_CONTEXT;
    return NULL;
  }

  if ((srcBuffer->getMemFlags() & (CL_MEM_HOST_WRITE_ONLY | CL_MEM_HOST_NO_ACCESS)) &&
      (map_flags & CL_MAP_READ)) {
    *not_null(errcode_ret) = CL_INVALID_OPERATION;
    return NULL;
  }

  if ((srcBuffer->getMemFlags() & (CL_MEM_HOST_READ_ONLY | CL_MEM_HOST_NO_ACCESS)) &&
      (map_flags & (CL_MAP_WRITE | CL_MAP_WRITE_INVALIDATE_REGION))) {
    *not_null(errcode_ret) = CL_INVALID_OPERATION;
    return NULL;
  }

  if (srcBuffer->getMemFlags() & CL_MEM_EXTERNAL_PHYSICAL_AMD) {
    *not_null(errcode_ret) = CL_INVALID_OPERATION;
    return NULL;
  }

  amd::Coord3D srcOffset(offset);
  amd::Coord3D srcSize(cb);

  if (!srcBuffer->validateRegion(srcOffset, srcSize)) {
    *not_null(errcode_ret) = CL_INVALID_VALUE;
    return NULL;
  }

  // Wait for possible pending operations
  amd::Command::EventWaitList eventWaitList;
  cl_int err =
      amd::clSetEventWaitList(eventWaitList, hostQueue, num_events_in_wait_list, event_wait_list);
  if (err != CL_SUCCESS) {
    *not_null(errcode_ret) = err;
    return (void*)0;
  }

  // Make sure we have memory for the command execution
  device::Memory* mem = srcBuffer->getDeviceMemory(hostQueue.device());
  if (NULL == mem) {
    LogPrintfError("Can't allocate memory size - 0x%08X bytes!", srcBuffer->getSize());
    *not_null(errcode_ret) = CL_MEM_OBJECT_ALLOCATION_FAILURE;
    return NULL;
  }
  // Attempt to allocate the map target now (whether blocking or non-blocking)
  void* mapPtr = mem->allocMapTarget(srcOffset, srcSize, map_flags);
  if (NULL == mapPtr) {
    *not_null(errcode_ret) = CL_MAP_FAILURE;
    return NULL;
  }

  // Allocate a map command for the queue thread
  amd::MapMemoryCommand* command = new amd::MapMemoryCommand(
      hostQueue, CL_COMMAND_MAP_BUFFER, eventWaitList, *srcBuffer, map_flags,
      blocking_map ? true : false, srcOffset, srcSize, nullptr, nullptr, mapPtr);
  if (command == NULL) {
    *not_null(errcode_ret) = CL_OUT_OF_HOST_MEMORY;
    return NULL;
  }

  // Make sure we have memory for the command execution
  if (!command->validateMemory()) {
    delete command;
    *not_null(errcode_ret) = CL_MEM_OBJECT_ALLOCATION_FAILURE;
    return NULL;
  }

  if (srcBuffer->getMemFlags() & CL_MEM_USE_PERSISTENT_MEM_AMD) {
    // [Windows VidMM restriction]
    // Runtime can't map persistent memory if it's still busy or
    // even wasn't submitted to HW from the worker thread yet
    hostQueue.finish();
  }

  // Send the map command for processing
  command->enqueue();

  // A blocking map has to wait for completion
  if (blocking_map) {
    command->awaitCompletion();
  }

  // Save the command event if applicaiton has requested it
  *not_null(event) = as_cl(&command->event());
  if (event == NULL) {
    command->release();
  }

  *not_null(errcode_ret) = CL_SUCCESS;
  srcBuffer->incMapCount();
  return mapPtr;
}
RUNTIME_EXIT

/*! \brief Enqueue a command to map a region in an image object given into
 *  the host address.
 *
 *  \param command_queue must be a valid command-queue.
 *
 *  \param image is a valid image object. The OpenCL context associated with
 *  \a command_queue and \a image must be the same.
 *
 *  \param blocking_map indicates if the map operation is blocking or
 *  non-blocking. If \a blocking_map is CL_TRUE, clEnqueueMapImage does not
 *  return until the specified region in image is mapped. If \a blocking_map is
 *  CL_FALSE i.e. map operation is non-blocking, the pointer to the mapped
 *  region returned by clEnqueueMapImage cannot be used until the map command
 *  has completed. The event argument returns an event object which can be used
 *  to query the execution status of the map command. When the map command is
 *  completed, the application can access the contents of the mapped region
 *  using the pointer returned by clEnqueueMapImage.
 *
 *  \param map_flags is a bit-field and can be set to CL_MAP_READ to indicate
 *  that the region specified by (\a origin, \a region) in the image object is
 *  being mapped for reading, and/or CL_MAP_WRITE to indicate that the region
 *  specified by (\a origin, \a region) in the image object is being mapped for
 *  writing.
 *
 *  \param origin defines the (x, y, z) offset in pixels in the image or (x, y)
 *  offset and the image index in the image array. If image is a 2D image
 *  object, origin[2] must be 0. If image is a 1D image or 1D image buffer
 *  object, origin[1] and origin[2] must be 0. If image is a 1D image array
 *  object, origin[2] must be 0. If image is a 1D image array object, origin[1]
 *  describes the image index in the 1D image array. If image is a 2D image
 *  array object, origin[2] describes the image index in the 2D image array.
 *
 *  \param region defines the (width, height, depth) in pixels of the 1D, 2D or
 *  3D rectangle or the (width, height) in pixels in pixels of the 1D or 2D
 *  rectangle and the image index of an image array. If image is a 2D image
 *  object, region[2] must be 1. If image is a 1D image or 1D image buffer
 *  object, region[1] and region[2] must be 1. If image is a 1D image array
 *  object, region[1] and region[2] must be 1. If image is a 2D image array
 *  object, region[2] must be 1.
 *
 *  \param origin define the (x, y, z) offset of the 2D or 3D rectangle region
 *  that is to be mapped. If image is a 2D image object, the z value given by
 *  \a origin[2] must be 0.
 *
 *  \param region define the (width, height, depth) of the 2D or 3D rectangle
 *  region that is to be mapped. If image is a 2D image object, the depth value
 *  given by \a region[2] must be 1.
 *
 *  \param image_row_pitch returns the scan-line pitch in bytes for the mapped
 *  region. This must be a non- NULL value.
 *
 *  \param image_slice_pitch returns the size in bytes of each 2D slice for the
 *  mapped region. For a 2D image this argument is ignored. For a 3D image this
 *  must be a non-NULL value.
 *
 *  \param num_events_in_wait_list specifies the number of event objects in
 *  \a event_wait_list.
 *
 *  \param event_wait_list specifies events that need to complete before
 *  clEnqueueMapImage can be executed. If \a event_wait_list is NULL, then
 *  clEnqueueMapImage does not wait on any event to complete. If
 *  \a event_wait_list is NULL, \a num_events_in_wait_list must be 0. If
 *  \a event_wait_list is not NULL, the list of events pointed to by
 *  \a event_wait_list must be valid and \a num_events_in_wait_list must be
 *  greater than 0. The events specified in \a event_wait_list act as
 *  synchronization points.
 *
 *  \param event returns an event object that identifies this particular command
 *  and can be used to query or queue a wait for this particular command to
 *  complete. \a event can be NULL in which case it will not be possible for the
 *  application to query the status of this command or queue a wait for this
 *  command to complete
 *
 *  \param errcode_ret will return an appropriate error code. If \a errcode_ret
 *  is NULL, no error code is returned.
 *
 *  \return A pointer to the mapped region if  image  is  a memory object
 *  created  with  clCreateImage {2D|3D},  and the 2D or 3D rectangle specified
 *  by  origin  and  region is a valid region in the image object  and can be
 *  mapped into the host address space.
 *  The \a errcode_ret is set to CL_SUCCESS. A NULL pointer is returned
 *  otherwise with one of the following error values returned in \a errcode_ret:
 *  - CL_INVALID_COMMAND_QUEUE if \a command_queue is not a valid command-queue.
 *  - CL_INVALID_CONTEXT if context associated with \a command_queue and
 *    \a image are not the same.
 *  - CL_INVALID_MEM_OBJECT if \a image is not a valid image object.
 *  - CL_INVALID_VALUE if region being mapped given by
 *    (\a origin, \a origin + \a region) is out of bounds or if values
 *    specified in \a map_flags are not valid.
 *  - CL_INVALID_VALUE if values in origin and region do not follow rules
 *    described in the argument description for origin and region.
 *  - CL_INVALID_VALUE if \a image is a 2D image object and \a origin[2] is not
 *    equal to 0 or \a region[2] is not equal to 1.
 *  - CL_INVALID_VALUE if \a image_row_pitch is NULL.
 *  - CL_INVALID_VALUE if \a image is a 3D image object and \a image_slice_pitch
 *    is NULL.
 *  - CL_INVALID_IMAGE_FORMAT if image format (image channel order and data
 *    type) for image are not supported by device associated with queue.
 *  - CL_INVALID_OPERATION if buffer has been created with
 *    CL_MEM_HOST_WRITE_ONLY or CL_MEM_HOST_NO_ACCESS and CL_MAP_READ
 *    is set in map_flags or if buffer has been created with
 *    CL_MEM_HOST_READ_ONLY or CL_MEM_HOST_NO_ACCESS and CL_MAP_WRITE or
 *    CL_MAP_WRITE_INVALIDATE_REGION is set in map_flags.
 *  - CL_INVALID_EVENT_WAIT_LIST if \a event_wait_list is NULL and
 *    \a num_events_in_wait_list > 0, or \a event_wait_list is not NULL and
 *    \a num_events_in_wait_list is 0, or if event objects in \a event_wait_list
 *    are not valid events.
 *  - CL_MEM_OBJECT_MAP_FAILURE  if there is a failure to map the  specified
 *    region in the host address space.
 *  - CL_OUT_OF_HOST_MEMORY if there is a failure to allocate resources required
 *    by the runtime.
 *
 * The pointer returned maps a 2D or 3D region starting at origin and is
 * at least (\a image_row_pitch * \a region[1] + \a region[0]) pixels in size
 * for a 2D image, and is at least (\a image_slice_pitch * \a region[2] +
 * \a image_row_pitch * \a region[1] + \a region[0]) pixels in size for a 3D
 * image. The result of a memory access outside this region is undefined.
 *
 *  \version 1.2r07
 */
RUNTIME_ENTRY_RET(void*, clEnqueueMapImage,
                  (cl_command_queue command_queue, cl_mem image, cl_bool blocking_map,
                   cl_map_flags map_flags, const size_t* origin, const size_t* region,
                   size_t* image_row_pitch, size_t* image_slice_pitch,
                   cl_uint num_events_in_wait_list, const cl_event* event_wait_list,
                   cl_event* event, cl_int* errcode_ret)) {
  if (!is_valid(command_queue)) {
    *not_null(errcode_ret) = CL_INVALID_COMMAND_QUEUE;
    return NULL;
  }

  if (!is_valid(image)) {
    *not_null(errcode_ret) = CL_INVALID_MEM_OBJECT;
    return NULL;
  }
  amd::Image* srcImage = as_amd(image)->asImage();
  if (srcImage == NULL) {
    *not_null(errcode_ret) = CL_INVALID_MEM_OBJECT;
    return NULL;
  }

  if (srcImage->getImageFormat().image_channel_order == CL_DEPTH_STENCIL) {
    *not_null(errcode_ret) = CL_INVALID_OPERATION;
    return NULL;
  }

  amd::HostQueue* queue = as_amd(command_queue)->asHostQueue();
  if (NULL == queue) {
    *not_null(errcode_ret) = CL_INVALID_COMMAND_QUEUE;
  }
  amd::HostQueue& hostQueue = *queue;

  if (hostQueue.context() != srcImage->getContext()) {
    *not_null(errcode_ret) = CL_INVALID_CONTEXT;
    return NULL;
  }

  if ((srcImage->getMemFlags() & (CL_MEM_HOST_WRITE_ONLY | CL_MEM_HOST_NO_ACCESS)) &&
      (map_flags & CL_MAP_READ)) {
    *not_null(errcode_ret) = CL_INVALID_OPERATION;
    return NULL;
  }

  if ((srcImage->getMemFlags() & (CL_MEM_HOST_READ_ONLY | CL_MEM_HOST_NO_ACCESS)) &&
      (map_flags & (CL_MAP_WRITE | CL_MAP_WRITE_INVALIDATE_REGION))) {
    *not_null(errcode_ret) = CL_INVALID_OPERATION;
    return NULL;
  }

  if ((srcImage->getDims() == 1) && ((region[1] != 1) || (region[2] != 1))) {
    *not_null(errcode_ret) = CL_INVALID_VALUE;
    return NULL;
  }

  if ((srcImage->getDims() == 2) && (region[2] != 1)) {
    *not_null(errcode_ret) = CL_INVALID_VALUE;
    return NULL;
  }

  amd::Coord3D srcOrigin(origin[0], origin[1], origin[2]);
  amd::Coord3D srcRegion(region[0], region[1], region[2]);

  ImageViewRef mip;
  if (srcImage->getMipLevels() > 1) {
    // Create a view for the specified mip level
    mip = srcImage->createView(srcImage->getContext(), srcImage->getImageFormat(), hostQueue.vdev(),
                               origin[srcImage->getDims()]);
    if (mip() == NULL) {
      *not_null(errcode_ret) = CL_OUT_OF_HOST_MEMORY;
      return NULL;
    }
    // Reset the mip level value to 0, since a view was created
    if (srcImage->getDims() < 3) {
      srcOrigin.c[srcImage->getDims()] = 0;
    }
    srcImage->incMapCount();
    srcImage = mip();
    // Retain this view until unmap is done
    srcImage->retain();
  }

  if (!srcImage->validateRegion(srcOrigin, srcRegion)) {
    *not_null(errcode_ret) = CL_INVALID_VALUE;
    return NULL;
  }

  // Wait for possible pending operations
  amd::Command::EventWaitList eventWaitList;
  cl_int err =
      amd::clSetEventWaitList(eventWaitList, hostQueue, num_events_in_wait_list, event_wait_list);
  if (err != CL_SUCCESS) {
    *not_null(errcode_ret) = err;
    return (void*)0;
  }

  // Make sure we have memory for the command execution
  device::Memory* mem = srcImage->getDeviceMemory(hostQueue.device());
  if (NULL == mem) {
    LogPrintfError("Can't allocate memory size - 0x%08X bytes!", srcImage->getSize());
    *not_null(errcode_ret) = CL_MEM_OBJECT_ALLOCATION_FAILURE;
    return NULL;
  }
  // Attempt to allocate the map target now (whether blocking or non-blocking)
  void* mapPtr =
      mem->allocMapTarget(srcOrigin, srcRegion, map_flags, image_row_pitch, image_slice_pitch);
  if (NULL == mapPtr) {
    *not_null(errcode_ret) = CL_MAP_FAILURE;
    return NULL;
  }

  // Allocate a map command for the queue thread
  amd::MapMemoryCommand* command = new amd::MapMemoryCommand(
      hostQueue, CL_COMMAND_MAP_IMAGE, eventWaitList, *srcImage, map_flags,
      blocking_map ? true : false, srcOrigin, srcRegion, nullptr, nullptr, mapPtr);
  if (command == NULL) {
    *not_null(errcode_ret) = CL_OUT_OF_HOST_MEMORY;
    return NULL;
  }

  // Make sure we have memory for the command execution
  if (!command->validateMemory()) {
    delete command;
    *not_null(errcode_ret) = CL_MEM_OBJECT_ALLOCATION_FAILURE;
    return NULL;
  }

  if (srcImage->getMemFlags() & CL_MEM_USE_PERSISTENT_MEM_AMD) {
    // [Windows VidMM restriction]
    // Runtime can't map persistent memory if it's still busy or
    // even wasn't submitted to HW from the worker thread yet
    hostQueue.finish();
  }

  // Send the map command for processing
  command->enqueue();

  // A blocking map has to wait for completion
  if (blocking_map) {
    command->awaitCompletion();
  }

  // Save the command event if applicaiton has requested it
  *not_null(event) = as_cl(&command->event());
  if (event == NULL) {
    command->release();
  }

  *not_null(errcode_ret) = CL_SUCCESS;
  srcImage->incMapCount();

  return mapPtr;
}
RUNTIME_EXIT

/*! \brief Enqueue a command to unmap a previously mapped region of a memory i
 *  object.
 *
 *  Reads or writes from the host using the pointer returned by
 *  clEnqueueMapBuffer or clEnqueueMapImage are considered to be complete.
 *
 *  \param command_queue must be a valid command-queue.
 *
 *  \param memobj is a valid memory object. The OpenCL context associated with
 *  \a command_queue and \a memobj must be the same.
 *
 *  \param mapped_ptr is the host address returned by a previous call to
 *  clEnqueueMapBuffer or clEnqueueMapImage for \a memobj.
 *
 *  \param num_events_in_wait_list specifies the number of event objects in
 *  \a event_wait_list.
 *
 *  \param event_wait_list specifies events that need to complete before
 *  clEnqueueUnmapMemObject can be executed. If \a event_wait_list is NULL,
 *  then clEnqueueUnmapMemObject does not wait on any event to complete. If
 *  \a event_wait_list is NULL, \a num_events_in_wait_list must be 0. If
 *  \a event_wait_list is not NULL, the list of events pointed to by
 *  \a event_wait_list must be valid and \a num_events_in_wait_list must be
 *  greater than 0.  The events specified in \a event_wait_list act as
 *  synchronization points.
 *
 *  \param event returns an event object that identifies this particular command
 *  and can be used to query or queue a wait for this particular command to
 *  complete. \a event can be NULL in which case it will not be possible for the
 *  application to query the status of this command or queue a wait for this
 *  command to complete. clEnqueueBarrier can be used instead.
 *
 *  \return One of the following values:
 *  - CL_SUCCESS if the function is executed successfully.
 *  - CL_INVALID_COMMAND_QUEUE if \a command_queue is not a valid command-queue
 *  - CL_INVALID_MEM_OBJECT if \a memobj is not a valid memory object.
 *  - CL_INVALID_VALUE if \a mapped_ptr is not a valid pointer returned by
 *    clEnqueueMapBuffer or clEnqueueMapImage for \a memobj.
 *  - CL_INVALID_EVENT_WAIT_LIST if \a event_wait_list is NULL and
 *    \a num_events_in_wait_list > 0, or if \a event_wait_list is not NULL and
 *    \a num_events_in_wait_list is 0, or if event objects in \a event_wait_list
 *    are not valid events.
 *  - CL_OUT_OF_HOST_MEMORY if there is a failure to allocate resources required
 *    by the runtime.
 *  - CL_INVALID_CONTEXT if context associated with \a command_queue and
 *    \a memobj are not the same.
 *
 * clEnqueueMapBuffer and clEnqueueMapImage increments the mapped count of the
 * memory object. Multiple calls to clEnqueueMapBuffer or clEnqueueMapImage on
 * the same memory object will increment this mapped count by appropriate number
 * of calls. clEnqueueUnmapMemObject decrements the mapped count of the memory
 * object. clEnqueueMapBuffer and clEnqueueMapImage act as synchronization
 * points for a region of the memory object being mapped.
 *
 *  \version 1.0r33
 */
RUNTIME_ENTRY(cl_int, clEnqueueUnmapMemObject,
              (cl_command_queue command_queue, cl_mem memobj, void* mapped_ptr,
               cl_uint num_events_in_wait_list, const cl_event* event_wait_list, cl_event* event)) {
  if (!is_valid(command_queue)) {
    return CL_INVALID_COMMAND_QUEUE;
  }

  if (!is_valid(memobj)) {
    return CL_INVALID_MEM_OBJECT;
  }

  amd::Memory* amdMemory = as_amd(memobj);

  amd::HostQueue* queue = as_amd(command_queue)->asHostQueue();
  if (NULL == queue) {
    return CL_INVALID_COMMAND_QUEUE;
  }
  amd::HostQueue& hostQueue = *queue;

  if (hostQueue.context() != amdMemory->getContext()) {
    return CL_INVALID_CONTEXT;
  }

  amd::Command::EventWaitList eventWaitList;
  cl_int err =
      amd::clSetEventWaitList(eventWaitList, hostQueue, num_events_in_wait_list, event_wait_list);
  if (err != CL_SUCCESS) {
    return err;
  }

  amd::UnmapMemoryCommand* command = new amd::UnmapMemoryCommand(
      hostQueue, CL_COMMAND_UNMAP_MEM_OBJECT, eventWaitList, *amdMemory, mapped_ptr);

  if (command == NULL) {
    return CL_OUT_OF_HOST_MEMORY;
  }

  // Make sure we have memory for the command execution
  if (!command->validateMemory()) {
    delete command;
    return CL_MEM_OBJECT_ALLOCATION_FAILURE;
  }

  device::Memory* mem = amdMemory->getDeviceMemory(hostQueue.device());
  bool blocking = false;
  if (mem->isPersistentMapped()) {
    blocking = true;
  }

  amdMemory->decMapCount();
  command->enqueue();

  if (blocking) {
    LogInfo("blocking wait in unmapping function");
    command->awaitCompletion();
  }

  *not_null(event) = as_cl(&command->event());
  if (event == NULL) {
    command->release();
  }
  return CL_SUCCESS;
}
RUNTIME_EXIT

/*! @}
 *  \addtogroup CL_MemObjQuery
 *  @{
 */

/*! \brief Get information that is common to all memory objects (buffer and
 *  image objects)
 *
 *  \param memobj specifies the memory object being queried.
 *
 *  \param param_name specifies the information to query.
 *
 *  \param param_value is a pointer to memory where the appropriate result being
 *  queried is returned. If \a param_value is NULL, it is ignored.
 *
 *  \param param_value_size is used to specify the size in bytes of memory
 *  pointed to by \a param_value. This size must be >= size of return type.
 *
 *  \param param_value_size_ret returns the actual size in bytes of data being
 *  queried by \a param_value. If \a param_value_size_ret is NULL, it is
 *  ignored.
 *
 *  \return One of the following values:
 *  - CL_SUCCESS if the function is executed successfully.
 *  - CL_INVALID_VALUE if \a param_name is not valid, or if size in bytes
 *    specified by \a param_value_size is < size of return type.
 *  - CL_INVALID_MEM_OBJECT if \a memobj is a not a valid memory object.
 *
 *  \version 1.0r33
 */
RUNTIME_ENTRY(cl_int, clGetMemObjectInfo,
              (cl_mem memobj, cl_mem_info param_name, size_t param_value_size, void* param_value,
               size_t* param_value_size_ret)) {
  if (!is_valid(memobj)) {
    return CL_INVALID_MEM_OBJECT;
  }

  switch (param_name) {
    case CL_MEM_TYPE: {
      cl_mem_object_type type = as_amd(memobj)->getType();
      return amd::clGetInfo(type, param_value_size, param_value, param_value_size_ret);
    }
    case CL_MEM_FLAGS: {
      cl_mem_flags flags = as_amd(memobj)->getMemFlags();
      return amd::clGetInfo(flags, param_value_size, param_value, param_value_size_ret);
    }
    case CL_MEM_SIZE: {
      size_t size = as_amd(memobj)->getSize();
      return amd::clGetInfo(size, param_value_size, param_value, param_value_size_ret);
    }
    case CL_MEM_HOST_PTR: {
      amd::Memory* memory = as_amd(memobj);
      const void* hostPtr =
          (memory->getMemFlags() & CL_MEM_USE_HOST_PTR) ? memory->getHostMem() : NULL;
      return amd::clGetInfo(hostPtr, param_value_size, param_value, param_value_size_ret);
    }
    case CL_MEM_MAP_COUNT: {
      cl_uint count = as_amd(memobj)->mapCount();
      return amd::clGetInfo(count, param_value_size, param_value, param_value_size_ret);
    }
    case CL_MEM_REFERENCE_COUNT: {
      cl_uint count = as_amd(memobj)->referenceCount();
      return amd::clGetInfo(count, param_value_size, param_value, param_value_size_ret);
    }
    case CL_MEM_CONTEXT: {
      cl_context context = as_cl(&as_amd(memobj)->getContext());
      return amd::clGetInfo(context, param_value_size, param_value, param_value_size_ret);
    }
    case CL_MEM_ASSOCIATED_MEMOBJECT: {
      amd::Memory* amdParent = as_amd(memobj)->parent();
      if ((NULL != amdParent) && (NULL != amdParent->getSvmPtr()) &&
          (NULL == amdParent->parent())) {
        amdParent = NULL;
      }
      cl_mem parent = as_cl(amdParent);
      return amd::clGetInfo(parent, param_value_size, param_value, param_value_size_ret);
    }
    case CL_MEM_OFFSET: {
      size_t mem_offset = as_amd(memobj)->getOrigin();
      return amd::clGetInfo(mem_offset, param_value_size, param_value, param_value_size_ret);
    }
    case CL_MEM_USES_SVM_POINTER: {
      cl_bool usesSvmPointer = as_amd(memobj)->usesSvmPointer();
      return amd::clGetInfo(usesSvmPointer, param_value_size, param_value, param_value_size_ret);
    }
#ifdef _WIN32
    case CL_MEM_D3D10_RESOURCE_KHR: {
      ID3D10Resource* pRes;

      amd::InteropObject* interop = ((amd::Memory*)as_amd(memobj))->getInteropObj();
      if (interop) {
        amd::D3D10Object* d3d10obj = interop->asD3D10Object();
        if (d3d10obj) {
          pRes = d3d10obj->getD3D10ResOrig();
          if (!pRes) {
            pRes = d3d10obj->getD3D10Resource();
          }
        }
        return amd::clGetInfo(pRes, param_value_size, param_value, param_value_size_ret);
      }
      break;
    }
    case CL_MEM_D3D11_RESOURCE_KHR: {
      ID3D11Resource* pRes;

      amd::InteropObject* interop = ((amd::Memory*)as_amd(memobj))->getInteropObj();
      if (interop) {
        amd::D3D11Object* d3d11obj = interop->asD3D11Object();
        if (d3d11obj) {
          pRes = d3d11obj->getD3D11ResOrig();
          if (!pRes) {
            pRes = d3d11obj->getD3D11Resource();
          }
        }
        return amd::clGetInfo(pRes, param_value_size, param_value, param_value_size_ret);
      }
      break;
    }
    case CL_MEM_DX9_MEDIA_SURFACE_INFO_KHR: {
      amd::InteropObject* interop = ((amd::Memory*)as_amd(memobj))->getInteropObj();
      if (interop) {
        amd::D3D9Object* d3d9obj = interop->asD3D9Object();
        if (d3d9obj)
          return amd::clGetInfo(d3d9obj->getSurfInfo(), param_value_size, param_value,
                                param_value_size_ret);
        else
          return CL_INVALID_MEM_OBJECT;
      } else
        return CL_INVALID_MEM_OBJECT;
      break;
    }
    case CL_MEM_DX9_MEDIA_ADAPTER_TYPE_KHR: {
      cl_dx9_media_adapter_type_khr adapterType;

      amd::InteropObject* interop = ((amd::Memory*)as_amd(memobj))->getInteropObj();
      if (interop) {
        amd::D3D9Object* d3d9obj = interop->asD3D9Object();
        if (d3d9obj) {
          adapterType = d3d9obj->getAdapterType();
        }
        return amd::clGetInfo(adapterType, param_value_size, param_value, param_value_size_ret);
      }
      break;
    }
#endif  //_WIN32
    default:
      break;
  }

  return CL_INVALID_VALUE;
}
RUNTIME_EXIT

/*! \brief Get information specific to an image object.
 *
 *  \param obj specifies the image object being queried.
 *
 *  \param param_name specifies the information to query.
 *
 *  \param param_value is a pointer to memory where the appropriate result being
 *  queried is returned. If \a param_value is NULL, it is ignored.
 *
 *  \param param_value_size is used to specify the size in bytes of memory
 *  pointed to by \a param_value.  This size must be >= size of return type.
 *
 *  \param param_value_size_ret returns the actual size in bytes of data being
 *  queried by \a param_value. If \a param_value_size_ret is NULL, it is
 *  ignored.
 *
 *  \return One of the following values:
 *  - CL_SUCCESS if the function is executed successfully
 *  - CL_INVALID_VALUE if \a param_name is not valid, or if size in bytes
 *    specified by \a param_value_size is < size of return type and
 *    \a param_value is not NULL.
 *  - CL_INVALID_MEM_OBJECT if \a image is a not a valid image object.
 *
 *  \version 1.2r09
 */
RUNTIME_ENTRY(cl_int, clGetImageInfo,
              (cl_mem memobj, cl_image_info param_name, size_t param_value_size, void* param_value,
               size_t* param_value_size_ret)) {
  if (!is_valid(memobj)) {
    return CL_INVALID_MEM_OBJECT;
  }
  amd::Image* image = as_amd(memobj)->asImage();
  if (image == NULL) {
    return CL_INVALID_MEM_OBJECT;
  }

  switch (param_name) {
    case CL_IMAGE_FORMAT: {
      cl_image_format format = image->getImageFormat();
      return amd::clGetInfo(format, param_value_size, param_value, param_value_size_ret);
    }
    case CL_IMAGE_ELEMENT_SIZE: {
      size_t elementSize = image->getImageFormat().getElementSize();
      return amd::clGetInfo(elementSize, param_value_size, param_value, param_value_size_ret);
    }
    case CL_IMAGE_ROW_PITCH: {
      size_t rowPitch = image->getRowPitch();
      return amd::clGetInfo(rowPitch, param_value_size, param_value, param_value_size_ret);
    }
    case CL_IMAGE_SLICE_PITCH: {
      size_t slicePitch = image->getSlicePitch();
      return amd::clGetInfo(slicePitch, param_value_size, param_value, param_value_size_ret);
    }
    case CL_IMAGE_WIDTH: {
      size_t width = image->getWidth();
      return amd::clGetInfo(width, param_value_size, param_value, param_value_size_ret);
    }
    case CL_IMAGE_HEIGHT: {
      size_t height = image->getHeight();
      if ((image->getType() == CL_MEM_OBJECT_IMAGE1D) ||
          (image->getType() == CL_MEM_OBJECT_IMAGE1D_ARRAY) ||
          (image->getType() == CL_MEM_OBJECT_IMAGE1D_BUFFER)) {
        height = 0;
      }
      return amd::clGetInfo(height, param_value_size, param_value, param_value_size_ret);
    }
    case CL_IMAGE_DEPTH: {
      size_t depth = image->getDepth();
      if ((image->getType() == CL_MEM_OBJECT_IMAGE1D_BUFFER) ||
          (image->getType() == CL_MEM_OBJECT_IMAGE1D_ARRAY) ||
          (image->getType() == CL_MEM_OBJECT_IMAGE2D_ARRAY) ||
          (image->getType() == CL_MEM_OBJECT_IMAGE1D) ||
          (image->getType() == CL_MEM_OBJECT_IMAGE2D)) {
        depth = 0;
      }
      return amd::clGetInfo(depth, param_value_size, param_value, param_value_size_ret);
    }
    case CL_IMAGE_ARRAY_SIZE: {
      size_t arraySize = 0;
      if (image->getType() == CL_MEM_OBJECT_IMAGE1D_ARRAY) {
        arraySize = image->getHeight();
      } else if (image->getType() == CL_MEM_OBJECT_IMAGE2D_ARRAY) {
        arraySize = image->getDepth();
      }
      return amd::clGetInfo(arraySize, param_value_size, param_value, param_value_size_ret);
    }
    case CL_IMAGE_BUFFER: {
      cl_mem buffer = 0;
      amd::Memory* parent = image->parent();
      while (parent && (parent->asBuffer() == NULL)) {
        parent = parent->parent();
      }
      buffer = as_cl(parent);
      return amd::clGetInfo(buffer, param_value_size, param_value, param_value_size_ret);
    }
    case CL_IMAGE_NUM_MIP_LEVELS: {
      cl_uint numMipLevels = image->getMipLevels();
      return amd::clGetInfo(numMipLevels, param_value_size, param_value, param_value_size_ret);
    }
    case CL_IMAGE_NUM_SAMPLES: {
      cl_uint numSamples = 0;
      return amd::clGetInfo(numSamples, param_value_size, param_value, param_value_size_ret);
    }
    case CL_IMAGE_BYTE_PITCH_AMD: {
      size_t bytePitch = image->getBytePitch();
      return amd::clGetInfo(bytePitch, param_value_size, param_value, param_value_size_ret);
    }
#ifdef _WIN32
    case CL_IMAGE_D3D10_SUBRESOURCE_KHR: {
      amd::InteropObject* interop = ((amd::Memory*)as_amd(memobj))->getInteropObj();
      if (!interop) {
        return CL_INVALID_MEM_OBJECT;
      }
      amd::D3D10Object* d3d10obj = interop->asD3D10Object();
      if (!d3d10obj) {
        return CL_INVALID_MEM_OBJECT;
      }
      UINT subresource = d3d10obj->getSubresource();
      return amd::clGetInfo(subresource, param_value_size, param_value, param_value_size_ret);
    }
    case CL_IMAGE_D3D11_SUBRESOURCE_KHR: {
      amd::InteropObject* interop = ((amd::Memory*)as_amd(memobj))->getInteropObj();
      if (!interop) {
        return CL_INVALID_MEM_OBJECT;
      }
      amd::D3D11Object* d3d11obj = interop->asD3D11Object();
      if (!d3d11obj) {
        return CL_INVALID_MEM_OBJECT;
      }
      UINT subresource = d3d11obj->getSubresource();
      return amd::clGetInfo(subresource, param_value_size, param_value, param_value_size_ret);
    }
    case CL_MEM_DX9_MEDIA_SURFACE_INFO_KHR: {
      amd::InteropObject* interop = ((amd::Memory*)as_amd(memobj))->getInteropObj();
      if (!interop) {
        return CL_INVALID_MEM_OBJECT;
      }
      amd::D3D9Object* d3d9obj = interop->asD3D9Object();
      if (!d3d9obj) {
        return CL_INVALID_MEM_OBJECT;
      }
      return amd::clGetInfo(d3d9obj->getSurfInfo(), param_value_size, param_value,
                            param_value_size_ret);
    }
    case CL_IMAGE_DX9_MEDIA_PLANE_KHR: {
      amd::InteropObject* interop = ((amd::Memory*)as_amd(memobj))->getInteropObj();
      if (!interop) {
        return CL_INVALID_MEM_OBJECT;
      }
      amd::D3D9Object* d3d9obj = interop->asD3D9Object();
      if (!d3d9obj) {
        return CL_INVALID_MEM_OBJECT;
      }
      cl_uint plane = d3d9obj->getPlane();
      return amd::clGetInfo(plane, param_value_size, param_value, param_value_size_ret);
    }
#endif  //_WIN32
    default:
      break;
  }
  return CL_INVALID_VALUE;
}
RUNTIME_EXIT

/*! \brief creates a 1D image, 1D image buffer, 1D image array, 2D image,
 *  2D image array and 3D image object
 *
 *  \param context is a valid OpenCL context on which the image object is
 *  to be created.
 *
 *  \param flags is a bit-field that is used to specify allocation and usage
 *  information about the image memory object being created and is described
 *  in table 5.3. If value specified for flags is 0, the default is used which
 *  is CL_MEM_READ_WRITE.
 *
 *  \param image_format is a pointer to a structure that describes format
 *  properties of the image to be allocated. Refer to section 5.3.1.1 for
 *  a detailed description of the image format descriptor.
 *
 *  \param image_desc is a pointer to a structure that describes type and
 *  dimensions of the image to be allocated. Refer to section 5.3.1.2 for
 *  a detailed description of the image descriptor.
 *
 *  \param host_ptr is a pointer to the image data that may already be
 *  allocated by the application. Refer to table below for a description of
 *  how large the buffer that host_ptr points to must be.
 *      CL_MEM_OBJECT_IMAGE1D >= image_row_pitch
 *      CL_MEM_OBJECT_IMAGE1D_BUFFER >= image_row_pitch
 *      CL_MEM_OBJECT_IMAGE2D >= image_row_pitch * image_height
 *      CL_MEM_OBJECT_IMAGE3D >= image_slice_pitch * image_depth
 *      CL_MEM_OBJECT_IMAGE1D_ARRAY >= image_slice_pitch * image_array_size
 *      CL_MEM_OBJECT_IMAGE2D_ARRAY >= image_slice_pitch * image_array_size
 *  For a 3D image or 2D image array, the image data specified by \a host_ptr
 *  is stored as a linear sequence of adjacent 2D image slices or 2D images
 *  respectively. Each 2D image is a linear sequence of adjacent scanlines.
 *  Each scanline is a linear sequence of image elements.
 *  For a 2D image array, the image data specified by \a host_ptr is stored
 *  as a linear sequence of adjacent scanlines. Each scanline is a linear
 *  sequence of image elements.
 *  For a 1D image array, the image data specified by \a host_ptr is stored
 *  as a linear sequence of adjacent 1D images respectively. Each 1D image
 *  or 1D image buffer is a single scanline which is a linear sequence of
 *  adjacent elements.
 *
 *  \param errcode_ret will return an appropriate error code.
 *  If \a errcode_ret is NULL, no error code is returned.
 *
 *  \return a valid non-zero image object created and the \a errcode_ret is
 *  set to CL_SUCCESS if the image object is created successfully. Otherwise,
 *  it returns a NULL value with one of the following error values
 *  returned in \a errcode_ret:
 *  - CL_INVALID_CONTEXT if \a context is not a valid context.
 *  - CL_INVALID_VALUE if values specified in \a flags are not valid.
 *  - CL_INVALID_IMAGE_FORMAT_DESCRIPTOR if values specified in \a image_format
 *    are not valid or if \a image_format is NULL.
 *  - CL_INVALID_IMAGE_DESCRIPTOR if values specified in \a image_desc are
 *    not valid or if \a image_desc is NULL.
 *  - CL_INVALID_HOST_PTR if \a host_ptr in \a image_desc is NULL and
 *    CL_MEM_USE_HOST_PTR or CL_MEM_COPY_HOST_PTR are set in \a flags or
 *    if \a host_ptr is not NULL, but CL_MEM_COPY_HOST_PTR or
 *    CL_MEM_USE_HOST_PTR are not set in \a flags.
 *  - CL_INVALID_VALUE if a 1D image buffer is being created and
 *    the buffer object was created with CL_MEM_WRITE_ONLY and \a flags
 *    specifies CL_MEM_READ_WRITE or CL_MEM_READ_ONLY, or if the buffer object
 *    was created with CL_MEM_READ_ONLY and \a flags specifies
 *    CL_MEM_READ_WRITE or CL_MEM_WRITE_ONLY, or if \a flags specifies
 *    CL_MEM_USE_HOST_PTR or CL_MEM_ALLOC_HOST_PTR or CL_MEM_COPY_HOST_PTR.
 *  - CL_IMAGE_FORMAT_NOT_SUPPORTED if the image_format is not supported.
 *  - CL_MEM_OBJECT_ALLOCATION_FAILURE if there is a failure to allocate memory
 *    for image object.
 *  - CL_INVALID_OPERATION if there are no devices in \a context that support
 *    images
 *  - CL_DEVICE_IMAGE_SUPPORT specified in table 4.3 is CL_FALSE).
 *  - CL_OUT_OF_RESOURCES if there is a failure to allocate resources required
 *    by the OpenCL implementation on the device.
 *  - CL_OUT_OF_HOST_MEMORY if there is a failure to allocate resources required
 *    by the OpenCL implementation on the host.
 *
 *  \version 1.2r07
 */
RUNTIME_ENTRY_RET(cl_mem, clCreateImage,
                  (cl_context context, cl_mem_flags flags, const cl_image_format* image_format,
                   const cl_image_desc* image_desc, void* host_ptr, cl_int* errcode_ret)) {
  if (!is_valid(context)) {
    *not_null(errcode_ret) = CL_INVALID_CONTEXT;
    LogWarning("invalid parameter: context");
    return (cl_mem)0;
  }
  // check flags for validity
  if (!validateFlags(flags)) {
    *not_null(errcode_ret) = CL_INVALID_VALUE;
    LogWarning("invalid parameter: flags");
    return (cl_mem)0;
  }
  // check format
  if (image_format == NULL) {
    *not_null(errcode_ret) = CL_INVALID_IMAGE_FORMAT_DESCRIPTOR;
    LogWarning("invalid parameter: image_format");
    return (cl_mem)0;
  }

  const amd::Image::Format imageFormat(*image_format);
  if (!imageFormat.isValid()) {
    *not_null(errcode_ret) = CL_INVALID_IMAGE_FORMAT_DESCRIPTOR;
    LogWarning("invalid parameter: image_format");
    return (cl_mem)0;
  }

  amd::Context& amdContext = *as_amd(context);

  if (!imageFormat.isSupported(amdContext, image_desc->image_type)) {
    *not_null(errcode_ret) = CL_IMAGE_FORMAT_NOT_SUPPORTED;
    LogWarning("invalid parameter: image_format");
    return (cl_mem)0;
  }

  // check host_ptr consistency
  if (host_ptr == NULL) {
    if (flags & (CL_MEM_USE_HOST_PTR | CL_MEM_COPY_HOST_PTR)) {
      *not_null(errcode_ret) = CL_INVALID_HOST_PTR;
      LogWarning("invalid parameter: host_ptr");
      return (cl_mem)0;
    }
  } else {
    if (!(flags & (CL_MEM_USE_HOST_PTR | CL_MEM_COPY_HOST_PTR))) {
      *not_null(errcode_ret) = CL_INVALID_HOST_PTR;
      LogWarning("invalid parameter: host_ptr");
      return (cl_mem)0;
    }
  }

  const std::vector<amd::Device*>& devices = as_amd(context)->devices();
  bool supportPass = false;
  for (auto& dev : devices) {
    if (dev->info().imageSupport_) {
      supportPass = true;
      break;
    }
  }

  if (!supportPass) {
    *not_null(errcode_ret) = CL_INVALID_OPERATION;
    LogWarning("there are no devices in context to support images");
    return (cl_mem)0;
  }

  if (!amd::Image::validateDimensions(devices, image_desc->image_type, image_desc->image_width,
                                      image_desc->image_height, image_desc->image_depth,
                                      image_desc->image_array_size)) {
    *not_null(errcode_ret) = CL_INVALID_IMAGE_SIZE;
    LogWarning("invalid parameter: image dimensions exceeding max");
    return (cl_mem)0;
  }

  size_t imageRowPitch = 0;
  size_t imageSlicePitch = 0;
  if (!validateImageDescriptor(devices, imageFormat, image_desc, host_ptr, imageRowPitch,
                               imageSlicePitch)) {
    *not_null(errcode_ret) = CL_INVALID_IMAGE_DESCRIPTOR;
    LogWarning("invalid parameter: image_desc");
    return (cl_mem)0;
  }

  // Validate mip level
  if (image_desc->num_mip_levels != 0) {
    size_t maxDim = std::max(image_desc->image_width, image_desc->image_height);
    maxDim = std::max(maxDim, image_desc->image_depth);
    uint mipLevels;
    for (mipLevels = 0; maxDim > 0; maxDim >>= 1, mipLevels++);
    if (mipLevels < image_desc->num_mip_levels) {
      *not_null(errcode_ret) = CL_INVALID_MIP_LEVEL;
      LogWarning("Invalid mip level");
      return (cl_mem)0;
    }
  }
  amd::Image* image = NULL;

  switch (image_desc->image_type) {
    case CL_MEM_OBJECT_IMAGE1D:
      image = new (amdContext)
          amd::Image(amdContext, CL_MEM_OBJECT_IMAGE1D, flags, imageFormat, image_desc->image_width,
                     1, 1, imageRowPitch, 0, image_desc->num_mip_levels);
      break;
    case CL_MEM_OBJECT_IMAGE2D:
      if (image_desc->mem_object != NULL) {
        amd::Buffer& buffer = *(as_amd(image_desc->mem_object)->asBuffer());
        if (&amdContext != &buffer.getContext()) {
          *not_null(errcode_ret) = CL_INVALID_CONTEXT;
          LogWarning("invalid parameter: context");
          return (cl_mem)0;
        }

        // host_ptr is not supported, the buffer object is used instead.
        if ((flags & (CL_MEM_USE_HOST_PTR | CL_MEM_ALLOC_HOST_PTR | CL_MEM_COPY_HOST_PTR)) != 0) {
          *not_null(errcode_ret) = CL_INVALID_VALUE;
          LogWarning("invalid parameter: flags");
          return (cl_mem)0;
        }

        cl_uint pitchAlignment = 0;
        for (unsigned int i = 0; i < devices.size(); ++i) {
          if (pitchAlignment < devices[i]->info().imagePitchAlignment_) {
            pitchAlignment = devices[i]->info().imagePitchAlignment_;
          }
        }
        if ((imageRowPitch % pitchAlignment) != 0) {
          *not_null(errcode_ret) = CL_INVALID_IMAGE_FORMAT_DESCRIPTOR;
          LogWarning("invalid parameter: flags");
          return (cl_mem)0;
        }

        image = new (amdContext) amd::Image(
            buffer, CL_MEM_OBJECT_IMAGE2D, (flags != 0) ? flags : buffer.getMemFlags(), imageFormat,
            image_desc->image_width, image_desc->image_height, 1, imageRowPitch, imageSlicePitch);
      } else {
        image = new (amdContext) amd::Image(amdContext, CL_MEM_OBJECT_IMAGE2D, flags, imageFormat,
                                            image_desc->image_width, image_desc->image_height, 1,
                                            imageRowPitch, 0, image_desc->num_mip_levels);
      }
      break;
    case CL_MEM_OBJECT_IMAGE3D:
      image = new (amdContext)
          amd::Image(amdContext, CL_MEM_OBJECT_IMAGE3D, flags, imageFormat, image_desc->image_width,
                     image_desc->image_height, image_desc->image_depth, imageRowPitch,
                     imageSlicePitch, image_desc->num_mip_levels);
      break;
    case CL_MEM_OBJECT_IMAGE1D_BUFFER: {
      amd::Buffer& buffer = *(as_amd(image_desc->mem_object)->asBuffer());
      if (&amdContext != &buffer.getContext()) {
        *not_null(errcode_ret) = CL_INVALID_CONTEXT;
        LogWarning("invalid parameter: context");
        return (cl_mem)0;
      }

      // host_ptr is not supported, the buffer object is used instead.
      if ((flags & (CL_MEM_USE_HOST_PTR | CL_MEM_ALLOC_HOST_PTR | CL_MEM_COPY_HOST_PTR)) != 0) {
        *not_null(errcode_ret) = CL_INVALID_VALUE;
        LogWarning("invalid parameter: flags");
        return (cl_mem)0;
      }

      // Mip-mapped images can not be created for IMAGE1D_BUFFER
      if (image_desc->num_mip_levels != 0) {
        *not_null(errcode_ret) = CL_INVALID_MIP_LEVEL;
        LogWarning("Invalid mip level");
        return (cl_mem)0;
      }

      image = new (amdContext) amd::Image(buffer, CL_MEM_OBJECT_IMAGE1D_BUFFER,
                                          (flags != 0) ? flags : buffer.getMemFlags(), imageFormat,
                                          image_desc->image_width, 1, 1, imageRowPitch,
                                          imageSlicePitch, image_desc->num_mip_levels);
    } break;
    case CL_MEM_OBJECT_IMAGE1D_ARRAY:
      image =
          new (amdContext) amd::Image(amdContext, CL_MEM_OBJECT_IMAGE1D_ARRAY, flags, imageFormat,
                                      image_desc->image_width, image_desc->image_array_size, 1,
                                      imageRowPitch, imageSlicePitch, image_desc->num_mip_levels);
      break;
    case CL_MEM_OBJECT_IMAGE2D_ARRAY:
      image = new (amdContext) amd::Image(
          amdContext, CL_MEM_OBJECT_IMAGE2D_ARRAY, flags, imageFormat, image_desc->image_width,
          image_desc->image_height, image_desc->image_array_size, imageRowPitch, imageSlicePitch,
          image_desc->num_mip_levels);
      break;
    default: {
      *not_null(errcode_ret) = CL_INVALID_IMAGE_DESCRIPTOR;
      LogWarning("invalid parameter: image_desc");
      return reinterpret_cast<cl_mem>(image);
    } break;
  }

  if (image == NULL) {
    *not_null(errcode_ret) = CL_OUT_OF_HOST_MEMORY;
    LogWarning("cannot allocate resources");
    return (cl_mem)0;
  }

  if (!image->create(host_ptr)) {
    *not_null(errcode_ret) = CL_MEM_OBJECT_ALLOCATION_FAILURE;
    image->release();
    return (cl_mem)0;
  }

  *not_null(errcode_ret) = CL_SUCCESS;
  return (cl_mem)as_cl<amd::Memory>(image);
}
RUNTIME_EXIT

/*! \brief Enqueues a command to fill a buffer object with
 *  a pattern of a given pattern size.
 *
 *  \param command_queue refers to the command-queue in which
 *  the fill command will be queued. The OpenCL context associated with
 *  command_queue and buffer must be the same.
 *
 *  \param buffer is a valid buffer object.
 *
 *  \param pattern is a pointer to the data pattern of size pattern_size
 *  in bytes. pattern will be used to fill a region in buffer starting
 *  at offset and is cb bytes in size. The data pattern must be a scalar or
 *  vector integer or floating-point data type supported by OpenCL
 *  as described in sections 6.1.1 and 6.1.2. For example, if buffer is
 *  to be filled with a pattern of float4 values, then pattern will be
 *  a pointer to a cl_float4 value and pattern_size will be sizeof(cl_float4).
 *  The maximum value of pattern_size is the size of the largest integer or
 *  floating-point vector data type supported by the OpenCL device.
 *
 *  \param offset is the location in bytes of the region being filled
 *  in buffer and must be a multiple of pattern_size. size is the size
 *  in bytes of region being filled in buffer and must be a multiple
 *  of pattern_size.
 *
 *  \param num_events_in_wait_list specifies the number of event objects in
 *  \a event_wait_list.
 *
 *  \param event_wait_list specifes events that need to complete before this
 *  particular command can be executed. If \a event_wait_list is NULL,
 *  then this particular command does not wait on any event to complete.
 *  If \a event_wait_list is NULL, \a num_events_in_wait_list must be 0.
 *  If \a event_wait_list is not NULL, the list of events pointed to by
 *  \a event_wait_list must be valid and a\ num_events_in_wait_list must be
 *  greater than 0. The events specified in \a event_wait_list act as
 *  synchronization points. The context associated with events in
 *  \a event_wait_list and \a command_queue must be the same.
 *  The memory associated with \a event_wait_list can be reused or
 *  freed after the function returns.
 *
 *  \param event returns an event object that identifies this particular command
 *  and can be used to query or queue a wait for this particular command to
 *  complete. \a event can be NULL in which case it will not be possible for the
 *  application to query the status of this command or queue a wait for this
 *  command to complete. clEnqueueBarrierWithWaitList can be used instead.
 *
 *  \return One of the following values:
 *  - CL_SUCCESS if the function is executed successfully.
 *  - CL_INVALID_CONTEXT if context associated with \a command_queue and
 *    \a buffer are not the same or if the \a context associated with
 *    \a command_queue and \a events in \a event_wait_list are not the same.
 *  - CL_INVALID_COMMAND_QUEUE if \a command_queue is not a valid command-queue
 *  - CL_INVALID_MEM_OBJECT if \a memobj is not a valid memory object.
 *  - CL_INVALID_VALUE if pattern is NULL or if pattern_size is 0 or if
 *    \a pattern_size is one of {1, 2, 4, 8, 16, 32, 64, 128}.
 *  - CL_INVALID_VALUE if \a offset or \a offset + \a size require accessing
 *    elements outside the \a buffer object respectively.
 *  - CL_INVALID_EVENT_WAIT_LIST if \a event_wait_list is NULL and
 *    \a num_events_in_wait_list > 0, or if \a event_wait_list is not NULL and
 *    \a num_events_in_wait_list is 0, or if event objects in \a event_wait_list
 *    are not valid events.
 *  - CL_OUT_OF_RESOURCES if there is a failure to allocate resources required
 *    by the OpenCL implementation on the device.
 *  - CL_OUT_OF_HOST_MEMORY if there is a failure to allocate resources
 *    required by the OpenCL implementation on the host.
 *
 *  \version 1.2r07
 */
RUNTIME_ENTRY(cl_int, clEnqueueFillBuffer,
              (cl_command_queue command_queue, cl_mem buffer, const void* pattern,
               size_t pattern_size, size_t offset, size_t size, cl_uint num_events_in_wait_list,
               const cl_event* event_wait_list, cl_event* event)) {
  amd::Buffer* fillBuffer;

  if (!is_valid(command_queue)) {
    return CL_INVALID_COMMAND_QUEUE;
  }

  if (!is_valid(buffer)) {
    return CL_INVALID_MEM_OBJECT;
  }

  fillBuffer = as_amd(buffer)->asBuffer();
  if (fillBuffer == NULL) {
    return CL_INVALID_MEM_OBJECT;
  }

  if ((pattern == NULL) || (pattern_size == 0) ||
      (pattern_size > amd::FillMemoryCommand::MaxFillPatterSize) ||
      ((pattern_size & (pattern_size - 1)) != 0)) {
    return CL_INVALID_VALUE;
  }

  // Offset and size must be multiple of pattern_size
  if (!(amd::isMultipleOf(offset, pattern_size) && amd::isMultipleOf(size, pattern_size))) {
    return CL_INVALID_VALUE;
  }

  amd::HostQueue* queue = as_amd(command_queue)->asHostQueue();
  if (NULL == queue) {
    return CL_INVALID_COMMAND_QUEUE;
  }
  amd::HostQueue& hostQueue = *queue;

  if (hostQueue.context() != fillBuffer->getContext()) {
    return CL_INVALID_CONTEXT;
  }

  amd::Coord3D fillOffset(offset, 0, 0);
  amd::Coord3D fillSize(size, 1, 1);
  // surface takes [pitch, width, height]
  amd::Coord3D surface(size, size, 1);
  if (!fillBuffer->validateRegion(fillOffset, fillSize)) {
    return CL_INVALID_VALUE;
  }

  amd::Command::EventWaitList eventWaitList;
  cl_int err =
      amd::clSetEventWaitList(eventWaitList, hostQueue, num_events_in_wait_list, event_wait_list);
  if (err != CL_SUCCESS) {
    return err;
  }

  amd::FillMemoryCommand* command =
      new amd::FillMemoryCommand(hostQueue, CL_COMMAND_FILL_BUFFER, eventWaitList, *fillBuffer,
                                 pattern, pattern_size, fillOffset, fillSize, surface);

  if (command == NULL) {
    return CL_OUT_OF_HOST_MEMORY;
  }

  // Make sure we have memory for the command execution
  if (!command->validateMemory()) {
    delete command;
    return CL_MEM_OBJECT_ALLOCATION_FAILURE;
  }

  command->enqueue();

  *not_null(event) = as_cl(&command->event());
  if (event == NULL) {
    command->release();
  }

  return CL_SUCCESS;
}
RUNTIME_EXIT

/*! \brief enqueues a command to fill an image object with
 *  a specified color.
 *
 *  \param command_queue refers to the command-queue in which
 *  the fill command will be queued. The OpenCL context associated with
 *  command_queue and buffer must be the same.
 *
 *  \param buffer is a valid buffer object.
 *
 *  \param fill_color is the fill color. The fill color is a four
 *  component RGBA floating-point color value if the image channel data type
 *  is not an unnormalized signed and unsigned integer type, is a four
 *  component signed integer value if the image channel data type is
 *  an unnormalized signed integer type and is a four component unsigned
 *  integer value if the image channel data type is an unormalized
 *  unsigned integer type. The fill color will be converted to
 *  the appropriate image channel format and order associated with image
 *  as described in sections 6.11.13 and 8.3.
 *
 *  \param origin defines the (x, y, z) offset in pixels in the image
 *  or (x, y) offset and the image index in the image array. If image is
 *  a 2D image object, origin[2] must be 0. If image is a 1D image or 1D
 *  image buffer object, origin[1] and origin[2] must be 0. If image is
 *  a 1D image array object, origin[2] must be 0. If image is a 1D image array
 *  object, origin[1] describes the image index in the 1D image array.
 *  If image is a 2D image array object, origin[2] describes the image index
 *  in the 2D image array.
 *
 *  \param region defines the (width, height, depth) in pixels of
 *  the 1D, 2D or 3D rectangle or the (width, height) in pixels in pixels of
 *  the 1D or 2D rectangle and the image index of an image array. If image is
 *  a 2D image object, region[2] must be 1. If image is a 1D image or
 *  1D image buffer object, region[1] and region[2] must be 1. If image is
 *  a 1D image array object, region[1] and region[2] must be 1.
 *  If image is a 2D image array object, region[2] must be 1.
 *
 *  \param num_events_in_wait_list specifies the number of event objects in
 *  \a event_wait_list.
 *
 *  \param event_wait_list specifes events that need to complete before this
 *  particular command can be executed. If \a event_wait_list is NULL,
 *  then this particular command does not wait on any event to complete.
 *  If \a event_wait_list is NULL, \a num_events_in_wait_list must be 0.
 *  If \a event_wait_list is not NULL, the list of events pointed to by
 *  \a event_wait_list must be valid and a\ num_events_in_wait_list must be
 *  greater than 0. The events specified in \a event_wait_list act as
 *  synchronization points. The context associated with events in
 *  \a event_wait_list and \a command_queue must be the same.
 *  The memory associated with \a event_wait_list can be reused or
 *  freed after the function returns.
 *
 *  \param event returns an event object that identifies this particular command
 *  and can be used to query or queue a wait for this particular command to
 *  complete. \a event can be NULL in which case it will not be possible for
 *  the application to query the status of this command or queue a wait for this
 *  command to complete. clEnqueueBarrierWithWaitList can be used instead.
 *
 *  \return One of the following values:
 *  - CL_SUCCESS if the function is executed successfully.
 *  - CL_INVALID_CONTEXT if context associated with \a command_queue and
 *    \a buffer are not the same or if the \a context associated with
 *    \a command_queue and \a events in \a event_wait_list are not the same.
 *  - CL_INVALID_COMMAND_QUEUE if \a command_queue is not a valid command-queue
 *  - CL_INVALID_MEM_OBJECT if \a memobj is not a valid memory object.
 *  - CL_INVALID_VALUE if fill_color is NULL.
 *  - CL_INVALID_VALUE if the region being filled as specified by origin and
 *    region is out of bounds.
 *  - CL_INVALID_VALUE if values in origin and region do not follow rules
 *    described in the argument description for origin and region.
 *  - CL_INVALID_EVENT_WAIT_LIST if \a event_wait_list is NULL and
 *    \a num_events_in_wait_list > 0, or if \a event_wait_list is not NULL and
 *    \a num_events_in_wait_list is 0, or if event objects in \a event_wait_list
 *    are not valid events.
 *  - CL_INVALID_IMAGE_SIZE if image dimensions (image width, height, specified
 *    or compute row
 *  - CL_INVALID_IMAGE_FORMAT if image format (image channel order and data type)
 *    for image are not supported by device associated with queue.
 *  - CL_OUT_OF_RESOURCES if there is a failure to allocate resources required
 *    by the OpenCL implementation on the device.
 *  - CL_OUT_OF_HOST_MEMORY if there is a failure to allocate resources
 *    required by the OpenCL implementation on the host.
 *
 *  \version 1.2r07
 */
RUNTIME_ENTRY(cl_int, clEnqueueFillImage,
              (cl_command_queue command_queue, cl_mem image, const void* fill_color,
               const size_t* origin, const size_t* region, cl_uint num_events_in_wait_list,
               const cl_event* event_wait_list, cl_event* event)) {
  amd::Image* fillImage;

  if (!is_valid(command_queue)) {
    return CL_INVALID_COMMAND_QUEUE;
  }

  if (!is_valid(image)) {
    return CL_INVALID_MEM_OBJECT;
  }

  if (fill_color == NULL) {
    return CL_INVALID_VALUE;
  }

  fillImage = as_amd(image)->asImage();
  if (fillImage == NULL) {
    return CL_INVALID_MEM_OBJECT;
  }

  amd::HostQueue* queue = as_amd(command_queue)->asHostQueue();
  if (NULL == queue) {
    return CL_INVALID_COMMAND_QUEUE;
  }
  amd::HostQueue& hostQueue = *queue;

  if (hostQueue.context() != fillImage->getContext()) {
    return CL_INVALID_CONTEXT;
  }

  if (fillImage->getImageFormat().image_channel_order == CL_DEPTH_STENCIL) {
    return CL_INVALID_OPERATION;
  }

  amd::Coord3D fillOrigin(origin[0], origin[1], origin[2]);
  amd::Coord3D fillRegion(region[0], region[1], region[2]);
  // surface takes [pitch, width, height]
  amd::Coord3D surface(region[0], region[0], region[2]);

  ImageViewRef mip;
  if (fillImage->getMipLevels() > 1) {
    // Create a view for the specified mip level
    mip = fillImage->createView(fillImage->getContext(), fillImage->getImageFormat(), nullptr,
                                origin[fillImage->getDims()]);
    if (mip() == nullptr) {
      return CL_OUT_OF_HOST_MEMORY;
    }
    // Reset the mip level value to 0, since a view was created
    if (fillImage->getDims() < 3) {
      fillOrigin.c[fillImage->getDims()] = 0;
    }
    fillImage = mip();
  }

  if (!fillImage->validateRegion(fillOrigin, fillRegion)) {
    return CL_INVALID_VALUE;
  }

  amd::Command::EventWaitList eventWaitList;
  cl_int err =
      amd::clSetEventWaitList(eventWaitList, hostQueue, num_events_in_wait_list, event_wait_list);
  if (err != CL_SUCCESS) {
    return err;
  }

  amd::FillMemoryCommand* command = new amd::FillMemoryCommand(
      hostQueue, CL_COMMAND_FILL_IMAGE, eventWaitList, *fillImage, fill_color,
      sizeof(cl_float4),  // @note color size is always 16 bytes value
      fillOrigin, fillRegion, surface);

  if (command == NULL) {
    return CL_OUT_OF_HOST_MEMORY;
  }

  // Make sure we have memory for the command execution
  if (!command->validateMemory()) {
    delete command;
    return CL_MEM_OBJECT_ALLOCATION_FAILURE;
  }

  command->enqueue();

  *not_null(event) = as_cl(&command->event());
  if (event == NULL) {
    command->release();
  }

  return CL_SUCCESS;
}
RUNTIME_EXIT

/*! \brief Enqueues a command to indicate which device a set of memory objects
 *  should be associated with. Typically, memory objects are implicitly
 *  migrated to a device for which enqueued commands, using the memory object,
 *  are targeted. \a clEnqueueMigrateMemObjects allows this migration to be
 *  explicitly performed ahead of the dependent commands. This allows a user to
 *  preemptively change the association of a memory object, through regular
 *  command queue scheduling, in order to prepare for another upcoming
 *  command. This also permits an application to overlap the placement of
 *  memory objects with other unrelated operations before these memory objects
 *  are needed potentially hiding transfer latencies. Once the event, returned
 *  from \a clEnqueueMigrateMemObjects, has been marked \a CL_COMPLETE
 *  the memory objects specified in \a mem_objects have been successfully
 *  migrated to the device associated with \a command_queue. The migrated memory
 *  object shall remain resident on the device until another command is enqueued
 *  that either implicitly or explicitly migrates it away.
 *  \a clEnqueueMigrateMemObjects can also be used to direct the initial
 *  placement of a memory object, after creation, possibly avoiding the initial
 *  overhead of instantiating the object on the first enqueued command to use it.
 *  The user is responsible for managing the event dependencies, associated with
 *  this command, in order to avoid overlapping access to memory objects.
 *  Improperly specified event dependencies passed to
 *  \a clEnqueueMigrateMemObjects could result in undefined results.
 *
 *  \param command_queue is a valid command-queue. The specified set of memory
 *  objects in \a mem_objects will be migrated to the OpenCL device associated
 *  with \a command_queue or to the host if the \a CL_MIGRATE_MEM_OBJECT_HOST
 *  has been specified.
 *
 *  \param num_mem_objects is the number of memory objects specified in
 *  \a mem_objects. \a mem_objects is a pointer to a list of memory objects.
 *
 *  \param flags is a bit-field that is used to specify migration options.
 *  The following table describes the possible values for flags.
 *  cl_mem_migration flags      Description
 *  CL_MIGRATE_MEM_OBJECT_HOST  This flag indicates that the specified set
 *                              of memory objects are to be migrated to the
 *                              host, regardless of the target command-queue.
 *  CL_MIGRATE_MEM_OBJECT_      This flag indicates that the contents of the set
 *  CONTENT_UNDEFINED           of memory objects are undefined after migration.
 *                              The specified set of memory objects are migrated
 *                              to the device associated with \a command_queue
 *                              without incurring
 *
 *  \param num_events_in_wait_list specifies the number of event objects in
 *  \a event_wait_list.
 *
 *  \param event_wait_list specifes events that need to complete before this
 *  particular command can be executed. If \a event_wait_list is NULL,
 *  then this particular command does not wait on any event to complete.
 *  If \a event_wait_list is NULL, \a num_events_in_wait_list must be 0.
 *  If \a event_wait_list is not NULL, the list of events pointed to by
 *  \a event_wait_list must be valid and a\ num_events_in_wait_list must be
 *  greater than 0. The events specified in \a event_wait_list act as
 *  synchronization points. The context associated with events in
 *  \a event_wait_list and \a command_queue must be the same.
 *  The memory associated with \a event_wait_list can be reused or
 *  freed after the function returns.
 *
 *  \return One of the following values:
 *  - CL_SUCCESS if the function is executed successfully.
 *  - CL_INVALID_COMMAND_QUEUE if \a command_queue is not a valid command-queue
 *  - CL_INVALID_CONTEXT if the context associated with \a command_queue
 *    and memory objects in \a mem_objects are not the same or if the context
 *    associated with \a command_queue and events in \a event_wait_list
 *    are not the same.
 *  - CL_INVALID_MEM_OBJECT if any of the memory objects in \a mem_objects
 *    is not a valid memory object.
 *  - CL_INVALID_VALUE if \a num_mem_objects is zero or
 *    if \a mem_objects is NULL.
 *  - CL_INVALID_VALUE if flags is not 0 or any of the values described
 *    in the table above
 *  - CL_INVALID_EVENT_WAIT_LIST if \a event_wait_list is NULL and
 *    \a num_events_in_wait_list > 0, or if \a event_wait_list is not NULL and
 *    \a num_events_in_wait_list is 0, or if event objects in \a event_wait_list
 *    are not valid events.
 *  - CL_MEM_OBJECT_ALLOCATION_FAILURE if there is a failure to allocate
 *    memory for the specified set of memory objects in \a mem_objects.
 *  - CL_OUT_OF_RESOURCES if there is a failure to allocate resources required
 *    by the OpenCL implementation on the device.
 *  - CL_OUT_OF_HOST_MEMORY if there is a failure to allocate resources
 *    required by the OpenCL implementation on the host.
 *
 *  \version 1.2r15
 */
RUNTIME_ENTRY(cl_int, clEnqueueMigrateMemObjects,
              (cl_command_queue command_queue, cl_uint num_mem_objects, const cl_mem* mem_objects,
               cl_mem_migration_flags flags, cl_uint num_events_in_wait_list,
               const cl_event* event_wait_list, cl_event* event)) {
  if (!is_valid(command_queue)) {
    return CL_INVALID_COMMAND_QUEUE;
  }

  amd::HostQueue* queue = as_amd(command_queue)->asHostQueue();
  if (NULL == queue) {
    return CL_INVALID_COMMAND_QUEUE;
  }
  amd::HostQueue& hostQueue = *queue;

  if ((num_mem_objects == 0) || (mem_objects == NULL)) {
    return CL_INVALID_VALUE;
  }

  if (flags & ~(CL_MIGRATE_MEM_OBJECT_HOST | CL_MIGRATE_MEM_OBJECT_CONTENT_UNDEFINED)) {
    return CL_INVALID_VALUE;
  }

  std::vector<amd::Memory*> memObjects;
  for (uint i = 0; i < num_mem_objects; ++i) {
    if (!is_valid(mem_objects[i])) {
      return CL_INVALID_MEM_OBJECT;
    }
    amd::Memory* memory = as_amd(mem_objects[i]);
    if (hostQueue.context() != memory->getContext()) {
      return CL_INVALID_CONTEXT;
    }
    memObjects.push_back(memory);
  }

  amd::Command::EventWaitList eventWaitList;
  cl_int err =
      amd::clSetEventWaitList(eventWaitList, hostQueue, num_events_in_wait_list, event_wait_list);
  if (err != CL_SUCCESS) {
    return err;
  }

  amd::MigrateMemObjectsCommand* command = new amd::MigrateMemObjectsCommand(
      hostQueue, CL_COMMAND_MIGRATE_MEM_OBJECTS, eventWaitList, memObjects, flags);

  if (command == NULL) {
    return CL_OUT_OF_HOST_MEMORY;
  }

  // Make sure we have memory for the command execution
  if (!command->validateMemory()) {
    delete command;
    return CL_MEM_OBJECT_ALLOCATION_FAILURE;
  }

  command->enqueue();

  *not_null(event) = as_cl(&command->event());
  if (event == NULL) {
    command->release();
  }

  return CL_SUCCESS;
}
RUNTIME_EXIT

RUNTIME_ENTRY_RET(cl_mem, clConvertImageAMD,
                  (cl_context context, cl_mem image, const cl_image_format* image_format,
                   cl_int* errcode_ret)) {
  if (!is_valid(context)) {
    *not_null(errcode_ret) = CL_INVALID_CONTEXT;
    LogWarning("invalid parameter: context");
    return (cl_mem)0;
  }
  // check format
  if (image_format == NULL) {
    *not_null(errcode_ret) = CL_INVALID_IMAGE_FORMAT_DESCRIPTOR;
    LogWarning("invalid parameter: image_format");
    return (cl_mem)0;
  }
  const amd::Image::Format imageFormat(*image_format);
  if (!imageFormat.isValid()) {
    *not_null(errcode_ret) = CL_INVALID_IMAGE_FORMAT_DESCRIPTOR;
    LogWarning("invalid parameter: image_format");
    return (cl_mem)0;
  }

  amd::Context& amdContext = *as_amd(context);
  if (!imageFormat.isSupported(amdContext)) {
    *not_null(errcode_ret) = CL_IMAGE_FORMAT_NOT_SUPPORTED;
    LogWarning("invalid parameter: image_format");
    return (cl_mem)0;
  }
  amd::Image* amdImage = as_amd(image)->asImage();
  amd::Image* converted_image = amdImage->createView(amdContext, imageFormat, NULL);

  if (converted_image == NULL) {
    *not_null(errcode_ret) = CL_INVALID_IMAGE_FORMAT_DESCRIPTOR;
    LogWarning("cannot allocate resources");
    return (cl_mem)0;
  }

  *not_null(errcode_ret) = CL_SUCCESS;
  return (cl_mem)as_cl<amd::Memory>(converted_image);
}
RUNTIME_EXIT

RUNTIME_ENTRY_RET(cl_mem, clCreateBufferFromImageAMD,
                  (cl_context context, cl_mem image, cl_int* errcode_ret)) {
  if (!is_valid(context)) {
    *not_null(errcode_ret) = CL_INVALID_CONTEXT;
    LogWarning("invalid parameter: context");
    return (cl_mem)0;
  }

  amd::Context& amdContext = *as_amd(context);
  const std::vector<amd::Device*>& devices = amdContext.devices();
  bool supportPass = false;
  for (auto& dev : devices) {
    if (dev->info().bufferFromImageSupport_) {
      supportPass = true;
      break;
    }
  }

  if (!supportPass) {
    *not_null(errcode_ret) = CL_INVALID_OPERATION;
    LogWarning("there are no devices in context to support buffer from image");
    return (cl_mem)0;
  }

  amd::Image* amdImage = as_amd(image)->asImage();
  if (!is_valid(image) || amdImage == NULL) {
    *not_null(errcode_ret) = CL_INVALID_MEM_OBJECT;
    return NULL;
  }

  amd::Memory* mem = new (amdContext) amd::Buffer(*amdImage, 0, 0, amdImage->getSize());
  if (mem == NULL) {
    *not_null(errcode_ret) = CL_OUT_OF_HOST_MEMORY;
    return (cl_mem)0;
  }

  if (!mem->create()) {
    *not_null(errcode_ret) = CL_MEM_OBJECT_ALLOCATION_FAILURE;
    mem->release();
    return NULL;
  }

  *not_null(errcode_ret) = CL_SUCCESS;
  return (cl_mem)as_cl<amd::Memory>(mem);
}
RUNTIME_EXIT

/*! @}
 *  @}
 *  @}
 */
