/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

#ifndef HIP_INCLUDE_AMD_HIP_GL_INTEROP_H
#define HIP_INCLUDE_AMD_HIP_GL_INTEROP_H

#if defined(__cplusplus)
extern "C" {
#endif

/**
 *
 * @addtogroup GlobalDefs
 * @{
 *
 */

/**
 * HIP Devices used by current OpenGL Context.
 */
typedef enum hipGLDeviceList {
  hipGLDeviceListAll = 1,           ///< All hip devices used by current OpenGL context.
  hipGLDeviceListCurrentFrame = 2,  ///< Hip devices used by current OpenGL context in current
                                    ///< frame
  hipGLDeviceListNextFrame = 3      ///< Hip devices used by current OpenGL context in next
                                    ///< frame.
} hipGLDeviceList;


/** GLuint as uint.*/
typedef unsigned int GLuint;
/** GLenum as uint.*/
typedef unsigned int GLenum;
/**
 * @}
 */

/**
 * @defgroup GL OpenGL Interoperability
 * @ingroup API
 * @{
 * This section describes OpenGL interoperability functions of HIP runtime API.
 */

/**
 * @brief Queries devices associated with the current OpenGL context.
 *
 * @param [out] pHipDeviceCount - Pointer of number of devices on the current GL context.
 * @param [out] pHipDevices - Pointer of devices on the current OpenGL context.
 * @param [in] hipDeviceCount - Size of device.
 * @param [in] deviceList - The setting of devices. It could be either hipGLDeviceListCurrentFrame
 * for the devices used to render the current frame, or hipGLDeviceListAll for all devices.
 * The default setting is Invalid deviceList value.
 *
 * @returns #hipSuccess, #hipErrorInvalidValue, #hipErrorNotSupported
 *
 */
hipError_t hipGLGetDevices(unsigned int* pHipDeviceCount, int* pHipDevices,
                           unsigned int hipDeviceCount, hipGLDeviceList deviceList);
/**
 * @brief Registers a GL Buffer for interop and returns corresponding graphics resource.
 *
 * @param [out] resource - Returns pointer of graphics resource.
 * @param [in] buffer - Buffer to be registered.
 * @param [in] flags - Register OpenGL flags.
 *
 * @returns #hipSuccess, #hipErrorInvalidValue, #hipErrorUnknown, #hipErrorInvalidResourceHandle
 *
 */
hipError_t hipGraphicsGLRegisterBuffer(hipGraphicsResource** resource, GLuint buffer,
                                       unsigned int flags);
/**
 * @brief Register a GL Image for interop and returns the corresponding graphic resource.
 *
 * @param [out] resource - Returns pointer of graphics resource.
 * @param [in] image - Image to be registered.
 * @param [in] target - Valid target value Id.
 * @param [in] flags - Register OpenGL flags.
 *
 * @returns #hipSuccess, #hipErrorInvalidValue, #hipErrorUnknown, #hipErrorInvalidResourceHandle
 *
 */
hipError_t hipGraphicsGLRegisterImage(hipGraphicsResource** resource, GLuint image, GLenum target,
                                      unsigned int flags);
/**
 * @}
 */
#if defined(__cplusplus)
}
#endif /* __cplusplus */
#endif /* HIP_INCLUDE_AMD_HIP_GL_INTEROP_H */
