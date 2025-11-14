/*
 ***********************************************************************************************************************
 *
 *  Copyright (c) 2014-2025 Advanced Micro Devices, Inc. All Rights Reserved.
 *
 *  Permission is hereby granted, free of charge, to any person obtaining a copy
 *  of this software and associated documentation files (the "Software"), to deal
 *  in the Software without restriction, including without limitation the rights
 *  to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 *  copies of the Software, and to permit persons to whom the Software is
 *  furnished to do so, subject to the following conditions:
 *
 *  The above copyright notice and this permission notice shall be included in all
 *  copies or substantial portions of the Software.
 *
 *  THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 *  IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 *  FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 *  AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 *  LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 *  OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 *  SOFTWARE.
 *
 **********************************************************************************************************************/
/**
 ***********************************************************************************************************************
 * @file  palGpuUtil.h
 * @brief Common include for the PAL GPU utility collection.  Defines common types, macros, enums, etc.
 ***********************************************************************************************************************
 */

#pragma once

#include "pal.h"

// Forward declarations.
namespace Pal
{
    struct DeviceProperties;
    class  IImage;
    class  IGpuMemory;
    struct ImageCopyRegion;
    struct TypedBufferCopyRegion;
    struct MemoryImageCopyRegion;
}

/// Library-wide namespace encapsulating all PAL GPU utility entities.
namespace GpuUtil
{

/// Validate image copy region.
///
/// @param [in] properties  The device properties.
/// @param [in] engineType  Engine to validate.
/// @param [in] src         Src image.
/// @param [in] dst         Des image.
/// @param [in] region      Copy region.
///
/// @returns true if the image copy is supported by the specific engine, otherwise false.
extern bool ValidateImageCopyRegion(
    const Pal::DeviceProperties& properties,
    Pal::EngineType              engineType,
    const Pal::IImage&           src,
    const Pal::IImage&           dst,
    const Pal::ImageCopyRegion&  region);

/// Validate typed buffer copy region.
///
/// @param [in] properties  The device properties.
/// @param [in] engineType  Engine to validate.
/// @param [in] region      Copy region.
///
/// @returns true if the typed buffer copy is supported by the specific engine, otherwise false.
extern bool ValidateTypedBufferCopyRegion(
    const Pal::DeviceProperties&      properties,
    Pal::EngineType                   engineType,
    const Pal::TypedBufferCopyRegion& region);

/// Validate image-memory copy region.
///
/// @param [in] properties  The device properties.
/// @param [in] engineType  Engine to validate.
/// @param [in] image       The IImage object.
/// @param [in] region      Copy region.
///
/// @returns true if the image-memory copy is supported by the specific engine, otherwise false.
extern bool ValidateMemoryImageRegion(
    const Pal::DeviceProperties&      properties,
    Pal::EngineType                   engineType,
    const Pal::IImage&                image,
    const Pal::IGpuMemory&            memory,
    const Pal::MemoryImageCopyRegion& region);

/// Generate a 64-bit uniqueId for a GPU memory allocation
///
/// @param [in] isInterprocess  Indicates this uniqueId is for an externally shareable GPU memory allocation
///
/// @returns 64-bit uniqueId
extern Pal::uint64 GenerateGpuMemoryUniqueId(
    bool isInterprocess);

} // GpuUtil

/**
 ***********************************************************************************************************************
 * @page GpuUtilOverview GPU Utility Collection
 *
 * In addition to the generic, OS-abstracted software utilities, PAL provides GPU-specific utilities in the @ref GpuUtil
 * namespace. The PAL GPU Utility Collection relies on both PAL core and PAL Utility. They are also available for use by
 * its clients.
 *
 * All available PAL GPU utilities are defined in the @ref GpuUtil namespace, and are briefly summarized below.  See the
 * Reference topics for more detailed information on specific classes, enums, etc.
 *
 * ### TextWriter
 * The TextWriter GPU utility class provides a method for clients to write text directly to an image. This can be used
 * for debugging purposes. PAL's internal DbgOverlay uses the TextWriter class to write information about the current
 * FPS and total allocated GPU video memory usage.
 *
 * The TextWriter class is broken up into palTextWriter.h and palTextWriterImpl.h. The intention is that palTextWriter.h
 * will be included from other header files that need a full TextWriter definition, while palTextWriterImpl.h will be
 * included by .cpp files that actually interact with the TextWriter. This should keep build times down versus putting
 * all implementations directly in palTextWriter.h.
 *
 * Also included in the TextWriter is the TextWriterFont namespace, which defines the shader IL for drawing the text via
 * a compute shader. It also defines the Font data, which is a packed binary that represents which pixels of a 10x16
 * rectangle to render. The font is monospaced.
 *
 * ### Helper Functions
 * ValidateImageCopyRegion - Validate the image copy region, returns true if the image copy is supported by the specific
 * engine, otherwise false.
 *
 * ValidateTypedBufferCopyRegion - Validate the typed buffer copy region, returns true if the typed buffer copy is
 * supported by the specific engine, otherwise false.
 *
 * ValidateMemoryImageRegion - Validate the image-memory copy region, returns true if the image-memory copy is supported
 * by the specific engine, otherwise false.
 *
 * Next: @ref Overview
 ***********************************************************************************************************************
 */
