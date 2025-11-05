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
 * @file  palDestroyable.h
 * @brief Defines the Platform Abstraction Library (PAL) IDestroyable interface.
 ***********************************************************************************************************************
 */

#pragma once

namespace Pal
{

/**
 ***********************************************************************************************************************
 * @interface IDestroyable
 * @brief     Interface inherited by objects that must be explicitly destroyed by the client.
 *
 * This includes all objects except:
 *
 * + @ref IColorTargetView, @ref IDepthStencilView - These classes are treated as SRDs by the DX12 runtime.  Therefore,
 *   PAL guarantees that no action needs to be taken at Destroy() - the client should just free the memory backing these
 *   classes.
 * + @ref IDevice - These objects are created during IPlatform::EnumerateDevices() and are automatically destroyed
 *   along with the Platform object.
 * + @ref IPrivateScreen - These objects are created as during IPlatform::EnumerateDevices() based on
 *   which screens are attached to each device.  They are automatically destroyed along with the Platform object.
 ***********************************************************************************************************************
 */
class IDestroyable
{
public:
    /// Frees all resources associated with this object.
    ///
    /// It is the client's responsibility to only call this method once there are no more existing references to this
    /// object.  This method does not free the system memory associated with the object (as specified in pPlacementAddr
    /// during creation); the client is responsible for freeing that memory since they allocated it.
    virtual void Destroy() = 0;

protected:
    /// @internal Destructor.  Prevent use of delete operator on this interface.  Client must destroy objects by
    /// explicitly calling IDestroyable::Destroy() and is responsible for freeing the system memory allocated for the
    /// object on their own.
    virtual ~IDestroyable() { }
};

} // Pal
