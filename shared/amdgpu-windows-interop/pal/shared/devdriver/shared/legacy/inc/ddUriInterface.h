/*
 ***********************************************************************************************************************
 *
 *  Copyright (c) 2021-2025 Advanced Micro Devices, Inc. All Rights Reserved.
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

#pragma once

#include "gpuopen.h"
#include "ddPlatform.h"

namespace DevDriver
{
    namespace TransferProtocol
    {
        class ServerBlock;
    }

    // The maximum allowed name for a service name
    DD_STATIC_CONST size_t kMaxUriServiceNameLength = 128;

    enum struct URIDataFormat : uint32
    {
        Unknown = 0,
        Text,
        Binary,
        Count
    };

    // An interface to write bytes.
    class IByteWriter
    {
    protected:
        virtual ~IByteWriter() {}

    public:
        // Finish all writing and return the last error.
        virtual Result End() = 0;

        // Write exactly `length` bytes.
        virtual void WriteBytes(const void* pBytes, size_t length) = 0;

        // Write a value as a byte array.
        // N.B.: Be mindful of your struct's implicit padding!
        template <typename T>
        void Write(const T& value)
        {
            static_assert(!Platform::IsPointer<T>::Value, "Writing a pointer is likely an error. Cast to an integer type if you mean it.");
            WriteBytes(&value, sizeof(value));
        }
    };

    // An interface to write and validate text.
    class ITextWriter
    {
    protected:
        virtual ~ITextWriter() {}

    public:
        // Finish all writing and return the last error.
        virtual Result End() = 0;

        // Write formatted text.
        // Try and only pass string literals as `pFmt`. Prefer: Write("%s", myGeneratedBuffer);
        virtual void Write(const char* pFmt, ...) = 0;

        // Write specific types
        virtual void Write(uint64 value) = 0;
        virtual void Write(uint32 value) = 0;
        virtual void Write(uint16 value) = 0;
        virtual void Write(uint8  value) = 0;
        virtual void Write(int64  value) = 0;
        virtual void Write(int32  value) = 0;
        virtual void Write(int16  value) = 0;
        virtual void Write(double value) = 0;
        virtual void Write(float  value) = 0;
        virtual void Write(bool   value) = 0;
        virtual void Write(char   value) = 0;
    };

    // An interface to write and validate structured data - e.g. json or message pack
    class IStructuredWriter
    {
    protected:
        virtual ~IStructuredWriter() {}

    public:
        // Finish all writing and return the last error.
        virtual Result End() = 0;

        // Structured data is often nullable.
        // Write a "null" value.
        virtual void ValueNull() = 0;

        // ===== Collection Writers ====================================================================================

        // Begin writing a new list collection.
        virtual void BeginList() = 0;

        // End the current list collection.
        virtual void EndList() = 0;

        // Begin writing a new map collection.
        virtual void BeginMap() = 0;

        // End the current map collection.
        virtual void EndMap() = 0;

        // Write a key into a map.
        virtual void Key(const char* pKey) = 0;

        // ===== Value Writers =========================================================================================

        virtual void Value(const char* pValue) = 0;
        virtual void Value(const char* pValue, size_t length) = 0;

        virtual void Value(uint64 value) = 0;
        virtual void Value(uint32 value) = 0;
        virtual void Value(uint16 value) = 0;
        virtual void Value(uint8  value) = 0;
        virtual void Value(int64  value) = 0;
        virtual void Value(int32  value) = 0;
        virtual void Value(int16  value) = 0;
        virtual void Value(int8   value) = 0;
        virtual void Value(double value) = 0;
        virtual void Value(float  value) = 0;
        virtual void Value(bool   value) = 0;
        virtual void Value(char   value) = 0;

        /// Writes an enum value as a String or hex value
        /// If DevDriver::ToString(Enum) returns NULL or an empty string, it will hex-encode the integer value.
        /// Otherwise, it will write that string
        template <typename Enum>
        void ValueEnumOrHex(Enum value)
        {
            const char* pString = ToString(value);
            if ((pString == nullptr) || (strcmp(pString, "") != 0))
            {
                Value(pString);
            }
            else
            {
                Valuef("0x%x", value);
            }
        }

        // Write a formatted string
        template <typename... Args>
        void Valuef(const char* pFmt, Args&&... args)
        {
            char buffer[1024];
            Platform::Snprintf(buffer, pFmt, args...);
            Value(buffer);
        }

        // ===== Key + Value Writers ===================================================================================

        // Write a key-value pair where the value will be a list.
        void KeyAndBeginList(const char* pKey) { Key(pKey); BeginList(); }

        // Write a key-value pair where the value will be a map.
        void KeyAndBeginMap(const char* pKey)  { Key(pKey); BeginMap(); }

        // Write a key-value pair.
        void KeyAndValue(const char* pKey, const char* pValue)                { Key(pKey); Value(pValue); }
        void KeyAndValue(const char* pKey, const char* pValue, size_t length) { Key(pKey); Value(pValue, length); }
        void KeyAndValue(const char* pKey, uint64      value)                 { Key(pKey); Value(value); }
        void KeyAndValue(const char* pKey, uint32      value)                 { Key(pKey); Value(value); }
        void KeyAndValue(const char* pKey, int64       value)                 { Key(pKey); Value(value); }
        void KeyAndValue(const char* pKey, int32       value)                 { Key(pKey); Value(value); }
        void KeyAndValue(const char* pKey, double      value)                 { Key(pKey); Value(value); }
        void KeyAndValue(const char* pKey, float       value)                 { Key(pKey); Value(value); }
        void KeyAndValue(const char* pKey, bool        value)                 { Key(pKey); Value(value); }

        template <typename Enum>
        void KeyAndValueEnumOrHex(const char* pKey, Enum value) { Key(pKey); ValueEnumOrHex(value); }

        // Write a key-value pair where the value will be a "null" value.
        void KeyAndValueNull(const char* pKey) { Key(pKey); ValueNull(); }

        // Write a key-value pair with a formatted value
        template <typename... Args>
        void KeyAndValuef(const char* pKey, const char* pFmt, Args&&... args) { Key(pKey); Valuef(pFmt, args...); }
    };

    // An aggregate of the POST metadata for a request.
    struct PostDataInfo
    {
        const void*   pData;  // Immutable view of the post data
        uint32        size;   // Size of the post data in bytes
        URIDataFormat format; // Format of the post data - i.e. how to read it

        // Zero initialize the struct.
        PostDataInfo()
        {
            memset(this, 0, sizeof(*this));
        }
    };

    // An interface that represents a unique URI request
    class IURIRequestContext
    {
    protected:
        virtual ~IURIRequestContext() {}

    public:
        // Retrieve the request argument string
        // N.B: This is non-const and designed to be mutated
        virtual char* GetRequestArguments() = 0;

        // Retrieve information about the post data of this request
        virtual const PostDataInfo& GetPostData() const = 0;

        // Creates and returns a Writer to copy bytes into the response block.
        // Only a single writer is allowed per request context.
        // Returns:
        //    - Result::Rejected if any writer of any type has already been returned
        //    - Result::Error if `ppWriter` is `nullptr`
        virtual Result BeginByteResponse(IByteWriter** ppWriter) = 0;

        // Creates and returns a Writer to copy text into the response block.
        // Only a single writer is allowed per request context.
        // Returns:
        //    - Result::Rejected if any writer of any type has already been returned
        //    - Result::Error if `ppWriter` is `nullptr`
        virtual Result BeginTextResponse(ITextWriter** ppWriter) = 0;

        // Creates and returns a Writer to copy json into the response block.
        // Only a single writer is allowed per request context.
        // Returns:
        //    - Result::Rejected if any writer of any type has already been returned
        //    - Result::Error if `ppWriter` is `nullptr`
        virtual Result BeginJsonResponse(IStructuredWriter** ppWriter) = 0;
    };

    struct URIResponseHeader
    {
        // The size of the response data in bytes
        size_t responseDataSizeInBytes;

        // The format of the response data
        URIDataFormat responseDataFormat;
    };

    // Base class for URI services
    class IService
    {
    public:
        virtual ~IService() {}

        // Returns the name of the service
        virtual const char* GetName() const = 0;

        // Returns the service version
        virtual Version GetVersion() const = 0;

        // Attempts to handle a request from a client
        virtual Result HandleRequest(IURIRequestContext* pContext) = 0;

        // Determines the size limit for post data requests for the client request.  By default services
        // will not accept any post data.  The pArguments paramter must remain non-const because the
        // service may need to manipulate it for further processing.
        virtual size_t QueryPostSizeLimit(char* pArguments) const
        {
            DD_UNUSED(pArguments);
            return 0;
        }

    protected:
        IService() {};
    };
} // DevDriver
