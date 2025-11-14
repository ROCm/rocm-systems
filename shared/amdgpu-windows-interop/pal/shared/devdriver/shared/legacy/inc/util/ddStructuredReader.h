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

#include <ddPlatform.h>

namespace DevDriver
{
    // The value half of a key-value pair from a StructuredReader.
    // This always wraps a valid IValue pointer, but the value semantically stored may be empty. (e.g. a Json null)
    class StructuredValue
    {
    public:
        // In order to avoid leaking internal headers, we treat this member as an opaque data type.
        // Its size and alignment is checked in the cpp file.
        /// This is an internal type, only exposed due to limitations in C++ semantics.
        struct OpaqueNode
        {
            void* blob[2] = {};
        };

        ~StructuredValue() = default;

        StructuredValue()
        : m_opaque()
        {}

        explicit StructuredValue(OpaqueNode opaque)
        : m_opaque(opaque)
        {}

        StructuredValue(StructuredValue&& other)      = default;
        StructuredValue(const StructuredValue& other) = default;

        StructuredValue& operator=(StructuredValue&& other)      = default;
        StructuredValue& operator=(const StructuredValue& other) = default;

        enum class Type
        {
            Null = 0,
            Array,
            Map,
            Str,
            Bool,
            Int,
            Uint,
            Double,
            Float,
        };

        // Type of data contained in this node.
        Type GetType() const;

        const char* GetTypeString() const
        {
            switch (GetType())
            {
                case StructuredValue::Type::Null:   return "Null";

                case StructuredValue::Type::Array:  return "Array";
                case StructuredValue::Type::Map:    return "Map";
                case StructuredValue::Type::Str:    return "Str";

                case StructuredValue::Type::Bool:   return "Bool";

                case StructuredValue::Type::Int:    return "Int";
                case StructuredValue::Type::Uint:   return "Uint";

                case StructuredValue::Type::Double: return "Double";
                case StructuredValue::Type::Float:  return "Float";
                default:
                    DD_WARN_ALWAYS();
                    return "Unknown";
            }
        }

        // Create a new empty value
        StructuredValue MakeNull() const;

        // Return whether this is an empty, or "null" node.
        bool IsNull() const;

        /// ===== Unsigned Integer Types

        /// Returns true when this node contains a Uint8. If pValue is not NULL, copy out the node's value
        DD_NODISCARD bool GetUint8(uint8*   pValue) const;
        // Returns true when this node contains a Uint16. If pValue is not NULL, copy out the node's value
        DD_NODISCARD bool GetUint16(uint16* pValue) const;
        // Returns true when this node contains a Uint32. If pValue is not NULL, copy out the node's value
        DD_NODISCARD bool GetUint32(uint32* pValue) const;
        // Returns true when this node contains a Uint64. If pValue is not NULL, copy out the node's value
        DD_NODISCARD bool GetUint64(uint64* pValue) const;

        /// ===== Signed Integer Types

        /// Returns true when this node contains a Int8. If pValue is not NULL, copy out the node's value
        DD_NODISCARD bool GetInt8(int8*     pValue) const;
        // Returns true when this node contains a Int16. If pValue is not NULL, copy out the node's value
        DD_NODISCARD bool GetInt16(int16*   pValue) const;
        // Returns true when this node contains a Int32. If pValue is not NULL, copy out the node's value
        DD_NODISCARD bool GetInt32(int32*   pValue) const;
        // Returns true when this node contains a Int64. If pValue is not NULL, copy out the node's value
        DD_NODISCARD bool GetInt64(int64*   pValue) const;

        /// ===== Floating Point Types

        /// Returns true when this node contains a Float. If pValue is not NULL, copy out the node's value
        DD_NODISCARD bool GetFloat(float*   pValue) const;
        // Returns true when this node contains a Double. If pValue is not NULL, copy out the node's value
        DD_NODISCARD bool GetDouble(double* pValue) const;

        /// ===== Other Types

        // Returns true when this node contains a Bool. If pValue is not NULL, copy out the node's value
        DD_NODISCARD bool GetBool(bool*     pValue) const;

        // Copy a string value from a node into a buffer
        //  If the StructuredValue is not a string,
        //      false is returned and no writes occur
        //
        //  If pStringSize is not NULL,
        //      the string length is written and processing continues
        //
        //  If pBuffer is not NULL,
        //      not more than bufferSize bytes (including a NULL terminator) are written.
        //      If the buffer is large enough to hold the entire string,
        //          true is returned
        //
        // If both pBuffer and pStringSize are NULL and the value *is* a string,
        //      true is returned
        //
        // TODO: ... this is complicated. Should we use a Result?
        //       The other types are simple enough that they benefit from using bool instead of a Result, but Strings may not.
        DD_NODISCARD bool GetStringCopy(char* pBuffer, size_t bufferSize, size_t* pStringSize) const;

        template <size_t BufferSize>
        DD_NODISCARD bool GetStringCopy(char(&buffer)[BufferSize]) const
        {
            return GetStringCopy(buffer, BufferSize, nullptr);
        }

        // Return a NULL-terminated string from the backing messagepack data.
        // This will fail and return NULL if the embedded string does not end with a NULL byte. Use GetStringCopy() if this is the case.
        DD_NODISCARD const char* GetStringPtr() const;

        // Lookup a value in a map by a string key
        // If the key does not exist, returns false and writes a Null value to `*pValue`
        DD_NODISCARD bool GetValueByKey(const char* pKey, StructuredValue* pValue) const;

        // Lookup a value in an array.
        // If `index` is out of bounds, returns false and writes a Null value to `*pValue`
        DD_NODISCARD bool GetValueByIndex(size_t index, StructuredValue* pValue) const;

        // Query information about Maps and Arrays

        // Returns whether this node has key-value pairs
        bool IsMap() const;

        // Returns whether this node has numeric indices
        bool IsArray() const;

        // Returns the length of the array if this node is an array, otherwise 0.
        size_t GetArrayLength() const;

        // Get-methods with defaults
        // If you don't want to check the `bool` value anyway, prefer these.

        uint8 GetUint8Or(uint8 defaultValue) const
        {
            const bool ok = GetUint8(&defaultValue);
            DD_UNUSED(ok);
            return defaultValue;
        }

        uint16 GetUint16Or(uint16 defaultValue) const
        {
            const bool ok = GetUint16(&defaultValue);
            DD_UNUSED(ok);
            return defaultValue;
        }

        uint32 GetUint32Or(uint32 defaultValue) const
        {
            const bool ok = GetUint32(&defaultValue);
            DD_UNUSED(ok);
            return defaultValue;
        }

        uint64 GetUint64Or(uint64 defaultValue) const
        {
            const bool ok = GetUint64(&defaultValue);
            DD_UNUSED(ok);
            return defaultValue;
        }

        int8 GetInt8Or(int8 defaultValue) const
        {
            const bool ok = GetInt8(&defaultValue);
            DD_UNUSED(ok);
            return defaultValue;
        }

        int16 GetInt16Or(int16 defaultValue) const
        {
            const bool ok = GetInt16(&defaultValue);
            DD_UNUSED(ok);
            return defaultValue;
        }

        int32 GetInt32Or(int32 defaultValue) const
        {
            const bool ok = GetInt32(&defaultValue);
            DD_UNUSED(ok);
            return defaultValue;
        }

        int64 GetInt64Or(int64 defaultValue) const
        {
            const bool ok = GetInt64(&defaultValue);
            DD_UNUSED(ok);
            return defaultValue;
        }

        float GetFloatOr(float defaultValue) const
        {
            const bool ok = GetFloat(&defaultValue);
            DD_UNUSED(ok);
            return defaultValue;
        }

        double GetDoubleOr(double defaultValue) const
        {
            const bool ok = GetDouble(&defaultValue);
            DD_UNUSED(ok);
            return defaultValue;
        }

        bool GetBoolOr(bool defaultValue) const
        {
            const bool ok = GetBool(&defaultValue);
            DD_UNUSED(ok);
            return defaultValue;
        }

        // Index methods

        StructuredValue operator[](const char* pKey) const
        {
            StructuredValue next = MakeNull();

            // Ignore the result of this fetch, `next` is already an empty value
            const bool ok = GetValueByKey(pKey, &next);
            DD_UNUSED(ok);

            return next;
        }

        StructuredValue operator[](size_t index) const
        {
            StructuredValue next = MakeNull();

            // Ignore the result of this fetch, `next` is already an empty value
            const bool ok = GetValueByIndex(index, &next);
            DD_UNUSED(ok);

            return next;
        }

        template <typename T>
        StructuredValue operator[](T index) const
        {
            // This overload exists so that we don't get ambiguous calls when calling operator[] with integer types.
            // If the type of the index can't be statically cast to a size_t, this will fail to compile.
            // Note: pointer types cannot be static_cast()'d, which is great!
            return this->operator[](static_cast<size_t>(index));
        }

    private:
        bool ResetInternalErrorStateImpl(const char* pFile, int line, const char* pCallingFunction) const;

        OpaqueNode m_opaque;
    };

    // Top level container of structured data
    class IStructuredReader
    {
    public:
        virtual ~IStructuredReader() {};

        DD_NODISCARD static Result CreateFromJson(
            const void*         pBytes,
            size_t              numBytes,
            const AllocCb&      allocCb,
            IStructuredReader** ppReader
        );

        DD_NODISCARD static Result CreateFromMessagePack(
            const uint8*        pBytes,
            size_t              numBytes,
            const AllocCb&      allocCb,
            IStructuredReader** ppReader
        );

        static void Destroy(IStructuredReader **ppReader);

        /// Get the root object being read
        virtual StructuredValue GetRoot() const = 0;

        /// Get the allocation callbacks
        virtual const AllocCb& GetAllocCb() const = 0;
    };

} // DevDriver
