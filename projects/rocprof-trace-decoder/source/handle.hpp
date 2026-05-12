// MIT License
//
// Copyright (c) 2024-2026 Advanced Micro Devices, Inc. All rights reserved.
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.

#pragma once
#include "trace_parser.hpp"

#include <cstdint>
#include <memory>
#include <mutex>
#include <unordered_map>
#include <condition_variable>

#ifndef ROCPROF_TRACE_DECODER_COMGR_DISABLED
using AddressTable = rocprof_trace_decoder::codeobj::CodeobjAddressTranslate;
using Instruction = rocprof_trace_decoder::codeobj::Instruction;
#endif

template<typename T>
class ReadLock
{
public:
    ReadLock(std::shared_ptr<T> instance, std::shared_mutex& mut): value(instance), lk(std::shared_lock{mut}) {}

    bool valid() const { return value != nullptr; }
    const T* operator->() const { if(!value) throw std::runtime_error("Invalid lock"); return value.get(); }

    std::shared_lock<std::shared_mutex> lk;
private:
    std::shared_ptr<const T> value;
};

template<typename T>
class WriteLock
{
public:
    WriteLock(std::shared_ptr<T> instance, std::shared_mutex& mut): value(instance), lk(std::unique_lock{mut}) {}

    bool valid() const { return value != nullptr; }
    T* operator->() { if(!value) throw std::runtime_error("Invalid lock"); return value.get(); }

    std::unique_lock<std::shared_mutex> lk;
private:
    std::shared_ptr<T> value;
};

class HandleData
{
public:
    rocprof_trace_decoder_isa_callback_t isa_cb{nullptr};
    void* isa_userdata{nullptr};

    rocprof_trace_decoder_se_data_callback_t se_data_cb{nullptr};
    void* se_data_userdata{nullptr};

    // quick_scan state data
    mutable std::condition_variable_any cv;
    int gfxip = 0;
    // Raw 8-byte chunk-0 header captured by quick_scan. Used by
    // build_standalone to prepend the original arch header to the rebuilt
    // standalone buffer (so a downstream decoder can identify the arch).
    // Currently only the gfx9 8-byte header is captured; future archs will
    // either reuse this slot (if their header fits in 8 bytes) or extend the
    // storage.
    uint64_t gfx9_header = 0;
    std::unordered_map<uint64_t, std::shared_ptr<CSRegisterHandler>> pipestate{};

#ifndef ROCPROF_TRACE_DECODER_COMGR_DISABLED
    WriteLock<AddressTable> decoder() { return {instance, decoder_mut}; }
#endif

    static std::mutex& get_map_mutex()
    {
        static std::mutex mtx;
        return mtx;
    }

    static std::unordered_map<uint64_t, std::shared_ptr<HandleData>>& get_map()
    {
        static std::unordered_map<uint64_t, std::shared_ptr<HandleData>> map;
        return map;
    }

    static ReadLock<HandleData> get_read_handle(rocprof_trace_decoder_handle_t handle)
    {
        static std::shared_mutex dummy_mut{};
    
        std::lock_guard<std::mutex> lock(get_map_mutex());
        auto& map = get_map();
        auto it = map.find(handle.handle);
        if (it != map.end()) return ReadLock<HandleData>{it->second, it->second->mtx};
        return {nullptr, dummy_mut};
    }

    static WriteLock<HandleData> get_write_handle(rocprof_trace_decoder_handle_t handle)
    {
        static std::shared_mutex dummy_mut{};
    
        std::lock_guard<std::mutex> lock(get_map_mutex());
        auto& map = get_map();
        auto it = map.find(handle.handle);
        if (it != map.end()) return WriteLock<HandleData>{it->second, it->second->mtx};
        return {nullptr, dummy_mut};
    }

private:
    std::shared_mutex mtx;

#ifndef ROCPROF_TRACE_DECODER_COMGR_DISABLED
    std::shared_mutex decoder_mut{};
    std::shared_ptr<AddressTable> instance;
#endif
};
