// MIT License
//
// Copyright (c) 2024 Advanced Micro Devices, Inc. All rights reserved.
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

#include <fstream>
#include <memory>
#include <shared_mutex>
#include <string>
#include <unordered_map>
#include <vector>

/**
 * Enum defines how code object is captured for ATT and PC Sampling
 */
enum codeobj_capture_mode_t
{
    /**
     * Capture file and memory paths for the loaded code object
     */
    ROCPROFILER_CODEOBJ_CAPTURE_SYMBOLS_ONLY = 0,
    /**
     * Capture symbols for file:// and memory:// type objects,
     * and generate a copy of all kernel code for objects under memory://
     */
    ROCPROFILER_CODEOBJ_CAPTURE_COPY_MEMORY = 1,
    /**
     * Capture symbols and all kernel code for file:// and memory:// type objects
     */
    ROCPROFILER_CODEOBJ_CAPTURE_COPY_FILE_AND_MEMORY = 2,
    ROCPROFILER_CODEOBJ_CAPTURE_LAST                 = 3,
};

/**
 * A class to keep track of currently loaded code objects.
 * Only the public static methods are thread-safe and expected to be used.
 */
class CodeobjCaptureInstance
{
public:
    CodeobjCaptureInstance(uint64_t               _addr,
                           uint64_t               _load_size,
                           const std::string&     _uri,
                           uint64_t               _mem_addr,
                           uint64_t               _mem_size,
                           uint64_t               id,
                           codeobj_capture_mode_t mode)
    : addr(_addr)
    , load_size(_load_size)
    , load_id(id)
    , URI(_uri)
    , mem_addr(_mem_addr)
    , mem_size(_mem_size)
    {
        reset(mode);
    };

    const uint64_t addr;
    const uint64_t load_size;
    const uint64_t load_id;

private:
    void reset(codeobj_capture_mode_t mode);

    std::pair<size_t, size_t> parse_uri();
    void                      DecodePath();
    void copyCodeobjFromFile(uint64_t offset, uint64_t size, const std::string& path);
    void copyCodeobjFromMemory(uint64_t, uint64_t);

    std::string       URI{};
    std::string       decoded_path{};
    std::string       protocol{};
    std::vector<char> buffer{};

    uint64_t               mem_addr     = 0;
    uint64_t               mem_size     = 0;
    codeobj_capture_mode_t capture_mode = ROCPROFILER_CODEOBJ_CAPTURE_SYMBOLS_ONLY;
};

typedef std::shared_ptr<CodeobjCaptureInstance> CodeobjPtr;

template <>
struct std::hash<CodeobjPtr>
{
    uint64_t operator()(const CodeobjPtr& p) const { return p->load_id; }
};

template <>
struct std::equal_to<CodeobjPtr>
{
    bool operator()(const CodeobjPtr& a, const CodeobjPtr& b) const
    {
        return (a->addr == b->addr) && (a->load_id == b->load_id);
    };
};

/**
 * A class to keep track of the history of loaded code objets.
 * Only the public static methods are thread-safe and expected to be used.
 */
class CodeobjRecorder
{
public:
    CodeobjRecorder(codeobj_capture_mode_t mode)
    : capture_mode(mode){};

    void Load(uint64_t           _addr,
              uint64_t           _load_size,
              const std::string& _uri,
              uint64_t           mem_addr,
              uint64_t           mem_size,
              uint64_t           id);
    void Load(CodeobjPtr capture)
    {
        std::lock_guard<std::shared_mutex> lk(mutex);
        captures[capture->load_id] = capture;
    }
    void Unload(uint64_t id)
    {
        std::lock_guard<std::shared_mutex> lk(mutex);
        captures.erase(id);
    };

public:
    std::shared_mutex mutex;

    std::vector<CodeobjPtr> get()
    {
        std::vector<CodeobjPtr>             vec;
        std::shared_lock<std::shared_mutex> lk(mutex);
        for(auto& [k, v] : captures)
            vec.push_back(v);
        return vec;
    };

private:
    codeobj_capture_mode_t                   capture_mode;
    std::unordered_map<uint64_t, CodeobjPtr> captures;
};
