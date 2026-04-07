/***************************************************************************
 Copyright (c) 2022, Advanced Micro Devices, Inc. All rights reserved.
***************************************************************************/

#ifndef MARKERS_H
#define MARKERS_H

#include <hip/hip_runtime.h>
#include <iostream>
#include <sstream>
#include <stdexcept>

#define CONCAT_(x, y) x##y
#define CONCAT(x, y)  CONCAT_(x, y)

#if defined(__HIP_PLATFORM_HCC__)
// begin HCC

#    include <roctracer_ext.h>
#    include <roctx.h>

#    define BEGIN_RANGE(name, group)                                                     \
        do                                                                               \
        {                                                                                \
            std::stringstream ss;                                                        \
            ss << (name) << "::" << (group);                                             \
            std::string str  = ss.str();                                                 \
            const char* cstr = str.c_str();                                              \
            roctxRangePush(cstr);                                                        \
        } while(0)  // must terminate with semi-colon

#    define END_RANGE()                                                                  \
        do                                                                               \
        {                                                                                \
            if(roctxRangePop() < 0)                                                      \
            {                                                                            \
                std::stringstream ss;                                                    \
                ss << "rocTX error popping range." << std::endl;                         \
                std::string str = ss.str();                                              \
                throw std::runtime_error(str);                                           \
            }                                                                            \
        } while(0)

// embed event recording in class to automatically pop when destroyed
class ScopedRange
{
public:
    ScopedRange(const char* name, const char* group)
    {
        std::stringstream ss;
        ss << (name) << "::" << (group);
        std::string str  = ss.str();
        const char* cstr = str.c_str();
        roctxRangePush(cstr);
    }
    ~ScopedRange() { roctxRangePop(); }
};

#    define SCOPED_RANGE(name, group)                                                    \
        ScopedRange CONCAT(range, __COUNTER__)((name), (group))

#    define MARKER(name, group)                                                          \
        do                                                                               \
        {                                                                                \
            std::stringstream ss;                                                        \
            ss << (name) << "::" << (group);                                             \
            std::string str  = ss.str();                                                 \
            const char* cstr = str.c_str();                                              \
            roctxMark(cstr);                                                             \
        } while(0)  // must terminate with semi-colon

static inline void
initialize_logger()
{
    roctracer_start();
}

static inline void
finalize_logger()
{
    roctracer_stop();
}

#else  // __HIP_PLATFORM_NVCC__
// Begin CUDA

#    include <cuda_profiler_api.h>
#    include <nvToolsExt.h>
static inline void
initialize_logger()
{
    if(cudaProfilerStart() != cudaSuccess)
        throw std::runtime_error("CUDA profiler could not be initialized!");
}

static inline void
finalize_logger()
{
    if(cudaProfilerStop() != cudaSuccess)
        throw std::runtime_error("CUDA profiler could not be stopped!");
}

// start recording an event
#    define BEGIN_RANGE(name, group)                                                     \
        do                                                                               \
        {                                                                                \
            std::stringstream ss;                                                        \
            ss << name << "::" << group;                                                 \
            std::string           str         = ss.str();                                \
            const char*           cstr        = str.c_str();                             \
            nvtxEventAttributes_t eventAttrib = { 0 };                                   \
            eventAttrib.version               = NVTX_VERSION;                            \
            eventAttrib.size                  = NVTX_EVENT_ATTRIB_STRUCT_SIZE;           \
            eventAttrib.messageType           = NVTX_MESSAGE_TYPE_ASCII;                 \
            eventAttrib.message.ascii         = cstr;                                    \
            nvtxRangePushEx(&eventAttrib);                                               \
        } while(0)  // must terminate with semi-colon

// stop recording an event
// must terminate with semi-colon
#    define END_RANGE()                                                                  \
        do                                                                               \
        {                                                                                \
            nvtxRangePop();                                                              \
        } while(0)

// embed event recording in class to automatically pop when destroyed
class ScopedRange
{
public:
    ScopedRange(const char* name, const char* group) { BEGIN_RANGE(name, group); }
    ~ScopedRange() { END_RANGE(); }
};

// call RANGE at beginning of function to push event recording
// destructor is automatically called on return to pop event recording
// must terminate with semi-colon
#    define SCOPED_RANGE(name, group) ScopedRange CONCAT(range, __COUNTER__)(name, group)

#    define MARKER(name, group)                                                          \
        do                                                                               \
        {                                                                                \
            std::stringstream ss;                                                        \
            ss << name << "::" << group;                                                 \
            std::string           str         = ss.str();                                \
            const char*           cstr        = str.c_str();                             \
            nvtxEventAttributes_t eventAttrib = { 0 };                                   \
            eventAttrib.version               = NVTX_VERSION;                            \
            eventAttrib.size                  = NVTX_EVENT_ATTRIB_STRUCT_SIZE;           \
            eventAttrib.messageType           = NVTX_MESSAGE_TYPE_ASCII;                 \
            eventAttrib.message.ascii         = cstr;                                    \
            nvtxMarkEx(&eventAttrib);                                                    \
        } while(0)  // must terminate with semi-colon

#endif  // __HIP_PLATFORM_NVCC__

#endif  // MARKERS_H
