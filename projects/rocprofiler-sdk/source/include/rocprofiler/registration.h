// MIT License
//
// Copyright (c) 2023 Advanced Micro Devices, Inc. All rights reserved.
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

#include <rocprofiler/defines.h>
#include <rocprofiler/fwd.h>

ROCPROFILER_EXTERN_C_INIT

/**
 * @defgroup REGISTRATION_GROUP Tool registration
 *
 * Data types and functions for tool registration with rocprofiler
 * @{
 */

/**
 * @brief A pointer to this data structure is provided to the client tool initialization function.
 * The name member can be set by the client to assist with debugging (e.g. rocprofiler cannot start
 * your context because there is a conflicting context started by `<name>` -- at least that is the
 * plan). The handle member is a unique identifer assigned by rocprofiler for the client and the
 * client can store it and pass it to the @ref rocprofiler_client_finalize_t function to force
 * finalization (i.e. deactivate all of it's contexts) for the client.
 */
typedef struct
{
    const char*    name;    ///< clients should set this value for debugging
    const uint32_t handle;  ///< internal handle
} rocprofiler_client_id_t;

typedef void (*rocprofiler_client_finalize_t)(rocprofiler_client_id_t);

typedef int (*rocprofiler_tool_initialize_t)(rocprofiler_client_finalize_t finalize_func,
                                             void*                         tool_data);

typedef void (*rocprofiler_tool_finalize_t)(void* tool_data);

/**
 * @brief Data structure containing a initialization, finalization, and data
 *
 */
typedef struct
{
    size_t                        size;        ///< in case of future extensions
    rocprofiler_tool_initialize_t initialize;  ///< context creation
    rocprofiler_tool_finalize_t   finalize;    ///< cleanup
    void*                         tool_data;   ///< data to provide to init and fini callbacks
} rocprofiler_tool_configure_result_t;

/**
 * @brief Query whether rocprofiler has already scanned the binary for all the instances of @ref
 * rocprofiler_configure (or is currently scanning). If rocprofiler has completed it's scan, clients
 * can directly register themselves with rocprofiler.
 *
 * @param [out] status 0 indicates rocprofiler has not been initialized (i.e. configured), 1
 * indicates rocprofiler has been initialized, -1 indicates rocprofiler is currently initializing.
 * @return ::rocprofiler_status_t
 */
rocprofiler_status_t
rocprofiler_is_initialized(int* status) ROCPROFILER_API;

/**
 * @brief Query rocprofiler finalization status.
 *
 * @param [out] status 0 indicates rocprofiler has not been finalized, 1 indicates rocprofiler has
 * been finalized, -1 indicates rocprofiler is currently finalizing.
 * @return ::rocprofiler_status_t
 */
rocprofiler_status_t
rocprofiler_is_finalized(int* status) ROCPROFILER_API;

/**
 * @brief This is the special function that tools define to enable rocprofiler support. The tool
 * should return a pointer to
 * @ref rocprofiler_tool_configure_result_t which will contain a function pointer to (1) an
 * initialization function where all the contexts are created, (2) a finalization function (if
 * necessary) which will be invoked when rocprofiler shutdown and, (3) a pointer to any data that
 * the tool wants communicated between the @ref rocprofiler_tool_configure_result_t::initialize and
 * @ref rocprofiler_tool_configure_result_t::finalize functions. If the user
 *
 * @param [in] version The version of rocprofiler: `(10000 * major) + (100 * minor) + patch`
 * @param [in] runtime_version String descriptor of the rocprofiler version and other relevant info.
 * @param [in] priority How many client tools were initialized before this client tool
 * @param [in, out] client_id tool identifier value.
 * @return rocprofiler_tool_configure_result_t*
 *
 * @code{.cpp}
 * #include <rocprofiler/registration.h>
 *
 * static rocprofiler_client_id_t       my_client_id;
 * static rocprofiler_client_finalize_t my_fini_func;
 * static int                           my_tool_data = 1234;
 *
 * static int my_init_func(rocprofiler_client_finalize_t fini_func,
 *                         void* tool_data)
 * {
 *      my_fini_func = fini_func;
 *
 *      assert(*static_cast<int*>(tool_data) == 1234 && "tool_data is wrong");
 *
 *      rocprofiler_context_id_t ctx;
 *      rocprofiler_create_context(&ctx);
 *
 *      if(int valid_ctx = 0;
 *         rocprofiler_context_is_valid(ctx, &valid_ctx) != ROCPROFILER_STATUS_SUCCESS ||
 *         valid_ctx != 0)
 *      {
 *          // notify rocprofiler that initialization failed
 *          // and all the contexts, buffers, etc. created
 *          // should be ignored
 *          return -1;
 *      }
 *
 *      if(rocprofiler_start_context(ctx) != ROCPROFILER_STATUS_SUCCESS)
 *      {
 *          // notify rocprofiler that initialization failed
 *          // and all the contexts, buffers, etc. created
 *          // should be ignored
 *          return -1;
 *      }
 *
 *      // no errors
 *      return 0;
 * }
 *
 * static int my_fini_func(void* tool_data)
 * {
 *      assert(*static_cast<int*>(tool_data) == 1234 && "tool_data is wrong");
 * }
 *
 * rocprofiler_tool_configure_result_t*
 * rocprofiler_configure(uint32_t version,
 *                       const char* runtime_version,
 *                       uint32_t priority,
 *                       rocprofiler_client_id_t* client_id)
 * {
 *      // only activate if main tool
 *      if(priority > 0) return nullptr;
 *
 *      // set the client name
 *      client_id->name = "ExampleTool";
 *
 *      // make a copy of client info
 *      my_client_id = *client_id;
 *
 *      // compute major/minor/patch version info
 *      uint32_t major = version / 10000;
 *      uint32_t minor = (version % 10000) / 100;
 *      uint32_t patch = version % 100;
 *
 *      // print info
 *      printf("Configuring rocprofiler (v%u.%u.%u) [%s]\n", major, minor, patch, runtime_version);
 *
 *      // create configure data
 *      static auto cfg = rocprofiler_tool_configure_result_t{ &my_init_func,
 *                                                             &my_fini_func,
 *                                                             &my_tool_data };
 *
 *      // return pointer to configure data
 *      return &cfg;
 * }
 * @endcode
 */
rocprofiler_tool_configure_result_t*
rocprofiler_configure(uint32_t                 version,
                      const char*              runtime_version,
                      uint32_t                 priority,
                      rocprofiler_client_id_t* client_id) ROCPROFILER_PUBLIC_API;

// NOTE: we use ROCPROFILER_PUBLIC_API above instead of ROCPROFILER_API because we always
// want the symbol to be visible when the user includes the header for the prototype

/**
 * @brief Function pointer typedef for @ref rocprofiler_configure function
 * @param [in] version The version of rocprofiler: `(10000 * major) + (100 * minor) + patch`
 * @param [in] runtime_version String descriptor of the rocprofiler version and other relevant info.
 * @param [in] priority How many client tools were initialized before this client tool
 * @param [in, out] client_id tool identifier value.
 */
typedef rocprofiler_tool_configure_result_t* (*rocprofiler_configure_func_t)(
    uint32_t                 version,
    const char*              runtime_version,
    uint32_t                 priority,
    rocprofiler_client_id_t* client_id);

/**
 * @brief Function for explicitly registering a configuration with rocprofiler. This can be invoked
 * before any ROCm runtimes (lazily) initialize and context(s) can be started before the runtimes
 * initialize.
 * @param [in] configure_func Address of @ref rocprofiler_configure function. A null pointer is
 * acceptable if the address is not known
 * @returns rocprofiler_status_t If rocprofiler has already been configured, or is currently being
 * configured, this function will return @ref ROCPROFILER_STATUS_ERROR_CONFIGURATION_LOCKED.
 */
rocprofiler_status_t
rocprofiler_force_configure(rocprofiler_configure_func_t configure_func) ROCPROFILER_API;

/** @} */

ROCPROFILER_EXTERN_C_FINI
