// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier:  MIT

#define ROCPROFILER_CALL(result, msg)                                                                  \
    {                                                                                                  \
        rocprofiler_status_t CHECKSTATUS = result;                                                     \
        if (CHECKSTATUS != ROCPROFILER_STATUS_SUCCESS)                                                 \
        {                                                                                              \
            std::string status_msg = rocprofiler_get_status_string(CHECKSTATUS);                       \
            std::cerr << "[" #result "][" << __FILE__ << ":" << __LINE__ << "] " << msg                \
                      << " failed with error code " << CHECKSTATUS << ": " << status_msg << std::endl; \
            std::stringstream errmsg{};                                                                \
            errmsg << "[" #result "][" << __FILE__ << ":" << __LINE__ << "] " << msg " failure ("      \
                   << status_msg << ")";                                                               \
            throw std::runtime_error(errmsg.str());                                                    \
        }                                                                                              \
    }
