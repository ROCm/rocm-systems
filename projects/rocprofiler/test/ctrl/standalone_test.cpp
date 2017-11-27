#include <hsa.h>
#include <string.h>
#include <iostream>

#include "ctrl/run_kernel.h"
#include "ctrl/test_aql.h"
#include "ctrl/test_hsa.h"
#include "inc/rocprofiler.h"
#include "simple_convolution/simple_convolution.h"
#include "util/test_assert.h"

int main(int argc, char** argv) {
    bool ret_val = false;
    // HSA status
    hsa_status_t status = HSA_STATUS_ERROR;
    // Profiling context
    rocprofiler_t* context = NULL;
    // Profiling properties
    rocprofiler_properties_t properties;
    // Number of context invocation
    uint32_t invocation = 0;

#if 0
    // Profiling info objects
    const unsigned info_count = 1;
    rocprofiler_info_t info[info_count];
    // PMC events
    memset(info, 0, sizeof(info));
    info[0].type = ROCPROFILER_TYPE_METRIC;
    info[0].name = "SQ_WAVES";
#else
    // Profiling info objects
    const unsigned info_count = 3;
    rocprofiler_info_t info[info_count];
    // PMC events
    memset(info, 0, sizeof(info));
    info[0].type = ROCPROFILER_TYPE_METRIC;
    info[0].name = "SQ_WAVES";
    info[1].type = ROCPROFILER_TYPE_METRIC;
    info[1].name = "SQ_ITEMS";
    // Tracing parameters
    const unsigned parameter_count = 2;
    rocprofiler_parameter_t parameters[parameter_count];
    info[2].name = "THREAD_TRACE";
    info[2].type = ROCPROFILER_TYPE_TRACE;
    info[2].parameters = parameters;
    info[2].parameter_count = parameter_count;
    parameters[0].parameter_name = HSA_VEN_AMD_AQLPROFILE_PARAMETER_NAME_MASK;
    parameters[0].value = 0;
    parameters[1].parameter_name = HSA_VEN_AMD_AQLPROFILE_PARAMETER_NAME_TOKEN_MASK;
    parameters[1].value = 0;
#endif

    // Creating profiling context
    properties = {};
    properties.queue_depth = 128;
    status = rocprofiler_open(TestHsa::HsaAgentId(), info, info_count, &context, ROCPROFILER_MODE_STANDALONE|ROCPROFILER_MODE_OWNQUEUE, &properties);
    TEST_STATUS(status == HSA_STATUS_SUCCESS);

    TestHsa::SetQueue(properties.queue);

    // Adding dispatch observer
    status = rocprofiler_dispatch_observer(rocprofiler_dispatch_callback, context);
    TEST_STATUS(status == HSA_STATUS_SUCCESS);

    // Querying the number of context invocation 
    status = rocprofiler_invocation(context, &invocation);
    TEST_STATUS(status == HSA_STATUS_SUCCESS);

    // Dispatching profiled kernel n-times to collect all counter groups data
    unsigned n = 0;
    while(1) {
        std::cout << "> " << n << "/" << invocation << std::endl;
#if 0
        status = rocprofiler_start(context);
        TEST_STATUS(status == HSA_STATUS_SUCCESS);
        ret_val = RunKernel<SimpleConvolution, TestAql>(argc, argv);
        status = rocprofiler_stop(context);
        TEST_STATUS(status == HSA_STATUS_SUCCESS);
#else
        ret_val = RunKernel<SimpleConvolution, TestAql>(argc, argv);
#endif
        status = rocprofiler_sample(context);
        TEST_STATUS(status == HSA_STATUS_SUCCESS);

        for (rocprofiler_info_t* p = info; p < info + info_count; ++p) {
          std::cout << (p - info) << ": " << p->name;
          switch (p->data.kind) {
            case ROCPROFILER_DATA_KIND_INT64:
              std::cout << std::dec << " result64 (" << p->data.result64 << ")" << std::endl;
              break;
            case ROCPROFILER_BYTES: {
              const char* ptr = reinterpret_cast<const char*>(p->data.result_bytes.ptr);
              uint64_t size = 0;
              for (unsigned i = 0; i < p->data.result_bytes.instance_count; ++i) {
                size = *reinterpret_cast<const uint64_t*>(ptr);
                const char* data = ptr + sizeof(size);
                std::cout << std::endl;
                std::cout << std::hex << "  data (" << (void*)data << ")" << std::endl;
                std::cout << std::dec << "  size (" << size << ")" << std::endl;
                ptr = data + size;
              }
              break;
            }
            default:
              std::cout << "result kind (" << p->data.kind << ")" << std::endl;
              TEST_ASSERT(false);
          }
        }

        ++n;
        if (n < invocation) {
            status = rocprofiler_next(context);
            TEST_STATUS(status == HSA_STATUS_SUCCESS);
            continue;
        }
        break;
    }

    // Finishing cleanup
    // Deleting profiling context will delete all allocated resources
    status = rocprofiler_close(context);
    TEST_STATUS(status == HSA_STATUS_SUCCESS);

    return (ret_val) ? 0 : 1;
}
