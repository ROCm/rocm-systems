ROCM_PATH ?=/opt/rocm
ROCPROFILER_LIBS_PATH ?=$(ROCM_PATH)/lib
ROCM_INCLUDES=-I$(ROCM_PATH)/include
ROCPROFILER_INCLUDES=-I$(ROCPROFILER_LIBS_PATH)/../include/rocprofiler/

LIBS=-L$(ROCPROFILER_LIBS_PATH) -lamd_comgr -lrocprofiler64v2

ifndef ROCPROFILER_LIBS_PATH
$(warning You may need to set ROCPROFILER_LIBS_PATH to the path of the rocprofiler source)
endif

.PHONY: all
all: profiler_kernel_no_replay profiler_device_profiling profiler_kernel_replay profiler_application_replay profiler_user_replay tracer_hip_hsa tracer_hip_hsa_async pcie_counters_sampler xgmi_counters_sampler pc_sampler

CXX=$(ROCM_PATH)/bin/hipcc -std=c++17 -Wall

DEBUG=1
ifeq ($(DEBUG), 1)
    CFLAGS =-DDEBUG -g
else
    CFLAGS=-DNDEBUG
endif

profiler_kernel_no_replay: profiler/kernel_profiling_no_replay_sample.cpp common/helper.cpp
	$(CXX) $^ $(LDFLAGS) -o $@ $(LIBS) $(ROCPROFILER_INCLUDES) $(ROCM_INCLUDES) $(CFLAGS)

profiler_device_profiling: profiler/device_profiling_sample.cpp common/helper.cpp
	$(CXX) $^ $(LDFLAGS) -o $@ $(LIBS) $(ROCPROFILER_INCLUDES) $(ROCM_INCLUDES) $(CFLAGS)

profiler_kernel_replay: profiler/kernel_replay_sample.cpp common/helper.cpp
	$(CXX) $^ $(LDFLAGS) -o $@ $(LIBS) $(ROCPROFILER_INCLUDES) $(ROCM_INCLUDES) $(CFLAGS)

profiler_application_replay: profiler/application_replay_sample.cpp common/helper.cpp
	$(CXX) $^ $(LDFLAGS) -o $@ $(LIBS) $(ROCPROFILER_INCLUDES) $(ROCM_INCLUDES) $(CFLAGS)

profiler_user_replay: profiler/user_replay_sample.cpp common/helper.cpp
	$(CXX) $^ $(LDFLAGS) -o $@ $(LIBS) $(ROCPROFILER_INCLUDES) $(ROCM_INCLUDES) $(CFLAGS)

tracer_hip_hsa: tracer/sample.cpp common/helper.cpp
	$(CXX) $^ $(LDFLAGS) -o $@ $(LIBS) $(ROCPROFILER_INCLUDES) $(ROCM_INCLUDES) $(CFLAGS)

tracer_hip_hsa_async: tracer/sample_async.cpp common/helper.cpp
	$(CXX) $^ $(LDFLAGS) -o $@ $(LIBS) $(ROCPROFILER_INCLUDES) $(ROCM_INCLUDES) $(CFLAGS)

pcie_counters_sampler: counters_sampler/pcie_counters_example.cpp common/helper.cpp
	$(CXX) $^ $(LDFLAGS) -o $@ $(LIBS) $(ROCPROFILER_INCLUDES) $(ROCM_INCLUDES) $(CFLAGS)

xgmi_counters_sampler: counters_sampler/xgmi_counters_sampler_example.cpp common/helper.cpp
	$(CXX) $^ $(LDFLAGS) -o $@ $(LIBS) $(ROCPROFILER_INCLUDES) $(ROCM_INCLUDES) $(CFLAGS)

pc_sampler :
	cd pcsampler/code_printing_sample && $(MAKE)



.PHONY: clean

clean:
	$(RM) -rf xgmi_counters_sampler pcie_counters_sampler tracer_hip_hsa_async tracer_hip_hsa profiler_user_replay profiler_application_replay profiler_kernel_replay profiler_device_profiling profiler_kernel_no_replay && cd pcsampler/code_printing_sample && $(MAKE) clean
