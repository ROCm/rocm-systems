ROCM_PATH ?=/opt/rocm
ROCPROFILER_LIBS_PATH ?=$(ROCM_PATH)/lib
ROCM_INCLUDES=-I$(ROCM_PATH)/include
ROCPROFILER_INCLUDES=-I$(ROCM_PATH)/include/rocprofiler/

LIBS=-L$(ROCPROFILER_LIBS_PATH) -lrocprofiler64 -lamd_comgr

ifndef ROCPROFILER_LIBS_PATH
$(warning You may need to set ROCPROFILER_LIBS_PATH to the path of the rocprofiler source)
endif

.PHONY: all
all: kernel_profiling_no_replay_sample device_profiling_sample kernel_replay_sample application_replay_sample user_replay_sample tracer_sample pc_sampler

CXX=hipcc -std=c++17 -Wall

DEBUG=1
ifeq ($(DEBUG), 1)
    CFLAGS =-DDEBUG -g
else
    CFLAGS=-DNDEBUG
endif

kernel_profiling_no_replay_sample: profiler/kernel_profiling_no_replay_sample.cpp common/helper.cpp
	$(CXX) $^ $(LDFLAGS) -o $@ $(LIBS) $(ROCPROFILER_INCLUDES) $(ROCM_INCLUDES) $(HIP_INCLUDES) $(CFLAGS)

device_profiling_sample: profiler/device_profiling_sample.cpp common/helper.cpp
	$(CXX) $^ $(LDFLAGS) -o $@ $(LIBS) $(ROCPROFILER_INCLUDES) $(ROCM_INCLUDES) $(HIP_INCLUDES) $(CFLAGS)

kernel_replay_sample: profiler/kernel_replay_sample.cpp common/helper.cpp
	$(CXX) $^ $(LDFLAGS) -o $@ $(LIBS) $(ROCPROFILER_INCLUDES) $(ROCM_INCLUDES) $(HIP_INCLUDES) $(CFLAGS)

application_replay_sample: profiler/application_replay_sample.cpp common/helper.cpp
	$(CXX) $^ $(LDFLAGS) -o $@ $(LIBS) $(ROCPROFILER_INCLUDES) $(ROCM_INCLUDES) $(HIP_INCLUDES) $(CFLAGS)

user_replay_sample: profiler/user_replay_sample.cpp common/helper.cpp
	$(CXX) $^ $(LDFLAGS) -o $@ $(LIBS) $(ROCPROFILER_INCLUDES) $(ROCM_INCLUDES) $(HIP_INCLUDES) $(CFLAGS)

tracer_sample: tracer/sample.cpp common/helper.cpp
	$(CXX) $^ $(LDFLAGS) -o $@ $(LIBS) $(ROCPROFILER_INCLUDES) $(ROCM_INCLUDES) $(HIP_INCLUDES) $(CFLAGS)

pc_sampler : 
	cd pcsampler/code_printing_sample && $(MAKE)
	


.PHONY: clean

clean:
	$(RM) -rf kernel_profiling_no_replay_sample device_profiling_sample kernel_replay_sample application_replay_sample user_replay_sample tracer_sample && cd pcsampler/code_printing_sample && $(MAKE) clean
