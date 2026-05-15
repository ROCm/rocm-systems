/*
##############################################################################bl
# MIT License
#
# Copyright (c) 2021 - 2023 Advanced Micro Devices, Inc. All Rights Reserved.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
##############################################################################el



A data-fabric exerciser example, written by Nicholas Curtis [AMD]

The test allows the user to control the:
  - The granularity of an allocation (Coarse vs Fine-grained),
  - The owner of an allocation (local HBM, CPU DRAM or remote HBM),
  - The size of an allocation (the default is ~4GiB), and
  - The type of operation we are executing (read, write, atomics of various flavors)

This lets the user explore the impact of these choices on the generated
data-fabric traffic.
*/


#include <getopt.h>
#include <hip/hip_runtime.h>

#include <iostream>
#include <vector>
#include <cassert>

#include "common.h"

enum class mtype : int { FineGrained = 0, CoarseGrained = 1, Undef = 3 };
enum class mowner : int { Device = 0, Host = 1, Remote = 2, Undef = 3 };
enum class mspace : int { Global = 0, Undef = 1 };
enum class mop : int {
  Read = 0,
  Write = 1,
  AtomicAdd = 2,
  AtomicCas = 3,
  AtomicOr = 4,
  AtomicMax = 5,
  Undef = 6
};
enum class mdata : int { Unsigned = 0, UnsignedLong = 1, Float = 2, Double = 3, Undef = 4 };

template<typename T>
T parse(const char* value) {
  int ivalue = std::atoi(value);
  if (ivalue < 0 || ivalue >= int(T::Undef)) {
    throw std::runtime_error("bad enum value!");
  }
  return T(ivalue);
}

void parse(int argc, char** argv, mtype& mytype, mowner& myowner,
           mspace& myspace, size_t& size, mop& myop, mdata& mydata,
           int& remoteId) {
  while (1) {
    static struct option long_options[] = {
        /* These options set a flag. */
        {"type", required_argument, 0, 't'},
        {"owner", required_argument, 0, 'o'},
        {"size", required_argument, 0, 'z'},
        {"op", required_argument, 0, 'p'},
        {"remote", required_argument, 0, 'r'},
        {"data", required_argument, 0, 'd'},
        {0, 0, 0, 0}};
    /* getopt_long stores the option index here. */
    int option_index = 0;

    int c =
        getopt_long(argc, argv, "t:o:z:p:r:d:", long_options, &option_index);

    /* Detect the end of the options. */
    if (c == -1) break;

    switch (c) {
      case 't':
        mytype = parse<mtype>(optarg);
        break;

      case 'o':
        myowner = parse<mowner>(optarg);
        break;

      case 'z':
        size = std::atoll(optarg);
        break;

      case 'p':
        myop = parse<mop>(optarg);
        break;

      case 'r':
        remoteId = std::atoi(optarg);
        break;

      case 'd':
        mydata = parse<mdata>(optarg);
        break;

      case '?':
        /* getopt_long already printed an error message. */
        break;

      default:
        abort();
    }
  }
  std::cout << "Using: " << std::endl;
  std::cout << "\tmtype:"
            << ((mytype == mtype::FineGrained) ? "FineGrained"
                                               : "CoarseGrained")
            << std::endl;
  std::cout << "\tmowner:"
            << ((myowner == mowner::Device)
                    ? "Device"
                    : ((myowner == mowner::Host) ? "Host" : "Remote"))
            << std::endl;
  std::cout << "\tmspace:Global" << std::endl;
  std::cout << "\tmop:" << ((myop == mop::Read) ? "Read" : (myop == mop::Write ? "Write" : (myop == mop::AtomicAdd ? "Add" : (myop == mop::AtomicCas ? "CAS" : (myop == mop::AtomicOr ? "Or" : "Max"))))) << std::endl;
  std::cout << "\tmdata:" << (mydata == mdata::Unsigned ? "Unsigned" : (mydata == mdata::UnsignedLong ? "Unsigned Long" : (mydata == mdata::Float ? "Float" : "Double"))) << std::endl;
  std::cout << "\tremoteId:" << remoteId << std::endl;
}

// dummy intialization kernel
__global__ void init() {}

template <typename T>
void alloc(mtype memory, mowner owner, T** ptr, size_t Nbytes, int devId,
           int remoteId) {
  bool is_device = (owner == mowner::Device) || (owner == mowner::Remote);
  if (owner == mowner::Remote) {
    // enable remote access
    hipCheck(hipDeviceEnablePeerAccess(remoteId, 0));
    // set id for alloc
    hipCheck(hipSetDevice(remoteId));
  }
  init<<<1, 1>>>();

  if (memory == mtype::FineGrained && is_device) {
    hipCheck(
        hipExtMallocWithFlags((void**)ptr, Nbytes, hipDeviceMallocFinegrained));
  } else if (memory == mtype::CoarseGrained && is_device) {
    hipCheck(hipMalloc(ptr, Nbytes));
  } else if (memory == mtype::FineGrained && owner == mowner::Host) {
    hipCheck(hipHostMalloc(ptr, Nbytes, hipHostMallocCoherent));
  } else if (memory == mtype::CoarseGrained && owner == mowner::Host) {
    hipCheck(hipHostMalloc(ptr, Nbytes, hipHostMallocNonCoherent));
  } else {
    assert(false && "unknown combo");
  }

  // set to random
  std::vector<T> host(Nbytes / sizeof(T), T(0));
  hipCheck(hipMemcpy(*ptr, &host[0], Nbytes,
                  (is_device ? hipMemcpyHostToDevice : hipMemcpyHostToHost)));

  if (owner == mowner::Remote) {
    // reset id for execution
    hipCheck(hipSetDevice(devId));
  }
}

template <typename T>
void release(mtype memory, mowner owner, T* ptr) {
  bool is_device = (owner == mowner::Device) || (owner == mowner::Remote);
  if (memory == mtype::FineGrained && is_device) {
    hipCheck(hipFree(ptr));
  } else if (memory == mtype::CoarseGrained && is_device) {
    hipCheck(hipFree(ptr));
  } else if (memory == mtype::FineGrained && owner == mowner::Host) {
    hipCheck(hipHostFree(ptr));
  } else if (memory == mtype::CoarseGrained && owner == mowner::Host) {
    hipCheck(hipHostFree(ptr));
  } else {
    assert(false && "unknown combo");
  }
}

// the main streaming kernel
template <mop op, typename T, int repeats = 10>
__global__ void kernel(T* x, size_t N, T zero, T foo) {
  int sum = 0;
  const size_t offset_start = threadIdx.x + blockIdx.x * blockDim.x;
  for (int i = 0; i < repeats; ++i) {
    for (size_t offset = offset_start; offset < N;
         offset += blockDim.x * gridDim.x) {
      T uniq = (foo + offset) + i;
      if constexpr (op == mop::Read) {
        sum += x[offset];
      } else if constexpr (op == mop::Write) {
        x[offset] = (T)offset;
      } else if constexpr (op == mop::AtomicAdd) {
        atomicAdd(&x[offset], uniq);
      } else if constexpr (op == mop::AtomicCas) {
        atomicCAS(&x[offset], uniq, uniq);
      } else if constexpr (op == mop::AtomicOr) {
        atomicOr(&x[offset], uniq);
      } else if constexpr (op == mop::AtomicMax) {
        atomicMax(&x[offset], uniq);
      }
    }
  }
  if constexpr (op == mop::Read) {
    if (sum != 0) {
      x[offset_start] = sum;
    }
  }
}

template <mop op, typename T, int nrepeats = 10>
void run_kernel(T* x, size_t size) {
  if constexpr (op == mop::AtomicOr && std::is_floating_point_v<T>) {
    throw std::runtime_error("bad");
  } else {
    kernel<op, T, nrepeats><<<4096, 1024>>>(x, size, 0, T(23456789));
    // then run once for data collection
    kernel<op, T, nrepeats><<<4096, 1024>>>(x, size, 0, T(23456789));
  }
}

template <mop op, typename T>
void run_atomic(mowner myowner, T* x, size_t size) {
  if (myowner == mowner::Host) {
    // speed it up
    run_kernel<op, T, 1>(x, size / 10);
  } else {
    run_kernel<op, T>(x, size);
  }
}

template <typename T>
void run(mtype mytype, mspace myspace, mowner myowner, mop myop, int remoteId,
         size_t size) {
  int devId = 0;
  if (myowner == mowner::Remote && remoteId == -1) {
    // need to find a remote GPU
    int ndevices;
    hipCheck(hipGetDeviceCount(&ndevices));
    if (ndevices <= 1) {
      throw std::runtime_error(
          "Need >=2 devices available for mowner = Remote");
    }
    for (int i = 0; i < ndevices; ++i) {
      if (i != devId) {
        remoteId = i;
        break;
      }
    }
  }

  T* x;
  alloc(mytype, myowner, &x, size * sizeof(T), devId, remoteId);

  // run the kernel once for warmup
  assert(4096 * 1024 < size);
  if (myop == mop::Read) {
    run_kernel<mop::Read>(x, size);
  } else if (myop == mop::Write) {
    run_kernel<mop::Write>(x, size);
  } else if (myop == mop::AtomicAdd) {
    run_atomic<mop::AtomicAdd>(myowner, x, size);
  } else if (myop == mop::AtomicCas) {
    run_atomic<mop::AtomicCas>(myowner, x, size);
  } else if (myop == mop::AtomicOr) {
    run_atomic<mop::AtomicOr>(myowner, x, size);
  } else if (myop == mop::AtomicMax) {
    run_atomic<mop::AtomicMax>(myowner, x, size);
  } else {
    throw std::runtime_error("bad");
  }
  hipCheck(hipDeviceSynchronize());
  release(mytype, myowner, x);
}

int main(int argc, char** argv) {
  mtype mytype = (mtype)0;
  mspace myspace = (mspace)0;
  mowner myowner = (mowner)0;
  mop myop = (mop)0;
  mdata mydata = (mdata)0;
  int remoteId = -1;
  size_t size = 1024ull * 1024ull *
                1024ull;  // 4 GiB, purposefully much larger than caches.
  parse(argc, argv, mytype, myowner, myspace, size, myop, mydata, remoteId);
  if (mydata == mdata::Unsigned)
    run<unsigned>(mytype, myspace, myowner, myop, remoteId, size);
  else if (mydata == mdata::UnsignedLong)
    run<unsigned long>(mytype, myspace, myowner, myop, remoteId, size);
  else if (mydata == mdata::Float)
    run<float>(mytype, myspace, myowner, myop, remoteId, size);
  else if (mydata == mdata::Double)
    run<double>(mytype, myspace, myowner, myop, remoteId, size);
  else {
    throw std::runtime_error("bad");
  }
}
