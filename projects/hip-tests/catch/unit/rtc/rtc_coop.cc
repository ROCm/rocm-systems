#include "warp_common.hh"
#include <hip/hip_cooperative_groups.h>
#include <hip/hip_runtime.h>
#include <tuple>

#define NELEMS(array) (sizeof(array) / sizeof(array[0]))

namespace cg = cooperative_groups;


template <class T, template <typename> class Op>
const char* functorToString()
{
  if constexpr (std::is_same<Op<T>, cg::plus<T>>::value) {
    return "cg::plus";
  }

  assert(false && "Missing conversion to string for type");
  return "";
}

template <template <typename> class Op, class T, typename... Types>
void compileProgram(hiprtcProgram& prog, const std::tuple<T, Types...>&) {
  std::string expression;
  std::tuple<Types...> remainingTypes;

  expression = std::string("reduceCoopKernel<") +
               functorToString<T, Op>() +
               ", " +
               typeToString<T>() + ">";
  HIPRTC_CHECK(hiprtcAddNameExpression(prog, expression.c_str()));
  compileProgram<Op>(prog, remainingTypes);
}

template <template <typename> class Op, class T = void>
void compileProgram(hiprtcProgram& prog, const std::tuple<>&) {
  size_t logSize;
  hiprtcResult compileResult;
  const char* options[] = {"-DHIP_ENABLE_WARP_SYNC_BUILTINS", "-DHIP_ENABLE_EXTRA_WARP_SYNC_TYPES"};

  compileResult = hiprtcResult{hiprtcCompileProgram(prog, NELEMS(options), options)};
  HIPRTC_CHECK(hiprtcGetProgramLogSize(prog, &logSize));

  if (compileResult != HIPRTC_SUCCESS || logSize > 0) {
    std::string log(logSize, '\0');

    HIPRTC_CHECK(hiprtcGetProgramLog(prog, &log[0]));
    std::cerr << "Runtime compilation failed or contained warnings\n";
    std::cerr << log << '\n';
    REQUIRE(false);
  }
}

template <class T, template <typename> class Op>
void runReduce(hiprtcProgram& prog) {
  using distribution = typename DistributionType<T>::type;

  static constexpr std::array<int, 6> tileSizes = {2, 4, 8, 16, 32, 64};
  unsigned int wavefrontSize = getWarpSize();
  const char* loweredName;
  hipFunction_t kernel;
  hipModule_t module;
  LinearAllocGuard<T> d_input(LinearAllocs::hipMalloc, wavefrontSize * sizeof(T));
  LinearAllocGuard<T> input(LinearAllocs::malloc, d_input.size_bytes());
  LinearAllocGuard<T> d_output(LinearAllocs::hipMalloc, wavefrontSize * sizeof(T) * tileSizes.size());
  LinearAllocGuard<T> output(LinearAllocs::malloc, d_output.size_bytes());
  std::mt19937_64 gen(Catch::rngSeed());
  // for float16, we generate any random unsigned short, but cap the exponent later on
  // to keep it in the range (-8.0..8.0) (just to avoid overflows)
  // On the rest of the types, just use a bigger reduced range of numbers to avoid overflows too
  T a = std::is_same<T, half>::value? std::numeric_limits<unsigned short>::lowest() : -1023;
  T b = std::is_same<T, half>::value? std::numeric_limits<unsigned short>::max() : 1023;
  distribution dist(a, b);

  genRandomBuffers(d_input, input, dist, gen, wavefrontSize);

  struct {
    const T* d_output;
    const T* d_input;
  } args{d_output.ptr(), d_input.ptr()};
  size_t size;
  void* config[] = {HIP_LAUNCH_PARAM_BUFFER_POINTER, &args, HIP_LAUNCH_PARAM_BUFFER_SIZE, &size,
                    HIP_LAUNCH_PARAM_END};
  std::vector<char> code;
  size_t codeSize;
  std::string expression =
      std::string("reduceCoopKernel<") + functorToString<T, Op>() + ", " + typeToString<T>() + ">";
  dim3 grdDim{1u};
  dim3 blkDim{wavefrontSize};
  int numTile = 0;

  HIPRTC_CHECK(hiprtcGetCodeSize(prog, &codeSize));
  code.resize(codeSize);
  HIPRTC_CHECK(hiprtcGetCode(prog, code.data()));
  HIP_CHECK(hipModuleLoadData(&module, code.data()));
  HIPRTC_CHECK(hiprtcGetLoweredName(prog, expression.c_str(), &loweredName));
  HIP_CHECK(hipModuleGetFunction(&kernel, module, loweredName));
  HIP_CHECK(hipModuleLaunchKernel(kernel, grdDim.x, grdDim.y, grdDim.z, blkDim.x, blkDim.y,
                                  blkDim.z, 0, 0, nullptr, config));
  HIP_CHECK(hipModuleUnload(module));
  HIP_CHECK(hipDeviceSynchronize());
  HIP_CHECK(hipMemcpy(output.ptr(), d_output.ptr(), d_output.size_bytes(), hipMemcpyDeviceToHost));

  for (auto tileSize : tileSizes) {
    for (int laneId = 0; laneId < wavefrontSize; laneId++) {
      if (tileSize <= wavefrontSize) {
        Op<T> op;
        unsigned long long mask = ~0ull >> (64 - tileSize);
        T expected;

        mask <<= (((laneId % wavefrontSize) / tileSize) * tileSize);
        expected = calculateExpected(input.host_ptr(), op, mask);
        INFO("Tile: " << tileSize << " laneId: " << laneId);
        REQUIRE(output.host_ptr()[numTile * laneId] == expected);
      }
    }

    numTile++;
  }
}

template <template <typename> class Op, class Type = void>
void runTestReduceForTypes(hiprtcProgram&, const std::tuple<>) {}

template <template <typename> class Op, class T, typename... Types>
void runTestReduceForTypes(hiprtcProgram& prog, const std::tuple<T, Types...>) {
  std::tuple<Types...> remainingTypes;

  runReduce<T, Op>(prog);
  runTestReduceForTypes<Op>(prog, remainingTypes);
}

template <template <typename> class Op, typename... Types>
void runAndCompileTest(const std::tuple<Types...> types) {
  std::string kernelStr;
  hiprtcProgram prog;

  kernelStr = R"(
    using cg = coooperative_groups;

    template <template <typename> class Op, class T, class TileSize = void>
    __global__ void reduceTiles(T& output, const T* input, const std::tuple<> tileSizes) 
    {
    }

    // run reduce for a specific type and for different tile sizes as  a variadic template parameter
    // @output the result, per lane
    template <template <typename> class Op, class T, size_t TileSize, typename... TileSizes>
    __global__ void reduceTiles(T* output, const T* input, const std::tuple<T, TileSizes...>) 
    {
      std::integer_sequence<TyleSizes...> remainingTypes;
      cg::thread_block group = cg::this_thread_block();
      auto tile = cg::tiled_partition<TileSize>(group);
      Op<T> op;
      T accum;

      output[threadIdx.x] = cg::reduce(tile, accum, op);
      reduceTiles(output + warpSize, input, remainingTypes);
    }

    // @output will receive a different result per tile size
    template <template <typename> class Op, class T, size_t WarpSize>
    __global__ void reduceCoopKernel(T* output, const T* input)
    {
      if constexpr (WarpSize <= 32) {
        std::integer_sequence<2, 4, 8, 16, 32> tileSizes;
        reduceTiles<Op, T>(output, tileSizes);
      } else {
        std::integer_sequence<2, 4, 8, 16, 32, 64> tileSizes;
        reduceTiles<Op, T>(input, tileSizes);
      }
    }
  })";

  HIPRTC_CHECK(
      hiprtcCreateProgram(&prog, kernelStr.c_str(), "coop_reduce.hip", 0, nullptr, nullptr));
  compileProgram<Op>(prog, types);
  runTestReduceForTypes<Op>(prog, types);
  HIPRTC_CHECK(hiprtcDestroyProgram(&prog));
}

TEST_CASE("Unit_Rtc_CoopReduce")
{
  const std::tuple<int/*, unsigned int, long long, unsigned long long, float, half, double*/> allTypes;
  //const std::tuple<int, unsigned int, long long, unsigned long long> integralTypes;

  SECTION("add") {
    runAndCompileTest<cooperative_groups::plus>(allTypes);
  }
}
