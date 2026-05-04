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
    return "cooperative_groups::plus";
  } else if constexpr (std::is_same<Op<T>, cg::less<T>>::value) {
    return "cooperative_groups::less";
  } else if constexpr (std::is_same<Op<T>, cg::greater<T>>::value) {
    return "cooperative_groups::greater";
  } else if constexpr (std::is_same<Op<T>, cg::bit_and<T>>::value) {
    return "cooperative_groups::bit_and";
  } else if constexpr (std::is_same<Op<T>, cg::bit_or<T>>::value) {
    return "cooperative_groups::bit_or";
  } else if constexpr (std::is_same<Op<T>, cg::bit_xor<T>>::value) {
    return "cooperative_groups::bit_xor";
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
               typeToString<T>() +
               ", " +
               std::to_string(getWarpSize()) + ">";
  HIPRTC_CHECK(hiprtcAddNameExpression(prog, expression.c_str()));
  compileProgram<Op>(prog, remainingTypes);
}

template <template <typename> class Op, class T = void>
void compileProgram(hiprtcProgram& prog, const std::tuple<>&) {
  size_t logSize;
  hiprtcResult compileResult;
  const char* options[] = {"-DHIP_ENABLE_WARP_SYNC_BUILTINS", "-DHIP_ENABLE_EXTRA_WARP_SYNC_TYPES", "-std=c++17"};

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

  static constexpr std::array<int, 7> tileSizes = {1, 2, 4, 8, 16, 32, 64};
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
  std::vector<const void*> args = { d_output.ptr(), d_input.ptr() };
  std::size_t sizeBytes = args.size() * sizeof(void*);
  void* config[] = {HIP_LAUNCH_PARAM_BUFFER_POINTER, args.data(), HIP_LAUNCH_PARAM_BUFFER_SIZE, &sizeBytes,
                    HIP_LAUNCH_PARAM_END};
  std::vector<char> code;
  size_t codeSize;
  std::string expression =
      std::string("reduceCoopKernel<") +
      functorToString<T, Op>() +
      ", " +
      typeToString<T>() +
      ", " +
      std::to_string(wavefrontSize) + ">";
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

  INFO("Type: " << typeToString<T>());
  for (auto tileSize : tileSizes) {
    UNSCOPED_INFO("Tile size: " << tileSize);

    for (unsigned int laneId = 0; laneId < wavefrontSize; laneId++) {
      unsigned long long mask = ~0ull >> (64 - tileSize);

      mask <<= (((laneId % wavefrontSize) / tileSize) * tileSize);

      if (tileSize <= wavefrontSize) {
        std::string inputStr;

        if constexpr (!std::is_same<T, half>::value) {
          inputStr = std::string(" input: ") + std::to_string(input.host_ptr()[laneId]);
        }

        UNSCOPED_INFO("laneId: " << laneId << " mask: " << mask << inputStr);
      }
    }

    for (unsigned int laneId = 0; laneId < wavefrontSize; laneId++) {
      unsigned long long mask = ~0ull >> (64 - tileSize);

      mask <<= ((laneId % wavefrontSize) / tileSize) * tileSize;

      if (tileSize <= wavefrontSize) {
        T expected;
        Op<T> op;
        expected = calculateExpected(input.host_ptr(), op, mask);
        REQUIRE(output.host_ptr()[numTile * wavefrontSize + laneId] == expected);
      }
    }

    numTile++;
  }
}

template <template <typename> class Op, class Type = void>
void runCoopTestReduceForTypes(hiprtcProgram&, const std::tuple<>) {}

template <template <typename> class Op, class T, typename... Types>
void runCoopTestReduceForTypes(hiprtcProgram& prog, const std::tuple<T, Types...>) {
  std::tuple<Types...> remainingTypes;

  runReduce<T, Op>(prog);
  runCoopTestReduceForTypes<Op>(prog, remainingTypes);
}

template <template <typename> class Op, typename... Types>
void runAndCompileTest(const std::tuple<Types...> types) {
  std::string kernelStr;
  hiprtcProgram prog;

  kernelStr = R"(
    namespace cg = cooperative_groups;

    template <template <typename> class Op, class T>
    __device__ void reduceTiles(T*, const T*, const __hip_internal::index_sequence<>) 
    {
    }

    // run reduce for a specific type and for different tile sizes as a variadic template parameter
    // @output the result, per lane
    template <template <typename> class Op, class T, size_t TileSize, size_t... TileSizes>
    __device__ void reduceTiles(T* output, const T* input, const __hip_internal::index_sequence<TileSize, TileSizes...>) 
    {
      const __hip_internal::index_sequence<TileSizes...> remainingTiles;
      cg::thread_block group = cg::this_thread_block();
      auto tile = cg::tiled_partition<TileSize>(group);
      Op<T> op;

      output[threadIdx.x] = cg::reduce(tile, input[threadIdx.x], op);
      reduceTiles<Op, T>(output + warpSize, input, remainingTiles);
    }

    // @output will receive a different result per tile size
    template <template <typename> class Op, class T, int WarpSize>
    __global__ void reduceCoopKernel(T* output, const T* input)
    {
      if constexpr (WarpSize <= 32) {
        __hip_internal::index_sequence<1, 2, 4, 8, 16, 32> tileSizes;
        reduceTiles<Op, T>(output, input, tileSizes);
      } else {
        __hip_internal::index_sequence<1, 2, 4, 8, 16, 32, 64> tileSizes;
        reduceTiles<Op, T>(output, input, tileSizes);
      }
    }
  )";

  HIPRTC_CHECK(
      hiprtcCreateProgram(&prog, kernelStr.c_str(), "coop_reduce.hip", 0, nullptr, nullptr));
  compileProgram<Op>(prog, types);
  runCoopTestReduceForTypes<Op>(prog, types);
  HIPRTC_CHECK(hiprtcDestroyProgram(&prog));
}

TEST_CASE("Unit_Rtc_CoopReduce")
{
  const std::tuple<int, unsigned int, long long, unsigned long long, float, half, double> allTypes;
  const std::tuple<int, unsigned int, long long, unsigned long long> integralTypes;

  SECTION("add") {
    runAndCompileTest<cooperative_groups::plus>(allTypes);
  }

  SECTION("less") {
    runAndCompileTest<cooperative_groups::less>(allTypes);
  }

  SECTION("greater") {
    runAndCompileTest<cooperative_groups::greater>(allTypes);
  }

  SECTION("and") {
    runAndCompileTest<cooperative_groups::bit_and>(integralTypes);
  }

  SECTION("or") {
    runAndCompileTest<cooperative_groups::bit_or>(integralTypes);
  }

  SECTION("xor") {
    runAndCompileTest<cooperative_groups::bit_xor>(integralTypes);
  }
}
