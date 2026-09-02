#!/usr/bin/env python3
import os
import sys
import subprocess
from dataclasses import dataclass
import shutil

# Order of colls, redops, tys, protos, algos must match src/include/device.h
# The empty entries are for collectives like Gather, Scatter, etc.
all_colls     = ["Broadcast", "Reduce", "AllGather", "ReduceScatter", "AllReduce", "SendRecv", "", "", "", "", "", "AlltoAllPivot", "AlltoAllGda", "AlltoAllvGda", "AllGatherV"]
all_redops    = ["Sum","Prod","MinMax","PreMulSum","SumPostDiv"]
all_tys       = ["i8","u8","i32","u32","i64","u64","f16","f32","f64","bf16","f8e4m3","f8e5m2"]
all_protos    = ["LL","LL128","SIMPLE"]
all_algos     = ["TREE","RING", "", "", "", "", "PAT"]
all_accs      = ["0", "1"]
all_pipelines = ["0", "1"]
all_unrolls   = ["1", "2", "4", "8", "16", "32"]
# User-buffer registration mode (compile-time UserRegMode template parameter):
#   "0" = runtime / not-applicable (single kernel, current behavior)
#   "1" = registered user buffer   (LL128 Direct path bypasses cache)
#   "2" = non-registered user buffer (LL128 Direct path uses plain/non-temporal)
# Only LL128 for the collectives in `ll128_reg_variant_colls` is split into the
# "1"/"2" pair; everything else stays a single "0" kernel.
all_regs      = ["0", "1", "2"]

all_params = [all_colls, all_algos, all_protos, all_redops, all_tys, all_accs, all_pipelines, all_unrolls, all_regs]

# Collectives whose LL128 kernels are specialized into separate registered /
# non-registered variants (their LL128 user-buffer path is Direct=1, so the
# compile-time split removes a runtime branch and its dead code). Keep this in
# sync with `ncclDevFuncIsLL128RegVariant()` in src/include/device.h and the
# selection logic in src/enqueue.cc.
ll128_reg_variant_colls = {"AllReduce", "AllGather", "Broadcast"}

def reg_values_of(coll, proto):
  if proto == "LL128" and coll in ll128_reg_variant_colls:
    return ["1", "2"]
  # SendRecv is generated as two latency-protocol kernel variants, selected on the
  # host by ncclDevFuncId_P2p(useLL128) (see src/enqueue.cc / src/include/device.h):
  #   reg "0" = legacy LL latency path (built on every arch; the default)
  #   reg "1" = LL128 latency path (built for gfx942/gfx950 only; used when
  #             NCCL_ALLOC_P2P_NET_LL_BUFFERS=1). The reg value is threaded into the
  #             SendRecv RunWorkBatch specialization as UserRegMode to pick LL vs LL128.
  if coll == "SendRecv":
    return ["0", "1"]
  return ["0"]

################################################################################
# The first command line argument is the path to the directory to generate and
# populate.

gensrc = sys.argv[1]

if os.path.exists(gensrc):
  for name in os.listdir(gensrc):
    path = os.path.join(gensrc, name)
    if os.path.isfile(path):
      os.remove(path)
    elif os.path.isdir(path):
      shutil.rmtree(path)
else:
  os.makedirs(gensrc)

################################################################################
# The command line argument is used as a regex to filter the functions
# which make it into librccl. This is helpful for reducing the binary when
# developing device code. The regex supports non-space containing globs '*',
# and union 'a|b'. The string representing the function has the form:
#
# <coll> <algo> <proto> <redop> <type>
#
# The possible values for redop, type, algo, proto can be found in the all_<foo>
# lists at the top of this file.
#
# Example use-cases:
#
# # Only send/recv:
# make ONLY_FUNCS="SendRecv"
#
# # Only AllReduce and Reduce
# make ONLY_FUNCS="AllReduce|Reduce"
#
# # Only non-reductions:
# make ONLY_FUNCS="AllGather * *|Broadcast * *|SendRecv"
#
# # Only AllReduce Sum int32_t (but all algos, protos)
# make ONLY_FUNCS="AllReduce * * Sum i32"
#
# # Only AllReduce RING Max float (but all protos)
# make ONLY_FUNCS="AllReduce RING * Max f32"
#
# # AllReduce TREE LL128 Prod rccl_bfloat16
# make ONLY_FUNCS="AllReduce TREE LL128 Prod bf16"
#
# # AllReduce RING SIMPLE and ReduceScatter RING LL float (but all redops, types for AllReduce and all redops for ReduceScatter)
# make ONLY_FUNCS="AllReduce RING SIMPLE * *|ReduceScatter RING LL * f32"
#                         --- or ---
# make ONLY_FUNCS="AllReduce RING SIMPLE|ReduceScatter RING LL * f32"
# make ONLY_FUNCS="AllReduce RING/TREE LL/SIMPLE Sum/MinMax i8/u8/f16/f32/f64/bf16/f8e4m3/f8e5m2|AllGather RING LL/SIMPLE Sum i8|AlltoAllPivot RING SIMPLE Sum i8|Broadcast RING LL/SIMPLE Sum i8|Reduce RING LL/SIMPLE Sum/MinMax i8/u8/f16/f32/f64/bf16/f8e4m3/f8e5m2|ReduceScatter RING LL/SIMPLE Sum/MinMax i8/u8/f16/f32/f64/bf16/f8e4m3/f8e5m2|SendRecv RING SIMPLE Sum i8"

################################################################################
# NCCL_EXACT_KERNEL_NAMES selects upstream's exact kernel naming, which only
# applies to the representative-ncclDevKernel mapping. RCCL emits one
# ncclDevFunc_* per primary variant plus a per-function specialized kernel, so
# every variant already has a distinct name and the option has no effect here.
def str_to_bool(s):
  s = s.lower()
  if s in ("1", "true", "on", "yes"):
    return True
  if s in ("", "0", "false", "off", "no"):
    return False
  raise ValueError("Invalid boolean value: " + s)
exact_kernel_names = str_to_bool(os.environ.get("NCCL_EXACT_KERNEL_NAMES", "0"))

# Paste all non-None arguments together with `sep`.
def paste(sep, *args):
  return sep.join(x for x in args if x is not None)

is_ifc             = 1 if sys.argv[2] == "ON" else 0
is_local_arch_only = 1 if sys.argv[4] == "ON" else 0
is_rocshmem        = 1 if sys.argv[5] == "ON" else 0

func_pattern = sys.argv[6:7]

if func_pattern and func_pattern[0]:
  func_pattern = func_pattern[0]
else:
  # GDA (rocSHMEM-based) kernels only when rocshmem build requested
  if is_rocshmem:
    func_pattern = "AllGather|AllGatherV|AllReduce|AlltoAllPivot|AlltoAllGda|AlltoAllvGda|Broadcast|Reduce|ReduceScatter|SendRecv"
  else:
    func_pattern = "AllGather|AllGatherV|AllReduce|AlltoAllPivot|Broadcast|Reduce|ReduceScatter|SendRecv"

################################################################################

algos_of_coll = {
  "AllGather":             ["RING", "PAT"],
  "AllGatherV":            ["RING"],
  "AllReduce":             ["RING", "TREE"],
  "AlltoAllPivot":         ["RING"],
  "AlltoAllGda":           ["RING"],
  "AlltoAllvGda":          ["RING"],
  "Broadcast":             ["RING"],
  "Reduce":                ["RING"],
  "ReduceScatter":         ["RING", "PAT"],
  "SendRecv":              ["RING"]
}

protos_of_coll = {
  "AllGather":              all_protos,
  "AllGatherV":             all_protos,
  "AllReduce":              all_protos,
  "AlltoAllPivot":          ["SIMPLE"],
  "AlltoAllGda":            ["SIMPLE"],
  "AlltoAllvGda":           ["SIMPLE"],
  "Broadcast":              all_protos,
  "Reduce":                 all_protos,
  "ReduceScatter":          all_protos,
  "SendRecv":               ["SIMPLE"]
}

redops_of_coll = {
  "AllGather":            ["Sum"],
  "AllGatherV":           ["Sum"],
  "AllReduce":            all_redops,
  "AlltoAllPivot":        ["Sum"],
  "AlltoAllGda":          ["Sum"],
  "AlltoAllvGda":         ["Sum"],
  "Broadcast":            ["Sum"],
  "Reduce":               all_redops,
  "ReduceScatter":        all_redops,
  "SendRecv":             ["Sum"]
}

tys_of_coll = {
  "AllGather":             ["i8"],
  "AllGatherV":            ["i8"],
  "AllReduce":             all_tys,
  "AlltoAllPivot":         ["i8"],
  "AlltoAllGda":           ["i8"],
  "AlltoAllvGda":          ["i8"],
  "Broadcast":             ["i8"],
  "Reduce":                all_tys,
  "ReduceScatter":         all_tys,
  "SendRecv":              ["i8"]
}

acc_of_coll = {
  "AllGather":             ["0"],
  "AllGatherV":            ["0"],
  "AllReduce":             all_accs,
  "AlltoAllPivot":         ["0"],
  "AlltoAllGda":           ["0"],
  "AlltoAllvGda":          ["0"],
  "Broadcast":             ["0"],
  "Reduce":                ["0"],
  "ReduceScatter":         ["0"],
  "SendRecv":              ["0"]
}

pipelines_of_coll = {
  "AllGather":             ["0"],
  "AllGatherV":            ["0"],
  "AllReduce":             all_pipelines,
  "AlltoAllPivot":         ["0"],
  "AlltoAllGda":           ["0"],
  "AlltoAllvGda":          ["0"],
  "Broadcast":             ["0"],
  "Reduce":                all_pipelines,
  "ReduceScatter":         all_pipelines,
  "SendRecv":              ["0"]
}
pipelined_types = ["bf16"]

coll_camel_to_lower = {
  "AllGather":             "all_gather",
  "AllGatherV":            "all_gather_v",
  "AllReduce":             "all_reduce",
  "AlltoAllPivot":         "alltoall_pivot",
  "AlltoAllGda":           "alltoall_gda",
  "AlltoAllvGda":          "alltoallv_gda",
  "Broadcast":             "broadcast",
  "Reduce":                "reduce",
  "ReduceScatter":         "reduce_scatter",
  "SendRecv":              "sendrecv"
}
coll_lower_to_camel = {coll_camel_to_lower[x]: x for x in coll_camel_to_lower}

################################################################################
@dataclass(frozen=True)
class Fn:
  coll: str
  algo: str
  proto: str
  redop: str
  ty: str
  acc: str
  pipeline: str
  unroll: str
  reg: str

  def __iter__(self):
    return iter((self.coll, self.algo, self.proto, self.redop, self.ty, self.acc, self.pipeline, self.unroll, self.reg))

def calc_unroll_and_pipeline_for_local_arch():

  if not is_local_arch_only:
    return (all_unrolls, all_pipelines)

  rocminfo_path = os.environ.get('ROCM_PATH') + "/bin/rocminfo"

  res = subprocess.run([rocminfo_path], stdout=subprocess.PIPE, universal_newlines=True)
  rocminfo_output = res.stdout

  # Parse rocminfo binary output
  gfx_targets = {}
  curr_name = None
  for line in rocminfo_output.splitlines():
    line = line.strip()

    if line.startswith("Name:"):
      name = line.split(':')[-1].strip()
      if "gfx" in name:
        curr_name = name
    if line.startswith("Compute Unit:") and curr_name:
      cu_count = int(line.split(':')[-1].strip())
      gfx_targets[(curr_name, cu_count)] = None
      curr_name = None

  # We want to remove duplicates but cannot use a dictionary since same gfx name can have different cu counts
  # Use (gfx_name, cu_count) as key for dictionary and convert it to list here
  gfx_targets = list(gfx_targets.keys())
  
  # Homogeneous system is required to build for only 1 variant of unroll factor (except for gfx950 and gfx1250)
  if len(gfx_targets) == 1:
    gfx_name, cu_count = gfx_targets[0]
    if "gfx950" == gfx_name:
      return (["1", "2"], ["0"])  # Disable pipelining for gfx950
    elif "gfx908" == gfx_name or ("gfx942" == gfx_name and cu_count > 80):
      return (["2"], all_pipelines)
    elif "gfx1250" == gfx_name:
      # gfx1250 (MI450) benefits from larger unrolls; Unroll 8 required for FP8 launch;
      # 32 is the default (commSetUnrollFactor).
      return (["8", "16", "32"], all_pipelines)
    else:
      return (["4"], all_pipelines)
  else:
    return (all_unrolls, all_pipelines)

# if building for local arch only, we only need to build for 1 variant of unroll for most gfx targets,
# except for gfx950. For gfx950, we also disable pipelining.
local_unroll, local_pipeline = calc_unroll_and_pipeline_for_local_arch()

# rocSHMEM/GDA-based collectives: only generated when ENABLE_ROCSHMEM build is requested
gda_colls = {"AlltoAllGda", "AlltoAllvGda"}

# Helper function to check if the conditions for the collective is being met
def func_validate(coll, algo, proto, redop, ty, acc,  pipeline, unroll, reg):
  if redop == "SumPostDiv" and ty[0] not in ("i","u"):
    return False
  if coll == "" or algo == "":
    return False
  # The LL128 SendRecv variant (reg=1) is only built/activated on gfx942/gfx950, which never
  # use the gfx1250-only unroll factors (8/16/32). Don't emit those nonsensical variants: the
  # device linker would skip compiling them for gfx942 while the dispatch table still expected
  # them (undefined-symbol link error).
  if coll == "SendRecv" and reg == "1" and unroll in ("8", "16", "32"):
    return False
  if not is_rocshmem and coll in gda_colls:
    return False
  if (algo not in algos_of_coll[coll] or
      proto not in protos_of_coll[coll] or
      redop not in redops_of_coll[coll] or
      ty not in tys_of_coll[coll] or
      acc not in acc_of_coll[coll] or
      pipeline not in pipelines_of_coll[coll] or (pipeline in ["1"] and ty not in pipelined_types) or
      pipeline not in local_pipeline or
      unroll not in local_unroll or
      reg not in reg_values_of(coll, proto)):
    return False
  return True

# A recursive helper to generate collective functions based on the input given
def func_filter(function_params, current_idx, item_list=None):
  if item_list is None:
    item_list = []

  # Check if current_idx exceeds the max depth
  if current_idx < len(all_params):
    # Current element is the config parameter
    current_element = function_params[current_idx]

    # If the paramter is equal to '*', include all possible cases for it
    if current_element == "*":
      # all_params list must be in the same order as function_params --> <coll> <algo> <proto> <redop> <type>
      # Get the current list from all_params
      current_list = all_params[current_idx]

      # Iterate over the items int the current_list
      for item in current_list:
        # Add item to item_list which will be used in the inner most loop
        item_list.append(item)
        yield from func_filter(function_params, current_idx+1, item_list)

        # For each loop layer remove the last element in item_list
        item_list.pop()
    else:
      # Check if the current element is recognized
      elements = current_element.split("/")
      current_param = all_params[current_idx]

      # Iterate over the elements in the elements list
      for item in elements:
        if item not in current_param:
          raise ValueError(f"Error: {item} is unrecognized or does not belong to this category {current_param}.")

      for item in elements:
        item_list.append(item)
        yield from func_filter(function_params, current_idx+1, item_list)

        # For each loop layer remove the last element in item_list
        item_list.pop()
  else:
    coll, algo, proto, redop, ty, acc, pipeline, unroll, reg = item_list
    if func_validate(coll, algo, proto, redop, ty, acc, pipeline, unroll, reg):
      yield(coll, algo, proto, redop, ty, acc, pipeline, unroll, reg)


# Parse ONLY_FUNCS input and feed it to func_filter
def parse_input(func_pattern):
  input_list = sorted(func_pattern.split("|"))

  for input in input_list:
    function_params = input.split()
    params_length = len(function_params)

    # If a parameter is missing, append '*'
    while params_length < len(all_params):
      function_params.append("*")
      params_length += 1

    # Filter functions/kernels based on input
    yield from func_filter(function_params, 0)

# Maps functions to the chosen representative for the equivalence class it
# belongs to. For instance (sum, signed int) maps to (sum, unsigned int).
def equivalent_primary(coll, algo, proto, redop, ty, acc, pipeline, unroll, reg):
  if coll in ("AllReduce", "Reduce", "ReduceScatter"):
    # map signed integer sum/prod to unsigned
    if redop in ("Sum","Prod","PreMulSum","SumPostDiv") and ty[0]=="i":
      ty = "u"+ty[1:]
    # map signed integer min/max to unsigned for non-NVLS
    elif redop=="MinMax" and ty[0]=="i" and ("NVLS" not in algo):
      ty = "u"+ty[1:]
    # map pipelined to non-pipelined for LL/LL128 to avoid extra device codegen
    if (pipeline != "0" and proto != "SIMPLE"):
      pipeline = "0"

  return (coll, algo, proto, redop, ty, acc, pipeline, unroll, reg)

# Order rows are enumerated must match formula of `ncclDevFuncId()`:
# outermost loop should be for unroll factor; refer to host_table section
def enumerate_func_rows():
  for unroll in local_unroll:
    for coll in all_colls:
      for algo in all_algos:
        for proto in all_protos:
          for redop in all_redops:
            for ty in all_tys:
              for acc in all_accs:
                for pipeline in local_pipeline:
                  for reg in all_regs:
                    if func_validate(coll, algo, proto, redop, ty, acc, pipeline, unroll, reg):
                      yield (coll, algo, proto, redop, ty, acc, pipeline, unroll, reg)

# Sort the hashmap based on custom key <coll> <algo> <proto> <redop> <ty>
def custom_sort_key(fn: Fn):
    return (
        local_unroll.index(fn.unroll),
        all_colls.index(fn.coll),
        all_algos.index(fn.algo),
        all_protos.index(fn.proto),
        all_redops.index(fn.redop),
        all_tys.index(fn.ty),
        all_accs.index(fn.acc),
        local_pipeline.index(fn.pipeline),
        all_regs.index(fn.reg)
    )

def get_arch_guard(fn):
  cond = None

  if fn.coll == "SendRecv" and fn.reg == "1":
      # LL128 SendRecv latency kernel: only build (and only activate) on gfx942/gfx950.
      # Every other arch keeps the legacy LL kernel (reg "0"), which has no guard.
      cond = "(defined(__gfx942__) || defined(__gfx950__)) && defined(ENABLE_LL128)"
  elif fn.unroll in ("8", "16", "32"):
      cond = "defined(__gfx1250__)"
  elif fn.proto == "LL128" and fn.acc == "1":
      cond = "(defined(__gfx942__) || defined(__gfx950__) || defined(__gfx1250__)) && defined(ENABLE_LL128)"
  elif fn.proto == "LL128":
      cond = "(defined(__gfx90a__) || defined(__gfx942__) || defined(__gfx950__) || defined(__gfx1250__)) && defined(ENABLE_LL128)"
  elif fn.acc == "1":
      cond = "defined(__gfx942__) || defined(__gfx950__) || defined(__gfx1250__)"
  return cond

# Build the mangled function symbol suffix. The user-buffer registration mode is
# only appended when it is meaningful (LL128 reg-variant kernels, reg == "1"/"2")
# so that all other kernels keep their historical names.
def fn_sym(fn):
  reg = fn.reg if fn.reg != "0" else None
  return paste("_", fn.coll, fn.algo, fn.proto, fn.redop, fn.ty, fn.acc, fn.pipeline, fn.unroll, reg)

################################################################################

# Corresponds to ncclDevFuncRowToId[]
func_rows = [Fn(*fn) for fn in enumerate_func_rows()]

# Corresponds to ncclDevFuncTable[]
primary_funcs = sorted(
    {Fn(*equivalent_primary(*fn)) for fn in parse_input(func_pattern)}, key=custom_sort_key
)

# primary_to_index[primary_funcs[i]] == i
primary_to_index = {fn: i for i, fn in enumerate(primary_funcs)}
primary_to_index = {fn: primary_to_index.get(Fn(*fn), -1) for fn in func_rows}

################################################################################

# Generate <gensrc>/device_table.h
with open(os.path.join(gensrc, "device_table.h"), "w") as f:
  print("-- Generating %s" % os.path.join(gensrc, "device_table.h"))
  out = f.write

  # Plain forward declarations; noinline (device-linker only) is controlled
  # solely by DEFINE_ncclDevFunc in common.h.
  for fn in primary_funcs:
    sym = "ncclDevFunc_" + fn_sym(fn)
    guard = get_arch_guard(fn)
    if guard:
      out("#if %s\n__device__ void %s();\n#endif\n" % (guard, sym))
    else:
      out("__device__ void %s();\n" % sym)
  out("\n")

  index = {val: None for val in all_unrolls}

  # Function-pointer table. Only the builds that dispatch through it at RUNTIME
  # emit it: the device-linker build and the legacy indirect-function-call build.
  out("#if defined(USE_INDIRECT_FUNCTION_CALL) || defined(RCCL_DEVICE_LINKER)\n")
  out("typedef void(*ncclDevFuncPtr_t)();\n\n")
  for unroll in all_unrolls:
    index[unroll] = 0
    out("static __device__ ncclDevFuncPtr_t const ncclDevFuncTable_%s[] = {\n" % unroll)
    for fn in primary_funcs:
      if fn.unroll != unroll: continue
      sym = "ncclDevFunc_" + fn_sym(fn)
      guard = get_arch_guard(fn)
      if guard:
        out("#if %s\n/*%4d*/ %s,\n#else\n/*%4d*/ nullptr,\n#endif\n" % (guard, index[unroll], sym, index[unroll]))
      else:
        out("/*%4d*/ %s,\n" % (index[unroll], sym))
      index[unroll] += 1
    out("nullptr};\n")
    out("\n")
  out("#endif // USE_INDIRECT_FUNCTION_CALL || RCCL_DEVICE_LINKER\n\n")

  if not is_ifc:
    # Pure-RDC dispatch: a compile-time binary search whose leaves call each
    # ncclDevFunc_* DIRECTLY BY NAME -- no function-pointer table is referenced,
    # so nothing is address-taken.
    out("#if !defined(USE_INDIRECT_FUNCTION_CALL) && !defined(RCCL_DEVICE_LINKER)\n")
    for unroll in all_unrolls:
      out(f"template<unsigned short f, unsigned short l>\n"
          f"struct Caller{unroll} {{\n"
          "  static __forceinline__ __device__\n"
          f"  void call{unroll}(unsigned short funcIndex) noexcept {{\n"
          "    constexpr unsigned short m = f + (l - f) / 2;\n"
          f"    return (funcIndex < m)\n"
          f"      ? Caller{unroll}<f, m>::call{unroll}(funcIndex)\n"
          f"      : Caller{unroll}<m, l>::call{unroll}(funcIndex);\n"
          "  }\n"
          "};\n\n")
      unroll_fns = [fn for fn in primary_funcs if fn.unroll == unroll]
      for i, fn in enumerate(unroll_fns):
        # Must match the symbol emitted for the forward declarations / table /
        # DEFINE_ncclDevFunc above: fn_sym() omits the reg suffix when reg=="0",
        # so calling by name here must use it too (plain *fn would append the
        # "_0" reg field and reference an undeclared symbol, breaking the
        # pure-RDC / --no-device-linker build).
        sym = "ncclDevFunc_" + fn_sym(fn)
        guard = get_arch_guard(fn)
        spec = f"template<> struct Caller{unroll}<{i}, {i+1}> {{ static __forceinline__ __device__ void call{unroll}(unsigned short) noexcept"
        if guard:
          out(f"#if {guard}\n")
          out(f"{spec} {{ {sym}(); }} }};\n")
          out("#else\n")
          # Arch-guarded-out slot: the function does not exist for this arch.
          # Trap instead of a silent no-op so an out-of-range/inconsistent funcId
          # fails fast, matching the nullptr entries of the function-pointer table.
          out(f"{spec} {{ __builtin_trap(); }} }};\n")
          out("#endif\n")
        else:
          out(f"{spec} {{ {sym}(); }} }};\n")
      out("\n")
      out(f"__forceinline__ __device__ void NCCL_CALL_FUNCTIONS_{unroll}(unsigned short funcIndex) noexcept {{\n")
      out(f"  Caller{unroll}<0, {len(unroll_fns)}>::call{unroll}(funcIndex);\n")
      out("}\n\n")
    out("#endif // !USE_INDIRECT_FUNCTION_CALL && !RCCL_DEVICE_LINKER\n")

# Generate <gensrc>/host_table.cpp
with open(os.path.join(gensrc, "host_table.cpp"), "w") as f:
  print("-- Generating %s" % os.path.join(gensrc, "host_table.cpp"))

  out = f.write
  out('#include "device.h"\n')
  out("\n")
  out("// The key for the ncclDevFuncNameToId map is a 64-bit unsigned integer.\n")
  out("// Each field (coll, algo, proto, redop, ty, acc, pipeline) is packed into 4 bits,\n")
  out("// This allows up to 16 unique values per field. The layout is:\n")
  out("//   bits  0-3:   coll index\n")
  out("//   bits  4-7:   algo index\n")
  out("//   bits  8-11:  proto index\n")
  out("//   bits 12-15:  redop index\n")
  out("//   bits 16-19:  ty index\n")
  out("//   bits 20-23:  accumulator index\n")
  out("//   bits 24-27:  pipeline index\n")
  out("//   bits 28-31:  user-buffer registration mode index (LL128 reg/noreg)\n")
  out("#include <unordered_map>\n")
  out("std::unordered_map<uint64_t, int> ncclDevFuncNameToId = {\n")

  # host_table entries map device functions based on collective, algorithm, protocol, redop, and datatype
  # For GPU targets that support multiple unrolls, e.g., gfx950
  # (or) for non-local builds, only a single set of functions are needed in the host_table.
  for fn in func_rows[:len(func_rows)//len(local_unroll)]:
    fn_id = -1
    if fn is not None:
      guard = get_arch_guard(fn)
      fn_id = primary_to_index[Fn(*equivalent_primary(*fn))]
      comment = " // " + paste(" ", *fn)
      # Build the function signature string: "<coll> <algo> <proto> <redop> <ty>"
      # get parts indexes in order (coll, algo, proto, redop, ty, acc, pipeline, unroll)
      coll_idx = all_colls.index(fn.coll)
      algo_idx = all_algos.index(fn.algo)
      proto_idx = all_protos.index(fn.proto)
      redop_idx = all_redops.index(fn.redop)
      ty_idx = all_tys.index(fn.ty)
      acc_idx = all_accs.index(fn.acc)
      pipeline_idx = all_pipelines.index(fn.pipeline)
      reg_idx = all_regs.index(fn.reg)
      # Assert that 4 bits (16 values) is enough to map all_colls, all_algos, etc.
      assert len(all_colls) <= 16, "Error: all_colls has more than 16 values, which exceeds 4-bit capacity."
      assert len(all_algos) <= 16, "Error: all_algos has more than 16 values, which exceeds 4-bit capacity."
      assert len(all_protos) <= 16, "Error: all_protos has more than 16 values, which exceeds 4-bit capacity."
      assert len(all_redops) <= 16, "Error: all_redops has more than 16 values, which exceeds 4-bit capacity."
      assert len(all_tys) <= 16, "Error: all_tys has more than 16 values, which exceeds 4-bit capacity."
      assert len(all_accs) <= 16, "Error: all_accs has more than 16 values, which exceeds 4-bit capacity."
      assert len(all_pipelines) <= 16, "Error: all_pipelines has more than 16 values, which exceeds 4-bit capacity."
      assert len(all_regs) <= 16, "Error: all_regs has more than 16 values, which exceeds 4-bit capacity."
      # Create a 64-bit unsigned integer key and pack the indices into 4 bits each
      key = (
        (coll_idx & 0xF)
        | ((algo_idx & 0xF) << 4)
        | ((proto_idx & 0xF) << 8)
        | ((redop_idx & 0xF) << 12)
        | ((ty_idx & 0xF) << 16)
        | ((acc_idx & 0xF) << 20)
        | ((pipeline_idx & 0xF) << 24)
        | ((reg_idx & 0xF) << 28)
      )
      if fn.coll == "Broadcast":
        key = ((coll_idx & 0x3F) | ((proto_idx & 0x3F) << 8) | ((reg_idx & 0xF) << 28))
      if fn.coll == "SendRecv":
        # SendRecv has two latency-protocol variants distinguished by reg (0=LL, 1=LL128).
        # reg=0 keeps the historical coll-only key for backward compatibility.
        key = ((coll_idx & 0x3F) | ((reg_idx & 0xF) << 28))
      if fn.coll in ["AlltoAllPivot", "AlltoAllGda", "AlltoAllvGda"]:
        key = ((coll_idx & 0x3F))
      
      out(f'  {{{key}, {fn_id}}}, {comment}\n')
  out("};\n")

  # Which unroll-factor tables were actually generated for this build. The host
  # (commSetUnrollFactor) uses this to reject an RCCL_UNROLL_FACTOR that maps to
  # an empty ncclDevFuncTable_* / NCCL_CALL_FUNCTIONS_* slot, which would
  # otherwise dispatch to a nullptr and segfault on the device.
  out("\n")
  out("// Indexed by unroll-factor enum (NCCL_UNROLL_1 .. NCCL_UNROLL_32).\n")
  out("bool const ncclDevFuncUnrollGenerated[NCCL_NUM_UNROLLS] = {\n")
  for u in all_unrolls:
    out("  %s, // unroll %s\n" % ("true" if u in local_unroll else "false", u))
  out("};\n")

# Maps to .cu filename which implements this func. The only constraint is that
# "coll" is reflected in the name: formally that no two funcs having different
# coll's map to the same filename.
def impl_filename(coll, algo, proto, redop, ty, acc, pipeline, unroll, reg):
  return "%s.cpp" % paste("_", coll_camel_to_lower[coll], redop and redop.lower(), ty)

# Partition the functions and kernels to the .cu filenames. The partition is
# a dictionary mapping filename to (coll, func-tuple list)
def partition_by_name(fns):
  ans = {}
  for fn in fns:
    name = impl_filename(*fn)
    coll = fn.coll
    if name not in ans:
      ans[name] = (coll, [])
    ans[name][1].append(fn)
  return ans

name_to_funcs = partition_by_name(fn for fn in primary_funcs if fn.coll !="Nop")

redop_to_cxx = {
  None: "FuncCopy",
  "Sum": "FuncSum",
  "Prod": "FuncProd",
  "MinMax": "FuncMinMax",
  "PreMulSum": "FuncPreMulSum",
  "SumPostDiv": "FuncSumPostDiv"
}

ty_to_cxx = {
  None: "int8_t",
  "i8": "int8_t",
  "u8": "uint8_t",
  "i32": "int32_t",
  "u32": "uint32_t",
  "i64": "int64_t",
  "u64": "uint64_t",
  "f16": "half",
  "f32": "float",
  "f64": "double",
  "bf16": "hip_bfloat16",
  "f8e4m3":  "rccl_float8",
  "f8e5m2": "rccl_bfloat8"
}

# Generate each <gensrc>/<impl>.cpp:
for name in name_to_funcs.keys():
  (coll, fns) = name_to_funcs[name]
  with open(os.path.join(gensrc, name), "w") as f:
    print("-- Generating %s" % os.path.join(gensrc, name))

    out = f.write
    out(
      '#include "device/common.h"\n'
      '#include "{lower_coll}.h"\n'
      .format(lower_coll=coll_camel_to_lower[coll])
    )

    for fn in fns:
      sym = fn_sym(fn)
      guard = get_arch_guard(fn)
      if guard:
        out("#if %s\n" % guard)
      out(
        "DEFINE_ncclDevFunc({sym}, ncclFunc{coll}, {redop_cxx}, {ty_cxx}, NCCL_ALGO_{algo}, NCCL_PROTO_{proto}, {acc}, {pipeline}, {unroll}, {reg})\n"
        .format(sym=sym, coll=fn.coll, redop_cxx=redop_to_cxx[fn.redop], ty_cxx=ty_to_cxx[fn.ty],
                algo=(fn.algo or "RING"), proto=(fn.proto or "SIMPLE"), acc=fn.acc, pipeline=fn.pipeline, unroll=fn.unroll, reg=fn.reg)
      )
      if guard: 
        out("#endif\n")

################################################################################
# Generate per-function specialized kernel .cpp files for the parallel build.
# Each file contains one device function + a kernel wrapper that calls it,
# enabling the compiler to optimize the device function in kernel context
# (LDS allocation, barriers, etc.) while keeping it as a separate linkable symbol.

specialized_dir = os.path.join(gensrc, "specialized")
if not os.path.exists(specialized_dir):
  os.makedirs(specialized_dir)
else:
  for name in os.listdir(specialized_dir):
    os.remove(os.path.join(specialized_dir, name))

specialized_filelist = []
for fn in primary_funcs:
  if fn.coll == "Nop":
    continue
  sym = fn_sym(fn)
  func_name = "ncclDevFunc_" + sym
  lower_coll = coll_camel_to_lower[fn.coll]
  guard = get_arch_guard(fn)

  filename = "specialized_%s.cpp" % sym.lower()
  filepath = os.path.join(specialized_dir, filename)
  specialized_filelist.append((filename, func_name, guard, fn))

  with open(filepath, "w") as f:
    out = f.write
    out('#include "device/common.h"\n')
    out('#include "%s.h"\n\n' % lower_coll)
    if guard:
      out("#if %s\n" % guard)
    out(
      "DEFINE_ncclDevFunc({sym}, ncclFunc{coll}, {redop_cxx}, {ty_cxx}, "
      "NCCL_ALGO_{algo}, NCCL_PROTO_{proto}, {acc}, {pipeline}, {unroll}, {reg})\n\n"
      "__launch_bounds__(NCCL_MAX_NTHREADS, 1)\n"
      "__global__ void ncclDevKernel_{sym}_Specialized(\n"
      "    ncclDevKernelArgsDefaultStorage NCCL_GRID_CONSTANT const argsStorage) {{\n"
      "  ncclShmemPerWarp[0].x = 0;\n"
      "  {func_name}();\n"
      "}}\n"
      .format(sym=sym, coll=fn.coll, redop_cxx=redop_to_cxx[fn.redop],
              ty_cxx=ty_to_cxx[fn.ty], algo=(fn.algo or "RING"),
              proto=(fn.proto or "SIMPLE"), acc=fn.acc, pipeline=fn.pipeline,
              unroll=fn.unroll, reg=fn.reg, func_name=func_name)
    )
    if guard:
      out("#endif\n")

# Sort specialized files so the heaviest kernels appear first in build.ninja.
# Ninja breaks scheduling ties by edge ID (= rule order in the manifest), so
# putting slow kernels first ensures they start early and don't form a long tail.
_ty_cost  = {t: (0 if t in ("f8e4m3", "f8e5m2") else 1) for t in all_tys}
_proto_cost = {"SIMPLE": 0, "LL": 1, "LL128": 2}
_algo_cost  = {"TREE": 0, "RING": 1, "PAT": 2}

def _compile_cost_key(entry):
  fn = entry[3]  # Fn object stashed as 4th element
  return (
    _ty_cost.get(fn.ty, 1),
    _proto_cost.get(fn.proto, 1),
    _algo_cost.get(fn.algo, 1),
    -int(fn.unroll),
  )

specialized_filelist.sort(key=_compile_cost_key)

# Write the list of specialized files for CMake consumption
with open(os.path.join(gensrc, "specialized_files.txt"), "w") as f:
  for filename, func_name, guard, fn in specialized_filelist:
    if fn.unroll in ("8", "16", "32"):
      cmake_guard = "defined(__gfx1250__)"
      if fn.proto == "LL128":
        cmake_guard += " && defined(ENABLE_LL128)"
    else:
      cmake_guard = guard or ""
    f.write("%s %s %s\n" % (filename, func_name, cmake_guard))

print("-- Generated %d specialized kernel files in %s" % (len(specialized_filelist), specialized_dir))
