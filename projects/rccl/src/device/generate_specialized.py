#!/usr/bin/env python3
"""
Generate specialized kernels for all valid RCCL operations.

This script mirrors the operation enumeration rules in generate.py so the
specialized set matches the generic kernel coverage.
"""
import os
import sys
import subprocess
import shutil
from dataclasses import dataclass

# Order of colls, redops, tys, protos, algos must match src/include/device.h
all_colls     = ["Broadcast", "Reduce", "AllGather", "ReduceScatter", "AllReduce", "SendRecv", "", "", "AlltoAllPivot", "AllToAllGda"]
all_redops    = ["Sum","Prod","MinMax","PreMulSum","SumPostDiv"]
all_tys       = ["i8","u8","i32","u32","i64","u64","f16","f32","f64","bf16","f8e4m3","f8e5m2"]
all_protos    = ["LL","LL128","SIMPLE"]
all_algos     = ["TREE","RING", "", "", "", "", "PAT"]
all_accs      = ["0", "1"]
all_pipelines = ["0", "1"]
all_unrolls   = ["1", "2", "4"]

all_params = [all_colls, all_algos, all_protos, all_redops, all_tys, all_accs, all_pipelines, all_unrolls]

algos_of_coll = {
  "AllGather":             ["RING", "PAT"],
  "AllReduce":             ["RING", "TREE"],
  "AlltoAllPivot":         ["RING"],
  "AllToAllGda":           ["RING"],
  "Broadcast":             ["RING"],
  "Reduce":                ["RING"],
  "ReduceScatter":         ["RING", "PAT"],
  "SendRecv":              ["RING"]
}

protos_of_coll = {
  "AllGather":              all_protos,
  "AllReduce":              all_protos,
  "AlltoAllPivot":          ["SIMPLE"],
  "AllToAllGda":            ["SIMPLE"],
  "Broadcast":              all_protos,
  "Reduce":                 all_protos,
  "ReduceScatter":          all_protos,
  "SendRecv":               ["SIMPLE"]
}

# All reduction ops are now supported in specialized builds.
specialized_redops = all_redops  # ["Sum","Prod","MinMax","PreMulSum","SumPostDiv"]

redops_of_coll = {
  "AllGather":            ["Sum"],
  "AllReduce":            specialized_redops,
  "AlltoAllPivot":        ["Sum"],
  "AllToAllGda":          ["Sum"],
  "Broadcast":            ["Sum"],
  "Reduce":               specialized_redops,
  "ReduceScatter":        specialized_redops,
  "SendRecv":             ["Sum"]
}

tys_of_coll = {
  "AllGather":             ["i8"],
  "AllReduce":             all_tys,
  "AlltoAllPivot":         ["i8"],
  "AllToAllGda":           ["i8"],
  "Broadcast":             ["i8"],
  "Reduce":                all_tys,
  "ReduceScatter":         all_tys,
  "SendRecv":              ["i8"]
}

acc_of_coll = {
  "AllGather":             ["0"],
  "AllReduce":             all_accs,
  "AlltoAllPivot":         ["0"],
  "AllToAllGda":           ["0"],
  "Broadcast":             ["0"],
  "Reduce":                ["0"],
  "ReduceScatter":         ["0"],
  "SendRecv":              ["0"]
}

pipelines_of_coll = {
  "AllGather":             ["0"],
  "AllReduce":             all_pipelines,
  "AlltoAllPivot":         ["0"],
  "AllToAllGda":           ["0"],
  "Broadcast":             ["0"],
  "Reduce":                all_pipelines,
  "ReduceScatter":         all_pipelines,
  "SendRecv":              ["0"]
}

pipelined_types = ["bf16"]

coll_camel_to_lower = {
  "AllGather":             "all_gather",
  "AllReduce":             "all_reduce",
  "AlltoAllPivot":         "alltoall_pivot",
  "AllToAllGda":           "alltoall_gda",
  "Broadcast":             "broadcast",
  "Reduce":                "reduce",
  "ReduceScatter":         "reduce_scatter",
  "SendRecv":              "sendrecv"
}

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

  def __iter__(self):
    return iter((self.coll, self.algo, self.proto, self.redop, self.ty, self.acc, self.pipeline, self.unroll))

def calc_unroll_and_pipeline_for_local_arch(is_local_arch_only):
  """Calculate unroll and pipeline settings.
  
  This must match the production generator (generate.py) exactly:
  - When is_local_arch_only=True: detect local GPU via rocminfo and use optimal unroll
  - When is_local_arch_only=False: generate all unrolls (1, 2, 4)
  """
  if not is_local_arch_only:
    return (all_unrolls, all_pipelines)

  # Detect local GPU via rocminfo (matches production generator)
  rocm_path = os.environ.get('ROCM_PATH', '/opt/rocm')
  rocminfo_path = os.path.join(rocm_path, "bin", "rocminfo")
  res = subprocess.run([rocminfo_path], stdout=subprocess.PIPE, universal_newlines=True)
  rocminfo_output = res.stdout

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

  gfx_targets = list(gfx_targets.keys())
  if len(gfx_targets) == 1:
    gfx_name, cu_count = gfx_targets[0]
    if "gfx950" == gfx_name:
      return (["1", "2"], ["0"])
    elif "gfx908" == gfx_name or ("gfx942" == gfx_name and cu_count > 80):
      return (["2"], all_pipelines)
    else:
      return (["4"], all_pipelines)
  return (all_unrolls, all_pipelines)

def func_validate(coll, algo, proto, redop, ty, acc, pipeline, unroll, local_unroll, local_pipeline):
  if redop == "SumPostDiv" and ty[0] not in ("i","u"):
    return False
  if coll == "" or algo == "":
    return False
  if (algo not in algos_of_coll[coll] or
      proto not in protos_of_coll[coll] or
      redop not in redops_of_coll[coll] or
      ty not in tys_of_coll[coll] or
      acc not in acc_of_coll[coll] or
      pipeline not in pipelines_of_coll[coll] or (pipeline in ["1"] and ty not in pipelined_types) or
      pipeline not in local_pipeline or
      unroll not in local_unroll):
    return False
  return True

def func_filter(function_params, current_idx, local_unroll, local_pipeline, item_list=None):
  if item_list is None:
    item_list = []

  if current_idx < len(all_params):
    current_element = function_params[current_idx]
    if current_element == "*":
      current_list = all_params[current_idx]
      for item in current_list:
        item_list.append(item)
        yield from func_filter(function_params, current_idx+1, local_unroll, local_pipeline, item_list)
        item_list.pop()
    else:
      elements = current_element.split("/")
      current_param = all_params[current_idx]
      for item in elements:
        if item not in current_param:
          raise ValueError(f"Error: {item} is unrecognized or does not belong to this category {current_param}.")
      for item in elements:
        item_list.append(item)
        yield from func_filter(function_params, current_idx+1, local_unroll, local_pipeline, item_list)
        item_list.pop()
  else:
    coll, algo, proto, redop, ty, acc, pipeline, unroll = item_list
    if func_validate(coll, algo, proto, redop, ty, acc, pipeline, unroll, local_unroll, local_pipeline):
      yield (coll, algo, proto, redop, ty, acc, pipeline, unroll)

def parse_input(func_pattern, local_unroll, local_pipeline):
  input_list = sorted(func_pattern.split("|"))
  for input in input_list:
    function_params = input.split()
    params_length = len(function_params)
    while params_length < len(all_params):
      function_params.append("*")
      params_length += 1
    yield from func_filter(function_params, 0, local_unroll, local_pipeline)

def equivalent_primary(coll, algo, proto, redop, ty, acc, pipeline, unroll):
  if coll in ("AllReduce", "Reduce", "ReduceScatter"):
    if redop in ("Sum","Prod","PreMulSum","SumPostDiv") and ty[0]=="i":
      ty = "u"+ty[1:]
    elif redop=="MinMax" and ty[0]=="i" and ("NVLS" not in algo):
      ty = "u"+ty[1:]
    if (pipeline != "0" and proto != "SIMPLE"):
      pipeline = "0"
  return (coll, algo, proto, redop, ty, acc, pipeline, unroll)

def custom_sort_key(fn: Fn, local_unroll, local_pipeline):
  return (
      local_unroll.index(fn.unroll),
      all_colls.index(fn.coll),
      all_algos.index(fn.algo),
      all_protos.index(fn.proto),
      all_redops.index(fn.redop),
      all_tys.index(fn.ty),
      all_accs.index(fn.acc),
      local_pipeline.index(fn.pipeline)
  )

def get_arch_guard(fn):
  """Get the preprocessor guard for a kernel.

  Note: __gfx942__ etc. are only defined during device code compilation,
  not host code compilation. Since getPtr is host code and needs to
  reference the kernel, we can't use architecture macros.

  Instead, we only use feature guards (ENABLE_LL128) and rely on cmake's
  --offload-arch to restrict to the correct architecture.
  """
  cond = None
  # Only use feature guards, not architecture guards
  if fn.proto == "LL128":
    cond = "defined(ENABLE_LL128)"
  # Don't use __gfx942__ etc. - cmake handles architecture targeting
  return cond

def generate_specialized_kernel_file(op_tuple, output_dir):
  coll, algo, proto, redop, ty, acc, pipeline, unroll = op_tuple
  coll_lower = coll_camel_to_lower[coll]
  # Include acc and unroll in filename to avoid overwriting
  acc_file_suffix = f"_acc{acc}" if acc != "0" else ""
  unroll_file_suffix = f"_unroll{unroll}"
  filename = f"specialized_{coll_lower}_{algo.lower()}_{proto.lower()}_{redop.lower()}_{ty}{acc_file_suffix}{unroll_file_suffix}.cpp"
  filepath = os.path.join(output_dir, filename)

  type_map = {
      "i8": "int8_t",
      "u8": "uint8_t",
      "i32": "int32_t",
      "u32": "uint32_t",
      "i64": "int64_t",
      "u64": "uint64_t",
      "f16": "__half",
      "f32": "float",
      "f64": "double",
      "bf16": "hip_bfloat16",
      "f8e4m3": "rccl_float8",
      "f8e5m2": "rccl_bfloat8",
  }

  redop_map = {
      "Sum": "FuncSum",
      "Prod": "FuncProd",
      "MinMax": "FuncMinMax",
      "PreMulSum": "FuncPreMulSum",
      "SumPostDiv": "FuncSumPostDiv",
  }

  func_map = {
      "AllReduce": "ncclFuncAllReduce",
      "Broadcast": "ncclFuncBroadcast",
      "Reduce": "ncclFuncReduce",
      "AllGather": "ncclFuncAllGather",
      "ReduceScatter": "ncclFuncReduceScatter",
      "SendRecv": "ncclFuncSendRecv",
      "AlltoAllPivot": "ncclFuncAlltoAllPivot",
      "AllToAllGda": "ncclFuncAllToAllGda",
  }

  algo_const = f"NCCL_ALGO_{algo}"
  proto_const = f"NCCL_PROTO_{proto}"
  cxx_type = type_map.get(ty, "float")
  redop_class = redop_map.get(redop, "FuncSum")
  func_const = func_map.get(coll, "ncclFuncAllReduce")

  # Include acc and unroll in kernel name to avoid duplicates
  acc_suffix = f"_acc{acc}" if acc != "0" else ""
  unroll_suffix = f"_unroll{unroll}"
  kernel_name = f"ncclDevKernel_{coll}_{algo}_{proto}_{redop}_{ty}{acc_suffix}{unroll_suffix}_Specialized"
  guard = get_arch_guard(Fn(*op_tuple))

  # Build ncclDevFunc name (matches production naming: Coll_ALGO_PROTO_Redop_ty_acc_pipeline_unroll)
  devfunc_suffix = f"{coll}_{algo}_{proto}_{redop}_{ty}_{acc}_{pipeline}_{unroll}"

  # Build kernel code
  kernel_code = f"""// ncclDevFunc - the noinline device function for this operation
// This is the entry point that will be used by the device linker
DEFINE_ncclDevFunc({devfunc_suffix}, {func_const}, {redop_class}, {cxx_type}, {algo_const}, {proto_const}, {acc}, {pipeline}, {unroll})

// Specialized kernel (includes ncclDevFunc above, used for resource calculation)
__launch_bounds__(NCCL_MAX_NTHREADS, 1)
__global__ void {kernel_name}(
    ncclDevKernelArgsDefaultStorage NCCL_GRID_CONSTANT const argsStorage) {{

  ncclKernelMain<
    {func_const},
    RunWorkBatch<{func_const}, {cxx_type}, {redop_class}<{cxx_type}>, {algo_const}, {proto_const}, {acc}, {unroll}, {pipeline}>,
    false,  // COLLTRACE
    {unroll}
  >(&argsStorage.args);
}}

// Host-side getter - exported for runtime kernel lookup
extern "C" __attribute__((visibility("default")))
void* {kernel_name}_getPtr() {{
  return (void*){kernel_name};
}}
"""

  # Wrap in guard if needed (same guard for kernel and getPtr)
  if guard:
    kernel_code = f"#if {guard}\n{kernel_code}#endif\n"

  content = f"""/*
 * GENERATED FILE - DO NOT EDIT
 * Specialized kernel for: {coll} {algo} {proto} {redop} {ty} acc={acc} pipeline={pipeline} unroll={unroll}
 */

#define NCCL_SPECIALIZED_KERNEL 1
#define NCCL_SHMEM_DECL __shared__
#include "common.h"
#include "{coll_lower}.h"

{kernel_code}"""

  with open(filepath, 'w') as f:
    f.write(content)

  return filename

def generate_kernel_selector(all_ops, output_dir):
  filepath = os.path.join(output_dir, "specialized_kernel_selector.h")

  func_map = {
      "AllReduce": "ncclFuncAllReduce",
      "Broadcast": "ncclFuncBroadcast",
      "Reduce": "ncclFuncReduce",
      "AllGather": "ncclFuncAllGather",
      "ReduceScatter": "ncclFuncReduceScatter",
      "SendRecv": "ncclFuncSendRecv",
      "AlltoAllPivot": "ncclFuncAlltoAllPivot",
      "AllToAllGda": "ncclFuncAllToAllGda",
  }

  type_map = {
      "i8": "ncclInt8", "u8": "ncclUint8",
      "i32": "ncclInt32", "u32": "ncclUint32",
      "i64": "ncclInt64", "u64": "ncclUint64",
      "f16": "ncclFloat16", "f32": "ncclFloat32",
      "f64": "ncclFloat64", "bf16": "ncclBfloat16",
      "f8e4m3": "ncclFloat8e4m3", "f8e5m2": "ncclFloat8e5m2",
  }

  redop_map = {
      "Sum": "ncclDevSum",
      "Prod": "ncclDevProd",
      "MinMax": "ncclDevMinMax",
      "PreMulSum": "ncclDevPreMulSum",
      "SumPostDiv": "ncclDevSumPostDiv",
  }

  content = """/*
 * GENERATED FILE - DO NOT EDIT
 * Kernel selector for specialized vs generic kernels
 */

#ifndef SPECIALIZED_KERNEL_SELECTOR_H_
#define SPECIALIZED_KERNEL_SELECTOR_H_

#include "nccl.h"
#include "device.h"

// Forward declarations of specialized kernels
"""

  # Group ops by guard and emit declarations
  for op in all_ops:
    coll, algo, proto, redop, ty, acc, pipeline, unroll = op
    acc_suffix = f"_acc{acc}" if acc != "0" else ""
    unroll_suffix = f"_unroll{unroll}"
    kernel_name = f"ncclDevKernel_{coll}_{algo}_{proto}_{redop}_{ty}{acc_suffix}{unroll_suffix}_Specialized"
    guard = get_arch_guard(Fn(*op))
    if guard:
      content += f"#if {guard}\n"
    content += f"extern \"C\" void* {kernel_name}_getPtr();\n"
    if guard:
      content += f"#endif\n"

  content += """
// Check if operation has a specialized kernel
inline bool hasSpecializedKernel(
    ncclFunc_t func, int algo, int proto, ncclDevRedOp_t redop,
    ncclDataType_t datatype, int acc, int pipeline, int unroll) {
"""

  first = True
  for op in all_ops:
    coll, algo, proto, redop, ty, acc_str, pipeline_str, unroll_str = op
    func_const = func_map.get(coll, "ncclFuncAllReduce")
    type_const = type_map.get(ty, "ncclFloat32")
    redop_const = redop_map.get(redop, "ncclDevSum")
    algo_const = f"NCCL_ALGO_{algo}"
    proto_const = f"NCCL_PROTO_{proto}"
    guard = get_arch_guard(Fn(*op))

    if guard:
      content += f"#if {guard}\n"
    if first:
      content += f"  if (func == {func_const} && algo == {algo_const} && proto == {proto_const} && \n"
      first = False
    else:
      content += f"  else if (func == {func_const} && algo == {algo_const} && proto == {proto_const} && \n"
    content += f"      redop == {redop_const} && datatype == {type_const}) {{\n"
    content += "    return true;\n"
    content += "  }\n"
    if guard:
      content += f"#endif\n"

  content += "  return false;\n}\n\n"

  content += """// Get specialized kernel pointer
inline void* getSpecializedKernel(
    ncclFunc_t func, int algo, int proto, ncclDevRedOp_t redop,
    ncclDataType_t datatype, int acc, int pipeline, int unroll) {
"""

  first = True
  for op in all_ops:
    coll, algo, proto, redop, ty, acc_str, pipeline_str, unroll_str = op
    func_const = func_map.get(coll, "ncclFuncAllReduce")
    type_const = type_map.get(ty, "ncclFloat32")
    redop_const = redop_map.get(redop, "ncclDevSum")
    algo_const = f"NCCL_ALGO_{algo}"
    proto_const = f"NCCL_PROTO_{proto}"
    # Include acc and unroll in kernel name to avoid duplicates
    acc_suffix = f"_acc{acc_str}" if acc_str != "0" else ""
    unroll_suffix = f"_unroll{unroll_str}"
    kernel_name = f"ncclDevKernel_{coll}_{algo}_{proto}_{redop}_{ty}{acc_suffix}{unroll_suffix}_Specialized"
    guard = get_arch_guard(Fn(*op))

    if guard:
      content += f"#if {guard}\n"
    if first:
      content += f"  if (func == {func_const} && algo == {algo_const} && proto == {proto_const} && \n"
      first = False
    else:
      content += f"  else if (func == {func_const} && algo == {algo_const} && proto == {proto_const} && \n"
    content += f"      redop == {redop_const} && datatype == {type_const} && acc == {acc_str} && unroll == {unroll_str}) {{\n"
    content += f"    return {kernel_name}_getPtr();\n"
    content += "  }\n"
    if guard:
      content += f"#endif\n"

  content += """  return nullptr;  // Fall back to generic kernel
}

#endif // SPECIALIZED_KERNEL_SELECTOR_H_
"""

  with open(filepath, 'w') as f:
    f.write(content)

  print(f"Generated kernel selector: {filepath}")

def generate_kernel_list(all_ops, output_dir):
  """Generate a header with array of all specialized kernel pointers for initialization."""
  filepath = os.path.join(output_dir, "specialized_kernel_list.h")

  content = """/*
 * GENERATED FILE - DO NOT EDIT
 * List of all specialized kernels for initialization
 */

#ifndef SPECIALIZED_KERNEL_LIST_H_
#define SPECIALIZED_KERNEL_LIST_H_

#include "nccl.h"

// Forward declarations of specialized kernel getters
"""

  for op in all_ops:
    coll, algo, proto, redop, ty, acc, pipeline, unroll = op
    acc_suffix = f"_acc{acc}" if acc != "0" else ""
    unroll_suffix = f"_unroll{unroll}"
    kernel_name = f"ncclDevKernel_{coll}_{algo}_{proto}_{redop}_{ty}{acc_suffix}{unroll_suffix}_Specialized"
    guard = get_arch_guard(Fn(*op))
    if guard:
      content += f"#if {guard}\n"
    content += f"extern \"C\" void* {kernel_name}_getPtr();\n"
    if guard:
      content += f"#endif\n"

  content += f"""
// Array of all specialized kernel pointers
inline void** getSpecializedKernelList() {{
  static void* kernels[] = {{
"""

  for i, op in enumerate(all_ops):
    coll, algo, proto, redop, ty, acc, pipeline, unroll = op
    acc_suffix = f"_acc{acc}" if acc != "0" else ""
    unroll_suffix = f"_unroll{unroll}"
    kernel_name = f"ncclDevKernel_{coll}_{algo}_{proto}_{redop}_{ty}{acc_suffix}{unroll_suffix}_Specialized"
    guard = get_arch_guard(Fn(*op))
    if guard:
      content += f"#if {guard}\n"
    content += f"    {kernel_name}_getPtr(),\n"
    if guard:
      content += f"#else\n    nullptr,\n#endif\n"

  content += f"""    nullptr
  }};
  return kernels;
}}

inline int getSpecializedKernelCount() {{
  return {len(all_ops)};
}}

#endif // SPECIALIZED_KERNEL_LIST_H_
"""

  with open(filepath, 'w') as f:
    f.write(content)

  print(f"Generated kernel list: {filepath}")

def main():
  if len(sys.argv) < 2:
    print("Usage: generate_specialized.py <output_dir> [BUILD_LOCAL_GPU_TARGET_ONLY] [ONLY_FUNCS]")
    sys.exit(1)

  output_dir = sys.argv[1]
  is_local_arch_only = 1 if (len(sys.argv) > 2 and sys.argv[2] == "ON") else 0

  if len(sys.argv) > 3 and sys.argv[3]:
    func_pattern = sys.argv[3]
  else:
    func_pattern = "AllGather|AllReduce|AlltoAllPivot|AllToAllGda|Broadcast|Reduce|ReduceScatter|SendRecv"

  if os.path.exists(output_dir):
    for name in os.listdir(output_dir):
      path = os.path.join(output_dir, name)
      if os.path.isfile(path):
        os.remove(path)
      elif os.path.isdir(path):
        shutil.rmtree(path)
  else:
    os.makedirs(output_dir)

  local_unroll, local_pipeline = calc_unroll_and_pipeline_for_local_arch(is_local_arch_only)
  primary_funcs = sorted(
      {Fn(*equivalent_primary(*fn)) for fn in parse_input(func_pattern, local_unroll, local_pipeline)},
      key=lambda fn: custom_sort_key(fn, local_unroll, local_pipeline)
  )

  print("========================================")
  print("Full Specialized Kernel Generation")
  print("========================================")
  print(f"Output directory: {output_dir}")
  print(f"Operations: {len(primary_funcs)}")
  print("")

  generated_files = []
  for op in primary_funcs:
    filename = generate_specialized_kernel_file(tuple(op), output_dir)
    generated_files.append(filename)

  print("")
  generate_kernel_selector([tuple(op) for op in primary_funcs], output_dir)
  generate_kernel_list([tuple(op) for op in primary_funcs], output_dir)
  print("")
  print(f"Total files generated: {len(generated_files) + 2}")
  print("")

  cmake_file = os.path.join(output_dir, "specialized_kernels.cmake")
  with open(cmake_file, 'w') as f:
    f.write("# CMake fragment for specialized kernels\n")
    f.write("set(SPECIALIZED_KERNEL_SOURCES\n")
    for filename in generated_files:
      f.write(f"  ${{GEN_DIR}}/specialized/{filename}\n")
    f.write(")\n")
  print(f"Generated CMake fragment: {cmake_file}")

  file_list = os.path.join(output_dir, "file_list.txt")
  with open(file_list, 'w') as f:
    for filename in generated_files:
      f.write(f"{filename}\n")
  print(f"Generated file list: {file_list}")

if __name__ == "__main__":
  main()
