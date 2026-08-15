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

local_gfx_name = None

def detect_local_gfx_targets():
  """Distinct (gfx_name, cu_count) pairs rocminfo reports, or [] if it can't run.

  Deduplicated by name AND compute-unit count, since the same gfx can appear
  with different CU counts (SPX/CPX partitioning), which is not a homogeneous
  system.
  """
  rocminfo_path = os.path.join(os.environ.get('ROCM_PATH', '/opt/rocm'),
                               'bin', 'rocminfo')
  try:
    res = subprocess.run([rocminfo_path], stdout=subprocess.PIPE,
                         universal_newlines=True, check=True)
  except (OSError, subprocess.CalledProcessError) as e:
    # Building every unroll factor is always correct, just slower, so warn and
    # carry on instead of failing the configure step.
    print("-- WARNING: cannot run %s (%s); building all unroll factors. Set "
          "ROCM_PATH to narrow the build to the local architecture."
          % (rocminfo_path, e), file=sys.stderr)
    return []

  gfx_targets = {}
  curr_name = None
  for line in res.stdout.splitlines():
    line = line.strip()
    if line.startswith("Name:"):
      name = line.split(':')[-1].strip()
      if "gfx" in name:
        curr_name = name
    if line.startswith("Compute Unit:") and curr_name:
      gfx_targets[(curr_name, int(line.split(':')[-1].strip()))] = None
      curr_name = None
  return list(gfx_targets.keys())

def calc_unroll_and_pipeline_for_local_arch():
  """The (unrolls, pipelines) this build needs, setting local_gfx_name when
  exactly one on-system gfx target was detected."""
  global local_gfx_name
  local_gfx_name = None

  if not is_local_arch_only:
    return (all_unrolls, all_pipelines)

  # A homogeneous system is required to narrow the unroll set at all.
  gfx_targets = detect_local_gfx_targets()
  if len(gfx_targets) != 1:
    return (all_unrolls, all_pipelines)

  gfx_name, cu_count = gfx_targets[0]
  local_gfx_name = gfx_name
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

# if building for local arch only, we only need to build for 1 variant of unroll for most gfx targets,
# except for gfx950. For gfx950, we also disable pipelining.
local_unroll, local_pipeline = calc_unroll_and_pipeline_for_local_arch()

# funcId indexes the first local_unroll slice (see host_table / enumerate_func_rows
# and dispatch_branches_for_unroll_table() below). Defined here, ahead of the
# unroll override table, because maybe_remap_unroll() below depends on it.
func_id_unroll = local_unroll[0]

# rocSHMEM/GDA-based collectives: only generated when ENABLE_ROCSHMEM build is requested
gda_colls = {"AlltoAllGda", "AlltoAllvGda"}

# Unroll overrides: force a collective identity's generic-kernel dispatch (in
# EVERY built ncclDevFuncTable_*/Caller* table) to use one specific unroll's
# kernel instead of each table's own native variant. Used today to route the
# 17 slowest gfx1250 SIMPLE identities (compile time exceeded 45s at unroll
# 16/32; see --kernel-compile-timing in install.sh) to their unroll-8 variant,
# but the mechanism is general: any (identity, unroll) pair can be added here,
# keyed by the gfx target it applies to. If the target unroll isn't otherwise
# produced by this build, an extra kernel is compiled just to serve the
# override -- see forced_override_funcs().
@dataclass(frozen=True)
class UnrollOverride:
  coll: str
  algo: str
  proto: str
  redop: str
  ty: str
  acc: str
  pipeline: str
  unroll: str  # kernel to substitute wherever this identity is dispatched

  def identity(self):
    return (self.coll, self.algo, self.proto, self.redop, self.ty, self.acc, self.pipeline)

# Unroll -> the single arch that exclusively owns it. get_arch_guard() compiles
# these unrolls only under that arch's guard, so no other arch's object ever
# holds a real (non-nullptr/non-trap) definition at one of them. That makes an
# override for the owning arch safe to skip a redundant native variant even in a
# multi-arch/fat build. An override targeting a *shared* unroll (1/2/4) cannot:
# the other archs in that binary still need their own native variant, so it gets
# a per-arch dispatch redirect instead (dispatch_branches_for_unroll_table()).
# This table is the single source of that fact -- get_arch_guard() and the
# specialized_files.txt guard column both derive from it.
_EXCLUSIVE_UNROLL_TIERS = {"8": "gfx1250", "16": "gfx1250", "32": "gfx1250"}

# gfx -> (datatypes, unroll): datatypes this arch may only ever launch at one
# fixed unroll factor, regardless of which table dispatches them. gfx1250 FP8
# kernels launch correctly only at unroll 8, and enqueue.cc clamps FP8 there at
# runtime, so no other FP8 variant of an unroll tier gfx1250 owns is ever
# reachable. This is a property of the ARCH, not of how the build was invoked,
# so it must hold in a multi-arch/fat build too -- not just under
# --local_gpu_only. Shared tiers (1/2/4) keep their FP8 variants: the other
# archs in a fat build still dispatch them.
_FIXED_UNROLL_TYPES = {"gfx1250": (("f8e4m3", "f8e5m2"), "8")}

def fixed_unroll_for_type(gfx, ty):
  """The only unroll `gfx` may launch `ty` at, or None when unconstrained.

  A gfx of None (no sole owner -- see sole_arch_for_unroll()) is unconstrained:
  a tier shared by several archs cannot be narrowed to one arch's rule.
  """
  entry = _FIXED_UNROLL_TYPES.get(gfx)
  if entry is None or ty not in entry[0]:
    return None
  return entry[1]

_UNROLL_OVERRIDES = {
  "gfx1250": {
    UnrollOverride("AllReduce", "TREE", "SIMPLE", "MinMax", "f16", "1", "0", "8"),
    UnrollOverride("AllReduce", "TREE", "SIMPLE", "MinMax", "f16", "0", "0", "8"),
    UnrollOverride("AllReduce", "RING", "SIMPLE", "MinMax", "f16", "1", "0", "8"),
    UnrollOverride("AllReduce", "RING", "SIMPLE", "MinMax", "u8", "1", "0", "8"),
    UnrollOverride("AllReduce", "TREE", "SIMPLE", "SumPostDiv", "u8", "1", "0", "8"),
    UnrollOverride("AllReduce", "TREE", "SIMPLE", "MinMax", "u8", "1", "0", "8"),
    UnrollOverride("AllReduce", "TREE", "SIMPLE", "Prod", "u8", "1", "0", "8"),
    UnrollOverride("AllReduce", "TREE", "SIMPLE", "PreMulSum", "bf16", "0", "1", "8"),
    UnrollOverride("AllReduce", "TREE", "SIMPLE", "PreMulSum", "u8", "1", "0", "8"),
    UnrollOverride("AllReduce", "TREE", "SIMPLE", "Sum", "u8", "1", "0", "8"),
    UnrollOverride("AllReduce", "TREE", "SIMPLE", "MinMax", "bf16", "0", "1", "8"),
    UnrollOverride("AllReduce", "TREE", "SIMPLE", "MinMax", "bf16", "1", "1", "8"),
    UnrollOverride("AllReduce", "TREE", "SIMPLE", "Prod", "bf16", "0", "1", "8"),
    UnrollOverride("AllReduce", "TREE", "SIMPLE", "Sum", "bf16", "0", "1", "8"),
    UnrollOverride("AllReduce", "TREE", "SIMPLE", "SumPostDiv", "u8", "0", "0", "8"),
    UnrollOverride("AllReduce", "TREE", "SIMPLE", "PreMulSum", "bf16", "1", "1", "8"),
    UnrollOverride("AllReduce", "RING", "SIMPLE", "SumPostDiv", "u8", "1", "0", "8"),
  },
}

def build_unroll_override_index():
  """gfx -> {identity -> override unroll}, validated once from
  _UNROLL_OVERRIDES above."""
  index = {}
  for gfx, overrides in _UNROLL_OVERRIDES.items():
    by_identity = {}
    for ov in overrides:
      assert ov.identity() not in by_identity, (
        "Duplicate unroll override for %s identity %s" % (gfx, ov.identity())
      )
      assert ov.unroll in all_unrolls, (
        "Unroll override for %s %s targets unroll %r, not one of %s"
        % (gfx, ov.identity(), ov.unroll, all_unrolls)
      )
      # An override for one arch targeting another arch's exclusive tier would
      # reference a symbol guarded out -- and therefore undeclared -- during its
      # own compile pass, silently trapping or null-dispatching at runtime
      # instead of failing to build. Catch it here rather than on hardware.
      exclusive_owner = _EXCLUSIVE_UNROLL_TIERS.get(ov.unroll)
      assert exclusive_owner is None or exclusive_owner == gfx, (
        "Unroll override for %s %s targets unroll %r, but that unroll is "
        "compiled exclusively for %s (see _EXCLUSIVE_UNROLL_TIERS / "
        "get_arch_guard()); this override would reference an undeclared "
        "symbol when compiling for %s. Pick an unroll that %s natively "
        "produces (see calc_unroll_and_pipeline_for_local_arch())."
        % (gfx, ov.identity(), ov.unroll, exclusive_owner, gfx, gfx)
      )
      by_identity[ov.identity()] = ov.unroll
    index[gfx] = by_identity
  return index

_unroll_override_by_identity = build_unroll_override_index()

# Which gfx targets' override entries are active for this generation run:
#   - True single local-arch build (local_gfx_name is set): only that arch is
#     ever compiled, so only its entry applies, and it applies unconditionally.
#   - Multi-arch/fat build (local_gfx_name is None: --local_gpu_only wasn't
#     requested, or rocminfo found zero/multiple gfx targets): generate.py is
#     never told the real GPU_TARGETS list (a CMake-level concept), so every gfx
#     key is treated as potentially present and gets its own
#     `#if defined(__gfxN__)` dispatch branch. Each branch is simply never
#     selected during another arch's compile pass, so this stays correct for any
#     subset of archs a build actually targets.
if local_gfx_name is not None:
  override_archs = [local_gfx_name] if local_gfx_name in _unroll_override_by_identity else []
else:
  override_archs = sorted(_unroll_override_by_identity.keys())

def unroll_override_for_gfx(gfx, coll, algo, proto, redop, ty, acc, pipeline):
  """Return gfx's override target unroll for this identity, or None.

  Also checks the equivalence-class representative (e.g. MinMax i8 folds into
  MinMax u8) so both share the same override. A gfx of None (no sole owner --
  see sole_arch_for_unroll()) simply has no overrides.
  """
  by_identity = _unroll_override_by_identity.get(gfx)
  if not by_identity:
    return None
  identity = (coll, algo, proto, redop, ty, acc, pipeline)
  if identity in by_identity:
    return by_identity[identity]
  eq_identity = equivalent_primary(coll, algo, proto, redop, ty, acc, pipeline, "1", "0")[:7]
  return by_identity.get(eq_identity)

def sole_arch_for_unroll(unroll):
  """The one gfx target whose overrides can apply to unroll `unroll`, or None
  when several archs share it.

  A true single-local-arch build compiles exactly one arch, so that arch owns
  every tier. Otherwise only an exclusive tier has a single owner; a shared
  tier (1/2/4) is reached by every arch in a fat build, which needs per-arch
  dispatch branches instead (see dispatch_branches_for_unroll_table()).
  """
  if local_gfx_name is not None:
    return local_gfx_name
  return _EXCLUSIVE_UNROLL_TIERS.get(unroll)

def maybe_remap_unroll(coll, algo, proto, redop, ty, acc, pipeline, unroll):
  """Skip generating a native variant that no arch could ever dispatch to.

  An override redirects every generic-kernel dispatch of this identity onto one
  compiled variant, so the other variants are dead code -- but only where a
  single arch owns this unroll (sole_arch_for_unroll()). A shared tier of a
  multi-arch build must keep every arch's native variant and gets a per-arch
  dispatch redirect instead. The func-id-axis anchor (func_id_unroll, needed
  for funcId / host_table bookkeeping) and the override's own target are always
  kept.
  """
  override_unroll = unroll_override_for_gfx(
    sole_arch_for_unroll(unroll), coll, algo, proto, redop, ty, acc, pipeline)
  if override_unroll is None or unroll in (func_id_unroll, override_unroll):
    return unroll
  return None

def yield_func_row(coll, algo, proto, redop, ty, acc, pipeline, unroll, reg):
  if not func_validate(coll, algo, proto, redop, ty, acc, pipeline, unroll, reg):
    return
  if maybe_remap_unroll(coll, algo, proto, redop, ty, acc, pipeline, unroll) is None:
    return
  yield (coll, algo, proto, redop, ty, acc, pipeline, unroll, reg)

# Helper function to check if the conditions for the collective is being met.
# ignore_unroll_membership bypasses only the "unroll must be in local_unroll"
# check, for forced_override_funcs() below to validate a kernel that an
# unroll override needs compiled at a factor this build wouldn't otherwise
# produce; every other structural check (algo/proto/redop/ty/acc/pipeline
# compatibility, FP8-requires-unroll-8, etc.) still applies.
def func_validate(coll, algo, proto, redop, ty, acc, pipeline, unroll, reg,
                   ignore_unroll_membership=False):
  if redop == "SumPostDiv" and ty[0] not in ("i","u"):
    return False
  if coll == "" or algo == "":
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
      (not ignore_unroll_membership and unroll not in local_unroll) or
      reg not in reg_values_of(coll, proto)):
    return False
  # Drop a datatype variant the arch owning this unroll tier could never launch
  # from it (gfx1250 FP8 anywhere but unroll 8).
  fixed_unroll = fixed_unroll_for_type(sole_arch_for_unroll(unroll), ty)
  if fixed_unroll is not None and fixed_unroll != unroll:
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
    yield from yield_func_row(coll, algo, proto, redop, ty, acc, pipeline, unroll, reg)


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
                    yield from yield_func_row(coll, algo, proto, redop, ty, acc, pipeline, unroll, reg)

# Sort the hashmap based on custom key <coll> <algo> <proto> <redop> <ty>
def custom_sort_key(fn: Fn):
    unroll_order = local_unroll.index(fn.unroll) if fn.unroll in local_unroll else len(local_unroll) + all_unrolls.index(fn.unroll)
    return (
        unroll_order,
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

  if fn.unroll in _EXCLUSIVE_UNROLL_TIERS:
      cond = "defined(__%s__)" % _EXCLUSIVE_UNROLL_TIERS[fn.unroll]
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
_primary_funcs_set = {Fn(*equivalent_primary(*fn)) for fn in parse_input(func_pattern)}

def fn_identity_key(fn):
  return (fn.coll, fn.algo, fn.proto, fn.redop, fn.ty, fn.acc, fn.pipeline, fn.reg)

def forced_override_funcs():
  """Extra Fn rows for overrides whose target unroll isn't produced by the
  normal local_unroll enumeration (e.g. an override to unroll 4 on a build
  that only compiles 8/16/32). These are compiled once and folded into every
  generic kernel's dispatch table for that identity (see
  dispatch_branches_for_unroll_table() below); without this there would be
  nothing for those tables to redirect to.

  In a multi-arch/fat build local_unroll is all_unrolls, so every override's
  target unroll is already produced by the normal enumeration and this is a
  no-op there; it only ever does something for a true single-local-arch build
  whose local_unroll is a strict subset (e.g. gfx1250-local's 8/16/32).
  """
  extra = []
  for gfx in override_archs:
    for ov in _UNROLL_OVERRIDES.get(gfx, ()):
      override_unroll = unroll_override_for_gfx(gfx, *ov.identity())
      if override_unroll is None or override_unroll in local_unroll:
        continue
      for reg in reg_values_of(ov.coll, ov.proto):
        if not func_validate(ov.coll, ov.algo, ov.proto, ov.redop, ov.ty, ov.acc,
                              ov.pipeline, override_unroll, reg,
                              ignore_unroll_membership=True):
          continue
        fn = Fn(*equivalent_primary(ov.coll, ov.algo, ov.proto, ov.redop, ov.ty,
                                     ov.acc, ov.pipeline, override_unroll, reg))
        if fn in _primary_funcs_set:
          continue
        print(
          "-- Unroll override (%s): %s is not built at unroll %s for this "
          "target (local unroll = %s); compiling an extra kernel so its "
          "generic-kernel dispatch can use unroll %s"
          % (gfx, paste(" ", *ov.identity()), override_unroll,
             "/".join(local_unroll), override_unroll)
        )
        extra.append(fn)
  return extra

_primary_funcs_set |= set(forced_override_funcs())
primary_funcs = sorted(_primary_funcs_set, key=custom_sort_key)

# primary_to_index[primary_funcs[i]] == i
primary_to_index = {fn: i for i, fn in enumerate(primary_funcs)}
primary_to_index = {fn: primary_to_index.get(Fn(*fn), -1) for fn in func_rows}

primary_by_identity_unroll = {
  (fn_identity_key(fn), fn.unroll): fn for fn in primary_funcs
}

func_id_axis = [fn for fn in primary_funcs if fn.unroll == func_id_unroll]

def dispatch_branches_for_unroll_table(unroll, base_fn):
  """Ordered [(gfx_or_None, fn), ...] branches selecting funcId=base_fn's row
  in the unroll-`unroll` generic-kernel dispatch table. Consumers emit an
  "#if defined(__gfx#) / #elif ... / #else" chain from this list, in order;
  the LAST branch's gfx is always None (the fallback taken by every arch not
  covered by an earlier branch).

  A single-branch [(None, fn)] result means no arch-conditional dispatch is
  needed -- the common case, and the only possible one when a single arch owns
  this unroll (see sole_arch_for_unroll()).

  Multiple branches only arise on a *shared* unroll tier (1/2/4) of a
  multi-arch/fat build, where several archs reuse the SAME table slot and each
  wants its own override (or none): each --offload-arch compile pass defines
  only its own __gfxNNN__ macro, so the preprocessor picks the right branch per
  arch. An override always wins over the native variant for its arch --
  including on this identity's own func_id_unroll table -- since the point of an
  override is to replace this identity's kernel wherever it is dispatched.

  On a tier with a sole owner, an explicit override is tried first and the
  owner's fixed-unroll rule for this datatype second, so an identity with no
  native variant here (gfx1250 FP8 in tables 16/32) lands on the same kernel the
  runtime clamp picks rather than drifting to the func-id axis. Any arch with
  neither falls through to the native variant generated at this table's unroll,
  or to the func-id-axis entry when none exists.
  """
  identity_key = fn_identity_key(base_fn)
  native = primary_by_identity_unroll.get((identity_key, unroll), base_fn)

  gfx = sole_arch_for_unroll(unroll)
  if gfx is not None:
    override_unroll = (unroll_override_for_gfx(gfx, *identity_key[:7])
                       or fixed_unroll_for_type(gfx, base_fn.ty))
    overridden = primary_by_identity_unroll.get((identity_key, override_unroll))
    return [(None, native if overridden is None else overridden)]

  # Shared tier of a multi-arch/fat build: one branch per override arch that
  # redirects this identity away from its native/default row here, in
  # deterministic (sorted) order so the emitted #if/#elif chain is stable.
  branches = []
  for gfx in override_archs:
    override_unroll = unroll_override_for_gfx(gfx, *identity_key[:7])
    if override_unroll is None:
      continue
    overridden = primary_by_identity_unroll.get((identity_key, override_unroll))
    if overridden is None or overridden == native:
      continue
    branches.append((gfx, overridden))
  branches.append((None, native))
  return branches

def unroll_table_aliases(unroll):
  """(funcId, base_fn, gfx_or_None, actual_fn) rows where a dispatch branch
  for this table's funcId does not use the native unroll-`unroll` variant --
  either because an override redirects it elsewhere for some/all archs, or
  because no native variant was generated at all. gfx_or_None identifies
  which arch's branch this is (None for the fallback branch, which is only
  reported if IT also differs from native, e.g. no native variant exists)."""
  aliases = []
  for index, base_fn in enumerate(func_id_axis):
    # No default here (unlike dispatch_branches_for_unroll_table's internal
    # `native`): a missing native variant must compare unequal to whatever
    # branch fn is actually used, so it's always reported as an alias (e.g.
    # func_id_unroll's own table when func_id_unroll == override_unroll).
    native = primary_by_identity_unroll.get((fn_identity_key(base_fn), unroll))
    for gfx, fn in dispatch_branches_for_unroll_table(unroll, base_fn):
      if fn != native:
        aliases.append((index, base_fn, gfx, fn))
  return aliases

alias_log_lines = []
for unroll in local_unroll:
  aliases = unroll_table_aliases(unroll)
  if not aliases:
    continue
  alias_log_lines.append(
    "ncclDevKernel_Generic_%s: %d specialized function(s) redirected away from "
    "their native unroll-%s variant"
    % (unroll, len(aliases), unroll)
  )
  for index, base_fn, gfx, actual_fn in aliases:
    sym = "ncclDevFunc_" + fn_sym(actual_fn)
    arch_note = " [%s only]" % gfx if gfx else ""
    alias_log_lines.append(
      "  funcId %4d  %s%s  -> %s (compiled at unroll %s)"
      % (index, paste(" ", base_fn.coll, base_fn.algo, base_fn.proto, base_fn.redop,
                      base_fn.ty, base_fn.acc, base_fn.pipeline), arch_note, sym, actual_fn.unroll)
    )

alias_log_path = os.path.join(gensrc, "unroll_table_aliases.log")
with open(alias_log_path, "w") as alias_log:
  alias_log.write("\n".join(alias_log_lines))
  if alias_log_lines:
    alias_log.write("\n")

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

  # Function-pointer table. Only the builds that dispatch through it at RUNTIME
  # emit it: the device-linker build and the legacy indirect-function-call build.
  out("#if defined(USE_INDIRECT_FUNCTION_CALL) || defined(RCCL_DEVICE_LINKER)\n")
  out("typedef void(*ncclDevFuncPtr_t)();\n\n")
  for unroll in all_unrolls:
    rows = func_id_axis if unroll in local_unroll else []
    out("static __device__ ncclDevFuncPtr_t const ncclDevFuncTable_%s[] = {\n" % unroll)
    for index, base_fn in enumerate(rows):
      # Usually a single (arch-agnostic) branch; multiple branches only arise
      # on a shared unroll tier of a multi-arch/fat build where different
      # archs want different overrides for this funcId -- see
      # dispatch_branches_for_unroll_table().
      branches = dispatch_branches_for_unroll_table(unroll, base_fn)
      multi = len(branches) > 1
      for i, (gfx, fn) in enumerate(branches):
        if multi:
          if i == 0:
            out("#if defined(__%s__)\n" % gfx)
          elif gfx is None:
            out("#else\n")
          else:
            out("#elif defined(__%s__)\n" % gfx)
        sym = "ncclDevFunc_" + fn_sym(fn)
        if gfx is not None:
          # An arch-selected branch: build_unroll_override_index() already
          # validated that fn's own get_arch_guard() (if any) is implied by this
          # branch's `defined(__gfxN__)` selector, so re-checking it would be a
          # redundant always-true nested guard.
          out("/*%4d*/ %s,\n" % (index, sym))
        else:
          guard = get_arch_guard(fn)
          if guard:
            out("#if %s\n/*%4d*/ %s,\n#else\n/*%4d*/ nullptr,\n#endif\n" % (guard, index, sym, index))
          else:
            out("/*%4d*/ %s,\n" % (index, sym))
      if multi:
        out("#endif\n")
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
      unroll_fns = func_id_axis if unroll in local_unroll else []
      for i, base_fn in enumerate(unroll_fns):
        # Must match the symbol emitted for the forward declarations / table /
        # DEFINE_ncclDevFunc above: fn_sym() omits the reg suffix when reg=="0",
        # so calling by name here must use it too (plain *fn would append the
        # "_0" reg field and reference an undeclared symbol, breaking the
        # pure-RDC / --no-device-linker build).
        # Usually a single (arch-agnostic) branch; multiple branches only
        # arise on a shared unroll tier of a multi-arch/fat build -- see
        # dispatch_branches_for_unroll_table().
        branches = dispatch_branches_for_unroll_table(unroll, base_fn)
        spec = f"template<> struct Caller{unroll}<{i}, {i+1}> {{ static __forceinline__ __device__ void call{unroll}(unsigned short) noexcept"
        multi = len(branches) > 1
        for bi, (gfx, fn) in enumerate(branches):
          if multi:
            if bi == 0:
              out("#if defined(__%s__)\n" % gfx)
            elif gfx is None:
              out("#else\n")
            else:
              out("#elif defined(__%s__)\n" % gfx)
          sym = "ncclDevFunc_" + fn_sym(fn)
          if gfx is not None:
            # See the matching note in the function-pointer table above.
            out(f"{spec} {{ {sym}(); }} }};\n")
            continue
          guard = get_arch_guard(fn)
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
        if multi:
          out("#endif\n")
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

  # host_table entries map device functions based on collective, algorithm, protocol, redop, and datatype.
  # funcId is independent of unroll, so only one row per identity is needed;
  # use the func_id_unroll slice (not a positional slice: some identities, e.g.
  # FP8 or unroll-8-clamped kernels, aren't generated at every unroll, so the
  # per-unroll slices of func_rows aren't equal size).
  # Each field is packed into 4 bits of the key below.
  for name, values in (("all_colls", all_colls), ("all_algos", all_algos),
                       ("all_protos", all_protos), ("all_redops", all_redops),
                       ("all_tys", all_tys), ("all_accs", all_accs),
                       ("all_pipelines", all_pipelines), ("all_regs", all_regs)):
    assert len(values) <= 16, \
      "Error: %s has more than 16 values, which exceeds 4-bit capacity." % name

  for fn in [fn for fn in func_rows if fn.unroll == func_id_unroll]:
    fn_id = primary_to_index[Fn(*equivalent_primary(*fn))]
    comment = " // " + paste(" ", *fn)
    # Field indexes in order (coll, algo, proto, redop, ty, acc, pipeline, reg)
    coll_idx = all_colls.index(fn.coll)
    algo_idx = all_algos.index(fn.algo)
    proto_idx = all_protos.index(fn.proto)
    redop_idx = all_redops.index(fn.redop)
    ty_idx = all_tys.index(fn.ty)
    acc_idx = all_accs.index(fn.acc)
    pipeline_idx = all_pipelines.index(fn.pipeline)
    reg_idx = all_regs.index(fn.reg)
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
    if fn.coll in ["SendRecv", "AlltoAllPivot", "AlltoAllGda", "AlltoAllvGda"]:
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
      '#include "common.h"\n'
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

# gensrc (including any previous `specialized/`) was already emptied above.
specialized_dir = os.path.join(gensrc, "specialized")
os.makedirs(specialized_dir)

specialized_filelist = []
for fn in primary_funcs:
  sym = fn_sym(fn)
  func_name = "ncclDevFunc_" + sym
  lower_coll = coll_camel_to_lower[fn.coll]
  guard = get_arch_guard(fn)

  filename = "specialized_%s.cpp" % sym.lower()
  filepath = os.path.join(specialized_dir, filename)
  specialized_filelist.append((filename, func_name, guard, fn))

  with open(filepath, "w") as f:
    out = f.write
    out('#include "common.h"\n')
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
    cmake_guard = guard or ""
    # In an exclusive tier get_arch_guard() reports only the owning arch (the
    # unroll check wins over the LL128 one), but CMake must still skip LL128
    # kernels when LL128 is disabled.
    if fn.unroll in _EXCLUSIVE_UNROLL_TIERS and fn.proto == "LL128":
      cmake_guard += " && defined(ENABLE_LL128)"
    f.write("%s %s %s\n" % (filename, func_name, cmake_guard))

print("-- Generated %d specialized kernel files in %s" % (len(specialized_filelist), specialized_dir))
