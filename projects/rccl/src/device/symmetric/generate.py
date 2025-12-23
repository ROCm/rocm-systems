# Modification Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT 

#!/usr/bin/env python3
import os
import sys

################################################################################
# The first command line argument is the path to the directory to generate and
# populate.

gensrc = sys.argv[1]

if os.path.exists(gensrc):
  for name in os.listdir(gensrc):
    os.remove(os.path.join(gensrc, name))
    #os.truncate(os.path.join(gensrc, name), 0)
else:
  os.mkdir(gensrc)

def paste(sep, *args):
  return sep.join(args)

indents = 0
def emitln(f, lines):
  global indents
  for ln in ((lines,) if isinstance(lines, str) else lines):
    f.write('  '*indents + ln + '\n')

def indent(s):
  return '\n'.join('  '+l for l in s.splitlines())

class Rec(object):
  def __init__(me, **kw):
    me.__dict__.update(kw)
  def __eq__(x, y):
    if len(x) != len(y): return False
    for k in x:
      if k not in y: return False
      if x[k] != y[k]: return False
    return True
  def __hash__(me):
    h = 0
    for k in me.__dict__:
      h += hash((k, me.__dict__[k]))
    return h

################################################################################
# Edit this region for introducing new algos etc

reductions = ["AllReduce","ReduceScatter"]
all_reds = ["sum"]
all_tys = ["f32","f16","bf16","f8e4m3","f8e5m2"]

nvls_algos_by_coll = {
  "AllReduce": ["AGxLLMC_R","RSxLDMC_AGxSTMC"],
  "ReduceScatter": ["LDMC"]
}
ldmc_algos = ["RSxLDMC_AGxSTMC", "LDMC"]

coll_to_lower = {
  "AllGather": "all_gather",
  "AllReduce": "all_reduce",
  "ReduceScatter": "reduce_scatter"
}

red_to_ncclDevRedOp = {
  "sum": "ncclDevSum"
}
red_to_Func = {
  "sum": "FuncSum"
}

ty_to_ncclDataType = {
  "f32": "ncclFloat32",
  "f16": "ncclFloat16",
  "bf16": "ncclBfloat16",
  "f8e4m3": "ncclFloat8e4m3",
  "f8e5m2": "ncclFloat8e5m2"
}
ty_to_cxxtype = {
  "f32": "float",
  "f16": "half",
  "bf16": "hip_bfloat16",
  "f8e4m3": "rccl_float8",
  "f8e5m2": "rccl_bfloat8"
}

def enumerate_kernels():
  for algo in ["LL","ST"]:
    yield Rec(coll="AllGather", algo=algo)
  for red in all_reds:
    for ty in all_tys:
      for algo in ["AGxLL_R","RSxLD_AGxST"]:
        yield Rec(coll="AllReduce", algo=algo, red=red, ty=ty)
      for algo in ["LL","LD"]:
        yield Rec(coll="ReduceScatter", algo=algo, red=red, ty=ty)

def required_cuda(k):
  cudart, arch, specific_sms  = 0, 0, None
  is_nvls = k.algo in nvls_algos_by_coll.get(k.coll, [])
  if is_nvls:
    cudart = max(cudart, 12010)
    arch = 900
  if k.coll in reductions:
    if k.ty == "bf16":
      cudart = max(cudart, 11000)
    if k.ty.startswith("f8"):
      cudart = max(cudart, 11080)
      arch = 900
      if k.algo in ldmc_algos:
        cudart = 12070
        arch = None
        specific_sms = ["100a", "101a", "100f", "101f", "120a", "121a"]
  return (cudart, arch, specific_sms)

################################################################################

def kernel_fdep(k):
  return coll_to_lower[k.coll] + '.cpp'

def kernel_fname(k):
  if k.coll in reductions:
    if k.algo in ldmc_algos and k.ty.startswith('f8'):
      return paste('_', coll_to_lower[k.coll], k.red, k.ty, k.algo) + '.cpp'
    else:
      return paste('_', coll_to_lower[k.coll], k.red, k.ty) + '.cpp'
  else:
    return coll_to_lower[k.coll] + '.cpp'

def kernel_gencode(k):
  if k.coll in reductions and k.algo in ldmc_algos and k.ty.startswith('f8'):
    return "$(NVCC_GENCODE_LDMC_FP8)"
  else:
    return "$(NVCC_GENCODE)"

def kernel_cname(k):
  if k.coll in reductions:
    return paste("_", "ncclSymDevKernel", k.coll, k.algo, k.red, k.ty)
  else:
    return paste("_", "ncclSymDevKernel", k.coll, k.algo)

def kernel_conds(k):
  cudart, arch, specific_sms = required_cuda(k)
  if cudart == 0: return (None, None)

  cudart_cond = "CUDART_VERSION >= %d"%cudart
  if not specific_sms:
    arch_cond = "__CUDA_ARCH__ >= %d"%arch
  else:
    arch_cond = " || ".join(["0"] + ["NCCL_CUDA_ARCH_%sSPECIFIC==%d"%("FAMILY_" if sm[-1] == "f" else "", 10*int(sm.replace('a', '').replace('f', ''))) for sm in specific_sms])
  return cudart_cond, arch_cond

def instantiate(k):
  form_red_ty = (
    "__global__ void {cname}(ncclSymDevArgs NCCL_GRID_CONSTANT const *args) {{\n"
    "  ncclSymRun_{id}<{red}, {ty}>(args);\n"
    "}}"
  )
  form = (
    "__global__ void {cname}(ncclSymDevArgs NCCL_GRID_CONSTANT const *args) {{\n"
    "  ncclSymRun_{id}(args);\n"
    "}}"
  )

  id = k.coll+'_'+k.algo
  cname = kernel_cname(k)
  if k.coll in reductions:
    inst = form_red_ty.format(cname=cname, id=id, red=red_to_Func[k.red], ty=ty_to_cxxtype[k.ty])
  else:
    inst = form.format(cname=cname, id=id)
  return inst

def prototype(k):
  return "__global__ void {cname}(ncclSymDevArgs const *args);".format(cname=kernel_cname(k))

################################################################################

def partition(vals, keyfn):
  ans = {}
  for x in vals:
    k = keyfn(x)
    if k not in ans:
      ans[k] = []
    ans[k].append(x)
  return ans


kernels_by_file = partition(enumerate_kernels(), lambda k: (kernel_fname(k), k.coll))

# Add dependency only files (e.g. allreduce.cpp)
for coll in set(k.coll for k in enumerate_kernels()):
  fname = coll_to_lower[coll]+'.cpp'
  if (fname, coll) not in kernels_by_file:
    kernels_by_file[fname, coll] = []

# Generate each kernel instantiation file
for (fname, coll), ks in kernels_by_file.items():
  with open(os.path.join(gensrc, fname), "w") as f:
    print("-- Generating %s" % os.path.join(gensrc, fname))
    emitln(f, '#include "symmetric.h"')
    emitln(f, '#include "symmetric/kernel.h"')
    emitln(f, '#include "symmetric/{coll}.h"'.format(coll=coll_to_lower[coll]))
    for k in ks:
      emitln(f, instantiate(k))

# Generate <gensrc>/symmetric_host.cc
with open(os.path.join(gensrc, "symmetric_kernels.cc"), "w") as f:
  print("-- Generating %s" % os.path.join(gensrc, "symmetric_kernels.cc"))
  emitln(f, '#include "symmetric.h"')
  emitln(f, '#include "device.h"')
  emitln(f, '')

  for k in enumerate_kernels():
    emitln(f, prototype(k))
  emitln(f, '')

  emitln(f, 'extern int const ncclSymKernelCount = %d;' % len(list(enumerate_kernels())))
  emitln(f, 'extern void* const ncclSymKernelList[] = {')
  for k in enumerate_kernels():
    emitln(f, '(void*){cname},'.format(cname=kernel_cname(k)))
  emitln(f, 'nullptr};')
  emitln(f, '')

  emitln(f, 'void* ncclSymGetKernelPtr(ncclSymKernelId id, int red, ncclDataType_t ty) {')
  indents += 1
  emitln(f, 'switch (id) {')
  emitln(f, 'default: return nullptr;')
  for (coll, algo), coll_algo_ks in partition(enumerate_kernels(), lambda k: (k.coll, k.algo)).items():
    emitln(f, 'case ncclSymKernelId_'+coll+'_'+algo+':')
    indents += 1
    if len(coll_algo_ks) == 1:
      emitln(f, 'return (void*)&'+kernel_cname(coll_algo_ks[0])+';')
    else:
      emitln(f, 'switch ((ncclDevRedOp_t)red) {')
      emitln(f, 'default: return nullptr;')
      for red, coll_algo_red_ks in partition(coll_algo_ks, lambda k: k.red).items():
        emitln(f, 'case '+red_to_ncclDevRedOp[red]+':')
        indents += 1
        emitln(f, 'switch (ty) {')
        emitln(f, 'default: return nullptr;')
        for k in coll_algo_red_ks:
          emitln(f, 'case '+ty_to_ncclDataType[k.ty]+': return (void*)'+kernel_cname(k)+';')
        emitln(f, '}')
        indents -= 1
      emitln(f, '}')
    indents -=1
  emitln(f, '}')
  indents -= 1
  emitln(f, '}')
