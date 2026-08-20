#!/usr/bin/env python3
"""
Generate one translation unit per (type, redop) instantiation of
ncclCeLocalReduceKernelVec.

Previously all 40 instantiations lived in a single ce_reduce.cc, which meant
the device linker paid for all of them in one single-threaded LLVM bitcode
link + O3 codegen pass (ld.lld). Two of those 40 kernels -- int8_t/uint8_t
Min and Max -- balloon to ~56K instructions each (vs ~5-6K for Sum/Prod on
the same types) because the compiler can't vectorize a byte-granularity
compare-select the way it does packed add/mul, and that alone made the whole
TU take ~48 minutes to compile. Splitting into one file per instantiation
lets ninja build them in parallel like every other device TU, bounding the
wall-clock cost by the slowest single kernel instead of the sum of all 40.

Output: one ce_reduce_impl.h (shared VecTrait/ReduceOp/kernel-template
definitions) plus one ce_reduce_<type>_<redop>.cpp per instantiation, each
defining a small host-callable launcher that ce_reduce.cc's dispatcher calls
into. The .cpp extension is intentional: it matches the existing
gensrc/*.cpp filter in src/CMakeLists.txt so these files are automatically
included in the non-device-linker (RDC) build and automatically excluded
(compiled separately by cmake/DeviceLinker.cmake instead) in the
device-linker build, with no additional filter rules needed.

The actual C++/HIP code lives in ce_reduce_impl.h.in (copied verbatim) and
ce_reduce_launcher.cpp.in (a string.Template with $tag/$ctype/$redname/
$redval/$vec_define placeholders) alongside this script, not inlined here,
so it reads and edits like normal device code.
"""
import os
import shutil
import string
import sys

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
out_dir = sys.argv[1]

if os.path.exists(out_dir):
    for name in os.listdir(out_dir):
        path = os.path.join(out_dir, name)
        if os.path.isfile(path) or os.path.islink(path):
            os.remove(path)
        elif os.path.isdir(path):
            shutil.rmtree(path)
else:
    os.makedirs(out_dir)


def read_template(name):
    with open(os.path.join(SCRIPT_DIR, name)) as f:
        return f.read()


COPYRIGHT = """/*
Copyright (c) 2025 Advanced Micro Devices, Inc. All rights reserved.

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
*/
"""

IMPL_HEADER = COPYRIGHT + read_template("ce_reduce_impl.h.in")
LAUNCHER_TEMPLATE = string.Template(COPYRIGHT + read_template("ce_reduce_launcher.cpp.in"))

TYPES = [
    ("f32", "float"),
    ("f64", "double"),
    ("f16", "__half"),
    ("bf16", "hip_bfloat16"),
    ("i32", "int32_t"),
    ("u32", "uint32_t"),
    ("i64", "int64_t"),
    ("u64", "uint64_t"),
    ("i8", "int8_t"),
    ("u8", "uint8_t"),
]

REDOPS = [
    ("Sum", 0),
    ("Prod", 1),
    ("Min", 2),
    ("Max", 3),
]

# (type, redop) pairs where measurement (gfx950, >=1MB/rank chunks) showed
# leaving the loop vectorizer enabled on the rank-reduction loop is faster
# than disabling it -- see the NOTE above ncclCeLocalReduceKernelVec. Not in
# this set => vectorize(disable), which is faster (or a wash) for everything
# else, most dramatically int32_t/uint32_t Sum/Prod (~2.2x).
VECTORIZE_OK = {
    ("i8", "Min"),
    ("i8", "Max"),
    ("i8", "Sum"),
    ("u8", "Sum"),
}

with open(os.path.join(out_dir, "ce_reduce_impl.h"), "w") as f:
    f.write(IMPL_HEADER)

# Per-type, redop-independent reporting wrapper: block count without launching.
# Emitted once per type (in the Sum TU) so it has exactly one definition; both
# the launcher and this wrapper route through ncclCeLocalReduceBlocksT<T>.
BLOCKS_FN = string.Template(
    "\n// Reporting wrapper: reduce-kernel block count for host-side impl selection.\n"
    "int ncclCeLocalReduceBlocks_${tag}(size_t chunkElems) {\n"
    "  return ncclCeLocalReduceBlocksT<${ctype}>(chunkElems);\n"
    "}\n")

for tag, ctype in TYPES:
    for redname, redval in REDOPS:
        fname = "ce_reduce_%s_%s.cpp" % (tag, redname)
        vec_define = "#define CE_REDUCE_VECTORIZE_OK\n" if (tag, redname) in VECTORIZE_OK else ""
        blocks_fn = BLOCKS_FN.substitute(tag=tag, ctype=ctype) if redname == "Sum" else ""
        with open(os.path.join(out_dir, fname), "w") as f:
            f.write(LAUNCHER_TEMPLATE.substitute(tag=tag, ctype=ctype, redname=redname, redval=redval,
                                                  vec_define=vec_define, blocks_fn=blocks_fn))

print("-- Generated %d CE-reduce kernel TUs in %s" % (len(TYPES) * len(REDOPS), out_dir))
