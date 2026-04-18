#!/usr/bin/env python3
"""AOT-compile a single Triton kernel file for both gfx942 and gfx1250, and
write a `<name>.sidecar.json` that compare_correctness consumes at run time.

Invoked from the compare_correctness Makefile, one invocation per kernel file.
The kernel file must define one or more ``@triton.jit`` functions plus a
``RECIPES = [...]`` list whose entries are recipe specifications.  See
``kernels/triton/README.md`` for the schema.

Outputs per recipe (one recipe per file is supported in Phase 1):

  <out_dir>/<name>.gfx942.co       — AOT-compiled code object for wave64
  <out_dir>/<name>.gfx1250.co      — AOT-compiled code object for wave32
  <out_dir>/<name>.sidecar.json    — signature / shape / grid / comparator +
                                     kernarg-layout metadata extracted from
                                     each .co via llvm-readelf.  The C++
                                     harness uses this to pack kernargs at
                                     the exact offsets the code object
                                     expects — no hand-written struct per
                                     Triton kernel.

Environment:
  Needs a working Triton install.  The Makefile points TRITON_VENV and
  TRITON_PYTHONPATH at the tree the rest of the repo already uses
  ($(HOME)/rocm-systems/triton).  PyYAML is required (shipped with that
  Triton venv) and llvm-readelf is located via $ROCM or $PATH.
"""
import argparse
import importlib.util
import json
import os
import subprocess
import sys
import tempfile
from shutil import which

import yaml  # provided by the Triton venv (PyYAML >= 6.0)

# Triton's compile surface.
import triton
from triton.compiler.compiler import (
    ASTSource,
    GPUTarget,
    compile as triton_compile,
)

# Per-target wave size.  Must match the hardware definition: gfx942 is wave64
# and gfx1250 is wave32.  Triton's GPUTarget takes the warp size as its
# third arg, which affects lowering decisions (reduction patterns, wmma
# sizing, etc.).
TARGETS = [
    ("gfx942", 64),
    ("gfx1250", 32),
]

# Places to look for llvm-readelf, in preference order.  We try the ROCm
# install first so the readelf matches the HSACO version we just produced.
READELF_CANDIDATES = [
    os.environ.get("LLVM_READELF", ""),
    "/opt/rocm-7.2.1/lib/llvm/bin/llvm-readelf",
    "/opt/rocm/lib/llvm/bin/llvm-readelf",
    "llvm-readelf",
]


def find_readelf() -> str:
    for p in READELF_CANDIDATES:
        if not p:
            continue
        if os.path.isabs(p) and os.path.exists(p):
            return p
        if not os.path.isabs(p):
            hit = which(p)
            if hit:
                return hit
    raise RuntimeError(
        "llvm-readelf not found. Tried: "
        + ", ".join(x for x in READELF_CANDIDATES if x)
    )


def load_kernel_module(path: str):
    """Import a .py kernel file as a fresh module and return it."""
    spec = importlib.util.spec_from_file_location(
        "compare_correctness_kernel", path
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import kernel file {path}")
    mod = importlib.util.module_from_spec(spec)
    # Triton's @jit decorator resolves source via inspect, which needs the
    # module registered under its name.
    sys.modules["compare_correctness_kernel"] = mod
    spec.loader.exec_module(mod)
    if not hasattr(mod, "RECIPES"):
        raise RuntimeError(
            f"{path}: missing module-level RECIPES = [...] list. "
            "See kernels/triton/README.md for the schema."
        )
    return mod


def compile_for_target(kernel_fn, signature, constexprs, num_warps, arch, warp_size):
    """AOT-compile one Triton kernel for one architecture.  Returns the raw
    .hsaco bytes."""
    target = GPUTarget("hip", arch, warp_size)
    src = ASTSource(
        fn=kernel_fn,
        signature=dict(signature),
        constexprs=dict(constexprs),
        attrs={"num_warps": num_warps},
    )
    compiled = triton_compile(src, target=target)
    hsaco = compiled.asm.get("hsaco")
    if hsaco is None:
        raise RuntimeError(
            f"triton_compile did not produce hsaco for {arch} "
            f"(keys: {list(compiled.asm.keys())})"
        )
    if isinstance(hsaco, str):
        with open(hsaco, "rb") as f:
            return f.read()
    return bytes(hsaco)


def extract_metadata(co_bytes: bytes, kernel_symbol: str) -> dict:
    """Parse .note.amdgpu_metadata from an HSACO and return the per-kernel
    layout info the dispatch needs.  Includes private_segment_fixed_size so
    the C++ harness can assert the kernel needs no scratch (Phase 1 limit:
    we don't allocate a scratch buffer for global_scratch_base implicit
    kernarg slots)."""
    readelf = find_readelf()
    with tempfile.NamedTemporaryFile(suffix=".co", delete=False) as tmp:
        tmp.write(co_bytes)
        tmp_path = tmp.name
    try:
        raw = subprocess.check_output([readelf, "--notes", tmp_path])
    finally:
        os.unlink(tmp_path)
    text = raw.decode("utf-8", errors="replace")

    # llvm-readelf wraps the AMDGPU metadata YAML in a non-YAML header.  The
    # payload starts with a '---' document marker and ends with '...'; we
    # extract everything between.
    lines = text.splitlines()
    start = end = None
    for i, ln in enumerate(lines):
        s = ln.strip()
        if s == "---" and start is None:
            start = i
        elif s == "..." and start is not None:
            end = i
            break
    if start is None or end is None:
        raise RuntimeError(
            "llvm-readelf output did not contain an AMDGPU metadata YAML "
            "block (missing '---' / '...' markers)"
        )
    block = "\n".join(lines[start + 1 : end])
    doc = yaml.safe_load(block)
    kernels = doc.get("amdhsa.kernels") or []
    for k in kernels:
        if k.get(".name") == kernel_symbol:
            return {
                "kernarg_segment_size": int(k.get(".kernarg_segment_size", 0)),
                "group_segment_fixed_size": int(
                    k.get(".group_segment_fixed_size", 0)
                ),
                "private_segment_fixed_size": int(
                    k.get(".private_segment_fixed_size", 0)
                ),
                "max_flat_workgroup_size": int(
                    k.get(".max_flat_workgroup_size", 256)
                ),
                # All kernarg slots, in offset order, INCLUDING the implicit
                # trailing args Triton appends past the user signature
                # (typically global_scratch_base and similar).  The C++
                # dispatch only writes the first N slots (N == len(signature))
                # and trusts that the trailing slots are safe to leave at
                # zero, which the matching private_segment_fixed_size == 0
                # assertion enforces.
                "args": [
                    {
                        "offset": int(a.get(".offset", 0)),
                        "size": int(a.get(".size", 0)),
                        "value_kind": a.get(".value_kind", ""),
                    }
                    for a in k.get(".args", [])
                ],
            }
    raise RuntimeError(
        f"kernel symbol {kernel_symbol!r} not found in AMDGPU metadata "
        f"(present: {[k.get('.name') for k in kernels]})"
    )


def validate_recipe(recipe: dict, path: str) -> None:
    required = [
        "name",
        "kernel_fn",
        "kernel_symbol",
        "signature",
        "num_warps",
        "shape_dim",
        "default_shapes",
        "grid",
        "inputs",
        "outputs",
        "comparator",
    ]
    missing = [k for k in required if k not in recipe]
    if missing:
        raise RuntimeError(f"{path}: recipe is missing keys: {missing}")
    if not isinstance(recipe["default_shapes"], list) or not recipe["default_shapes"]:
        raise RuntimeError(f"{path}: default_shapes must be a non-empty list")
    if recipe["comparator"].get("kind") not in ("abs", "rel"):
        raise RuntimeError(f"{path}: comparator.kind must be 'abs' or 'rel'")
    for axis in ("x", "y", "z"):
        if axis not in recipe["grid"]:
            raise RuntimeError(f"{path}: grid missing axis {axis!r}")
    sig_names = set(recipe["signature"].keys())
    for b in recipe["inputs"] + recipe["outputs"]:
        if b["name"] not in sig_names:
            raise RuntimeError(
                f"{path}: input/output {b['name']!r} not declared in signature"
            )
    # Mirror the C++ harness's Phase-1 mixed-dtype refusal at AOT time so a
    # mis-shaped recipe fails the build instead of producing a sidecar that
    # crashes compare_correctness on first launch.  Keep both checks in
    # sync; tritonCompare reinterprets the entire output blob as the first
    # output's dtype, so any mismatch is silently wrong.
    outs = recipe["outputs"]
    if not outs:
        raise RuntimeError(f"{path}: at least one output is required")
    first_dtype = outs[0]["dtype"]
    for b in outs[1:]:
        if b["dtype"] != first_dtype:
            raise RuntimeError(
                f"{path}: Phase 1 requires all outputs share dtype "
                f"(got {first_dtype!r} and {b['dtype']!r}); split the "
                f"recipe or extend tritonCompare in compare_correctness.cpp"
            )


def build_recipe(recipe: dict, out_dir: str, kernel_file: str) -> None:
    name = recipe["name"]
    expected_stem = os.path.splitext(os.path.basename(kernel_file))[0]
    if name != expected_stem:
        # We use the file stem as the Makefile's % pattern and as the .co
        # filename root, so keeping recipe name == file stem avoids a whole
        # class of "which file produced which recipe?" confusions.
        raise RuntimeError(
            f"{kernel_file}: recipe name {name!r} must match file stem "
            f"{expected_stem!r} (Phase 1 uses one recipe per file)"
        )

    metadata = {}
    for arch, warp_size in TARGETS:
        data = compile_for_target(
            recipe["kernel_fn"],
            recipe["signature"],
            recipe.get("constexprs", {}),
            recipe["num_warps"],
            arch,
            warp_size,
        )
        co_path = os.path.join(out_dir, f"{name}.{arch}.co")
        with open(co_path, "wb") as f:
            f.write(data)
        md = extract_metadata(data, recipe["kernel_symbol"])
        metadata[arch] = md
        print(
            f"  [{arch:7s}] wrote {co_path} "
            f"({len(data):5d} bytes, kernarg={md['kernarg_segment_size']}, "
            f"lds={md['group_segment_fixed_size']}, "
            f"scratch={md['private_segment_fixed_size']}, "
            f"wg={md['max_flat_workgroup_size']})"
        )

    sidecar = {
        "name": name,
        "kernel_symbol": recipe["kernel_symbol"],
        "num_warps": recipe["num_warps"],
        "signature": [
            {"name": k, "type": v} for k, v in recipe["signature"].items()
        ],
        "constexprs": dict(recipe.get("constexprs", {})),
        "shape": {
            "dim": recipe["shape_dim"],
            "values": [int(v) for v in recipe["default_shapes"]],
        },
        "grid": {k: str(v) for k, v in recipe["grid"].items()},
        "inputs": [
            {"name": b["name"], "dtype": b["dtype"], "elems": str(b["elems"])}
            for b in recipe["inputs"]
        ],
        "outputs": [
            {"name": b["name"], "dtype": b["dtype"], "elems": str(b["elems"])}
            for b in recipe["outputs"]
        ],
        "comparator": {
            "kind": recipe["comparator"]["kind"],
            "tol": float(recipe["comparator"]["tol"]),
        },
        "metadata": metadata,
    }
    sidecar_path = os.path.join(out_dir, f"{name}.sidecar.json")
    with open(sidecar_path, "w") as f:
        json.dump(sidecar, f, indent=2)
        f.write("\n")
    print(f"  wrote {sidecar_path}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--kernel-file", required=True, help="path to a .py file")
    ap.add_argument("--out-dir", required=True, help="output directory for .co/.sidecar.json")
    args = ap.parse_args()

    # Pin the Triton version into the build log.  Triton's internal compile
    # API (ASTSource / GPUTarget / triton_compile) has shifted shape across
    # 2.x → 3.x and intra-3.x; if a sidecar layout regresses, the version
    # printed here is the first thing to compare against the working build.
    print(f"Triton {triton.__version__} (from {os.path.dirname(triton.__file__)})")

    os.makedirs(args.out_dir, exist_ok=True)
    mod = load_kernel_module(args.kernel_file)
    recipes = mod.RECIPES
    if len(recipes) != 1:
        raise RuntimeError(
            f"{args.kernel_file}: expected exactly 1 recipe, found {len(recipes)}. "
            "Phase 1 supports one recipe per file (file stem == recipe name)."
        )
    for recipe in recipes:
        validate_recipe(recipe, args.kernel_file)
        print(f"Building {recipe['name']} …")
        build_recipe(recipe, args.out_dir, args.kernel_file)
    return 0


if __name__ == "__main__":
    sys.exit(main())
