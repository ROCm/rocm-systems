#!/usr/bin/env python3
"""
generate_wrappers.py — Generates hip_clr_dispatch_wrappers.cpp.

Reads:
  include/hip/amd_detail/hip_api_trace.hpp   — typedef signatures
  src/hip_api_trace.cpp                       — UpdateDispatchTable field list

Outputs:
  src/profiler/hip_clr_dispatch_wrappers.cpp

Each wrapper uses the two-phase API:
  auto* _rec = HipClrProfilerPreCall(api_id);
  const uint64_t _s = NowNsW();
  auto _r = g_next.hipFoo_fn(...);   // or g_next.hipFoo_fn(...); for void
  HipClrProfilerPostCall(_rec, _s, NowNsW());
  return _r;  // omitted for void
"""

import re
import sys
import os

SCRIPT_DIR  = os.path.dirname(os.path.abspath(__file__))
HIPAMD_DIR  = os.path.normpath(os.path.join(SCRIPT_DIR, "..", ".."))
TRACE_HPP   = os.path.join(HIPAMD_DIR, "include/hip/amd_detail/hip_api_trace.hpp")
TRACE_CPP   = os.path.join(HIPAMD_DIR, "src/hip_api_trace.cpp")
OUT_FILE    = os.path.join(SCRIPT_DIR, "hip_clr_dispatch_wrappers.cpp")

# ---------------------------------------------------------------------------
# Step 1 — Parse typedefs from hip_api_trace.hpp
#   typedef <ret> (*t_<name>)(<params>);
# ---------------------------------------------------------------------------
typedef_re = re.compile(
    r'typedef\s+([\w\s\*]+?)\s*\(\s*\*\s*t_(\w+)\s*\)\s*\(([^)]*)\)\s*;'
)

typedefs = {}   # name -> (ret_type, params_str)
with open(TRACE_HPP, encoding="utf-8") as f:
    content = f.read()

for m in typedef_re.finditer(content):
    ret  = m.group(1).strip()
    name = m.group(2).strip()
    params = m.group(3).strip()
    typedefs[name] = (ret, params)

print(f"Parsed {len(typedefs)} typedefs from {TRACE_HPP}")

# ---------------------------------------------------------------------------
# Step 2 — Parse field assignments from UpdateDispatchTable in hip_api_trace.cpp
#   tbl-><name>_fn = <name>;   or  tbl-><name>_fn = hip::<name>;
# ---------------------------------------------------------------------------
assign_re = re.compile(r'\w+\s*->\s*(\w+)_fn\s*=\s*(?:hip::)?(\w+)\s*;')

field_order = []  # list of (field_name, impl_name) in source order
with open(TRACE_CPP, encoding="utf-8") as f:
    cpp_content = f.read()

# Find the UpdateDispatchTable(HipDispatchTable* ...) overload body.
# There are multiple overloads; pick the one with HipDispatchTable parameter.
body_m = re.search(
    r'UpdateDispatchTable\s*\(\s*HipDispatchTable\s*\*[^)]*\)\s*\{(.*?)\n\}',
    cpp_content, re.DOTALL)
if not body_m:
    sys.exit("ERROR: Could not find UpdateDispatchTable(HipDispatchTable*) body in " + TRACE_CPP)

seen_fields = set()
for m in assign_re.finditer(body_m.group(1)):
    name = m.group(1)
    if name not in seen_fields:
        seen_fields.add(name)
        field_order.append((name, m.group(2)))

print(f"Parsed {len(field_order)} field assignments from {TRACE_CPP}")

# ---------------------------------------------------------------------------
# Step 3 — Parse param names from a params string
#   Returns list of bare parameter names (stripping types).
#   Handles void, pointers, arrays, const, etc.
# ---------------------------------------------------------------------------
def _split_params(params_str):
    """Split a parameter string into individual parameter strings."""
    if not params_str or params_str.strip() in ('', 'void'):
        return []
    parts = []
    depth = 0
    current = []
    for ch in params_str:
        if ch in '(<':
            depth += 1; current.append(ch)
        elif ch in ')>':
            depth -= 1; current.append(ch)
        elif ch == ',' and depth == 0:
            parts.append(''.join(current).strip()); current = []
        else:
            current.append(ch)
    if current:
        parts.append(''.join(current).strip())
    return parts

def _find_param_of_type(params_str, type_name):
    """Return the name of the first value (non-pointer) parameter of type_name, or None."""
    for p in _split_params(params_str):
        toks = p.split()
        for j, tok in enumerate(toks):
            if tok == type_name:
                if j + 1 < len(toks) and toks[j + 1].startswith('*'):
                    continue
                return toks[-1].lstrip('*').strip()
    return None

def find_stream_param(params_str):
    """Return the name of the first hipStream_t (non-pointer) parameter, or None."""
    return _find_param_of_type(params_str, 'hipStream_t')

def find_copy_kind_param(params_str):
    """Return the name of the first hipMemcpyKind parameter, or None."""
    return _find_param_of_type(params_str, 'hipMemcpyKind')

def parse_param_names(params_str):
    if not params_str or params_str.strip() in ('', 'void'):
        return []
    parts = []
    depth = 0
    current = []
    for ch in params_str:
        if ch in '(<':
            depth += 1
            current.append(ch)
        elif ch in ')>':
            depth -= 1
            current.append(ch)
        elif ch == ',' and depth == 0:
            parts.append(''.join(current).strip())
            current = []
        else:
            current.append(ch)
    if current:
        parts.append(''.join(current).strip())

    names = []
    for p in parts:
        p = p.strip()
        if not p or p == 'void':
            continue
        # Strip trailing array specifiers: foo[n]
        p_base = re.sub(r'\[.*?\]', '', p).strip()
        # Last token is the name (handle pointer * attached to name)
        toks = re.split(r'\s+', p_base)
        last = toks[-1].lstrip('*').strip()
        if last:
            names.append(last)
        elif len(toks) > 1:
            names.append(toks[-2].lstrip('*').strip())
    return names

# ---------------------------------------------------------------------------
# Step 4 — Emit the wrapper file
# ---------------------------------------------------------------------------
HEADER = """\
/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 *
 * AUTO-GENERATED — do not edit by hand.
 * Regenerate with:  python generate_wrappers.py
 *
 * Dispatch table wrappers for the HIP CLR built-in profiling layer.
 * Pattern mirrors the reference hip_tracer.cpp:
 *   auto* record = HipGetActiveRecordExt(api_id);  // allocs slot, sets correlation_id TLS
 *   [call real function]
 *   if (record) record->end_ns = NowNs();
 */

#include "hip/amd_detail/hip_api_trace.hpp"
#include "hip_clr_profiler.hpp"
#include "rocclr/os/os.hpp"

#include <atomic>

static inline uint64_t NowNs() { return amd::Os::timeNanos(); }

// Saved original dispatch table (the "next layer").
static HipDispatchTable g_next{};

// Idempotency guard — true while wrappers are installed.
static std::atomic<bool> g_wrapped{false};

"""

missing = []
lines = [HEADER]

# Enumerate wrappers preserving UpdateDispatchTable order
for api_id, (field_name, impl_name) in enumerate(field_order):
    if field_name not in typedefs:
        missing.append(field_name)
        continue

    ret, params_str = typedefs[field_name]
    param_names = parse_param_names(params_str)
    is_void = (ret.strip() == 'void')

    # Build parameter list for the wrapper signature
    if params_str.strip() in ('', 'void'):
        sig_params = 'void'
        call_args  = ''
    else:
        sig_params = params_str
        call_args  = ', '.join(param_names)

    layer_name = f'{field_name}Layer'

    stream_param = find_stream_param(params_str)

    lines.append(f'// api_id = {api_id}')
    lines.append(f'static {ret} {layer_name}({sig_params}) {{')
    lines.append(f'  auto* _rec = HipGetActiveRecordExt({api_id}u);')
    if stream_param:
        lines.append(f'  _rec->stream = {stream_param};')
    if is_void:
        lines.append(f'  g_next.{field_name}_fn({call_args});')
    else:
        lines.append(f'  auto _r = g_next.{field_name}_fn({call_args});')
    lines.append(f'  _rec->end_ns = NowNs();')
    if not is_void:
        lines.append(f'  return _r;')
    lines.append('}')
    lines.append('')

if missing:
    print(f"WARNING: {len(missing)} fields not found in typedefs: {missing[:10]}")

# API name table — indexed by api_id, same order as field_order
lines.append('// API name table — indexed by api_id (same order as UpdateDispatchTable).')
lines.append('const char* const kHipApiNamesExt[] = {')
for field_name, _ in field_order:
    lines.append(f'  "{field_name}",')
lines.append('};')
lines.append(f'const size_t kHipApiNamesCountExt = {len(field_order)};')
lines.append('')

# Install / Remove — build full replacement table first, then memcpy in one shot.
# Each function pointer is 8-byte aligned on x86-64, so individual pointer writes
# are naturally atomic; memcpy emits a burst of such stores.  A thread_fence
# prevents the compiler from reordering the stores relative to g_wrapped.
lines.append('#include <cstring>')
lines.append('')
lines.append('void HipProfilerInstallWrappersExt(HipDispatchTable* tbl) {')
lines.append('  if (g_wrapped.exchange(true)) return;')
lines.append('  g_next = *tbl;')
lines.append('  HipDispatchTable wrapper_tbl = g_next;  // start from a full valid copy')
for field_name, _ in field_order:
    if field_name in typedefs:
        lines.append(f'  wrapper_tbl.{field_name}_fn = {field_name}Layer;')
lines.append('  std::atomic_thread_fence(std::memory_order_release);')
lines.append('  std::memcpy(tbl, &wrapper_tbl, sizeof(HipDispatchTable));')
lines.append('}')
lines.append('')
lines.append('void HipProfilerRemoveWrappersExt(HipDispatchTable* tbl) {')
lines.append('  if (!g_wrapped.exchange(false)) return;')
lines.append('  std::atomic_thread_fence(std::memory_order_release);')
lines.append('  std::memcpy(tbl, &g_next, sizeof(HipDispatchTable));')
lines.append('}')
lines.append('')

with open(OUT_FILE, 'w', encoding='utf-8') as f:
    f.write('\n'.join(lines))

print(f"Wrote {len(field_order) - len(missing)} wrappers -> {OUT_FILE}")
if missing:
    print(f"Skipped {len(missing)} (no typedef): {missing}")
