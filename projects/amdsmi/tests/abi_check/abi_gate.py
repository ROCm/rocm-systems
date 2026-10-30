#!/usr/bin/env python3
# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
"""Blocking ABI gate for amdsmi.h.

Parses amdsmi.h with libclang (no GPU, no library build) into a layout
snapshot: size, alignment and field offsets of every public record, enumerator
values, function signatures and integer macros. Snapshots the PR, its base and
the latest release branch from git and compares them. A break passes only when
projects/amdsmi/abi/abi_exceptions.json lists that exact change and
CHANGELOG.md names it; the file is owned through CODEOWNERS.

Run ``abi_gate.py --help`` for the subcommands; README.md in this directory
describes the workflow.
"""

import argparse
import difflib
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile

HEADER = "projects/amdsmi/include/amd_smi/amdsmi.h"
EXCEPTIONS = "projects/amdsmi/abi/abi_exceptions.json"
CHANGELOG = "projects/amdsmi/CHANGELOG.md"
WORKTREE = "WORKTREE"

DEFINES = ["ENABLE_ESMI_LIB=1"]
# Every macro a public #if may test. Each value is parsed (C11 with and without
# ENABLE_ESMI_LIB, and C++17) and the layouts must agree; anything else is refused.
CONDITIONALS = {"__AMDSMI_H__", "__cplusplus", "ENABLE_ESMI_LIB"}
RESERVED = re.compile(r"^_*(reserved|rsvd|padding|pad|unused)", re.IGNORECASE)
BREAK_WORD = re.compile(r"\b(ABI|breaking)\b", re.IGNORECASE)
SENTINEL = re.compile(r"_(LAST|MAX|COUNT|END)$")
MARKER = re.compile(r"_(UNKNOWN|INVALID|RESERVED|ALL\w*|NONE|UNDEFINED|UNSUPPORTED)$")
MARKER_VALUES = {0xFF, 0xFFFF, 0xFFFFFFFF, (1 << 63), (1 << 64) - 1, -1}
UNPINNED_MACROS = re.compile(r"^(AMDSMI_LIB_VERSION_\w+|__AMDSMI_H__)$")
ANON = re.compile(r"\((?:anonymous|unnamed)(?: struct| union| enum)? at ([^)]*)\)")
INCLUDE = re.compile(r'^\s*#\s*include\s*([<"])([^>"]+)[>"]', re.MULTILINE)
CONDITIONAL = re.compile(r"^\s*#\s*(?:if|ifdef|ifndef|elif|elifdef|elifndef)\b(.*)$", re.MULTILINE)
DIRECTION = re.compile(r"\b(IN|OUT|INPUT|OUTPUT|IN/OUT|INOUT)\b")
READS = re.compile(r"\bin(?:out|put)?\b", re.IGNORECASE)
ATTRIBUTE_TOKENS = {"__attribute__", "__declspec", "alignas", "_Alignas"}

# Per-field and per-parameter meaning, read from the doc comments of caller-allocated
# structs and of parameters. A change blocks like a layout change; see semantic_annotations.
SEMANTIC_KEYS = ("dir", "unit", "sentinel", "encoding")
UNITS = [
    (re.compile(r"\b([GMKk]?Hz)\b"), lambda m: m.group(1).replace("KHz", "kHz")),
    (re.compile(r"\b([munµ]?W)\b"), lambda m: m.group(1).replace("µ", "u")),
    (
        re.compile(r"\b(milli|micro|nano)?watts?\b", re.IGNORECASE),
        lambda m: {"milli": "mW", "micro": "uW", "nano": "nW"}.get((m.group(1) or "").lower(), "W"),
    ),
    (re.compile(r"\b(m?V)\b"), lambda m: m.group(1)),
    (re.compile(r"\b(milli)?volts?\b", re.IGNORECASE), lambda m: "mV" if m.group(1) else "V"),
    (re.compile(r"\bmA\b"), lambda m: "mA"),
    (re.compile(r"°C|\bdegrees? C(?:elsius)?\b|\bCelsius\b"), lambda m: "C"),
    (re.compile(r"\bmillidegrees?\b"), lambda m: "mC"),
    (re.compile(r"(?:\d\s?|\bin )(ms|us|µs|ns|s)\b"), lambda m: m.group(1).replace("µ", "u")),
    (
        re.compile(r"\b(milli|micro|nano)?seconds?\b", re.IGNORECASE),
        lambda m: {"milli": "ms", "micro": "us", "nano": "ns"}.get((m.group(1) or "").lower(), "s"),
    ),
    (re.compile(r"%|\bpercent(?:age)?\b", re.IGNORECASE), lambda m: "%"),
    (re.compile(r"\b([KMGT]B/s|[GM]T/s)"), lambda m: m.group(1)),
    (re.compile(r"\b([KMGT]i?B)\b(?!/)"), lambda m: m.group(1)),
    (re.compile(r"\bbytes?\b", re.IGNORECASE), lambda m: "B"),
    (re.compile(r"\b([mu]?J)\b"), lambda m: m.group(1)),
    (
        re.compile(r"\b(milli|micro)?joules?\b", re.IGNORECASE),
        lambda m: {"milli": "mJ", "micro": "uJ"}.get((m.group(1) or "").lower(), "J"),
    ),
    (re.compile(r"\bRPM\b"), lambda m: "RPM"),
]
SENTINELS = [
    (re.compile(r"\b0[xX][0-9A-Fa-f]+\b"), lambda m: hex(int(m.group(0), 16))),
    (
        re.compile(r"\b(U?)INT(8|16|32|64)_MAX\b"),
        lambda m: hex((1 << (int(m.group(2)) - (not m.group(1)))) - 1),
    ),
    (
        re.compile(r"\(\s*u?int(8|16|32|64)_t\s*\)\s*-1\b"),
        lambda m: hex((1 << int(m.group(1))) - 1),
    ),
    (re.compile(r"(?<![\w.)])-1\b"), lambda m: "-1"),
]
ENCODINGS = [
    (re.compile(r"\bbit[- ]?(?:mask|map|field)s?\b|\bbit [A-Z0-9]\b", re.IGNORECASE), "bitmask"),
    (re.compile(r"\bnul(?:l)?[- ]terminated\b", re.IGNORECASE), "cstring"),
]

# amdsmi.h is parsed with -nostdinc against these, so the snapshot does not
# depend on the host's libc and an untrusted header cannot include host files.
SHIMS = {
    "stdbool.h": "#ifndef __cplusplus\n#define bool _Bool\n#define true 1\n#define false 0\n#endif\n",
    "stddef.h": "typedef unsigned long size_t;\ntypedef long ptrdiff_t;\n#define NULL ((void*)0)\n",
    "stdint.h": (
        "typedef signed char int8_t;\ntypedef short int16_t;\ntypedef int int32_t;\n"
        "typedef long int64_t;\ntypedef unsigned char uint8_t;\ntypedef unsigned short uint16_t;\n"
        "typedef unsigned int uint32_t;\ntypedef unsigned long uint64_t;\n"
        "typedef long intptr_t;\ntypedef unsigned long uintptr_t;\n"
        "#define UINT8_MAX 0xff\n#define UINT16_MAX 0xffff\n#define UINT32_MAX 0xffffffffU\n"
        "#define UINT64_MAX 0xffffffffffffffffUL\n#define INT32_MAX 0x7fffffff\n"
    ),
    "time.h": (
        "typedef unsigned long size_t;\ntypedef long time_t;\n"
        "struct timespec { time_t tv_sec; long tv_nsec; };\n"
    ),
    "stdlib.h": "typedef unsigned long size_t;\n#define NULL ((void*)0)\n",
    "string.h": "typedef unsigned long size_t;\n#define NULL ((void*)0)\n",
    "limits.h": (
        "#define CHAR_BIT 8\n#define INT_MAX 0x7fffffff\n#define UINT_MAX 0xffffffffU\n"
        "#define LONG_MAX 0x7fffffffffffffffL\n#define ULONG_MAX 0xffffffffffffffffUL\n"
    ),
}
LLP64_SHIMS = {
    name: text.replace("unsigned long", "unsigned long long").replace(
        "typedef long ", "typedef long long "
    )
    for name, text in SHIMS.items()
}
for _c, _cxx in (
    ("stdint.h", "cstdint"),
    ("stddef.h", "cstddef"),
    ("stdbool.h", "cstdbool"),
    ("stdlib.h", "cstdlib"),
    ("string.h", "cstring"),
    ("limits.h", "climits"),
    ("time.h", "ctime"),
):
    SHIMS[_cxx] = SHIMS[_c]
    LLP64_SHIMS[_cxx] = LLP64_SHIMS[_c]
LLP64_TARGET = "x86_64-w64-windows-gnu"
TARGET = "x86_64-unknown-linux-gnu"

MISSING = object()


class GateError(Exception):
    pass


def _cindex():
    try:
        import clang.cindex as cindex
    except ImportError as exc:
        raise GateError(
            "libclang Python bindings missing: pip install -r requirements.txt"
        ) from exc
    return cindex


def _norm_doc(raw):
    if not raw:
        return ""
    text = re.sub(r"^\s*(/\*[*!]?<?|\*/|//[/!]<?|\*(?!/))", " ", raw, flags=re.MULTILINE)
    text = re.sub(r"\*/|/\*[*!]?<?|//[/!]<?", " ", text)
    return " ".join(text.split())


def _param_docs(raw):
    docs = {}
    text = _norm_doc(raw)
    for m in re.finditer(
        r"@param(\[[^\]]*\])?\s+(\w+)\s*(.*?)(?=\s@(?:param|return|retval|note|details|brief|code|warning|see|platform|ingroup|cond)\b|$)",
        text,
    ):
        direction = (m.group(1) or "").replace(" ", "").lower()
        docs[m.group(2)] = {"dir": direction, "text": m.group(3).strip()}
    return docs


class _Walker:
    def __init__(self, cindex, header_path):
        self.ci = cindex
        self.header = os.path.realpath(header_path)
        with open(header_path, encoding="utf-8") as f:
            self.lines = f.read().splitlines()
        self.labels = {}
        self.snap = {
            "types": {},
            "functions": {},
            "params": {},
            "macros": {},
            "docs": {},
            "aliases": {},
        }
        self.cxx_linkage = set()
        self.pointees = set()
        self.record_params = set()
        self.anon_enums = []
        self.inline = {}

    def in_header(self, cursor):
        f = cursor.location.file
        return f is not None and os.path.realpath(f.name) == self.header

    def doc(self, cursor):
        """The cursor's doc comment; clang drops a trailing //!< after a wrapped declaration."""
        if cursor.raw_comment:
            return _norm_doc(cursor.raw_comment)
        end = cursor.extent.end
        if end.line < 1 or end.line > len(self.lines):
            return ""
        rest = self.lines[end.line - 1][end.column - 1 :]
        m = re.search(r"(//[/!]<|/\*[*!]<)", rest)
        if not m:
            return ""
        parts = [rest[m.start() :]]
        for line in self.lines[end.line :]:
            if not re.match(r"\s*//[/!]<", line):
                break
            parts.append(line)
        return _norm_doc("\n".join(parts))

    def loc(self, cursor):
        l = cursor.location
        return f"{l.file.name}:{l.line}:{l.column}"

    def norm(self, spelling):
        def repl(m):
            return self.labels.get(m.group(1), "(anonymous)")

        return ANON.sub(repl, spelling)

    def is_anon(self, decl):
        if not decl.spelling or ANON.search(decl.spelling):
            return True
        tokens = decl.get_tokens()
        next(tokens, None)
        tok = next(tokens, None)
        while tok is not None and tok.spelling in ATTRIBUTE_TOKENS:
            depth = 0
            for tok in tokens:
                depth += {"(": 1, ")": -1}.get(tok.spelling, 0)
                if depth == 0:
                    break
            tok = next(tokens, None)
        return tok is not None and tok.spelling in ("{", ":")

    def record_kind(self, decl):
        K = self.ci.CursorKind
        return {K.STRUCT_DECL: "struct", K.UNION_DECL: "union", K.ENUM_DECL: "enum"}.get(decl.kind)

    def walk(self, parent):
        K = self.ci.CursorKind
        for c in parent.get_children():
            if not self.in_header(c):
                continue
            if c.kind in (K.LINKAGE_SPEC, K.UNEXPOSED_DECL):
                self.walk(c)
            elif c.kind == K.TYPEDEF_DECL:
                self.typedef(c)
            elif self.record_kind(c) and c.is_definition() and not self.is_anon(c):
                self.record(c, self.norm(c.type.get_canonical().spelling))
            elif c.kind == K.ENUM_DECL and c.is_definition():
                self.anon_enums.append(c)
            elif c.kind == K.FUNCTION_DECL:
                self.function(c)
        if parent.kind == K.TRANSLATION_UNIT:
            for c in self.anon_enums:
                first = next(c.get_children(), None)
                if first is not None and self.loc(c) not in self.labels:
                    self.record(c, f"enum {{{first.spelling}}}")

    def typedef(self, c):
        canon = c.underlying_typedef_type.get_canonical()
        decl = canon.get_declaration()
        if (
            self.record_kind(decl)
            and decl.is_definition()
            and self.is_anon(decl)
            and self.in_header(decl)
        ):
            label = self.labels.setdefault(self.loc(decl), c.spelling)
            if label != c.spelling:
                self.snap["aliases"][c.spelling] = label
            self.record(decl, c.spelling)
            return
        self.snap["types"][c.spelling] = {
            "kind": "typedef",
            "type": self.norm(canon.spelling),
            "size": c.type.get_size(),
        }

    def record(self, decl, key):
        if key in self.snap["types"]:
            return
        kind = self.record_kind(decl)
        entry = {"kind": kind, "size": decl.type.get_size(), "align": decl.type.get_align()}
        self.snap["types"][key] = entry
        if kind == "enum":
            entry["enumerators"] = {}
            for e in decl.get_children():
                entry["enumerators"][e.spelling] = e.enum_value
                doc = self.doc(e)
                if doc:
                    self.snap["docs"][f"enumerator:{key}.{e.spelling}"] = doc
            return
        K = self.ci.CursorKind
        children = list(decl.get_children())
        field_types = {
            self.loc(f.type.get_canonical().get_declaration())
            for f in children
            if f.kind == K.FIELD_DECL and f.type.get_canonical().get_declaration().location.file
        }
        fields = []
        nested = []
        anon_index = 0
        for c in children:
            if self.record_kind(c) and c.is_definition():
                if self.is_anon(c):
                    if self.loc(c) in field_types:
                        nested.append(c)
                        continue
                    label = f"(anon #{anon_index})"
                    anon_index += 1
                    self.labels[self.loc(c)] = label
                    member_key = self.norm(c.type.get_canonical().spelling)
                    self.record(c, member_key)
                    fields.append([label, self.anon_offset(decl, c), c.type.get_size(), member_key])
                else:
                    self.record(c, self.norm(c.type.get_canonical().spelling))
            elif c.kind == K.FIELD_DECL:
                fdecl = c.type.get_canonical().get_declaration()
                if fdecl in nested:
                    self.labels[self.loc(fdecl)] = c.spelling
                    self.record(fdecl, self.norm(fdecl.type.get_canonical().spelling))
                ftype = self.norm(c.type.get_canonical().spelling)
                bits = c.get_field_offsetof()
                if c.is_bitfield():
                    fields.append(
                        [c.spelling, f"{bits // 8}:{bits % 8}", f"{c.get_bitfield_width()}b", ftype]
                    )
                else:
                    fields.append([c.spelling, bits // 8, c.type.get_size(), ftype])
                doc = self.doc(c)
                if doc:
                    self.snap["docs"][f"field:{key}.{c.spelling}"] = doc
        entry["fields"] = fields

    def anon_offset(self, parent, anon):
        K = self.ci.CursorKind
        for f in anon.get_children():
            if f.kind == K.FIELD_DECL and f.spelling:
                outer = parent.type.get_offset(f.spelling)
                inner = anon.type.get_offset(f.spelling)
                if outer >= 0 and inner >= 0:
                    return (outer - inner) // 8
        return 0

    def function(self, c):
        static = c.storage_class == self.ci.StorageClass.STATIC
        sig = self.norm(c.type.get_canonical().spelling)
        self.snap["functions"][c.spelling] = "static " + sig if static else sig
        if not static and c.mangled_name != c.spelling:
            self.cxx_linkage.add(c.spelling)
        self.snap["params"][c.spelling] = [a.spelling for a in c.get_arguments()]
        for name, doc in _param_docs(c.raw_comment).items():
            self.snap["docs"][f"param:{c.spelling}.{name}"] = doc
        for arg in c.get_arguments():
            t = arg.type.get_canonical()
            while t.kind in (self.ci.TypeKind.POINTER, self.ci.TypeKind.CONSTANTARRAY):
                t = (
                    t.get_pointee() if t.kind == self.ci.TypeKind.POINTER else t.element_type
                ).get_canonical()
                self.pointees.add(self.norm(t.spelling))
                if t.kind == self.ci.TypeKind.RECORD:
                    self.record_params.add(f"{c.spelling}.{arg.spelling}")
        if c.is_definition():
            body = " ".join(tok.spelling for tok in c.get_tokens())
            self.inline[c.spelling] = _fnv1a(body)


def _fnv1a(text):
    h = 0xCBF29CE484222325
    for b in text.encode():
        h = ((h ^ b) * 0x100000001B3) & 0xFFFFFFFFFFFFFFFF
    return f"{h:016x}"


def _macro_candidates(cindex, tu, walker):
    """Object-like macros as text, and function-like ones as {"fn": tokens}."""
    K = cindex.CursorKind
    found, fnlike = {}, {}
    for c in tu.cursor.get_children():
        if c.kind != K.MACRO_DEFINITION or not walker.in_header(c):
            continue
        tokens = list(c.get_tokens())
        if len(tokens) < 2:
            continue
        text = " ".join(t.spelling for t in tokens[1:])
        if (
            tokens[1].spelling == "("
            and tokens[1].extent.start.column == tokens[0].extent.end.column
        ):
            fnlike[c.spelling] = {"fn": text}
            continue
        found[c.spelling] = text
    return found, fnlike


def _base_type(spelling):
    return re.sub(r"^(?:const |volatile )*|(?:\s*\[\d*\])+$", "", spelling)


def layout_hashes(types):
    """FNV-1a 64 over each record's size, alignment and non-reserved fields (name, offset,
    size, and the hash of a nested record), and over each enum's enumerators. Equal hashes
    mean an identical layout; a binding or a runtime amdsmi_get_type_layout() can compare them."""
    memo = {}

    def h(key):
        if key in memo:
            return memo[key] or ""
        t = types.get(key)
        if not t or t["kind"] not in ("struct", "union", "enum"):
            return ""
        memo[key] = None
        parts = [t["kind"], str(t["size"]), str(t["align"])]
        for name, off, size, ftype in t.get("fields", []):
            if not RESERVED.match(name):
                parts.append(f"{name}@{off}:{size}:{h(_base_type(ftype))}")
        parts += [f"{n}={v}" for n, v in t.get("enumerators", {}).items()]
        memo[key] = _fnv1a("|".join(parts))
        return memo[key]

    return {key: h(key) for key in sorted(types) if h(key)}


def _annotate(text, direction=""):
    ann = {}
    if direction:
        ann["dir"] = direction
    elif text:
        found = sorted(set(DIRECTION.findall(text)))
        if found:
            ann["dir"] = "/".join(found)
    for key, table in (("unit", UNITS), ("sentinel", SENTINELS)):
        values = sorted({fmt(m) for rx, fmt in table for m in rx.finditer(text or "")})
        if values:
            ann[key] = values
    enc = sorted({name for rx, name in ENCODINGS if rx.search(text or "")})
    if enc:
        ann["encoding"] = enc
    return ann


def caller_allocated(snap, pointees):
    """Records a caller passes by pointer, and the records they contain or point to."""
    todo = [p for p in pointees if p in snap["types"]]
    seen = set()
    while todo:
        key = todo.pop()
        if key in seen:
            continue
        seen.add(key)
        t = snap["types"][key]
        if t["kind"] == "typedef":
            todo.append(_base_type(t["type"]))
            continue
        for f in t.get("fields", []):
            base = re.sub(r"(?:\s*\*)+(?:\s*const)?$", "", _base_type(f[3]))
            if base in snap["types"]:
                todo.append(base)
    return sorted(k for k in seen if "fields" in snap["types"][k])


def semantic_annotations(snap, records, record_params=()):
    """{"field:T.f" or "param:fn.p": {dir, unit, sentinel, encoding}} for annotated subjects.
    A parameter pointing to a struct gets only its direction; its fields carry the rest."""
    out = {}
    for key in records:
        for f in snap["types"][key]["fields"]:
            if RESERVED.match(f[0]) or f[3].rstrip().endswith("*"):
                continue
            ann = _annotate(snap["docs"].get(f"field:{key}.{f[0]}", ""))
            if ann:
                out[f"field:{key}.{f[0]}"] = ann
    for key, doc in snap["docs"].items():
        if key.startswith("param:"):
            ann = _annotate(doc["text"], doc["dir"].strip("[]"))
            if key.partition(":")[2] in record_params:
                ann = {k: v for k, v in ann.items() if k == "dir"}
            if ann:
                out[key] = ann
    return out


def check_header_text(header_text):
    """Refuse what the parse cannot see: other headers, and #if on unknown macros."""
    for quote, inc in INCLUDE.findall(header_text):
        if quote != "<" or inc not in SHIMS:
            raise GateError(
                f"amdsmi.h includes {quote}{inc}; public types must live in amdsmi.h. "
                "For a standard header, add a shim to abi_gate.py SHIMS"
            )
    guard = re.search(r"^\s*#\s*ifndef\s+(\w+)\s*\n\s*#\s*define\s+\1\b", header_text, re.MULTILINE)
    allowed = CONDITIONALS | {"defined"} | ({guard.group(1)} if guard else set())
    for cond in CONDITIONAL.findall(header_text):
        cond = re.sub(r"//.*|/\*.*?\*/", "", cond)
        unknown = set(re.findall(r"[A-Za-z_]\w*", cond)) - allowed
        if unknown:
            raise GateError(
                f"amdsmi.h tests {', '.join(sorted(unknown))} in #if; the gate parses only "
                f"{', '.join(sorted(CONDITIONALS))}. Add it to abi_gate.py CONDITIONALS first"
            )


def _parse(cindex, tmp, header_text, defines, lang, target=TARGET):
    shims = LLP64_SHIMS if target == LLP64_TARGET else SHIMS
    sysdir = os.path.join(tmp, "sys-" + target)
    if not os.path.isdir(sysdir):
        os.mkdir(sysdir)
        for name, text in shims.items():
            with open(os.path.join(sysdir, name), "w") as f:
                f.write(text)
    header = os.path.join(tmp, "amdsmi.h")
    with open(header, "w") as f:
        f.write(header_text)
    std = "-std=c11" if lang == "c" else "-std=c++17"
    args = ["-x", lang, std, "-nostdinc", "-nostdinc++", "-isystem", sysdir]
    args += ["-target", target, "-Wno-everything"]
    args += [f"-D{d}" for d in defines]
    index = cindex.Index.create()
    opts = cindex.TranslationUnit.PARSE_DETAILED_PROCESSING_RECORD
    tu = index.parse(header, args=args, options=opts)
    errors = [str(d) for d in tu.diagnostics if d.severity >= cindex.Diagnostic.Error]
    if errors:
        what = "C11" if lang == "c" else "C++17"
        raise GateError(f"amdsmi.h does not parse as {what}:\n  " + "\n  ".join(errors[:10]))
    walker = _Walker(cindex, header)
    walker.walk(tu.cursor)
    return index, tu, walker, args


def snapshot_text(header_text, defines=DEFINES):
    """Layout snapshot of an amdsmi.h given as text."""
    cindex = _cindex()
    check_header_text(header_text)
    with tempfile.TemporaryDirectory(prefix="abi_gate_") as tmp:
        index, tu, walker, args = _parse(cindex, tmp, header_text, defines, "c")
        macros, fnlike = _macro_candidates(cindex, tu, walker)

        probe = os.path.join(tmp, "probe.c")
        names = sorted(n for n, v in macros.items() if '"' not in v)
        with open(probe, "w") as f:
            f.write('#include "amdsmi.h"\n')
            for i, name in enumerate(names):
                f.write(f"enum {{ abi_gate_probe_{i} = ({name}) }};\n")
        ptu = index.parse(probe, args=args)
        bad = {
            d.location.line
            for d in ptu.diagnostics
            if d.severity >= cindex.Diagnostic.Error
            and d.location.file
            and os.path.realpath(d.location.file.name) == os.path.realpath(probe)
        }
        values = {}
        for c in ptu.cursor.walk_preorder():
            if c.kind == cindex.CursorKind.ENUM_CONSTANT_DECL and c.spelling.startswith(
                "abi_gate_probe_"
            ):
                if c.location.line not in bad:
                    values[names[int(c.spelling.rsplit("_", 1)[1])]] = c.enum_value
        for name, text in macros.items():
            walker.snap["macros"][name] = values.get(name, {"text": text})
        walker.snap["macros"].update(fnlike)
    snap = walker.snap
    for key, value in layout_hashes(snap["types"]).items():
        snap["types"][key]["layout_hash"] = value
    snap["semantics"] = semantic_annotations(
        snap, caller_allocated(snap, walker.pointees), walker.record_params
    )
    snap["inline"] = walker.inline
    return snap


def _lang_neutral(spelling):
    """A type as both C and C++ spell it: no tag keyword or enclosing record, bool, ()."""
    spelling = re.sub(r"\b(?:struct|union|enum) |\b\w+::(?!\()", "", spelling)
    return spelling.replace("_Bool", "bool").replace("(void)", "()")


def _layout_view(snap, types=False):
    """Size, alignment and field placement keyed without the C tag keyword, for comparing parses;
    with types, also each field's type."""
    view = {}
    for key, t in snap["types"].items():
        if t["kind"] == "typedef":
            continue
        k = re.sub(r"\b(struct|union|enum) ", "", key)
        if "::" in k and not k.rsplit("::", 1)[1].startswith("("):
            k = k.rsplit("::", 1)[1]
        view[k] = (
            t["size"],
            t["align"],
            tuple(
                (f[0], f[1], f[2]) + ((_lang_neutral(f[3]),) if types else ())
                for f in t.get("fields", [])
            ),
            tuple(sorted(t.get("enumerators", {}).items())),
        )
    return view


def consistency_errors(header_text):
    """Layouts must not depend on the language, on ENABLE_ESMI_LIB or on the data model:
    Windows is LLP64, so a `long` that is 8 bytes on Linux is 4 there. C++ must also see the
    same field types and functions, with C linkage."""
    cindex = _cindex()
    check_header_text(header_text)
    errors = []
    with tempfile.TemporaryDirectory(prefix="abi_gate_") as tmp:
        c = _parse(cindex, tmp, header_text, DEFINES, "c")[2].snap
        for label, defines, lang, target in (
            ("C++17", DEFINES, "c++", TARGET),
            ("C11 without ENABLE_ESMI_LIB", [], "c", TARGET),
            ("C11 on LLP64 (Windows)", DEFINES, "c", LLP64_TARGET),
        ):
            try:
                walker = _parse(cindex, tmp, header_text, defines, lang, target)[2]
            except GateError as exc:
                errors.append(str(exc))
                continue
            ref = _layout_view(c, types=lang == "c++")
            other = _layout_view(walker.snap, types=lang == "c++")
            for key in sorted(set(ref) & set(other)):
                if ref[key] != other[key]:
                    errors.append(
                        f"{key} has a different layout in {label} than in C11: "
                        f"{_describe_view(ref[key])} vs {_describe_view(other[key])}"
                    )
            if lang != "c++":
                continue
            for key in sorted(set(ref) ^ set(other)):
                errors.append(f"{key} is declared only in {'C11' if key in ref else label}")
            cfuncs, xfuncs = c["functions"], walker.snap["functions"]
            for name in sorted(set(cfuncs) | set(xfuncs)):
                csig, xsig = cfuncs.get(name, "(nothing)"), xfuncs.get(name, "(nothing)")
                if _lang_neutral(csig) != _lang_neutral(xsig):
                    errors.append(f"{name} is {csig} in C11 but {xsig} in {label}")
                elif name in walker.cxx_linkage:
                    errors.append(f'{name} has C++ linkage in {label}; declare it in extern "C"')
    return errors


def _describe_view(v):
    size, align, fields, enums = v
    if enums:
        return "enumerators " + ", ".join(f"{n}={x}" for n, x in enums)
    return f"size {size} align {align} fields " + ", ".join(
        f"{f[0]}@{f[1]}" + "".join(f" {t}" for t in f[3:]) for f in fields
    )


def path_str(path):
    return "/".join(path)


class Break:
    def __init__(self, path, old, new, rows):
        self.path = tuple(path)
        self.old = old
        self.new = new
        self.rows = rows

    @property
    def entity(self):
        return self.path[1]

    def to_json(self):
        return {"path": path_str(self.path), "rows": self.rows}

    def entry(self):
        def value(v):
            return None if v is MISSING else v

        return {"path": path_str(self.path), "old": value(self.old), "new": value(self.new)}


def _fmt(v):
    if v is MISSING:
        return "(removed)"
    if isinstance(v, dict):
        return v.get("text", json.dumps(v))
    return str(v)


def _alias_map(*snaps):
    """Typedef names that denote the same anonymous record, mapped to one representative."""
    parent = {}

    def find(x):
        while parent.get(x, x) != x:
            x = parent[x]
        return x

    for snap in snaps:
        for alias, label in snap.get("aliases", {}).items():
            a, b = find(alias), find(label)
            if a != b:
                parent[max(a, b)] = min(a, b)
    if not parent:
        return None
    names = sorted(set(parent) | set(parent.values()), key=len, reverse=True)
    rx = re.compile(r"\b(" + "|".join(map(re.escape, names)) + r")\b")
    return lambda v: rx.sub(lambda m: find(m.group(1)), v) if isinstance(v, str) else v


def _same(a, b, canon):
    if a == b:
        return True
    if canon is None or a is MISSING or b is MISSING:
        return False
    if isinstance(a, list) and isinstance(b, list):
        return len(a) == len(b) and all(canon(x) == canon(y) for x, y in zip(a, b))
    return canon(a) == canon(b)


def _field_breaks(key, old, new, canon):
    breaks = []
    old_fields = {f[0]: f[1:] for f in old["fields"]}
    new_fields = {f[0]: f[1:] for f in new["fields"]}
    same_size = old["size"] == new["size"]
    for name, ofield in old_fields.items():
        nfield = new_fields.get(name, MISSING)
        if _same(ofield, nfield, canon):
            continue
        if RESERVED.match(name) and same_size and _within(ofield, nfield):
            continue
        path = ("types", key, "fields", name)
        if nfield is MISSING:
            breaks.append(
                Break(path, ofield, MISSING, [[name, "removed", _fmt(ofield), "(removed)"]])
            )
            continue
        rows = [
            [name, attr, _fmt(o), _fmt(n)]
            for attr, o, n in zip(("offset", "size", "type"), ofield, nfield)
            if not _same(o, n, canon)
        ]
        breaks.append(Break(path, ofield, nfield, rows))
    return breaks


def _bits(offset, size):
    """A field's [start, end) in bits; a bitfield is recorded as "byte:bit" and "Nb"."""
    if isinstance(offset, int):
        start = offset * 8
    else:
        byte, _, bit = offset.partition(":")
        start = int(byte) * 8 + int(bit)
    return start, start + (size * 8 if isinstance(size, int) else int(size[:-1]))


def _within(ofield, nfield):
    """A reserved field may shrink or move inside its old bits: that is a carve."""
    if nfield is MISSING:
        return True
    ostart, oend = _bits(*ofield[:2])
    nstart, nend = _bits(*nfield[:2])
    return ostart <= nstart and nend <= oend


def _sentinels(enums):
    """End-of-range enumerators such as AMDSMI_GPU_BLOCK_LAST, which move up when values are
    added: a *_LAST, *_MAX, *_COUNT or *_END at the highest value that is not a marker such as
    *_INVALID = 0xFFFFFFFF. A *_MAX next to a *_MIN is a value, as in amdsmi_freq_ind_t."""
    top = max(
        (v for n, v in enums.items() if not MARKER.search(n) and v not in MARKER_VALUES),
        default=None,
    )
    return {
        n
        for n, v in enums.items()
        if SENTINEL.search(n) and v == top and not (n.endswith("_MAX") and n[:-4] + "_MIN" in enums)
    }


def _sentinel_grew(name, oenums, nenums):
    return name in _sentinels(oenums) and name in _sentinels(nenums) and nenums[name] > oenums[name]


def _enum_breaks(key, oenums, nenums, retired):
    """Enums are append-only: a new enumerator may not take a value that a removed or
    renumbered one had."""
    breaks = []
    osent = _sentinels(oenums)
    gone = {
        v: n
        for n, v in oenums.items()
        if nenums.get(n, MISSING) != v and (n not in osent or n not in nenums)
    }
    nsent = _sentinels(nenums)
    for name, value in nenums.items():
        if name in oenums or name in nsent or (value not in gone and value not in retired):
            continue
        was = gone.get(value, "a removed enumerator")
        path = ("types", key, "enumerators", name)
        breaks.append(
            Break(path, MISSING, value, [[name, f"reuses the value of {was}", "-", value]])
        )
    return breaks


def _adds_meaning(attr, o, n):
    """A new direction into the library, or a unit, sentinel or encoding the old docs did not
    state. Dropping one only loses documentation, so it does not block."""
    if attr == "dir":
        return bool(READS.search(n or "")) and not READS.search(o or "")
    return bool(set(n or []) - set(o or []))


def _semantic_breaks(old, new):
    """Meaning added to a field or parameter present on both sides (see _adds_meaning)."""
    breaks = []
    renamed = {
        f"param:{fn}.{n}": f"param:{fn}.{o}"
        for fn, names in new["params"].items()
        for o, n in zip(old["params"].get(fn, []), names)
    }
    osem = old["semantics"]
    nsem = {renamed.get(k, k): v for k, v in new["semantics"].items()}
    for key in sorted(set(osem) | set(nsem)):
        kind, _, name = key.partition(":")
        owner, _, member = name.rpartition(".")
        if kind == "field":
            present = all(
                any(f[0] == member for f in s["types"].get(owner, {}).get("fields", []))
                for s in (old, new)
            )
        else:
            present = owner in old["functions"] and owner in new["functions"]
        o = {k: v for k, v in osem.get(key, {}).items() if k in SEMANTIC_KEYS}
        n = {k: v for k, v in nsem.get(key, {}).items() if k in SEMANTIC_KEYS}
        if not present or not any(_adds_meaning(a, o.get(a), n.get(a)) for a in SEMANTIC_KEYS):
            continue
        rows = [
            [member, attr, _fmt(o.get(attr, "-")), _fmt(n.get(attr, "-"))]
            for attr in SEMANTIC_KEYS
            if o.get(attr) != n.get(attr)
        ]
        breaks.append(Break(("semantics", key), o, n, rows))
    return breaks


def diff(old, new, policy=None):
    """Breaking changes from snapshot old to snapshot new."""
    append_only = (policy or {}).get("append_only", {})
    canon = _alias_map(old, new)
    breaks = []
    for key, otype in old["types"].items():
        ntype = new["types"].get(key, MISSING)
        if ntype is MISSING:
            breaks.append(
                Break(
                    ("types", key),
                    otype["kind"],
                    MISSING,
                    [["", "type", otype["kind"], "(removed)"]],
                )
            )
            continue
        for attr in ("kind", "type"):
            if not _same(otype.get(attr), ntype.get(attr), canon):
                path = ("types", key, attr)
                breaks.append(
                    Break(
                        path,
                        otype.get(attr),
                        ntype.get(attr),
                        [["", attr, _fmt(otype.get(attr)), _fmt(ntype.get(attr))]],
                    )
                )
        if otype["kind"] != ntype["kind"]:
            continue
        for attr in ("size", "align"):
            o, n = otype.get(attr), ntype.get(attr)
            if o == n or (attr == "size" and key in append_only and n > o):
                continue
            breaks.append(Break(("types", key, attr), o, n, [["", attr, _fmt(o), _fmt(n)]]))
        if "fields" in otype:
            breaks += _field_breaks(key, otype, ntype, canon)
        if "enumerators" in otype:
            for name, value in otype["enumerators"].items():
                nvalue = ntype["enumerators"].get(name, MISSING)
                if nvalue == value or _sentinel_grew(
                    name, otype["enumerators"], ntype["enumerators"]
                ):
                    continue
                path = ("types", key, "enumerators", name)
                breaks.append(
                    Break(path, value, nvalue, [[name, "value", _fmt(value), _fmt(nvalue)]])
                )
            retired = (policy or {}).get("retired", {}).get(key, set())
            breaks += _enum_breaks(key, otype["enumerators"], ntype["enumerators"], retired)
    for name, sig in old["functions"].items():
        nsig = new["functions"].get(name, MISSING)
        if not _same(sig, nsig, canon):
            breaks.append(
                Break(("functions", name), sig, nsig, [["", "signature", sig, _fmt(nsig)]])
            )
    enumerators = _all_enumerators(new)
    for name, value in old["macros"].items():
        if UNPINNED_MACROS.match(name):
            continue
        nvalue = new["macros"].get(name, MISSING)
        if nvalue == value or (
            isinstance(value, dict) and "text" in value and nvalue is not MISSING
        ):
            continue
        if nvalue is MISSING and enumerators.get(name, MISSING) == value:
            continue
        breaks.append(
            Break(("macros", name), value, nvalue, [["", "value", _fmt(value), _fmt(nvalue)]])
        )
    breaks += _semantic_breaks(old, new)
    return breaks


def _all_enumerators(snap):
    return {n: v for t in snap["types"].values() for n, v in t.get("enumerators", {}).items()}


def load_exceptions(text):
    """The exceptions file: acknowledged breaks, append_only structs and notes."""
    data = json.loads(text) if text else {}
    unknown = set(data) - {"append_only", "notes", "breaks"}
    if unknown:
        raise GateError(f"{EXCEPTIONS}: unknown keys {', '.join(sorted(unknown))}")
    for entry in data.get("breaks", []):
        if set(entry) != {"path", "old", "new", "why"} or not entry["why"]:
            raise GateError(f"{EXCEPTIONS}: break entries need path, old, new and why: {entry}")
    return {"append_only": {}, "notes": {}, "breaks": [], **data}


def render_exceptions(exc):
    """One break per line, so a review shows exactly what was acknowledged."""
    out = ["{"]
    for section in ("append_only", "notes"):
        items = [f"    {json.dumps(k)}: {json.dumps(v)}" for k, v in sorted(exc[section].items())]
        out.append(
            f'  "{section}": {{' + ("\n" + ",\n".join(items) + "\n  " if items else "") + "},"
        )
    items = [f"    {json.dumps(e)}" for e in exc["breaks"]]
    out.append('  "breaks": [' + ("\n" + ",\n".join(items) + "\n  " if items else "") + "]")
    out.append("}")
    return "\n".join(out) + "\n"


def _canon(value):
    return json.dumps(value, sort_keys=True)


def _retires(entry):
    """(enum, value) when entry removes or renumbers an enumerator: no new one may take the value."""
    parts = entry["path"].split("/")
    if len(parts) == 4 and parts[2] == "enumerators" and isinstance(entry["old"], int):
        return parts[1], entry["old"]
    return None


def policy_of(exc):
    """append_only, and the enumerator values acknowledged breaks retired."""
    retired = {}
    for enum, value in filter(None, map(_retires, exc["breaks"])):
        retired.setdefault(enum, set()).add(value)
    return {"append_only": exc["append_only"], "retired": retired}


def acknowledged(brk, entries):
    """True when entries for brk's path lead from its old value to its new one, so an entry
    acknowledged in one PR and one in a later PR together cover the change since a release."""
    want = brk.entry()
    edges = [(_canon(e["old"]), _canon(e["new"])) for e in entries if e["path"] == want["path"]]
    goal, todo, seen = _canon(want["new"]), [_canon(want["old"])], set()
    while todo:
        value = todo.pop()
        if value == goal:
            return True
        if value not in seen:
            seen.add(value)
            todo += [new for old, new in edges if old == value]
    return False


def doc_changes(old, new):
    """Changed doc comments on fields, enumerators and parameters present on both sides."""
    notes = []
    odocs, ndocs = old.get("docs", {}), new.get("docs", {})
    for key in sorted(set(odocs) | set(ndocs)):
        kind, _, name = key.partition(":")
        owner, _, member = name.rpartition(".")
        if kind in ("field", "enumerator"):
            fields = new["types"].get(owner, {})
            exists_old = owner in old["types"] and (
                member in old["types"][owner].get("enumerators", {})
                or any(f[0] == member for f in old["types"][owner].get("fields", []))
            )
            exists_new = member in fields.get("enumerators", {}) or any(
                f[0] == member for f in fields.get("fields", [])
            )
        else:
            exists_old = owner in old["functions"]
            exists_new = owner in new["functions"]
        if not (exists_old and exists_new):
            continue
        o, n = odocs.get(key), ndocs.get(key)
        if kind == "param":
            o = o or {"dir": "", "text": ""}
            n = n or {"dir": "", "text": ""}
            if o == n:
                continue
            if o["dir"] != n["dir"] and o["dir"] and n["dir"]:
                notes.append([name, "direction", o["dir"], n["dir"]])
            if o["text"] != n["text"]:
                notes.append([name, "doc", o["text"], n["text"]])
            continue
        if (o or "") == (n or ""):
            continue
        od, nd = set(DIRECTION.findall(o or "")), set(DIRECTION.findall(n or ""))
        if od != nd:
            notes.append(
                [name, "direction", "/".join(sorted(od)) or "-", "/".join(sorted(nd)) or "-"]
            )
        notes.append([name, "doc", o or "", n or ""])
    for key, otype in old["types"].items():
        nenums = new["types"].get(key, {}).get("enumerators", {})
        oenums = otype.get("enumerators", {})
        sentinels = {n: oenums[n] for n in _sentinels(oenums)}
        nsentinels = _sentinels(nenums)
        real = {
            n: v
            for n, v in oenums.items()
            if n not in sentinels and not MARKER.search(n) and v not in MARKER_VALUES
        }
        top_sentinel = max(sentinels.items(), key=lambda kv: kv[1], default=None)
        for name, value in oenums.items():
            if _sentinel_grew(name, oenums, nenums):
                notes.append([f"{key}.{name}", "sentinel", value, nenums[name]])
        values = {v: n for n, v in oenums.items() if n in nenums}
        for name, value in nenums.items():
            if name in oenums or name in nsentinels:
                continue
            if value in values:
                notes.append([f"{key}.{name}", "alias", values[value], value])
            elif real and value < max(real.values()) and value not in oenums.values():
                notes.append([f"{key}.{name}", "not appended", f"max {max(real.values())}", value])
            if top_sentinel and nenums.get(top_sentinel[0]) == top_sentinel[1] <= value:
                notes.append(
                    [f"{key}.{name}", "beyond sentinel", "=".join(map(str, top_sentinel)), value]
                )
    enumerators = _all_enumerators(new)
    for name, value in old["macros"].items():
        if name not in new["macros"] and enumerators.get(name, MISSING) == value:
            notes.append(
                [
                    name,
                    "macro became enumerator",
                    "#define",
                    "enumerator (#ifdef no longer sees it)",
                ]
            )
    for name, body in old.get("inline", {}).items():
        nbody = new.get("inline", {}).get(name)
        if nbody and nbody != body:
            notes.append([name, "inline body", body, nbody])
    return notes


def _changelog_added(old_text, new_text):
    return "\n".join(
        line[1:]
        for line in difflib.unified_diff(
            (old_text or "").splitlines(), (new_text or "").splitlines(), lineterm="", n=0
        )
        if line.startswith("+") and not line.startswith("+++")
    )


def _changelog_names(brk):
    if brk.path[0] == "policy":
        return {brk.path[1]}
    if brk.path[0] == "semantics":
        owner, _, member = brk.entity.partition(":")[2].rpartition(".")
        return {owner, member}
    names = {re.sub(r"^(struct|union|enum) ", "", brk.entity.split("::")[0])}
    if len(brk.path) > 3:
        names.add(brk.path[3])
    return names


def _git_show(repo, ref, path):
    if ref == WORKTREE:
        full = os.path.join(repo, path)
        if not os.path.exists(full):
            return None
        with open(full, encoding="utf-8") as f:
            return f.read()
    try:
        return subprocess.run(
            ["git", "-C", repo, "show", f"{ref}:{path}"], check=True, capture_output=True, text=True
        ).stdout
    except subprocess.CalledProcessError:
        return None


def latest_release(repo, remote="origin"):
    """Highest release/therock-* branch on the remote."""
    out = subprocess.run(
        ["git", "-C", repo, "ls-remote", "--heads", remote, "refs/heads/release/therock-*"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    branches = [
        line.split("refs/heads/", 1)[1] for line in out.splitlines() if "refs/heads/" in line
    ]

    def version(name):
        return [
            int(p) if p.isdigit() else -1 for p in re.split(r"[.-]", name.rsplit("therock-", 1)[1])
        ]

    branches = [b for b in branches if re.fullmatch(r"release/therock-\d+(\.\d+)*", b)]
    return max(branches, key=version) if branches else None


def _snapshot_ref(repo, ref, cache={}):
    if ref != WORKTREE:
        res = subprocess.run(
            ["git", "-C", repo, "rev-parse", "--verify", "-q", f"{ref}^{{commit}}"],
            capture_output=True,
            text=True,
        )
        ref = res.stdout.strip() or ref
    key = (repo, ref)
    if key not in cache or ref == WORKTREE:
        text = _git_show(repo, ref, HEADER)
        cache[key] = None if text is None else snapshot_text(text)
    return cache[key]


def _c_int(value):
    return f"{value}ULL" if value >= 0 else f"{value}LL"


def render_asserts(snap, append_only=(), skip=()):
    """C/C++ static assertions pinning the layouts in snap, except the break paths in skip.
    A changed kind skips the whole type."""
    skip = {re.sub(r"/kind$", "", p) for p in skip}

    def pinned(*path):
        return not any(path_str(path[:i]) in skip for i in range(2, len(path) + 1))

    out = [
        "#ifdef __cplusplus",
        "#include <cstddef>",
        "#define AMDSMI_ABI_ASSERT(cond, msg) static_assert(cond, msg)",
        "#define AMDSMI_ABI_ALIGNOF(t) alignof(t)",
        "#else",
        "#include <stddef.h>",
        "#define AMDSMI_ABI_ASSERT(cond, msg) _Static_assert(cond, msg)",
        "#define AMDSMI_ABI_ALIGNOF(t) _Alignof(t)",
        "#endif",
    ]
    for key in sorted(snap["types"]):
        t = snap["types"][key]
        if "::" in key or t["kind"] == "typedef" or t["size"] < 0 or not pinned("types", key):
            continue
        if t["kind"] == "enum":
            sentinels = _sentinels(t["enumerators"])
            for name, value in t["enumerators"].items():
                op = ">=" if name in sentinels else "=="
                if pinned("types", key, "enumerators", name):
                    out.append(f'AMDSMI_ABI_ASSERT({name} {op} {_c_int(value)}, "{name} value");')
            continue
        op = ">=" if key in append_only else "=="
        tagged = " " in key
        if tagged:
            out.append("#ifndef __cplusplus  // C++ scopes a nested tag inside its parent")
        if pinned("types", key, "size"):
            out.append(f'AMDSMI_ABI_ASSERT(sizeof({key}) {op} {t["size"]}, "sizeof({key})");')
        if pinned("types", key, "align"):
            out.append(
                f'AMDSMI_ABI_ASSERT(AMDSMI_ABI_ALIGNOF({key}) == {t["align"]}, "alignof({key})");'
            )
        for name, off, _size, _type in t.get("fields", []):
            if name.startswith("(") or RESERVED.match(name) or not isinstance(off, int):
                continue
            if pinned("types", key, "fields", name):
                out.append(
                    f'AMDSMI_ABI_ASSERT(offsetof({key}, {name}) == {off}, "offsetof({key}, {name})");'
                )
        if tagged:
            out.append("#endif")
    for name, value in sorted(snap["macros"].items()):
        if isinstance(value, int) and not UNPINNED_MACROS.match(name) and pinned("macros", name):
            out.append(f'AMDSMI_ABI_ASSERT(({name}) == {_c_int(value)}, "{name}");')
    return "\n".join(out) + "\n"


def compile_asserts(header_text, asserts_text):
    """Compile static asserts against a header with gcc and g++: a check of the layouts that
    does not depend on libclang."""
    failures = []
    with tempfile.TemporaryDirectory(prefix="abi_gate_cc_") as tmp:
        src = os.path.join(tmp, "asserts.c")
        with open(src, "w") as f:
            f.write(header_text + "\n" + asserts_text)
        for cc, lang in (("gcc", ["-x", "c", "-std=c11"]), ("g++", ["-x", "c++", "-std=c++17"])):
            if shutil.which(cc) is None:
                failures.append(f"{cc} not found")
                continue
            cmd = [cc, *lang, "-fsyntax-only", "-w", *[f"-D{d}" for d in DEFINES], src]
            res = subprocess.run(cmd, capture_output=True, text=True)
            if res.returncode:
                failures.append(f"{cc}:\n{res.stderr.strip()}")
    return failures


def _list_dir(repo, ref, path):
    if ref == WORKTREE:
        d = os.path.join(repo, path)
        return sorted(os.listdir(d)) if os.path.isdir(d) else []
    res = subprocess.run(
        ["git", "-C", repo, "ls-tree", "--name-only", f"{ref}:{path}"],
        capture_output=True,
        text=True,
    )
    return sorted(res.stdout.split()) if res.returncode == 0 else []


def merge_base(repo, base, new):
    """Where new forked from base, so what base gained since is not blamed on new. A shallow CI
    clone may not have it, but there the base is already the fork point."""
    res = subprocess.run(
        ["git", "-C", repo, "merge-base", base, "HEAD" if new == WORKTREE else new],
        capture_output=True,
        text=True,
    )
    return res.stdout.strip() or base


def gate(repo, old_ref, new_ref, release_ref=None, compile_check=True):
    """Run every check; returns the result dict (result['blocking'] is the verdict)."""
    old_ref = merge_base(repo, old_ref, new_ref)
    new = _snapshot_ref(repo, new_ref)
    if new is None:
        raise GateError(f"{HEADER} missing at {new_ref}")
    exc = load_exceptions(_git_show(repo, new_ref, EXCEPTIONS))
    old_exc_text = _git_show(repo, old_ref, EXCEPTIONS)
    old_exc = load_exceptions(old_exc_text)
    # A PR must not exempt the types it breaks: comparisons use the base's append_only and
    # retired values, and every entry the PR adds needs a CHANGELOG line.
    old_policy = policy_of(old_exc)
    old_keys = {_canon(e) for e in old_exc["breaks"]}
    added = [e for e in exc["breaks"] if _canon(e) not in old_keys]
    result = {
        "refs": {"base": old_ref, "release": release_ref, "new": new_ref},
        "breaks": [],
        "errors": [],
        "advisory": [],
    }
    result["errors"] += consistency_errors(_git_show(repo, new_ref, HEADER))
    kept = {_retires(e) for e in exc["breaks"]}
    for e in old_exc["breaks"]:
        if _retires(e) and _retires(e) not in kept:
            result["errors"].append(
                f"{EXCEPTIONS} drops {e['path']}, which keeps {e['old']} from being reused"
            )
    include_dir = os.path.dirname(HEADER)
    for name in _list_dir(repo, new_ref, include_dir):
        if name in SHIMS:
            result["errors"].append(
                f"{include_dir}/{name} shadows the standard header the gate parses amdsmi.h with"
            )

    refs = [("base", old_ref)] + ([("release", release_ref)] if release_ref else [])
    base_acked = set()
    for label, ref in refs:
        old = _snapshot_ref(repo, ref)
        if old is None:
            result["errors"].append(f"{HEADER} missing at {label} ref {ref}")
            continue
        for brk in diff(old, new, old_policy):
            if label == "base":
                acked = acknowledged(brk, added)
                if acked:
                    base_acked.add(path_str(brk.path))
            else:
                acked = acknowledged(brk, exc["breaks"])
            result["breaks"].append({"ref": label, "acknowledged": acked, **brk.to_json()})
        if label == "base":
            result["advisory"] += [["base", *n] for n in doc_changes(old, new)]
            if compile_check:
                asserts = render_asserts(old, old_policy["append_only"], base_acked)
                result["errors"] += compile_asserts(_git_show(repo, new_ref, HEADER), asserts)

    newly_acked = [Break(e["path"].split("/"), e["old"], e["new"], []) for e in added]
    for section, kind in (("append_only", "policy"), ("notes", "semantics")):
        for key in sorted(set(exc[section]) | set(old_exc[section])):
            if exc[section].get(key) != old_exc[section].get(key):
                newly_acked.append(Break((kind, key), None, exc[section].get(key), []))

    if newly_acked and old_exc_text is not None:
        added_lines = _changelog_added(
            _git_show(repo, old_ref, CHANGELOG), _git_show(repo, new_ref, CHANGELOG)
        )
        if not BREAK_WORD.search(added_lines):
            result["errors"].append(
                f"{EXCEPTIONS} acknowledges an ABI break but no added {CHANGELOG} line says "
                "'ABI' or 'breaking'"
            )
        unnamed = {
            brk.entity
            for brk in newly_acked
            if not any(
                re.search(rf"\b{re.escape(n)}\b", added_lines) for n in _changelog_names(brk)
            )
        }
        for entity in sorted(unnamed):
            result["errors"].append(
                f"ABI break in {entity} is acknowledged in {EXCEPTIONS} but no added "
                f"{CHANGELOG} line names it"
            )
        if release_ref:
            rel = _snapshot_ref(repo, release_ref)
            if rel and rel["macros"].get("AMDSMI_LIB_VERSION_MAJOR") == new["macros"].get(
                "AMDSMI_LIB_VERSION_MAJOR"
            ):
                result["advisory"].append(
                    [
                        "release",
                        "AMDSMI_LIB_VERSION_MAJOR",
                        "note",
                        "unchanged since the release",
                        "this PR acknowledges an ABI break; decide whether the major version and SONAME move",
                    ]
                )

    result["acknowledged_new"] = sorted({path_str(b.path) for b in newly_acked})
    result["blocking"] = bool(result["errors"]) or any(
        not b["acknowledged"] for b in result["breaks"]
    )
    return result


def _cell(text, limit=120):
    text = str(text).replace("|", "\\|").replace("\n", " ")
    return text if len(text) <= limit else text[: limit - 3] + "..."


def render_summary(result):
    out = ["## AMDSMI ABI gate", ""]
    refs = result["refs"]
    out.append(
        f"Compared `{refs['new']}` against base `{refs['base']}` and release `{refs['release'] or '-'}`."
    )
    out.append("")
    if result["blocking"]:
        out.append(
            "**FAIL**: unacknowledged ABI breaks or gate errors. For an intended break, run "
            "`abi_gate.py acknowledge` and add a CHANGELOG line; see README in "
            "`projects/amdsmi/tests/abi_check/`."
        )
    else:
        out.append("**PASS**")
    out.append("")
    if result["breaks"]:
        out += [
            "| vs | entity | member | change | old | new | status |",
            "|---|---|---|---|---|---|---|",
        ]
        for b in result["breaks"]:
            status = "acknowledged" if b["acknowledged"] else "**BLOCKING**"
            entity = b["path"].split("/")[1]
            for member, attr, old, new in b["rows"]:
                out.append(
                    f"| {b['ref']} | `{_cell(entity)}` | {_cell(member) or '-'} | {attr} | {_cell(old)} | {_cell(new)} | {status} |"
                )
        out.append("")
    for err in result["errors"]:
        out.append(f"- **error**: {err}")
    if result["errors"]:
        out.append("")
    if result["advisory"]:
        out += ["### Semantic review (advisory, not blocking)", ""]
        out += ["| vs | entity | change | old | new |", "|---|---|---|---|---|"]
        for ref, entity, attr, old, new in result["advisory"]:
            out.append(
                f"| {ref} | `{_cell(entity)}` | {attr} | {_cell(old, 80)} | {_cell(new, 80)} |"
            )
        out.append("")
    return "\n".join(out)


def acknowledge(repo, base_ref, release_ref, why):
    """Append an entry to the exceptions file for each break of the working tree against base
    and release that the gate would not accept; returns the new entries."""
    base_ref = merge_base(repo, base_ref, WORKTREE)
    new = _snapshot_ref(repo, WORKTREE)
    exc = load_exceptions(_git_show(repo, WORKTREE, EXCEPTIONS))
    base_exc = load_exceptions(_git_show(repo, base_ref, EXCEPTIONS))
    base_keys = {_canon(e) for e in base_exc["breaks"]}
    policy = policy_of(base_exc)
    added = []
    for ref in [base_ref] + ([release_ref] if release_ref else []):
        old = _snapshot_ref(repo, ref)
        if old is None:
            raise GateError(f"{HEADER} missing at {ref}")
        for brk in diff(old, new, policy):
            pool = exc["breaks"]
            if ref == base_ref:
                pool = [e for e in pool if _canon(e) not in base_keys]
            if not acknowledged(brk, pool):
                if _canon({**brk.entry(), "why": why}) in base_keys:
                    raise GateError(
                        f"{path_str(brk.path)}: the base already has this entry; "
                        "give a --why that names this PR"
                    )
                added.append({**brk.entry(), "why": why})
                exc["breaks"].append(added[-1])
    os.makedirs(os.path.dirname(os.path.join(repo, EXCEPTIONS)), exist_ok=True)
    with open(os.path.join(repo, EXCEPTIONS), "w") as f:
        f.write(render_exceptions(exc))
    return added


def _release(repo, release):
    if release != "auto":
        return release
    latest = latest_release(repo)
    return f"origin/{latest}" if latest else None


def main(argv=None):
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--repo", default=".", help="rocm-systems checkout (default: cwd)")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("gate", help="Fail on unacknowledged ABI breaks")
    p.add_argument("--base", required=True, help="Base commit of the PR")
    p.add_argument("--new", default="HEAD", help=f"Commit to check, or {WORKTREE} (default: HEAD)")
    p.add_argument(
        "--release",
        help="Release ref to compare against too; 'auto' picks the highest release/therock-*",
    )
    p.add_argument(
        "--summary-out", help="Append the markdown summary here (e.g. $GITHUB_STEP_SUMMARY)"
    )
    p.add_argument("--json-out", help="Write the full result as JSON here")
    p.add_argument(
        "--no-compile", action="store_true", help="Skip compiling the asserts with gcc/g++"
    )

    p = sub.add_parser(
        "diff", help=f"Print breaking changes between two refs, ignoring {EXCEPTIONS}"
    )
    p.add_argument("old")
    p.add_argument("new")

    p = sub.add_parser("snapshot", help="Print the layout snapshot of a header file as JSON")
    p.add_argument("header")

    p = sub.add_parser(
        "acknowledge", help=f"Add the working tree's unacknowledged breaks to {EXCEPTIONS}"
    )
    p.add_argument("--base", required=True, help="Base commit of the PR")
    p.add_argument("--release", help="Release ref, or 'auto' as for gate")
    p.add_argument("--why", required=True, help="Reason and PR reference for every new entry")

    p = sub.add_parser(
        "latest-release", help="Print the highest release/therock-* branch on origin"
    )

    args = parser.parse_args(argv)
    repo = os.path.abspath(args.repo)
    try:
        if args.cmd == "snapshot":
            with open(args.header, encoding="utf-8") as f:
                print(json.dumps(snapshot_text(f.read()), indent=1, sort_keys=True))
        elif args.cmd == "latest-release":
            print(latest_release(repo) or "")
        elif args.cmd == "acknowledge":
            added = acknowledge(repo, args.base, _release(repo, args.release), args.why)
            for entry in added:
                print(json.dumps(entry))
            print(f"Added {len(added)} entries to {EXCEPTIONS}")
        elif args.cmd == "diff":
            old, new = _snapshot_ref(repo, args.old), _snapshot_ref(repo, args.new)
            breaks = diff(old, new)
            notes = doc_changes(old, new)
            print(
                render_summary(
                    {
                        "refs": {"base": args.old, "release": None, "new": args.new},
                        "breaks": [
                            {"ref": "old", "acknowledged": False, **b.to_json()} for b in breaks
                        ],
                        "errors": [],
                        "advisory": [["old", *n] for n in notes],
                        "blocking": bool(breaks),
                    }
                )
            )
            return 1 if breaks else 0
        else:
            release = _release(repo, args.release)
            result = gate(repo, args.base, args.new, release, compile_check=not args.no_compile)
            summary = render_summary(result)
            print(summary)
            if args.summary_out:
                with open(args.summary_out, "a", encoding="utf-8") as f:
                    f.write(summary + "\n")
            if args.json_out:
                with open(args.json_out, "w", encoding="utf-8") as f:
                    json.dump(result, f, indent=1)
            return 1 if result["blocking"] else 0
    except GateError as exc:
        print(f"abi_gate: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
