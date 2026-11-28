#!/usr/bin/env python3
"""Generate the remapper's gfx1250 instruction table from AMD's machine-readable ISA.

Usage: gen_isa_table.py AMDGPU_ISA_CDNA5_XML OUT_JSON

The XML is `amdgpu_isa_cdna5.xml` from the GPUOpen machine-readable ISA archive
(shared/machine-readable-isa/download.py fetches it). The table keeps what the remapper needs to decode an
instruction LLVM's disassembler rejects: each encoding's identification mask and identifiers, its bit fields and
named conditions, and for each (encoding, opcode) the instruction's variants (encoding, condition) with their
operand fields. Names and operand lists are interned.
"""

import hashlib
import json
import sys
import xml.etree.ElementTree as ET


def bits(text):
    return int(text, 2)


def expr(node):
    """An XML condition expression as nested lists: [op, args...], ["field", ENC, FIELD] or ["lit", value]."""
    tag = node.tag
    if tag == "lit":
        return ["lit", int(node.get("val"), 0)]
    if tag == "id":
        return ["id", node.get("val")]
    if tag != "op":
        return ["unknown", tag]
    kind = node.get("type")
    args = [c for c in node if c.tag != "ty"]
    if kind == ".fieldderef":
        inner = expr(args[0])
        name = args[1].get("val")
        if inner[0] == "id":
            return ["path", inner[1], name]
        if inner[0] == "path":
            return ["field", inner[2], name]
        return ["unknown", kind]
    if kind == ".cons_array":
        return ["array"] + [expr(a) for a in args]
    return [kind] + [expr(a) for a in args]


def main():
    src, out = sys.argv[1], sys.argv[2]
    data = open(src, "rb").read()
    root = ET.fromstring(data)
    encodings = []
    for e in root.iter("Encoding"):
        fields = {}
        for f in e.iter("Field"):
            fields[f.findtext("FieldName")] = [
                [int(r.findtext("BitOffset")), int(r.findtext("BitCount"))]
                for r in f.iter("Range")
            ]
        conds = {}
        ec = e.find("EncodingConditions")
        for c in [] if ec is None else ec.iter("EncodingCondition"):
            x = c.find("CondtionExpression")
            body = [n for n in x] if x is not None else []
            conds[c.findtext("ConditionName")] = expr(body[0]) if body else ["lit", 1]
        encodings.append(
            {
                "name": e.findtext("EncodingName"),
                "bits": int(e.findtext("BitCount")),
                "mask": format(bits(e.findtext("EncodingIdentifierMask")), "x"),
                "ids": [
                    format(bits(i.text), "x") for i in e.iter("EncodingIdentifier")
                ],
                "fields": fields,
                "conds": conds,
            }
        )
    # Variants are [name, encoding, condition, operand list], each an index into its interned list.
    names, opsets, interned = [], [], {}
    enc_index = {e["name"]: k for k, e in enumerate(encodings)}
    cond_index = {}
    instructions = {}
    for i in root.iter("Instruction"):
        name = i.findtext("InstructionName")
        for ie in i.iter("InstructionEncoding"):
            enc = ie.findtext("EncodingName")
            ops = []
            for o in sorted(ie.iter("Operand"), key=lambda o: int(o.get("Order", "0"))):
                ops.append(
                    [
                        o.findtext("FieldName") or "",
                        o.findtext("OperandType") or "",
                        int(o.findtext("OperandSize") or 0),
                        o.get("Output") == "true",
                        o.get("IsImplicit") == "true",
                    ]
                )
            key = json.dumps(ops)
            if key not in interned:
                interned[key] = len(opsets)
                opsets.append(ops)
            if not names or names[-1] != name:
                names.append(name)
            cond = ie.findtext("EncodingCondition")
            if cond not in cond_index:
                cond_index[cond] = len(cond_index)
            variant = [len(names) - 1, enc_index[enc], cond_index[cond], interned[key]]
            instructions.setdefault(f"{enc}:{ie.findtext('Opcode')}", []).append(
                variant
            )
    arch = root.find(".//Architecture")
    table = {
        "source": {
            "file": src.rsplit("/", 1)[-1],
            "sha256": hashlib.sha256(data).hexdigest(),
            "architecture": (
                arch.findtext("ArchitectureName") if arch is not None else ""
            ),
        },
        "encodings": encodings,
        "names": names,
        "conditions": sorted(cond_index, key=cond_index.get),
        "operand_lists": opsets,
        "instructions": instructions,
    }
    with open(out, "w") as f:
        json.dump(table, f, separators=(",", ":"), sort_keys=True)
        f.write("\n")


if __name__ == "__main__":
    main()
