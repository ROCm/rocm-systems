#!/usr/bin/env python3

# Copyright (c) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Audit the gfx1251 delta using only its recorded public-source evidence."""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any


class ProvenanceError(ValueError):
    """The delta cannot be reproduced from its public provenance manifest."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ProvenanceError(message)


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ProvenanceError(
            f"{path}: cannot read provenance manifest: {error}"
        ) from error
    _require(isinstance(value, dict), f"{path}: manifest root must be an object")
    return value


def _parse_xml(path: Path) -> ET.Element:
    try:
        return ET.parse(path).getroot()
    except (OSError, ET.ParseError) as error:
        raise ProvenanceError(f"{path}: cannot read XML: {error}") from error


def _validate_xml_text_layout(
    element: ET.Element, path: Path, context: str | None = None
) -> None:
    element_context = element.tag if context is None else f"{context}/{element.tag}"
    if len(element) and (element.text or "").strip():
        raise ProvenanceError(
            f"{path}: mixed container text is not permitted at {element_context}"
        )
    for child in element:
        _validate_xml_text_layout(child, path, element_context)
    if (element.tail or "").strip():
        raise ProvenanceError(
            f"{path}: non-whitespace tail is not permitted after {element_context}"
        )


def _parse_audited_delta_xml(path: Path) -> ET.Element:
    try:
        parser = ET.iterparse(path, events=("start", "comment", "pi"))
        root = None
        for event, element in parser:
            if event == "comment":
                raise ProvenanceError(f"{path}: XML comment is not permitted")
            if event == "pi":
                raise ProvenanceError(
                    f"{path}: XML processing instruction is not permitted"
                )
            if root is None:
                root = element
    except (OSError, ET.ParseError) as error:
        raise ProvenanceError(f"{path}: cannot read XML: {error}") from error
    _require(root is not None, f"{path}: XML document has no root element")
    _validate_xml_text_layout(root, path)
    return root


def _find_named_instruction(root: ET.Element, name: str, context: str) -> ET.Element:
    matches = [
        instruction
        for instruction in root.findall(".//Instruction")
        if instruction.findtext("InstructionName") == name
    ]
    _require(len(matches) == 1, f"{context}: expected exactly one instruction {name!r}")
    return matches[0]


def _find_encoding(root: ET.Element, name: str) -> ET.Element:
    matches = [
        encoding
        for encoding in root.findall(".//Encoding")
        if encoding.findtext("EncodingName") == name
    ]
    _require(len(matches) == 1, f"base XML: expected exactly one encoding {name!r}")
    return matches[0]


def _canonical(element: ET.Element) -> tuple[Any, ...]:
    return (
        element.tag,
        tuple(sorted(element.attrib.items())),
        (element.text or "").strip(),
        tuple(_canonical(child) for child in element),
    )


def _encoding_identifier_from_mc(
    record: dict[str, Any], mask: str, bit_count: int
) -> str:
    byte_values = bytes(int(value, 0) for value in record["bytes"])
    _require(
        len(byte_values) * 8 == bit_count,
        f"{record['path']}:{record['assembly_line']}: expected {bit_count // 8} encoding bytes",
    )
    encoded = int.from_bytes(byte_values, byteorder="little")
    return f"{encoded & int(mask, 2):0{bit_count}b}"


def _validate_operand_profile(
    instruction: ET.Element,
    template: ET.Element,
    record: dict[str, Any],
) -> None:
    name = record["instruction"]
    actual_encodings = instruction.findall("./InstructionEncodings/InstructionEncoding")
    template_encodings = template.findall("./InstructionEncodings/InstructionEncoding")
    _require(
        len(actual_encodings) == len(template_encodings),
        f'{name}: encoding count does not match public base template {record["base_template"]}',
    )

    profile = record["operands"]
    for actual, base in zip(actual_encodings, template_encodings, strict=True):
        encoding_name = actual.findtext("EncodingName")
        condition = actual.findtext("EncodingCondition")
        context = f"{name}/{encoding_name}/{condition}"
        _require(
            actual.attrib == base.attrib,
            f"{context}: encoding attributes differ from template",
        )
        _require(
            [child.tag for child in actual]
            == ["EncodingName", "EncodingCondition", "Opcode", "Operands"],
            f"{context}: unexpected encoding fields",
        )
        _require(
            encoding_name == base.findtext("EncodingName")
            and condition == base.findtext("EncodingCondition"),
            f"{context}: encoding name/condition differs from public base template",
        )
        for child_name in ("EncodingName", "Operands"):
            _require(
                actual.find(child_name).attrib == base.find(child_name).attrib,
                f"{context}: {child_name} attributes differ from public base template",
            )
        _require(
            actual.find("EncodingCondition").attrib
            == base.find("EncodingCondition").attrib,
            f"{context}: encoding-condition attributes differ from public base template",
        )
        opcode = actual.find("Opcode")
        _require(
            opcode is not None
            and opcode.attrib == {"Radix": "10"}
            and opcode.text == str(record["opcode"]),
            f'{context}: opcode must be decimal {record["opcode"]}',
        )

        literal_sources = {
            int(index) for index in re.findall(r"has_lit_(\d+)", condition or "")
        }
        operands = actual.findall("./Operands/Operand")
        base_operands = base.findall("./Operands/Operand")
        _require(
            len(operands) == len(profile) == len(base_operands),
            f"{context}: operand count differs from profile/template",
        )
        for index, (operand, base_operand, expected) in enumerate(
            zip(operands, base_operands, profile, strict=True)
        ):
            expected_field = expected["field"]
            if index > 0 and index - 1 in literal_sources:
                expected_field = "LITERAL"
            expected_attributes = {
                "Input": str(expected["input"]).lower(),
                "Output": str(expected["output"]).lower(),
                "IsImplicit": "false",
                "IsBinaryMicrocodeRequired": "true",
                "Order": str(index + 1),
            }
            _require(
                operand.attrib == expected_attributes == base_operand.attrib,
                f"{context}: operand {index + 1} attributes differ from profile/template",
            )
            _require(
                [child.tag for child in operand]
                == ["FieldName", "DataFormatName", "OperandType", "OperandSize"],
                f"{context}: operand {index + 1} has unexpected fields",
            )
            _require(
                all(not child.attrib for child in operand),
                f"{context}: operand {index + 1} field attributes are not provenance-approved",
            )
            actual_values = (
                operand.findtext("FieldName"),
                operand.findtext("DataFormatName"),
                operand.findtext("OperandType"),
                operand.findtext("OperandSize"),
            )
            expected_values = (
                expected_field,
                expected["format"],
                expected["type"],
                str(expected["size"]),
            )
            _require(
                actual_values == expected_values,
                f"{context}: operand {index + 1} is {actual_values}, expected {expected_values}",
            )


def _validate_delta(manifest: dict[str, Any], isa_dir: Path) -> None:
    _require(
        manifest.get("schema_version") == 1, "unsupported provenance schema version"
    )
    records = manifest.get("instructions")
    _require(
        isinstance(records, list) and records,
        "manifest instructions must be a nonempty list",
    )

    base = _parse_xml(isa_dir / manifest["base_xml"])
    delta = _parse_audited_delta_xml(isa_dir / manifest["delta_xml"])
    _require(delta.tag == "IsaAdditions", "delta root must be IsaAdditions")
    _require(
        delta.attrib
        == {
            "Id": "gfx1251-instructions",
            "BaseArchitecture": "AMD CDNA 5",
            "BaseSchemaVersion": "1.2.0",
        },
        "delta root attributes are not provenance-approved",
    )
    _require(
        [child.tag for child in delta]
        == ["EncodingIdentifierAdditions", "InstructionAdditions"],
        "delta root children are not provenance-approved",
    )
    identifier_additions = delta.find("EncodingIdentifierAdditions")
    instructions = delta.find("InstructionAdditions")
    _require(
        identifier_additions is not None and not identifier_additions.attrib,
        "EncodingIdentifierAdditions attributes are not provenance-approved",
    )
    _require(
        instructions is not None and not instructions.attrib,
        "InstructionAdditions attributes are not provenance-approved",
    )
    delta_instructions = delta.findall("./InstructionAdditions/Instruction")
    expected_names = [record["instruction"] for record in records]
    actual_names = [
        instruction.findtext("InstructionName") for instruction in delta_instructions
    ]
    _require(
        actual_names == expected_names,
        "delta instruction order/set differs from provenance manifest",
    )

    expected_identifiers: dict[tuple[str, int], str] = {}
    for record in records:
        name = record["instruction"]
        instruction = _find_named_instruction(delta, name, "delta XML")
        template = _find_named_instruction(
            base, record["base_template"], "public base XML"
        )
        _require(
            not instruction.attrib,
            f"{name}: instruction attributes are not provenance-approved",
        )
        _require(
            [child.tag for child in instruction]
            == [
                "InstructionFlags",
                "InstructionName",
                "Description",
                "InstructionEncodings",
                "FunctionalGroup",
            ],
            f"{name}: unexpected instruction metadata (aliases are not provenance-approved)",
        )
        for child in instruction:
            _require(
                not child.attrib,
                f"{name}: {child.tag} attributes are not provenance-approved",
            )
        expected_description = (
            f"Encoding model for {name.lower()}, derived from LLVM gfx1251 "
            "instruction definitions."
        )
        _require(
            instruction.findtext("Description") == expected_description,
            f"{name}: description is not the repository-authored provenance template",
        )
        _require(
            _canonical(instruction.find("InstructionFlags"))
            == _canonical(template.find("InstructionFlags")),
            f'{name}: flags differ from public base template {record["base_template"]}',
        )
        _require(
            _canonical(instruction.find("FunctionalGroup"))
            == _canonical(template.find("FunctionalGroup")),
            f'{name}: functional group differs from public base template {record["base_template"]}',
        )
        _validate_operand_profile(instruction, template, record)

        for encoding_name, mc_key in (
            ("ENC_VOP3P", "primary_mc"),
            ("VOP3P_INST_LITERAL", "literal_mc"),
        ):
            if mc_key not in record:
                continue
            encoding = _find_encoding(base, encoding_name)
            bit_count = int(encoding.findtext("BitCount"))
            mask_element = encoding.find("EncodingIdentifierMask")
            _require(
                mask_element is not None and mask_element.attrib == {"Radix": "2"},
                f"base XML {encoding_name}: identifier mask must be binary",
            )
            mask = (mask_element.text or "").strip()
            _require(
                len(mask) == bit_count, f"base XML {encoding_name}: mask width mismatch"
            )
            expected_identifiers[(encoding_name, record["opcode"])] = (
                _encoding_identifier_from_mc(record[mc_key], mask, bit_count)
            )

    actual_identifiers: dict[tuple[str, int], str] = {}
    actual_identifier_order: list[tuple[str, int]] = []
    for index, addition in enumerate(
        delta.findall("./EncodingIdentifierAdditions/EncodingIdentifierAddition"),
        start=1,
    ):
        context = f"identifier addition {index}"
        _require(
            not addition.attrib, f"{context}: attributes are not provenance-approved"
        )
        _require(
            [child.tag for child in addition]
            == ["EncodingName", "Opcode", "EncodingIdentifier"],
            f"{context}: unexpected fields",
        )
        encoding_name_element = addition.find("EncodingName")
        opcode_element = addition.find("Opcode")
        _require(
            encoding_name_element is not None
            and not encoding_name_element.attrib
            and opcode_element is not None
            and not opcode_element.attrib,
            f"{context}: name/opcode attributes are not provenance-approved",
        )
        encoding_name = addition.findtext("EncodingName")
        opcode_text = addition.findtext("Opcode")
        _require(
            opcode_text is not None and opcode_text.isdecimal(),
            f"{context}: malformed opcode",
        )
        key = (encoding_name, int(opcode_text))
        identifier = addition.find("EncodingIdentifier")
        _require(
            identifier is not None and identifier.attrib == {"Radix": "2"},
            f"{context}: identifier must be binary",
        )
        _require(key not in actual_identifiers, f"{context}: duplicate {key}")
        actual_identifiers[key] = (identifier.text or "").strip()
        actual_identifier_order.append(key)
    _require(
        actual_identifiers == expected_identifiers,
        "delta identifiers are not the public MC bytes masked by the public base XML",
    )
    _require(
        actual_identifier_order == list(expected_identifiers),
        "delta identifier order differs from the provenance manifest",
    )


def _verify_source_line(
    llvm_root: Path, reference: dict[str, Any], context: str
) -> str:
    path = llvm_root / reference["path"]
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError as error:
        raise ProvenanceError(f"{context}: cannot read {path}: {error}") from error
    line_number = reference["line"]
    _require(
        1 <= line_number <= len(lines), f"{context}: line {line_number} is out of range"
    )
    actual = lines[line_number - 1].strip()
    _require(
        actual == reference["text"].strip(),
        f'{context}: {reference["path"]}:{line_number} no longer matches the recorded source',
    )
    return actual


def _verify_mc_record(llvm_root: Path, record: dict[str, Any], context: str) -> None:
    path = llvm_root / record["path"]
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError as error:
        raise ProvenanceError(f"{context}: cannot read {path}: {error}") from error
    assembly_line = record["assembly_line"]
    check_line = record["check_line"]
    _require(
        1 <= assembly_line <= len(lines),
        f"{context}: assembly line {assembly_line} is out of range",
    )
    _require(
        1 <= check_line <= len(lines),
        f"{context}: check line {check_line} is out of range",
    )
    _require(
        lines[assembly_line - 1].strip() == record["assembly"],
        f'{context}: assembly at {record["path"]}:{assembly_line} does not match',
    )
    check = lines[check_line - 1]
    _require(f'// {record["test"]}:' in check, f"{context}: wrong MC check prefix")
    encoded_text = check.split("encoding:", maxsplit=1)
    _require(len(encoded_text) == 2, f"{context}: MC check has no encoding")
    actual_bytes = re.findall(r"0x[0-9a-fA-F]{2}", encoded_text[1])
    _require(
        [value.lower() for value in actual_bytes]
        == [value.lower() for value in record["bytes"]],
        f"{context}: MC encoding bytes do not match",
    )


def _validate_llvm(manifest: dict[str, Any], llvm_root: Path) -> None:
    try:
        head = subprocess.run(
            ["git", "-C", str(llvm_root), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError) as error:
        raise ProvenanceError(
            f"cannot identify LLVM checkout {llvm_root}: {error}"
        ) from error
    _require(
        head == manifest["llvm_commit"],
        f'LLVM HEAD is {head}; provenance requires {manifest["llvm_commit"]}',
    )

    for record in manifest["instructions"]:
        name = record["instruction"]
        _verify_source_line(
            llvm_root, record["tablegen_definition"], f"{name} definition"
        )
        _verify_source_line(llvm_root, record["tablegen_profile"], f"{name} profile")
        opcode_line = _verify_source_line(
            llvm_root, record["tablegen_opcode"], f"{name} opcode"
        )
        opcode_match = re.search(r"<0x([0-9a-fA-F]+)", opcode_line)
        _require(
            opcode_match is not None
            and int(opcode_match.group(1), 16) == record["opcode"],
            f'{name}: TableGen opcode does not match manifest opcode {record["opcode"]}',
        )
        _verify_mc_record(llvm_root, record["primary_mc"], f"{name} primary MC vector")
        if "literal_mc" in record:
            _verify_mc_record(
                llvm_root, record["literal_mc"], f"{name} literal MC vector"
            )


def validate(
    manifest_path: Path,
    *,
    isa_dir: Path | None = None,
    llvm_root: Path | None = None,
) -> None:
    """Validate the delta and, when supplied, its pinned public LLVM checkout."""

    manifest = _read_json(manifest_path)
    resolved_isa_dir = isa_dir if isa_dir is not None else manifest_path.parent
    _validate_delta(manifest, resolved_isa_dir)
    if llvm_root is not None:
        _validate_llvm(manifest, llvm_root)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path(__file__).with_name("amdgpu_isa_cdna5_gfx1251_provenance.json"),
    )
    parser.add_argument(
        "--llvm-root",
        type=Path,
        help="also verify every recorded source line against this pinned public LLVM checkout",
    )
    args = parser.parse_args(argv)
    try:
        validate(args.manifest.resolve(), llvm_root=args.llvm_root)
    except ProvenanceError as error:
        print(f"error: {error}", file=sys.stderr)
        return 1
    print("gfx1251 delta provenance verified")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
