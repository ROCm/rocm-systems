# Copyright (c) 2025-2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""AMD machine-readable ISA specification XML parser.

Parses the machine-readable XML spec and populates an ``IsaSpec`` with
encoding, instruction, and operand-type data. ISA-specific encoding rules
(naming conventions, implied-literal detection, etc.) are delegated to the
``IsaProfile`` provided at construction time.

Known XML spec bugs handled by this parser (as of spec version 1.1.1):

1. **CondtionExpression typo** - misspelled element name in all versions;
   matched verbatim via ``xml_schema.COND_EXPR``.
2. **EncodingIdentifer typo** - misspelled singular form (missing 'i');
   matched via ``xml_schema.ENCODING_IDENTIFER``. The plural form
   ``EncodingIdentifiers`` is spelled correctly.
3. **Reserved field omissions** - versions 1.0.0 and 1.1.0 omit padding
   fields from the MicrocodeFormat. The parser synthesizes them by
   detecting gaps in the declared bit offsets.
4. **ENC_VOP3PX2 (CDNA4)** - has zero encoding identifier entries and an
   all-zeros mask; skipped via ``CdnaProfile.skip_encodings``.
5. **VOPDXY dual-opcode format (RDNA3/4)** - uses ``opx``/``opy`` fields
   instead of a single ``op`` field; skipped via profile skip_encodings.
6. **V_SWAP_B32 (CDNA4)** - operands marked output-only even though the
   instruction reads both registers; compensated in codegen execute().
7. **V_FMAMK/V_FMAAK (CDNA4)** - missing ``simm32`` operand; codegen
   falls back to ``inst_.simm32`` directly.
8. **Operand direction bug** - some read-modify-write destinations are
   marked output-only instead of input+output; codegen detects these
   by checking instruction semantics that require the old value.
"""

import re
import xml.etree.ElementTree as elem_tree


from amdisa import xml_schema as xs
from amdisa.gpuisa import (
    DecodeTableEntry,
    InstEncoding,
    Instruction,
    IsaSpec,
    MicrocodeField,
    Operand,
    OperandNamePattern,
    OperandSelector,
)
from amdisa.isa_profile import IsaProfile

class Parser:
    """Parses a machine-readable AMD GPU ISA specification file.

    Attributes:
        isa_xml: Path to XML file for the ISA specification.
        profile: ISA-specific encoding rules used during parsing.
        tree: XML element tree obtained by parsing the XML file.
        root: Root of the XML element tree.
        isa_spec: ISA specification object.
        encodings_node: Element tree node pointing to the encodings.
        insts_node: Element tree node pointing to the instructions.
        operand_types_node: Element tree node pointing to the operand types.
    """

    def __init__(self, isa_xml: str, profile: IsaProfile) -> None:
        self.isa_xml = isa_xml
        self.profile = profile
        self.tree = elem_tree.parse(self.isa_xml)
        self.root = self.tree.getroot()

        isa_node = xs.get_node(self.root, xs.ISA)
        arch_node = xs.get_node(isa_node, xs.ARCH)
        arch_name_node = xs.get_node(arch_node, xs.ARCH_NAME)
        doc_node = xs.get_node(self.root, xs.DOCUMENT)
        version_node = xs.get_node(doc_node, xs.SCHEMA_VERSION)

        version = xs.get_node_text(version_node)
        arch_name_raw = xs.get_node_text(arch_name_node)
        arch_parts = arch_name_raw.split()
        arch_family = arch_parts[1].lower()
        arch_version = arch_parts[2].replace('.', '_')
        arch_name = f'{arch_family}{arch_version}'
        self.isa_spec = IsaSpec(arch_name, version, profile)

        self.encodings_node = xs.get_node(isa_node, xs.ENCODINGS)
        self.insts_node = xs.get_node(isa_node, xs.INSTS)
        self.operand_types_node = xs.get_node(isa_node, xs.OPERAND_TYPES)

    def parse(self) -> IsaSpec:
        """Parse the spec and populate the IsaSpec.

        Returns:
            Populated IsaSpec object.
        """
        self.parse_encodings()
        self.parse_insts()
        self.parse_operand_types()
        return self.isa_spec

    def parse_expr(self, expr_node: elem_tree.Element) -> str:
        """Recursively parse an expression AST into a C++ expression string."""
        expr_type = expr_node.attrib[xs.EXPR_ATTR_TYPE]

        if expr_type == xs.EXPR_TYPE_VAL_OPERATOR:
            operator = xs.get_node_text(expr_node.find(xs.OPERATOR))
            sub_expr = [
                self.parse_expr(s)
                for s in expr_node.find(xs.SUB_EXPR).findall(xs.EXPR)
            ]
            if len(sub_expr) < 2:
                raise ValueError(
                    f"Expected at least 2 sub-expressions for operator "
                    f"'{operator}', got {len(sub_expr)}"
                )
            if operator == '.fieldderef':
                if '.' in sub_expr[0]:
                    return f'{sub_expr[0].split(".")[0]}_.{sub_expr[1]}'
                return f'{sub_expr[0]}.{sub_expr[1]}'
            return f'{sub_expr[0]} {operator} {sub_expr[1]}'
        elif expr_type == xs.EXPR_TYPE_VAL_ID:
            label = xs.get_node_text(expr_node.find(xs.LABEL))
            return label.lower()
        elif expr_type == xs.EXPR_TYPE_VAL_LITERAL:
            val_node = expr_node.find(xs.VALUE)
            literal = val_node.text if val_node is not None and val_node.text else '0'
            lit_size = xs.get_node_text(expr_node.find(
                f'{xs.VALUE_TYPE}/{xs.SIZE}'
            ))
            if lit_size == '1':
                return 'true' if literal == '1' else 'false'
            return literal
        elif expr_type == xs.EXPR_TYPE_VAL_RETURN:
            # ReturnType annotations don't contribute to the C++ expression.
            return ''
        raise ValueError(
            f"Unrecognized expression type '{expr_type}' in encoding "
            f"condition AST"
        )

    def parse_condition(self, cond_node: elem_tree.Element) -> tuple[str, str]:
        """Parse an encoding condition into a (name, expression) pair."""
        cond_name = xs.get_node_text(cond_node.find(f'.//{xs.COND_NAME}'))
        cond_expr_node = cond_node.find(xs.COND_EXPR)
        if cond_expr_node is None:
            cond_expr_node = cond_node.find(xs.COND_EXPR_ALT)
        if cond_expr_node is None:
            raise xs.SchemaValueError(
                f'neither {xs.COND_EXPR} nor {xs.COND_EXPR_ALT} found '
                f'in encoding condition {cond_name!r}'
            )
        expr_node = cond_expr_node.find(xs.EXPR)
        if expr_node is None:
            raise xs.SchemaValueError(
                f'{xs.EXPR} not found in condition expression for {cond_name!r}'
            )
        expr = self.parse_expr(expr_node)

        if cond_name == 'default':
            cond_name += '_encoding'

        return (cond_name, expr)

    def parse_encoding_conditions(
        self, conds_node: elem_tree.Element
    ) -> list[tuple[str, str]]:
        """Parse all encoding conditions under the given node."""
        return [
            self.parse_condition(cond_node)
            for cond_node in conds_node.findall(xs.ENCODING_COND)
        ]

    def parse_ucode_bitmap(
        self, enc_node: elem_tree.Element, bit_cnt: int
    ) -> tuple[list[MicrocodeField], int, int, int]:
        """Parse an encoding's microcode bitmap.

        Returns:
            Tuple of (microcode fields, encoding field bit count, opcode
            field bit count, opm field bit count).

        Raises:
            ValueError: If the bitmap is missing the required ``encoding``
                or ``op`` field.
        """
        ucode_fields = []
        enc_field_bit_cnt: int | None = None
        op_field_bit_cnt: int | None = None
        opm_field_bit_cnt = 0
        for field in enc_node.findall(
            f'./{xs.UCODE_FMT}/{xs.BITMAP}/{xs.FIELD}'
        ):
            field_name = xs.get_node_text(field.find(xs.FIELD_NAME)).lower()
            field_bit_cnt = int(xs.get_node_text(
                field.find(f'{xs.BIT_LAYOUT}/{xs.RANGE}/{xs.BIT_CNT}')
            ))
            field_bit_offset = int(xs.get_node_text(
                field.find(f'{xs.BIT_LAYOUT}/{xs.RANGE}/{xs.BIT_OFF}')
            ))
            ucode_fields.append(
                MicrocodeField(field_name, field_bit_cnt, field_bit_offset)
            )
            if field_name == 'encoding':
                enc_field_bit_cnt = field_bit_cnt
            elif field_name == 'op':
                op_field_bit_cnt = field_bit_cnt
            elif field_name == 'opm':
                opm_field_bit_cnt = field_bit_cnt
        ucode_fields.sort(key=lambda x: x.bit_offset)

        # XML bug: versions 1.0.0 and 1.1.0 omit reserved/padding fields
        # from the MicrocodeFormat. We synthesize them by detecting gaps
        # between declared fields. The generated C++ bitfield structs need
        # every bit accounted for to match the hardware encoding layout.
        ucode_fields_bit_cnt = sum(f.bit_cnt for f in ucode_fields)
        if ucode_fields_bit_cnt != bit_cnt:
            next_bit_off = 0
            ucode_pads = []
            for ucode_field in ucode_fields:
                if next_bit_off != int(ucode_field.bit_offset):
                    pad_bit_cnt = int(ucode_field.bit_offset) - next_bit_off
                    pad_name = f'pad_{next_bit_off}'
                    if pad_bit_cnt > 1:
                        pad_name += f'_{next_bit_off + pad_bit_cnt - 1}'
                    ucode_pads.append(
                        MicrocodeField(pad_name, pad_bit_cnt, next_bit_off)
                    )
                next_bit_off = (
                    int(ucode_field.bit_offset) + int(ucode_field.bit_cnt)
                )
            # Also check for a trailing gap after the last declared field.
            if next_bit_off < bit_cnt:
                pad_bit_cnt = bit_cnt - next_bit_off
                pad_name = f'pad_{next_bit_off}'
                if pad_bit_cnt > 1:
                    pad_name += f'_{next_bit_off + pad_bit_cnt - 1}'
                ucode_pads.append(
                    MicrocodeField(pad_name, pad_bit_cnt, next_bit_off)
                )
            ucode_fields.extend(ucode_pads)

        enc_name = xs.get_node_text(xs.get_node(enc_node, xs.ENCODING_NAME))
        if enc_field_bit_cnt is None:
            raise ValueError(
                f"Encoding '{enc_name}' bitmap missing required 'encoding' "
                f"field"
            )
        if op_field_bit_cnt is None:
            op_field_bit_cnt = 0
        return ucode_fields, enc_field_bit_cnt, op_field_bit_cnt, opm_field_bit_cnt

    def parse_encoding_identifers(
        self, enc_node: elem_tree.Element, inst_enc: InstEncoding
    ) -> None:
        """Parse encoding identifier masks to populate the primary decode table.

        For alternate encodings, this method also derives two behavioral
        properties from the XML data:

        * **primary decode**: If every encoding identifier value already
          exists in the primary table (placed by the parent encoding),
          the alternate reuses those entries and is marked as a
          primary-decode alternate.
        * **unique ops**: If the alternate's opcode values do not collide
          with the parent's, it has unique opcodes and gets its own
          entries in the sub-decode table.
        """
        max_enc_bits = self.profile.max_enc_bits
        enc_ids_node = xs.get_node(enc_node, xs.ENCODING_IDENTIFERS)

        enc_id_radix = int(
            enc_node.find(xs.ENCODING_IDENTIFIER_MASK).attrib[
                xs.ENC_IDENTIFER_ATTR_RADIX
            ]
        )

        enc_id_mask = xs.get_node_text(enc_node.find(xs.ENCODING_IDENTIFIER_MASK))
        bit_masks = [
            (x.start(), x.end()) for x in re.finditer(r'1+', enc_id_mask)
        ]

        # The encoding identifier mask is a binary string where '1' bits
        # mark the encoding field and '0' bits separate encoding from opcode.
        # Most encodings have a single contiguous run of '1' bits (the
        # encoding field followed by the opcode field). Some encodings
        # (e.g., VOP3 variants) have a split mask with two separate runs
        # of '1' bits, the first run is the encoding field and the second
        # is the opcode field.
        flat_enc_mask = bit_masks[0]
        if len(bit_masks) == 1:
            op_mask = (
                bit_masks[0][0] + inst_enc.enc_field_bit_cnt,
                bit_masks[0][0]
                + inst_enc.enc_field_bit_cnt
                + inst_enc.op_field_bit_cnt,
            )
            if (flat_enc_mask[1] - flat_enc_mask[0]) > max_enc_bits:
                flat_enc_mask = (
                    flat_enc_mask[0],
                    flat_enc_mask[0] + max_enc_bits,
                )
        else:
            op_mask = (
                bit_masks[1][0],
                bit_masks[1][0] + inst_enc.op_field_bit_cnt,
            )

        dont_care_bits = max_enc_bits - (
            flat_enc_mask[1] - flat_enc_mask[0]
        )
        effective_op_bits = (
            inst_enc.op_field_bit_cnt + inst_enc.opm_field_bit_cnt
        )
        max_num_opcodes = pow(2, effective_op_bits)
        sub_decode_funcs = ['decodeInvalid'] * max_num_opcodes
        primary_dt_ptrs = None
        dt = self.isa_spec.primary_decode_table

        if inst_enc.is_alt:
            primary_dt_ptrs = inst_enc.parent_enc.primary_dt_ptrs
        else:
            inst_enc.primary_dt_ptrs = [-1] * max_num_opcodes
            primary_dt_ptrs = inst_enc.primary_dt_ptrs

        all_primary_slots_exist = True
        has_unique_opcode = False
        deferred_entries: list[int] = []

        for enc_id in enc_ids_node:
            enc_id_text = xs.get_node_text(enc_id)
            enc_val = (
                int(
                    enc_id_text[flat_enc_mask[0] : flat_enc_mask[1]],
                    enc_id_radix,
                )
                << dont_care_bits
            )
            if dt[enc_val]:
                if inst_enc.is_alt:
                    if not inst_enc.parent_enc.is_primary_decode:
                        inst_enc.is_primary_decode = False
                else:
                    inst_enc.is_primary_decode = False
                    if dt[enc_val].is_primary:
                        dt[enc_val].is_primary = False
                        dt[enc_val].sub_decode_funcs = sub_decode_funcs
                    if (
                        dt[enc_val].enc.enc_name != inst_enc.enc_name
                        and dt[enc_val].enc.enc_name
                        != inst_enc.parent_enc.enc_name
                    ):
                        raise ValueError(
                            f'Double-mapped encoding in primary decode table: '
                            f'{inst_enc.enc_name} conflicts with '
                            f'{dt[enc_val].enc.enc_name} at index {enc_val}'
                        )
            elif inst_enc.is_alt:
                # Primary table slot doesn't exist yet. Defer creation
                # until we know whether this alternate has unique opcodes
                # (only unique-ops alternates need new primary entries).
                all_primary_slots_exist = False
                deferred_entries.append(enc_val)
            else:
                dt[enc_val] = DecodeTableEntry(
                    inst_enc, pow(2, dont_care_bits)
                )
            if inst_enc.op_field_bit_cnt == 0:
                opcode = 0
            else:
                opcode = int(
                    enc_id_text[op_mask[0] : op_mask[1]], enc_id_radix
                )
            if primary_dt_ptrs[opcode] != -1:
                if inst_enc.is_alt:
                    continue
                if primary_dt_ptrs[opcode] == enc_val:
                    continue
                raise ValueError(
                    f'Double-mapped opcode {opcode} in {inst_enc.enc_name}: '
                    f'slot already occupied'
                )
            has_unique_opcode = True
            primary_dt_ptrs[opcode] = enc_val

        # When an opm (opcode modification) field is present, the encoding
        # identifiers only encode the base op value. Mirror the populated
        # slots into the upper half so that instructions with opm=1 (whose
        # XML opcode = op + 2^op_bits) resolve to the same decode entry.
        if inst_enc.opm_field_bit_cnt > 0:
            base_count = pow(2, inst_enc.op_field_bit_cnt)
            for opm_val in range(1, pow(2, inst_enc.opm_field_bit_cnt)):
                offset = opm_val * base_count
                for op in range(base_count):
                    if primary_dt_ptrs[op] != -1:
                        primary_dt_ptrs[op + offset] = primary_dt_ptrs[op]

        if inst_enc.is_alt:
            inst_enc.has_unique_ops = has_unique_opcode
            if inst_enc.has_unique_ops:
                self.isa_spec.alt_encs_with_unique_ops.add(inst_enc.enc_name)
                # Find an existing parent entry to copy sub-decode state from.
                parent_entry = next(
                    (dt[v] for v in inst_enc.parent_enc.primary_dt_ptrs
                     if v != -1 and dt[v] is not None),
                    None,
                )
                for enc_val in deferred_entries:
                    if dt[enc_val] is None:
                        new_entry = DecodeTableEntry(
                            inst_enc.parent_enc, pow(2, dont_care_bits)
                        )
                        if parent_entry is not None:
                            new_entry.is_primary = parent_entry.is_primary
                            new_entry.decode_func = parent_entry.decode_func
                            new_entry.sub_decode_table = (
                                parent_entry.sub_decode_table
                            )
                            new_entry.sub_decode_funcs = (
                                parent_entry.sub_decode_funcs
                            )
                        dt[enc_val] = new_entry
            if all_primary_slots_exist:
                self.isa_spec.alt_encs_primary_decode.add(inst_enc.enc_name)

        if not inst_enc.is_primary_decode and not inst_enc.is_alt:
            for i in primary_dt_ptrs:
                if i != -1:
                    dte = dt[i]
                    dte.decode_func = (
                        f'subDecode{inst_enc.fmt_enc_name}'
                    )
                    dte.sub_decode_table = (
                        f'sub_decode_{inst_enc.fmt_enc_name}'.lower()
                    )
                    if dte.sub_decode_funcs is None:
                        dte.is_primary = False
                        dte.sub_decode_funcs = list(sub_decode_funcs)
                    for j in range(1, dte.num_dupe_entries):
                        dt[i + j] = dte

    def parse_encodings(self) -> None:
        """Parse the XML and generate the internal encoding objects.

        For each encoding in the XML:

        1. Parse microcode bitmap and encoding conditions.
        2. Determine if the encoding is alternate using the ISA profile.
        3. For alternates: link to parent, propagate conditions, and detect
           implied-literal status from the encoding conditions.
        4. Parse encoding identifiers for all primary encodings and all
           alternate encodings (the identifier parser derives whether the
           alternate has unique opcodes or shares the parent's decode).
        """
        for enc_node in self.encodings_node:
            enc_name_node = xs.get_node(enc_node, xs.ENCODING_NAME)
            bit_cnt_node = xs.get_node(enc_node, xs.BIT_CNT)

            enc_name = xs.get_node_text(enc_name_node)
            if enc_name in self.profile.skip_encodings:
                continue
            order = int(enc_node.attrib[xs.ENC_ATTR_ORDER])
            bit_cnt = int(xs.get_node_text(bit_cnt_node))

            ucode_fields, enc_field_bit_cnt, op_field_bit_cnt, opm_field_bit_cnt = (
                self.parse_ucode_bitmap(enc_node, bit_cnt)
            )
            enc_conds_node = enc_node.find(xs.ENCODING_CONDS)
            enc_conds = self.parse_encoding_conditions(enc_conds_node)

            ucode_fields.sort(key=lambda x: x.bit_offset)
            inst_enc = InstEncoding(
                enc_name,
                order,
                bit_cnt,
                enc_field_bit_cnt,
                op_field_bit_cnt,
                ucode_fields,
                enc_conds,
            )
            inst_enc.opm_field_bit_cnt = opm_field_bit_cnt

            if self.profile.is_alt_encoding(enc_name):
                inst_enc.is_alt = True
                parent_name = self.profile.derive_parent_enc_name(enc_name)
                if parent_name in self.profile.skip_encodings:
                    continue
                if parent_name not in self.isa_spec.encoding_map:
                    raise KeyError(
                        f'Parent encoding {parent_name!r} not found for '
                        f'alternate encoding {enc_name!r}. Ensure primary '
                        f'encodings appear before their alternates in the XML.'
                    )
                inst_enc.parent_enc = self.isa_spec.encoding_map[parent_name]

                for enc_cond in inst_enc.enc_conds:
                    if enc_cond not in inst_enc.parent_enc.enc_conds:
                        inst_enc.parent_enc.enc_conds.append(enc_cond)

                has_implied_literal = self.profile.is_implied_literal_encoding(
                    enc_name,
                    inst_enc.enc_conds,
                    inst_enc.bit_cnt,
                    inst_enc.parent_enc.bit_cnt,
                )
                if has_implied_literal:
                    inst_enc.is_implied_literal_enc = True
                    self.isa_spec.alt_encs_with_implied_literal.add(enc_name)

            if not inst_enc.is_alt or not self.profile.skip_inst_encoding(
                enc_name, 'default'
            ):
                self.parse_encoding_identifers(enc_node, inst_enc)

            self.isa_spec.inst_encodings.append(inst_enc)
            if enc_name in self.isa_spec.encoding_map:
                raise KeyError(f'Duplicate encoding found: {enc_name}')
            self.isa_spec.encoding_map[enc_name] = inst_enc

    def parse_insts(self) -> None:
        """Parse instructions and populate the decode table.

        Instructions under non-default encoding conditions (DPP, SDWA,
        etc.) and instructions under skipped alternate encodings (e.g.,
        FLAT segment variants) are filtered out by the ISA profile's
        ``skip_inst_encoding()`` method.
        """
        for inst_node in self.insts_node:
            inst_name_node = xs.get_node(inst_node, xs.INST_NAME)
            inst_encs_node = xs.get_node(inst_node, xs.INST_ENCODINGS)
            inst_name = inst_name_node.text
            for inst_enc_node in inst_encs_node:
                enc_name_node = xs.get_node(inst_enc_node, xs.ENCODING_NAME)
                enc_cond_node = xs.get_node(inst_enc_node, xs.ENCODING_COND)
                opcode_node = xs.get_node(inst_enc_node, xs.OPCODE)
                operands_node = xs.get_node(inst_enc_node, xs.OPERANDS)
                enc_name = xs.get_node_text(enc_name_node)
                if enc_name in self.profile.skip_encodings:
                    continue
                enc_cond = xs.get_node_text(enc_cond_node)
                if self.profile.skip_inst_encoding(enc_name, enc_cond):
                    continue
                opcode = int(xs.get_node_text(opcode_node))
                opnds = []
                for opnd in operands_node:
                    is_in = (
                        opnd.attrib[xs.OPERAND_ATTR_INPUT].lower() == 'true'
                    )
                    is_out = (
                        opnd.attrib[xs.OPERAND_ATTR_OUTPUT].lower() == 'true'
                    )
                    is_implicit = (
                        opnd.attrib[xs.OPERAND_ATTR_IS_IMPLICIT].lower()
                        == 'true'
                    )
                    is_bin_ucode_required = (
                        opnd.attrib[
                            xs.OPERAND_ATTR_IS_BINARY_MICROCODE_REQUIRED
                        ].lower()
                        == 'true'
                    )
                    order = int(opnd.attrib[xs.OPERAND_ATTR_ORDER])
                    field_name_node = opnd.find(xs.FIELD_NAME)
                    opnd_size = int(xs.get_node_text(opnd.find(xs.OPERAND_SIZE)))
                    opnd_type = xs.get_node_text(opnd.find(xs.OPERAND_TYPE))
                    if field_name_node is not None:
                        field_name = xs.get_node_text(field_name_node).lower()
                        opnds.append(
                            Operand(
                                field_name,
                                opnd_size,
                                opnd_type,
                                is_in,
                                is_out,
                                is_implicit,
                                is_bin_ucode_required,
                                order,
                            )
                        )
                opnds.sort(key=lambda x: x.order)

                enc = self.isa_spec.encoding_map[enc_name]
                is_implied_literal = (
                    enc_name in self.isa_spec.alt_encs_with_implied_literal
                )
                inst = Instruction(
                    inst_name, enc_name, opcode, opnds, is_implied_literal
                )

                if enc.is_alt:
                    enc.parent_enc.insts.append(inst)
                else:
                    enc.insts.append(inst)

                if is_implied_literal:
                    enc.parent_enc.implied_literal_ops.append(
                        str(inst.opcode)
                    )

                if (
                    not enc.is_alt
                    or enc.enc_name
                    in self.isa_spec.alt_encs_with_unique_ops
                ):
                    if enc.is_alt:
                        primary_dt_ptrs = enc.parent_enc.primary_dt_ptrs
                    else:
                        primary_dt_ptrs = enc.primary_dt_ptrs
                    dt = self.isa_spec.primary_decode_table
                    dt_ptr = primary_dt_ptrs[inst.opcode]
                    decode_func = f'decode{inst.fmt_name}'
                    if enc.is_primary_decode:
                        dt[dt_ptr].decode_func = decode_func
                        dt[dt_ptr].inst_name = inst.fmt_name
                        for i in range(1, dt[dt_ptr].num_dupe_entries):
                            if dt[dt_ptr + i] is not None:
                                raise ValueError(
                                    f'Entry already exists in the decode '
                                    f'table: opcode {inst.opcode}, inst '
                                    f'{inst.mnemonic}'
                                )
                            else:
                                dt[dt_ptr + i] = dt[dt_ptr]
                    else:
                        dt[dt_ptr].sub_decode_funcs[inst.opcode] = decode_func

    def parse_operand_types(self) -> None:
        """Parse operand type definitions and build OpSel enum data."""
        for opnd_type in self.operand_types_node:
            opnd_type_name = xs.get_node_text(
                opnd_type.find(f'.//{xs.OPERAND_TYPE_NAME}')
            )
            opnd_predefined_val = opnd_type.find(
                f'.//{xs.OPERAND_PREDEFINED_VALS}'
            )
            if opnd_predefined_val is not None:
                predef_vals_list: list[tuple[str, str]] = []
                name_patterns: list[OperandNamePattern] = []
                first = True
                last = False
                current_range_prefix = ''
                current_range_min_enum = ''
                current_range_max_idx = -1
                current_int_min_enum = ''
                pairs = list(opnd_predefined_val)
                for pair_idx, predef_val_pair in enumerate(pairs):
                    predef_name = xs.get_node_text(
                        predef_val_pair.find(xs.NAME)
                    )
                    predef_val = xs.get_node_text(
                        predef_val_pair.find(xs.VALUE)
                    )
                    original_name = predef_name
                    reg_match = re.match(
                        r'^\s*(v|s|ttmp|acc)([0-9]+)$', predef_name
                    )
                    if reg_match:
                        prefix = reg_match.group(1)
                        reg_idx = int(reg_match.group(2))
                        label_map = {
                            'v': 'VGPR', 's': 'SGPR',
                            'ttmp': 'TTMP', 'acc': 'ACC',
                        }
                        label = label_map[prefix]
                        # Reset state when transitioning between register
                        # prefixes so that min/max markers are emitted
                        # correctly for each category.
                        if prefix != current_range_prefix:
                            first = True
                        if first:
                            first = False
                            predef_name = f'{opnd_type_name}_{label}_MIN'
                            current_range_prefix = prefix
                            current_range_min_enum = predef_name
                            current_range_max_idx = reg_idx
                        else:
                            if reg_idx > current_range_max_idx:
                                current_range_max_idx = reg_idx
                            # Peek ahead: emit MAX if the next entry is not
                            # the same register prefix (or this is the last).
                            next_pair = None
                            if pair_idx + 1 < len(pairs):
                                next_name = xs.get_node_text(
                                    pairs[pair_idx + 1].find(xs.NAME)
                                )
                                next_match = re.match(
                                    r'^\s*(v|s|ttmp|acc)([0-9]+)$',
                                    next_name,
                                )
                                if next_match:
                                    next_pair = next_match.group(1)
                            if next_pair != prefix:
                                last = True
                                predef_name = (
                                    f'{opnd_type_name}_{label}_MAX'
                                )
                                name_patterns.append(OperandNamePattern(
                                    OperandNamePattern.REG_RANGE,
                                    prefix=current_range_prefix,
                                    min_enum=current_range_min_enum,
                                    max_enum=predef_name,
                                ))
                            else:
                                continue
                    else:
                        try:
                            int_val = int(predef_name)
                            if int_val < 0:
                                if int_val == -1:
                                    predef_name = (
                                        opnd_type_name + '_NEG_INT_MIN'
                                    )
                                    current_int_min_enum = predef_name
                                elif int_val == -16:
                                    predef_name = (
                                        opnd_type_name + '_NEG_INT_MAX'
                                    )
                                    name_patterns.append(OperandNamePattern(
                                        OperandNamePattern.NEG_INT,
                                        min_enum=current_int_min_enum,
                                        max_enum=predef_name,
                                    ))
                                else:
                                    continue
                            else:
                                if int_val == 0:
                                    predef_name = (
                                        opnd_type_name + '_POS_INT_MIN'
                                    )
                                    current_int_min_enum = predef_name
                                elif int_val == 64:
                                    predef_name = (
                                        opnd_type_name + '_POS_INT_MAX'
                                    )
                                    name_patterns.append(OperandNamePattern(
                                        OperandNamePattern.POS_INT,
                                        min_enum=current_int_min_enum,
                                        max_enum=predef_name,
                                    ))
                                else:
                                    continue
                        except ValueError:
                            try:
                                flt_val = float(predef_name)
                                predef_name = (
                                    f'{opnd_type_name}_FLOAT_'
                                    f'{self.profile.flt_name_map[flt_val]}'
                                )
                                name_patterns.append(OperandNamePattern(
                                    OperandNamePattern.FLOAT_CONST,
                                    operand_name=original_name,
                                    enum_name=predef_name,
                                ))
                            except ValueError:
                                predef_name = (
                                    f'{opnd_type_name}_{predef_name.upper()}'
                                )
                                if original_name.lower() == 'src_literal':
                                    name_patterns.append(OperandNamePattern(
                                        OperandNamePattern.LITERAL,
                                        enum_name=predef_name,
                                    ))
                                else:
                                    name_patterns.append(OperandNamePattern(
                                        OperandNamePattern.NAMED,
                                        operand_name=original_name,
                                        enum_name=predef_name,
                                    ))
                    if last:
                        first = True
                        last = False
                    predef_vals_list.append((predef_name, predef_val))
                self.isa_spec.opnd_selectors.append(
                    OperandSelector(
                        opnd_type_name, predef_vals_list, name_patterns
                    )
                )
            self.isa_spec.operand_types.append(opnd_type_name)
