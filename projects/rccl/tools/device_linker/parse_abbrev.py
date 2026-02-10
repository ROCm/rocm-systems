#!/usr/bin/env python3
"""
Parse .debug_abbrev section from ELF file and output in llvm-dwarfdump format.
Correctly decodes ULEB128 and shows symbolic names for tags, attributes, and forms.
"""

import sys
import struct
from collections import defaultdict

# DWARF tag constants
DW_TAG = {
    0x01: "DW_TAG_array_type",
    0x02: "DW_TAG_class_type",
    0x03: "DW_TAG_entry_point",
    0x04: "DW_TAG_enumeration_type",
    0x05: "DW_TAG_formal_parameter",
    0x08: "DW_TAG_imported_declaration",
    0x0a: "DW_TAG_label",
    0x0b: "DW_TAG_lexical_block",
    0x0d: "DW_TAG_member",
    0x0f: "DW_TAG_pointer_type",
    0x10: "DW_TAG_reference_type",
    0x11: "DW_TAG_compile_unit",
    0x12: "DW_TAG_string_type",
    0x13: "DW_TAG_structure_type",
    0x15: "DW_TAG_subroutine_type",
    0x16: "DW_TAG_typedef",
    0x17: "DW_TAG_union_type",
    0x18: "DW_TAG_unspecified_parameters",
    0x19: "DW_TAG_variant",
    0x1a: "DW_TAG_common_block",
    0x1b: "DW_TAG_common_inclusion",
    0x1c: "DW_TAG_inheritance",
    0x1d: "DW_TAG_inlined_subroutine",
    0x1e: "DW_TAG_module",
    0x1f: "DW_TAG_ptr_to_member_type",
    0x20: "DW_TAG_set_type",
    0x21: "DW_TAG_subrange_type",
    0x22: "DW_TAG_with_stmt",
    0x23: "DW_TAG_access_declaration",
    0x24: "DW_TAG_base_type",
    0x25: "DW_TAG_catch_block",
    0x26: "DW_TAG_const_type",
    0x27: "DW_TAG_constant",
    0x28: "DW_TAG_enumerator",
    0x29: "DW_TAG_file_type",
    0x2a: "DW_TAG_friend",
    0x2b: "DW_TAG_namelist",
    0x2c: "DW_TAG_namelist_item",
    0x2d: "DW_TAG_packed_type",
    0x2e: "DW_TAG_subprogram",
    0x2f: "DW_TAG_template_type_parameter",
    0x30: "DW_TAG_template_value_parameter",
    0x31: "DW_TAG_thrown_type",
    0x32: "DW_TAG_try_block",
    0x33: "DW_TAG_variant_part",
    0x34: "DW_TAG_variable",
    0x35: "DW_TAG_volatile_type",
    0x36: "DW_TAG_dwarf_procedure",
    0x37: "DW_TAG_restrict_type",
    0x38: "DW_TAG_interface_type",
    0x39: "DW_TAG_namespace",
    0x3a: "DW_TAG_imported_module",
    0x3b: "DW_TAG_unspecified_type",
    0x3c: "DW_TAG_partial_unit",
    0x3d: "DW_TAG_imported_unit",
    0x3e: "DW_TAG_condition",
    0x3f: "DW_TAG_shared_type",
    0x40: "DW_TAG_type_unit",
    0x41: "DW_TAG_rvalue_reference_type",
    0x42: "DW_TAG_template_alias",
    0x43: "DW_TAG_coarray_type",
    0x44: "DW_TAG_generic_subrange",
    0x45: "DW_TAG_dynamic_type",
    0x46: "DW_TAG_atomic_type",
    0x47: "DW_TAG_call_site",
    0x48: "DW_TAG_call_site_parameter",
    0x49: "DW_TAG_skeleton_unit",
    0x4a: "DW_TAG_immutable_type",
}

# DWARF attribute constants
DW_AT = {
    0x01: "DW_AT_sibling",
    0x02: "DW_AT_location",
    0x03: "DW_AT_name",
    0x09: "DW_AT_ordering",
    0x0a: "DW_AT_byte_size",
    0x0b: "DW_AT_bit_offset",
    0x0c: "DW_AT_bit_size",
    0x0d: "DW_AT_stmt_list",
    0x0e: "DW_AT_low_pc",
    0x0f: "DW_AT_high_pc",
    0x10: "DW_AT_language",
    0x11: "DW_AT_member",
    0x12: "DW_AT_discr",
    0x13: "DW_AT_discr_value",
    0x14: "DW_AT_visibility",
    0x15: "DW_AT_import",
    0x16: "DW_AT_string_length",
    0x17: "DW_AT_common_reference",
    0x18: "DW_AT_comp_dir",
    0x19: "DW_AT_const_value",
    0x1a: "DW_AT_containing_type",
    0x1b: "DW_AT_default_value",
    0x1c: "DW_AT_inline",
    0x1d: "DW_AT_is_optional",
    0x1e: "DW_AT_lower_bound",
    0x20: "DW_AT_producer",
    0x21: "DW_AT_prototyped",
    0x22: "DW_AT_return_addr",
    0x25: "DW_AT_start_scope",
    0x27: "DW_AT_bit_stride",
    0x2a: "DW_AT_upper_bound",
    0x2c: "DW_AT_abstract_origin",
    0x2e: "DW_AT_accessibility",
    0x2f: "DW_AT_address_class",
    0x30: "DW_AT_artificial",
    0x31: "DW_AT_base_types",
    0x32: "DW_AT_calling_convention",
    0x33: "DW_AT_count",
    0x34: "DW_AT_data_member_location",
    0x35: "DW_AT_decl_column",
    0x36: "DW_AT_decl_file",
    0x37: "DW_AT_decl_line",
    0x38: "DW_AT_declaration",
    0x39: "DW_AT_discr_list",
    0x3a: "DW_AT_encoding",
    0x3b: "DW_AT_external",
    0x3c: "DW_AT_frame_base",
    0x3e: "DW_AT_friend",
    0x3f: "DW_AT_identifier_case",
    0x40: "DW_AT_macro_info",
    0x41: "DW_AT_namelist_item",
    0x42: "DW_AT_priority",
    0x43: "DW_AT_segment",
    0x44: "DW_AT_specification",
    0x45: "DW_AT_static_link",
    0x46: "DW_AT_type",
    0x47: "DW_AT_use_location",
    0x48: "DW_AT_variable_parameter",
    0x49: "DW_AT_virtuality",
    0x4a: "DW_AT_vtable_elem_location",
    0x4b: "DW_AT_allocated",
    0x4c: "DW_AT_associated",
    0x4d: "DW_AT_data_location",
    0x4e: "DW_AT_byte_stride",
    0x4f: "DW_AT_entry_pc",
    0x50: "DW_AT_use_UTF8",
    0x51: "DW_AT_extension",
    0x52: "DW_AT_ranges",
    0x53: "DW_AT_trampoline",
    0x54: "DW_AT_call_column",
    0x55: "DW_AT_call_file",
    0x56: "DW_AT_call_line",
    0x57: "DW_AT_description",
    0x58: "DW_AT_binary_scale",
    0x59: "DW_AT_decimal_scale",
    0x5a: "DW_AT_small",
    0x5b: "DW_AT_decimal_sign",
    0x5c: "DW_AT_digit_count",
    0x5d: "DW_AT_picture_string",
    0x5e: "DW_AT_mutable",
    0x5f: "DW_AT_threads_scaled",
    0x60: "DW_AT_explicit",
    0x61: "DW_AT_object_pointer",
    0x62: "DW_AT_endianity",
    0x63: "DW_AT_elemental",
    0x64: "DW_AT_pure",
    0x65: "DW_AT_recursive",
    0x66: "DW_AT_signature",
    0x67: "DW_AT_main_subprogram",
    0x68: "DW_AT_data_bit_offset",
    0x69: "DW_AT_const_expr",
    0x6a: "DW_AT_enum_class",
    0x6b: "DW_AT_linkage_name",
    0x6c: "DW_AT_string_length_bit",
    0x6d: "DW_AT_string_length_byte",
    0x6e: "DW_AT_rank",
    0x6f: "DW_AT_str_offsets_base",
    0x70: "DW_AT_addr_base",
    0x71: "DW_AT_rnglists_base",
    0x72: "DW_AT_dwo_name",
    0x73: "DW_AT_reference",
    0x74: "DW_AT_rvalue_reference",
    0x75: "DW_AT_macros",
    0x76: "DW_AT_call_all_calls",
    0x77: "DW_AT_call_all_source_calls",
    0x78: "DW_AT_call_all_tail_calls",
    0x79: "DW_AT_call_return_pc",
    0x7a: "DW_AT_call_value",
    0x7b: "DW_AT_call_origin",
    0x7c: "DW_AT_call_parameter",
    0x7d: "DW_AT_call_pc",
    0x7e: "DW_AT_call_tail_call",
    0x7f: "DW_AT_call_target",
    0x80: "DW_AT_call_target_clobbered",
    0x81: "DW_AT_call_data_location",
    0x82: "DW_AT_call_data_value",
    0x83: "DW_AT_noreturn",
    0x84: "DW_AT_alignment",
    0x85: "DW_AT_export_symbols",
    0x86: "DW_AT_deleted",
    0x87: "DW_AT_defaulted",
    0x88: "DW_AT_loclists_base",
}

# DWARF form constants
DW_FORM = {
    0x01: "DW_FORM_addr",
    0x03: "DW_FORM_block2",
    0x04: "DW_FORM_block4",
    0x05: "DW_FORM_data2",
    0x06: "DW_FORM_data4",
    0x07: "DW_FORM_data8",
    0x08: "DW_FORM_string",
    0x09: "DW_FORM_block",
    0x0a: "DW_FORM_block1",
    0x0b: "DW_FORM_data1",
    0x0c: "DW_FORM_flag",
    0x0d: "DW_FORM_sdata",
    0x0e: "DW_FORM_strp",
    0x0f: "DW_FORM_udata",
    0x10: "DW_FORM_ref_addr",
    0x11: "DW_FORM_ref1",
    0x12: "DW_FORM_ref2",
    0x13: "DW_FORM_ref4",
    0x14: "DW_FORM_ref8",
    0x15: "DW_FORM_ref_udata",
    0x16: "DW_FORM_indirect",
    0x17: "DW_FORM_sec_offset",
    0x18: "DW_FORM_exprloc",
    0x19: "DW_FORM_flag_present",
    # 0x1a (26) and 0x1b (27) are INVALID - gap in DWARF spec
    # These should never appear in valid DWARF
    0x1a: "DW_FORM_<INVALID_0x1a>",  # INVALID FORM VALUE
    0x1b: "DW_FORM_<INVALID_0x1b>",  # INVALID FORM VALUE
    0x20: "DW_FORM_ref_sig8",
    0x21: "DW_FORM_strx",
    0x22: "DW_FORM_addrx",
    0x23: "DW_FORM_ref_sup4",
    0x24: "DW_FORM_strp_sup",
    0x25: "DW_FORM_data16",
    0x26: "DW_FORM_line_strp",
    0x27: "DW_FORM_ref_sup8",
    0x28: "DW_FORM_implicit_const",
    0x29: "DW_FORM_loclistx",
    0x2a: "DW_FORM_rnglistx",
    0x2b: "DW_FORM_ref_sup",
    0x2c: "DW_FORM_strx1",
    0x2d: "DW_FORM_strx2",
    0x2e: "DW_FORM_strx3",
    0x2f: "DW_FORM_strx4",
    0x30: "DW_FORM_addrx1",
    0x31: "DW_FORM_addrx2",
    0x32: "DW_FORM_addrx3",
    0x33: "DW_FORM_addrx4",
}

DW_CHILDREN = {
    0x00: "DW_CHILDREN_no",
    0x01: "DW_CHILDREN_yes",
}

def decode_uleb128(data, offset):
    """Decode ULEB128 value, return (value, bytes_consumed)"""
    result = 0
    shift = 0
    bytes_read = 0
    
    while offset + bytes_read < len(data):
        byte = data[offset + bytes_read]
        bytes_read += 1
        result |= (byte & 0x7f) << shift
        if (byte & 0x80) == 0:
            break
        shift += 7
    
    return result, bytes_read

def parse_elf_sections(elf_data):
    """Parse ELF file and return section headers"""
    if len(elf_data) < 64:
        raise ValueError("ELF file too small")
    
    # Check ELF magic
    if elf_data[0:4] != b'\x7fELF':
        raise ValueError("Not an ELF file")
    
    ei_class = elf_data[4]
    ei_data = elf_data[5]
    
    if ei_class == 1:  # 32-bit
        raise ValueError("32-bit ELF not supported")
    
    # 64-bit ELF
    if ei_data == 1:  # little endian
        endian = '<'
    else:  # big endian
        endian = '>'
    
    # Read ELF header
    e_shoff = struct.unpack_from(endian + 'Q', elf_data, 0x28)[0]
    e_shentsize = struct.unpack_from(endian + 'H', elf_data, 0x3a)[0]
    e_shnum = struct.unpack_from(endian + 'H', elf_data, 0x3c)[0]
    e_shstrndx = struct.unpack_from(endian + 'H', elf_data, 0x3e)[0]
    
    # Read section string table
    shstrtab_offset = e_shoff + e_shstrndx * e_shentsize
    shstrtab_offset_in_file = struct.unpack_from(endian + 'Q', elf_data, shstrtab_offset + 0x18)[0]
    shstrtab_size = struct.unpack_from(endian + 'Q', elf_data, shstrtab_offset + 0x20)[0]
    shstrtab = elf_data[shstrtab_offset_in_file:shstrtab_offset_in_file + shstrtab_size]
    
    # Read all section headers
    sections = {}
    for i in range(e_shnum):
        sh_offset = e_shoff + i * e_shentsize
        sh_name_idx = struct.unpack_from(endian + 'I', elf_data, sh_offset)[0]
        sh_type = struct.unpack_from(endian + 'I', elf_data, sh_offset + 0x4)[0]
        sh_offset_in_file = struct.unpack_from(endian + 'Q', elf_data, sh_offset + 0x18)[0]
        sh_size = struct.unpack_from(endian + 'Q', elf_data, sh_offset + 0x20)[0]
        
        # Get section name
        name_end = shstrtab.find(b'\x00', sh_name_idx)
        if name_end == -1:
            name = shstrtab[sh_name_idx:].decode('utf-8', errors='ignore')
        else:
            name = shstrtab[sh_name_idx:name_end].decode('utf-8', errors='ignore')
        
        if sh_type == 1:  # SHT_PROGBITS
            sections[name] = elf_data[sh_offset_in_file:sh_offset_in_file + sh_size]
    
    return sections

def parse_abbrev_table(data):
    """Parse .debug_abbrev section and return list of abbreviations"""
    abbrevs = []
    offset = 0
    
    while offset < len(data):
        # Decode abbreviation code
        code, n = decode_uleb128(data, offset)
        offset += n
        
        if code == 0:
            # End of table
            break
        
        # Decode tag
        tag, n = decode_uleb128(data, offset)
        offset += n
        
        # Read children flag
        if offset >= len(data):
            break
        children_byte = data[offset]
        offset += 1
        has_children = children_byte != 0
        
        # Read attribute/form pairs
        attrs_forms = []
        while offset < len(data):
            attr, n = decode_uleb128(data, offset)
            offset += n
            if offset >= len(data):
                break
            form, n = decode_uleb128(data, offset)
            offset += n
            
            if attr == 0 and form == 0:
                # End of this abbreviation
                break
            
            attrs_forms.append((attr, form))
        
        abbrevs.append({
            'code': code,
            'tag': tag,
            'has_children': has_children,
            'attrs_forms': attrs_forms
        })
    
    return abbrevs

def format_abbrev_output(abbrevs, filename):
    """Format abbreviation table output similar to llvm-dwarfdump"""
    lines = []
    lines.append(f"{filename}:\tfile format elf64-amdgpu")
    lines.append("")
    lines.append(".debug_abbrev contents:")
    lines.append("Abbrev table for offset: 0x00000000")
    
    for abbrev in abbrevs:
        children_str = "DW_CHILDREN_yes" if abbrev['has_children'] else "DW_CHILDREN_no"
        tag_name = DW_TAG.get(abbrev['tag'], f"DW_TAG_<unknown_{abbrev['tag']}>")
        lines.append(f"[{abbrev['code']}] {tag_name}\t{children_str}")
        
        for attr, form in abbrev['attrs_forms']:
            attr_name = DW_AT.get(attr, f"DW_AT_<unknown_{attr}>")
            form_name = DW_FORM.get(form, f"DW_FORM_<unknown_{form}>")
            lines.append(f"\t{attr_name}\t{form_name}")
    
    return "\n".join(lines)

def main():
    if len(sys.argv) != 3:
        print(f"Usage: {sys.argv[0]} <input_elf> <output_file>", file=sys.stderr)
        sys.exit(1)
    
    input_file = sys.argv[1]
    output_file = sys.argv[2]
    
    # Read ELF file
    with open(input_file, 'rb') as f:
        elf_data = f.read()
    
    # Parse ELF sections
    sections = parse_elf_sections(elf_data)
    
    if '.debug_abbrev' not in sections:
        print(f"Error: .debug_abbrev section not found in {input_file}", file=sys.stderr)
        sys.exit(1)
    
    # Parse abbreviation table
    abbrev_data = sections['.debug_abbrev']
    abbrevs = parse_abbrev_table(abbrev_data)
    
    # Format output
    output = format_abbrev_output(abbrevs, input_file)
    
    # Write output file
    with open(output_file, 'w') as f:
        f.write(output)
    
    print(f"Abbreviation table parsed and written to: {output_file}")

if __name__ == '__main__':
    main()
